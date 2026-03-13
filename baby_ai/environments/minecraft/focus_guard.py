"""
Focus guard — prevents user input from reaching Minecraft by keeping it unfocused.

Instead of using error-prone global low-level hooks that can trap the cursor
and interfere with the OS, this guard takes a fundamentally simpler approach:

1. Minecraft's ``pauseOnLostFocus`` is set to ``false`` (handled by the launcher)
   so the game keeps ticking even when it's not the foreground window.
2. The AI sends all input via ``PostMessage`` which injects directly into MC's
   message queue — this works regardless of focus state.
3. This guard runs a lightweight background thread that monitors focus.  If the
   user accidentally clicks on the MC window (giving it focus), the guard
   immediately steals focus back to the previous foreground window.

Result: **zero global hooks**, no cursor trapping, no blocked mouse movement.
The user's keyboard and mouse work normally at all times.  MC never receives
real hardware input because it's never the foreground window.

Architecture::

    ┌──────────────┐
    │  User KB/Mouse│──▶ Goes to whatever window is focused (not MC)
    └──────────────┘

    ┌──────────────┐
    │  AI PostMsg  │──▶ MC Window Queue ──▶ Game processes it
    └──────────────┘    (works without focus)

    ┌──────────────┐
    │  FocusGuard  │──▶ Monitors foreground window
    └──────────────┘    If MC gains focus → steal it back immediately

Safety:
    - Scroll Lock toggles the guard on/off (same as the old InputGuard)
    - If the guard is off, the user CAN click on MC and interact normally

NOTE: FocusGuard is exported via __init__.py but is **not currently
instantiated** by env.py.  The environment uses InputGuard instead.
FocusGuard is a newer alternative that avoids global hooks; it can
be wired in when PostMessage-based input is stabilised.

Usage::

    guard = FocusGuard(mc_hwnd=0x12345)
    guard.start()
    # ... AI training — user can freely use mouse/keyboard ...
    guard.stop()
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import threading
import time
from typing import Optional

from baby_ai.utils.logging import get_logger

log = get_logger("mc_guard")

# ── Win32 bindings ──────────────────────────────────────────────
user32 = ctypes.windll.user32

user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wt.HWND

user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.SetForegroundWindow.restype = wt.BOOL

user32.IsWindow.argtypes = [wt.HWND]
user32.IsWindow.restype = wt.BOOL

user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short

VK_SCROLL = 0x91  # Scroll Lock — emergency toggle


class FocusGuard:
    """
    Prevents the Minecraft window from being the foreground window.

    Runs a daemon thread that polls ``GetForegroundWindow()`` at ~60 Hz.
    If MC becomes the foreground window, focus is immediately switched
    back to the previously active window.  This means:

    - The user's real keyboard/mouse input never reaches MC.
    - The AI's ``PostMessage`` input still works (it bypasses focus).
    - The user can freely use every other application.

    Press **Scroll Lock** to toggle the guard on/off.  When off, the
    user can click on and interact with MC normally (useful for manual
    inspection of the world).

    Args:
        mc_hwnd: The window handle of the Minecraft window to protect.
    """

    def __init__(self, mc_hwnd: int):
        self._mc_hwnd = mc_hwnd
        self._enabled = False
        self._paused = False  # Scroll Lock toggle
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_good_hwnd: Optional[int] = None  # last non-MC foreground window
        self._stats = {"focus_stolen_back": 0}
        self._scroll_was_pressed = False

    def start(self) -> None:
        """Start the focus monitoring thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("Focus guard is already running.")
            return

        self._enabled = True
        self._paused = False
        self._stop_event.clear()

        # Remember the current foreground window (likely the terminal/IDE)
        fg = user32.GetForegroundWindow()
        if fg and int(fg) != self._mc_hwnd:
            self._last_good_hwnd = int(fg)

        self._thread = threading.Thread(
            target=self._monitor_loop,
            name="FocusGuard",
            daemon=True,
        )
        self._thread.start()

        log.info(
            "Focus guard ACTIVE — MC (hwnd=%s) will be kept unfocused. "
            "Press Scroll Lock to toggle.",
            hex(self._mc_hwnd),
        )

    def stop(self) -> None:
        """Stop the focus monitoring thread and release any cursor clip."""
        self._enabled = False
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                log.warning("Focus guard thread did not exit cleanly.")
            self._thread = None

        # Just in case MC trapped the OS cursor before getting killed.
        try:
            user32.ClipCursor(None)
        except Exception:
            pass

        log.info("Focus guard stopped. Stats: %s", self._stats)

    def _monitor_loop(self) -> None:
        """
        Main loop: poll foreground window and steal focus back from MC.

        Runs at ~60 Hz (16 ms sleep).  When MC gains focus, we call
        ``SetForegroundWindow`` on the last known non-MC window to
        push MC back to the background.
        """
        while not self._stop_event.is_set():
            # ── Scroll Lock toggle ──────────────────────────────
            scroll_state = user32.GetAsyncKeyState(VK_SCROLL)
            scroll_pressed = bool(scroll_state & 0x0001)  # toggled since last check
            if scroll_pressed and not self._scroll_was_pressed:
                self._paused = not self._paused
                state = "PAUSED (user can interact)" if self._paused else "ACTIVE"
                log.info("Focus guard toggled: %s", state)
            self._scroll_was_pressed = scroll_pressed

            if not self._paused:
                fg = user32.GetForegroundWindow()
                if fg is None:
                    time.sleep(0.016)
                    continue

                fg_int = int(fg)

                if fg_int == self._mc_hwnd:
                    # MC grabbed focus — steal it back
                    target = self._last_good_hwnd
                    if target and user32.IsWindow(target):
                        user32.SetForegroundWindow(target)
                        self._stats["focus_stolen_back"] += 1
                        log.debug("Stole focus back from MC → hwnd=%s", hex(target))
                    else:
                        # Fallback: focus the desktop (shell)
                        desktop = user32.GetDesktopWindow()
                        if desktop:
                            user32.SetForegroundWindow(desktop)
                            self._stats["focus_stolen_back"] += 1
                elif fg_int != 0:
                    # Track the last non-MC foreground window
                    self._last_good_hwnd = fg_int

            time.sleep(0.016)  # ~60 Hz polling

    # ── Public API ──────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._paused

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def update_hwnd(self, new_hwnd: int) -> None:
        """Update the guarded window handle (e.g. after MC restart)."""
        self._mc_hwnd = new_hwnd
        log.info("Focus guard target updated: hwnd=%s", hex(new_hwnd))
