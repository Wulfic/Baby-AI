"""
Input guard — blocks user keyboard/mouse from reaching the Minecraft window.

Uses Win32 low-level hooks (``WH_KEYBOARD_LL`` and ``WH_MOUSE_LL``) to
intercept hardware input *before* it reaches any application.  When the
Minecraft window is the foreground window, the hooks swallow the events
so the user's physical keyboard and mouse cannot interfere with the AI.

The AI's input is unaffected because it uses ``PostMessage`` which
injects messages directly into the window's message queue, bypassing
the hook chain entirely.

Architecture::

    ┌──────────────┐
    │  User KB/Mouse│──▶ Low-Level Hook ──▶ BLOCKED (when MC focused)
    └──────────────┘         │
                             └──▶ PASSED (when MC not focused)

    ┌──────────────┐
    │  AI PostMsg  │──▶ MC Window Queue ──▶ Game processes it
    └──────────────┘    (bypasses hooks)

Thread model:
    The hooks must run on a thread with a Windows message loop.  We
    spin up a daemon thread that installs the hooks and pumps messages
    until :meth:`InputGuard.stop` posts ``WM_QUIT``.

Usage::

    guard = InputGuard(mc_hwnd=0x12345)
    guard.start()
    # ... AI training ...
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

# ── Win32 constants ─────────────────────────────────────────────
WH_KEYBOARD_LL = 13
WH_MOUSE_LL    = 14
WM_QUIT        = 0x0012

# Return 1 from a low-level hook proc to block the event
_BLOCK = 1

# ── Win32 bindings ──────────────────────────────────────────────
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# Hook procedure type: LRESULT CALLBACK (int nCode, WPARAM, LPARAM)
HOOKPROC = ctypes.WINFUNCTYPE(
    ctypes.c_longlong,   # LRESULT (64-bit)
    ctypes.c_int,        # nCode
    ctypes.c_ulonglong,  # WPARAM (64-bit unsigned)
    ctypes.c_longlong,   # LPARAM (64-bit signed)
)

# SetWindowsHookExW
user32.SetWindowsHookExW.argtypes = [
    ctypes.c_int,        # idHook
    HOOKPROC,            # lpfn
    ctypes.c_void_p,     # hMod (NULL for LL hooks)
    wt.DWORD,            # dwThreadId (0 = all threads)
]
user32.SetWindowsHookExW.restype = ctypes.c_void_p

# UnhookWindowsHookEx
user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
user32.UnhookWindowsHookEx.restype = wt.BOOL

# CallNextHookEx
user32.CallNextHookEx.argtypes = [
    ctypes.c_void_p,     # hhk
    ctypes.c_int,        # nCode
    ctypes.c_ulonglong,  # WPARAM
    ctypes.c_longlong,   # LPARAM
]
user32.CallNextHookEx.restype = ctypes.c_longlong

# GetForegroundWindow
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wt.HWND

# GetMessage / PostThreadMessage
user32.GetMessageW.argtypes = [
    ctypes.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT,
]
user32.GetMessageW.restype = wt.BOOL

user32.TranslateMessage.argtypes = [ctypes.POINTER(wt.MSG)]
user32.TranslateMessage.restype = wt.BOOL

user32.DispatchMessageW.argtypes = [ctypes.POINTER(wt.MSG)]
user32.DispatchMessageW.restype = ctypes.c_longlong

user32.PostThreadMessageW.argtypes = [wt.DWORD, wt.UINT, ctypes.c_ulonglong, ctypes.c_longlong]
user32.PostThreadMessageW.restype = wt.BOOL

# GetCurrentThreadId
kernel32.GetCurrentThreadId.argtypes = []
kernel32.GetCurrentThreadId.restype = wt.DWORD

# Allowed keys — even when the guard is active, these keys pass through
# so the user can always escape the AI lock (e.g. Alt+Tab, Ctrl+Alt+Del)
_ALWAYS_PASS_VKS = {
    0x09,   # VK_TAB (for Alt+Tab)
    0x5B,   # VK_LWIN
    0x5C,   # VK_RWIN
    0x2E,   # VK_DELETE (for Ctrl+Alt+Del)
    0x91,   # VK_SCROLL (Scroll Lock — emergency toggle)
}

# Virtual key codes for hotkey detection
_VK_Q = 0x51
_VK_M = 0x4D
_VK_P = 0x50
_VK_CONTROL = 0x11

# GetAsyncKeyState — lets us check modifier state inside the hook
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short

# KBDLLHOOKSTRUCT layout
class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wt.DWORD),
        ("scanCode", wt.DWORD),
        ("flags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


# MSLLHOOKSTRUCT layout (for WH_MOUSE_LL)
class MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", wt.POINT),
        ("mouseData", wt.DWORD),
        ("flags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class InputGuard:
    """
    Blocks user keyboard and mouse input from reaching the Minecraft window.

    When enabled, low-level hooks intercept hardware events.  If the
    foreground window is the guarded MC window, the events are swallowed.
    Otherwise, they pass through normally.

    Safety features:
    - Alt+Tab, Win key, and Ctrl+Alt+Del ALWAYS pass through.
    - Scroll Lock acts as an emergency toggle to disable the guard.
    - The guard auto-disables if the MC window is no longer valid.

    Args:
        mc_hwnd: The window handle of the Minecraft window to protect.
    """

    def __init__(self, mc_hwnd: int):
        self._mc_hwnd = mc_hwnd
        self._enabled = False
        self._emergency_off = False  # Scroll Lock toggle
        self._mouse_blocked = True   # Ctrl+M toggle for mouse blocking
        self._kb_blocked = True      # Ctrl+M toggle for keyboard blocking
        self._quit_requested = False  # Ctrl+Q sets this flag
        self._ai_paused = False      # Ctrl+P toggle for pausing AI
        self._thread: Optional[threading.Thread] = None
        self._thread_id: Optional[int] = None
        self._kb_hook: Optional[ctypes.c_void_p] = None
        self._mouse_hook: Optional[ctypes.c_void_p] = None
        self._ready = threading.Event()
        self._stats = {"kb_blocked": 0, "mouse_blocked": 0, "passed": 0}

        # ── Player input tracking (for imitation learning) ──────
        # When the guard is NOT blocking (imitation mode), these
        # capture what the human player is actually pressing so the
        # replay buffer can store the real demonstrated actions.
        self._player_lock = threading.Lock()
        self._player_held_keys: set[int] = set()     # VK codes currently held
        self._player_held_buttons: set[str] = set()   # "left"/"right"/"middle"

        # Keep references to prevent garbage collection of callbacks
        self._kb_proc = HOOKPROC(self._keyboard_hook_proc)
        self._mouse_proc = HOOKPROC(self._mouse_hook_proc)

    def start(self) -> None:
        """Install hooks and start the message loop thread."""
        if self._thread is not None and self._thread.is_alive():
            log.warning("Input guard is already running.")
            return

        self._enabled = True
        self._emergency_off = False
        self._ready.clear()

        self._thread = threading.Thread(
            target=self._hook_thread,
            name="InputGuard",
            daemon=True,
        )
        self._thread.start()

        # Wait for hooks to be installed
        if not self._ready.wait(timeout=5.0):
            log.error("Input guard thread did not start in time!")
            self._enabled = False
            return

        log.info(
            "Input guard ACTIVE — blocking user input to MC window (hwnd=%s). "
            "Scroll Lock=emergency-toggle | Ctrl+Q=save&quit | Ctrl+M=toggle input block | Ctrl+P=pause AI",
            hex(self._mc_hwnd),
        )

    def stop(self) -> None:
        """Remove hooks and stop the message loop."""
        self._enabled = False

        if self._thread_id is not None:
            # Post WM_QUIT to break the message loop
            user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning("Input guard thread did not exit cleanly.")
            self._thread = None
            self._thread_id = None

        log.info("Input guard stopped.")

    def _should_block(self) -> bool:
        """Check if we should block input right now."""
        if not self._enabled or self._emergency_off:
            return False

        # Only block when MC window is the foreground window.
        # GetForegroundWindow() can return None when no window is focused.
        fg = user32.GetForegroundWindow()
        if fg is None:
            return False
        return int(fg) == self._mc_hwnd

    def _is_ctrl_held(self) -> bool:
        """Check if Ctrl is currently held (via GetAsyncKeyState)."""
        return (user32.GetAsyncKeyState(_VK_CONTROL) & 0x8000) != 0

    def _keyboard_hook_proc(
        self, nCode: int, wParam: int, lParam: int,
    ) -> int:
        """Low-level keyboard hook callback."""
        if nCode >= 0 and self._should_block():
            # Parse the key info
            kb = ctypes.cast(lParam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            vk = kb.vkCode

            # Only react to key-down events (WM_KEYDOWN=0x0100, WM_SYSKEYDOWN=0x0104)
            is_keydown = wParam in (0x0100, 0x0104)

            # ── Hotkeys (checked on key-down only) ──────────────
            if is_keydown and self._is_ctrl_held():
                if vk == _VK_Q:
                    # Ctrl+Q  →  request save & quit
                    self._quit_requested = True
                    log.info("Ctrl+Q pressed — save & quit requested.")
                    return _BLOCK

                if vk == _VK_M:
                    # Ctrl+M  →  toggle keyboard + mouse blocking
                    self._kb_blocked = not self._kb_blocked
                    self._mouse_blocked = self._kb_blocked
                    state = "ON" if self._kb_blocked else "OFF"
                    log.info("Ctrl+M pressed — input blocking %s", state)
                    return _BLOCK

                if vk == _VK_P:
                    # Ctrl+P  →  toggle AI pause + unlock/lock controls
                    self._ai_paused = not self._ai_paused
                    if self._ai_paused:
                        # Pausing → unlock controls so user can play
                        self._kb_blocked = False
                        self._mouse_blocked = False
                    else:
                        # Resuming → re-lock controls
                        self._kb_blocked = True
                        self._mouse_blocked = True
                    state = "PAUSED (controls unlocked)" if self._ai_paused else "RUNNING (controls locked)"
                    log.info("Ctrl+P pressed — AI %s", state)
                    return _BLOCK

            # Scroll Lock toggles emergency off
            if vk == 0x91:  # VK_SCROLL
                self._emergency_off = not self._emergency_off
                state_str = "OFF (emergency)" if self._emergency_off else "ON"
                log.info("Input guard toggled: %s", state_str)
                # Let the Scroll Lock key itself pass through
                return user32.CallNextHookEx(None, nCode, wParam, lParam)

            # Always let safety keys pass
            if vk in _ALWAYS_PASS_VKS:
                self._stats["passed"] += 1
                return user32.CallNextHookEx(None, nCode, wParam, lParam)

            # Also let Alt through (for Alt+Tab, Alt+F4)
            if vk in (0xA4, 0xA5, 0x12):  # VK_LMENU, VK_RMENU, VK_MENU
                self._stats["passed"] += 1
                return user32.CallNextHookEx(None, nCode, wParam, lParam)

            # Also let Ctrl through (for Ctrl+Alt+Del)
            if vk in (0xA2, 0xA3, 0x11):  # VK_LCONTROL, VK_RCONTROL, VK_CONTROL
                self._stats["passed"] += 1
                return user32.CallNextHookEx(None, nCode, wParam, lParam)

            # If keyboard blocking is disabled via Ctrl+M, let keys through
            # but also TRACK them for imitation learning.
            if not self._kb_blocked:
                self._stats["passed"] += 1
                with self._player_lock:
                    if is_keydown:
                        self._player_held_keys.add(vk)
                    else:
                        self._player_held_keys.discard(vk)
                return user32.CallNextHookEx(None, nCode, wParam, lParam)

            # Block this key event
            self._stats["kb_blocked"] += 1
            return _BLOCK

        return user32.CallNextHookEx(None, nCode, wParam, lParam)

    def _mouse_hook_proc(
        self, nCode: int, wParam: int, lParam: int,
    ) -> int:
        """
        Low-level mouse hook callback.

        Blocks PHYSICAL mouse clicks and scroll so the user can't interfere
        with the game. Always lets INJECTED (AI) moves/clicks pass through!
        We also allow physical WM_MOUSEMOVE to pass through so the user doesn't
        get a frozen OS cursor and can still use Alt-Tab or move their mouse
        to click the AI Pause button.

        Ctrl+M toggles mouse blocking — when OFF, all physical mouse events
        pass through so the user can interact with MC normally.
        """
        if nCode >= 0 and self._should_block():
            # If mouse blocking is disabled via Ctrl+M, let everything through
            # but also TRACK button presses for imitation learning.
            if not self._mouse_blocked:
                self._stats["passed"] += 1
                # Track mouse button state for imitation learning
                with self._player_lock:
                    if wParam == 0x0201:    # WM_LBUTTONDOWN
                        self._player_held_buttons.add("left")
                    elif wParam == 0x0202:  # WM_LBUTTONUP
                        self._player_held_buttons.discard("left")
                    elif wParam == 0x0204:  # WM_RBUTTONDOWN
                        self._player_held_buttons.add("right")
                    elif wParam == 0x0205:  # WM_RBUTTONUP
                        self._player_held_buttons.discard("right")
                    elif wParam == 0x0207:  # WM_MBUTTONDOWN
                        self._player_held_buttons.add("middle")
                    elif wParam == 0x0208:  # WM_MBUTTONUP
                        self._player_held_buttons.discard("middle")
                return user32.CallNextHookEx(None, nCode, wParam, lParam)

            struct = ctypes.cast(lParam, ctypes.POINTER(MSLLHOOKSTRUCT)).contents

            # Check if this input was injected by software (like SetCursorPos or PostMessage AI)
            is_injected = (struct.flags & 1) != 0 or (struct.flags & 2) != 0
            if is_injected:
                self._stats["passed"] += 1
                return user32.CallNextHookEx(None, nCode, wParam, lParam)

            # If physical hardware:
            # 0x0200 = WM_MOUSEMOVE
            if wParam == 0x0200:
                # Let the user physically move their mouse (otherwise cursor is fully frozen system-wide)
                # Note: this WILL rotate the camera slightly if the user wiggles their mouse while watching,
                # but it allows them to move their mouse to click the AI UI.
                self._stats["passed"] += 1
                return user32.CallNextHookEx(None, nCode, wParam, lParam)

            # Block physical clicks and scroll
            self._stats["mouse_blocked"] += 1
            return _BLOCK

        return user32.CallNextHookEx(None, nCode, wParam, lParam)

    def _hook_thread(self) -> None:
        """Thread entry point — installs hooks and runs message loop."""
        self._thread_id = kernel32.GetCurrentThreadId()

        # Install keyboard hook
        self._kb_hook = user32.SetWindowsHookExW(
            WH_KEYBOARD_LL, self._kb_proc, None, 0,
        )
        if not self._kb_hook:
            log.error("Failed to install keyboard hook (error=%d)", ctypes.GetLastError())
            self._ready.set()
            return

        # Install mouse hook
        self._mouse_hook = user32.SetWindowsHookExW(
            WH_MOUSE_LL, self._mouse_proc, None, 0,
        )
        if not self._mouse_hook:
            log.error("Failed to install mouse hook (error=%d)", ctypes.GetLastError())
            user32.UnhookWindowsHookEx(self._kb_hook)
            self._kb_hook = None
            self._ready.set()
            return

        log.info("Low-level hooks installed (kb=%s, mouse=%s)",
                 self._kb_hook, self._mouse_hook)
        self._ready.set()

        # ── Message loop (required for low-level hooks) ─────────
        msg = wt.MSG()
        while self._enabled:
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret <= 0:  # 0 = WM_QUIT, -1 = error
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        # ── Cleanup ─────────────────────────────────────────────
        if self._kb_hook:
            user32.UnhookWindowsHookEx(self._kb_hook)
            self._kb_hook = None
        if self._mouse_hook:
            user32.UnhookWindowsHookEx(self._mouse_hook)
            self._mouse_hook = None

        log.info("Hook thread exiting. Stats: %s", self._stats)

    # ── Public API ──────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled and not self._emergency_off

    @property
    def quit_requested(self) -> bool:
        """True after the user presses Ctrl+Q."""
        return self._quit_requested

    @property
    def mouse_blocked(self) -> bool:
        """True when physical mouse clicks are being blocked."""
        return self._mouse_blocked

    @property
    def kb_blocked(self) -> bool:
        """True when physical keyboard input is being blocked."""
        return self._kb_blocked

    @property
    def ai_paused(self) -> bool:
        """True when the user has paused the AI via Ctrl+P."""
        return self._ai_paused

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def update_hwnd(self, new_hwnd: int) -> None:
        """Update the guarded window handle (e.g. after MC restart)."""
        self._mc_hwnd = new_hwnd
        log.info("Input guard target updated: hwnd=%s", hex(new_hwnd))

    # ── Imitation learning helpers ──────────────────────────────

    def snapshot_player_input(self) -> tuple[frozenset[int], frozenset[str]]:
        """
        Atomically read which keys and mouse buttons the player is
        currently holding.

        Returns:
            (held_keys, held_buttons) where held_keys is a frozenset
            of VK codes and held_buttons is a frozenset of button names
            ("left", "right", "middle").
        """
        with self._player_lock:
            return frozenset(self._player_held_keys), frozenset(self._player_held_buttons)

    def clear_player_input(self) -> None:
        """Reset all tracked player input state (e.g. on mode switch)."""
        with self._player_lock:
            self._player_held_keys.clear()
            self._player_held_buttons.clear()