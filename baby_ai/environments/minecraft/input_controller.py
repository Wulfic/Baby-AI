"""
Low-level input controller for Minecraft.

Sends keyboard and mouse events to the Minecraft window using Win32
``PostMessage`` — this targets the window by *handle*, so the user's
real keyboard and mouse are **not** affected.

Two operating modes:

``"background"``  (default)
    All input via PostMessage.  Keyboard and mouse *clicks* work
    reliably.  Camera look (mouse movement) is **not available**
    because Minecraft's LWJGL/GLFW uses raw-input for mouse deltas
    which cannot be faked via PostMessage.

``"active"``
    Keyboard still via PostMessage.  Camera look works by briefly
    warping the OS cursor with ``SetCursorPos``.  This *does* move
    the real cursor but only within the Minecraft window and the
    game re-centres it every frame.  The window must be focused.

In both modes the user keeps full control of their physical
keyboard.  In *active* mode the mouse cursor is shared for look
actions only.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import time
from typing import Dict, Optional, Set

from baby_ai.environments.minecraft.window import WindowManager, user32
from baby_ai.utils.logging import get_logger

log = get_logger("mc_input")

# Forward-declare; set at runtime via set_controls_state().
_controls_state = None

def set_controls_state(state) -> None:
    """Inject the shared AIControlsState so the controller can respect it.

    Called once from main.py after creating the env and controls_state.
    Must be called before the training loop starts.
    """
    global _controls_state
    _controls_state = state

# ── Win32 message constants ─────────────────────────────────────
WM_KEYDOWN      = 0x0100
WM_KEYUP        = 0x0101
WM_MOUSEMOVE    = 0x0200
WM_LBUTTONDOWN  = 0x0201
WM_LBUTTONUP    = 0x0202
WM_RBUTTONDOWN  = 0x0204
WM_RBUTTONUP    = 0x0205
WM_MBUTTONDOWN  = 0x0207
WM_MBUTTONUP    = 0x0208

MK_LBUTTON = 0x0001
MK_RBUTTON = 0x0002
MK_MBUTTON = 0x0010

MAPVK_VK_TO_VSC = 0

# PostMessage signature
user32.PostMessageW.argtypes = [wt.HWND, wt.UINT, ctypes.c_ulonglong, ctypes.c_longlong]
user32.PostMessageW.restype = wt.BOOL

# MapVirtualKeyW — convert virtual-key code to scan code
user32.MapVirtualKeyW.argtypes = [wt.UINT, wt.UINT]
user32.MapVirtualKeyW.restype = wt.UINT

# Cursor manipulation (active mode only)
user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
user32.SetCursorPos.restype = wt.BOOL

user32.GetCursorPos.argtypes = [ctypes.POINTER(wt.POINT)]
user32.GetCursorPos.restype = wt.BOOL

# ── Virtual-key codes used by Minecraft ─────────────────────────
VK = {
    "W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44,
    "E": 0x45, "Q": 0x51, "F": 0x46,
    "SPACE": 0x20, "LSHIFT": 0xA0, "LCTRL": 0xA2,
    "1": 0x31, "2": 0x32, "3": 0x33, "4": 0x34, "5": 0x35,
    "6": 0x36, "7": 0x37, "8": 0x38, "9": 0x39, "0": 0x30,
    # Kept for launcher/wrapper use only — blocked from AI actions:
    "ESCAPE": 0x1B, "TAB": 0x09,
    "F3": 0x72, "F5": 0x74,
}

# Keys the AI is NEVER allowed to send during gameplay.
# ESC opens pause/settings, F-keys toggle debug overlays.
# These are only usable by the launcher/wrapper code directly.
BLOCKED_KEYS: Set[int] = {VK["ESCAPE"], VK["TAB"], VK["F3"], VK["F5"]}

# Toggle / momentary keys — these are tapped (down+up in one step)
# rather than held.  Minecraft treats E, Q, and number keys as
# instant-action triggers; holding them via continuous KEYDOWN
# confuses the keybind system and can cause keys to "unbind".
_TAP_KEYS: Set[int] = {
    VK["E"],   # inventory toggle
    VK["Q"],   # drop item
    VK["F"],   # swap hands
    # Hotbar keys — only need a single tap to switch slots
    VK["1"], VK["2"], VK["3"], VK["4"], VK["5"],
    VK["6"], VK["7"], VK["8"], VK["9"], VK["0"],
}

# Extended keys that need the bit-24 flag
_EXTENDED_KEYS: Set[int] = set()  # add VK codes here if needed


def _make_key_lparam(scan_code: int, extended: bool = False, down: bool = True) -> int:
    """
    Build the lParam bitfield for WM_KEYDOWN / WM_KEYUP.

    Layout (32-bit value):
        bits  0-15  repeat count (always 1)
        bits 16-23  scan code
        bit  24     extended-key flag
        bit  29     context code (0)
        bit  30     previous key state (0=new press, 1=was down)
        bit  31     transition state (0=pressing, 1=releasing)
    """
    lp = 1                              # repeat = 1
    lp |= (scan_code & 0xFF) << 16     # scan code
    if extended:
        lp |= 1 << 24
    if not down:
        lp |= 1 << 30                  # was pressed
        lp |= 1 << 31                  # being released
    return lp


def _makelparam(x: int, y: int) -> int:
    """Pack two 16-bit values into a single LPARAM (low=x, high=y)."""
    return ((y & 0xFFFF) << 16) | (x & 0xFFFF)


class InputController:
    """
    Sends keyboard and mouse input to Minecraft via PostMessage.

    Usage::

        ctrl = InputController(window_mgr, mode="background")
        ctrl.press_key("W")      # hold forward
        ctrl.mouse_click("left") # swing / mine
        ctrl.release_all()       # let go of everything

    Args:
        window: WindowManager for the Minecraft window.
        mode: ``"background"`` or ``"active"``.
    """

    def __init__(self, window: WindowManager, mode: str = "background"):
        if mode not in ("background", "active"):
            raise ValueError(f"mode must be 'background' or 'active', got '{mode}'")
        self._window = window
        self._mode = mode

        # Track currently held keys / buttons to release them properly
        self._held_keys: Set[int] = set()           # VK codes
        self._held_buttons: Set[str] = set()         # "left", "right", "middle"

        # Cache scan codes (VK → SC mapping doesn't change)
        self._scan_cache: Dict[int, int] = {}

        # ── Inventory cooldown (max 2 taps per 5 s) ───────────
        self._inventory_tap_times: list[float] = []
        self._INVENTORY_MAX_TAPS: int = 2
        self._INVENTORY_WINDOW: float = 5.0
        self.inventory_spam_blocked: int = 0  # counter for reward penalty

        # When True, ALL inputs (keys, mouse, look) are silently
        # dropped.  Set by the inference thread during System 2/3
        # planning while the mod freezes server ticks.
        self.paused: bool = False

        # Background-mode tracked cursor position (client coords).
        # Look deltas accumulate here so the AI can reach any part
        # of the screen (inventory slots, crafting grid, etc.).
        # Lazily initialised to window center on first use.
        self._bg_cursor_x: int = -1
        self._bg_cursor_y: int = -1

    # ── Keyboard ────────────────────────────────────────────────

    def _scan(self, vk: int) -> int:
        """Get the scan code for a virtual-key code (cached)."""
        if vk not in self._scan_cache:
            self._scan_cache[vk] = user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)
        return self._scan_cache[vk]

    def press_key(self, key: str | int, *, force: bool = False) -> None:
        """
        Send a key-down event. *key* is a name from the VK table
        (e.g. ``"W"``) or a raw VK code (int).

        Blocked keys (ESC, F3, F5, TAB) are silently dropped unless
        *force=True* (used only by the launcher/wrapper code).

        Keys disabled via the AI Controls panel are also silently
        dropped unless *force=True*.
        """
        vk = VK[key] if isinstance(key, str) else key
        if not force and self.paused:
            return
        if not force and vk in BLOCKED_KEYS:
            log.debug("Blocked key %s (vk=0x%02X) — not allowed for AI.", key, vk)
            return
        # Check AI Controls state (UI toggles).
        if not force and _controls_state is not None:
            if not _controls_state.is_vk_allowed(vk, VK):
                return
        sc = self._scan(vk)
        ext = vk in _EXTENDED_KEYS
        lp = _make_key_lparam(sc, extended=ext, down=True)
        user32.PostMessageW(self._window.hwnd, WM_KEYDOWN, vk, lp)
        self._held_keys.add(vk)

    def release_key(self, key: str | int, *, force: bool = False) -> None:
        """Send a key-up event. Blocked keys require *force=True*."""
        vk = VK[key] if isinstance(key, str) else key
        if not force and vk in BLOCKED_KEYS:
            return
        sc = self._scan(vk)
        ext = vk in _EXTENDED_KEYS
        lp = _make_key_lparam(sc, extended=ext, down=False)
        user32.PostMessageW(self._window.hwnd, WM_KEYUP, vk, lp)
        self._held_keys.discard(vk)

    # ── Mouse buttons ──────────────────────────────────────────

    def _get_cursor_client_pos(self) -> tuple[int, int]:
        """
        Return the current cursor position in client coordinates.

        Active mode: reads the real OS cursor via ``GetCursorPos``
        and converts to client-relative coords.
        Background mode: returns the tracked ``_bg_cursor`` position
        (initialised to center on first use).
        """
        _, _, cw, ch = self._window.get_client_rect()
        if self._mode == "active":
            pt = wt.POINT()
            user32.GetCursorPos(ctypes.byref(pt))
            sx, sy, _, _ = self._window.get_client_rect()
            cx = max(0, min(pt.x - sx, cw - 1))
            cy = max(0, min(pt.y - sy, ch - 1))
            return cx, cy
        # Background — use tracked cursor
        if self._bg_cursor_x < 0:
            self._bg_cursor_x = cw // 2
            self._bg_cursor_y = ch // 2
        return self._bg_cursor_x, self._bg_cursor_y

    def mouse_down(self, button: str = "left", x: int = -1, y: int = -1) -> None:
        """
        Send a mouse-button-down event at (x, y) in client coords.

        If x/y are -1, the click is sent at the **current cursor
        position** — the real OS cursor in active mode, or the
        tracked virtual cursor in background mode.  This allows
        the AI to click on inventory slots, crafting grids, etc.
        after positioning the cursor with look actions.

        This uses PostMessage which targets the MC window by handle,
        so clicks NEVER affect other windows.

        Buttons disabled via the AI Controls panel are silently dropped.
        """
        if not self._window.is_valid:
            return
        if self.paused:
            return
        # Check AI Controls state (UI toggles).
        if _controls_state is not None:
            if not _controls_state.is_button_allowed(button):
                return

        if x < 0 or y < 0:
            x, y = self._get_cursor_client_pos()

        lp = _makelparam(x, y)

        msg_map = {
            "left":   (WM_LBUTTONDOWN, MK_LBUTTON),
            "right":  (WM_RBUTTONDOWN, MK_RBUTTON),
            "middle": (WM_MBUTTONDOWN, MK_MBUTTON),
        }
        msg, wp = msg_map[button]
        user32.PostMessageW(self._window.hwnd, msg, wp, lp)
        self._held_buttons.add(button)

    def mouse_up(self, button: str = "left", x: int = -1, y: int = -1) -> None:
        """Send a mouse-button-up event at the current cursor position."""
        if x < 0 or y < 0:
            x, y = self._get_cursor_client_pos()

        lp = _makelparam(x, y)

        msg_map = {
            "left":   WM_LBUTTONUP,
            "right":  WM_RBUTTONUP,
            "middle": WM_MBUTTONUP,
        }
        user32.PostMessageW(self._window.hwnd, msg_map[button], 0, lp)
        self._held_buttons.discard(button)

    def mouse_click(self, button: str = "left", x: int = -1, y: int = -1,
                     hold_ms: float = 50.0) -> None:
        """Press + release a mouse button with a short hold."""
        self.mouse_down(button, x, y)
        time.sleep(hold_ms / 1000.0)
        self.mouse_up(button, x, y)

    # ── Camera look (active mode) ──────────────────────────────

    def mouse_look(self, dx: int, dy: int) -> bool:
        """
        Rotate the camera by (dx, dy) pixels.
        
        In **background** mode, this sends WM_MOUSEMOVE via PostMessage. 
        Note: This only actually rotates the in-game camera if Minecraft's
        'rawMouseInput' setting is set to 'false' in options.txt.

        In **active** mode this warps the OS cursor relative to the
        window center.

        Camera look disabled via the AI Controls panel is silently dropped.
        """
        if not self._window.is_valid:
            return False
        if self.paused:
            return False
        # Check AI Controls state (UI toggles).
        if _controls_state is not None:
            if not _controls_state.is_look_allowed():
                return False

        if self._mode == "background":
            # ── Background mode: accumulate cursor position ──────
            # Camera rotation via PostMessage doesn't work (LWJGL
            # raw-input), but cursor positioning DOES work for GUI
            # screens (inventory, crafting table, etc.).
            _, _, cw, ch = self._window.get_client_rect()

            # Lazy init to window center
            if self._bg_cursor_x < 0:
                self._bg_cursor_x = cw // 2
                self._bg_cursor_y = ch // 2

            # Accumulate delta (clamped to window bounds)
            self._bg_cursor_x = max(1, min(self._bg_cursor_x + dx, cw - 2))
            self._bg_cursor_y = max(1, min(self._bg_cursor_y + dy, ch - 2))

            # Send cursor to accumulated position so GUI screens
            # see the cursor move to the intended slot / button.
            lp = _makelparam(self._bg_cursor_x, self._bg_cursor_y)
            user32.PostMessageW(self._window.hwnd, WM_MOUSEMOVE, 0, lp)
            return True

        # ── Active mode: accumulate from real cursor position ───
        # Read where the OS cursor actually is right now.  In-game
        # Minecraft re-centres each frame, so GetCursorPos ≈ center.
        # But in GUI screens the cursor stays where we placed it,
        # allowing deltas to accumulate across steps and reach the
        # entire screen — inventory slots, crafting grid, etc.
        if not self._window.is_focused:
            return False

        pt = wt.POINT()
        user32.GetCursorPos(ctypes.byref(pt))

        sx, sy, cw, ch = self._window.get_client_rect()

        target_x = max(sx + 1, min(pt.x + dx, sx + cw - 2))
        target_y = max(sy + 1, min(pt.y + dy, sy + ch - 2))

        user32.SetCursorPos(target_x, target_y)
        return True

    # ── Bulk operations ─────────────────────────────────────────

    def release_all(self) -> None:
        """Release every held key and mouse button."""
        for vk in list(self._held_keys):
            self.release_key(vk)
        for btn in list(self._held_buttons):
            self.mouse_up(btn)

    def set_keys(self, desired_keys: Set[int]) -> None:
        """
        Transition to exactly the set of *desired_keys* being held.

        Keys not in the desired set are released; keys not currently
        held are pressed.  This minimises message count.

        **Toggle keys** (E, Q, F, hotbar numbers) are excluded from
        hold tracking.  They receive a single down+up tap so that
        Minecraft sees one clean press event without a lingering
        KEYDOWN that conflicts with the game's own keybind toggle
        logic.

        Keys disabled via the AI Controls panel are filtered out
        before processing.
        """
        # Filter through AI Controls state.
        if _controls_state is not None:
            desired_keys = _controls_state.filter_keys(desired_keys, VK)

        # Separate tap-only keys from holdable keys
        hold_desired = desired_keys - _TAP_KEYS
        tap_desired = desired_keys & _TAP_KEYS

        # Release / press holdable keys
        to_release = self._held_keys - hold_desired
        to_press = hold_desired - self._held_keys
        for vk in to_release:
            self.release_key(vk)
        for vk in to_press:
            self.press_key(vk)

        # Tap toggle keys (press then immediate release)
        for vk in tap_desired:
            # Inventory cooldown: max 2 taps per 5 seconds
            if vk == VK["E"]:
                now = time.monotonic()
                cutoff = now - self._INVENTORY_WINDOW
                self._inventory_tap_times = [
                    t for t in self._inventory_tap_times if t > cutoff
                ]
                if len(self._inventory_tap_times) >= self._INVENTORY_MAX_TAPS:
                    self.inventory_spam_blocked += 1
                    log.debug("Inventory tap suppressed (cooldown)")
                    continue
                self._inventory_tap_times.append(now)
            self.press_key(vk)
            time.sleep(0.035)          # 35 ms hold — long enough for MC to register
            self.release_key(vk)

    def set_buttons(self, desired_buttons: Set[str]) -> None:
        """Transition to exactly the set of *desired_buttons* being held.

        Buttons disabled via the AI Controls panel are filtered out.
        """
        # Filter through AI Controls state.
        if _controls_state is not None:
            desired_buttons = _controls_state.filter_buttons(desired_buttons)

        to_release = self._held_buttons - desired_buttons
        to_press = desired_buttons - self._held_buttons
        for btn in to_release:
            self.mouse_up(btn)
        for btn in to_press:
            self.mouse_down(btn)

    # ── Properties ──────────────────────────────────────────────

    @property
    def mode(self) -> str:
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        if value not in ("background", "active"):
            raise ValueError(f"mode must be 'background' or 'active', got '{value}'")
        self._mode = value

    @property
    def held_keys(self) -> Set[int]:
        return set(self._held_keys)

    @property
    def held_buttons(self) -> Set[str]:
        return set(self._held_buttons)
