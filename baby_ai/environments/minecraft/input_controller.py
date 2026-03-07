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
        """
        vk = VK[key] if isinstance(key, str) else key
        if not force and vk in BLOCKED_KEYS:
            log.debug("Blocked key %s (vk=0x%02X) — not allowed for AI.", key, vk)
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

    def mouse_down(self, button: str = "left", x: int = -1, y: int = -1) -> None:
        """
        Send a mouse-button-down event at (x, y) in client coords.

        If x/y are -1, the click is sent at the client-area center
        (the crosshair location in-game).

        This uses PostMessage which targets the MC window by handle,
        so clicks NEVER affect other windows.
        """
        if not self._window.is_valid:
            return

        if x < 0 or y < 0:
            _, _, cw, ch = self._window.get_client_rect()
            # Shift Y slightly up to hit 'Respawn' and avoid 'Title Screen' button
            x, y = cw // 2, (ch // 2) - int(ch * 0.15)

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
        """Send a mouse-button-up event."""
        if x < 0 or y < 0:
            _, _, cw, ch = self._window.get_client_rect()
            # Shift Y slightly up to hit 'Respawn' and avoid 'Title Screen' button
            x, y = cw // 2, (ch // 2) - int(ch * 0.15)

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
        """
        if not self._window.is_valid:
            return False

        if self._mode == "background":
            # Send the delta relative to the center of the window
            _, _, cw, ch = self._window.get_client_rect()
            
            # Since Minecraft is windowed, local client coordinates start at 0,0
            local_cx = cw // 2
            local_cy = ch // 2
            
            target_x = local_cx + dx
            target_y = local_cy + dy
            
            # 1. Send the offset to simulate the mouse moving
            lp_move = _makelparam(target_x, target_y)
            user32.PostMessageW(self._window.hwnd, WM_MOUSEMOVE, 0, lp_move)
            
            # 2. Immediately send a "reset to center" message.
            # Because Minecraft is backgrounded, its native SetCursorPos(center) 
            # won't trigger real mouse events to reset its internal last_x/last_y.
            # We must trick it into thinking the cursor snapped back to center.
            lp_reset = _makelparam(local_cx, local_cy)
            user32.PostMessageW(self._window.hwnd, WM_MOUSEMOVE, 0, lp_reset)
            return True

        # ACTIVE MODE SAFETY:
        if not self._window.is_focused:
            return False

        cx, cy = self._window.get_client_center()
        sx, sy, cw, ch = self._window.get_client_rect()

        target_x = max(sx + 1, min(cx + dx, sx + cw - 2))
        target_y = max(sy + 1, min(cy + dy, sy + ch - 2))

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
        """
        to_release = self._held_keys - desired_keys
        to_press = desired_keys - self._held_keys
        for vk in to_release:
            self.release_key(vk)
        for vk in to_press:
            self.press_key(vk)

    def set_buttons(self, desired_buttons: Set[str]) -> None:
        """Transition to exactly the set of *desired_buttons* being held."""
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
