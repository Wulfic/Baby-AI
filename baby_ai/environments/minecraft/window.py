"""
Win32 window management for Minecraft.

Uses ctypes to interact with the Windows API directly — no pywin32
dependency required. Finds the Minecraft window by title, tracks its
position, and manages focus when needed.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
from typing import List, Optional, Tuple

from baby_ai.utils.logging import get_logger

log = get_logger("mc_window")

# ── Win32 API bindings ──────────────────────────────────────────
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# EnumWindows callback type
WNDENUMPROC = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

# Function signatures
user32.EnumWindows.argtypes = [WNDENUMPROC, wt.LPARAM]
user32.EnumWindows.restype = wt.BOOL

user32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
user32.GetWindowTextW.restype = ctypes.c_int

user32.GetWindowTextLengthW.argtypes = [wt.HWND]
user32.GetWindowTextLengthW.restype = ctypes.c_int

user32.IsWindowVisible.argtypes = [wt.HWND]
user32.IsWindowVisible.restype = wt.BOOL

user32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetClientRect.restype = wt.BOOL

user32.GetWindowRect.argtypes = [wt.HWND, ctypes.POINTER(wt.RECT)]
user32.GetWindowRect.restype = wt.BOOL

user32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(wt.POINT)]
user32.ClientToScreen.restype = wt.BOOL

user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wt.HWND

user32.SetForegroundWindow.argtypes = [wt.HWND]
user32.SetForegroundWindow.restype = wt.BOOL

user32.IsWindow.argtypes = [wt.HWND]
user32.IsWindow.restype = wt.BOOL

user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetWindowThreadProcessId.restype = wt.DWORD


def find_windows_by_title(search: str) -> List[Tuple[int, str]]:
    """
    Find all visible windows whose title contains *search* (case-insensitive).

    Returns:
        List of (hwnd, title) tuples.
    """
    results: List[Tuple[int, str]] = []

    @WNDENUMPROC
    def _callback(hwnd: wt.HWND, _lparam: wt.LPARAM) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value
        if search.lower() in title.lower():
            results.append((int(hwnd), title))
        return True

    user32.EnumWindows(_callback, 0)
    return results


class WindowManager:
    """
    Tracks a specific window by its handle (HWND).

    Provides helpers for querying geometry, focus state, and
    client-area screen coordinates (used by the screen capture
    and input controller modules).

    Args:
        hwnd: Explicit window handle. If None, auto-discovers the
              first window matching *title_search*.
        title_search: Substring to search in window titles.
    """

    def __init__(
        self,
        hwnd: Optional[int] = None,
        title_search: str = "Minecraft",
    ):
        if hwnd is not None:
            self._hwnd = hwnd
        else:
            matches = find_windows_by_title(title_search)
            if not matches:
                raise RuntimeError(
                    f"No visible window found matching '{title_search}'. "
                    "Make sure Minecraft is running in windowed mode."
                )
            # Prefer the main game window (usually the longest title)
            matches.sort(key=lambda m: len(m[1]), reverse=True)
            self._hwnd = matches[0][0]
            log.info("Found Minecraft window: hwnd=%s title='%s'", hex(self._hwnd), matches[0][1])

        if not user32.IsWindow(self._hwnd):
            raise RuntimeError(f"Window handle {hex(self._hwnd)} is not valid.")

    # ── Properties ──────────────────────────────────────────────

    @property
    def hwnd(self) -> int:
        """Raw Win32 window handle."""
        return self._hwnd

    @property
    def title(self) -> str:
        length = user32.GetWindowTextLengthW(self._hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(self._hwnd, buf, length + 1)
        return buf.value

    @property
    def is_valid(self) -> bool:
        return bool(user32.IsWindow(self._hwnd))

    @property
    def is_focused(self) -> bool:
        return user32.GetForegroundWindow() == self._hwnd

    # ── Geometry ────────────────────────────────────────────────

    def get_window_rect(self) -> Tuple[int, int, int, int]:
        """Return (left, top, right, bottom) in screen pixels."""
        rect = wt.RECT()
        user32.GetWindowRect(self._hwnd, ctypes.byref(rect))
        return (rect.left, rect.top, rect.right, rect.bottom)

    def get_client_rect(self) -> Tuple[int, int, int, int]:
        """
        Return (screen_x, screen_y, width, height) of the client area.

        The client area excludes title bar and window borders — this is
        the actual game rendering area.
        """
        rect = wt.RECT()
        user32.GetClientRect(self._hwnd, ctypes.byref(rect))
        width, height = rect.right, rect.bottom

        # Convert client (0,0) to screen coordinates
        origin = wt.POINT(0, 0)
        user32.ClientToScreen(self._hwnd, ctypes.byref(origin))

        return (origin.x, origin.y, width, height)

    def get_client_center(self) -> Tuple[int, int]:
        """Screen coordinates of the client area center."""
        x, y, w, h = self.get_client_rect()
        return (x + w // 2, y + h // 2)

    # ── Focus management ────────────────────────────────────────

    def set_foreground(self) -> bool:
        """Attempt to bring the window to the foreground."""
        return bool(user32.SetForegroundWindow(self._hwnd))

    def __repr__(self) -> str:
        return f"WindowManager(hwnd={hex(self._hwnd)}, title='{self.title}')"
