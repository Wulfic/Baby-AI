"""
Shared UI theme constants and helpers for the Baby-AI control panel.

Extracted from ``control_panel.py`` to keep individual modules small
and to allow reuse if additional UI windows are added later.
"""

from __future__ import annotations

import ctypes


def get_dpi_scale() -> float:
    """Return a UI scale factor based on the primary monitor DPI.

    On standard 96-dpi screens this returns 1.0.
    On 4K / HiDPI screens (typically 144-192 dpi) it returns 1.5-2.0+.
    Falls back to 1.0 if DPI detection fails.
    """
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        hdc = ctypes.windll.user32.GetDC(0)
        dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX
        ctypes.windll.user32.ReleaseDC(0, hdc)
        return max(dpi / 96.0, 1.0)
    except Exception:
        return 1.0


# ── Colour palette (dark-mode inspired) ────────────────────────
BG          = "#1e1e2e"
BG_FRAME    = "#2a2a3c"
BG_GROUP    = "#33334d"
FG          = "#cdd6f4"
FG_DIM      = "#6c7086"
ACCENT      = "#89b4fa"
ACCENT_DARK = "#585b70"
PAUSE_ON    = "#a6e3a1"
STOP_BG     = "#f38ba8"
BTN_BG      = "#45475a"
BTN_FG      = "#cdd6f4"

# Number of columns for the reward-channel grid.
CHANNEL_COLS = 3
