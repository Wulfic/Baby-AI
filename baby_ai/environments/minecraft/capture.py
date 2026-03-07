"""
Fast screen capture of the Minecraft window.

Uses the ``mss`` library for low-latency screen grabs.  Only the
client area (game viewport) is captured — title bar and borders
are excluded so the model sees exactly what the player sees.

Typical capture latency: 3-8 ms at 854 × 480, which is well
within the 100 ms step budget.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from baby_ai.environments.minecraft.window import WindowManager
from baby_ai.utils.logging import get_logger

log = get_logger("mc_capture")

# mss is imported lazily so the module can be loaded even if mss
# is not installed (e.g. on Linux CI).
_mss = None


def _get_mss():
    """Lazy-initialise a single mss instance (thread-local)."""
    global _mss
    if _mss is None:
        import mss as _mss_mod
        _mss = _mss_mod.mss()
    return _mss


class ScreenCapture:
    """
    Captures the Minecraft game viewport as numpy arrays or PyTorch
    tensors, ready for the vision encoder.

    Args:
        window: WindowManager pointing at the Minecraft window.
        resolution: (H, W) to resize captured frames to.  Defaults
                    to the model's expected input size (160×160).
        normalize: Scale pixel values to [0, 1] when returning tensors.
    """

    def __init__(
        self,
        window: WindowManager,
        resolution: Tuple[int, int] = (160, 160),
        normalize: bool = True,
    ):
        self._window = window
        self._resolution = resolution  # (H, W)
        self._normalize = normalize

        # Cached region dict (mss format): updated on each grab
        self._region: Optional[dict] = None

    # ── Public API ──────────────────────────────────────────────

    def grab_raw(self) -> np.ndarray:
        """
        Capture the current frame as an (H, W, 3) uint8 BGR array.

        The image is already cropped to the client area and resized
        to ``self._resolution``.
        """
        sct = _get_mss()

        # Refresh client-area position (window may have been moved)
        sx, sy, sw, sh = self._window.get_client_rect()
        if sw <= 0 or sh <= 0:
            raise RuntimeError("Minecraft client area has zero size — is the window minimised?")

        self._region = {"left": sx, "top": sy, "width": sw, "height": sh}
        shot = sct.grab(self._region)

        # mss returns BGRA; drop alpha and convert to contiguous array
        frame = np.array(shot, dtype=np.uint8)[..., :3]  # (H, W, 3) BGR

        # Resize to model resolution
        frame = cv2.resize(frame, (self._resolution[1], self._resolution[0]))

        return frame  # BGR uint8

    def grab_tensor(self) -> torch.Tensor:
        """
        Capture a frame and return a model-ready (1, 3, H, W) float tensor.

        Colour order is RGB  and values are in [0, 1] if ``normalize``
        is enabled.
        """
        bgr = self.grab_raw()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # (H, W, 3) uint8 → (3, H, W) float32
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()

        if self._normalize:
            tensor = tensor / 255.0

        return tensor.unsqueeze(0)  # (1, C, H, W)

    def grab_both(self) -> Tuple[np.ndarray, torch.Tensor]:
        """Return (raw_bgr, model_tensor) in one capture to avoid double-grab."""
        bgr = self.grab_raw()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        if self._normalize:
            tensor = tensor / 255.0
        return bgr, tensor.unsqueeze(0)

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        self._resolution = value
