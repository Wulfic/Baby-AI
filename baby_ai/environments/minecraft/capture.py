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
                    to the model's expected input size (640×360).
        normalize: Scale pixel values to [0, 1] when returning tensors.
    """

    def __init__(
        self,
        window: WindowManager,
        resolution: Tuple[int, int] = (360, 640),
        normalize: bool = True,
    ):
        self._window = window
        self._resolution = resolution  # (H, W)
        self._normalize = normalize

        # Cached region dict (mss format): updated on each grab
        self._region: Optional[dict] = None

        # Track the initial client-area size so we can detect resizes
        # and log a warning.  The capture pipeline handles arbitrary
        # sizes gracefully (all coordinates are fractional), but it's
        # good to know when the window has been resized.
        self._initial_size: Optional[Tuple[int, int]] = None

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
            log.warning("Minecraft client area has zero size. Returning black frame.")
            return np.zeros((self._resolution[0], self._resolution[1], 3), dtype=np.uint8)

        self._region = {"left": sx, "top": sy, "width": sw, "height": sh}
        shot = sct.grab(self._region)

        # mss returns BGRA (Blue, Green, Red, Alpha); drop alpha channel.
        # We keep BGR order here because cv2 (used downstream) expects BGR.
        # RGB conversion happens only when building the model tensor.
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
        # Convert BGR (OpenCV/mss convention) → RGB (PyTorch convention)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # (H, W, 3) uint8 → (3, H, W) float32
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()

        if self._normalize:
            tensor = tensor / 255.0

        return tensor.unsqueeze(0)  # (1, C, H, W)

    def grab_both(self) -> Tuple[np.ndarray, torch.Tensor]:
        """Return ``(native_bgr, model_tensor)`` from a single capture.

        ``native_bgr`` is at the Minecraft window's native client-area
        resolution (e.g. 854 x 480) so that screen-region analysers have
        enough pixel detail to detect hotbar changes, block-crack
        animations, etc.

        ``model_tensor`` is resized to ``self._resolution`` (typically
        640 x 360) for the neural-network forward pass.
        """
        sct = _get_mss()

        sx, sy, sw, sh = self._window.get_client_rect()
        if sw <= 0 or sh <= 0:
            log.warning("Minecraft client area has zero size. Returning black frame.")
            black = np.zeros(
                (self._resolution[0], self._resolution[1], 3), dtype=np.uint8,
            )
            tensor = torch.zeros(
                1, 3, self._resolution[0], self._resolution[1],
            )
            return black, tensor

        # ── Window-resize detection ────────────────────────────
        current_size = (sh, sw)
        if self._initial_size is None:
            self._initial_size = current_size
            log.info("Initial capture size: %dx%d", sw, sh)
        elif current_size != self._initial_size:
            log.warning(
                "Minecraft window resized from %dx%d to %dx%d — "
                "capture pipeline adapts automatically (fractional coords).",
                self._initial_size[1], self._initial_size[0], sw, sh,
            )
            self._initial_size = current_size

        self._region = {"left": sx, "top": sy, "width": sw, "height": sh}
        shot = sct.grab(self._region)

        # Native-resolution BGR (no resize) — screen analysers need full
        # pixel detail for hotbar diff and death-screen colour analysis.
        native_bgr = np.array(shot, dtype=np.uint8)[..., :3]

        # Downscale to model resolution for the vision encoder
        model_bgr = cv2.resize(
            native_bgr,
            (self._resolution[1], self._resolution[0]),
        )
        rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
        if self._normalize:
            tensor = tensor / 255.0

        return native_bgr, tensor.unsqueeze(0)

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        self._resolution = value
