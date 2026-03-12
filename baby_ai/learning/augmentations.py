"""
Observation augmentations for distillation.

Applies stochastic transforms to the *Student's* input during distillation
while the Teacher sees clean (unaugmented) observations.  This "asymmetric
augmentation" strategy forces the Student to learn robust representations
and prevents overfitting to the narrow observation distribution in replay.

References:
    - BYOL (Grill et al., 2020): asymmetric augmentation for self-supervised learning
    - DAgger (Ross et al., 2011): distribution shift mitigation in imitation

All transforms operate on batched (B, C, H, W) float tensors in [0, 1] range
and are designed to be GPU-friendly (no CPU round-trips).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class DistillAugmentor:
    """
    GPU-friendly observation augmentations for distillation.

    During distillation the Teacher sees clean frames and the Student
    sees augmented frames.  This prevents the Student from memorising
    pixel-level artefacts that don't generalise.

    Augmentations applied (stochastically):
        1. Random crop + resize  (spatial robustness)
        2. Color jitter           (lighting / gamma robustness)
        3. Gaussian noise         (sensor noise robustness)

    Each augmentation has an independent probability of being applied,
    so on any given batch the Student might see none, some, or all of them.

    Args:
        crop_ratio: Fraction of image area to keep (e.g. 0.85 = 85%).
        jitter_strength: Maximum brightness/contrast shift magnitude.
        noise_std: Std-dev of additive Gaussian noise.
        crop_prob: Probability of applying random crop.
        jitter_prob: Probability of applying color jitter.
        noise_prob: Probability of applying Gaussian noise.
    """

    def __init__(
        self,
        crop_ratio: float = 0.85,
        jitter_strength: float = 0.15,
        noise_std: float = 0.02,
        crop_prob: float = 0.5,
        jitter_prob: float = 0.4,
        noise_prob: float = 0.3,
    ):
        self.crop_ratio = crop_ratio
        self.jitter_strength = jitter_strength
        self.noise_std = noise_std
        self.crop_prob = crop_prob
        self.jitter_prob = jitter_prob
        self.noise_prob = noise_prob

    @torch.no_grad()
    def __call__(self, vision: torch.Tensor) -> torch.Tensor:
        """
        Apply stochastic augmentations to a vision batch.

        Args:
            vision: (B, C, H, W) float tensor, expected range [0, 1].

        Returns:
            Augmented (B, C, H, W) tensor, same device and dtype.
        """
        x = vision

        # 1. Random crop + resize back to original size
        if torch.rand(1).item() < self.crop_prob:
            x = self._random_crop_resize(x)

        # 2. Color jitter (brightness + contrast shift)
        if torch.rand(1).item() < self.jitter_prob:
            x = self._color_jitter(x)

        # 3. Additive Gaussian noise
        if torch.rand(1).item() < self.noise_prob:
            x = self._gaussian_noise(x)

        return x

    def _random_crop_resize(self, x: torch.Tensor) -> torch.Tensor:
        """Crop a random sub-region and resize back to original dims."""
        B, C, H, W = x.shape
        crop_h = int(H * self.crop_ratio)
        crop_w = int(W * self.crop_ratio)

        # Random top-left corner (same for whole batch — fast)
        top = torch.randint(0, H - crop_h + 1, (1,)).item()
        left = torch.randint(0, W - crop_w + 1, (1,)).item()

        cropped = x[:, :, top:top + crop_h, left:left + crop_w]
        # Bilinear resize back to original resolution
        return F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)

    def _color_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Random brightness and contrast adjustment."""
        s = self.jitter_strength
        # Brightness: shift all pixels by a random offset
        brightness = (torch.rand(1, device=x.device) * 2 - 1) * s  # [-s, +s]
        x = x + brightness
        # Contrast: scale deviation from mean
        contrast = 1.0 + (torch.rand(1, device=x.device) * 2 - 1) * s  # [1-s, 1+s]
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = (x - mean) * contrast + mean
        return x.clamp(0.0, 1.0)

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(x) * self.noise_std
        return (x + noise).clamp(0.0, 1.0)
