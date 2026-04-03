"""
Vision encoder — MobileNetV2-style depthwise-separable CNN.

Converts raw image frames (B, C, H, W) → (B, embed_dim) embeddings.
Designed to be tiny (~1-3M params for student, ~5-10M for teacher).

When ``use_slot_attention=True`` the GAP+Linear head is replaced by a
SlotAttentionVisionHead that extracts K object-centric slot embeddings
before pooling.  This improves object binding, which is critical for
Minecraft's multi-object inventory/scene understanding.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_ai.encoders.slot_attention import SlotAttentionVisionHead


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
            groups=in_ch, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.bn1(self.depthwise(x)))
        x = F.relu6(self.bn2(self.pointwise(x)))
        return x


class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block with expansion."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expand_ratio: int = 2):
        super().__init__()
        mid = in_ch * expand_ratio
        # Residual connections only work when stride=1 (spatial dims
        # preserved) AND in_ch==out_ch (channel dims match for addition).
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            # Depthwise
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
            # Pointwise linear
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_residual:
            out = out + x
        return out


class VisionEncoder(nn.Module):
    """
    Lightweight MobileNetV2-inspired vision encoder.

    Args:
        in_channels:        Input channels (3 for RGB, 1 for grayscale).
        embed_dim:          Output embedding dimension.
        width_mult:         Width multiplier to scale channel counts.
        use_slot_attention: If True, replace GAP head with SlotAttentionVisionHead.
        num_slots:          Number of object slots (only used when use_slot_attention=True).
        slot_dim:           Slot feature dimension (only used when use_slot_attention=True).
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 128,
        width_mult: float = 0.5,
        use_slot_attention: bool = False,
        num_slots: int = 8,
        slot_dim: int = 64,
    ):
        super().__init__()
        self.use_slot_attention = use_slot_attention

        def _ch(c: int) -> int:
            return max(8, int(c * width_mult))

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, _ch(32), 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_ch(32)),
            nn.ReLU6(inplace=True),
        )

        # Inverted residual blocks — aggressively small
        # (expand, out_ch, stride, repeats)
        block_configs = [
            (1, _ch(16),  1, 1),
            (2, _ch(24),  2, 2),
            (2, _ch(32),  2, 2),
            (2, _ch(64),  2, 2),
            (2, _ch(96),  1, 1),
            (2, _ch(160), 2, 1),
        ]

        blocks = []
        in_ch = _ch(32)
        for expand, out_ch, stride, n_repeat in block_configs:
            for i in range(n_repeat):
                s = stride if i == 0 else 1
                blocks.append(InvertedResidual(in_ch, out_ch, stride=s, expand_ratio=expand))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        # Head: either GAP+linear (default) or Slot Attention
        if use_slot_attention:
            self.head = SlotAttentionVisionHead(
                in_channels=in_ch,
                num_slots=num_slots,
                slot_dim=slot_dim,
                output_dim=embed_dim,
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(in_ch, _ch(256), 1, bias=False),
                nn.BatchNorm2d(_ch(256)),
                nn.ReLU6(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(_ch(256), embed_dim),
            )
        self._last_slots: torch.Tensor | None = None  # exposed for downstream use

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor, values in [0, 1] or normalized.

        Returns:
            (B, embed_dim) embedding.
        """
        x = self.stem(x)
        x = self.blocks(x)
        if self.use_slot_attention:
            embed, slots = self.head(x)
            self._last_slots = slots  # (B, K, slot_dim) — stored for downstream access
            return embed
        return self.head(x)
