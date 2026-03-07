"""
Audio encoder — 1D conv on log-mel spectrograms.

Converts (B, 1, n_mels, T) log-mel spectrograms → (B, embed_dim) embeddings.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """Conv1D + BN + ReLU + optional pooling."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pool: int = 2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x))))


class AudioEncoder(nn.Module):
    """
    Lightweight 1D convolutional audio encoder.

    Processes log-mel spectrograms by treating mel bins as channels
    (after a frequency-folding strategy) or as a 2D → 1D pipeline.

    Args:
        n_mels: Number of mel frequency bins.
        embed_dim: Output embedding dimension.
        width_mult: Channel width multiplier for scaling student/teacher.
    """

    def __init__(
        self,
        n_mels: int = 64,
        embed_dim: int = 128,
        width_mult: float = 1.0,
    ):
        super().__init__()

        def _ch(c: int) -> int:
            return max(8, int(c * width_mult))

        # Treat the mel spectrogram as (B, n_mels, T) — mel bins are "channels"
        # over time dimension T. First reduce mel dimension with a linear projection.
        self.mel_proj = nn.Linear(n_mels, _ch(64))

        # 1D conv stack over time
        self.conv_stack = nn.Sequential(
            ConvBlock1D(_ch(64), _ch(64), kernel=3, pool=2),
            ConvBlock1D(_ch(64), _ch(128), kernel=3, pool=2),
            ConvBlock1D(_ch(128), _ch(128), kernel=3, pool=2),
            ConvBlock1D(_ch(128), _ch(256), kernel=3, pool=2),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(_ch(256), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_mels, T) log-mel spectrogram.

        Returns:
            (B, embed_dim) embedding.
        """
        # x: (B, n_mels, T) → transpose to (B, T, n_mels) for linear
        x = self.mel_proj(x.transpose(1, 2))  # (B, T, proj_dim)
        x = x.transpose(1, 2)                 # (B, proj_dim, T)
        x = self.conv_stack(x)
        x = self.head(x)
        return x
