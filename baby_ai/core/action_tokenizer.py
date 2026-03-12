"""
VQ-BeT: Vector-Quantized Behavior Transformer action tokenizer.

Converts continuous action vectors into discrete codebook indices
using Residual Vector Quantization, capturing multimodal action
distributions (mining, building, exploring, combat, etc.).

Architecture:
    Encoder:  23-dim action → project to code_dim → Residual VQ → indices
    Decoder:  codebook vectors → project back → 23-dim action

The tokenizer sits *after* the policy head:
    core_state → FlowMatchingHead → 23-dim continuous → VQ Tokenizer → codebook indices

Reference: arXiv 2403.03181 (VQ-BeT)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Single VQ layer with EMA codebook updates.

    Encoder: continuous vector → nearest codebook embedding
    Decoder: codebook embedding → quantized vector

    Uses straight-through estimator for gradient flow and optional
    EMA updates for codebook stability (avoids codebook collapse).
    """

    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 64,
        ema_decay: float = 0.99,
        commitment_weight: float = 0.25,
        ema_update: bool = True,
    ):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.ema_update = ema_update
        self.ema_decay = ema_decay

        # Codebook embeddings
        self.embedding = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1 / num_codes, 1 / num_codes)

        if ema_update:
            # EMA tracking buffers (not learnable parameters)
            self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
            self.register_buffer(
                "_ema_embed_sum", self.embedding.weight.data.clone()
            )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize input vectors to nearest codebook entries.

        Args:
            z: (B, code_dim) continuous vectors to quantize.

        Returns:
            z_q:     (B, code_dim) quantized vectors (straight-through grad).
            indices: (B,) codebook indices.
            vq_loss: Scalar commitment + codebook loss.
        """
        # Find nearest codebook vector via L2 distance
        # distances: (B, num_codes)
        distances = torch.cdist(
            z.unsqueeze(0), self.embedding.weight.unsqueeze(0)
        ).squeeze(0)
        indices = distances.argmin(dim=-1)  # (B,)
        z_q = self.embedding(indices)  # (B, code_dim)

        if self.training:
            if self.ema_update:
                self._ema_update(z, indices)
                # With EMA, only commitment loss is needed (codebook updates implicitly)
                vq_loss = self.commitment_weight * F.mse_loss(z.detach(), z_q)
            else:
                # Standard VQ-VAE loss: codebook + commitment
                codebook_loss = F.mse_loss(z_q, z.detach())
                commitment_loss = F.mse_loss(z_q.detach(), z)
                vq_loss = codebook_loss + self.commitment_weight * commitment_loss
        else:
            vq_loss = torch.tensor(0.0, device=z.device)

        # Straight-through estimator: gradient flows through z_q as if it were z
        z_q = z + (z_q - z).detach()

        return z_q, indices, vq_loss

    def _ema_update(self, z: torch.Tensor, indices: torch.Tensor) -> None:
        """Update codebook via exponential moving averages (Laplace smoothing)."""
        one_hot = F.one_hot(indices, self.num_codes).float()  # (B, K)
        cluster_size = one_hot.sum(dim=0)  # (K,)
        embed_sum = one_hot.T @ z  # (K, code_dim)

        self._ema_cluster_size.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        self._ema_embed_sum.mul_(self.ema_decay).add_(
            embed_sum, alpha=1 - self.ema_decay
        )

        # Laplace smoothing to prevent division by zero
        n = self._ema_cluster_size.sum()
        smoothed = (
            (self._ema_cluster_size + 1e-5)
            / (n + self.num_codes * 1e-5)
            * n
        )
        self.embedding.weight.data.copy_(
            self._ema_embed_sum / smoothed.unsqueeze(1)
        )


class ResidualVQ(nn.Module):
    """
    Hierarchical Residual Vector Quantization.

    Applies multiple VQ layers in sequence, each quantizing the
    residual from the previous layer.  This captures coarse behavior
    modes at level 0 (mine, build, explore) and fine adjustments
    at subsequent levels (exact camera angle, hotbar slot).
    """

    def __init__(
        self,
        num_levels: int = 2,
        num_codes: int = 512,
        code_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.levels = nn.ModuleList(
            [
                VectorQuantizer(num_codes=num_codes, code_dim=code_dim, **kwargs)
                for _ in range(num_levels)
            ]
        )

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Args:
            z: (B, code_dim) continuous vectors.

        Returns:
            z_q:         (B, code_dim) final quantized vector (sum of all levels).
            all_indices: List of (B,) index tensors, one per RVQ level.
            total_loss:  Scalar sum of VQ losses across levels.
        """
        residual = z
        z_q = torch.zeros_like(z)
        all_indices: list[torch.Tensor] = []
        total_loss = torch.tensor(0.0, device=z.device)

        for vq in self.levels:
            z_q_level, indices, vq_loss = vq(residual)
            z_q = z_q + z_q_level
            residual = residual - z_q_level.detach()
            all_indices.append(indices)
            total_loss = total_loss + vq_loss

        return z_q, all_indices, total_loss


class ActionTokenizer(nn.Module):
    """
    Full VQ-BeT action tokenizer.

    Encoder: 23-dim action → project → Residual VQ → codebook indices
    Decoder: codebook vectors → project → 23-dim action

    Wraps around the policy: the policy generates continuous actions,
    the tokenizer discretizes them, and the decoder maps back to
    the continuous action space expected by the ActionDecoder.

    Args:
        action_dim:        Continuous action vector size (default 23).
        code_dim:          Internal codebook embedding dimension.
        num_codes:         Number of codebook entries per VQ level.
        num_residual:      Number of hierarchical VQ levels.
        commitment_weight: VQ commitment loss weight.
        ema_update:        Use EMA codebook updates (recommended).
        ema_decay:         EMA decay rate.
    """

    def __init__(
        self,
        action_dim: int = 23,
        code_dim: int = 64,
        num_codes: int = 512,
        num_residual: int = 2,
        commitment_weight: float = 0.25,
        ema_update: bool = True,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.code_dim = code_dim

        # Encoder: action → latent code space
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
        )

        # Residual VQ
        self.rvq = ResidualVQ(
            num_levels=num_residual,
            num_codes=num_codes,
            code_dim=code_dim,
            commitment_weight=commitment_weight,
            ema_update=ema_update,
            ema_decay=ema_decay,
        )

        # Decoder: latent → action space
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, action_dim),
        )

    def encode(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Encode continuous actions → codebook indices.

        Args:
            action: (B, action_dim) continuous action vectors.

        Returns:
            z_q:     (B, code_dim) quantized latent vectors.
            indices: List of (B,) index tensors per RVQ level.
            vq_loss: Scalar VQ loss.
        """
        z = self.encoder(action)
        z_q, indices, vq_loss = self.rvq(z)
        return z_q, indices, vq_loss

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latent → continuous action.

        Args:
            z_q: (B, code_dim) quantized latent vectors.

        Returns:
            (B, action_dim) reconstructed continuous actions.
        """
        return self.decoder(z_q)

    def forward(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Full encode → VQ → decode pass (for training the tokenizer).

        Args:
            action: (B, action_dim) ground-truth continuous actions.

        Returns:
            reconstructed: (B, action_dim) reconstructed actions.
            indices:       List of (B,) index tensors per RVQ level.
            total_loss:    Scalar VQ loss + reconstruction MSE.
        """
        z_q, indices, vq_loss = self.encode(action)
        reconstructed = self.decode(z_q)
        recon_loss = F.mse_loss(reconstructed, action)
        return reconstructed, indices, vq_loss + recon_loss

    def decode_from_indices(
        self, indices_list: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Decode directly from codebook indices (for inference).

        Args:
            indices_list: List of (B,) index tensors, one per RVQ level.

        Returns:
            (B, action_dim) decoded continuous actions.
        """
        z_q = torch.zeros(
            indices_list[0].size(0),
            self.code_dim,
            device=indices_list[0].device,
        )
        for indices, vq in zip(indices_list, self.rvq.levels):
            z_q = z_q + vq.embedding(indices)
        return self.decoder(z_q)
