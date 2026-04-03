"""
Slot Attention encoder.

Converts a spatial feature map (B, N, D) — output of a CNN backbone
before global-average pooling — into K object-centric slot embeddings
(B, K, slot_dim).

References:
  Locatello et al., "Object-Centric Learning with Slot Attention", NeurIPS 2020.

Architecture:
    Input feature map (B, N, D)
        ↓  LayerNorm + Linear projection to slot_dim
    SlotAttentionModule  (T iterations of competitive soft-attention)
        ↓  (B, K, slot_dim)
    Slot pooling → (B, slot_dim)  [mean over K slots]
        ↓  Linear projection to output_dim
    Output (B, output_dim)  [plug-in replacement for GAP-pooled vision embed]

The module also exposes raw slots (B, K, slot_dim) for richer downstream
use (e.g. per-object attention in the Jamba core).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttentionModule(nn.Module):
    """
    Single slot-attention iteration block.

    Args:
        num_slots:  K — number of object slots.
        slot_dim:   Dimension of each slot.
        input_dim:  Dimension of projected input features.
        num_iters:  Number of competitive attention iterations.
        eps:        Numerical stability for softmax normalisation.
    """

    def __init__(
        self,
        num_slots: int = 8,
        slot_dim: int = 64,
        input_dim: int = 64,
        num_iters: int = 3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iters = num_iters
        self.eps = eps
        self.scale = math.sqrt(slot_dim)

        # Slot initialisation: learned mean + log-std
        self.slot_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, slot_dim))

        # Projections
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(input_dim, slot_dim, bias=False)

        # Slot update GRU
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # Residual MLP within each slot
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (B, N, input_dim) spatial feature tokens.

        Returns:
            slots: (B, K, slot_dim)
        """
        B, N, _ = inputs.shape
        K = self.num_slots

        inputs = self.norm_inputs(inputs)  # (B, N, D)
        k = self.k_proj(inputs)            # (B, N, slot_dim)
        v = self.v_proj(inputs)            # (B, N, slot_dim)

        # Sample initial slots from learned distribution
        mu = self.slot_mu.expand(B, K, -1)
        sigma = self.slot_log_sigma.exp().expand(B, K, -1)
        slots = mu + sigma * torch.randn_like(mu)  # (B, K, slot_dim)

        for _ in range(self.num_iters):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention weights
            q = self.q_proj(slots)  # (B, K, slot_dim)
            # (B, K, N) dot-product similarity
            attn = torch.bmm(q, k.transpose(1, 2)) / self.scale  # (B, K, N)

            # Competitive: normalise over slots (columns), not inputs
            attn = F.softmax(attn, dim=1)  # (B, K, N) — competition over slots
            attn_weights = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)  # (B, K, N)

            # Weighted mean of values
            updates = torch.bmm(attn_weights, v)  # (B, K, slot_dim)

            # GRU update per slot
            slots = self.gru(
                updates.reshape(B * K, self.slot_dim),
                slots_prev.reshape(B * K, self.slot_dim),
            ).reshape(B, K, self.slot_dim)

            # Residual MLP
            slots = slots + self.mlp(slots)

        return slots  # (B, K, slot_dim)


class SlotAttentionVisionHead(nn.Module):
    """
    Drop-in replacement for the GAP+projection head in VisionEncoder.

    Wraps SlotAttentionModule to accept a spatial feature map
    (B, C, H, W) and produce a pooled embedding (B, output_dim)
    plus raw object slots (B, K, slot_dim).

    Args:
        in_channels:  Channels of the CNN feature map (before head).
        num_slots:    Number of object-centric slots.
        slot_dim:     Dimension of each slot.
        output_dim:   Final pooled embedding dimension (= vision_embed_dim).
        num_iters:    Slot attention iterations.
    """

    def __init__(
        self,
        in_channels: int = 160,
        num_slots: int = 8,
        slot_dim: int = 64,
        output_dim: int = 128,
        num_iters: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim

        # Project CNN channels to slot_dim for attention
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, slot_dim, 1, bias=False),
            nn.BatchNorm2d(slot_dim),
            nn.ReLU6(inplace=True),
        )

        self.slot_attn = SlotAttentionModule(
            num_slots=num_slots,
            slot_dim=slot_dim,
            input_dim=slot_dim,
            num_iters=num_iters,
        )

        # Pool K slots → single vector
        self.pool = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, output_dim),
        )

    def forward(
        self,
        feature_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_map: (B, C, H, W) spatial CNN features.

        Returns:
            embed:  (B, output_dim) pooled embedding.
            slots:  (B, K, slot_dim) raw object slots.
        """
        B = feature_map.size(0)
        x = self.input_proj(feature_map)  # (B, slot_dim, H, W)
        H, W = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, slot_dim)

        slots = self.slot_attn(x)         # (B, K, slot_dim)
        embed = self.pool(slots).mean(dim=1)  # (B, output_dim)

        return embed, slots
