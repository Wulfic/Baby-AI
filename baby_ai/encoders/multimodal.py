"""
Multimodal fusion encoder.

Combines embeddings from vision, audio, code, and sensor encoders
into a single fused representation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """
    Gated fusion of multiple modality embeddings.

    Uses a learned gating mechanism to weight modality contributions,
    allowing the model to attend to the most informative modalities
    at each timestep.

    Args:
        modality_dims: Dict mapping modality name → embedding dim.
        fused_dim: Output fused embedding dimension.
    """

    def __init__(
        self,
        modality_dims: Optional[dict[str, int]] = None,
        fused_dim: int = 256,
    ):
        super().__init__()

        if modality_dims is None:
            modality_dims = {
                "vision": 128,
                "audio": 128,
                "code": 128,
                "sensor": 64,
            }

        self.modality_names = sorted(modality_dims.keys())
        total_dim = sum(modality_dims.values())

        # Per-modality projection to a common dimension
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, fused_dim)
            for name, dim in modality_dims.items()
        })

        # Gating network: takes concatenated raw embeddings → gate per modality
        self.gate = nn.Sequential(
            nn.Linear(total_dim, len(modality_dims)),
            nn.Softmax(dim=-1),
        )

        # Final projection
        self.out_proj = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
        )

        self._total_dim = total_dim

    def forward(
        self,
        embeddings: dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Dict mapping modality name → (B, modality_dim) tensor.
                        Missing modalities should be passed as None or omitted.

        Returns:
            (B, fused_dim) fused embedding.
        """
        # Determine batch size from any available modality
        batch_size = None
        device = None
        for v in embeddings.values():
            if v is not None:
                batch_size = v.shape[0]
                device = v.device
                break

        if batch_size is None:
            raise ValueError("At least one modality embedding must be provided.")

        # Project each modality; zero-fill missing ones
        projected = []
        raw_parts = []
        for name in self.modality_names:
            emb = embeddings.get(name)
            proj = self.projections[name]
            if emb is not None:
                projected.append(proj(emb))
                raw_parts.append(emb)
            else:
                # Zero embedding for missing modality
                projected.append(torch.zeros(batch_size, proj.out_features, device=device))
                raw_parts.append(torch.zeros(batch_size, proj.in_features, device=device))

        # Compute gates from raw concatenation
        raw_cat = torch.cat(raw_parts, dim=-1)  # (B, total_dim)
        gates = self.gate(raw_cat)               # (B, num_modalities)

        # Gated weighted sum of projected embeddings
        stacked = torch.stack(projected, dim=1)  # (B, num_modalities, fused_dim)
        gates = gates.unsqueeze(-1)               # (B, num_modalities, 1)
        fused = (stacked * gates).sum(dim=1)      # (B, fused_dim)

        return self.out_proj(fused)
