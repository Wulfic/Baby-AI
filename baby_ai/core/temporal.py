"""
Temporal integration module — GRU-based recurrent core.

Maintains a hidden state that integrates multimodal embeddings over time,
providing the backbone state representation for the agent.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalCore(nn.Module):
    """
    GRU-based temporal integration.

    Takes fused multimodal embeddings at each timestep and maintains
    a recurrent hidden state for sequential decision making.

    Args:
        input_dim: Dimension of input (fused embedding).
        hidden_dim: GRU hidden state dimension.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout between GRU layers (only if num_layers > 1).
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Pre-norm for input stability
        self.input_norm = nn.LayerNorm(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Post-projection (optional, for matching downstream dims)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create zero-initialized hidden state."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim) or (B, input_dim) fused embeddings.
               If 2D, unsqueeze to (B, 1, input_dim) for single-step.
            hidden: (num_layers, B, hidden_dim) previous hidden state.

        Returns:
            output: (B, T, hidden_dim) or (B, hidden_dim) GRU output.
            hidden: (num_layers, B, hidden_dim) updated hidden state.
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, input_dim)
            squeeze = True

        x = self.input_norm(x)

        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        output, hidden = self.gru(x, hidden)
        output = self.output_proj(output)

        if squeeze:
            output = output.squeeze(1)  # (B, hidden_dim)

        return output, hidden

    def detach_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Detach hidden state from computation graph (for TBPTT)."""
        return hidden.detach()
