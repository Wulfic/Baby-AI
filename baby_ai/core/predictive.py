"""
Predictive head — next-state prediction for intrinsic motivation.

Predicts the next fused embedding given current state + action,
used by the ICM module to compute prediction error (curiosity signal).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveHead(nn.Module):
    """
    Forward dynamics model: predicts next state embedding from
    current state + action.

    Also includes an inverse model for feature learning stability
    (predicts action from consecutive states).

    Args:
        state_dim: Dimension of the fused state embedding.
        action_dim: Number of discrete actions (for one-hot encoding).
        hidden_dim: Hidden layer size.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()

        # Forward model: (state, action) → predicted next state
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # Inverse model: (state, next_state) → predicted action
        self.inverse_model = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.action_dim = action_dim

    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward dynamics: predict next state embedding.

        Args:
            state: (B, state_dim) current state embedding.
            action: (B,) discrete action indices.

        Returns:
            (B, state_dim) predicted next state embedding.
        """
        # One-hot encode action
        action_oh = F.one_hot(action.long(), num_classes=self.action_dim).float()
        x = torch.cat([state, action_oh], dim=-1)
        return self.forward_model(x)

    def predict_action(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inverse dynamics: predict action from state transition.

        Args:
            state: (B, state_dim) current state.
            next_state: (B, state_dim) next state.

        Returns:
            (B, action_dim) action logits.
        """
        x = torch.cat([state, next_state], dim=-1)
        return self.inverse_model(x)

    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Returns:
            predicted_next: (B, state_dim) predicted next state.
            action_logits: (B, action_dim) predicted action from inverse model.
        """
        predicted_next = self.predict_next_state(state, action)
        action_logits = self.predict_action(state, next_state)
        return predicted_next, action_logits
