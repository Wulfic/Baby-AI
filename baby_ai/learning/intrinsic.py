"""
Intrinsic Curiosity Module (ICM) + Learning Progress estimator.

Provides intrinsic reward signals based on:
1. Forward prediction error (novelty / curiosity)
2. Learning progress (reduction in prediction error over time)

These drive the agent to explore and learn even without external rewards.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (Pathak et al., 2017).

    Uses the PredictiveHead's forward/inverse models to compute:
    - Curiosity reward: forward model prediction error
    - Inverse model accuracy (ensures features are useful)

    Args:
        state_dim: Fused embedding dimension.
        action_dim: Number of discrete actions.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Feature encoder: projects raw fused embeddings to a learned feature space
        # where curiosity is measured (avoids trivially unpredictable noise)
        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Forward model: predict next state features from current features + action
        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Inverse model: predict action from consecutive feature pairs
        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
        action: torch.Tensor,
    ) -> dict:
        """
        Compute ICM outputs.

        Args:
            state: (B, state_dim) current fused embedding.
            next_state: (B, state_dim) next fused embedding.
            action: (B,) discrete action taken.

        Returns:
            dict with:
                curiosity_reward: (B,) prediction error as reward.
                forward_loss: scalar, forward model MSE.
                inverse_loss: scalar, inverse model CE.
                inverse_accuracy: scalar, inverse model accuracy.
        """
        # Encode features
        phi_s = self.feature_encoder(state)      # (B, hidden)
        phi_ns = self.feature_encoder(next_state)  # (B, hidden)

        # Forward model
        action_oh = F.one_hot(action.long(), num_classes=self.action_dim).float()
        forward_input = torch.cat([phi_s, action_oh], dim=-1)
        phi_ns_pred = self.forward_model(forward_input)  # (B, hidden)

        # Curiosity reward = L2 prediction error
        forward_error = 0.5 * (phi_ns_pred - phi_ns.detach()).pow(2).sum(dim=-1)  # (B,)
        forward_loss = forward_error.mean()

        # Inverse model
        inverse_input = torch.cat([phi_s, phi_ns], dim=-1)
        action_logits = self.inverse_model(inverse_input)  # (B, action_dim)
        inverse_loss = F.cross_entropy(action_logits, action.long())
        inverse_acc = (action_logits.argmax(dim=-1) == action.long()).float().mean()

        return {
            "curiosity_reward": forward_error.detach(),
            "forward_loss": forward_loss,
            "inverse_loss": inverse_loss,
            "inverse_accuracy": inverse_acc,
        }


class LearningProgressEstimator:
    """
    Tracks learning progress as the reduction in prediction error over time.

    Learning progress = moving average of (old_error - new_error).
    Positive = model is improving → still interesting to learn from.
    Zero or negative = fully learned or getting worse.

    This is used to prioritize replay transitions for distillation.
    """

    def __init__(self, window: int = 1000, ema_alpha: float = 0.01):
        self._window = window
        self._alpha = ema_alpha
        self._error_history: dict[str, deque] = {}
        self._ema: dict[str, float] = {}

    def update(self, key: str, error: float) -> float:
        """
        Record a new prediction error for a transition/key.

        Args:
            key: Transition identifier.
            error: Current prediction error.

        Returns:
            Learning progress estimate (positive = improving).
        """
        if key not in self._error_history:
            self._error_history[key] = deque(maxlen=self._window)
            self._ema[key] = error

        old_ema = self._ema[key]
        self._ema[key] = self._alpha * error + (1 - self._alpha) * old_ema
        progress = old_ema - self._ema[key]

        self._error_history[key].append(error)
        return progress

    def get_progress(self, key: str) -> float:
        """Get current learning progress estimate for a key."""
        if key not in self._error_history or len(self._error_history[key]) < 2:
            return float("inf")  # Unknown → high priority

        hist = list(self._error_history[key])
        mid = len(hist) // 2
        old_mean = sum(hist[:mid]) / max(mid, 1)
        new_mean = sum(hist[mid:]) / max(len(hist) - mid, 1)
        return old_mean - new_mean

    def batch_priorities(self, keys: list[str]) -> list[float]:
        """Get prioritization scores for a batch of transitions."""
        return [max(self.get_progress(k), 1e-6) for k in keys]
