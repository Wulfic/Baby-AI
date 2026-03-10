"""
JEPA-based Intrinsic Curiosity + Learning Progress estimator.

Provides intrinsic reward signals based on:
1. Latent prediction error in JEPA space (curiosity) — filters stochastic noise
2. Learning progress (reduction in latent prediction error over time)

These drive the agent to explore and learn even without external rewards.

The legacy ICM class is retained for backward compatibility.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# JEPA-based Curiosity Module
# ────────────────────────────────────────────────────────────────────────────

class JEPACuriosity(nn.Module):
    """
    JEPA-style intrinsic curiosity module.

    Computes curiosity reward as the L2 prediction error between
    the LatentWorldModel's predicted next-latent and the EMA target
    encoder's actual next-latent.  This filters out stochastic noise
    (weather, random entity movement) because the target encoder
    only captures learnable structure.

    Unlike ICM, there is no inverse model — curiosity is purely
    driven by forward latent dynamics error.

    This module does NOT own its own encoders; it delegates to the
    LatentWorldModel that already maintains online + target encoders.

    Args:
        world_model: The shared LatentWorldModel from core/predictive.py.
        reward_scale: Multiplier on the raw prediction error.
        max_reward: Clamp curiosity reward to prevent spikes.
    """

    def __init__(
        self,
        world_model: nn.Module,
        reward_scale: float = 1.0,
        max_reward: float = 5.0,
    ):
        super().__init__()
        self.world_model = world_model
        self.reward_scale = reward_scale
        self.max_reward = max_reward

    def forward(
        self,
        core_state: torch.Tensor,
        next_fused: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Compute JEPA curiosity outputs.

        Args:
            core_state: (B, state_dim) from temporal core at time t.
            next_fused: (B, state_dim) fused embedding at time t+1.
            action:     (B,) discrete action taken.

        Returns:
            dict with:
                curiosity_reward: (B,) scaled latent prediction error.
                dynamics_loss:    Scalar latent dynamics loss (for training).
                kl_loss:          Scalar KL loss (from stochastic path).
        """
        wm_out = self.world_model(core_state, next_fused, action)

        # Scale and clamp
        raw_reward = wm_out["curiosity_reward"]  # (B,)
        curiosity = torch.clamp(raw_reward * self.reward_scale, max=self.max_reward)

        return {
            "curiosity_reward": curiosity,
            "dynamics_loss": wm_out["dynamics_loss"],
            "kl_loss": wm_out["kl_loss"],
        }


# ────────────────────────────────────────────────────────────────────────────
# Legacy ICM (deprecated — kept for backward compatibility)
# ────────────────────────────────────────────────────────────────────────────

class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (Pathak et al., 2017)  *(deprecated)*.

    Uses forward/inverse models to compute:
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

        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
        phi_s = self.feature_encoder(state)
        phi_ns = self.feature_encoder(next_state)

        action_oh = F.one_hot(action.long(), num_classes=self.action_dim).float()
        forward_input = torch.cat([phi_s, action_oh], dim=-1)
        phi_ns_pred = self.forward_model(forward_input)

        forward_error = 0.5 * (phi_ns_pred - phi_ns.detach()).pow(2).mean(dim=-1)
        forward_loss = forward_error.mean()

        inverse_input = torch.cat([phi_s, phi_ns], dim=-1)
        action_logits = self.inverse_model(inverse_input)
        inverse_loss = F.cross_entropy(action_logits, action.long())
        inverse_acc = (action_logits.argmax(dim=-1) == action.long()).float().mean()

        return {
            "curiosity_reward": forward_error.detach(),
            "forward_loss": forward_loss,
            "inverse_loss": inverse_loss,
            "inverse_accuracy": inverse_acc,
        }


# ────────────────────────────────────────────────────────────────────────────
# Learning Progress Estimator (shared by ICM and JEPA)
# ────────────────────────────────────────────────────────────────────────────

class LearningProgressEstimator:
    """
    Tracks learning progress as the reduction in prediction error over time.

    Learning progress = moving average of (old_error - new_error).
    Positive = model is improving → still interesting to learn from.
    Zero or negative = fully learned or getting worse.

    Used to prioritize replay transitions for distillation.
    """

    def __init__(self, window: int = 1000, ema_alpha: float = 0.01):
        self._window = window
        self._alpha = ema_alpha
        self._error_history: dict[str, deque] = {}
        self._ema: dict[str, float] = {}

    def update(self, key: str, error: float) -> float:
        if key not in self._error_history:
            self._error_history[key] = deque(maxlen=self._window)
            self._ema[key] = error

        old_ema = self._ema[key]
        self._ema[key] = self._alpha * error + (1 - self._alpha) * old_ema
        progress = old_ema - self._ema[key]

        self._error_history[key].append(error)
        return progress

    def get_progress(self, key: str) -> float:
        if key not in self._error_history or len(self._error_history[key]) < 2:
            return float("inf")

        hist = list(self._error_history[key])
        mid = len(hist) // 2
        old_mean = sum(hist[:mid]) / max(mid, 1)
        new_mean = sum(hist[mid:]) / max(len(hist) - mid, 1)
        return old_mean - new_mean

    def batch_priorities(self, keys: list[str]) -> list[float]:
        return [max(self.get_progress(k), 1e-6) for k in keys]
