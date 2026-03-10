"""
JEPA-based Intrinsic Curiosity + Learning Progress estimator.

Provides intrinsic reward signals based on:
1. Latent prediction error in JEPA space (curiosity) — filters stochastic noise
2. Learning progress (reduction in latent prediction error over time)

These drive the agent to explore and learn even without external rewards.
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

    Unlike classic ICM, there is no inverse model — curiosity is purely
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
            action:     (B, action_dim) continuous action vector.

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
# Learning Progress Estimator
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
