"""
System 2 Planner — Latent Monte Carlo Tree Search.

When uncertainty spikes (novel situations, hostile mobs, complex crafting),
the agent pauses System 1 (fast reactive policy) and allocates ~150ms to
simulate futures in its "mind" using the LatentWorldModel.

The planner:
  1. Proposes N candidate first-actions via the DiffusionPolicy.
  2. Rolls out T latent steps per trajectory using the WorldModel.
  3. Scores each trajectory via the value head.
  4. Returns the first action of the highest-scoring trajectory.

All operations are batched tensor ops for GPU efficiency.
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn


class LatentMCTS(nn.Module):
    """
    Batched latent-space Monte Carlo Tree Search.

    Uses the LatentWorldModel for forward simulation and the
    DiffusionPolicyHead's value head for trajectory scoring.
    All rollouts are executed as a single batched tensor operation.

    Args:
        world_model:      The shared LatentWorldModel.
        policy:           The DiffusionPolicyHead (for action proposals + value).
        num_trajectories: Number of parallel trajectories to evaluate.
        horizon:          Number of latent steps per trajectory.
        discount:         Discount factor for trajectory returns.
        budget_ms:        Maximum planning time in milliseconds.
    """

    def __init__(
        self,
        world_model: nn.Module,
        policy: nn.Module,
        num_trajectories: int = 8,
        horizon: int = 5,
        discount: float = 0.99,
        budget_ms: float = 150.0,
    ):
        super().__init__()
        self.world_model = world_model
        self.policy = policy
        self.num_trajectories = num_trajectories
        self.horizon = horizon
        self.discount = discount
        self.budget_ms = budget_ms

    @torch.no_grad()
    def plan(
        self,
        core_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Run batched latent planning from the current state.

        Args:
            core_state: (1, state_dim) current temporal core output.

        Returns:
            dict with:
                best_action:    (1, action_dim) best first action.
                expected_value: (1,) expected return of best trajectory.
                uncertainty:    (1,) value spread across trajectories.
                planning_ms:    float, time taken.
                num_complete:   int, trajectories completed within budget.
        """
        t_start = time.perf_counter()
        device = core_state.device
        N = self.num_trajectories

        # Replicate current state for N parallel trajectories
        state = core_state.expand(N, -1).contiguous()  # (N, state_dim)

        # Storage for first actions and cumulative returns
        first_actions: torch.Tensor | None = None
        returns = torch.zeros(N, device=device)

        # ── Latent rollout ──
        for t in range(self.horizon):
            # Check time budget
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            if elapsed_ms > self.budget_ms:
                break

            # Propose actions from diffusion policy (DDIM)
            actions, _, _ = self.policy.act(state, deterministic=False)  # (N, action_dim)

            if t == 0:
                first_actions = actions.clone()

            # Forward dynamics via world model (continuous actions supported)
            next_latent = self.world_model.predict_next_latent(state, actions)

            # Score the new state via value head
            step_value = self.policy.value_head(next_latent).squeeze(-1)  # (N,)
            returns += (self.discount ** t) * step_value

            # Advance state (use predicted latent as next state for planning)
            state = next_latent

        # ── Select best trajectory ──
        if first_actions is None:
            # Budget expired before even one step — fall back to policy
            first_actions, _, _ = self.policy.act(core_state, deterministic=False)
            returns = torch.zeros(1, device=device)

        best_idx = returns.argmax()
        best_action = first_actions[best_idx].unsqueeze(0)   # (1, action_dim)
        expected_value = returns[best_idx].unsqueeze(0)       # (1,)

        # Uncertainty = std of trajectory returns
        uncertainty = returns.std() if N > 1 else torch.tensor(0.0, device=device)

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        return {
            "best_action": best_action,
            "expected_value": expected_value,
            "uncertainty": uncertainty,
            "planning_ms": elapsed_ms,
            "num_trajectories": N,
            "horizon_reached": min(self.horizon, max(1, int(elapsed_ms / (self.budget_ms / self.horizon)))),
        }

    # _continuous_to_discrete removed — world model now accepts
    # continuous action vectors via its action_proj layer.


class UncertaintyEstimator:
    """
    Estimates agent uncertainty to trigger System 2 planning.

    Computes a rolling estimate of how uncertain the agent is about
    its current situation. When uncertainty exceeds a threshold,
    the inference thread should invoke the LatentMCTS planner.

    Uncertainty signals:
      1. Diffusion denoising variance across action samples.
      2. World model prediction error spike.
      3. Value prediction spread across imagined trajectories.
    """

    def __init__(self, threshold: float = 0.5, ema_alpha: float = 0.1):
        self.threshold = threshold
        self._ema_alpha = ema_alpha
        self._running_uncertainty: float = 0.0

    def update(
        self,
        value: torch.Tensor | None = None,
        log_prob: torch.Tensor | None = None,
    ) -> float:
        """
        Update uncertainty estimate from the latest inference result.

        Args:
            value:    (B, 1) state value — low absolute value = uncertain.
            log_prob: (B,) log probability — low = uncertain.

        Returns:
            Current uncertainty estimate (0 = certain, higher = uncertain).
        """
        u = 0.0

        # Low log_prob means the policy is uncertain about which action to take
        if log_prob is not None:
            # Negate and normalise: large negative log_prob → high uncertainty
            u += torch.clamp(-log_prob.mean(), 0.0, 5.0).item() / 5.0

        # Low absolute value means the model can't distinguish good/bad states
        if value is not None:
            # Small |value| = uncertain about state quality
            u += torch.clamp(1.0 - value.abs().mean(), 0.0, 1.0).item()

        u = u / 2.0  # average the signals, range [0, 1]

        # EMA smoothing
        self._running_uncertainty = (
            self._ema_alpha * u
            + (1 - self._ema_alpha) * self._running_uncertainty
        )
        return self._running_uncertainty

    @property
    def should_plan(self) -> bool:
        """Whether current uncertainty exceeds the threshold."""
        return self._running_uncertainty > self.threshold

    @property
    def current(self) -> float:
        return self._running_uncertainty
