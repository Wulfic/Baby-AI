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
import torch.nn.functional as F


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
        goal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Run batched latent planning from the current state.

        Args:
            core_state: (1, state_dim) current temporal core output.
            goal:       (1, goal_dim) optional System 3 goal embedding.
                        When provided, a cosine proximity bonus biases
                        trajectory scoring toward the active subgoal.

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

        # Pre-expand goal for cosine bonus if provided
        goal_expanded = None
        if goal is not None:
            goal_dim = goal.size(-1)
            goal_expanded = goal.expand(N, -1).contiguous()  # (N, goal_dim)

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

            # Goal proximity bonus (System 3) — bias toward trajectories
            # that move latent state closer to the active subgoal.
            if goal_expanded is not None:
                # Compare the first goal_dim dims of next_latent to goal
                latent_prefix = next_latent[:, :goal_dim]  # (N, goal_dim)
                cos_sim = F.cosine_similarity(latent_prefix, goal_expanded, dim=-1)  # (N,)
                step_value = step_value + 0.5 * cos_sim

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

    Includes cooldown and warmup logic to prevent constant triggering
    on an untrained model where all signals saturate at maximum.

    Uncertainty signals:
      1. Diffusion denoising variance across action samples.
      2. World model prediction error spike.
      3. Value prediction spread across imagined trajectories.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ema_alpha: float = 0.1,
        warmup_steps: int = 100,
        cooldown_steps: int = 20,
    ):
        self.threshold = threshold
        self._ema_alpha = ema_alpha
        self._warmup_steps = warmup_steps
        self._cooldown_steps = cooldown_steps
        self._running_uncertainty: float = 0.0
        self._step: int = 0
        self._last_trigger_step: int = -999  # last step System 2 fired

        # Running statistics for adaptive normalisation.
        # Without these, the raw signals saturate and
        # should_plan is always True on an untrained model.
        self._lp_ema: float = 0.0   # EMA of raw log_prob signal
        self._lp_sq_ema: float = 1.0  # EMA of squared log_prob signal

    def update(
        self,
        value: torch.Tensor | None = None,
        log_prob: torch.Tensor | None = None,
    ) -> float:
        """
        Update uncertainty estimate from the latest inference result.

        Uses z-score normalisation against a running mean/variance
        so signals don't saturate on an early-stage untrained model.

        Args:
            value:    (B, 1) state value — low absolute value = uncertain.
            log_prob: (B,) log probability — low = uncertain.

        Returns:
            Current uncertainty estimate (0 = certain, higher = uncertain).
        """
        self._step += 1
        u = 0.0
        n_signals = 0

        if log_prob is not None:
            raw_lp = -log_prob.mean().item()  # positive = more uncertain
            # Adaptive z-score normalisation
            self._lp_ema = 0.99 * self._lp_ema + 0.01 * raw_lp
            self._lp_sq_ema = 0.99 * self._lp_sq_ema + 0.01 * (raw_lp ** 2)
            lp_std = max((self._lp_sq_ema - self._lp_ema ** 2) ** 0.5, 1e-4)
            # z-score: how many stdev above mean?
            z = (raw_lp - self._lp_ema) / lp_std
            # Map z to [0, 1]: z=0 → 0.5, z=+2 → ~0.88
            u += max(0.0, min(1.0, 0.5 + 0.25 * z))
            n_signals += 1

        if value is not None:
            # Low |value| means the model can't distinguish good/bad states.
            # Sigmoid-scale so large absolute values → low uncertainty.
            val_mag = value.abs().mean().item()
            u += 1.0 / (1.0 + val_mag)  # asymptotes to 0 as |v| grows
            n_signals += 1

        if n_signals > 0:
            u = u / n_signals

        # EMA smoothing
        self._running_uncertainty = (
            self._ema_alpha * u
            + (1 - self._ema_alpha) * self._running_uncertainty
        )
        return self._running_uncertainty

    @property
    def should_plan(self) -> bool:
        """Whether current uncertainty exceeds the threshold.

        Respects warmup (model needs N steps before planning makes
        sense) and cooldown (don't plan two steps in a row).
        """
        if self._step < self._warmup_steps:
            return False
        if (self._step - self._last_trigger_step) < self._cooldown_steps:
            return False
        return self._running_uncertainty > self.threshold

    def mark_triggered(self) -> None:
        """Call this after System 2 actually fires to start cooldown."""
        self._last_trigger_step = self._step

    @property
    def current(self) -> float:
        return self._running_uncertainty
