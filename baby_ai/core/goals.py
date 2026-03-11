"""
System 3 — Hierarchical long-horizon planning via latent goals.

System 3 sits above the reactive policy (System 1) and deliberative
planner (System 2).  It generates a *subgoal sequence* that conditions
every downstream action.  System 1 still picks the low-level motor
command each tick, but now pursues a higher-level objective.

Components:
  GoalProposer     — proposes K candidate goal embeddings from core state.
  SubgoalPlanner   — decomposes a top-level goal into ordered subgoals.
  GoalConditioner  — FiLM-modulates core_state with the active subgoal.
  GoalMonitor      — detects achievement / stuck → advance / replan.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Goal Proposer
# ────────────────────────────────────────────────────────────────────────────

class GoalProposer(nn.Module):
    """
    Proposes candidate high-level goals from the agent's current state.

    Uses the temporal core's hidden state (which encodes long history
    via Mamba's infinite context) to predict what goals are achievable
    and valuable from here.

    Architecture:
        core_state → shared MLP trunk → K parallel heads → K goal embeddings
        + a small value head that scores each candidate.

    Training:  hindsight goal relabelling from replay
        (target = projected fused embedding K steps into the future).

    Args:
        state_dim:      Dimension of the temporal core output.
        goal_dim:       Dimension of each goal embedding.
        num_candidates: Number of goal candidates to propose (K).
        hidden_dim:     MLP hidden layer size.
    """

    def __init__(
        self,
        state_dim: int = 512,
        goal_dim: int = 64,
        num_candidates: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.goal_dim = goal_dim
        self.num_candidates = num_candidates

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # K goal heads  (output K * goal_dim, then reshape)
        self.goal_heads = nn.Linear(hidden_dim, num_candidates * goal_dim)

        # Value head: scores each candidate
        self.score_head = nn.Linear(hidden_dim, num_candidates)

        # Projection from fused_dim → goal_dim (used during hindsight training)
        self.goal_proj = nn.Linear(state_dim, goal_dim)

    def forward(
        self,
        core_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Propose candidate goals and their estimated value.

        Args:
            core_state: (B, state_dim) temporal core output.

        Returns:
            goals:  (B, K, goal_dim)
            scores: (B, K)
        """
        h = self.trunk(core_state)                            # (B, hidden_dim)
        goals = self.goal_heads(h)                            # (B, K * goal_dim)
        goals = goals.view(-1, self.num_candidates, self.goal_dim)
        scores = self.score_head(h)                           # (B, K)
        return {"goals": goals, "scores": scores}

    def project_fused(self, fused: torch.Tensor) -> torch.Tensor:
        """Project a fused embedding to goal_dim for hindsight targets."""
        return self.goal_proj(fused)


# ────────────────────────────────────────────────────────────────────────────
# Subgoal Planner  (small Transformer decoder)
# ────────────────────────────────────────────────────────────────────────────

class SubgoalPlanner(nn.Module):
    """
    Autoregressive subgoal sequence generator.

    Given a top-level goal and current state, generates an ordered
    sequence of subgoal embeddings.  Each "token" is a continuous
    goal_dim-dimensional vector (not a discrete token).

    Architecture:  2-layer causal Transformer with continuous output heads.
    A learned [DONE] score indicates when the plan is complete.

    Training:  teacher-forced on hindsight subgoal sequences extracted
               from successful replay trajectories (waypoints).

    Args:
        state_dim:    Dimension of the temporal core output.
        goal_dim:     Dimension of each subgoal embedding.
        max_subgoals: Maximum sequence length.
        num_layers:   Transformer decoder layers.
        num_heads:    Attention heads.
    """

    def __init__(
        self,
        state_dim: int = 512,
        goal_dim: int = 64,
        max_subgoals: int = 12,
        num_layers: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.goal_dim = goal_dim
        self.max_subgoals = max_subgoals

        # Condition on state + top-level goal
        self.context_proj = nn.Linear(state_dim + goal_dim, goal_dim)

        # Learnable position embeddings for each slot in the plan
        self.pos_embed = nn.Parameter(torch.randn(1, max_subgoals, goal_dim) * 0.02)

        # Causal Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=goal_dim,
            nhead=num_heads,
            dim_feedforward=goal_dim * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head: continuous subgoal + done probability
        self.subgoal_head = nn.Linear(goal_dim, goal_dim)
        self.done_head = nn.Linear(goal_dim, 1)

        # Causal mask (registered as buffer so it moves to the right device)
        mask = nn.Transformer.generate_square_subsequent_mask(max_subgoals)
        self.register_buffer("_causal_mask", mask)

    def forward(
        self,
        core_state: torch.Tensor,
        goal: torch.Tensor,
        max_steps: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Generate a subgoal plan.

        Args:
            core_state: (B, state_dim) from temporal core.
            goal:       (B, goal_dim) top-level goal embedding.
            max_steps:  Override max_subgoals for early stopping.

        Returns:
            subgoals:   (B, T, goal_dim) ordered subgoal embeddings.
            done_probs: (B, T) probability each slot is the terminal subgoal.
        """
        B = core_state.size(0)
        T = max_steps or self.max_subgoals
        device = core_state.device

        # Context: project concatenated (state, goal) to goal_dim
        ctx = self.context_proj(
            torch.cat([core_state, goal], dim=-1)
        ).unsqueeze(1)  # (B, 1, goal_dim)

        # Build decoder input: positional embeddings for T slots
        tgt = self.pos_embed[:, :T, :].expand(B, -1, -1)  # (B, T, goal_dim)

        # Causal mask for autoregressive generation
        mask = self._causal_mask[:T, :T].to(device)

        # Decode:  memory = context, tgt = positional queries
        decoded = self.decoder(
            tgt, memory=ctx, tgt_mask=mask,
        )  # (B, T, goal_dim)

        subgoals = self.subgoal_head(decoded)                 # (B, T, goal_dim)
        done_logits = self.done_head(decoded).squeeze(-1)     # (B, T)
        done_probs = torch.sigmoid(done_logits)

        return {"subgoals": subgoals, "done_probs": done_probs}


# ────────────────────────────────────────────────────────────────────────────
# Goal Conditioner  (FiLM modulation)
# ────────────────────────────────────────────────────────────────────────────

class GoalConditioner(nn.Module):
    """
    FiLM conditioning layer: injects goal information into core_state.

    Generates per-channel scale and shift from the goal embedding:
        conditioned = scale * core_state + shift

    When no goal is active (goal=None), the output is an exact
    identity: scale=1, shift=0.  This guarantees zero regression
    when System 3 is disabled.

    Args:
        state_dim: Dimension of the temporal core output.
        goal_dim:  Dimension of the goal embedding.
    """

    def __init__(self, state_dim: int = 512, goal_dim: int = 64):
        super().__init__()
        # Two linear maps: goal → (scale, shift)
        self.film = nn.Linear(goal_dim, state_dim * 2)

        # Initialise to identity: scale=1, shift=0
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # bias first half = 1 (scale), second half = 0 (shift)
        with torch.no_grad():
            self.film.bias[:state_dim] = 1.0

    def forward(
        self,
        core_state: torch.Tensor,
        goal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            core_state: (B, state_dim).
            goal:       (B, goal_dim) or None.

        Returns:
            (B, state_dim) — conditioned core state.
        """
        if goal is None:
            return core_state

        film_params = self.film(goal)                  # (B, state_dim * 2)
        scale, shift = film_params.chunk(2, dim=-1)    # each (B, state_dim)
        return scale * core_state + shift


# ────────────────────────────────────────────────────────────────────────────
# Goal Monitor  (pure Python, no learnable params)
# ────────────────────────────────────────────────────────────────────────────

class GoalMonitor:
    """
    Monitors subgoal execution and triggers replanning.

    Tracks cosine similarity between the agent's current latent state
    and the target subgoal embedding.  When similarity exceeds a
    threshold the subgoal is considered achieved; when too many steps
    pass without progress the agent is declared stuck.

    Transitions:
        similarity > achieve_threshold           → 'advance'
        steps_on_subgoal > patience_steps        → 'replan'
        otherwise                                → 'continue'

    Args:
        achieve_threshold:    Cosine similarity to mark a subgoal achieved.
        patience_steps:       Max steps before declaring stuck.
        min_replan_interval:  Minimum steps between full replans.
    """

    def __init__(
        self,
        achieve_threshold: float = 0.85,
        patience_steps: int = 200,
        min_replan_interval: int = 50,
    ):
        self.achieve_threshold = achieve_threshold
        self.patience_steps = patience_steps
        self.min_replan_interval = min_replan_interval

        self._steps_on_subgoal: int = 0
        self._total_steps: int = 0
        self._last_replan_step: int = -999
        self._best_similarity: float = -1.0

    def step(
        self,
        core_state: torch.Tensor,
        current_subgoal: torch.Tensor,
        goal_proj: nn.Module | None = None,
    ) -> str:
        """
        Evaluate one step of subgoal execution.

        Args:
            core_state:     (1, state_dim) or (state_dim,) current latent.
            current_subgoal: (goal_dim,) target subgoal embedding.
            goal_proj:       Optional projection core_state → goal_dim space.

        Returns:
            'continue' — keep pursuing current subgoal.
            'advance'  — current subgoal achieved, advance to next.
            'replan'   — stuck, trigger System 3 replan.
        """
        self._total_steps += 1
        self._steps_on_subgoal += 1

        # Project core_state to goal space if dimensions differ
        state = core_state.detach()
        if state.dim() > 1:
            state = state.squeeze(0)
        goal = current_subgoal.detach()
        if goal.dim() > 1:
            goal = goal.squeeze(0)
        if state.shape[-1] != goal.shape[-1]:
            if goal_proj is not None:
                state = goal_proj(state)
            else:
                # Default: slice the first goal_dim dims from state
                state = state[:goal.shape[-1]]

        # Cosine similarity
        sim = F.cosine_similarity(
            state.unsqueeze(0), goal.unsqueeze(0),
        ).item()

        self._best_similarity = max(self._best_similarity, sim)

        # Check achievement
        if sim >= self.achieve_threshold:
            self._steps_on_subgoal = 0
            self._best_similarity = -1.0
            return "advance"

        # Check stuck
        if self._steps_on_subgoal >= self.patience_steps:
            if (self._total_steps - self._last_replan_step) >= self.min_replan_interval:
                self._steps_on_subgoal = 0
                self._best_similarity = -1.0
                self._last_replan_step = self._total_steps
                return "replan"

        return "continue"

    def reset(self) -> None:
        """Reset state for a new subgoal."""
        self._steps_on_subgoal = 0
        self._best_similarity = -1.0

    def reset_full(self) -> None:
        """Full reset (e.g. new episode / death)."""
        self._steps_on_subgoal = 0
        self._total_steps = 0
        self._last_replan_step = -999
        self._best_similarity = -1.0
        self._last_replan_step = -999
