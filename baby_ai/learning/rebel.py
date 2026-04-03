"""
REBEL: Reinforcement Learning via Regressing Relative Rewards.

Reduces policy optimization to regression on relative reward differences
between pairs of trajectory completions.  This completely removes the need
for clipped surrogate objectives, GAE, or accurate log-probability
estimates from the flow-matching policy.

Core algorithm:
    1. Sample two transitions: (s, a_w, r_w) and (s, a_l, r_l)
    2. Designate the higher-reward action as "winner"
    3. L = -log σ(β · (score(a_w|s) - score(a_l|s)))
    4. score(a|s) = policy.evaluate(state, action) → log-prob proxy

This is equivalent to a Bradley-Terry preference model and reduces to
Natural Policy Gradient in the tabular limit.

Reference: arXiv 2404.16767

Also provides GRPOLoss (Group Relative Policy Optimization, DeepSeek-R1),
which normalizes rewards within the batch group instead of pairwise pairing,
eliminating the state-mismatch noise from REBEL's random batch halving.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class REBELLoss(nn.Module):
    """
    Computes the REBEL policy optimization loss.

    Given pairs of (state, action, reward):
        (s, a_w, r_w) — the "winner" (higher reward)
        (s, a_l, r_l) — the "loser"  (lower reward)

    Loss = -log σ(β · Δr · (score(a_w|s) - score(a_l|s)))

    Where:
        - β controls the KL regularisation strength toward the reference policy
        - Δr = r_w - r_l is the clipped relative reward
        - score = policy.evaluate() log-prob proxy

    Args:
        beta:        KL regularisation weight (higher = more conservative updates).
        reward_clip: Maximum absolute value for the relative reward.
    """

    def __init__(self, beta: float = 0.1, reward_clip: float = 5.0, entropy_weight: float = 0.0):
        super().__init__()
        self.beta = beta
        self.reward_clip = reward_clip
        self.entropy_weight = entropy_weight

    def forward(
        self,
        state: torch.Tensor,
        action_w: torch.Tensor,
        action_l: torch.Tensor,
        reward_w: torch.Tensor,
        reward_l: torch.Tensor,
        policy: nn.Module,
    ) -> torch.Tensor:
        """
        Compute REBEL loss for a batch of winner/loser pairs.

        Args:
            state:    (B, D)          core hidden state.
            action_w: (B, action_dim) winner action (higher reward).
            action_l: (B, action_dim) loser action  (lower reward).
            reward_w: (B,)            winner reward.
            reward_l: (B,)            loser reward.
            policy:   nn.Module with .evaluate(state, action) → (log_prob, entropy, value).

        Returns:
            Scalar REBEL loss (lower = better policy alignment with rewards).
        """
        # Get policy scores for both actions
        log_prob_w, entropy_w, _ = policy.evaluate(state, action_w)  # (B,)
        log_prob_l, entropy_l, _ = policy.evaluate(state, action_l)  # (B,)

        # Relative reward — clipped for gradient stability
        delta_r = torch.clamp(
            reward_w - reward_l,
            -self.reward_clip,
            self.reward_clip,
        )

        # Score difference weighted by β
        score_diff = self.beta * (log_prob_w - log_prob_l)

        # Bradley-Terry sigmoid cross-entropy:
        # When delta_r > 0: push score_diff positive (prefer winner)
        # When delta_r < 0: push score_diff negative (prefer loser — swapped)
        # When delta_r ≈ 0: near-zero gradient (actions equally good)
        loss = -F.logsigmoid(delta_r * score_diff).mean()

        # Entropy bonus: maximise policy entropy to prevent collapse during
        # long reward deserts (e.g. mining, exploration with no dense reward).
        if self.entropy_weight > 0.0:
            mean_entropy = (entropy_w + entropy_l).mean() / 2.0
            loss = loss - self.entropy_weight * mean_entropy

        return loss


class GRPOLoss(nn.Module):
    """
    Group Relative Policy Optimization (GRPO).

    DeepSeek-R1 (2025) showed that normalizing rewards *within a group*
    of completions from the same prompt gives a cleaner advantage signal
    than pairwise winner/loser comparisons.  For Minecraft:

    - The entire batch is treated as one group (diverse rewards from
      different states; group normalization filters the state-value noise).
    - Advantage = (r - mean(r)) / (std(r) + ε) — zero-mean, unit-variance.
    - Policy gradient: maximize E[log_prob * advantage] — no clipped ratio,
      no reference model, no GAE needed.
    - KL penalty via log_prob² regularization keeps updates conservative.

    Improvements over REBEL (same-batch random pairing):
    - No state-mismatch noise from pairing different states.
    - Group normalization is robust to reward scale drift across episodes.
    - Full batch participates in the gradient (REBEL uses only half).

    Args:
        beta:           KL regularization weight (penalizes large log_prob).
        reward_clip:    Clip normalized advantages to this absolute value.
        entropy_weight: Entropy bonus weight (prevent collapse in reward deserts).
    """

    def __init__(
        self,
        beta: float = 0.1,
        reward_clip: float = 5.0,
        entropy_weight: float = 0.01,
    ):
        super().__init__()
        self.beta = beta
        self.reward_clip = reward_clip
        self.entropy_weight = entropy_weight

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        policy: nn.Module,
    ) -> torch.Tensor:
        """
        Compute GRPO loss for a batch of (state, action, reward) tuples.

        Args:
            state:   (B, D)          core hidden states.
            actions: (B, action_dim) collected actions.
            rewards: (B,)            scalar rewards.
            policy:  nn.Module with .evaluate(state, action) → (log_prob, entropy, value).

        Returns:
            Scalar GRPO loss.
        """
        B = state.size(0)
        if B < 2:
            return torch.tensor(0.0, device=state.device)

        # Group-normalize rewards → zero-mean, unit-variance advantages
        reward_mean = rewards.mean()
        reward_std = rewards.std().clamp(min=1e-8)
        advantages = (rewards - reward_mean) / reward_std
        advantages = torch.clamp(advantages, -self.reward_clip, self.reward_clip)

        # Policy evaluation for all actions in the batch
        log_probs, entropies, _ = policy.evaluate(state, actions)  # (B,), (B,)

        # Policy gradient objective: maximize E[log_prob * advantage]
        pg_loss = -(log_probs * advantages).mean()

        # KL regularization: penalize deviation from a broad reference by
        # discouraging very large |log_prob| values (implicit KL to uniform).
        kl_reg = self.beta * (log_probs.pow(2)).mean()

        loss = pg_loss + kl_reg

        # Entropy bonus to prevent premature mode collapse
        if self.entropy_weight > 0.0:
            loss = loss - self.entropy_weight * entropies.mean()

        return loss

    """
    Computes the REBEL policy optimization loss.

    Given pairs of (state, action, reward):
        (s, a_w, r_w) — the "winner" (higher reward)
        (s, a_l, r_l) — the "loser"  (lower reward)

    Loss = -log σ(β · Δr · (score(a_w|s) - score(a_l|s)))

    Where:
        - β controls the KL regularisation strength toward the reference policy
        - Δr = r_w - r_l is the clipped relative reward
        - score = policy.evaluate() log-prob proxy

    Args:
        beta:        KL regularisation weight (higher = more conservative updates).
        reward_clip: Maximum absolute value for the relative reward.
    """

    def __init__(self, beta: float = 0.1, reward_clip: float = 5.0, entropy_weight: float = 0.0):
        super().__init__()
        self.beta = beta
        self.reward_clip = reward_clip
        self.entropy_weight = entropy_weight

    def forward(
        self,
        state: torch.Tensor,
        action_w: torch.Tensor,
        action_l: torch.Tensor,
        reward_w: torch.Tensor,
        reward_l: torch.Tensor,
        policy: nn.Module,
    ) -> torch.Tensor:
        """
        Compute REBEL loss for a batch of winner/loser pairs.

        Args:
            state:    (B, D)          core hidden state.
            action_w: (B, action_dim) winner action (higher reward).
            action_l: (B, action_dim) loser action  (lower reward).
            reward_w: (B,)            winner reward.
            reward_l: (B,)            loser reward.
            policy:   nn.Module with .evaluate(state, action) → (log_prob, entropy, value).

        Returns:
            Scalar REBEL loss (lower = better policy alignment with rewards).
        """
        # Get policy scores for both actions
        log_prob_w, entropy_w, _ = policy.evaluate(state, action_w)  # (B,)
        log_prob_l, entropy_l, _ = policy.evaluate(state, action_l)  # (B,)

        # Relative reward — clipped for gradient stability
        delta_r = torch.clamp(
            reward_w - reward_l,
            -self.reward_clip,
            self.reward_clip,
        )

        # Score difference weighted by β
        score_diff = self.beta * (log_prob_w - log_prob_l)

        # Bradley-Terry sigmoid cross-entropy:
        # When delta_r > 0: push score_diff positive (prefer winner)
        # When delta_r < 0: push score_diff negative (prefer loser — swapped)
        # When delta_r ≈ 0: near-zero gradient (actions equally good)
        loss = -F.logsigmoid(delta_r * score_diff).mean()

        # Entropy bonus: maximise policy entropy to prevent collapse during
        # long reward deserts (e.g. mining, exploration with no dense reward).
        if self.entropy_weight > 0.0:
            mean_entropy = (entropy_w + entropy_l).mean() / 2.0
            loss = loss - self.entropy_weight * mean_entropy

        return loss
