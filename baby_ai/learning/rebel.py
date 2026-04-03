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
