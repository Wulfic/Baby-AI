"""
Grounded Successor Features (SF&GPI, Barreto et al. 2017/2018).

Successor features decompose the value function as::

    V(s) = ψ(s) · w

where ``ψ(s) ∈ ℝ^C`` is the expected discounted future **cumulant**
vector and ``w ∈ ℝ^C`` is a reward-weight vector.

**Grounding.** Unlike the original (learned-cumulant) formulation, here
the cumulants φ are *observed*: φ_t is the real per-channel reward vector
emitted by the environment each step (see
:mod:`baby_ai.learning.channels`).  So ψ(s)[i] literally means "expected
discounted future reward from channel i" (block_break, crafting,
idle_penalty, …) and ``w`` is exactly the live UI weight slider vector.

Key benefits for Minecraft:
  * **Zero-shot re-weighting** — change a slider and only ``w`` changes;
    ψ(s) is reused, so every state is instantly re-valued with no
    retraining (cf. ``RewardComposer`` buffer recomposition).
  * **Explainability** — ``ψ(s) ⊙ w`` is a per-channel attribution of the
    scalar value, answering "what is the agent being rewarded for".
  * **Transfer / GPI** — a bank of task weight vectors ``{w_k}`` lets the
    agent evaluate many objectives from one ψ and act greedily over the
    best (Generalised Policy Improvement).

Architecture::

    SuccessorHead:   core_state → ψ(s) ∈ ℝ^C
    RewardWeightBank: learnable per-task weight vectors w_k ∈ ℝ^C  (GPI)
    SuccessorLoss:    TD(0) on observed cumulants
                      ψ(s_t) ≈ φ_t + γ · stop_grad(ψ(s_{t+1}))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuccessorHead(nn.Module):
    """Maps the temporal-core state → successor-feature vector ψ(s) ∈ ℝ^C.

    The cumulants φ are observed (the real reward channels), so this head
    only predicts ψ — there is no learned feature extractor.

    Args:
        state_dim:    Dimension of the temporal-core output (hidden_dim).
        num_channels: Number of reward channels C (the cumulant dimension).
        hidden_dim:   MLP hidden width.
    """

    def __init__(
        self,
        state_dim: int = 512,
        num_channels: int = 29,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.num_channels = num_channels

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_channels),
        )

    def forward(self, core_state: torch.Tensor) -> torch.Tensor:
        """Args: ``core_state`` ``(B, state_dim)`` → ``ψ`` ``(B, C)``."""
        return self.net(core_state)


class RewardWeightBank(nn.Module):
    """Learnable per-task weight vectors for Generalised Policy Improvement.

    ``V(s, task_k) = ψ(s) · w_k``.  Holds a small library of task weight
    vectors (e.g. "explore", "build", "survive") so the agent can evaluate
    many objectives from a single ψ and act greedily over the best one
    (GPI).  Not on the critical path for the base single-``w`` setup; kept
    for the transfer extension.

    Args:
        num_tasks:    Number of distinct task/goal weight vectors.
        num_channels: Cumulant dimension C (must match :class:`SuccessorHead`).
    """

    def __init__(self, num_tasks: int = 16, num_channels: int = 29):
        super().__init__()
        self.weights = nn.Embedding(num_tasks, num_channels)

    def value(self, psi: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """``(B, C)`` ψ and ``(B,)`` task ids → ``(B,)`` values."""
        w = self.weights(task_ids)  # (B, C)
        return (psi * w).sum(dim=-1)  # (B,)

    def best_task_value(self, psi: torch.Tensor) -> torch.Tensor:
        """GPI: max value over all task weight vectors → ``(B,)``."""
        all_w = self.weights.weight  # (T, C)
        vals = psi @ all_w.T  # (B, T)
        return vals.max(dim=-1).values  # (B,)


class SuccessorLoss(nn.Module):
    """TD(0) loss for ψ against *observed* cumulants.

    ``L = SmoothL1(ψ(s_t),  φ_t + γ · stop_grad(ψ(s_{t+1})) · (1 - done))``

    where φ_t is the real per-channel reward vector for the transition.

    Args:
        gamma: Discount factor.
    """

    def __init__(self, gamma: float = 0.99):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        psi_t: torch.Tensor,
        phi_t: torch.Tensor,
        psi_next: torch.Tensor,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            psi_t:    ``(B, C)`` current successor features.
            phi_t:    ``(B, C)`` observed instantaneous cumulants (channels).
            psi_next: ``(B, C)`` next successor features (stop-grad applied).
            dones:    ``(B,)`` float, 1.0 at episode end (bootstrap masked).

        Returns:
            Scalar TD loss.
        """
        target = phi_t + self.gamma * psi_next.detach() * (1.0 - dones.unsqueeze(-1))
        return F.smooth_l1_loss(psi_t, target)
