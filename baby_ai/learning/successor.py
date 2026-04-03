"""
Successor Representations / Successor Features.

Successor features (SFs) decompose the value function as:
    V(s) = φ(s)ᵀ w
where φ(s) is the expected discounted future occupancy of features
and w is a reward weight vector.

Key benefit for Minecraft: when the reward changes (e.g. switch goal
from "collect wood" to "build shelter"), *only* w changes — φ(s) is
reused, giving instant zero-shot task transfer.

Architecture:
    SuccessorHead:
        core_state → SF vector ψ(s) ∈ ℝ^{sf_dim}  (LSTM encodes history)
    RewardWeightBank:
        Learnable per-task weight vectors w_k ∈ ℝ^{sf_dim}
        V(s, task_k) = ψ(s) · w_k
    SuccessorLoss:
        One-step TD target:  ψ(s_t) ≈ φ(s_t) + γ * ψ(s_{t+1})
        Loss: MSE(ψ(s_t), φ(s_t) + γ * ψ(s_{t+1}).detach())
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SuccessorHead(nn.Module):
    """
    Maps temporal core output → successor feature vector ψ(s).

    Args:
        state_dim:   Dimension of temporal core output (hidden_dim).
        sf_dim:      Dimension of successor feature vector.
        hidden_dim:  MLP hidden dim.
    """

    def __init__(
        self,
        state_dim: int = 512,
        sf_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.sf_dim = sf_dim

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, sf_dim),
        )

        # Feature extractor: maps core_state → instantaneous features φ(s)
        # These are the "primitives" that successor features accumulate.
        self.phi_net = nn.Sequential(
            nn.Linear(state_dim, sf_dim),
            nn.LayerNorm(sf_dim),
            nn.Tanh(),  # bounded features stabilise TD learning
        )

    def forward(
        self,
        core_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            core_state: (B, state_dim)

        Returns:
            psi: (B, sf_dim) successor features
            phi: (B, sf_dim) instantaneous features
        """
        psi = self.net(core_state)
        phi = self.phi_net(core_state)
        return {"psi": psi, "phi": phi}


class RewardWeightBank(nn.Module):
    """
    Learnable per-task reward weight vectors.

    V(s, task_k) = ψ(s) · w_k

    Args:
        num_tasks:  Number of distinct task/goal types.
        sf_dim:     Dimension of successor features.
    """

    def __init__(self, num_tasks: int = 16, sf_dim: int = 128):
        super().__init__()
        self.weights = nn.Embedding(num_tasks, sf_dim)

    def value(
        self,
        psi: torch.Tensor,
        task_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute value estimates for given tasks.

        Args:
            psi:       (B, sf_dim)
            task_ids:  (B,) long tensor of task indices

        Returns:
            (B,) value estimates
        """
        w = self.weights(task_ids)  # (B, sf_dim)
        return (psi * w).sum(dim=-1)  # (B,)

    def best_task_value(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Return the maximum value over all tasks (GPI policy).

        Args:
            psi: (B, sf_dim)

        Returns:
            (B,) max value across all tasks
        """
        # all_w: (num_tasks, sf_dim)
        all_w = self.weights.weight  # (T, sf_dim)
        vals = psi @ all_w.T  # (B, T)
        return vals.max(dim=-1).values  # (B,)


class SuccessorLoss(nn.Module):
    """
    TD(1) loss for training the successor head.

    Loss = MSE(ψ(s_t), φ(s_t) + γ · stop_grad(ψ(s_{t+1})))

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
            psi_t:    (B, sf_dim) current successor features
            phi_t:    (B, sf_dim) current instantaneous features
            psi_next: (B, sf_dim) next successor features (stop-grad applied)
            dones:    (B,) float tensor, 1.0 at episode end

        Returns:
            Scalar loss.
        """
        target = phi_t + self.gamma * psi_next.detach() * (1.0 - dones.unsqueeze(-1))
        return F.mse_loss(psi_t, target)
