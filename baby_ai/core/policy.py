"""
Policy head — action selection from core hidden state.

Maps the temporal core's hidden state to a distribution over discrete actions.
Supports both greedy and sampled action selection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class PolicyHead(nn.Module):
    """
    MLP-based policy head for discrete action selection.

    Args:
        input_dim: Core hidden state dimension.
        hidden_dim: MLP hidden size.
        action_dim: Number of discrete actions.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        action_dim: int = 64,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head for actor-critic
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, input_dim) from temporal core.

        Returns:
            logits: (B, action_dim) action logits.
            value: (B, 1) state value estimate.
        """
        logits = self.net(state)
        value = self.value_head(state)
        return logits, value

    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select an action.

        Returns:
            action: (B,) selected action indices.
            log_prob: (B,) log probability of selected action.
            value: (B, 1) state value.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)

        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob, value

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a given action (for PPO-style training).

        Returns:
            log_prob: (B,) log probability of the given action.
            entropy: (B,) distribution entropy.
            value: (B, 1) state value.
        """
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value
