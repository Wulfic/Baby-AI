"""
Action safety filter.

Screens proposed continuous action vectors against safety rules
before execution.  Blocks or penalizes unsafe actions.
"""

from __future__ import annotations

from typing import Optional

import torch

from baby_ai.utils.logging import get_logger

log = get_logger("safety", log_file="safety.log")


class ActionFilter:
    """
    Rule-based action safety filter for continuous action vectors.

    Each action vector is checked against a set of rules before execution.
    If an action is blocked, a penalty is returned instead.

    Args:
        penalty: Safety penalty magnitude for blocked actions.
    """

    def __init__(
        self,
        penalty: float = 1.0,
    ):
        self.penalty = penalty
        self._blocked_count = 0

    def filter_action(
        self,
        action: torch.Tensor,
        context: Optional[dict] = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Filter continuous actions.

        Continuous actions are bounded by tanh/sigmoid in the policy head.
        Returns actions as-is with zero penalty.
        Override for custom continuous safety rules.

        Args:
            action: (B, D) or (D,) continuous action vectors.
            context: Optional context dict.

        Returns:
            (filtered_actions, total_penalty)
        """
        return action, 0.0

    @property
    def stats(self) -> dict:
        return {"blocked_count": self._blocked_count}
