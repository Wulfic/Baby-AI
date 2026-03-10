"""
Action safety filter.

Screens proposed actions against a set of rules before execution.
Blocks or penalizes unsafe, disallowed, or risky actions.
"""

from __future__ import annotations

from typing import Optional

import torch

from baby_ai.utils.logging import get_logger

log = get_logger("safety", log_file="safety.log")


class ActionFilter:
    """
    Rule-based action safety filter.

    Each action is checked against a set of rules before execution.
    If an action is blocked, a penalty is returned instead.

    Rules are simple callables: (action_idx, context) → (allowed: bool, reason: str).

    Args:
        human_in_loop: If True, high-risk actions require human confirmation.
        penalty: Safety penalty magnitude for blocked actions.
    """

    def __init__(
        self,
        human_in_loop: bool = True,
        penalty: float = 1.0,
    ):
        self.human_in_loop = human_in_loop
        self.penalty = penalty
        self._rules: list = []
        self._blocked_count = 0

        # Default: block a reserved range of action indices (e.g., "dangerous" actions)
        self.add_rule(self._default_block_rule)

    def add_rule(self, rule_fn) -> None:
        """Add a safety rule. Rule signature: (action_idx: int, context: dict) → (bool, str)."""
        self._rules.append(rule_fn)

    def check(self, action: int, context: Optional[dict] = None) -> tuple[bool, float, str]:
        """
        Check if an action is safe.

        Returns:
            (allowed, safety_penalty, reason)
        """
        context = context or {}

        for rule in self._rules:
            allowed, reason = rule(action, context)
            if not allowed:
                self._blocked_count += 1
                log.warning("Action %d BLOCKED: %s", action, reason)
                return False, self.penalty, reason

        return True, 0.0, "OK"

    def filter_action(
        self,
        action: torch.Tensor,
        context: Optional[dict] = None,
    ) -> tuple[torch.Tensor, float]:
        """
        Filter a batch of continuous actions.

        Continuous actions are bounded by tanh/sigmoid in the policy head,
        so discrete rule checking is not applicable. Returns actions as-is
        with zero penalty. Override for custom continuous safety rules.

        Args:
            action: (B, D) continuous action vectors.
            context: Optional context dict.

        Returns:
            (filtered_actions, total_penalty)
        """
        return action, 0.0

    @staticmethod
    def _default_block_rule(action_idx: int, context: dict) -> tuple[bool, str]:
        """Default: no actions blocked (override with custom rules)."""
        return True, ""

    @property
    def stats(self) -> dict:
        return {"blocked_count": self._blocked_count, "num_rules": len(self._rules)}
