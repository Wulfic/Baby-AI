"""
Reward composition and normalization.

Combines intrinsic (curiosity + learning progress), extrinsic,
communication, and safety reward channels into a single scalar signal.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch

from baby_ai.utils.logging import RewardMonitor


class RewardComposer:
    """
    Composes multiple reward channels with configurable weights.

    Channels:
    - intrinsic: curiosity (ICM prediction error) + learning progress
    - communication: reward for useful communication
    - extrinsic: sparse task rewards
    - safety: penalty for unsafe / disallowed actions

    Weights are annealed over time (intrinsic decays from high to low).

    Args:
        intrinsic_weight_start: Initial weight for intrinsic reward.
        intrinsic_weight_end: Final weight after decay.
        intrinsic_decay_steps: Steps to anneal intrinsic weight.
        comm_weight: Weight for communication reward.
        extrinsic_weight: Weight for extrinsic reward.
        safety_weight: Weight for safety penalty (negative reward).
        normalize: Whether to normalize each channel via running stats.
    """

    def __init__(
        self,
        intrinsic_weight_start: float = 1.0,
        intrinsic_weight_end: float = 0.1,
        intrinsic_decay_steps: int = 500_000,
        comm_weight: float = 0.5,
        extrinsic_weight: float = 1.0,
        safety_weight: float = 2.0,
        normalize: bool = True,
        normalize_window: int = 1000,
    ):
        self.intrinsic_start = intrinsic_weight_start
        self.intrinsic_end = intrinsic_weight_end
        self.intrinsic_decay_steps = intrinsic_decay_steps
        self.comm_weight = comm_weight
        self.extrinsic_weight = extrinsic_weight
        self.safety_weight = safety_weight
        self.normalize = normalize

        # Running stats for normalization
        self._stats: dict[str, dict] = {}
        self._window = normalize_window
        self._step = 0

        # Monitoring
        self._monitor = RewardMonitor(window=normalize_window)

    @property
    def intrinsic_weight(self) -> float:
        """Current annealed intrinsic reward weight."""
        frac = min(1.0, self._step / max(1, self.intrinsic_decay_steps))
        return self.intrinsic_start + frac * (self.intrinsic_end - self.intrinsic_start)

    def _normalize_channel(self, channel: str, value: float) -> float:
        """Normalize a reward channel by running mean/std."""
        if not self.normalize:
            return value

        if channel not in self._stats:
            self._stats[channel] = {"values": deque(maxlen=self._window)}

        self._stats[channel]["values"].append(value)

        if len(self._stats[channel]["values"]) < 10:
            return value  # not enough data yet

        arr = np.array(self._stats[channel]["values"])
        mean, std = arr.mean(), arr.std() + 1e-8
        return (value - mean) / std

    def compose(
        self,
        intrinsic: float = 0.0,
        communication: float = 0.0,
        extrinsic: float = 0.0,
        safety_penalty: float = 0.0,
    ) -> float:
        """
        Compose a single reward from all channels.

        Args:
            intrinsic: Curiosity / learning progress signal.
            communication: Communication reward.
            extrinsic: External task reward.
            safety_penalty: Penalty for unsafe actions (should be >= 0, applied as negative).

        Returns:
            Composed scalar reward.
        """
        # Normalize each channel
        intrinsic_n = self._normalize_channel("intrinsic", intrinsic)
        comm_n = self._normalize_channel("communication", communication)
        extrinsic_n = self._normalize_channel("extrinsic", extrinsic)
        safety_n = safety_penalty  # don't normalize safety — raw penalty

        # Weighted sum
        reward = (
            self.intrinsic_weight * intrinsic_n
            + self.comm_weight * comm_n
            + self.extrinsic_weight * extrinsic_n
            - self.safety_weight * safety_n
        )

        # Monitor
        self._monitor.record("total", reward)
        self._monitor.record("intrinsic", intrinsic)
        self._monitor.record("extrinsic", extrinsic)

        self._step += 1
        return reward

    def stats(self) -> dict:
        return {
            "step": self._step,
            "intrinsic_weight": self.intrinsic_weight,
            "monitor": self._monitor.summary(),
        }
