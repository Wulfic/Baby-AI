"""
Reward composition and normalization.

Combines intrinsic (curiosity + learning progress), extrinsic,
communication, safety, and **creation-focused** reward channels
into a single scalar signal.

The creation channels (crafting, block placement, building streaks,
creative sequences) carry the highest weights to steer the agent
towards building, crafting, and creating in Minecraft.
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
    - extrinsic: sparse task rewards (survival, etc.)
    - exploration: reward for reaching visually new areas
    - interaction: reward for breaking/placing blocks, hitting entities
    - action_diversity: reward for trying different actions
    - movement: reward for locomotion confirmed by visual change
    - block_break: reward for breaking blocks (crosshair crack detection)
    - item_pickup: reward for collecting items (hotbar change detection)
    - death_penalty: penalty for dying (death screen detection)
    - safety: penalty for unsafe / disallowed actions

    * Creation-focused channels (highest weights):
    - block_place: reward for placing blocks (building)
    - crafting: reward for crafting items (inventory UI + hotbar change)
    - building_streak: bonus for consecutive block placements
    - creative_sequence: bonus for gather->craft->build pipeline

    Weights are annealed over time (intrinsic decays from high to low,
    creation channels stay high to keep the agent goal-directed).

    Args:
        intrinsic_weight_start: Initial weight for intrinsic reward.
        intrinsic_weight_end: Final weight after decay.
        intrinsic_decay_steps: Steps to anneal intrinsic weight.
        comm_weight: Weight for communication reward.
        extrinsic_weight: Weight for extrinsic reward.
        exploration_weight: Weight for long-horizon exploration bonus.
        interaction_weight: Weight for block-interaction reward.
        action_diversity_weight: Weight for action diversity bonus.
        movement_weight: Weight for movement reward.
        block_break_weight: Weight for block-break reward.
        item_pickup_weight: Weight for item-pickup reward.
        block_place_weight: Weight for block-placement reward (building).
        crafting_weight: Weight for crafting events.
        building_streak_weight: Weight for consecutive placements.
        creative_sequence_weight: Weight for gather->craft->build cycles.
        death_penalty_weight: Weight for death penalty (applied as negative).
        safety_weight: Weight for safety penalty (negative reward).
        normalize: Whether to normalize each channel via running stats.
    """

    def __init__(
        self,
        intrinsic_weight_start: float = 1.0,
        intrinsic_weight_end: float = 0.1,
        intrinsic_decay_steps: int = 500_000,
        comm_weight: float = 0.3,
        extrinsic_weight: float = 1.0,
        exploration_weight: float = 1.0,
        interaction_weight: float = 1.5,
        action_diversity_weight: float = 0.5,
        movement_weight: float = 0.3,
        block_break_weight: float = 3.0,
        item_pickup_weight: float = 4.0,
        # * Creation-focused weights (intentionally the highest)
        block_place_weight: float = 5.0,
        crafting_weight: float = 8.0,
        building_streak_weight: float = 4.0,
        creative_sequence_weight: float = 10.0,
        # Penalties
        death_penalty_weight: float = 5.0,
        safety_weight: float = 2.0,
        normalize: bool = True,
        normalize_window: int = 1000,
    ):
        self.intrinsic_start = intrinsic_weight_start
        self.intrinsic_end = intrinsic_weight_end
        self.intrinsic_decay_steps = intrinsic_decay_steps
        self.comm_weight = comm_weight
        self.extrinsic_weight = extrinsic_weight
        self.exploration_weight = exploration_weight
        self.interaction_weight = interaction_weight
        self.action_diversity_weight = action_diversity_weight
        self.movement_weight = movement_weight
        self.block_break_weight = block_break_weight
        self.item_pickup_weight = item_pickup_weight
        self.block_place_weight = block_place_weight
        self.crafting_weight = crafting_weight
        self.building_streak_weight = building_streak_weight
        self.creative_sequence_weight = creative_sequence_weight
        self.death_penalty_weight = death_penalty_weight
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
        exploration: float = 0.0,
        interaction: float = 0.0,
        action_diversity: float = 0.0,
        movement: float = 0.0,
        block_break: float = 0.0,
        item_pickup: float = 0.0,
        block_place: float = 0.0,
        crafting: float = 0.0,
        building_streak: float = 0.0,
        creative_sequence: float = 0.0,
        death_penalty: float = 0.0,
        safety_penalty: float = 0.0,
    ) -> float:
        """
        Compose a single reward from all channels.

        Args:
            intrinsic: Curiosity / learning progress signal.
            communication: Communication reward.
            extrinsic: External task reward (survival, etc.).
            exploration: Long-horizon area-change bonus.
            interaction: Block break / place / entity hit bonus.
            action_diversity: Bonus for trying many different actions.
            movement: Locomotion bonus (confirmed by visual change).
            block_break: Reward for breaking blocks (crosshair crack).
            item_pickup: Reward for collecting items (hotbar change).
            block_place: Reward for placing blocks (building).
            crafting: Reward for crafting items.
            building_streak: Bonus for consecutive placements.
            creative_sequence: Bonus for gather->craft->build pipeline.
            death_penalty: Penalty for dying (>= 0, applied as negative).
            safety_penalty: Penalty for unsafe actions (>= 0, applied as negative).

        Returns:
            Composed scalar reward.
        """
        # Normalize each channel
        intrinsic_n = self._normalize_channel("intrinsic", intrinsic)
        comm_n = self._normalize_channel("communication", communication)
        extrinsic_n = self._normalize_channel("extrinsic", extrinsic)
        exploration_n = self._normalize_channel("exploration", exploration)
        interaction_n = self._normalize_channel("interaction", interaction)
        diversity_n = self._normalize_channel("action_diversity", action_diversity)
        movement_n = self._normalize_channel("movement", movement)
        block_break_n = self._normalize_channel("block_break", block_break)
        item_pickup_n = self._normalize_channel("item_pickup", item_pickup)
        block_place_n = self._normalize_channel("block_place", block_place)
        crafting_n = self._normalize_channel("crafting", crafting)
        building_streak_n = self._normalize_channel("building_streak", building_streak)
        creative_seq_n = self._normalize_channel("creative_sequence", creative_sequence)
        death_n = death_penalty  # don't normalize — raw penalty
        safety_n = safety_penalty  # don't normalize safety — raw penalty

        # Weighted sum — creation channels carry the highest weights
        reward = (
            self.intrinsic_weight * intrinsic_n
            + self.comm_weight * comm_n
            + self.extrinsic_weight * extrinsic_n
            + self.exploration_weight * exploration_n
            + self.interaction_weight * interaction_n
            + self.action_diversity_weight * diversity_n
            + self.movement_weight * movement_n
            + self.block_break_weight * block_break_n
            + self.item_pickup_weight * item_pickup_n
            # * Creation-focused channels
            + self.block_place_weight * block_place_n
            + self.crafting_weight * crafting_n
            + self.building_streak_weight * building_streak_n
            + self.creative_sequence_weight * creative_seq_n
            # Penalties
            - self.death_penalty_weight * death_n
            - self.safety_weight * safety_n
        )

        # Monitor all channels
        self._monitor.record("total", reward)
        self._monitor.record("intrinsic", intrinsic)
        self._monitor.record("extrinsic", extrinsic)
        self._monitor.record("exploration", exploration)
        self._monitor.record("interaction", interaction)
        self._monitor.record("action_diversity", action_diversity)
        self._monitor.record("movement", movement)
        self._monitor.record("block_break", block_break)
        self._monitor.record("item_pickup", item_pickup)
        self._monitor.record("block_place", block_place)
        self._monitor.record("crafting", crafting)
        self._monitor.record("building_streak", building_streak)
        self._monitor.record("creative_sequence", creative_sequence)
        self._monitor.record("death_penalty", death_penalty)

        self._step += 1
        return reward

    def stats(self) -> dict:
        return {
            "step": self._step,
            "intrinsic_weight": self.intrinsic_weight,
            "monitor": self._monitor.summary(),
        }
