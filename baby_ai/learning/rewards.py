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
    - intrinsic: curiosity (JEPA prediction error) + learning progress
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
        intrinsic_weight_start: float = 0.5,
        intrinsic_weight_end: float = 0.05,
        intrinsic_decay_steps: int = 500_000,
        comm_weight: float = 0.1,
        extrinsic_weight: float = 0.3,
        exploration_weight: float = 0.3,
        interaction_weight: float = 0.4,
        action_diversity_weight: float = 0.15,
        movement_weight: float = 0.1,
        block_break_weight: float = 0.8,
        item_pickup_weight: float = 1.0,
        # * Creation-focused weights (highest, but bounded for z-scored channels)
        block_place_weight: float = 1.2,
        crafting_weight: float = 2.0,
        building_streak_weight: float = 1.0,
        creative_sequence_weight: float = 2.5,
        # Penalties
        death_penalty_weight: float = 2.0,
        safety_weight: float = 1.0,
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

    # Channels whose normalized value should NEVER go negative.
    #
    # Two categories:
    # 1. **Sparse events** (block_break, crafting …): 0 most of the
    #    time, occasional positive spike.  Z-scoring them creates a
    #    systematic negative bias on idle steps.
    # 2. **Inherently-positive continuous** (intrinsic, extrinsic):
    #    always > 0 but the level can shift (e.g. after distillation
    #    the curiosity baseline drops).  The rolling z-score window
    #    (1000 steps) lags behind, producing hundreds of negative
    #    z-scores until the mean catches up — this was the primary
    #    driver of the runaway-negative episode reward.
    #
    # Clamping the floor to 0 ensures "below average" is neutral (0),
    # not punishing.  Positive deviations still get full credit.
    _POSITIVE_ONLY_CHANNELS: frozenset[str] = frozenset({
        # Continuous positive signals
        "intrinsic", "extrinsic",
        # Sparse event channels
        "communication", "exploration", "interaction", "action_diversity",
        "movement", "block_break", "item_pickup", "block_place",
        "crafting", "building_streak", "creative_sequence",
        "new_chunk", "healing", "food_reward", "xp_reward",
    })

    def _normalize_channel(self, channel: str, value: float) -> float:
        """Normalize a reward channel by running mean/std.

        Values are z-scored against a rolling window, then **clamped**
        to prevent catastrophic reward spikes from sparse channels.

        For *sparse positive* channels (block_break, crafting, etc.) the
        lower clamp is 0 so that the common-case value of 0 never
        contributes a negative reward.  Continuous channels (intrinsic,
        extrinsic) keep the symmetric [-3, 3] clamp.

        Key safeguards:
        - Minimum std floor of 0.1 prevents divide-by-near-zero on
          sparse binary channels (block_break, crafting, etc.)
        - Warmup period returns scaled raw values until we have
          enough data for meaningful statistics.
        - Tighter clamp keeps per-channel contributions bounded even
          after multiplication by channel weights.
        """
        if not self.normalize:
            return value

        # Determine clamp bounds: positive-only channels floor at 0.
        lo = 0.0 if channel in self._POSITIVE_ONLY_CHANNELS else -3.0

        if channel not in self._stats:
            self._stats[channel] = {"values": deque(maxlen=self._window)}

        self._stats[channel]["values"].append(value)

        # Warmup: not enough data for reliable statistics.
        # Return value scaled down to prevent spikes from raw magnitudes.
        if len(self._stats[channel]["values"]) < 30:
            return float(np.clip(value, lo, 3.0))

        arr = np.array(self._stats[channel]["values"])
        mean = arr.mean()
        # Minimum std floor of 0.1 — critical for sparse channels where
        # std is near-zero (e.g., block_break: 99% zeros, 1% non-zero).
        # Without this, a single event gets z-scored to z=10+ even though
        # it's a perfectly normal game event.
        std = max(arr.std(), 0.1)
        z = (value - mean) / std
        # Tight clamp: even with weights up to 2.5, max single-channel
        # contribution is 2.5 * 3.0 = 7.5 (vs old 10.0 * 5.0 = 50.0)
        return float(np.clip(z, lo, 3.0))

    def compose(
        self,
        intrinsic: float = 0.0,
        communication: float = 0.0,
        extrinsic: float = 0.0,
        # NOTE: compose() is legacy code — the Minecraft loop uses
        # compose_dynamic() exclusively.  Kept for reference / testing.
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

        # Clamp total reward to [-5, 5].  With reduced weights and
        # tighter z-score clamps, the theoretical max is ~15 from all
        # channels firing at once — but the value head and advantage
        # computation work best with a compact reward range.
        reward = float(np.clip(reward, -5.0, 5.0))

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

    # ── Penalty channels: values that should be subtracted, not added ──
    _PENALTY_CHANNELS: frozenset[str] = frozenset({
        "death_penalty", "safety_penalty", "idle_penalty",
        "stagnation_penalty", "item_drop_penalty", "damage_taken",
        "hotbar_spam_penalty", "height_penalty", "pitch_penalty",
    })

    # Channels that skip z-score normalization (applied raw).
    _RAW_CHANNELS: frozenset[str] = frozenset({
        "death_penalty", "safety_penalty", "idle_penalty",
        "stagnation_penalty", "item_drop_penalty", "damage_taken",
        "hotbar_spam_penalty", "height_penalty", "pitch_penalty",
    })

    def compose_dynamic(
        self,
        channel_values: dict[str, float],
        weight_overrides: dict[str, float] | None = None,
    ) -> float:
        """Compose reward from arbitrary channels with optional weight overrides.

        This is the preferred entry-point for the Minecraft loop.  It
        accepts the raw per-channel values from the environment together
        with the live GUI weight overrides and produces a single scalar
        reward with full z-score normalization and intrinsic weight
        annealing.

        Args:
            channel_values: Mapping of channel_name → raw float value.
                           Must include 'intrinsic' for curiosity.
                           NOTE: must NOT include 'total' — that key
                           is skipped to prevent double-counting.
            weight_overrides: Optional mapping of channel_name → weight
                             from the GUI reward-weight sliders.  When a
                             channel is absent from this dict the fallback
                             weight is 1.0.  In practice, the GUI always
                             provides weights for all known channels (see
                             RewardWeightsState.snapshot()), so the 1.0
                             fallback only applies to novel / unknown
                             channels.

        Returns:
            Composed, clamped scalar reward in [-5, 5].
        """
        w = weight_overrides or {}
        reward = 0.0

        for channel, raw_value in channel_values.items():
            if raw_value == 0.0:
                continue

            # Skip pre-composed totals — including them would
            # double-count the reward that compose_dynamic() itself
            # is about to compute.  The env's reward_breakdown dict
            # includes a "total" key for diagnostic purposes only.
            if channel == "total":
                continue

            # Look up weight — special-case 'intrinsic' to use annealed weight
            if channel == "intrinsic":
                weight = self.intrinsic_weight * w.get("intrinsic", 1.0)
            else:
                weight = w.get(channel, 1.0)

            # Normalize (skip for penalty channels — apply raw)
            if channel in self._RAW_CHANNELS:
                normed = raw_value
            else:
                normed = self._normalize_channel(channel, raw_value)

            # Penalty channels are subtracted
            if channel in self._PENALTY_CHANNELS:
                reward -= weight * normed
            else:
                reward += weight * normed

            # Monitor
            self._monitor.record(channel, raw_value)

        reward = float(np.clip(reward, -5.0, 5.0))
        self._monitor.record("total", reward)
        self._step += 1
        return reward
