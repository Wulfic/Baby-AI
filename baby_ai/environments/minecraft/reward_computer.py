"""
Reward computer — multi-channel extrinsic reward calculation.

Extracted from :class:`MinecraftEnv` to keep env.py focused on
environment lifecycle (init, reset, step, close) while this module
handles the complex per-step reward shaping logic.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np
import torch

from baby_ai.environments.minecraft.action_categories import (
    ATTACK_ACTIONS,
    BACKWARD_ACTIONS,
    BLOCK_INTERACTION_ACTIONS,
    DROP_ACTIONS,
    FORWARD_ACTIONS,
    HOTBAR_ACTIONS,
    JUMP_ACTIONS,
    LOOK_ACTIONS,
    MOVEMENT_ACTIONS,
    PURE_STRAFE_ACTIONS,
    SPRINT_ACTIONS,
)
from baby_ai.learning.item_rewards import get_item_reward
from baby_ai.utils.logging import get_logger

log = get_logger("mc_reward")


class RewardComputer:
    """Stateful per-step reward calculator for the Minecraft environment.

    Holds all reward-shaping accumulators (frame history, action history,
    idle streaks, etc.) and exposes :meth:`compute` to produce a dict of
    per-channel rewards plus a weighted ``total``.

    The ``env`` reference is used *read-only* for position / mod-bridge
    data and *write* for a small set of shared counters (attack_streak,
    blocks_broken_total, is_dead, last_drop_time).
    """

    def __init__(self, env: Any) -> None:
        self._env = env

        # ── Frame / action history ──────────────────────────────
        self.frame_history: deque[np.ndarray] = deque(maxlen=10)
        self.action_history: deque[int] = deque(maxlen=50)
        self.idle_streak: int = 0
        self.interaction_streak: int = 0
        self.baseline_frame: Optional[np.ndarray] = None
        self.baseline_frame_step: int = 0
        self.hotbar_streak: int = 0

        # Chunk-linger tracking: penalise staying in the same
        # area for too long.  ``chunk_linger_steps`` ticks up
        # every step and only resets after 3 *distinct new*
        # chunks have been visited.
        self.chunk_linger_steps: int = 0
        self.new_chunks_since_linger: int = 0
        self._linger_known_chunks: set[tuple[int, int]] = set()

    def reset(self) -> None:
        """Clear per-episode accumulators."""
        self.frame_history.clear()
        self.action_history.clear()
        self.idle_streak = 0
        self.interaction_streak = 0
        self.baseline_frame = None
        self.baseline_frame_step = 0
        self.hotbar_streak = 0
        self.chunk_linger_steps = 0
        self.new_chunks_since_linger = 0
        self._linger_known_chunks.clear()

    # ────────────────────────────────────────────────────────────
    #  Main entry point
    # ────────────────────────────────────────────────────────────

    def compute(
        self,
        obs: Dict[str, torch.Tensor],
        action_id: int,
        reward_weights: Any,
        step_count: int,
        observation_only: bool = False,
        hotbar_slot: int | None = None,
        is_block_interaction: bool = False,
    ) -> Dict[str, float]:
        """Compute multi-channel extrinsic reward from frame analysis and action.

        Returns a dict with per-channel values AND a ``"total"`` key with
        the combined extrinsic signal.

        When *observation_only* is ``True`` (imitation learning mode),
        action-based channels (idle penalty, hotbar spam, action
        diversity, movement, interaction, item drop) are zeroed out
        and their accumulators are not updated.  Frame-based and
        mod-event-based channels still fire normally.
        """
        env = self._env
        rewards: Dict[str, float] = {}

        # Read sub-weights once (gracefully falls back to defaults).
        _sw = reward_weights.snapshot() if reward_weights is not None else {}

        # ── 1. Survival bonus ───────────────────────────────────
        rewards["survival"] = 0.005

        # ── 2. Visual change ────────────────────────────────────
        vision_tensor = obs["vision"]
        frame_np = vision_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_u8 = (
            (frame_np * 255).astype(np.uint8)
            if frame_np.max() <= 1.0
            else frame_np.astype(np.uint8)
        )

        visual_change = 0.0
        if self.frame_history:
            prev = self.frame_history[-1]
            diff = (
                np.abs(frame_u8.astype(np.float32) - prev.astype(np.float32)).mean()
                / 255.0
            )
            if diff > 0.04:
                visual_change = min(diff * 3.0, 1.0)
        rewards["visual_change"] = visual_change
        self.frame_history.append(frame_u8)

        # ── 3. Action diversity ─────────────────────────────────
        action_div = 0.0
        if not observation_only and len(self.action_history) >= 10:
            recent = list(self.action_history)[-30:]
            unique_ratio = len(set(recent)) / max(len(recent), 1)
            action_div = unique_ratio * 0.3
        rewards["action_diversity"] = action_div

        # ── 4. Interaction bonus ──────────────────────────────
        _int_impact = _sw.get("int_impact", 0.5)
        _int_sustained = _sw.get("int_sustained", 0.2)

        interaction_bonus = 0.0
        if observation_only:
            pass  # Don't update interaction_streak during observation
        elif action_id in BLOCK_INTERACTION_ACTIONS:
            self.interaction_streak += 1
            if visual_change > 0.06:
                interaction_bonus = _int_impact + min(visual_change * 2.0, 0.5)
            elif self.interaction_streak >= 5 and visual_change > 0.03:
                interaction_bonus = _int_sustained
        else:
            self.interaction_streak = 0
        rewards["interaction"] = interaction_bonus

        # ── 5. Exploration ──────────────────────────────────────
        exploration_bonus = 0.0
        baseline_interval = 100
        if self.baseline_frame is None:
            self.baseline_frame = frame_u8.copy()
            self.baseline_frame_step = step_count
        elif step_count - self.baseline_frame_step >= baseline_interval:
            base_diff = (
                np.abs(
                    frame_u8.astype(np.float32)
                    - self.baseline_frame.astype(np.float32)
                ).mean()
                / 255.0
            )
            if base_diff > 0.08:
                exploration_bonus = min(base_diff * 3.0, 1.0)
            self.baseline_frame = frame_u8.copy()
            self.baseline_frame_step = step_count
        rewards["exploration"] = exploration_bonus

        # ── 6. Movement bonus ───────────────────────────────────
        movement_bonus = 0.0
        if observation_only:
            # In imitation mode, reward movement from actual position
            # changes (mod bridge) rather than action_id matching.
            movement_bonus = self._compute_movement_from_position(
                visual_change, _sw, env
            )
        else:
            movement_bonus = self._compute_movement(
                action_id, visual_change, _sw, env
            )
        # Chunk-based movement decay
        chunk_decay = max(0.1, 1.0 - env._steps_in_chunk * 0.02)
        movement_bonus *= chunk_decay
        rewards["movement"] = movement_bonus

        # ── 6b. New chunk exploration bonus ─────────────────────
        new_chunk_bonus = 0.0
        if (
            env._player_x is not None
            and env._player_z is not None
            and env._steps_in_chunk == 0
            and step_count > 0
        ):
            new_chunk_bonus = 0.3
        rewards["new_chunk"] = new_chunk_bonus

        # ── 7. Idle penalty ─────────────────────────────────────
        if observation_only:
            # Action-level idle tracking doesn't apply in imitation,
            # but chunk-linger still ticks so the agent learns that
            # the human also needs to explore.
            idle_penalty = self._compute_chunk_linger_penalty(env)
            self.idle_streak = 0
        else:
            idle_penalty = self._compute_idle_penalty(action_id, env)
        rewards["idle_penalty"] = idle_penalty

        # ── 7b. Hotbar spam penalty ───────────────────────────
        if observation_only:
            hotbar_spam_penalty = 0.0
            self.hotbar_streak = 0
        else:
            hotbar_spam_penalty = self._compute_hotbar_spam(hotbar_slot)
        rewards["hotbar_spam_penalty"] = hotbar_spam_penalty

        # ── Mod events ──────────────────────────────────────────
        # Events were already drained and fed to the sensor packer
        # inside _observe() so all modalities share the same temporal
        # snapshot.  Re-use the cached list here for reward signals.
        mod_events = getattr(env, '_latest_mod_events', [])

        mod_blocks_broken = [e for e in mod_events if e.get("event") == "block_broken"]
        mod_blocks_placed = [e for e in mod_events if e.get("event") == "block_placed"]
        mod_items_crafted = [e for e in mod_events if e.get("event") == "item_crafted"]
        mod_items_picked = [e for e in mod_events if e.get("event") == "item_picked_up"]
        mod_deaths = [e for e in mod_events if e.get("event") == "player_death"]
        mod_health = [e for e in mod_events if e.get("event") == "health_changed"]
        mod_food = [e for e in mod_events if e.get("event") == "food_changed"]
        mod_xp = [e for e in mod_events if e.get("event") == "xp_gained"]
        mod_position = [e for e in mod_events if e.get("event") == "position_update"]
        mod_home_set = [e for e in mod_events if e.get("event") == "home_set"]

        # ── Update player position ──────────────────────────────
        self._process_position_updates(mod_position, env)

        # ── Process /sethome from mod ────────────────────────
        if mod_home_set:
            latest = mod_home_set[-1]
            env._home_x = latest.get("x")
            env._home_z = latest.get("z")
            env._home_y = latest.get("y")
            log.info("Home location updated via mod event: (%.1f, %.1f)",
                     env._home_x, env._home_z)
            # Persist to settings store if available
            if hasattr(env, '_persist_home'):
                env._persist_home()

        # Reset attack streak when a block actually breaks
        if mod_blocks_broken:
            env._attack_streak = 0
            env._blocks_broken_total += len(mod_blocks_broken)

        use_mod = bool(mod_events) or (
            env._mod_bridge is not None and env._mod_bridge.connected
        )

        if mod_events and step_count % 50 == 0:
            log.info(
                "Mod events step %d: %s",
                step_count,
                [e["event"] for e in mod_events],
            )

        # ── 8. Block-break detection (mod-only) ────────────────
        block_break_reward = 0.0
        if use_mod:
            block_break_reward = sum(
                get_item_reward(e.get("block", ""), "block_broken")
                for e in mod_blocks_broken
            )
        rewards["block_break"] = block_break_reward

        # ── 9. Crafting detection (mod-only) ────────────────────
        craft_score = 0.0
        if use_mod:
            craft_score = sum(
                get_item_reward(e.get("item", ""), "item_crafted")
                * e.get("count", 1)
                for e in mod_items_crafted
            )

        # ── 10. Item pickup (mod-only) ──────────────────────────
        item_pickup = 0.0
        if use_mod:
            item_pickup = sum(
                get_item_reward(e.get("item", ""), "item_picked_up")
                * min(e.get("count", 1), 10)
                for e in mod_items_picked
            )

        # Suppress pickup reward shortly after a drop action
        if item_pickup > 0 and (
            time.monotonic() - env._last_drop_time
        ) < env._drop_pickup_cooldown:
            log.debug(
                "Suppressed item_pickup reward (%.3f) — within %.0fs of drop",
                item_pickup,
                env._drop_pickup_cooldown,
            )
            item_pickup = 0.0
        rewards["item_pickup"] = item_pickup

        # ── Item drop penalty ───────────────────────────────────
        if observation_only:
            item_drop_penalty = 0.0
        else:
            item_drop_penalty = 0.3 if action_id in DROP_ACTIONS else 0.0
        rewards["item_drop_penalty"] = item_drop_penalty

        # ── 11. Death detection (mod-only) ──────────────────────
        if mod_deaths:
            log.info("Death detected via mod at step %d", step_count)
            env._is_dead = True
        rewards["death_penalty"] = 1.0 if mod_deaths else 0.0

        # ── 11b. Damage / Healing ───────────────────────────────
        damage_taken, healing = self._compute_health_rewards(use_mod, mod_health)
        rewards["damage_taken"] = damage_taken
        rewards["healing"] = healing

        # ── 11c. Food / Hunger reward ───────────────────────────
        food_reward = 0.0
        if use_mod:
            for evt in mod_food:
                delta = evt.get("delta", 0)
                if delta > 0:
                    food_reward += float(delta)
        rewards["food_reward"] = food_reward

        # ── 11d. Experience points reward ───────────────────────
        xp_reward = 0.0
        if use_mod:
            for evt in mod_xp:
                xp_reward += float(evt.get("amount", 0))
        rewards["xp_reward"] = xp_reward

        # ── 11e. Height / cave / fall penalties ─────────────────
        height_penalty = self._compute_height_penalty(_sw, env)
        rewards["height_penalty"] = height_penalty

        # ── 11e-2. Extreme pitch penalty ────────────────────────
        pitch_penalty = self._compute_pitch_penalty(env, is_block_interaction)
        rewards["pitch_penalty"] = pitch_penalty

        # ── 11f. Home proximity bonus / distance penalty ────────
        home_bonus = self._compute_home_proximity(env)
        rewards["home_proximity"] = home_bonus

        # ── 12. Block placement (mod-only) ──────────────────────
        block_place_score = 0.0
        if use_mod:
            block_place_score = sum(
                get_item_reward(e.get("block", ""), "block_placed")
                for e in mod_blocks_placed
            )
        rewards["block_place"] = block_place_score

        # ── 13. Crafting reward ─────────────────────────────────
        rewards["crafting"] = craft_score

        # ── 14. Building streak ─────────────────────────────────
        placed_this_step = block_place_score > 0.5
        streak_info = env._building_streak.update(placed_this_step)
        rewards["building_streak"] = streak_info["streak_bonus"]

        # ── 15. Creative sequence ───────────────────────────────
        seq_info = env._creative_sequence.update(
            block_break=block_break_reward,
            item_pickup=item_pickup,
            craft_score=craft_score,
            block_place=block_place_score,
        )
        rewards["creative_sequence"] = seq_info["stage_reward"] + seq_info["cycle_bonus"]

        # ── 16. Stagnation penalty ──────────────────────────────
        if observation_only:
            # Freeze the stagnation timer during imitation — the
            # player controls whether blocks are broken/crafted.
            stagnation_penalty = 0.0
        else:
            stagnation_penalty = self._compute_stagnation(
                block_break_reward, craft_score, block_place_score,
                item_pickup, step_count, env,
            )
        rewards["stagnation_penalty"] = stagnation_penalty

        # ── Diagnostic logging ──────────────────────────────────
        if step_count % 50 == 0:
            self._log_diagnostics(
                step_count, env, block_break_reward, item_pickup,
                block_place_score, craft_score, visual_change,
                interaction_bonus,
            )

        # ── Combine with weights ────────────────────────────────
        total = self._combine_rewards(rewards, reward_weights)
        rewards["total"] = total

        return rewards

    # ────────────────────────────────────────────────────────────
    #  Private helpers — one per reward channel / sub-computation
    # ────────────────────────────────────────────────────────────

    def _compute_movement(
        self,
        action_id: int,
        visual_change: float,
        _sw: dict,
        env: Any,
    ) -> float:
        _mv_forward = _sw.get("mv_forward", 3.0)
        _mv_backward = _sw.get("mv_backward", 1.0)
        _mv_strafe = _sw.get("mv_strafe", 0.4)
        _mv_look = _sw.get("mv_look", 0.02)
        _mv_jump = _sw.get("mv_jump", 1.5)
        _mv_sprint = _sw.get("mv_sprint", 1.5)

        movement_bonus = 0.0
        _actual_moved = False
        if (
            env._player_x is not None
            and env._prev_player_x is not None
            and env._player_z is not None
            and env._prev_player_z is not None
        ):
            dx = env._player_x - env._prev_player_x
            dz = env._player_z - env._prev_player_z
            horiz_dist = (dx * dx + dz * dz) ** 0.5
            _actual_moved = horiz_dist > 0.1

        if action_id in MOVEMENT_ACTIONS and visual_change > 0.02 and _actual_moved:
            base_move = visual_change * 0.3
            if action_id in FORWARD_ACTIONS:
                base_move *= _mv_forward
            elif action_id in BACKWARD_ACTIONS:
                base_move *= _mv_backward
            elif action_id in PURE_STRAFE_ACTIONS:
                base_move *= _mv_strafe
            if action_id in JUMP_ACTIONS:
                base_move *= _mv_jump
            if action_id in SPRINT_ACTIONS:
                base_move *= _mv_sprint
            movement_bonus = base_move

        # Camera look bonus
        _mod_has_position = (
            env._player_x is not None and env._prev_player_x is not None
        )
        if action_id in LOOK_ACTIONS and action_id not in PURE_STRAFE_ACTIONS:
            if _mod_has_position:
                if _actual_moved:
                    movement_bonus += _mv_look
            else:
                movement_bonus += _mv_look * 0.5

        return movement_bonus

    @staticmethod
    def _compute_movement_from_position(
        visual_change: float,
        _sw: dict,
        env: Any,
    ) -> float:
        """Compute movement reward from actual position changes.

        Used in imitation (observation_only) mode where the AI doesn't
        choose the action, so ``action_id``-based gating is wrong.
        Instead, reward is based purely on whether the player actually
        moved (from mod-bridge position updates).
        """
        _mv_forward = _sw.get("mv_forward", 3.0)
        _mv_look = _sw.get("mv_look", 0.02)

        if (
            env._player_x is None
            or env._prev_player_x is None
            or env._player_z is None
            or env._prev_player_z is None
        ):
            return 0.0

        dx = env._player_x - env._prev_player_x
        dz = env._player_z - env._prev_player_z
        horiz_dist = (dx * dx + dz * dz) ** 0.5

        bonus = 0.0
        if horiz_dist > 0.1 and visual_change > 0.02:
            # Treat all imitation movement like forward movement
            bonus = visual_change * 0.3 * _mv_forward

        # Camera look component — reward looking around while moving
        if env._player_yaw is not None and env._prev_player_yaw is not None:
            dyaw = abs(env._player_yaw - env._prev_player_yaw)
            if dyaw > 180:
                dyaw = 360 - dyaw  # shortest arc
            if dyaw > 2.0:
                bonus += _mv_look

        return bonus

    def _compute_idle_penalty(self, action_id: int, env: Any) -> float:
        idle_penalty = 0.0

        # ── Action-level noop streak ────────────────────────────
        if action_id == 0:
            self.idle_streak += 1
            if self.idle_streak > 3:
                idle_penalty = min(0.08 * (self.idle_streak - 3), 0.8)
        else:
            self.idle_streak = max(0, self.idle_streak - 2)

        if len(self.action_history) >= 10:
            last_10 = list(self.action_history)[-10:]
            if len(set(last_10)) == 1:
                idle_penalty += 0.15

        # ── Chunk-linger penalty ────────────────────────────────
        idle_penalty += self._compute_chunk_linger_penalty(env)

        return idle_penalty

    def _compute_chunk_linger_penalty(self, env: Any) -> float:
        """Progressive penalty for staying in the same area too long.

        Ticks up every step. Only resets after 3 distinct NEW chunks
        have been visited (re-entering already-known chunks doesn't
        count). This encourages continuous exploration.

        Returns:
            float penalty value (0.0 during grace period, ramps to 0.6).
        """
        self.chunk_linger_steps += 1

        # Track new chunk arrivals toward the 3-chunk reset
        current = env._current_chunk
        if current is not None and current not in self._linger_known_chunks:
            self._linger_known_chunks.add(current)
            self.new_chunks_since_linger += 1

        if self.new_chunks_since_linger >= 3:
            # Reset — agent has explored enough
            self.chunk_linger_steps = 0
            self.new_chunks_since_linger = 0
            self._linger_known_chunks.clear()

        # Grace period of 30 steps, then progressive ramp
        _LINGER_GRACE = 30
        if self.chunk_linger_steps > _LINGER_GRACE:
            over = self.chunk_linger_steps - _LINGER_GRACE
            # Ramps: 0.02 at step 31, 0.10 at step 35, caps at 0.6
            return min(0.02 * over, 0.6)
        return 0.0

    def _compute_hotbar_spam(self, hotbar_slot: int | None) -> float:
        """Detect hotbar spam from keyboard 0-9 presses.

        Uses the decoded ``hotbar_slot`` (1-9 or None) directly from
        the continuous action decoder, instead of relying on fuzzy
        discrete action-id matching.  This correctly catches every
        number-key press regardless of what other keys are held.
        """
        hotbar_spam_penalty = 0.0
        is_hotbar = hotbar_slot is not None

        if is_hotbar:
            self.hotbar_streak += 1
            if self.hotbar_streak >= 2:
                hotbar_spam_penalty = min(0.1 * self.hotbar_streak, 0.6)
        else:
            self.hotbar_streak = max(0, self.hotbar_streak - 1)

        # Also check recent history for excessive hotbar usage
        if len(self.action_history) >= 10:
            last_10 = list(self.action_history)[-10:]
            hotbar_count = sum(1 for a in last_10 if a in HOTBAR_ACTIONS)
            if hotbar_count > 3:
                hotbar_spam_penalty += 0.05 * (hotbar_count - 3)
        return hotbar_spam_penalty

    @staticmethod
    def _process_position_updates(mod_position: list, env: Any) -> None:
        if not mod_position:
            return
        latest_pos = mod_position[-1]
        env._prev_player_y = env._player_y
        env._prev_player_x = env._player_x
        env._prev_player_z = env._player_z
        env._prev_player_yaw = env._player_yaw
        env._player_y = latest_pos.get("y", env._player_y)
        env._player_x = latest_pos.get("x", env._player_x)
        env._player_z = latest_pos.get("z", env._player_z)
        env._player_on_ground = latest_pos.get("on_ground", True)
        env._sky_light = latest_pos.get("light", 15)
        env._player_pitch = latest_pos.get("pitch", env._player_pitch)
        env._player_yaw = latest_pos.get("yaw", env._player_yaw)

        # First position_update ever -> set as home location
        if env._home_x is None and env._player_x is not None:
            env._home_x = env._player_x
            env._home_z = env._player_z
            env._home_y = env._player_y
            log.info("Home location set: (%.1f, %.1f)", env._home_x, env._home_z)
            # Persist and notify GUI
            if hasattr(env, '_persist_home'):
                env._persist_home()
            if hasattr(env, '_on_home_changed') and env._on_home_changed:
                env._on_home_changed()

        # Calibrate underground threshold
        if not env._surface_y_calibrated and env._player_y is not None:
            env._surface_y = env._player_y - 5.0
            env._surface_y_calibrated = True
            log.info(
                "Surface threshold calibrated: Y < %.1f = underground (player Y=%.1f)",
                env._surface_y,
                env._player_y,
            )

        # Chunk tracking
        if env._player_x is not None and env._player_z is not None:
            cx = int(env._player_x) // 16
            cz = int(env._player_z) // 16
            new_chunk = (cx, cz)
            if new_chunk != env._current_chunk:
                env._current_chunk = new_chunk
                env._steps_in_chunk = 0
                env._visited_chunks.add(new_chunk)
            else:
                env._steps_in_chunk += 1

    @staticmethod
    def _compute_health_rewards(
        use_mod: bool, mod_health: list
    ) -> tuple[float, float]:
        damage_taken = 0.0
        healing = 0.0
        if use_mod:
            for evt in mod_health:
                delta = evt.get("delta", 0.0)
                if delta < 0:
                    damage_taken += abs(delta)
                elif delta > 0:
                    healing += delta
        return damage_taken, healing

    @staticmethod
    def _compute_height_penalty(_sw: dict, env: Any) -> float:
        _h_underground = _sw.get("height_underground", 1.0)
        _h_fall = _sw.get("height_fall", 1.0)
        _h_darkness = _sw.get("height_darkness", 1.0)

        height_penalty = 0.0
        if env._player_y is not None:
            # Underground penalty
            if env._player_y < env._surface_y:
                env._underground_steps += 1
                depth = max(0.0, env._surface_y - env._player_y)
                underground_base = min(0.03 + 0.005 * depth, 0.3)
                if env._underground_steps > 50:
                    overshoot = (env._underground_steps - 50) / 100.0
                    underground_base += min(0.1 * overshoot, 0.4)
                height_penalty += underground_base * _h_underground
            else:
                env._underground_steps = max(0, env._underground_steps - 5)

            # Fall penalty
            if env._prev_player_y is not None:
                y_drop = env._prev_player_y - env._player_y
                if y_drop > 3.0:
                    height_penalty += min(0.15 * y_drop, 1.0) * _h_fall
                    log.debug(
                        "Fall detected: %.1f blocks (Y %.1f -> %.1f)",
                        y_drop,
                        env._prev_player_y,
                        env._player_y,
                    )

            # Darkness penalty
            if env._sky_light <= 4:
                height_penalty += 0.05 * _h_darkness
        return height_penalty

    @staticmethod
    def _compute_pitch_penalty(env: Any, is_block_interaction: bool = False) -> float:
        """Penalise extreme camera pitch (>45° up or down).

        Rules:
        - Threshold is **45°** (not 60°).  Looking up is penalised
          more steeply than looking down.
        - **Exempt when interacting with a block** (attack / use):
          mining straight down or placing high are legitimate.
        - **3-second grace period** (~30 steps at 10 steps/s):
          only activates after the camera has been stuck at an
          extreme angle for 30+ consecutive steps.
        """
        _GRACE_STEPS = 30  # ~3 seconds at 100ms step delay

        pitch_penalty = 0.0
        if env._player_pitch is not None:
            pitch = env._player_pitch          # negative = up, positive = down
            abs_pitch = abs(pitch)
            looking_up = pitch < 0

            # No penalty while actively interacting with a block
            if is_block_interaction:
                # Slowly decay the counter so brief interactions
                # don't permanently reset a long stare.
                env._extreme_pitch_steps = max(0, env._extreme_pitch_steps - 2)
                return 0.0

            if abs_pitch > 45:
                env._extreme_pitch_steps += 1
                # Only penalise after the grace period
                if env._extreme_pitch_steps > _GRACE_STEPS:
                    over = env._extreme_pitch_steps - _GRACE_STEPS
                    if looking_up:
                        # Steeper ramp for sky-staring
                        pitch_penalty = 0.15 + 0.03 * min(over, 40)
                    else:
                        # Gentler ramp for looking down
                        pitch_penalty = 0.08 + 0.015 * min(over, 40)
            else:
                env._extreme_pitch_steps = max(0, env._extreme_pitch_steps - 3)
        return pitch_penalty

    @staticmethod
    def _compute_home_proximity(env: Any) -> float:
        home_bonus = 0.0
        if env._home_x is not None and env._player_x is not None:
            dx = env._player_x - env._home_x
            dz = env._player_z - env._home_z
            dist = (dx * dx + dz * dz) ** 0.5
            if dist <= env._home_radius:
                home_bonus = 0.02
            else:
                # Logarithmic penalty so it grows slowly at extreme
                # distance instead of instantly capping at -0.5.
                import math
                overshoot = (dist - env._home_radius) / env._home_radius
                home_bonus = -min(0.02 * math.log1p(overshoot), 0.15)
        return home_bonus

    @staticmethod
    def _compute_stagnation(
        block_break_reward: float,
        craft_score: float,
        block_place_score: float,
        item_pickup: float,
        step_count: int,
        env: Any,
    ) -> float:
        productive = (
            block_break_reward > 0
            or craft_score > 0
            or block_place_score > 0.3
            or item_pickup > 0
        )
        if productive:
            env._last_productive_step = step_count

        steps_since_productive = step_count - env._last_productive_step
        stagnation_penalty = 0.0
        if steps_since_productive > env._stagnation_timeout:
            overshoot = (
                (steps_since_productive - env._stagnation_timeout)
                / env._stagnation_timeout
            )
            stagnation_penalty = min(0.15 + 0.25 * overshoot, 1.5)
        return stagnation_penalty

    @staticmethod
    def _log_diagnostics(
        step_count: int,
        env: Any,
        block_break_reward: float,
        item_pickup: float,
        block_place_score: float,
        craft_score: float,
        visual_change: float,
        interaction_bonus: float,
    ) -> None:
        raw_h, raw_w = (
            env._raw_frame.shape[:2] if env._raw_frame is not None else (0, 0)
        )
        if env._mod_bridge and env._mod_bridge.connected:
            hb = "alive" if env._mod_bridge.pipeline_alive else "NO-HB"
            bridge_status = (
                f"connected({hb},tick={env._mod_bridge.last_heartbeat_tick},"
                f"evt={env._mod_bridge.total_events_received})"
            )
        else:
            bridge_status = "disconnected"
        log.info(
            "Detector diagnostics step %d | mod_bridge=%s | raw_frame=%dx%d"
            " | blk_brk=%.3f item=%.3f place=%.3f"
            " | craft=%.3f death=%.1f"
            " | vis_change=%.4f interact=%.3f"
            " | long_brk=%d atk_streak=%d"
            " | Y=%.1f pitch=%.1f sky_light=%d underground=%d"
            " | home_dist=%.1f",
            step_count,
            bridge_status,
            raw_h,
            raw_w,
            block_break_reward,
            item_pickup,
            block_place_score,
            craft_score,
            0.0,
            visual_change,
            interaction_bonus,
            env._long_break_count,
            env._attack_streak,
            env._player_y if env._player_y is not None else -1.0,
            env._player_pitch if env._player_pitch is not None else 0.0,
            env._sky_light,
            env._underground_steps,
            (
                (
                    ((env._player_x or 0) - (env._home_x or 0)) ** 2
                    + ((env._player_z or 0) - (env._home_z or 0)) ** 2
                )
                ** 0.5
            )
            if env._home_x is not None
            else -1.0,
        )

    @staticmethod
    def _combine_rewards(
        rewards: Dict[str, float], reward_weights: Any
    ) -> float:
        if reward_weights is not None:
            w = reward_weights.snapshot()
        else:
            # Hardcoded fallback (matches RewardWeightsState defaults)
            w = {
                "intrinsic": 0.1,
                "survival": 1.0,
                "visual_change": 0.1,
                "action_diversity": 0.5,
                "interaction": 0.8,
                "exploration": 0.8,
                "movement": 0.3,
                "block_break": 4.0,
                "item_pickup": 6.0,
                "block_place": 4.0,
                "crafting": 25.0,
                "building_streak": 3.0,
                "creative_sequence": 6.0,
                "idle_penalty": 2.0,
                "death_penalty": 5.0,
                "stagnation_penalty": 3.0,
                "item_drop_penalty": 3.0,
                "damage_taken": 1.5,
                "hotbar_spam_penalty": 2.0,
                "height_penalty": 2.5,
                "pitch_penalty": 3.0,
                "healing": 1.0,
                "food_reward": 0.8,
                "xp_reward": 0.1,
                "home_proximity": 1.5,
                # Sub-weights (internal multipliers)
                "mv_forward": 3.0,
                "mv_backward": 1.0,
                "mv_strafe": 0.4,
                "mv_look": 0.02,
                "mv_jump": 1.5,
                "mv_sprint": 1.5,
                "int_impact": 0.5,
                "int_sustained": 0.2,
                "height_underground": 1.0,
                "height_fall": 1.0,
                "height_darkness": 1.0,
            }

        return (
            # Baseline
            rewards["survival"] * w.get("survival", 1.0)
            + rewards["visual_change"] * w.get("visual_change", 0.1)
            # Exploration
            + rewards["action_diversity"] * w.get("action_diversity", 0.5)
            + rewards["interaction"] * w.get("interaction", 0.8)
            + rewards["exploration"] * w.get("exploration", 0.8)
            + rewards["movement"] * w.get("movement", 0.3)
            + rewards["new_chunk"] * w.get("new_chunk", 1.0)
            # Resource gathering
            + rewards["block_break"] * w.get("block_break", 4.0)
            + rewards["item_pickup"] * w.get("item_pickup", 6.0)
            # Creation
            + rewards["block_place"] * w.get("block_place", 4.0)
            + rewards["crafting"] * w.get("crafting", 25.0)
            + rewards["building_streak"] * w.get("building_streak", 3.0)
            + rewards["creative_sequence"] * w.get("creative_sequence", 6.0)
            # Penalties (subtracted)
            - rewards["idle_penalty"] * w.get("idle_penalty", 2.0)
            - rewards["death_penalty"] * w.get("death_penalty", 5.0)
            - rewards["stagnation_penalty"] * w.get("stagnation_penalty", 3.0)
            - rewards["item_drop_penalty"] * w.get("item_drop_penalty", 3.0)
            - rewards["damage_taken"] * w.get("damage_taken", 1.5)
            - rewards["hotbar_spam_penalty"] * w.get("hotbar_spam_penalty", 2.0)
            - rewards["height_penalty"] * w.get("height_penalty", 2.5)
            - rewards["pitch_penalty"] * w.get("pitch_penalty", 3.0)
            # Sustain
            + rewards["healing"] * w.get("healing", 1.0)
            + rewards["food_reward"] * w.get("food_reward", 0.8)
            + rewards["xp_reward"] * w.get("xp_reward", 0.1)
            # Home proximity (can be positive or negative)
            + rewards["home_proximity"] * w.get("home_proximity", 1.5)
        )
