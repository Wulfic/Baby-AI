"""
Minecraft environment — ties window capture and input together.

This is the top-level class used by the training loop.  It implements
the :class:`~baby_ai.environments.base.GameEnvironment` interface so
it can be swapped for any other environment in the future.

Usage::

    from baby_ai.environments.minecraft import MinecraftEnv
    from baby_ai.config import MinecraftConfig

    env = MinecraftEnv(MinecraftConfig())
    obs = env.reset()
    for _ in range(1000):
        obs, reward, done, info = env.step(action_id=42)
    env.close()
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from baby_ai.environments.base import GameEnvironment
from baby_ai.environments.minecraft.actions import (
    MINECRAFT_ACTIONS,
    NUM_ACTIONS,
    action_name,
    has_look,
)
from baby_ai.environments.minecraft.capture import ScreenCapture
from baby_ai.environments.minecraft.input_guard import InputGuard
from baby_ai.environments.minecraft.input_controller import InputController
from baby_ai.environments.minecraft.launcher import MinecraftLauncher
from baby_ai.environments.minecraft.screen_analyzer import (
    detect_death,
    BlockBreakTracker,
    HotbarTracker,
    CraftingTracker,
    PlacementTracker,
    BuildingStreakTracker,
    CreativeSequenceTracker,
)
from baby_ai.environments.minecraft.mod_bridge import ModBridge
from baby_ai.environments.minecraft.window import WindowManager
from baby_ai.learning.item_rewards import get_item_reward
from baby_ai.utils.logging import get_logger

log = get_logger("mc_env")

# Forward-declare; set at runtime via set_reward_weights().
_reward_weights = None

def set_reward_weights(state) -> None:
    """Inject the shared RewardWeightsState so the env reads dynamic weights.

    Called once from main.py after creating the env and reward weights state.
    Must be called before the training loop starts.
    """
    global _reward_weights
    _reward_weights = state

# ── Action categories for reward shaping ────────────────────────
# Which action indices involve actual interaction with the world
_INTERACTION_ACTIONS = set()
_MOVEMENT_ACTIONS = set()
for _i, _a in enumerate(MINECRAFT_ACTIONS):
    # Actions that involve left/right mouse (attack / use / place)
    if _a.buttons:
        _INTERACTION_ACTIONS.add(_i)
    # Actions that involve movement keys (W, A, S, D, SPACE)
    if _a.keys:
        _INTERACTION_ACTIONS.add(_i)  # any non-noop is "doing something"
    # Specifically movement (walking, jumping, strafing)
    from baby_ai.environments.minecraft.input_controller import VK as _VK
    _move_vks = {_VK.get(k) for k in ("W", "S", "A", "D", "SPACE") if k in _VK}
    if _a.keys & _move_vks:
        _MOVEMENT_ACTIONS.add(_i)
# Actions that specifically attack or place/use blocks
_BLOCK_INTERACTION_ACTIONS = set()
_ATTACK_ACTIONS = set()
_USE_ACTIONS = set()
_DROP_ACTIONS = set()
for _i, _a in enumerate(MINECRAFT_ACTIONS):
    if "left" in _a.buttons or "right" in _a.buttons:
        _BLOCK_INTERACTION_ACTIONS.add(_i)
    if "left" in _a.buttons:
        _ATTACK_ACTIONS.add(_i)
    if "right" in _a.buttons:
        _USE_ACTIONS.add(_i)
    # Q key = drop item — used to suppress pickup reward after drops
    if _VK.get("Q") in _a.keys:
        _DROP_ACTIONS.add(_i)

# ── Movement quality categories ─────────────────────────────────
# Actions that involve forward movement (W key) — preferred locomotion
_FORWARD_ACTIONS = set()
# Actions that involve backward movement (S key without W)
_BACKWARD_ACTIONS = set()
# Actions that involve camera look — preferred for situational awareness
_LOOK_ACTIONS = set()
# Pure strafe actions (A/D without W) — less desirable wandering
_PURE_STRAFE_ACTIONS = set()
# Jump actions (SPACE key)
_JUMP_ACTIONS = set()
# Sprint actions (LCTRL key)
_SPRINT_ACTIONS = set()
# Hotbar selection actions — spamming these produces no useful behaviour
_HOTBAR_ACTIONS = set()
_w_vk = _VK.get("W")
_s_vk = _VK.get("S")
_a_vk = _VK.get("A")
_d_vk = _VK.get("D")
_space_vk = _VK.get("SPACE")
_lctrl_vk = _VK.get("LCTRL")
for _i, _a in enumerate(MINECRAFT_ACTIONS):
    if _w_vk and _w_vk in _a.keys:
        _FORWARD_ACTIONS.add(_i)
    if _s_vk and _s_vk in _a.keys and not (_w_vk and _w_vk in _a.keys):
        _BACKWARD_ACTIONS.add(_i)
    if _a.look is not None:
        _LOOK_ACTIONS.add(_i)
    # Pure strafe = has A or D but NOT W
    has_strafe = (_a_vk and _a_vk in _a.keys) or (_d_vk and _d_vk in _a.keys)
    has_forward = _w_vk and _w_vk in _a.keys
    if has_strafe and not has_forward:
        _PURE_STRAFE_ACTIONS.add(_i)
    if _space_vk and _space_vk in _a.keys:
        _JUMP_ACTIONS.add(_i)
    if _lctrl_vk and _lctrl_vk in _a.keys:
        _SPRINT_ACTIONS.add(_i)

# Hotbar actions: indices that ONLY press a number key (no movement/mouse)
for _i, _a in enumerate(MINECRAFT_ACTIONS):
    _name = _a.name
    if _name.startswith("hotbar_") and not _a.buttons and not _a.look:
        _HOTBAR_ACTIONS.add(_i)
# Also include inventory and drop-only as "non-productive" tap actions
# Note: drop is already penalised via item_drop_penalty, so only hotbar here.


class MinecraftEnv(GameEnvironment):
    """
    Full Minecraft environment for Baby-AI.

    Captures the game window, preprocesses frames into model-ready
    tensors, and sends actions as keyboard/mouse events via PostMessage.

    **No external mods or APIs required** — works with any Minecraft
    version (Java or Bedrock) running in windowed mode.

    Args:
        window_title: Substring to match in window title (case-insensitive).
        input_mode: ``"background"`` (safe) or ``"active"`` (with camera look).
        resolution: ``(H, W)`` for captured frames.
        step_delay_ms: Minimum time between consecutive steps.
        sensor_channels: Number of extra sensor channels (filled with game
                         metadata when available, zeros otherwise).
    """

    def __init__(
        self,
        window_title: str = "Minecraft",
        input_mode: str = "background",
        resolution: Tuple[int, int] = (360, 640),
        step_delay_ms: float = 100.0,
        sensor_channels: int = 16,
        # ── Auto-launcher options ───────────────────────────────
        auto_launch: bool = False,
        mc_dir: str = "",
        mc_version: str = "1.21.11",
        world_name: str = "",
        player_name: str = "",
        player_uuid: str = "",
        max_memory_mb: int = 4096,
        window_width: int = 1920,
        window_height: int = 1080,
        launch_timeout_sec: float = 120.0,
        # ── Input guard ─────────────────────────────────────────
        block_user_input: bool = False,
        # ── Mod bridge ──────────────────────────────────────────
        mod_bridge_port: int = 5556,
    ):
        # ── Auto-launch Minecraft if requested ──────────────────
        self._launcher: Optional[MinecraftLauncher] = None
        self._guard: Optional[InputGuard] = None
        hwnd: Optional[int] = None

        if auto_launch and mc_dir:
            log.info("Auto-launching Minecraft %s ...", mc_version)
            self._launcher = MinecraftLauncher(
                mc_dir=mc_dir,
                version=mc_version,
                player_name=player_name,
                player_uuid=player_uuid,
                max_memory_mb=max_memory_mb,
                window_width=window_width,
                window_height=window_height,
            )

            # Ensure MC will keep running when unfocused
            self._launcher.ensure_background_options()

            hwnd = self._launcher.launch(
                world=world_name or None,
                timeout_sec=launch_timeout_sec,
            )
            log.info("Minecraft launched — hwnd=%s", hex(hwnd))

            # Wait for the world to actually finish loading.
            # Returns the (possibly new) hwnd — MC may create a new window.
            if world_name:
                new_hwnd = self._launcher.wait_for_world_ready(timeout_sec=launch_timeout_sec)
                if new_hwnd is not None:
                    hwnd = new_hwnd
                    log.info("Using post-load hwnd=%s", hex(hwnd))

        # ── Components ──────────────────────────────────────────
        self._window = WindowManager(hwnd=hwnd, title_search=window_title)
        self._capture = ScreenCapture(self._window, resolution=resolution)
        self._input = InputController(self._window, mode=input_mode)

        # ── Input guard ─────────────────────────────────────────
        # InputGuard intercepts physical keyboard/mouse if MC is focused.
        if block_user_input:
            self._guard = InputGuard(mc_hwnd=self._window.hwnd)
            self._guard.start()

        self._resolution = resolution
        self._step_delay = step_delay_ms / 1000.0
        self._sensor_channels = sensor_channels

        # ── Episode state ───────────────────────────────────────
        self._step_count = 0
        self._last_step_time = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self._prev_action_id: int = 0

        # ── Reward-shaping state ────────────────────────────────
        # Frame history for visual change detection
        self._frame_history: deque[np.ndarray] = deque(maxlen=10)
        # Action history for diversity tracking
        self._action_history: deque[int] = deque(maxlen=50)
        # Rolling noop/idle counter — penalise extended inactivity
        self._idle_streak: int = 0
        # Track sustained interaction (holding attack on a block)
        self._interaction_streak: int = 0
        # Baseline frame for longer-horizon exploration detection
        self._baseline_frame: Optional[np.ndarray] = None
        self._baseline_frame_step: int = 0

        # ── Screen-based event detectors ────────────────────────
        self._block_break_tracker = BlockBreakTracker()
        self._hotbar_tracker = HotbarTracker()
        self._crafting_tracker = CraftingTracker()
        self._placement_tracker = PlacementTracker()
        self._building_streak = BuildingStreakTracker()
        self._creative_sequence = CreativeSequenceTracker()
        self._is_dead: bool = False
        self._raw_frame: Optional[np.ndarray] = None  # full-res BGR

        # ── Stagnation penalty ──────────────────────────────────
        # Penalise if no crafting, breaking, or building for ~30 s.
        # Convert 30 real-time seconds into steps using step_delay.
        self._stagnation_timeout = max(1, int(30.0 / max(self._step_delay, 0.01)))
        self._last_productive_step: int = 0

        # ── Long-break system ───────────────────────────────────
        # Some blocks (dirt w/o shovel, stone w/o pick) take up to
        # 12 s of sustained left-click.  If the agent repeatedly
        # issues attack actions without a block_broken event, we
        # override with a single sustained hold.
        self._attack_streak: int = 0       # consecutive attack steps
        self._blocks_broken_total: int = 0 # lifetime counter (mod)
        # Trigger long break after this many consecutive attack steps
        # with no block break.  15 steps × 100 ms ≈ 1.5 s of trying.
        self._long_break_threshold: int = 15
        # How long (seconds) to hold attack during a long break.
        self._long_break_duration: float = 8.0
        # Minimum seconds between long breaks to avoid stalling.
        self._long_break_cooldown: float = 15.0
        self._last_long_break_time: float = 0.0
        self._long_break_count: int = 0    # total long breaks done

        # ── Drop-pickup suppression ─────────────────────────────
        # Suppress item_pickup rewards for 5 s after a drop action
        # to prevent the agent gaming rewards by dropping + picking
        # up the same item repeatedly.
        self._last_drop_time: float = 0.0
        self._drop_pickup_cooldown: float = 5.0

        # ── Position / height tracking (from mod position_update) ──
        # Surface Y in Minecraft 1.21 is ~63 (sea level).  Below ~58
        # the player is likely in a cave or ravine.
        self._player_y: Optional[float] = None      # last known Y
        self._prev_player_y: Optional[float] = None  # Y from previous update
        self._player_on_ground: bool = True
        self._sky_light: int = 15                    # 0=total darkness, 15=full sky
        self._underground_steps: int = 0             # consecutive steps below threshold
        _SURFACE_Y = 58  # below this = underground penalty
        self._surface_y = _SURFACE_Y

        # ── Camera pitch / yaw tracking (from mod position_update) ──
        # Minecraft pitch: -90 = straight up (sky), 0 = horizontal,
        #                  +90 = straight down (feet).
        # We clamp look deltas so the AI cannot reach extreme angles
        # and get stuck staring at the sky (or feet).
        self._player_pitch: Optional[float] = None   # degrees
        self._player_yaw: Optional[float] = None     # degrees
        self._PITCH_LIMIT_UP: float = -55.0          # max upward pitch (degrees)
        self._PITCH_LIMIT_DOWN: float = 70.0         # max downward pitch (degrees)
        # Track how long the camera has been at extreme pitch
        self._extreme_pitch_steps: int = 0

        # ── Hotbar spam tracking ────────────────────────────────
        self._hotbar_streak: int = 0  # consecutive hotbar-only actions

        # ── Home location (set from first position_update) ──────
        # The spawn point is treated as "home base" where the agent
        # stores items and builds.  Actions within 100 blocks of
        # home get a small proximity bonus; actions far away get a
        # gentle distance penalty that grows with distance.
        self._home_x: Optional[float] = None
        self._home_z: Optional[float] = None
        self._player_x: Optional[float] = None
        self._player_z: Optional[float] = None
        self._home_radius: float = 100.0  # blocks — full bonus zone

        # ── Mod bridge (authoritative game events via TCP) ──────
        self._mod_bridge = ModBridge(port=mod_bridge_port)
        self._mod_bridge.start()

        log.info(
            "MinecraftEnv ready: window='%s' mode=%s resolution=%s guard=%s launcher=%s",
            self._window.title, input_mode, resolution,
            "ON" if self._guard else "OFF",
            "ON" if self._launcher else "OFF",
        )

    # ── GameEnvironment interface ───────────────────────────────

    def reset(self) -> Dict[str, torch.Tensor]:
        """Release all inputs and capture the initial frame."""
        self._input.release_all()
        self._step_count = 0
        self._prev_action_id = 0
        self._last_step_time = time.perf_counter()

        # Reset reward-shaping state
        self._frame_history.clear()
        self._action_history.clear()
        self._idle_streak = 0
        self._interaction_streak = 0
        self._baseline_frame = None
        self._baseline_frame_step = 0

        # Reset screen-based detectors
        self._block_break_tracker.reset()
        self._hotbar_tracker.reset()
        self._crafting_tracker.reset()
        self._placement_tracker.reset()
        self._building_streak.reset()
        self._creative_sequence.reset()
        self._is_dead = False
        self._last_productive_step = 0

        # Reset long-break state
        self._attack_streak = 0
        self._blocks_broken_total = 0
        self._last_long_break_time = 0.0
        self._long_break_count = 0
        self._last_drop_time = 0.0

        # Reset position/underground tracking but KEEP home location
        # (home persists across episode resets — it's the world spawn).
        self._player_y = None
        self._prev_player_y = None
        self._player_on_ground = True
        self._sky_light = 15
        self._underground_steps = 0
        self._hotbar_streak = 0
        self._player_pitch = None
        self._player_yaw = None
        self._extreme_pitch_steps = 0

        # Drain any stale mod events from the previous episode
        if self._mod_bridge is not None:
            self._mod_bridge.drain_events()

        # Brief pause to let key releases propagate
        time.sleep(0.05)

        obs = self._observe()
        log.info("Environment reset — episode starts.")
        return obs

    def step(self, action_id: int) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Execute *action_id* and return (obs, reward, done, info).

        1. Translate action_id → key/button/look specification.
        2. Send input to the Minecraft window.
        3. Wait for step delay (pacing).
        4. Capture the next frame.
        5. Compute simple extrinsic reward signals.
        """
        action_id = int(action_id) % NUM_ACTIONS
        action = MINECRAFT_ACTIONS[action_id]

        # ── Long-break check ────────────────────────────────────
        # If the agent has been issuing attack actions repeatedly
        # without any block breaking, override with a sustained
        # hold so blocks that take many seconds actually break.
        is_attack = action_id in _ATTACK_ACTIONS
        long_break_triggered = False

        # Track item drops for pickup-reward suppression
        if action_id in _DROP_ACTIONS:
            self._last_drop_time = time.monotonic()

        if is_attack:
            self._attack_streak += 1
        else:
            self._attack_streak = 0

        now_mono = time.monotonic()
        cooldown_ok = (now_mono - self._last_long_break_time) >= self._long_break_cooldown
        mod_alive = (self._mod_bridge is not None and self._mod_bridge.connected)

        if (is_attack
                and self._attack_streak >= self._long_break_threshold
                and cooldown_ok
                and mod_alive
                and self._window.is_valid):
            long_break_triggered = True
            self._long_break_count += 1
            self._last_long_break_time = now_mono
            log.info(
                "Long-break #%d triggered at step %d "
                "(streak=%d, duration=%.1fs)",
                self._long_break_count, self._step_count,
                self._attack_streak, self._long_break_duration,
            )
            obs = self._execute_long_break(action)
            self._attack_streak = 0  # reset after long break
        else:
            # ── Normal action execution ─────────────────────────
            if self._window.is_valid:
                self._input.set_keys(action.keys)
                self._input.set_buttons(action.buttons)

                if action.look is not None:
                    dx, dy = action.look
                    # ── Pitch clamping ──────────────────────────
                    # Suppress vertical look deltas when the camera
                    # is near the pitch limits to prevent the AI
                    # from getting stuck staring at the sky/feet.
                    # dy < 0 = looking UP (pitch decreasing toward -90)
                    # dy > 0 = looking DOWN (pitch increasing toward +90)
                    if self._player_pitch is not None:
                        if dy < 0 and self._player_pitch <= self._PITCH_LIMIT_UP:
                            # Already at/beyond upward limit — block
                            dy = 0
                        elif dy < 0 and self._player_pitch < self._PITCH_LIMIT_UP + 15:
                            # Approaching upward limit — scale down
                            room = self._player_pitch - self._PITCH_LIMIT_UP
                            scale = max(0.0, room / 15.0)
                            dy = int(dy * scale)
                        elif dy > 0 and self._player_pitch >= self._PITCH_LIMIT_DOWN:
                            # Already at/beyond downward limit — block
                            dy = 0
                        elif dy > 0 and self._player_pitch > self._PITCH_LIMIT_DOWN - 15:
                            # Approaching downward limit — scale down
                            room = self._PITCH_LIMIT_DOWN - self._player_pitch
                            scale = max(0.0, room / 15.0)
                            dy = int(dy * scale)

                    if dx != 0 or dy != 0:
                        self._input.mouse_look(dx, dy)

            # ── Pacing ──────────────────────────────────────────
            elapsed = time.perf_counter() - self._last_step_time
            remaining = self._step_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_step_time = time.perf_counter()

            # ── Observe ─────────────────────────────────────────
            obs = self._observe()

        # ── Track action for reward shaping ─────────────────────
        self._action_history.append(action_id)

        # ── Reward heuristics ───────────────────────────────────
        reward_info = self._compute_reward(obs, action_id)
        reward = reward_info["total"]

        done = not self._window.is_valid
        info = {
            "step": self._step_count,
            "action_name": action.name,
            "has_look": action.look is not None,
            "window_focused": self._window.is_focused,
            "reward_breakdown": reward_info,
            "long_break": long_break_triggered,
            "attack_streak": self._attack_streak,
            "long_break_count": self._long_break_count,
        }

        self._step_count += 1
        self._prev_action_id = action_id

        return obs, reward, done, info

    # ── Long-break helper ───────────────────────────────────────

    def _execute_long_break(self, action) -> Dict[str, torch.Tensor]:
        """
        Hold left-click (attack) for an extended duration to break
        tough blocks that need sustained mining.

        Instead of the normal 100 ms tap, holds the attack button for
        ``_long_break_duration`` seconds (default 8 s).  Polls for
        ``block_broken`` mod events every 0.25 s and aborts early if
        the target block breaks, avoiding wasted time.

        During the hold we also keep any movement/modifier keys the
        original action specified, but suppress look deltas to avoid
        drifting the camera mid-break.

        Returns the latest observation tensor dict.
        """
        # Hold the same keys/buttons as the triggering action
        if self._window.is_valid:
            self._input.set_keys(action.keys)
            self._input.set_buttons(action.buttons | frozenset({"left"}))

        poll_interval = 0.25   # check for break every 250 ms
        start = time.perf_counter()
        end = start + self._long_break_duration
        broke_during_hold = False

        while time.perf_counter() < end:
            time.sleep(poll_interval)
            if not self._window.is_valid:
                break

            # Check mod bridge for block_broken events
            if self._mod_bridge is not None and self._mod_bridge.connected:
                events = self._mod_bridge.drain_events()
                breaks = [e for e in events if e.get("event") == "block_broken"]
                if breaks:
                    broke_during_hold = True
                    elapsed_s = time.perf_counter() - start
                    log.info(
                        "Long-break: block broke after %.1fs (%d events)",
                        elapsed_s, len(breaks),
                    )
                    # Put any non-break events back (pickups, etc.)
                    for e in events:
                        if e.get("event") != "block_broken":
                            self._mod_bridge._events.append(e)
                    break

        # Release the sustained hold and capture final observation
        self._input.release_all()
        self._last_step_time = time.perf_counter()
        obs = self._observe()

        hold_time = time.perf_counter() - start
        if not broke_during_hold:
            log.info("Long-break: no block broke after %.1fs hold", hold_time)

        return obs

    def close(self) -> None:
        """Release all held inputs, stop input guard, and stop Minecraft."""
        self._input.release_all()

        if self._mod_bridge is not None:
            self._mod_bridge.stop()
            self._mod_bridge = None

        if self._guard is not None:
            self._guard.stop()
            self._guard = None

        if self._launcher is not None:
            self._launcher.stop()
            self._launcher = None

        log.info("MinecraftEnv closed after %d steps.", self._step_count)

    @property
    def action_space_size(self) -> int:
        return NUM_ACTIONS

    @property
    def name(self) -> str:
        return "Minecraft"

    def render(self) -> Optional[np.ndarray]:
        """Return the last captured BGR frame for display."""
        return self._last_frame

    def set_home(self) -> None:
        """Set the home location to the player's current position.

        Called from the UI 'Set New Home' button.  Grabs the latest
        known coordinates from the mod bridge position updates and
        makes them the new home base for the proximity reward channel.
        """
        if self._player_x is not None and self._player_z is not None:
            self._home_x = self._player_x
            self._home_z = self._player_z
            log.info("Home location updated: (%.1f, %.1f)",
                     self._home_x, self._home_z)
        else:
            log.warning("Cannot set home — player position not yet known "
                        "(waiting for first position_update from mod).")

    # ── Internals ───────────────────────────────────────────────

    def _observe(self) -> Dict[str, torch.Tensor]:
        """Capture a frame and package it as a model-ready observation dict."""
        raw_bgr, vision_tensor = self._capture.grab_both()
        self._last_frame = raw_bgr
        self._raw_frame = raw_bgr  # native window resolution for analysers

        # Audio placeholder — zero mel spectrogram with correct shape:
        # (B, n_mels, T) where n_mels=64, T≈100 for 1 sec at 16kHz/160 hop
        audio_tensor = torch.zeros(1, 64, 100)

        # Sensor placeholder — could be populated with game-state
        # extracted via OCR (health bar, hunger, position estimates)
        sensor_tensor = torch.zeros(1, self._sensor_channels)

        return {
            "vision": vision_tensor,    # (1, 3, H, W)
            "audio": audio_tensor,      # (1, 1, T)
            "sensor": sensor_tensor,    # (1, S)
        }

    def _compute_reward(self, obs: Dict[str, torch.Tensor], action_id: int) -> Dict[str, float]:
        """
        Compute multi-channel extrinsic reward from frame analysis and action.

        Returns a dict with per-channel values AND a "total" key with
        the combined extrinsic signal.  The intrinsic (ICM) reward is
        computed externally — this only handles environment-side shaping.

        Channels (* = new creation-focused channels)
        --------
        survival        : tiny constant bonus for staying alive
        visual_change   : reward for frames that look meaningfully different
        action_diversity: reward for trying many different actions recently
        interaction     : reward when attack/use is held while visual
                          change is detected (= block broken / placed)
        exploration     : reward when the scene looks very different from
                          a baseline captured N steps ago
        idle_penalty    : negative reward for noop / standing still
        block_break     : crosshair crack detection (mining)
        item_pickup     : hotbar change detection (collecting items)
        death_penalty   : penalty for dying
        movement        : reward for locomotion with visual confirmation
        * block_place     : right-click that changes forward view (building)
        * crafting         : crafting UI interaction + hotbar change
        * building_streak  : bonus for consecutive placements (structures)
        * creative_sequence: bonus for gather->craft->build pipeline
        """
        rewards: Dict[str, float] = {}

        # ────────────────────────────────────────────────────────
        # 1. Survival bonus (very small, prevents total-zero signal)
        # ────────────────────────────────────────────────────────
        rewards["survival"] = 0.005

        # ────────────────────────────────────────────────────────
        # 2. Visual change — compare current frame to previous
        # ────────────────────────────────────────────────────────
        vision_tensor = obs["vision"]  # (1, 3, H, W)
        frame_np = vision_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        frame_u8 = (frame_np * 255).astype(np.uint8) if frame_np.max() <= 1.0 else frame_np.astype(np.uint8)

        visual_change = 0.0
        if self._frame_history:
            prev = self._frame_history[-1]
            diff = np.abs(frame_u8.astype(np.float32) - prev.astype(np.float32)).mean() / 255.0
            if diff > 0.02:
                visual_change = min(diff * 5.0, 1.0)
        rewards["visual_change"] = visual_change

        self._frame_history.append(frame_u8)

        # ────────────────────────────────────────────────────────
        # 3. Action diversity — entropy-like bonus over recent actions
        # ────────────────────────────────────────────────────────
        action_div = 0.0
        if len(self._action_history) >= 10:
            recent = list(self._action_history)[-30:]
            unique_ratio = len(set(recent)) / max(len(recent), 1)
            action_div = unique_ratio * 0.3
        rewards["action_diversity"] = action_div

        # ────────────────────────────────────────────────────────
        # 4. Interaction bonus — attack/use that causes visual change
        #    Only rewards MEANINGFUL interaction: requires significant
        #    visual change to prove the button press did something
        #    real in the world (not just clicking in empty air).
        #    Sub-weights: int_impact (base impact reward),
        #                 int_sustained (sustained mining reward).
        # ────────────────────────────────────────────────────────
        # Read sub-weights (gracefully falls back to defaults).
        _sw = _reward_weights.snapshot() if _reward_weights is not None else {}
        _int_impact = _sw.get("int_impact", 0.5)
        _int_sustained = _sw.get("int_sustained", 0.2)

        interaction_bonus = 0.0
        if action_id in _BLOCK_INTERACTION_ACTIONS:
            self._interaction_streak += 1
            if visual_change > 0.06:
                # Large visual change while interacting — real impact
                interaction_bonus = _int_impact + min(visual_change * 2.0, 0.5)
            elif self._interaction_streak >= 5 and visual_change > 0.03:
                # Sustained interaction with moderate change (mining)
                interaction_bonus = _int_sustained
            # Single clicks with tiny/no visual change = no reward
        else:
            self._interaction_streak = 0
        rewards["interaction"] = interaction_bonus

        # ────────────────────────────────────────────────────────
        # 5. Exploration — long-horizon scene difference
        # ────────────────────────────────────────────────────────
        exploration_bonus = 0.0
        baseline_interval = 100
        if self._baseline_frame is None:
            self._baseline_frame = frame_u8.copy()
            self._baseline_frame_step = self._step_count
        elif self._step_count - self._baseline_frame_step >= baseline_interval:
            base_diff = np.abs(
                frame_u8.astype(np.float32) - self._baseline_frame.astype(np.float32)
            ).mean() / 255.0
            if base_diff > 0.08:
                exploration_bonus = min(base_diff * 3.0, 1.0)
            self._baseline_frame = frame_u8.copy()
            self._baseline_frame_step = self._step_count
        rewards["exploration"] = exploration_bonus

        # ────────────────────────────────────────────────────────
        # 6. Movement bonus — reward confirmed movement with a
        #    bias towards forward locomotion and camera rotation
        #    over aimless strafing.
        #    Sub-weights (internal multipliers, NOT outer channel weight):
        #      mv_forward  — multiplier on forward (W) base_move
        #      mv_backward — multiplier on backward (S) base_move
        #      mv_strafe   — multiplier on strafe (A/D) base_move
        #      mv_look     — flat bonus for camera rotations
        #      mv_jump     — extra multiplier when jumping
        #      mv_sprint   — extra multiplier when sprinting
        # ────────────────────────────────────────────────────────
        _mv_forward  = _sw.get("mv_forward", 3.0)
        _mv_backward = _sw.get("mv_backward", 1.0)
        _mv_strafe   = _sw.get("mv_strafe", 0.4)
        _mv_look     = _sw.get("mv_look", 0.02)
        _mv_jump     = _sw.get("mv_jump", 1.5)
        _mv_sprint   = _sw.get("mv_sprint", 1.5)

        movement_bonus = 0.0
        if action_id in _MOVEMENT_ACTIONS and visual_change > 0.02:
            base_move = visual_change * 0.3
            if action_id in _FORWARD_ACTIONS:
                base_move *= _mv_forward
            elif action_id in _BACKWARD_ACTIONS:
                base_move *= _mv_backward
            elif action_id in _PURE_STRAFE_ACTIONS:
                base_move *= _mv_strafe
            # Jump bonus — stacks on top of directional multiplier
            if action_id in _JUMP_ACTIONS:
                base_move *= _mv_jump
            # Sprint bonus — stacks on top of directional multiplier
            if action_id in _SPRINT_ACTIONS:
                base_move *= _mv_sprint
            movement_bonus = base_move
        # Camera look (even while standing still) gets a small bonus
        # to encourage situational awareness / scanning the environment.
        if action_id in _LOOK_ACTIONS and action_id not in _PURE_STRAFE_ACTIONS:
            movement_bonus += _mv_look
        rewards["movement"] = movement_bonus

        # ────────────────────────────────────────────────────────
        # 7. Idle penalty — discourage noop / repeated same action
        # ────────────────────────────────────────────────────────
        idle_penalty = 0.0
        if action_id == 0:
            self._idle_streak += 1
            if self._idle_streak > 3:
                # Escalates faster: 0.08/step, caps at 0.8
                idle_penalty = min(0.08 * (self._idle_streak - 3), 0.8)
        else:
            self._idle_streak = max(0, self._idle_streak - 2)

        if len(self._action_history) >= 10:
            last_10 = list(self._action_history)[-10:]
            if len(set(last_10)) == 1:
                idle_penalty += 0.15
        rewards["idle_penalty"] = idle_penalty

        # ────────────────────────────────────────────────────────
        # 7b. Hotbar spam penalty — pressing number keys repeatedly
        #     without interacting with items is pointless.
        # ────────────────────────────────────────────────────────
        hotbar_spam_penalty = 0.0
        if action_id in _HOTBAR_ACTIONS:
            self._hotbar_streak += 1
            # Escalating penalty: first hotbar press in a while is
            # fine (switching tools), but repeated presses trigger.
            if self._hotbar_streak >= 2:
                hotbar_spam_penalty = min(0.1 * self._hotbar_streak, 0.6)
        else:
            # Decay streak when doing something else
            self._hotbar_streak = max(0, self._hotbar_streak - 1)

        # Also check the recent history — more than 3 hotbar
        # actions out of the last 10 is excessive switching.
        if len(self._action_history) >= 10:
            last_10 = list(self._action_history)[-10:]
            hotbar_count = sum(1 for a in last_10 if a in _HOTBAR_ACTIONS)
            if hotbar_count > 3:
                hotbar_spam_penalty += 0.05 * (hotbar_count - 3)
        rewards["hotbar_spam_penalty"] = hotbar_spam_penalty

        # ────────────────────────────────────────────────────────
        # Drain authoritative game events from the Fabric mod bridge.
        # When connected, these override the pixel-based detectors
        # for block break, item pickup, block place, crafting, and
        # death — giving accurate, no-false-positive signals.
        # ────────────────────────────────────────────────────────
        mod_events = (
            self._mod_bridge.drain_events()
            if self._mod_bridge is not None and self._mod_bridge.connected
            else []
        )
        mod_blocks_broken = [e for e in mod_events if e.get("event") == "block_broken"]
        mod_blocks_placed = [e for e in mod_events if e.get("event") == "block_placed"]
        mod_items_crafted = [e for e in mod_events if e.get("event") == "item_crafted"]
        mod_items_picked  = [e for e in mod_events if e.get("event") == "item_picked_up"]
        mod_deaths        = [e for e in mod_events if e.get("event") == "player_death"]
        mod_health        = [e for e in mod_events if e.get("event") == "health_changed"]
        mod_food          = [e for e in mod_events if e.get("event") == "food_changed"]
        mod_xp            = [e for e in mod_events if e.get("event") == "xp_gained"]
        mod_position      = [e for e in mod_events if e.get("event") == "position_update"]

        # ── Update player position from latest position_update ──
        if mod_position:
            latest_pos = mod_position[-1]  # use most recent
            self._prev_player_y = self._player_y
            self._player_y = latest_pos.get("y", self._player_y)
            self._player_x = latest_pos.get("x", self._player_x)
            self._player_z = latest_pos.get("z", self._player_z)
            self._player_on_ground = latest_pos.get("on_ground", True)
            self._sky_light = latest_pos.get("light", 15)
            self._player_pitch = latest_pos.get("pitch", self._player_pitch)
            self._player_yaw = latest_pos.get("yaw", self._player_yaw)

            # First position_update ever → set as home location
            if self._home_x is None and self._player_x is not None:
                self._home_x = self._player_x
                self._home_z = self._player_z
                log.info("Home location set: (%.1f, %.1f)",
                         self._home_x, self._home_z)

        # Reset attack streak when a block actually breaks —
        # prevents unnecessary long-break triggers.
        if mod_blocks_broken:
            self._attack_streak = 0
            self._blocks_broken_total += len(mod_blocks_broken)

        use_mod = bool(mod_events) or (
            self._mod_bridge is not None and self._mod_bridge.connected
        )

        if mod_events and self._step_count % 50 == 0:
            log.info("Mod events step %d: %s",
                     self._step_count,
                     [e["event"] for e in mod_events])

        # ────────────────────────────────────────────────────────
        # 8. Block-break detection
        #    MOD: per-item reward via item_rewards table
        #    PIXEL FALLBACK: crosshair crack animation (flat 1.0)
        # ────────────────────────────────────────────────────────
        if use_mod:
            # Sum per-block rewards based on rarity / difficulty.
            # Mod only fires block_broken on COMPLETE breaks —
            # no partial-damage reward is possible here.
            block_break_reward = sum(
                get_item_reward(e.get("block", ""), "block_broken")
                for e in mod_blocks_broken
            )
            # Still update pixel tracker state so it stays in sync
            if self._raw_frame is not None:
                is_attacking = action_id in _ATTACK_ACTIONS
                self._block_break_tracker.update(
                    self._raw_frame, is_attacking=is_attacking,
                )
            block_break_score = block_break_reward
        else:
            block_break_score = 0.0
            if self._raw_frame is not None:
                is_attacking = action_id in _ATTACK_ACTIONS
                block_break_score = self._block_break_tracker.update(
                    self._raw_frame, is_attacking=is_attacking,
                )
            # Only reward near-complete breaks (score ≥ 0.85).
            # This prevents partial damage from generating reward —
            # the agent must fully break the block to get credit.
            block_break_reward = block_break_score if block_break_score > 0.85 else 0.0
        rewards["block_break"] = block_break_reward

        # ────────────────────────────────────────────────────────
        # 9. Crafting detection
        #    MOD: per-item reward via item_rewards table (2× base)
        #    PIXEL FALLBACK: inventory UI + hotbar change
        # ────────────────────────────────────────────────────────
        craft_info = {"ui_open": False, "craft_score": 0.0, "frames_in_ui": 0}
        if self._raw_frame is not None:
            craft_info = self._crafting_tracker.update(self._raw_frame)

        if use_mod:
            # Per-item crafting reward scaled by rarity (2× multiplier)
            craft_score_mod = sum(
                get_item_reward(e.get("item", ""), "item_crafted")
                * e.get("count", 1)
                for e in mod_items_crafted
            )
            # Override pixel tracker score with mod score
            craft_info = dict(craft_info)  # copy
            craft_info["craft_score"] = craft_score_mod

        # ────────────────────────────────────────────────────────
        # 10. Item pickup
        #    MOD: per-item reward via item_rewards table
        #    PIXEL FALLBACK: hotbar region change (suppressed during UI)
        # ────────────────────────────────────────────────────────
        if use_mod:
            # Per-item pickup reward scaled by rarity
            item_pickup = sum(
                get_item_reward(e.get("item", ""), "item_picked_up")
                * min(e.get("count", 1), 10)  # cap count to curb inflation
                for e in mod_items_picked
            )
            # Keep pixel tracker in sync
            if self._raw_frame is not None:
                self._hotbar_tracker.update(self._raw_frame)
        else:
            item_pickup = 0.0
            if self._raw_frame is not None and not craft_info["ui_open"]:
                item_pickup = self._hotbar_tracker.update(self._raw_frame)
            elif self._raw_frame is not None:
                self._hotbar_tracker.update(self._raw_frame)

        # Suppress pickup reward shortly after a drop action —
        # prevents the agent from gaming rewards by dropping items
        # and immediately picking them back up.
        if item_pickup > 0 and (time.monotonic() - self._last_drop_time) < self._drop_pickup_cooldown:
            log.debug("Suppressed item_pickup reward (%.3f) — within %.0fs of drop",
                      item_pickup, self._drop_pickup_cooldown)
            item_pickup = 0.0

        rewards["item_pickup"] = item_pickup

        # ── Item drop penalty ───────────────────────────────────
        # Dropping items (Q key) is almost never beneficial early
        # on.  Apply a penalty proportional to the item's value
        # to discourage carelessly discarding inventory.
        item_drop_penalty = 0.0
        if action_id in _DROP_ACTIONS:
            # Flat penalty per drop action — the agent shouldn't
            # be pressing Q unless there's a very good reason.
            item_drop_penalty = 0.3
        rewards["item_drop_penalty"] = item_drop_penalty

        # ────────────────────────────────────────────────────────
        # 11. Death penalty
        #    MOD: 1.0 per death event
        #    PIXEL FALLBACK: red "You Died!" overlay
        # ────────────────────────────────────────────────────────
        if use_mod:
            death_penalty = float(len(mod_deaths))
            if death_penalty > 0:
                log.info("Death detected via mod at step %d", self._step_count)
            # Keep pixel tracker in sync
            if self._raw_frame is not None:
                self._is_dead = detect_death(self._raw_frame)
        else:
            death_penalty = 0.0
            if self._raw_frame is not None:
                died_now = detect_death(self._raw_frame)
                if died_now and not self._is_dead:
                    death_penalty = 1.0
                    log.info("Death detected at step %d", self._step_count)
                self._is_dead = died_now
        rewards["death_penalty"] = death_penalty

        # ────────────────────────────────────────────────────────
        # 11b. Damage / Healing (from health_changed events)
        #    Damage taken → penalty proportional to hearts lost
        #    Healing      → small reward proportional to hearts gained
        # ────────────────────────────────────────────────────────
        damage_taken = 0.0
        healing = 0.0
        if use_mod:
            for evt in mod_health:
                delta = evt.get("delta", 0.0)
                if delta < 0:
                    # delta is negative when damaged; store as positive penalty
                    damage_taken += abs(delta)
                elif delta > 0:
                    healing += delta
        rewards["damage_taken"] = damage_taken
        rewards["healing"] = healing

        # ────────────────────────────────────────────────────────
        # 11c. Food / Hunger reward
        #    Positive food delta = filling hunger bar → reward
        #    Losing hunger is not penalised (natural drain)
        # ────────────────────────────────────────────────────────
        food_reward = 0.0
        if use_mod:
            for evt in mod_food:
                delta = evt.get("delta", 0)
                if delta > 0:
                    food_reward += float(delta)
        rewards["food_reward"] = food_reward

        # ────────────────────────────────────────────────────────
        # 11d. Experience points reward
        #    Small reward for each XP point gained
        # ────────────────────────────────────────────────────────
        xp_reward = 0.0
        if use_mod:
            for evt in mod_xp:
                xp_reward += float(evt.get("amount", 0))
        rewards["xp_reward"] = xp_reward

        # ────────────────────────────────────────────────────────
        # 11e. Height / cave / fall penalties
        #    Uses position_update events from the mod to detect:
        #    - Being underground (Y below ~58, sea level is ~63)
        #    - Falling (rapid Y decrease between updates)
        #    - Darkness (low sky light = cave/underground)
        #    Combined into a single "height_penalty" channel.
        #    Sub-weights scale individual components:
        #      height_underground, height_fall, height_darkness.
        # ────────────────────────────────────────────────────────
        _h_underground = _sw.get("height_underground", 1.0)
        _h_fall        = _sw.get("height_fall", 1.0)
        _h_darkness    = _sw.get("height_darkness", 1.0)

        height_penalty = 0.0
        if self._player_y is not None:
            # Underground penalty — escalates the longer the AI
            # stays below the surface threshold.
            if self._player_y < self._surface_y:
                self._underground_steps += 1
                # Depth-proportional: deeper = bigger penalty
                depth = max(0.0, self._surface_y - self._player_y)
                # Base 0.03 per step underground, scaled by depth
                underground_base = min(0.03 + 0.005 * depth, 0.3)
                # Extra escalation for prolonged underground time
                # (kicks in after ~50 steps = ~5 seconds)
                if self._underground_steps > 50:
                    overshoot = (self._underground_steps - 50) / 100.0
                    underground_base += min(0.1 * overshoot, 0.4)
                height_penalty += underground_base * _h_underground
            else:
                self._underground_steps = max(0, self._underground_steps - 5)

            # Fall penalty — rapid Y decrease means falling into a
            # hole/cave.  Only triggers for drops > 3 blocks.
            if self._prev_player_y is not None:
                y_drop = self._prev_player_y - self._player_y
                if y_drop > 3.0:
                    # Proportional to fall distance
                    height_penalty += min(0.15 * y_drop, 1.0) * _h_fall
                    log.debug("Fall detected: %.1f blocks (Y %.1f → %.1f)",
                              y_drop, self._prev_player_y, self._player_y)

            # Darkness penalty — low sky light means the AI is in
            # a covered/underground area (cave, ravine, dense canopy).
            if self._sky_light <= 4:
                height_penalty += 0.05 * _h_darkness

        rewards["height_penalty"] = height_penalty

        # ────────────────────────────────────────────────────────
        # 11e-2. Extreme pitch (sky-staring / feet-staring) penalty
        #    When the camera pitch is far from horizontal the AI
        #    sees mostly sky or ground — no useful visual data.
        #    Escalating penalty discourages lingering at extremes.
        #
        #    Pitch convention: -90 = sky, 0 = horizontal, +90 = feet
        #    "soft zone" starts at ±40° with a gentle nudge;
        #    "hard zone" beyond ±60° is aggressively penalised.
        # ────────────────────────────────────────────────────────
        pitch_penalty = 0.0
        if self._player_pitch is not None:
            abs_pitch = abs(self._player_pitch)

            if abs_pitch > 60:
                # Hard zone — strong penalty, escalates with duration
                self._extreme_pitch_steps += 1
                pitch_penalty = 0.15 + 0.02 * min(self._extreme_pitch_steps, 30)
            elif abs_pitch > 40:
                # Soft zone — gentle nudge back toward horizontal
                self._extreme_pitch_steps += 1
                overshoot = (abs_pitch - 40) / 20.0  # 0.0 at 40°, 1.0 at 60°
                pitch_penalty = 0.03 * overshoot
            else:
                # Healthy range — decay the streak counter
                self._extreme_pitch_steps = max(0, self._extreme_pitch_steps - 3)

        rewards["pitch_penalty"] = pitch_penalty

        # ────────────────────────────────────────────────────────
        # 11f. Home proximity bonus / distance penalty
        #    Spawn point = home base.  Staying within 100 blocks
        #    gives a small bonus (all productive actions there are
        #    implicitly more valuable).  Beyond 100 blocks, a
        #    gentle penalty grows to discourage aimless wandering
        #    far from where the AI keeps its items/buildings.
        # ────────────────────────────────────────────────────────
        home_bonus = 0.0
        if (self._home_x is not None
                and self._player_x is not None):
            dx = self._player_x - self._home_x
            dz = self._player_z - self._home_z
            dist = (dx * dx + dz * dz) ** 0.5

            if dist <= self._home_radius:
                # Inside home zone — small flat bonus that makes
                # every other reward channel slightly more valuable
                home_bonus = 0.02
            else:
                # Outside home zone — gentle escalating penalty
                overshoot = (dist - self._home_radius) / self._home_radius
                home_bonus = -min(0.03 * overshoot, 0.5)
        rewards["home_proximity"] = home_bonus

        # ============================================
        # * CREATION-FOCUSED REWARD CHANNELS
        # ============================================

        # ────────────────────────────────────────────────────────
        # 12. Block placement detection
        #    MOD: per-block reward via item_rewards table (1.2× base)
        #    PIXEL FALLBACK: right-click + forward visual change
        # ────────────────────────────────────────────────────────
        if use_mod:
            # Per-block placement reward — incentivises building with
            # rarer materials (1.2× multiplier for intentional building)
            block_place_score = sum(
                get_item_reward(e.get("block", ""), "block_placed")
                for e in mod_blocks_placed
            )
            # Keep pixel tracker in sync
            if self._raw_frame is not None:
                is_using = action_id in _USE_ACTIONS
                self._placement_tracker.update(
                    self._raw_frame, is_using=is_using,
                )
        else:
            block_place_score = 0.0
            if self._raw_frame is not None:
                is_using = action_id in _USE_ACTIONS
                block_place_score = self._placement_tracker.update(
                    self._raw_frame, is_using=is_using,
                )
        rewards["block_place"] = block_place_score

        # ────────────────────────────────────────────────────────
        # 13. Crafting reward (craft_info from section 9)
        # ────────────────────────────────────────────────────────
        rewards["crafting"] = craft_info["craft_score"]

        # Small bonus for engaging with the crafting UI long enough
        # to actually craft — only when pixel fallback is in use
        # and the tracker hasn't already fired a real craft_score.
        if (not use_mod
                and craft_info["ui_open"]
                and craft_info["frames_in_ui"] > 15
                and craft_info["craft_score"] == 0.0):
            rewards["crafting"] += 0.02

        # ────────────────────────────────────────────────────────
        # 14. Building streak — consecutive placements build
        #     structures; super-linear reward scaling.
        # ────────────────────────────────────────────────────────
        placed_this_step = block_place_score > 0.5
        streak_info = self._building_streak.update(placed_this_step)
        rewards["building_streak"] = streak_info["streak_bonus"]

        # ────────────────────────────────────────────────────────
        # 15. Creative sequence — gather → craft → build pipeline
        #     Large bonus for completing creative workflow cycles.
        # ────────────────────────────────────────────────────────
        seq_info = self._creative_sequence.update(
            block_break=block_break_reward,
            item_pickup=item_pickup,
            craft_score=craft_info["craft_score"],
            block_place=block_place_score,
        )
        rewards["creative_sequence"] = seq_info["stage_reward"] + seq_info["cycle_bonus"]

        # ────────────────────────────────────────────────────────
        # 16. Stagnation penalty — no crafting, breaking, or
        #     building for ~30 seconds triggers an escalating
        #     penalty to push the agent out of idle wandering.
        # ────────────────────────────────────────────────────────
        productive = (
            block_break_reward > 0
            or craft_info["craft_score"] > 0
            or block_place_score > 0.3
            or item_pickup > 0
        )
        if productive:
            self._last_productive_step = self._step_count

        steps_since_productive = self._step_count - self._last_productive_step
        stagnation_penalty = 0.0
        if steps_since_productive > self._stagnation_timeout:
            # How many multiples of the timeout we've exceeded
            overshoot = (steps_since_productive - self._stagnation_timeout) / self._stagnation_timeout
            # Escalates from 0.15 up to a cap of 1.5 — the agent
            # should feel *real* pressure to do something useful.
            stagnation_penalty = min(0.15 + 0.25 * overshoot, 1.5)
        rewards["stagnation_penalty"] = stagnation_penalty

        # ────────────────────────────────────────────────────────
        # Diagnostic logging — raw detector scores every 50 steps
        # so we can verify detectors are actually producing signal.
        # ────────────────────────────────────────────────────────
        if self._step_count % 50 == 0:
            raw_h, raw_w = (self._raw_frame.shape[:2] if self._raw_frame is not None else (0, 0))
            if self._mod_bridge and self._mod_bridge.connected:
                hb = "alive" if self._mod_bridge.pipeline_alive else "NO-HB"
                bridge_status = f"connected({hb},tick={self._mod_bridge.last_heartbeat_tick},evt={self._mod_bridge.total_events_received})"
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
                self._step_count, bridge_status, raw_h, raw_w,
                block_break_reward, item_pickup, block_place_score,
                craft_info["craft_score"], death_penalty,
                visual_change, interaction_bonus,
                self._long_break_count, self._attack_streak,
                self._player_y if self._player_y is not None else -1.0,
                self._player_pitch if self._player_pitch is not None else 0.0,
                self._sky_light,
                self._underground_steps,
                ((((self._player_x or 0) - (self._home_x or 0)) ** 2
                  + ((self._player_z or 0) - (self._home_z or 0)) ** 2) ** 0.5)
                if self._home_x is not None else -1.0,
            )

        # ────────────────────────────────────────────────────────
        # Combine — weights are read from the UI-editable
        # RewardWeightsState when available; otherwise use defaults.
        # ────────────────────────────────────────────────────────
        if _reward_weights is not None:
            w = _reward_weights.snapshot()
        else:
            # Hardcoded fallback (matches RewardWeightsState defaults)
            w = {
                "survival": 1.0, "visual_change": 0.2,
                "action_diversity": 0.5, "interaction": 0.8,
                "exploration": 0.8, "movement": 0.3,
                "block_break": 4.0, "item_pickup": 6.0,
                "block_place": 4.0, "crafting": 25.0,
                "building_streak": 3.0, "creative_sequence": 6.0,
                "idle_penalty": 2.0, "death_penalty": 10.0,
                "stagnation_penalty": 3.0, "item_drop_penalty": 3.0,
                "damage_taken": 1.5, "hotbar_spam_penalty": 2.0,
                "height_penalty": 2.5, "pitch_penalty": 3.0,
                "healing": 1.0, "food_reward": 0.8,
                "xp_reward": 0.1, "home_proximity": 1.5,
                # Sub-weights (internal multipliers)
                "mv_forward": 3.0, "mv_backward": 1.0,
                "mv_strafe": 0.4, "mv_look": 0.02,
                "mv_jump": 1.5, "mv_sprint": 1.5,
                "int_impact": 0.5, "int_sustained": 0.2,
                "height_underground": 1.0, "height_fall": 1.0,
                "height_darkness": 1.0,
            }

        total = (
            # Baseline
            rewards["survival"]          * w.get("survival", 1.0)
            + rewards["visual_change"]   * w.get("visual_change", 0.2)
            # Exploration
            + rewards["action_diversity"]* w.get("action_diversity", 0.5)
            + rewards["interaction"]     * w.get("interaction", 0.8)
            + rewards["exploration"]     * w.get("exploration", 0.8)
            + rewards["movement"]        * w.get("movement", 0.3)
            # Resource gathering
            + rewards["block_break"]     * w.get("block_break", 4.0)
            + rewards["item_pickup"]     * w.get("item_pickup", 6.0)
            # Creation
            + rewards["block_place"]     * w.get("block_place", 4.0)
            + rewards["crafting"]        * w.get("crafting", 25.0)
            + rewards["building_streak"] * w.get("building_streak", 3.0)
            + rewards["creative_sequence"]* w.get("creative_sequence", 6.0)
            # Penalties (subtracted)
            - rewards["idle_penalty"]    * w.get("idle_penalty", 2.0)
            - rewards["death_penalty"]   * w.get("death_penalty", 10.0)
            - rewards["stagnation_penalty"]* w.get("stagnation_penalty", 3.0)
            - rewards["item_drop_penalty"]* w.get("item_drop_penalty", 3.0)
            - rewards["damage_taken"]    * w.get("damage_taken", 1.5)
            - rewards["hotbar_spam_penalty"]* w.get("hotbar_spam_penalty", 2.0)
            - rewards["height_penalty"]  * w.get("height_penalty", 2.5)
            - rewards["pitch_penalty"]   * w.get("pitch_penalty", 3.0)
            # Sustain
            + rewards["healing"]         * w.get("healing", 1.0)
            + rewards["food_reward"]     * w.get("food_reward", 0.8)
            + rewards["xp_reward"]       * w.get("xp_reward", 0.1)
            # Home proximity (can be positive or negative)
            + rewards["home_proximity"]  * w.get("home_proximity", 1.5)
        )
        rewards["total"] = total

        return rewards

    # ── Diagnostics ─────────────────────────────────────────────

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "step_count": self._step_count,
            "window_valid": self._window.is_valid,
            "window_focused": self._window.is_focused,
            "input_mode": self._input.mode,
            "held_keys": len(self._input.held_keys),
            "held_buttons": len(self._input.held_buttons),
        }
