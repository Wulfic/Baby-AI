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
from baby_ai.environments.minecraft.action_categories import (
    ATTACK_ACTIONS,
    DROP_ACTIONS,
    LOOK_ACTIONS,
)
from baby_ai.environments.minecraft.actions import (
    MINECRAFT_ACTIONS,
    NUM_ACTIONS,
    has_look,
)
from baby_ai.environments.minecraft.action_decoder import (
    ContinuousActionDecoder,
    decode_continuous_action,
)
from baby_ai.environments.minecraft.capture import ScreenCapture
from baby_ai.environments.minecraft.input_guard import InputGuard
from baby_ai.environments.minecraft.input_controller import InputController
from baby_ai.environments.minecraft.launcher import MinecraftLauncher
from baby_ai.environments.minecraft.reward_computer import RewardComputer
from baby_ai.environments.minecraft.screen_analyzer import (
    BuildingStreakTracker,
    CreativeSequenceTracker,
)
from baby_ai.environments.minecraft.mod_bridge import ModBridge
from baby_ai.environments.minecraft.window import WindowManager
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
        # Reward computation is delegated to RewardComputer which
        # owns frame history, action history, idle/interaction streaks.
        self._reward_computer = RewardComputer(self)
        # Action history reference — also used by step() for diversity
        self._action_history = self._reward_computer.action_history

        # ── Event trackers ───────────────────────────────────────
        # BuildingStreakTracker and CreativeSequenceTracker are pure
        # counters (no pixel analysis) — keep them.
        # Old pixel-based trackers are removed; all detection now
        # comes exclusively from the Fabric mod bridge.
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
        self._prev_player_x: Optional[float] = None  # X from previous step
        self._prev_player_z: Optional[float] = None  # Z from previous step
        self._player_on_ground: bool = True
        self._sky_light: int = 15                    # 0=total darkness, 15=full sky
        self._underground_steps: int = 0             # consecutive steps below threshold
        # Dynamic surface threshold — set from the player's starting Y
        # on first position_update.  Avoids false "underground" on flat
        # worlds (Y≈57) where the old hard-coded 58 was 1 block too high.
        self._surface_y: float = 50.0          # safe default until calibrated
        self._surface_y_calibrated: bool = False

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
        # Previous yaw/pitch for computing per-step deltas (imitation learning)
        self._prev_yaw: Optional[float] = None
        self._prev_pitch: Optional[float] = None

        # ── Hotbar spam tracking ────────────────────────────────
        self._hotbar_streak: int = 0  # consecutive hotbar-only actions

        # ── Home location (set from first position_update) ──────
        # The spawn point is treated as "home base" where the agent
        # stores items and builds.  Actions within 100 blocks of
        # home get a small proximity bonus; actions far away get a
        # gentle distance penalty that grows with distance.
        self._home_x: Optional[float] = None
        self._home_y: Optional[float] = None
        self._home_z: Optional[float] = None
        self._player_x: Optional[float] = None
        self._player_z: Optional[float] = None
        self._home_radius: float = 100.0  # blocks — full bonus zone

        # ── Settings store (injected via set_settings_store) ─────
        self._settings_store: Optional[Any] = None

        # ── Home-change callback (set via set_on_home_changed) ───
        # Called whenever home location changes so the GUI can update.
        self._on_home_changed: Optional[Any] = None

        # ── Chunk-based exploration tracking ─────────────────────
        # A Minecraft chunk is 16×16 blocks.  We track:
        #   - visited_chunks: set of (cx, cz) entered this episode
        #   - current chunk coords for detecting transitions
        #   - steps_in_chunk: how long we’ve been in the current chunk
        # Movement reward decays the longer we stay in one chunk.
        self._visited_chunks: set[tuple[int, int]] = set()
        self._current_chunk: Optional[tuple[int, int]] = None
        self._steps_in_chunk: int = 0

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

        # ── Respawn safety net ───────────────────────────────
        # The mod handles respawn automatically (server + client).
        # Just wait briefly for chunks to load if still flagged dead.
        if self._is_dead:
            log.info("Player flagged dead at reset — waiting for mod auto-respawn.")
            time.sleep(1.5)

        self._step_count = 0
        self._prev_action_id = 0
        self._last_step_time = time.perf_counter()

        # Reset reward-shaping state
        self._reward_computer.reset()

        # Reset event trackers
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
        self._prev_player_x = None
        self._prev_player_z = None
        self._player_on_ground = True
        self._sky_light = 15
        self._underground_steps = 0
        self._surface_y_calibrated = False
        self._player_pitch = None
        self._player_yaw = None
        self._extreme_pitch_steps = 0
        self._prev_yaw = None
        self._prev_pitch = None

        # Reset chunk tracking (new episode = fresh exploration)
        self._visited_chunks.clear()
        self._current_chunk = None
        self._steps_in_chunk = 0

        # Drain any stale mod events from the previous episode
        if self._mod_bridge is not None:
            self._mod_bridge.drain_events()

        # Brief pause to let key releases propagate
        time.sleep(0.05)

        obs = self._observe()
        log.info("Environment reset — episode starts.")
        return obs

    def step(self, action_id: torch.Tensor, observation_only: bool = False) -> Tuple[Dict[str, torch.Tensor], float, bool, Dict[str, Any]]:
        """
        Execute an action and return (obs, reward, done, info).

        Args:
            action_id: (20,) continuous action vector from the diffusion policy.
            observation_only: If True, skip AI input sending (imitation mode).

        1. Decode continuous vector → key/button/look specification.
        2. Send input to the Minecraft window.
        3. Wait for step delay (pacing).
        4. Capture the next frame.
        5. Compute simple extrinsic reward signals.

        When *observation_only* is ``True`` (imitation learning mode):
        the AI does **not** send any inputs, action-based accumulators
        and penalties are frozen, but frame capture and mod-event
        rewards (block break, crafting, etc.) still fire normally.
        """
        # ── Auto-respawn handled by mod ────────────────────────
        if self._is_dead:
            obs = self._observe()
            from baby_ai.environments.minecraft.screen_analyzer import detect_death
            frame = self._capture.grab_raw()
            if frame is not None and not detect_death(frame):
                log.info("Death screen dismissed by mod — resuming.")
                self._is_dead = False
            return obs, -5.0, False, {
                "step": self._step_count,
                "action_name": "noop (dead)",
                "has_look": False,
                "window_focused": self._window.is_focused,
                "reward_breakdown": {
                    "total": -5.0, "death_penalty": 1.0,
                    "movement": 0.0, "new_chunk": 0.0,
                },
                "long_break": False,
                "attack_streak": 0,
                "long_break_count": self._long_break_count,
            }

        # ── Decode continuous action vector ──────────────────────
        decoded = decode_continuous_action(action_id)
        action_keys = decoded["keys"]
        action_buttons = decoded["buttons"]
        action_look = decoded["look"]
        action_name_str = decoded["action_name"]
        is_attack = decoded["is_attack"]
        is_drop = False  # not in the 20-dim layout
        disc_action_id = decoded["approx_action_id"]

        # ── Observation-only mode (imitation learning) ──────────
        # Skip all AI input sending and action-based tracking.
        # Still capture frames, process mod events, and compute
        # non-action-based rewards.
        if observation_only:
            # Pacing
            elapsed = time.perf_counter() - self._last_step_time
            remaining = self._step_delay - elapsed
            if remaining > 0:
                time.sleep(remaining)
            self._last_step_time = time.perf_counter()

            obs = self._observe()

            # Reward computation — action-based channels skipped
            reward_info = self._reward_computer.compute(
                obs, disc_action_id, _reward_weights, self._step_count,
                observation_only=True,
            )
            reward = reward_info["total"]

            done = not self._window.is_valid
            info = {
                "step": self._step_count,
                "action_name": f"observe ({action_name_str})",
                "has_look": False,
                "window_focused": self._window.is_focused,
                "reward_breakdown": reward_info,
                "long_break": False,
                "attack_streak": 0,
                "long_break_count": self._long_break_count,
            }
            self._step_count += 1
            self._prev_action_id = disc_action_id
            return obs, reward, done, info

        # ── Long-break check ────────────────────────────────────
        # If the agent has been issuing attack actions repeatedly
        # without any block breaking, override with a sustained
        # hold so blocks that take many seconds actually break.
        long_break_triggered = False

        # Track item drops for pickup-reward suppression
        if is_drop:
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
            obs = self._execute_long_break_raw(action_keys, action_buttons, action_look)
            self._attack_streak = 0  # reset after long break
        else:
            # ── Normal action execution ─────────────────────────
            if self._window.is_valid:
                self._input.set_keys(action_keys)
                self._input.set_buttons(action_buttons)

                if action_look is not None:
                    dx, dy = action_look
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
        self._action_history.append(disc_action_id)

        # ── Reward heuristics ───────────────────────────────────
        reward_info = self._reward_computer.compute(
            obs, disc_action_id, _reward_weights, self._step_count,
        )
        reward = reward_info["total"]

        done = not self._window.is_valid
        info = {
            "step": self._step_count,
            "action_name": action_name_str,
            "has_look": action_look is not None,
            "window_focused": self._window.is_focused,
            "reward_breakdown": reward_info,
            "long_break": long_break_triggered,
            "attack_streak": self._attack_streak,
            "long_break_count": self._long_break_count,
        }

        self._step_count += 1
        self._prev_action_id = disc_action_id

        return obs, reward, done, info

    # ── Long-break helper ───────────────────────────────────────

    def _execute_long_break(self, action) -> Dict[str, torch.Tensor]:
        """Long-break using a MinecraftAction dataclass."""
        return self._execute_long_break_raw(action.keys, action.buttons, action.look)

    def _execute_long_break_raw(
        self,
        keys: frozenset,
        buttons: frozenset,
        look=None,
    ) -> Dict[str, torch.Tensor]:
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
            self._input.set_keys(keys)
            self._input.set_buttons(buttons | frozenset({"left"}))

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
        Also persists the location to the settings store.
        """
        if self._player_x is not None and self._player_z is not None:
            self._home_x = self._player_x
            self._home_z = self._player_z
            self._home_y = self._player_y
            log.info("Home location updated: (%.1f, %.1f)",
                     self._home_x, self._home_z)
            self._persist_home()
            if self._on_home_changed:
                self._on_home_changed()
        else:
            log.warning("Cannot set home — player position not yet known "
                        "(waiting for first position_update from mod).")

    def set_home_coords(self, x: float, y: float, z: float) -> None:
        """Set the home location to specific coordinates.

        Called from the GUI manual coordinate entry fields.
        """
        self._home_x = x
        self._home_y = y
        self._home_z = z
        log.info("Home location set manually: (%.1f, %.1f, %.1f)", x, y, z)
        self._persist_home()
        if self._on_home_changed:
            self._on_home_changed()

    def get_home(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Return the current home coordinates (x, y, z)."""
        return self._home_x, self._home_y, self._home_z

    def set_settings_store(self, store: Any) -> None:
        """Inject the settings store for home location persistence."""
        self._settings_store = store
        # Restore persisted home location if available
        saved = store.get("home_location")
        if saved and isinstance(saved, dict):
            sx = saved.get("x")
            sz = saved.get("z")
            sy = saved.get("y")
            if sx is not None and sz is not None:
                self._home_x = float(sx)
                self._home_z = float(sz)
                self._home_y = float(sy) if sy is not None else None
                log.info("Restored persisted home location: (%.1f, %.1f)",
                         self._home_x, self._home_z)

    def set_on_home_changed(self, callback: Any) -> None:
        """Register a callback for when home location changes."""
        self._on_home_changed = callback

    def _persist_home(self) -> None:
        """Save current home location to the settings store."""
        if self._settings_store is not None and self._home_x is not None:
            self._settings_store.set("home_location", {
                "x": round(self._home_x, 2),
                "y": round(self._home_y, 2) if self._home_y is not None else None,
                "z": round(self._home_z, 2),
            })

    def get_look_delta(self) -> tuple[float, float]:
        """
        Return (dyaw, dpitch) in degrees since the last call,
        then reset the baseline to the current yaw/pitch.

        Used during imitation learning to infer which direction the
        player moved their camera between env steps.

        Returns (0.0, 0.0) if yaw/pitch data is not yet available
        from the mod bridge.
        """
        yaw = self._player_yaw
        pitch = self._player_pitch

        if yaw is None or pitch is None:
            return 0.0, 0.0

        if self._prev_yaw is None or self._prev_pitch is None:
            # First call — establish baseline, return zero
            self._prev_yaw = yaw
            self._prev_pitch = pitch
            return 0.0, 0.0

        dyaw = yaw - self._prev_yaw
        dpitch = pitch - self._prev_pitch

        # Handle yaw wrap-around (-180 to 180)
        if dyaw > 180:
            dyaw -= 360
        elif dyaw < -180:
            dyaw += 360

        self._prev_yaw = yaw
        self._prev_pitch = pitch
        return dyaw, dpitch

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
