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
from baby_ai.environments.minecraft.focus_guard import FocusGuard
from baby_ai.environments.minecraft.input_controller import InputController
from baby_ai.environments.minecraft.launcher import MinecraftLauncher
from baby_ai.environments.minecraft.window import WindowManager
from baby_ai.utils.logging import get_logger

log = get_logger("mc_env")


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
        resolution: Tuple[int, int] = (160, 160),
        step_delay_ms: float = 100.0,
        sensor_channels: int = 16,
        # ── Auto-launcher options ───────────────────────────────
        auto_launch: bool = False,
        mc_dir: str = "",
        mc_version: str = "1.21.11",
        world_name: str = "",
        player_name: str = "",
        player_uuid: str = "",
        max_memory_mb: int = 2048,
        launch_timeout_sec: float = 120.0,
        # ── Input guard ─────────────────────────────────────────
        block_user_input: bool = False,
    ):
        # ── Auto-launch Minecraft if requested ──────────────────
        self._launcher: Optional[MinecraftLauncher] = None
        self._guard: Optional[FocusGuard] = None
        hwnd: Optional[int] = None

        if auto_launch and mc_dir:
            log.info("Auto-launching Minecraft %s ...", mc_version)
            self._launcher = MinecraftLauncher(
                mc_dir=mc_dir,
                version=mc_version,
                player_name=player_name,
                player_uuid=player_uuid,
                max_memory_mb=max_memory_mb,
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
        # FocusGuard keeps MC unfocused so real user input never reaches it.
        # The AI's PostMessage input works regardless of focus state.
        # MC's pauseOnLostFocus:false keeps the game ticking in background.
        if block_user_input:
            self._guard = FocusGuard(mc_hwnd=self._window.hwnd)
            self._guard.start()

        self._resolution = resolution
        self._step_delay = step_delay_ms / 1000.0
        self._sensor_channels = sensor_channels

        # ── Episode state ───────────────────────────────────────
        self._step_count = 0
        self._last_step_time = 0.0
        self._last_frame: Optional[np.ndarray] = None
        self._prev_action_id: int = 0

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

        # ── Execute action ──────────────────────────────────────
        # Only send input if the MC window is still alive
        if self._window.is_valid:
            self._input.set_keys(action.keys)
            self._input.set_buttons(action.buttons)

            if action.look is not None:
                # mouse_look already guards against unfocused window
                self._input.mouse_look(action.look[0], action.look[1])

        # ── Pacing ──────────────────────────────────────────────
        elapsed = time.perf_counter() - self._last_step_time
        remaining = self._step_delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_step_time = time.perf_counter()

        # ── Observe ─────────────────────────────────────────────
        obs = self._observe()

        # ── Reward heuristics ───────────────────────────────────
        reward = self._compute_reward(obs)

        done = not self._window.is_valid
        info = {
            "step": self._step_count,
            "action_name": action.name,
            "has_look": action.look is not None,
            "window_focused": self._window.is_focused,
        }

        self._step_count += 1
        self._prev_action_id = action_id

        return obs, reward, done, info

    def close(self) -> None:
        """Release all held inputs, stop input guard, and stop Minecraft."""
        self._input.release_all()

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

    # ── Internals ───────────────────────────────────────────────

    def _observe(self) -> Dict[str, torch.Tensor]:
        """Capture a frame and package it as a model-ready observation dict."""
        raw_bgr, vision_tensor = self._capture.grab_both()
        self._last_frame = raw_bgr

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

    def _compute_reward(self, obs: Dict[str, torch.Tensor]) -> float:
        """
        Compute a lightweight extrinsic reward from frame analysis.

        For V1 this returns 0.0 — the primary learning signal comes
        from intrinsic curiosity (ICM) in the learning module.

        Future enhancements:
        - Health bar delta detection (template matching)
        - Death screen detection ("You Died!" text)
        - Time-alive bonus
        - Inventory change detection
        """
        # Small survival bonus to encourage continued play
        return 0.01

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
