"""
Baby-AI: Main entry point.

Initializes the full system and runs the continuous learning loop.
Can be run directly or imported as a module.

Usage:
    python main.py                    # Start with default config
    python main.py --config my.yaml   # Start with custom config
    python main.py --demo             # Run a quick demo with dummy data
    python main.py --profile          # Profile models and exit    python main.py --minecraft        # Play Minecraft with AI control"""

from __future__ import annotations

import argparse
import signal
import sys
import time
import atexit
import ctypes
from pathlib import Path

# --- Fallback to always release mouse cursor on crash ---
def _emergency_release_cursor():
    try:
        ctypes.windll.user32.ClipCursor(None)
    except Exception:
        pass

atexit.register(_emergency_release_cursor)
# --------------------------------------------------------

import torch

from baby_ai.config import BabyAIConfig, DEFAULT_CONFIG, ensure_dirs
from baby_ai.runtime.orchestrator import Orchestrator
from baby_ai.utils.logging import get_logger
from baby_ai.utils.profiling import (
    count_parameters,
    model_size_mb,
    gpu_memory_report,
    profile_inference,
    full_system_report,
)

log = get_logger("main", log_file="main.log")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baby-AI: Self-Evolving Multimodal Agent")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    parser.add_argument("--demo", action="store_true", help="Run a quick demo with dummy data.")
    parser.add_argument("--profile", action="store_true", help="Profile models and exit.")
    parser.add_argument("--minecraft", action="store_true", help="Play Minecraft with AI control.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint.")
    return parser.parse_args()


def profile_models(config: BabyAIConfig) -> None:
    """Profile Student and Teacher models."""
    from baby_ai.models.student import StudentModel
    from baby_ai.models.teacher import TeacherModel
    from baby_ai.preprocessing.video import VideoPreprocessor
    from baby_ai.preprocessing.audio import AudioPreprocessor

    device = config.device

    log.info("=" * 60)
    log.info("PROFILING BABY-AI MODELS")
    log.info("=" * 60)

    # Build models
    student = StudentModel(config.student).to(device)
    teacher = TeacherModel(config.teacher).to(device)

    s_params = count_parameters(student)
    t_params = count_parameters(teacher)

    log.info("Student: %s params (%.1f M) | %.1f MB", f"{s_params:,}", s_params / 1e6, model_size_mb(student))
    log.info("Teacher: %s params (%.1f M) | %.1f MB", f"{t_params:,}", t_params / 1e6, model_size_mb(teacher))

    # Create dummy inputs
    vid_prep = VideoPreprocessor(
        resolution=config.video.resolution,
        grayscale=config.video.grayscale,
    )
    aud_prep = AudioPreprocessor(
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels,
    )

    B = 1
    dummy_vision = vid_prep.dummy_input(B).to(device)
    dummy_audio = aud_prep.dummy_input(B).to(device)
    dummy_sensor = torch.randn(B, config.sensor.max_channels).to(device)

    # Profile Student inference
    log.info("\n--- Student Inference Profile ---")
    student.eval()
    with torch.no_grad():
        with torch.amp.autocast(device, enabled=config.training.use_amp):
            t0 = time.perf_counter()
            for _ in range(20):
                result = student.act(
                    vision=dummy_vision,
                    audio=dummy_audio,
                    sensor=dummy_sensor,
                )
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000 / 20
    log.info("Student avg inference: %.1f ms", elapsed)

    # GPU memory
    mem = gpu_memory_report()
    log.info("GPU memory: allocated=%.0f MB, reserved=%.0f MB, total=%.0f MB",
             mem["allocated_mb"], mem["reserved_mb"], mem["total_mb"])

    # Full report
    full_system_report(student, teacher)
    log.info("Profiling complete.")


def run_demo(config: BabyAIConfig) -> None:
    """Run a quick demo with dummy observations."""
    log.info("=" * 60)
    log.info("BABY-AI DEMO MODE")
    log.info("=" * 60)

    orchestrator = Orchestrator(config)

    # Load checkpoint if available
    from baby_ai.config import CHECKPOINT_DIR
    latest = CHECKPOINT_DIR / "checkpoint_latest.pt"
    if latest.exists():
        log.info("Loading checkpoint from %s", latest)
        orchestrator.load_checkpoint(latest)

    orchestrator.start()

    # Create dummy preprocessors
    from baby_ai.preprocessing.video import VideoPreprocessor
    from baby_ai.preprocessing.audio import AudioPreprocessor
    from baby_ai.preprocessing.sensors import SensorPreprocessor

    vid_prep = VideoPreprocessor(
        resolution=config.video.resolution,
        grayscale=config.video.grayscale,
    )
    aud_prep = AudioPreprocessor(
        sample_rate=config.audio.sample_rate,
        n_mels=config.audio.n_mels,
    )
    sensor_prep = SensorPreprocessor(max_channels=config.sensor.max_channels)

    log.info("\nRunning demo loop (50 steps)...\n")

    try:
        for step in range(50):
            # Generate dummy observations
            observation = {
                "vision": vid_prep.dummy_input(1),
                "audio": aud_prep.dummy_input(1),
                "sensor": sensor_prep.dummy_input(1),
            }

            # Get action from Student
            result = orchestrator.step(observation)

            # Create a dummy transition for the replay buffer
            fused = result.get("fused", torch.randn(1, config.student.encoder.fused_dim))
            transition = {
                "vision": observation["vision"].squeeze(0),
                "audio": observation["audio"].squeeze(0),
                "sensor": observation["sensor"].squeeze(0),
                "action": result.get("action", torch.tensor([0])),
                "reward": torch.tensor(0.1),  # dummy reward
                "fused": fused.squeeze(0) if fused.dim() > 1 else fused,
            }
            orchestrator.add_experience(transition, priority=1.0)

            log.info(
                "Step %3d | action=%s | value=%.3f | latency=%.1f ms",
                step,
                result.get("action", "?"),
                result.get("value", torch.tensor([[0]])).item() if isinstance(result.get("value"), torch.Tensor) else 0,
                result.get("latency_ms", 0),
            )

            time.sleep(0.05)  # simulate real-time pacing

    except KeyboardInterrupt:
        log.info("Demo interrupted.")
    finally:
        # Save and stop
        orchestrator.save_checkpoint("demo")
        orchestrator.stop()

    log.info("Demo complete. System stats:")
    stats = orchestrator.system_stats()
    for section, data in stats.items():
        log.info("  %s: %s", section, data)


def run_minecraft(config: BabyAIConfig, checkpoint_path: str | None = None) -> None:
    """
    Train the agent by playing Minecraft.

    1. Finds the Minecraft window (must be running in windowed mode).
    2. Captures screen frames as observations.
    3. Sends keyboard/mouse input via Win32 PostMessage.
    4. Uses JEPA curiosity as the primary intrinsic reward signal.

    The user's real keyboard and mouse are NOT affected — all input
    goes directly to the Minecraft window handle.
    """
    from baby_ai.environments.minecraft import MinecraftEnv
    from baby_ai.environments.minecraft.actions import player_input_to_continuous
    from baby_ai.environments.minecraft.action_decoder import continuous_action_name
    from baby_ai.config import CHECKPOINT_DIR

    mc = config.minecraft

    log.info("=" * 60)
    log.info("BABY-AI  ×  MINECRAFT")
    log.info("=" * 60)
    log.info("Input mode  : %s", mc.input_mode)
    log.info("Step delay  : %.0f ms", mc.step_delay_ms)
    log.info("Resolution  : %s", mc.capture_resolution)
    log.info("Auto-launch : %s", mc.auto_launch)
    log.info("World       : %s", mc.world_name)
    log.info("Input guard : %s", mc.block_user_input)

    # ── Countdown (only when NOT auto-launching) ────────────────
    if not mc.auto_launch and mc.initial_pause_sec > 0:
        log.info("")
        for remaining in range(int(mc.initial_pause_sec), 0, -1):
            log.info("  Starting in %d …  (switch to Minecraft now)", remaining)
            time.sleep(1.0)
        log.info("")

    # ── Environment (auto-launches MC if configured) ────────────
    env = MinecraftEnv(
        window_title=mc.window_title,
        input_mode=mc.input_mode,
        resolution=mc.capture_resolution,
        step_delay_ms=mc.step_delay_ms,
        sensor_channels=config.sensor.max_channels,
        auto_launch=mc.auto_launch,
        mc_dir=mc.mc_dir,
        mc_version=mc.mc_version,
        world_name=mc.world_name,
        player_name=mc.player_name,
        player_uuid=mc.player_uuid,
        max_memory_mb=mc.max_memory_mb,
        window_width=mc.window_width,
        window_height=mc.window_height,
        launch_timeout_sec=mc.launch_timeout_sec,
        block_user_input=mc.block_user_input,
        mod_bridge_port=mc.mod_bridge_port,
    )

    # ── Orchestrator (Student + Teacher + Learner + Distill) ────
    orchestrator = Orchestrator(config)

    # Resume from checkpoint if available
    if checkpoint_path:
        orchestrator.load_checkpoint(checkpoint_path)
    else:
        latest = CHECKPOINT_DIR / "checkpoint_minecraft.pt"
        if latest.exists():
            log.info("Resuming from %s", latest)
            orchestrator.load_checkpoint(latest)

    orchestrator.start()

    # ── Wire mod bridge into inference thread for System 2 pause ─
    # The env creates the ModBridge, but InferenceThread is built
    # before the env.  Now that both exist, connect them so that
    # System 2 planning can freeze/resume game ticks.
    if env._mod_bridge is not None:
        orchestrator.inference_thread.set_mod_bridge(env._mod_bridge)

    # ── JEPA curiosity for intrinsic reward ─────────────────────────────
    curiosity = orchestrator.curiosity

    # ── UI Control Panel + Reward Toggles ───────────────────────
    from baby_ai.ui.control_panel import AIControlPanel, get_imitation_enabled
    from baby_ai.ui.reward_toggles import RewardToggleState
    from baby_ai.ui.controls_state import AIControlsState
    from baby_ai.ui.reward_weights import RewardWeightsState
    from baby_ai.ui.settings_store import SettingsStore
    from baby_ai.environments.minecraft.input_controller import set_controls_state
    from baby_ai.environments.minecraft.env import set_reward_weights

    settings_store = SettingsStore()
    toggle_state = RewardToggleState()
    controls_state = AIControlsState()
    reward_weights = RewardWeightsState()

    # Wire AI Controls state into the input controller so it can
    # filter disabled keys/buttons/look before sending them.
    set_controls_state(controls_state)

    # Wire reward weights into the env so it reads dynamic weights.
    set_reward_weights(reward_weights)

    # Inject settings store into env for home location persistence.
    env.set_settings_store(settings_store)

    # Set-home callback: grabs current coords and updates env.
    def _on_set_home() -> None:
        env.set_home()

    # Manual home coordinate callback from GUI entry fields.
    def _on_set_home_coords(x: float, y: float, z: float) -> None:
        env.set_home_coords(x, y, z)

    control_panel = AIControlPanel(
        toggle_state=toggle_state,
        controls_state=controls_state,
        reward_weights=reward_weights,
        on_set_home=_on_set_home,
        on_set_home_coords=_on_set_home_coords,
        input_guard=env._guard,
        settings_store=settings_store,
    )
    control_panel.start()

    # Wire home-change notifications so the GUI updates when
    # /sethome is used in-game or set_home() is called.
    def _notify_gui_home_changed() -> None:
        hx, hy, hz = env.get_home()
        if hx is not None and hz is not None:
            control_panel.update_home_display(
                hx, hy if hy is not None else 64.0, hz
            )

    env.set_on_home_changed(_notify_gui_home_changed)

    # ── 10-second warm-up (no input lock, no training) ──────────
    # Give the user time to arrange windows, alt-tab, etc.
    # Temporarily disable the input guard so keyboard/mouse are free.
    _WARMUP_SEC = 10
    if env._guard is not None:
        env._guard._kb_blocked = False
        env._guard._mouse_blocked = False
    log.info("")
    for remaining in range(_WARMUP_SEC, 0, -1):
        log.info("  AI starts in %d s — arrange your windows now …", remaining)
        time.sleep(1.0)
    log.info("  GO!  Training/inference starting now.")
    log.info("")
    # Re-enable input guard blocking.
    if env._guard is not None:
        env._guard._kb_blocked = True
        env._guard._mouse_blocked = True

    # ── Training loop ──────────────────────────────────────────
    log.info("Minecraft training loop started — Ctrl+C to stop.")
    prev_fused: torch.Tensor | None = None
    prev_action: torch.Tensor | None = None
    prev_obs: dict | None = None
    raw_ir: float = 0.0        # raw JEPA curiosity (before normalisation)
    intrinsic_r: float = 0.0   # normalised + clamped curiosity reward
    episode_reward = 0.0
    episode_steps = 0
    last_distill_count = 0  # track distillation rounds for reward reset

    # Running RMS normalizer for intrinsic (JEPA) reward.  Keeps the
    # curiosity signal bounded regardless of forward-model training
    # state.  EMA of squared values → divide by RMS → clamp [0, 5].
    _ir_ema_sq: float = 1.0   # EMA of intrinsic_r² (init 1.0 to avoid /0)
    _IR_EMA_DECAY: float = 0.99
    _IR_MAX: float = 1.0      # hard ceiling after normalisation

    # Accumulators for reward channels between log intervals.
    # Without these, events that happen between logged steps are invisible.
    _acc_keys = (
        "block_break", "item_pickup", "block_place",
        "crafting", "building_streak", "creative_sequence",
        "interaction", "exploration", "death_penalty",
        "item_drop_penalty", "damage_taken", "healing",
        "food_reward", "xp_reward",
        "hotbar_spam_penalty", "height_penalty",
        "pitch_penalty", "stagnation_penalty",
        "home_proximity", "visual_change", "movement",
        "new_chunk",
    )
    _acc = {k: 0.0 for k in _acc_keys}

    # Track whether imitation mode was active on the previous tick so
    # we can toggle the input guard exactly once on state transitions.
    _prev_imitation_active: bool = False

    try:
        obs = env.reset()

        while True:
            # Check UI controls
            if control_panel.is_stopped:
                log.info("Stop requested from Control Panel.")
                break

            # Ctrl+Q via InputGuard — save & quit
            if env._guard is not None and env._guard.quit_requested:
                log.info("Ctrl+Q — save & quit requested.")
                break

            if control_panel.is_paused:
                time.sleep(0.5)
                continue

            # Ctrl+P via InputGuard — pause AI
            if env._guard is not None and env._guard.ai_paused:
                time.sleep(0.5)
                continue

            # ── Imitation learning mode ──────────────────────────
            # When active: suppress AI outputs (noop), unblock user
            # keyboard/mouse so the human can play and demonstrate.
            # Observations and reward collection continue normally.
            _imitation_active = get_imitation_enabled()

            if _imitation_active and not _prev_imitation_active:
                # Just switched ON — release AI held keys, unblock user
                if env._input is not None:
                    env._input.release_all()
                if env._guard is not None:
                    env._guard._kb_blocked = False
                    env._guard._mouse_blocked = False
                    env._guard.clear_player_input()
                # Reset yaw/pitch baseline so first delta is clean
                env._prev_yaw = None
                env._prev_pitch = None
                log.info("Imitation learning ON — AI inputs suppressed, user controls unlocked.")
            elif not _imitation_active and _prev_imitation_active:
                # Just switched OFF — re-lock user input, clear tracked state
                if env._guard is not None:
                    env._guard._kb_blocked = True
                    env._guard._mouse_blocked = True
                    env._guard.clear_player_input()
                log.info("Imitation learning OFF — AI inputs resumed, user controls locked.")
            _prev_imitation_active = _imitation_active

            # ── Model inference ──────────────────────────────────
            result = orchestrator.step(obs)
            # Action is now a 20-dim continuous tensor from DiffusionPolicy
            action_tensor = result["action"]   # (20,) continuous
            fused = result.get("fused")

            # ── Store transition (using PREVIOUS step's data) ───
            if prev_fused is not None and fused is not None:
                with torch.no_grad():
                    # Ensure all inputs have a batch dimension (B=1)
                    _cur_state = prev_fused.to(config.device)
                    _cur_next  = fused.to(config.device)
                    _cur_act   = prev_action.to(config.device)
                    if _cur_state.dim() == 1:
                        _cur_state = _cur_state.unsqueeze(0)
                    if _cur_next.dim() == 1:
                        _cur_next = _cur_next.unsqueeze(0)
                    if _cur_act.dim() == 1:
                        _cur_act = _cur_act.unsqueeze(0)  # (20,) → (1, 20)
                    cur_out = curiosity(_cur_state, _cur_next, _cur_act)
                raw_ir = cur_out["curiosity_reward"].mean().item()
                # Running RMS normalisation — keeps magnitude stable
                # even while the forward model is poorly trained.
                _ir_ema_sq = (_IR_EMA_DECAY * _ir_ema_sq
                              + (1.0 - _IR_EMA_DECAY) * (raw_ir ** 2 + 1e-8))
                intrinsic_r = raw_ir / max(_ir_ema_sq ** 0.5, 1e-8)
                intrinsic_r = max(0.0, min(intrinsic_r, _IR_MAX))
                # Respect the intrinsic toggle — if disabled, zero it.
                if not toggle_state.is_enabled("intrinsic"):
                    intrinsic_r = 0.0

                # Extract per-channel extrinsic rewards from env info
                rb_raw = info.get("reward_breakdown", {})
                # Apply reward-channel toggles (disabled channels → 0).
                rb = toggle_state.filter_channels(rb_raw)

                # ── Compute total reward from env breakdown ─────
                # The env computes raw per-channel values; the Reward
                # Weights UI sets the multipliers; toggles zero disabled
                # channels.  We recompute the weighted total here using
                # the *filtered* breakdown so that:
                #   1. ALL channels are included (not just a subset)
                #   2. Toggles take effect instantly
                #   3. Weight slider changes are reflected live
                #   4. JEPA intrinsic is layered on top
                w = reward_weights.snapshot()

                # Positive channels (reward)
                extrinsic_total = (
                    rb.get("survival", 0.005)      * w.get("survival", 1.0)
                    + rb.get("visual_change", 0.0) * w.get("visual_change", 0.1)
                    + rb.get("action_diversity", 0.0) * w.get("action_diversity", 0.5)
                    + rb.get("interaction", 0.0)   * w.get("interaction", 0.8)
                    + rb.get("exploration", 0.0)   * w.get("exploration", 0.8)
                    + rb.get("movement", 0.0)      * w.get("movement", 0.3)
                    + rb.get("new_chunk", 0.0)     * w.get("new_chunk", 1.0)
                    + rb.get("block_break", 0.0)   * w.get("block_break", 4.0)
                    + rb.get("item_pickup", 0.0)   * w.get("item_pickup", 6.0)
                    + rb.get("block_place", 0.0)   * w.get("block_place", 4.0)
                    + rb.get("crafting", 0.0)      * w.get("crafting", 25.0)
                    + rb.get("building_streak", 0.0) * w.get("building_streak", 3.0)
                    + rb.get("creative_sequence", 0.0) * w.get("creative_sequence", 6.0)
                    + rb.get("healing", 0.0)       * w.get("healing", 1.0)
                    + rb.get("food_reward", 0.0)   * w.get("food_reward", 0.8)
                    + rb.get("xp_reward", 0.0)     * w.get("xp_reward", 0.1)
                    + rb.get("home_proximity", 0.0) * w.get("home_proximity", 1.5)
                )

                # Penalty channels (subtracted)
                penalty_total = (
                    rb.get("idle_penalty", 0.0)    * w.get("idle_penalty", 2.0)
                    + rb.get("death_penalty", 0.0) * w.get("death_penalty", 5.0)
                    + rb.get("stagnation_penalty", 0.0) * w.get("stagnation_penalty", 3.0)
                    + rb.get("item_drop_penalty", 0.0) * w.get("item_drop_penalty", 3.0)
                    + rb.get("damage_taken", 0.0)  * w.get("damage_taken", 1.5)
                    + rb.get("hotbar_spam_penalty", 0.0) * w.get("hotbar_spam_penalty", 2.0)
                    + rb.get("height_penalty", 0.0) * w.get("height_penalty", 2.5)
                    + rb.get("pitch_penalty", 0.0) * w.get("pitch_penalty", 3.0)
                )

                # Intrinsic curiosity — added on top of env rewards
                intrinsic_contribution = intrinsic_r * w.get("intrinsic", 0.1)

                total_r = extrinsic_total - penalty_total + intrinsic_contribution

                # Accumulate filtered channel values for periodic logging
                for _k in _acc_keys:
                    _acc[_k] += rb.get(_k, 0.0)

                # Include next_fused so the JEPA world model gets a
                # training signal from replay (dynamics + KL loss).
                _cur_fused = prev_fused.squeeze(0) if prev_fused.dim() > 1 else prev_fused
                _next_fused = fused.squeeze(0) if fused.dim() > 1 else fused
                _action = prev_action.squeeze(0) if prev_action.dim() > 1 else prev_action
                transition = {
                    "vision": prev_obs["vision"].squeeze(0),
                    "audio": prev_obs["audio"].squeeze(0),
                    "sensor": prev_obs["sensor"].squeeze(0),
                    "action": _action,
                    "reward": torch.tensor(total_r),
                    "fused": _cur_fused,
                    "next_fused": _next_fused.detach(),
                }
                # System 3: attach active goal embedding for hindsight training
                _goal_emb = result.get("goal_embedding")
                if _goal_emb is not None:
                    transition["goal_embedding"] = (
                        _goal_emb.squeeze(0) if _goal_emb.dim() > 1 else _goal_emb
                    )
                priority = abs(intrinsic_r) + 0.01
                orchestrator.add_experience(transition, priority=priority)
                episode_reward += total_r

            # ── Execute action in Minecraft ─────────────────────
            obs, ext_reward, done, info = env.step(
                action_tensor, observation_only=_imitation_active,
            )
            episode_steps += 1

            # ── Push live stats to control panel ────────────────
            control_panel.update_live_stats(
                reward=episode_reward, step=episode_steps,
            )

            # ── Reset reward counter after distillation ─────────
            cur_distill = orchestrator.distill_thread._distill_count
            if cur_distill > last_distill_count:
                log.info(
                    "Distillation round %d complete — resetting "
                    "episode_reward (was %.2f) to 0.",
                    cur_distill, episode_reward,
                )
                episode_reward = 0.0
                last_distill_count = cur_distill

                # ── Save screenshot of what the AI sees ─────────
                # Captures the last frame the env observed, which is
                # exactly what the vision encoder processed.
                try:
                    import cv2, datetime
                    from baby_ai.config import SCREENSHOT_DIR
                    _ss_dir = SCREENSHOT_DIR
                    _ss_dir.mkdir(parents=True, exist_ok=True)
                    frame = getattr(env, "_last_frame", None)
                    if frame is not None:
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        fname = _ss_dir / f"distill_{cur_distill:04d}_{ts}.png"
                        cv2.imwrite(str(fname), frame)
                        log.info("Distillation screenshot saved -> %s", fname)
                except Exception as _ss_err:
                    log.warning("Failed to save distillation screenshot: %s", _ss_err)

            # ── Logging (every 50 steps) ────────────────────────
            if episode_steps % 50 == 0:
                ir_raw_str = f"{raw_ir:.4f}" if prev_fused is not None else "N/A"
                ir_str = f"{intrinsic_r:.4f}" if prev_fused is not None else "N/A"
                # Show accumulated rewards since the last log interval
                # (so events between log steps are visible).
                log.info(
                    "Step %5d | action=%-28s | curiosity=%s(raw=%s)"
                    " | move=%.3f | vis=%.3f | interact=%.3f | explore=%.3f"
                    " | blk_break=%.3f | item=%.3f | place=%.3f"
                    " | craft=%.3f | bld_streak=%.3f | creative=%.3f"
                    " | death=%.1f | drop=%.2f | stag=%.2f"
                    " | dmg=%.2f | heal=%.2f | food=%.2f | xp=%.2f"
                    " | hbar=%.2f | height=%.2f | pitch=%.2f | home=%.2f"
                    " | ep_reward=%.2f | latency=%.0f ms",
                    episode_steps,
                    continuous_action_name(action_tensor),
                    ir_str,
                    ir_raw_str,
                    _acc["movement"],
                    _acc["visual_change"],
                    _acc["interaction"],
                    _acc["exploration"],
                    _acc["block_break"],
                    _acc["item_pickup"],
                    _acc["block_place"],
                    _acc["crafting"],
                    _acc["building_streak"],
                    _acc["creative_sequence"],
                    _acc["death_penalty"],
                    _acc["item_drop_penalty"],
                    _acc["stagnation_penalty"],
                    _acc["damage_taken"],
                    _acc["healing"],
                    _acc["food_reward"],
                    _acc["xp_reward"],
                    _acc["hotbar_spam_penalty"],
                    _acc["height_penalty"],
                    _acc["pitch_penalty"],
                    _acc["home_proximity"],
                    episode_reward,
                    result.get("latency_ms", 0),
                )
                # Reset accumulators after logging
                _acc = {k: 0.0 for k in _acc_keys}

                # Extra imitation mode info — show the player's actual action
                if _imitation_active and prev_action is not None:
                    log.info(
                        "  [IMITATION] player_action=%s",
                        continuous_action_name(prev_action),
                    )

            # ── Episode boundary ────────────────────────────────
            max_steps = mc.max_episode_steps
            if done or (max_steps > 0 and episode_steps >= max_steps):
                log.info(
                    "Episode ended after %d steps  (reward=%.2f)",
                    episode_steps, episode_reward,
                )
                # If the Minecraft window is gone, break out
                # instead of trying to reset (prevents zombie loop
                # and ensures the tkinter GUI closes too).
                if not env._window.is_valid:
                    log.info("Minecraft window lost — exiting.")
                    break
                orchestrator.inference_thread.reset_hidden()
                orchestrator.inference_thread.reset_system3()
                obs = env.reset()
                prev_fused, prev_action, prev_obs = None, None, None
                episode_reward = 0.0
                episode_steps = 0
                _acc = {k: 0.0 for k in _acc_keys}
                continue

            # ── Periodic checkpoint (every 2000 steps) ──────────
            if episode_steps % 2000 == 0:
                orchestrator.save_checkpoint("minecraft")

            prev_fused = fused
            # ── Record the action for the NEXT transition ───────
            # During imitation mode, use the *player's actual input*
            # instead of the AI's prediction — this is the entire
            # point of imitation learning.
            if _imitation_active and env._guard is not None:
                held_keys, held_buttons = env._guard.snapshot_player_input()
                dyaw, dpitch = env.get_look_delta()
                # Build a 20-dim continuous vector from the player's
                # physical input so curiosity / replay stay consistent.
                prev_action = player_input_to_continuous(
                    held_keys, held_buttons, dyaw, dpitch,
                )
            else:
                prev_action = action_tensor
            prev_obs = obs

    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
    finally:
        env.close()
        orchestrator.save_checkpoint("minecraft")
        orchestrator.stop()
        # Ensure the tkinter control panel closes when the
        # training loop exits for any reason.
        control_panel.request_close()

    log.info("Minecraft session complete.")
    stats = orchestrator.system_stats()
    for section, data in stats.items():
        log.info("  %s: %s", section, data)


def main() -> None:
    args = parse_args()

    # Load config
    if args.config:
        config = BabyAIConfig.load(args.config)
        log.info("Config loaded from %s", args.config)
    else:
        config = DEFAULT_CONFIG

    ensure_dirs()

    if args.profile:
        profile_models(config)
        return

    if args.demo:
        run_demo(config)
        return

    if args.minecraft:
        run_minecraft(config, args.checkpoint)
        return

    # --- Full continuous learning mode ---
    log.info("=" * 60)
    log.info("BABY-AI: Starting continuous learning system")
    log.info("=" * 60)

    orchestrator = Orchestrator(config)

    if args.checkpoint:
        orchestrator.load_checkpoint(args.checkpoint)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        log.info("Received shutdown signal.")
        orchestrator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    orchestrator.start()

    log.info(
        "System running. Feed observations via orchestrator.step().\n"
        "Press Ctrl+C to stop."
    )

    # Keep main thread alive
    try:
        while True:
            time.sleep(10)
            stats = orchestrator.system_stats()
            log.info("Status: learner=%d steps | distill=%d rounds | replay=%d",
                     stats["learner"]["step"],
                     stats["distillation"]["distill_count"],
                     stats["replay"]["size"])
    except KeyboardInterrupt:
        orchestrator.stop()


if __name__ == "__main__":
    main()
