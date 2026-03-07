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
    4. Uses intrinsic curiosity (ICM) as the primary reward signal.

    The user's real keyboard and mouse are NOT affected — all input
    goes directly to the Minecraft window handle.
    """
    from baby_ai.environments.minecraft import MinecraftEnv
    from baby_ai.environments.minecraft.actions import action_name
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

    # ── ICM for intrinsic reward ────────────────────────────────
    icm = orchestrator.icm
    reward_composer = orchestrator.reward_composer

    # ── UI Control Panel ─────────────────────────────────────────
    from baby_ai.ui.control_panel import AIControlPanel
    control_panel = AIControlPanel()
    control_panel.start()

    # ── Training loop ──────────────────────────────────────────
    log.info("Minecraft training loop started — Ctrl+C to stop.")
    prev_fused: torch.Tensor | None = None
    prev_action: torch.Tensor | None = None
    prev_obs: dict | None = None
    episode_reward = 0.0
    episode_steps = 0

    try:
        obs = env.reset()

        while True:
            # Check UI controls
            if control_panel.is_stopped:
                log.info("Stop requested from Control Panel.")
                break
                
            if control_panel.is_paused:
                time.sleep(0.5)
                continue

            # ── Model inference ──────────────────────────────────
            result = orchestrator.step(obs)
            action_id = result["action"].item()
            fused = result.get("fused")

            # ── Store transition (using PREVIOUS step's data) ───
            if prev_fused is not None and fused is not None:
                with torch.no_grad():
                    icm_out = icm(
                        prev_fused.to(config.device),
                        fused.to(config.device),
                        prev_action.to(config.device),
                    )
                intrinsic_r = icm_out["curiosity_reward"].mean().item()
                extrinsic_r = 0.01  # survival bonus
                total_r = reward_composer.compose(
                    extrinsic=extrinsic_r, intrinsic=intrinsic_r,
                )

                transition = {
                    "vision": prev_obs["vision"].squeeze(0),
                    "audio": prev_obs["audio"].squeeze(0),
                    "sensor": prev_obs["sensor"].squeeze(0),
                    "action": prev_action,
                    "reward": torch.tensor(total_r),
                    "fused": prev_fused.squeeze(0) if prev_fused.dim() > 1 else prev_fused,
                }
                priority = abs(intrinsic_r) + 0.01
                orchestrator.add_experience(transition, priority=priority)
                episode_reward += total_r

            # ── Execute action in Minecraft ─────────────────────
            obs, ext_reward, done, info = env.step(action_id)
            episode_steps += 1

            # ── Logging (every 50 steps) ────────────────────────
            if episode_steps % 50 == 0:
                ir_str = f"{intrinsic_r:.4f}" if prev_fused is not None else "N/A"
                log.info(
                    "Step %5d | action=%-28s | curiosity=%s | ep_reward=%.2f | latency=%.0f ms",
                    episode_steps,
                    action_name(action_id),
                    ir_str,
                    episode_reward,
                    result.get("latency_ms", 0),
                )

            # ── Episode boundary ────────────────────────────────
            max_steps = mc.max_episode_steps
            if done or (max_steps > 0 and episode_steps >= max_steps):
                log.info(
                    "Episode ended after %d steps  (reward=%.2f)",
                    episode_steps, episode_reward,
                )
                orchestrator.inference_thread.reset_hidden()
                obs = env.reset()
                prev_fused, prev_action, prev_obs = None, None, None
                episode_reward = 0.0
                episode_steps = 0
                continue

            # ── Periodic checkpoint (every 2000 steps) ──────────
            if episode_steps % 2000 == 0:
                orchestrator.save_checkpoint("minecraft")

            prev_fused = fused
            prev_action = result["action"]
            prev_obs = obs

    except KeyboardInterrupt:
        log.info("Training interrupted by user.")
    finally:
        env.close()
        orchestrator.save_checkpoint("minecraft")
        orchestrator.stop()

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
