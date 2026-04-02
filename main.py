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

# --- Emergency cursor release ---
# Windows ClipCursor() confines the mouse to a rectangle.  If the process
# crashes while the cursor is clipped (e.g. during Minecraft input mode),
# the user's mouse would be stuck.  This atexit handler guarantees the
# cursor is always freed, even on unhandled exceptions.
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
    parser.add_argument("--offline", action="store_true",
                        help="Offline training: replay existing imitation data for multiple epochs (no Minecraft).")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for --offline training (default: 10).")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use all matching GPUs for --offline training. "
                             "Each GPU trains independently, then checkpoints are averaged.")
    parser.add_argument("--gpu-rank", type=int, default=None,
                        help=argparse.SUPPRESS)  # internal: set by multi-gpu spawner
    parser.add_argument("--gpu-total", type=int, default=None,
                        help=argparse.SUPPRESS)  # internal: set by multi-gpu spawner
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint.")
    return parser.parse_args()


def profile_models(config: BabyAIConfig) -> None:
    """Profile Student and Teacher models.

    Builds both models on the configured device, reports parameter counts
    and memory usage, then runs 20 dummy inference passes to measure
    average latency.  Useful for verifying the model fits in GPU memory
    and meets the <200 ms inference target.
    """
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
    """Run a quick demo with dummy observations.

    Creates an Orchestrator and feeds 50 steps of random noise through
    the full pipeline (encode → fuse → temporal → policy → replay).
    Loads the latest checkpoint if available, so this also serves as
    a quick smoke-test that a saved model loads and runs.
    """
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
        orchestrator.save_checkpoint("latest")
        orchestrator.stop()

    log.info("Demo complete. System stats:")
    stats = orchestrator.system_stats()
    for section, data in stats.items():
        log.info("  %s: %s", section, data)


def run_offline_training(
    config: BabyAIConfig,
    checkpoint_path: str | None = None,
    epochs: int = 10,
    gpu_rank: int | None = None,
    gpu_total: int | None = None,
) -> None:
    """Train the Teacher and Student offline on existing replay data.

    This is the fast-iteration mode: no Minecraft window, no GUI, no
    inference thread.  The function:

    1. Builds models and loads the checkpoint (if any).
    2. Rebuilds the replay buffer's SumTree from on-disk chunk data.
    3. Runs ``epochs`` full passes over the replay data via the
       LearnerThread, interleaving distillation rounds.
    4. Saves a checkpoint at the end of each epoch.

    Usage::

        python main.py --offline                   # 10 epochs (default)
        python main.py --offline --epochs 50       # 50 epochs
        python main.py --offline --checkpoint latest  # resume + fine-tune
        python main.py --offline --epochs 10 --multi-gpu  # split across GPUs

    When ``gpu_rank`` is set (via ``--multi-gpu``), the final
    checkpoint is saved as ``offline_final_gpu{rank}`` instead of
    ``offline_final`` so the parent process can average them.

    Because the GPU never has to wait for Minecraft frames or input
    latency, training is **10-50× faster** than online mode.
    """
    log.info("=" * 60)
    if gpu_rank is not None:
        log.info("OFFLINE TRAINING MODE  (%d epochs)  [GPU worker %d / %d]",
                 epochs, gpu_rank, gpu_total or 0)
    else:
        log.info("OFFLINE TRAINING MODE  (%d epochs)", epochs)
    log.info("=" * 60)

    ensure_dirs()

    # ── Build orchestrator (models, replay, threads) ────────────
    orchestrator = Orchestrator(config)

    if checkpoint_path:
        orchestrator.load_checkpoint(checkpoint_path)
    else:
        # Try the default latest checkpoint
        from baby_ai.config import CHECKPOINT_DIR
        latest = CHECKPOINT_DIR / "checkpoint_latest.pt"
        if latest.exists():
            orchestrator.load_checkpoint(latest)
            log.info("Loaded checkpoint: %s", latest)
        else:
            log.warning("No checkpoint found -- models start from random init.")

    # ── Rebuild replay from on-disk chunk files ─────────────────
    n_recovered = orchestrator.replay.rebuild_from_disk(default_priority=1.0)
    if n_recovered == 0:
        log.error(
            "No replay data found on disk.  Run at least one online session "
            "with --minecraft first to collect imitation data."
        )
        return

    # ── Enable sequential (temporal) sampling ───────────────────
    # Rolling prefetch: the sequential sampler automatically keeps
    # the current + next 2 chunks (~1 500 transitions, ~500 MB) in
    # RAM and evicts consumed chunks.  No bulk preload needed.
    # Transitions are replayed in the order they were recorded,
    # preserving temporal coherence for the Jamba temporal core
    # and JEPA world model.
    orchestrator.replay.enable_sequential_mode()

    replay_size = orchestrator.replay.size
    batch_size = config.training.micro_batch_size
    steps_per_epoch = max(replay_size // batch_size, 1)
    distill_interval = config.training.distill_every_n_steps

    log.info("Replay: %d transitions | batch_size=%d | steps/epoch=%d",
             replay_size, batch_size, steps_per_epoch)
    log.info("Distillation every %d teacher steps", distill_interval)
    log.info("Consolidation every %d steps (Fisher recompute + EWC snapshot)",
             config.training.consolidation_every_n_steps)

    # ── Get direct references to threads ────────────────────────
    # We drive training synchronously instead of starting background
    # threads -- this avoids thread overhead and gives us deterministic
    # epoch boundaries.
    learner = orchestrator.learner_thread
    distill = orchestrator.distill_thread

    # Move models to device
    device = config.device
    orchestrator.teacher.to(device).train()
    orchestrator.student.to(device)

    # ── Cosine LR schedule for offline training ─────────────────
    total_train_steps = steps_per_epoch * epochs
    learner.enable_cosine_schedule(
        total_steps=total_train_steps,
        warmup_steps=min(500, total_train_steps // 10),
        peak_lr_mult=6.0,
        eta_min_mult=0.1,
    )

    # ── Verbose progress helpers ────────────────────────────────
    _LOG_EVERY = 10          # print a progress line every N steps
    _epoch_t0 = time.perf_counter()
    _global_t0 = time.perf_counter()
    _step_errors = 0

    # ── Training loop ───────────────────────────────────────────
    total_steps = 0
    try:
        for epoch in range(1, epochs + 1):
            epoch_steps = 0
            _epoch_t0 = time.perf_counter()

            # Reset sequential cursor so we iterate the full dataset
            # from the beginning each epoch.
            orchestrator.replay.reset_sequential()

            log.info("=" * 60)
            log.info("EPOCH %d / %d  (teacher_step=%d)", epoch, epochs, learner.step_count)
            log.info("=" * 60)

            while not orchestrator.replay.sequential_exhausted:
                if orchestrator.replay.sequential_remaining < batch_size:
                    # Not enough transitions left for a full batch — end epoch.
                    break

                _step_t0 = time.perf_counter()

                try:
                    # _train_step already handles: loss computation,
                    # backward, optimizer step, priority updates,
                    # EWC consolidation, and n-step sequence training
                    # every 4 steps. Its internal logging fires every
                    # 100 steps.
                    learner._train_step()
                except Exception as e:
                    _step_errors += 1
                    log.warning("Train step error (step %d): %s",
                                total_steps, e, exc_info=(_step_errors <= 3))
                    # Recover memory after allocation failures on long runs.
                    import gc; gc.collect()
                    torch.cuda.empty_cache()
                    continue

                _step_dt = time.perf_counter() - _step_t0
                epoch_steps += 1
                total_steps += 1

                # ── Periodic distillation ───────────────────────
                # In offline mode the distill thread isn't running, so
                # we call _distill_round directly on the same thread.
                if total_steps % distill_interval == 0:
                    try:
                        _d_t0 = time.perf_counter()
                        distill._distill_round()
                        _d_dt = time.perf_counter() - _d_t0
                        log.info(
                            ">>> DISTILL round %d at step %d  (%.1f s)",
                            distill._distill_count, total_steps, _d_dt,
                        )
                    except Exception as e:
                        log.warning("Distill error: %s", e, exc_info=True)

                # ── Verbose progress line ───────────────────────
                if epoch_steps % _LOG_EVERY == 0 or epoch_steps == 1:
                    pct = 100.0 * epoch_steps / steps_per_epoch
                    elapsed_epoch = time.perf_counter() - _epoch_t0
                    steps_per_sec = epoch_steps / max(elapsed_epoch, 1e-6)
                    remaining_steps = steps_per_epoch - epoch_steps
                    eta_sec = remaining_steps / max(steps_per_sec, 1e-6)
                    # Get LR from optimizer
                    lr_vals = [pg["lr"] for pg in learner.optimizer.param_groups]
                    lr_str = "/".join(f"{v:.2e}" for v in lr_vals)
                    log.info(
                        "  [%3.0f%%] step %d/%d (global %d) | "
                        "%.0f ms/step | %.1f steps/s | "
                        "lr=%s | ETA %dm%02ds",
                        pct, epoch_steps, steps_per_epoch, learner.step_count,
                        _step_dt * 1000, steps_per_sec,
                        lr_str,
                        int(eta_sec) // 60, int(eta_sec) % 60,
                    )

            # ── End-of-epoch distillation + swap ────────────────
            # Skip if sequential data is exhausted — all transitions
            # have already been consumed by learner + periodic distills.
            if (orchestrator.replay.size >= batch_size
                    and not orchestrator.replay.sequential_exhausted):
                try:
                    _d_t0 = time.perf_counter()
                    distill._distill_round()
                    _d_dt = time.perf_counter() - _d_t0
                    log.info(">>> End-of-epoch distill round %d  (%.1f s)",
                             distill._distill_count, _d_dt)
                except Exception as e:
                    log.warning("End-of-epoch distill error: %s", e)

            # ── Checkpoint ──────────────────────────────────────
            # Only rank 0 saves shared checkpoints to avoid file
            # race conditions on network drives.
            _save_t0 = time.perf_counter()
            if gpu_rank in (None, 0):
                orchestrator.save_checkpoint(f"offline_epoch_{epoch:03d}")
                # Only update 'latest' during single-GPU offline training.
                # Multi-GPU workers must NOT overwrite latest — the parent
                # process will write the properly averaged result.
                if gpu_rank is None:
                    orchestrator.save_checkpoint("latest")  # overwrite rolling
            _save_dt = time.perf_counter() - _save_t0

            _epoch_dt = time.perf_counter() - _epoch_t0
            _total_dt = time.perf_counter() - _global_t0
            log.info("-" * 60)
            log.info(
                "EPOCH %d DONE | %d steps in %.1fs (%.1f steps/s) | "
                "teacher_step=%d | distill_rounds=%d | "
                "checkpoint saved (%.1fs) | errors=%d",
                epoch, epoch_steps, _epoch_dt,
                epoch_steps / max(_epoch_dt, 1e-6),
                learner.step_count, distill._distill_count,
                _save_dt, _step_errors,
            )
            remaining_epochs = epochs - epoch
            if remaining_epochs > 0:
                eta_total = _epoch_dt * remaining_epochs
                log.info(
                    "  Remaining: %d epochs, ETA ~%dm%02ds  (wall time so far: %dm%02ds)",
                    remaining_epochs,
                    int(eta_total) // 60, int(eta_total) % 60,
                    int(_total_dt) // 60, int(_total_dt) % 60,
                )
            log.info("-" * 60)
            _step_errors = 0  # reset per-epoch error counter

    except KeyboardInterrupt:
        log.info("Offline training interrupted by user at epoch/step %d/%d.",
                 epoch, total_steps)

    # ── Final save ──────────────────────────────────────────────
    # When running as a multi-GPU worker, save with a per-GPU tag
    # so the parent process can find and average all of them.
    # Single-GPU just saves straight to "latest".
    if gpu_rank is not None:
        final_tag = f"offline_final_gpu{gpu_rank}"
    else:
        final_tag = "latest"
    final_path = orchestrator.save_checkpoint(final_tag)

    # ── Clean up intermediate epoch checkpoints ─────────────────
    # These are only useful during training for crash recovery.
    # Now that the final checkpoint is saved, they waste disk space
    # (~1.3 GB each).  Multi-GPU workers also clean their own epochs
    # so the parent only needs to average the final per-GPU files.
    if gpu_rank in (None, 0):
        import glob as _glob
        epoch_pattern = str(CHECKPOINT_DIR / "checkpoint_offline_epoch_*.pt")
        stale_epochs = sorted(_glob.glob(epoch_pattern))
        if stale_epochs:
            log.info("Cleaning up %d intermediate epoch checkpoints...", len(stale_epochs))
            for ep_path in stale_epochs:
                try:
                    Path(ep_path).unlink()
                    log.info("  Removed: %s", Path(ep_path).name)
                except OSError as exc:
                    log.warning("  Could not remove %s: %s", ep_path, exc)

    _total_wall = time.perf_counter() - _global_t0
    log.info("=" * 60)
    if gpu_rank is not None:
        log.info("OFFLINE TRAINING COMPLETE  (GPU worker %d / %d)", gpu_rank, gpu_total or 0)
    else:
        log.info("OFFLINE TRAINING COMPLETE")
    log.info("  Total teacher steps : %d", learner.step_count)
    log.info("  Distillation rounds : %d", distill._distill_count)
    log.info("  Wall time           : %dm%02ds", int(_total_wall) // 60, int(_total_wall) % 60)
    log.info("  Checkpoint saved as : '%s'", final_tag)
    log.info("=" * 60)


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
        camera_smooth_steps=mc.camera_smooth_steps,
    )

    # ── Orchestrator (Student + Teacher + Learner + Distill) ────
    orchestrator = Orchestrator(config)

    # Resume from checkpoint if available
    if checkpoint_path:
        orchestrator.load_checkpoint(checkpoint_path)
    else:
        latest = CHECKPOINT_DIR / "checkpoint_latest.pt"
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

    # ── Wire InputController for ESC-based pause ──
    # System 2/3 press ESC to open the MC pause menu (visible freeze).
    if hasattr(env, '_input') and env._input is not None:
        orchestrator.inference_thread.set_input_controller(env._input)

    # ── JEPA curiosity for intrinsic reward ─────────────────────────────
    curiosity = orchestrator.curiosity

    # ── Reward composer (z-score normalization + intrinsic annealing) ──
    reward_composer = orchestrator.reward_composer

    # ── Learning progress estimator (for replay priorities) ───────────
    lp_estimator = orchestrator.lp_estimator

    # ── UI Control Panel + Reward Toggles ───────────────────────
    from baby_ai.ui.control_panel import AIControlPanel, get_imitation_enabled, get_record_only
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

    # Wire live reward recomposition into the learner so that
    # weight/toggle changes take effect across ALL replay data
    # immediately, not just new transitions.
    orchestrator.learner_thread._reward_weights = reward_weights
    orchestrator.learner_thread._toggle_state = toggle_state

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
    lp_priority: float = 0.0   # learning progress priority bonus
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
        "hotbar_spam_penalty", "inventory_spam_penalty",
        "height_penalty", "pitch_penalty", "stagnation_penalty",
        "home_proximity", "visual_change", "movement",
        "new_chunk", "entity_hit", "mob_killed",
    )
    _acc = {k: 0.0 for k in _acc_keys}

    # Track whether imitation mode was active on the previous tick so
    # we can toggle the input guard exactly once on state transitions.
    _prev_imitation_active: bool = False
    _prev_record_only_active: bool = False

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

            # ── Record-only mode ─────────────────────────────────
            # When active: skip ALL inference and curiosity.  Only
            # capture raw observations + player's actions + mod-event
            # rewards.  Stored transitions are trainable offline later.
            _record_only_active = get_record_only()

            if _record_only_active and not _prev_record_only_active:
                # Transition ON — unlock user input
                if env._input is not None:
                    env._input.release_all()
                if env._guard is not None:
                    env._guard._kb_blocked = False
                    env._guard._mouse_blocked = False
                    env._guard.clear_player_input()
                env._prev_yaw = None
                env._prev_pitch = None
                prev_fused = None          # avoid stale graph refs
                log.info("Record-Only ON -- AI disabled, recording player data only.")
            elif not _record_only_active and _prev_record_only_active:
                # Transition OFF — re-lock user input, reset refs
                if env._guard is not None:
                    env._guard._kb_blocked = True
                    env._guard._mouse_blocked = True
                    env._guard.clear_player_input()
                prev_fused = None
                log.info("Record-Only OFF -- resuming normal operation.")
            _prev_record_only_active = _record_only_active

            if _record_only_active:
                # Lightweight recording path — no GPU inference at all.
                # Use a zero placeholder for fused; the learner's
                # forward pass will recompute from raw obs anyway.
                # Use fused_dim (encoder output) not hidden_dim (Jamba core);
                # they happen to be equal for Student (512) but are semantically
                # different — fused is what gets stored in replay transitions.
                _fused_dim = config.student.encoder.fused_dim
                fused = torch.zeros(_fused_dim)

                # Store transition from the PREVIOUS tick
                if prev_fused is not None and prev_obs is not None and prev_action is not None:
                    _action = prev_action.squeeze(0) if prev_action.dim() > 1 else prev_action
                    transition = {
                        "vision":     prev_obs["vision"].squeeze(0),
                        "audio":      prev_obs["audio"].squeeze(0),
                        "sensor":     prev_obs["sensor"].squeeze(0),
                        "action":     _action,
                        "reward":     torch.tensor(0.0),
                        "fused":      prev_fused,
                        "next_fused": fused.detach(),
                        "is_demo":    torch.tensor(1.0),
                    }
                    _demo_boost = getattr(config.training, "demo_priority_boost", 5.0)
                    orchestrator.add_experience(transition, priority=_demo_boost)

                # Advance env — observation only, no AI action
                obs, ext_reward, done, info = env.step(
                    torch.zeros(23), observation_only=True,
                )
                episode_steps += 1

                # Capture the player's physical input as a 23-dim tensor
                if env._guard is not None:
                    held_keys, held_buttons = env._guard.snapshot_player_input()
                    dyaw, dpitch = env.get_look_delta()
                    prev_action = player_input_to_continuous(
                        held_keys, held_buttons, dyaw, dpitch,
                    )
                else:
                    prev_action = torch.zeros(23)

                prev_fused = fused
                prev_obs = obs

                control_panel.update_live_stats(
                    reward=episode_reward, step=episode_steps,
                )
                continue  # skip normal inference + curiosity path

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
            # Action is a 23-dim continuous tensor from the policy head
            action_tensor = result["action"]   # (23,) continuous
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
                        _cur_act = _cur_act.unsqueeze(0)  # (23,) → (1, 23)
                    # Project Student fused (512) → Teacher state dim (1024)
                    # so it matches the Teacher's LatentWorldModel input.
                    if orchestrator.curiosity_proj is not None:
                        _cur_state = orchestrator.curiosity_proj(_cur_state)
                        _cur_next  = orchestrator.curiosity_proj(_cur_next)
                    cur_out = curiosity(_cur_state, _cur_next, _cur_act)
                raw_ir = cur_out["curiosity_reward"].mean().item()
                # Update Learning Progress Estimator — tracks whether the
                # world model is still improving on this observation.
                # Positive progress = still learning = higher replay priority.
                lp_priority = max(lp_estimator.update("global", raw_ir), 0.0)
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

                # ── Compose total reward via RewardComposer ─────
                # Uses z-score normalization, intrinsic weight annealing,
                # and the live GUI weight overrides — all in one place.
                w = reward_weights.snapshot()
                channel_values = dict(rb)  # shallow copy of filtered env channels
                channel_values["intrinsic"] = intrinsic_r  # layer curiosity on top
                total_r = reward_composer.compose_dynamic(channel_values, w)

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
                    # Store RAW per-channel breakdown (pre-toggle) so the
                    # learner can recompose rewards with live weight/toggle
                    # changes.  Using rb_raw (not the filtered rb) ensures
                    # that enabling a previously-disabled channel will
                    # retroactively include its values from old replay data.
                    "reward_channels": {**rb_raw, "intrinsic": intrinsic_r},
                    "fused": _cur_fused,
                    "next_fused": _next_fused.detach(),
                    "is_demo": torch.tensor(1.0 if _imitation_active else 0.0),
                }
                # System 3: attach active goal embedding for hindsight training
                _goal_emb = result.get("goal_embedding")
                if _goal_emb is not None:
                    transition["goal_embedding"] = (
                        _goal_emb.squeeze(0) if _goal_emb.dim() > 1 else _goal_emb
                    )
                priority = abs(intrinsic_r) + lp_priority + 0.01
                # Human demonstrations are high-value — boost priority
                if _imitation_active:
                    _demo_boost = getattr(config.training, "demo_priority_boost", 5.0)
                    priority *= _demo_boost
                # Only save AI transitions with positive reward;
                # demos are always saved regardless of reward.
                if _imitation_active or total_r > 0:
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
                    " | hbar=%.2f | inv_spam=%.2f | height=%.2f | pitch=%.2f | home=%.2f"
                    " | new_chunk=%.2f | entity_hit=%.2f | mob_kill=%.2f"
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
                    _acc["inventory_spam_penalty"],
                    _acc["height_penalty"],
                    _acc["pitch_penalty"],
                    _acc["home_proximity"],
                    _acc["new_chunk"],
                    _acc["entity_hit"],
                    _acc["mob_killed"],
                    episode_reward,
                    result.get("latency_ms", 0),
                )
                # Reset accumulators after logging
                _acc = {k: 0.0 for k in _acc_keys}

                # System 2/3 status
                _inf_stats = orchestrator.inference_thread.stats
                if _inf_stats.get("system2_trigger_count", 0) > 0 or _inf_stats.get("system3_trigger_count", 0) > 0:
                    log.info(
                        "  [THINKING] S2 triggers=%d | S3 triggers=%d | uncertainty=%.3f"
                        " | active_goal=%s | subgoals=%d (idx=%d)",
                        _inf_stats.get("system2_trigger_count", 0),
                        _inf_stats.get("system3_trigger_count", 0),
                        _inf_stats.get("uncertainty", 0.0),
                        _inf_stats.get("active_goal", False),
                        _inf_stats.get("subgoal_queue_len", 0),
                        _inf_stats.get("subgoal_idx", 0),
                    )

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
                orchestrator.mark_episode_boundary()
                obs = env.reset()
                prev_fused, prev_action, prev_obs = None, None, None
                episode_reward = 0.0
                episode_steps = 0
                _acc = {k: 0.0 for k in _acc_keys}
                continue

            # ── Periodic checkpoint (every 2000 steps) ──────────
            if episode_steps % 2000 == 0:
                orchestrator.save_checkpoint("latest")

            prev_fused = fused
            # ── Record the action for the NEXT transition ───────
            # During imitation mode, use the *player's actual input*
            # instead of the AI's prediction — this is the entire
            # point of imitation learning.
            if _imitation_active and env._guard is not None:
                held_keys, held_buttons = env._guard.snapshot_player_input()
                dyaw, dpitch = env.get_look_delta()
                # Build a 23-dim continuous vector from the player's
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
        orchestrator.save_checkpoint("latest")
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

    # Apply any model-size overrides saved by the GUI's Model tab.
    from baby_ai.ui.settings_store import SettingsStore as _SettingsStore
    _boot_store = _SettingsStore()
    _saved_model = _boot_store.get("model_config")
    if _saved_model and isinstance(_saved_model, dict):
        from baby_ai.ui.model_tab import apply_model_config
        apply_model_config(config, _saved_model)
        log.info("Applied saved model config (%d overrides)", len(_saved_model))

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

    if args.offline:
        # ── Multi-GPU: spawn parallel workers + merge ────────
        if args.multi_gpu and args.gpu_rank is None:
            from baby_ai.utils.multigpu import run_multi_gpu_offline
            run_multi_gpu_offline(
                epochs=args.epochs,
                checkpoint_path=args.checkpoint,
                config_path=args.config,
            )
            return
        # ── Single-GPU (or a spawned multi-gpu worker) ──────
        run_offline_training(
            config,
            args.checkpoint,
            epochs=args.epochs,
            gpu_rank=args.gpu_rank,
            gpu_total=args.gpu_total,
        )
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
