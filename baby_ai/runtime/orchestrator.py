"""
Orchestrator — coordinates all runtime threads and the learning loop.

This is the top-level controller that:
1. Initializes all models, memory, and preprocessors
2. Starts inference, learner, and distillation threads
3. Handles checkpointing and monitoring
4. Provides clean shutdown
"""

from __future__ import annotations

import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

import torch

from baby_ai.config import (
    BabyAIConfig, DEFAULT_CONFIG, DEVICE,
    CHECKPOINT_DIR, ensure_dirs,
)
from baby_ai.models.student import StudentModel
from baby_ai.models.teacher import TeacherModel
from baby_ai.memory.replay_buffer import PrioritizedReplayBuffer
from baby_ai.memory.consolidation import Consolidator
from baby_ai.learning.intrinsic import ICM
from baby_ai.learning.distillation import DistillationEngine
from baby_ai.learning.rewards import RewardComposer
from baby_ai.runtime.inference_thread import InferenceThread
from baby_ai.runtime.learner_thread import LearnerThread
from baby_ai.runtime.distill_thread import DistillThread
from baby_ai.utils.logging import get_logger
from baby_ai.utils.profiling import count_parameters, full_system_report

log = get_logger("orchestrator", log_file="orchestrator.log")


class Orchestrator:
    """
    Top-level system controller.

    Call .start() to launch all threads, .stop() to shut down.
    Call .step() to feed an observation and get an action (through the inference thread).

    Args:
        config: Full system configuration.
    """

    def __init__(self, config: BabyAIConfig | None = None):
        self.config = config or DEFAULT_CONFIG
        ensure_dirs()

        device = self.config.device
        log.info("Initializing Baby-AI on device=%s", device)

        # --- Models ---
        log.info("Building Student model...")
        self.student = StudentModel(self.config.student)
        student_params = count_parameters(self.student)
        log.info("Student: %s parameters (%.1f M)", f"{student_params:,}", student_params / 1e6)

        log.info("Building Teacher model...")
        self.teacher = TeacherModel(self.config.teacher)
        teacher_params = count_parameters(self.teacher)
        log.info("Teacher: %s parameters (%.1f M)", f"{teacher_params:,}", teacher_params / 1e6)

        # --- Memory ---
        self.replay = PrioritizedReplayBuffer(
            capacity=self.config.training.replay_capacity,
            disk_cap_gb=self.config.training.replay_disk_cap_gb,
        )

        # --- Learning modules ---
        self.icm = ICM(
            state_dim=self.config.student.encoder.fused_dim,
            action_dim=self.config.student.action_dim,
        ).to(device)

        self.consolidator = Consolidator(
            ewc_lambda=self.config.training.ewc_lambda,
        )

        self.reward_composer = RewardComposer(
            intrinsic_weight_start=self.config.training.intrinsic_weight_start,
            intrinsic_weight_end=self.config.training.intrinsic_weight_end,
            intrinsic_decay_steps=self.config.training.intrinsic_decay_steps,
        )

        # --- Distillation ---
        # Lock to prevent concurrent forward/backward on the Teacher
        # from the learner and distillation threads (cuDNN GRU is not
        # thread-safe — its internal reserve buffers get corrupted).
        self._teacher_lock = threading.Lock()

        self.distill_engine = DistillationEngine(
            student=self.student,
            teacher=self.teacher,
            lr=self.config.training.distill_lr,
            kl_weight=self.config.training.distill_kl_weight,
            feature_weight=self.config.training.distill_feature_weight,
            use_amp=self.config.training.use_amp,
            teacher_lock=self._teacher_lock,
        )

        # --- Runtime threads ---
        self.inference_thread = InferenceThread(
            student=self.student,
            device=device,
            target_latency_ms=self.config.runtime.inference_target_ms,
        )

        self.learner_thread = LearnerThread(
            teacher=self.teacher,
            replay=self.replay,
            icm=self.icm,
            consolidator=self.consolidator,
            config=self.config.training,
            device=device,
            teacher_lock=self._teacher_lock,
        )

        self.distill_thread = DistillThread(
            distill_engine=self.distill_engine,
            replay=self.replay,
            learner_step_fn=lambda: self.learner_thread.step_count,
            config=self.config.training,
            device=device,
        )

        self._started = False

    def start(self) -> None:
        """Launch all threads."""
        if self._started:
            log.warning("System already started.")
            return

        log.info("=" * 60)
        log.info("Starting Baby-AI system...")
        log.info("=" * 60)

        # System report
        report = full_system_report(self.student, self.teacher)
        log.info("System report: %s", report)

        self.inference_thread.start()
        self.learner_thread.start()
        self.distill_thread.start()

        self._started = True
        log.info("All threads running.")

    def stop(self) -> None:
        """Gracefully shut down all threads and save checkpoint."""
        log.info("Shutting down Baby-AI system...")

        self.distill_thread.stop()
        self.learner_thread.stop()
        self.inference_thread.stop()

        self.save_checkpoint("shutdown")
        self._started = False
        log.info("System stopped.")

    def step(self, observation: dict) -> dict:
        """
        Submit an observation and get an action from the Student.

        Args:
            observation: Dict with raw/preprocessed modality tensors.

        Returns:
            Dict with action, value, utterance, latency_ms, etc.
        """
        if not self._started:
            raise RuntimeError("System not started. Call .start() first.")
        return self.inference_thread.submit(observation)

    def add_experience(
        self,
        transition: dict,
        priority: float | None = None,
    ) -> None:
        """
        Add a transition to the replay buffer.

        Called after the environment returns the next observation
        and reward. This feeds the Teacher's training.

        Args:
            transition: Dict with state, action, reward, next_state, etc.
            priority: Optional priority (higher = more important).
        """
        self.replay.add(transition, priority=priority)

    def save_checkpoint(self, tag: str = "latest") -> Path:
        """Save model weights, optimizer state, and replay metadata."""
        ensure_dirs()
        path = CHECKPOINT_DIR / f"checkpoint_{tag}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        torch.save({
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(),
            "icm_state_dict": self.icm.state_dict(),
            "config": self.config,
            "learner_step": self.learner_thread.step_count,
            "distill_count": self.distill_thread.stats["distill_count"],
            "replay_stats": self.replay.stats(),
        }, tmp_path)

        # Atomic rename — prevents corrupted checkpoints if the process
        # crashes mid-write (especially on network storage like Z:\).
        tmp_path.replace(path)
        log.info("Checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Load a checkpoint.  Tolerates corrupt / truncated files."""
        path = Path(path)
        if not path.exists():
            log.warning("Checkpoint not found: %s — starting fresh.", path)
            return

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except (RuntimeError, zipfile.BadZipFile, EOFError, Exception) as exc:
            log.error(
                "Checkpoint file is corrupt or unreadable: %s (%s). "
                "Starting fresh and renaming bad file.",
                path, exc,
            )
            # Keep the bad file for debugging but don't block startup
            bad_path = path.with_suffix(".pt.bad")
            try:
                path.rename(bad_path)
                log.info("Renamed corrupt checkpoint → %s", bad_path)
            except OSError:
                pass
            return

        self.student.load_state_dict(ckpt["student_state_dict"])
        self.teacher.load_state_dict(ckpt["teacher_state_dict"])
        if "icm_state_dict" in ckpt:
            self.icm.load_state_dict(ckpt["icm_state_dict"])
        log.info("Checkpoint loaded from: %s", path)

    def system_stats(self) -> dict:
        """Get a full system status report."""
        return {
            "inference": self.inference_thread.stats,
            "learner": self.learner_thread.stats,
            "distillation": self.distill_thread.stats,
            "replay": self.replay.stats(),
            "rewards": self.reward_composer.stats(),
        }
