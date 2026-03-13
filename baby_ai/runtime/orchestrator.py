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
from baby_ai.learning.intrinsic import JEPACuriosity
import torch.nn as nn
from baby_ai.learning.distillation import DistillationEngine
from baby_ai.learning.rewards import RewardComposer
from baby_ai.runtime.inference_thread import InferenceThread
from baby_ai.runtime.learner_thread import LearnerThread
from baby_ai.runtime.distill_thread import DistillThread
from baby_ai.config import System2Config
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
        # Use the Teacher's world model for curiosity so the intrinsic
        # signal reflects the best dynamics model, not the lagging Student.
        # Previously this used student.predictive, causing a bootstrap
        # problem where curiosity quality depended on distillation lag.
        self.curiosity = JEPACuriosity(
            world_model=self.teacher.predictive,
            reward_scale=1.0,
            max_reward=5.0,
        ).to(device)

        # Projection: Student fused_dim → Teacher hidden_dim so that
        # Student embeddings can be fed into the Teacher's world model
        # for curiosity computation (Student=512, Teacher=1024).
        student_fused = self.config.student.encoder.fused_dim
        teacher_state = self.config.teacher.hidden_dim
        if student_fused != teacher_state:
            self.curiosity_proj = nn.Linear(student_fused, teacher_state).to(device)
        else:
            self.curiosity_proj = None

        self.consolidator = Consolidator(
            ewc_lambda=self.config.training.ewc_lambda,
        )

        self.reward_composer = RewardComposer(
            intrinsic_weight_start=self.config.training.intrinsic_weight_start,
            intrinsic_weight_end=self.config.training.intrinsic_weight_end,
            intrinsic_decay_steps=self.config.training.intrinsic_decay_steps,
        )

        # --- Distillation ---
        # Lock hierarchy (always acquire in this order to avoid deadlock):
        #   1. _teacher_lock   — guards Teacher forward/backward
        #   2. _model_swap_lock — guards Student weight swap
        # A thread must NEVER acquire _teacher_lock while holding
        # _model_swap_lock.
        #
        # _teacher_lock: prevents concurrent forward/backward on the
        # Teacher from the learner and distillation threads (cuDNN/SSM
        # is not thread-safe — internal reserve buffers get corrupted).
        self._teacher_lock = threading.Lock()

        # _model_swap_lock: shared between inference and distillation
        # so swap_to_live() doesn't write weights while _infer() reads.
        self._model_swap_lock = threading.Lock()

        self.distill_engine = DistillationEngine(
            student=self.student,
            teacher=self.teacher,
            lr=self.config.training.distill_lr,
            kl_weight=self.config.training.distill_kl_weight,
            feature_weight=self.config.training.distill_feature_weight,
            use_amp=self.config.training.use_amp,
            teacher_lock=self._teacher_lock,
            swap_lock=self._model_swap_lock,
        )

        # Enable asymmetric augmentation for distillation
        if getattr(self.config.training, "distill_augment", True):
            self.distill_engine.enable_augmentation()

        # Configure progressive curriculum for loss weights
        self.distill_engine.configure_curriculum(
            kl_warmup_steps=getattr(self.config.training, "distill_kl_warmup_steps", 1000),
            feat_warmup_steps=getattr(self.config.training, "distill_feature_warmup_steps", 2000),
        )

        # --- Runtime threads ---
        self.inference_thread = InferenceThread(
            student=self.student,
            device=device,
            target_latency_ms=self.config.runtime.inference_target_ms,
            system2_config=self.config.student.system2,
            system3_config=self.config.student.system3,
            swap_lock=self._model_swap_lock,
            runtime_config=self.config.runtime,
        )

        self.learner_thread = LearnerThread(
            teacher=self.teacher,
            replay=self.replay,
            curiosity=self.curiosity,
            consolidator=self.consolidator,
            config=self.config.training,
            device=device,
            teacher_lock=self._teacher_lock,
            runtime_config=self.config.runtime,
            curiosity_proj=self.curiosity_proj,
        )

        self.distill_thread = DistillThread(
            distill_engine=self.distill_engine,
            replay=self.replay,
            learner_step_fn=lambda: self.learner_thread.step_count,
            config=self.config.training,
            device=device,
        )

        # Wire event-driven coordination: learner → distill
        self.learner_thread._distill_ready_callback = (
            self.distill_thread.notify_distill_ready
        )
        # Wire teacher version bumps for soft-label cache coherence
        self.learner_thread._distill_engine = self.distill_engine

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
        # Signal the learner thread that new data is available so
        # it doesn't have to wait for the next poll-timeout cycle.
        if hasattr(self, 'learner_thread') and hasattr(self.learner_thread, 'notify_data_ready'):
            self.learner_thread.notify_data_ready()

    def mark_episode_boundary(self) -> None:
        """Notify the replay buffer that the current episode has ended.

        Must be called on every env.reset() so that
        :meth:`PrioritizedReplayBuffer.sample_sequence` respects
        episode boundaries and avoids cross-episode sequences.
        """
        self.replay.mark_episode_boundary()

    def save_checkpoint(self, tag: str = "latest") -> Path:
        """Save model weights, optimizer state, and replay metadata."""
        ensure_dirs()
        path = CHECKPOINT_DIR / f"checkpoint_{tag}.pt"
        tmp_path = path.with_suffix(".pt.tmp")
        ckpt_data = {
            "student_state_dict": self.student.state_dict(),
            "teacher_state_dict": self.teacher.state_dict(),
            "curiosity_state_dict": self.curiosity.state_dict(),
            "system3_state_dict": self.inference_thread.system3_state_dict(),
            "config": self.config,
            "learner_step": self.learner_thread.step_count,
            "distill_count": self.distill_thread.stats["distill_count"],
            "replay_stats": self.replay.stats(),
            "reward_composer_step": self.reward_composer._step,
            # Save learner optimizer state (Adam momentum/variance) so
            # training resumes smoothly without a cold-start period.
            "learner_optimizer_state_dict": self.learner_thread.optimizer.state_dict(),
        }
        if self.curiosity_proj is not None:
            ckpt_data["curiosity_proj_state_dict"] = self.curiosity_proj.state_dict()
        torch.save(ckpt_data, tmp_path)

        # Atomic rename — prevents corrupted checkpoints if the process
        # crashes mid-write (especially on network storage like Z:\).
        # Retry with backoff because Windows / network drives can
        # briefly lock the destination file (antivirus, indexer, SMB).
        # Also catches FileNotFoundError which can occur if the .tmp
        # file is moved/deleted between torch.save and rename (rare
        # race on network drives or when two processes collide).
        for _attempt in range(5):
            try:
                tmp_path.replace(path)
                break
            except (PermissionError, FileNotFoundError, OSError) as exc:
                if _attempt == 4:
                    log.error(
                        "Could not replace checkpoint after 5 attempts (%s): %s",
                        type(exc).__name__, path,
                    )
                    break
                import time as _t
                _t.sleep(0.3 * (2 ** _attempt))  # 0.3, 0.6, 1.2, 2.4 s
        log.info("Checkpoint saved: %s", path)
        return path

    def load_checkpoint(self, path: Path | str) -> None:
        """Load a checkpoint.  Tolerates corrupt / truncated files
        and shape mismatches from architecture changes (e.g. action
        dim expansion)."""
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

        self._safe_load_state_dict(self.student, ckpt["student_state_dict"], "student")
        self._safe_load_state_dict(self.teacher, ckpt["teacher_state_dict"], "teacher")
        # Load curiosity module (supports old "icm_state_dict" key)
        curiosity_sd = ckpt.get("curiosity_state_dict") or ckpt.get("icm_state_dict")
        if curiosity_sd is not None:
            self._safe_load_state_dict(self.curiosity, curiosity_sd, "curiosity")
        # Restore the Student→Teacher fused-dim projection for curiosity
        if self.curiosity_proj is not None and "curiosity_proj_state_dict" in ckpt:
            self._safe_load_state_dict(
                self.curiosity_proj, ckpt["curiosity_proj_state_dict"], "curiosity_proj",
            )
        # Load System 3 module weights (GoalProposer, SubgoalPlanner)
        s3_sd = ckpt.get("system3_state_dict")
        if s3_sd:
            self.inference_thread.load_system3_state_dict(s3_sd)
        # Restore reward composer step counter so intrinsic weight
        # decay continues across sessions instead of resetting.
        if "reward_composer_step" in ckpt:
            self.reward_composer._step = ckpt["reward_composer_step"]
        # Restore learner step counter so consolidation timing,
        # distillation signalling, and periodic logging continue
        # from where they left off instead of restarting at 0.
        if "learner_step" in ckpt:
            self.learner_thread._step = ckpt["learner_step"]
        # Restore learner optimizer state (Adam momentum/variance)
        # to avoid a cold-start training instability period.
        opt_sd = ckpt.get("learner_optimizer_state_dict")
        if opt_sd is not None:
            try:
                self.learner_thread.optimizer.load_state_dict(opt_sd)
                log.info("Learner optimizer state restored.")
            except (ValueError, RuntimeError) as exc:
                log.warning(
                    "Could not restore learner optimizer state "
                    "(likely architecture change): %s", exc,
                )
        log.info("Checkpoint loaded from: %s", path)

    @staticmethod
    def _safe_load_state_dict(
        module: torch.nn.Module,
        state_dict: dict,
        label: str,
    ) -> None:
        """Load a state dict, skipping tensors whose shapes don't match.

        This handles architecture changes (e.g. action_dim 20→23)
        gracefully: matching weights are loaded, mismatched layers
        keep their fresh random initialization so training can
        adapt them quickly.
        """
        model_sd = module.state_dict()
        filtered = {}
        skipped = []
        for key, val in state_dict.items():
            if key in model_sd:
                if val.shape == model_sd[key].shape:
                    filtered[key] = val
                else:
                    skipped.append(
                        f"  {key}: checkpoint {tuple(val.shape)} "
                        f"→ model {tuple(model_sd[key].shape)}"
                    )
            else:
                skipped.append(f"  {key}: not in current model (removed)")
        missing = set(model_sd.keys()) - set(filtered.keys())
        if skipped:
            log.warning(
                "[%s] Skipped %d checkpoint tensors (shape mismatch):\n%s",
                label, len(skipped), "\n".join(skipped),
            )
        if missing - set(state_dict.keys()):
            new_keys = missing - set(state_dict.keys())
            log.info(
                "[%s] %d new parameters initialized randomly: %s",
                label, len(new_keys),
                ", ".join(sorted(new_keys)[:10])
                + ("..." if len(new_keys) > 10 else ""),
            )
        module.load_state_dict(filtered, strict=False)

    @property
    def lp_estimator(self) -> "LearningProgressEstimator":
        """Expose learner thread's LearningProgressEstimator for priority computation."""
        return self.learner_thread.lp_estimator

    def system_stats(self) -> dict:
        """Get a full system status report."""
        return {
            "inference": self.inference_thread.stats,
            "learner": self.learner_thread.stats,
            "distillation": self.distill_thread.stats,
            "replay": self.replay.stats(),
            "rewards": self.reward_composer.stats(),
        }
