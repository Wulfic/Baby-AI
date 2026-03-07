"""
Teacher → Student distillation engine.

Performs knowledge distillation using:
- KL divergence on action logits and communication logits
- Feature matching on fused embeddings
- Prioritized sampling from replay buffer

Supports atomic Student weight swaps to avoid blocking inference.
"""

from __future__ import annotations

import copy
import threading
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_ai.utils.logging import get_logger

log = get_logger("distillation", log_file="distill.log")


class DistillationEngine:
    """
    Distills knowledge from Teacher to Student.

    Strategy:
    1. Sample prioritized minibatch from replay
    2. Get Teacher soft targets (action + comm logits + features)
    3. Compute distillation loss (KL + feature matching)
    4. Update a staging copy of Student weights
    5. Atomic pointer swap: staging → live Student

    Args:
        student: Live Student model (used for inference).
        teacher: Teacher model (provides soft targets).
        lr: Distillation learning rate (small for stability).
        kl_weight: Weight for KL divergence loss.
        feature_weight: Weight for feature matching loss.
        temperature: Softmax temperature for softer targets.
    """

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module,
        lr: float = 1e-5,
        kl_weight: float = 1.0,
        feature_weight: float = 0.5,
        temperature: float = 2.0,
        use_amp: bool = True,
    ):
        self.student = student
        self.teacher = teacher
        self.lr = lr
        self.kl_weight = kl_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.use_amp = use_amp

        # Create a staging copy of Student for offline updates
        self._staging_student = copy.deepcopy(student)
        self._optimizer = torch.optim.Adam(
            self._staging_student.parameters(), lr=lr
        )
        self._scaler = torch.amp.GradScaler("cuda") if use_amp else None
        self._swap_lock = threading.Lock()
        self._step = 0

    def _kl_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(Teacher || Student) with temperature scaling.

        Higher temperature → softer distributions → more knowledge transfer.
        """
        T = self.temperature
        s_log_probs = F.log_softmax(student_logits / T, dim=-1)
        t_probs = F.softmax(teacher_logits / T, dim=-1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (T * T)
        return kl

    def _feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        L2 feature matching loss.

        Since Student and Teacher may have different feature dims,
        we project Teacher features to Student dim if needed.
        """
        if student_features.shape != teacher_features.shape:
            # Simple linear projection (lazy init)
            if not hasattr(self, "_feature_proj"):
                self._feature_proj = nn.Linear(
                    teacher_features.shape[-1],
                    student_features.shape[-1],
                ).to(teacher_features.device)
            teacher_features = self._feature_proj(teacher_features)

        return F.mse_loss(student_features, teacher_features.detach())

    def distill_step(
        self,
        batch: dict,
        device: str = "cuda",
    ) -> dict:
        """
        Perform one distillation step on a minibatch.

        Args:
            batch: Dict with input tensors (vision, audio, etc.)
                   ready to be passed to model.forward().

        Returns:
            dict with loss components and total loss.
        """
        self._staging_student.train()
        # Ensure we don't accidentally set teacher to eval if it's shared with learner_thread
        # self.teacher.eval()

        # Move batch to device
        inputs = {}
        valid_keys = {"vision", "audio", "code_x", "code_edge_index", "code_batch", "sensor", "hidden"}
        for k, v in batch.items():
            if k in valid_keys:
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v

        # Get Teacher targets (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(**inputs)

        # Get Student outputs
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                student_out = self._staging_student(**inputs)
                loss = self._compute_loss(student_out, teacher_out)
        else:
            student_out = self._staging_student(**inputs)
            loss = self._compute_loss(student_out, teacher_out)

        # Backward
        self._optimizer.zero_grad()
        if self._scaler is not None:
            self._scaler.scale(loss["total"]).backward()
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            loss["total"].backward()
            self._optimizer.step()

        self._step += 1

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss.items()}

    def _compute_loss(self, student_out: dict, teacher_out: dict) -> dict:
        """Compute combined distillation loss."""
        # KL on action logits
        action_kl = self._kl_loss(
            student_out["action_logits"],
            teacher_out["action_logits"],
        )

        # KL on communication logits
        comm_kl = self._kl_loss(
            student_out["comm_logits"],
            teacher_out["comm_logits"],
        )

        # Feature matching on fused embeddings
        feat_loss = self._feature_loss(
            student_out["fused"],
            teacher_out["fused"],
        )

        total = (
            self.kl_weight * (action_kl + comm_kl)
            + self.feature_weight * feat_loss
        )

        return {
            "total": total,
            "action_kl": action_kl,
            "comm_kl": comm_kl,
            "feature_loss": feat_loss,
        }

    def swap_to_live(self) -> None:
        """
        Atomic weight swap: copy staging Student weights → live Student.

        Uses a lock to ensure inference thread doesn't read partial state.
        """
        with self._swap_lock:
            live_sd = self._staging_student.state_dict()
            self.student.load_state_dict(live_sd)
        log.info("Distillation swap at step %d", self._step)

    def sync_staging_from_live(self) -> None:
        """Sync staging Student from live (e.g., after external updates)."""
        with self._swap_lock:
            self._staging_student.load_state_dict(self.student.state_dict())

    @property
    def step_count(self) -> int:
        return self._step
