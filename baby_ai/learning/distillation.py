"""
Teacher → Student distillation engine.

Performs knowledge distillation using:
- MSE on continuous action vectors
- KL divergence on communication logits
- Feature matching on fused embeddings

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
        teacher_lock: threading.Lock | None = None,
        swap_lock: threading.Lock | None = None,
    ):
        self.student = student
        self.teacher = teacher
        self.lr = lr
        self.kl_weight = kl_weight
        self.feature_weight = feature_weight
        self.temperature = temperature
        self.use_amp = use_amp
        self._teacher_lock = teacher_lock or threading.Lock()

        # Create a staging copy of Student for offline updates.
        # Avoid copy.deepcopy — it fails if the model has been used
        # for inference (non-leaf tensors cached in the graph).  Instead
        # construct a fresh instance from the class + load weights.
        self._staging_student = type(student).__new__(type(student))
        nn.Module.__init__(self._staging_student)
        # Re-initialise using the same class constructor
        try:
            self._staging_student.__init__()  # StudentModel() uses DEFAULT_CONFIG
        except Exception:
            # Fallback: deepcopy under no_grad (works if no forward ran)
            with torch.no_grad():
                self._staging_student = copy.deepcopy(student)
        # Load the live weights into the staging copy
        self._staging_student.load_state_dict(student.state_dict())
        self._optimizer = torch.optim.Adam(
            self._staging_student.parameters(), lr=lr
        )
        # Learning Rate Warmup for distillation
        # Prevent wild target-driven jumps early on
        self._scheduler = torch.optim.lr_scheduler.LinearLR(
            self._optimizer, start_factor=0.01, total_iters=1000
        )
        
        self._scaler = torch.amp.GradScaler("cuda") if use_amp else None
        self._swap_lock = swap_lock or threading.Lock()
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
        # Ensure staging student is on the correct device (lazy move —
        # the deepcopy at init time happens before .to(device) is called
        # on the live student, so the copy starts on CPU).
        self._staging_student.to(device)
        self._staging_student.train()

        # Move batch to device
        inputs = {}
        valid_keys = {"vision", "audio", "code_x", "code_edge_index", "code_batch", "sensor", "hidden"}
        for k, v in batch.items():
            if k in valid_keys:
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v

        # Pass goal_embedding for goal-conditioned distillation
        if "goal_embedding" in batch:
            ge = batch["goal_embedding"]
            inputs["goal"] = ge.to(device) if isinstance(ge, torch.Tensor) else ge

        # Get Teacher targets (no grad)
        # Wrap in autocast so teacher outputs match the dtype that
        # autocast will produce for the student forward pass.
        # Acquire teacher lock to prevent concurrent forward on the
        # same SSM temporal core from the learner thread (state buffers
        # are not thread-safe).
        with self._teacher_lock:
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        teacher_out = self.teacher(**inputs)
                else:
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
            self._scaler.unscale_(self._optimizer)
            torch.nn.utils.clip_grad_norm_(self._staging_student.parameters(), max_norm=1.0)
            
            scale_before = self._scaler.get_scale()
            self._scaler.step(self._optimizer)
            self._scaler.update()
            skip_scheduler = (self._scaler.get_scale() < scale_before)
        else:
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(self._staging_student.parameters(), max_norm=1.0)
            self._optimizer.step()
            skip_scheduler = False

        if hasattr(self, "_scheduler") and not skip_scheduler:
            self._scheduler.step()

        self._step += 1

        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss.items()}

    def _compute_loss(self, student_out: dict, teacher_out: dict) -> dict:
        """Compute combined distillation loss."""
        # Continuous action MSE (diffusion policy outputs)
        action_loss = F.mse_loss(
            student_out["action"],
            teacher_out["action"].detach(),
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
            self.kl_weight * (action_loss + comm_kl)
            + self.feature_weight * feat_loss
        )

        return {
            "total": total,
            "action_loss": action_loss,
            "comm_kl": comm_kl,
            "feature_loss": feat_loss,
        }

    def swap_to_live(self) -> None:
        """
        Atomic weight swap: copy staging Student weights → live Student.

        Uses the swap lock to ensure the inference thread doesn't
        read partial state mid-update.
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
