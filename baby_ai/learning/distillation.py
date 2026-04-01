"""
Teacher → Student distillation engine.

Performs knowledge distillation using:
- MSE on continuous action vectors
- KL divergence on communication logits (temperature-scaled)
- Feature matching on fused embeddings (with lazy linear projection)

Key design choices:
- **Staging copy**: All gradient updates happen on an offline copy of
  the Student.  Once a distillation round is complete, weights are
  atomically swapped into the live Student used by the inference thread.
  This avoids blocking inference during backprop.
- **Teacher label cache**: An LRU cache keyed by (replay_index,
  teacher_version) avoids redundant Teacher forward passes when the
  same high-priority transitions are re-sampled.
- **Asymmetric augmentation**: Student sees cropped/jittered/noisy
  frames while Teacher sees clean frames, following BYOL.
- **Curriculum warmup**: KL weight ramps from 20%→100% over
  kl_warmup_steps; feature weight ramps from 0→100% over
  feat_warmup_steps.  This avoids unstable gradients while
  the Student's representations are still random.
"""

from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_ai.learning.augmentations import DistillAugmentor
from baby_ai.utils.logging import get_logger

log = get_logger("distillation", log_file="distill.log")


class _TeacherLabelCache:
    """Thread-safe LRU cache for Teacher forward-pass outputs.

    Keyed by (replay_index, teacher_version).  When the Teacher updates
    (version bump), stale entries are naturally evicted because new
    lookups use the new version and old entries are LRU-oldest.

    Motivation:
        During a single distillation round, the same replay indices
        are often re-sampled (especially high-priority transitions).
        A Teacher forward pass is ~3× more expensive than a Student
        pass, so caching saves significant GPU time.

    The cache stores *detached* CPU tensors to avoid holding GPU memory.
    """

    def __init__(self, max_size: int = 5000):
        self._max_size = max_size
        self._cache: OrderedDict[tuple[int, int], dict[str, torch.Tensor]] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, idx: int, version: int) -> Optional[dict[str, torch.Tensor]]:
        """Lookup cached teacher outputs.  Returns None on miss."""
        key = (idx, version)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None

    def put(self, idx: int, version: int, outputs: dict[str, torch.Tensor]) -> None:
        """Store teacher outputs (detached, on CPU)."""
        key = (idx, version)
        # Detach and move to CPU to avoid GPU memory leak
        cpu_out: dict[str, torch.Tensor] = {}
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                cpu_out[k] = v.detach().cpu()
        with self._lock:
            self._cache[key] = cpu_out
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self) -> None:
        """Clear the entire cache (e.g. after Teacher weight update)."""
        with self._lock:
            self._cache.clear()

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / max(total, 1)

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
        }


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
        # deepcopy is safe here because the staging copy is created at
        # orchestrator init time, before any forward pass runs on the
        # live student (which would leave non-leaf tensors in the graph
        # and cause deepcopy to fail).
        try:
            with torch.no_grad():
                self._staging_student = copy.deepcopy(student)
        except Exception:
            # Fallback: construct a fresh instance via __new__ + __init__.
            # WARNING: __init__() uses DEFAULT_CONFIG — if the user's
            # config differs (e.g. policy_type='diffusion'), this will
            # create a mismatched architecture that load_state_dict
            # may fail on.
            log.warning(
                "deepcopy failed for staging Student — falling back to "
                "DEFAULT_CONFIG re-init.  Custom configs may mismatch."
            )
            self._staging_student = type(student).__new__(type(student))
            nn.Module.__init__(self._staging_student)
            self._staging_student.__init__()
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

        # Teacher soft-label LRU cache.  Eliminates redundant teacher
        # forward passes when the same high-priority replay transitions
        # are re-sampled within a distillation round.
        self._label_cache = _TeacherLabelCache(max_size=5000)
        self._teacher_version = 0  # bumped when teacher weights change

        # Asymmetric augmentation: Student sees augmented vision,
        # Teacher sees clean.  Initialised lazily from config.
        self._augmentor: DistillAugmentor | None = None

        # Progressive curriculum ramp parameters — set by caller
        # via configure_curriculum().
        self._kl_warmup_steps: int = 1000
        self._feat_warmup_steps: int = 2000
        self._kl_weight_final: float = kl_weight
        self._feat_weight_final: float = feature_weight

    # ── Configuration helpers ───────────────────────────────────────────

    def enable_augmentation(
        self,
        crop_ratio: float = 0.85,
        jitter_strength: float = 0.15,
        noise_std: float = 0.02,
    ) -> None:
        """Enable asymmetric vision augmentation for the Student."""
        self._augmentor = DistillAugmentor(
            crop_ratio=crop_ratio,
            jitter_strength=jitter_strength,
            noise_std=noise_std,
        )
        log.info("Distillation augmentation enabled (crop=%.2f, jitter=%.2f, noise=%.3f).",
                 crop_ratio, jitter_strength, noise_std)

    def configure_curriculum(
        self,
        kl_warmup_steps: int = 1000,
        feat_warmup_steps: int = 2000,
    ) -> None:
        """Configure progressive loss weight warmup.

        During early distillation the Student's representations are
        random, so feature matching noise would dominate.  We ramp:
           - kl_weight:      0.2 → full over kl_warmup_steps
           - feature_weight:  0  → full over feat_warmup_steps
        """
        self._kl_warmup_steps = max(kl_warmup_steps, 1)
        self._feat_warmup_steps = max(feat_warmup_steps, 1)
        log.info("Distillation curriculum: kl warmup=%d steps, feat warmup=%d steps.",
                 kl_warmup_steps, feat_warmup_steps)

    @property
    def effective_kl_weight(self) -> float:
        """Current KL weight after curriculum warmup."""
        frac = min(1.0, self._step / self._kl_warmup_steps)
        # Ramp from 20% → 100% of the configured weight
        return self._kl_weight_final * (0.2 + 0.8 * frac)

    @property
    def effective_feature_weight(self) -> float:
        """Current feature-matching weight after curriculum warmup."""
        frac = min(1.0, self._step / self._feat_warmup_steps)
        return self._feat_weight_final * frac

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
        replay_indices: list[int] | None = None,
    ) -> dict:
        """
        Perform one distillation step on a minibatch.

        Args:
            batch: Dict with input tensors (vision, audio, etc.)
                   ready to be passed to model.forward().
            replay_indices: Optional replay buffer indices for cache lookups.
                            When provided, cached Teacher outputs are reused
                            for transitions that were already forward-passed.

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
        valid_keys = {"vision", "audio", "code_x", "code_edge_index", "code_batch", "sensor", "hidden", "actions"}
        for k, v in batch.items():
            if k in valid_keys:
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v
            # Map 'action' key from replay to 'actions' key expected by model.forward()
            elif k == "action":
                if isinstance(v, torch.Tensor):
                    inputs["actions"] = v.to(device)
                else:
                    inputs["actions"] = v

        # Pass goal_embedding for goal-conditioned distillation
        if "goal_embedding" in batch:
            ge = batch["goal_embedding"]
            inputs["goal"] = ge.to(device) if isinstance(ge, torch.Tensor) else ge

        # Guard: at least one modality must survive collation.
        # When torch.stack fails for all modality keys (shape mismatch
        # across replay transitions), the batch has no encodable inputs
        # and the fusion layer would raise ValueError.
        _modality_keys = {"vision", "audio", "sensor", "code_x"}
        if not (_modality_keys & inputs.keys()):
            log.warning("Skipping distill step: batch has no modality inputs "
                        "(keys present: %s)", list(batch.keys()))
            return {"total": 0.0, "skipped": True}

        # Get Teacher targets (no grad, clean inputs)
        # Use soft-label cache when replay_indices are provided.
        # For each sample in the batch, check the cache first; only run
        # the full Teacher forward for cache-miss samples, then stitch
        # the results together.
        teacher_out = self._get_teacher_outputs(
            inputs, device, replay_indices,
        )

        # ── Asymmetric augmentation: Student sees augmented vision ──
        student_inputs = dict(inputs)
        if self._augmentor is not None and "vision" in student_inputs:
            student_inputs["vision"] = self._augmentor(student_inputs["vision"])

        # Get Student outputs
        if self.use_amp:
            with torch.amp.autocast("cuda"):
                student_out = self._staging_student(**student_inputs)
                loss = self._compute_loss(student_out, teacher_out)
        else:
            student_out = self._staging_student(**student_inputs)
            loss = self._compute_loss(student_out, teacher_out)

        # Backward
        self._optimizer.zero_grad()

        if self._scaler is not None:
            # Use guarded handling so that update() runs even if step() fails
            self._scaler.scale(loss["total"]).backward()
            scale_before = self._scaler.get_scale()
            step_failed = False
            try:
                try:
                    self._scaler.unscale_(self._optimizer)
                except RuntimeError as e:
                    # If unscale_ was already called, log and continue
                    log.warning("GradScaler.unscale_ warning during distill step %d: %s", self._step, e)
                torch.nn.utils.clip_grad_norm_(self._staging_student.parameters(), max_norm=1.0)
                try:
                    self._scaler.step(self._optimizer)
                except Exception as e:
                    step_failed = True
                    log.exception("GradScaler.step failed during distill step %d: %s", self._step, e)
            finally:
                try:
                    self._scaler.update()
                except Exception as e:
                    log.exception("GradScaler.update failed during distill step %d: %s", self._step, e)
            try:
                new_scale = self._scaler.get_scale()
                skip_scheduler = (new_scale < scale_before) or step_failed
            except Exception:
                skip_scheduler = True
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

        # Use curriculum-ramped weights instead of fixed values
        kl_w = self.effective_kl_weight
        feat_w = self.effective_feature_weight

        total = (
            kl_w * (action_loss + comm_kl)
            + feat_w * feat_loss
        )

        # VQ codebook index distillation (Phase E)
        # If both Student and Teacher have action tokenizers,
        # add cross-entropy loss on level-0 (coarse behavior mode) indices.
        if (
            "vq_indices" in teacher_out
            and "vq_indices" in student_out
        ):
            # Level-0 indices capture coarse behavior modes
            teacher_idx_0 = teacher_out["vq_indices"][0].detach()  # (B,) long
            # We need soft logits from the student's VQ encoder, not hard argmax.
            # Approximate: use the student's encoder output distances as logits.
            if (
                hasattr(self._staging_student, "action_tokenizer")
                and self._staging_student.action_tokenizer is not None
            ):
                s_z = self._staging_student.action_tokenizer.encoder(
                    student_out["action"]
                )
                s_dists = -torch.cdist(
                    s_z.unsqueeze(0),
                    self._staging_student.action_tokenizer.rvq.levels[0]
                    .embedding.weight.unsqueeze(0),
                ).squeeze(0)  # (B, K) negative distances → logits
                codebook_distill_loss = F.cross_entropy(s_dists, teacher_idx_0)
                total = total + 0.1 * codebook_distill_loss

        # VQ loss pass-through (auxiliary — keeps tokenizer codebook accurate)
        if "vq_loss" in student_out:
            total = total + 0.05 * student_out["vq_loss"]

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

    def bump_teacher_version(self) -> None:
        """Notify the engine that Teacher weights have been updated.

        Called by the learner thread after an optimizer step.
        Increments the version counter so that stale cache entries
        are no longer matched.
        """
        self._teacher_version += 1

    def _get_teacher_outputs(
        self,
        inputs: dict,
        device: str,
        replay_indices: list[int] | None,
    ) -> dict[str, torch.Tensor]:
        """Get Teacher outputs, using the soft-label cache when possible.

        For each sample in the batch:
          - If ``replay_indices`` is provided and the sample is in cache
            (same teacher version), reuse the cached output.
          - Otherwise, run a Teacher forward pass and store the result.

        This avoids redundant GPU work when high-priority replay
        transitions are re-sampled across distillation steps.
        """
        B = None
        for v in inputs.values():
            if isinstance(v, torch.Tensor) and v.dim() >= 1:
                B = v.shape[0]
                break

        # Fast path: no indices → full forward, no caching
        if replay_indices is None or B is None:
            return self._teacher_forward(inputs)

        # Check cache for each sample
        cached = [None] * B
        miss_positions: list[int] = []
        ver = self._teacher_version
        for i, idx in enumerate(replay_indices):
            cached[i] = self._label_cache.get(idx, ver)
            if cached[i] is None:
                miss_positions.append(i)

        # If all cached, reconstruct and return
        if not miss_positions:
            return self._stitch_cached(cached, device)

        # If all missed, full forward
        if len(miss_positions) == B:
            teacher_out = self._teacher_forward(inputs)
            # Cache individual samples
            for i, idx in enumerate(replay_indices):
                sample_out = {k: v[i] for k, v in teacher_out.items()
                              if isinstance(v, torch.Tensor) and v.dim() >= 1}
                self._label_cache.put(idx, ver, sample_out)
            return teacher_out

        # Partial cache hit: build sub-batch for misses
        miss_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == B:
                miss_inputs[k] = v[miss_positions]
            else:
                miss_inputs[k] = v

        miss_out = self._teacher_forward(miss_inputs)

        # Cache the miss outputs
        for local_i, global_i in enumerate(miss_positions):
            idx = replay_indices[global_i]
            sample_out = {k: v[local_i] for k, v in miss_out.items()
                          if isinstance(v, torch.Tensor) and v.dim() >= 1}
            self._label_cache.put(idx, ver, sample_out)
            cached[global_i] = {k: v[local_i].detach().cpu()
                                for k, v in miss_out.items()
                                if isinstance(v, torch.Tensor) and v.dim() >= 1}

        return self._stitch_cached(cached, device)

    def _teacher_forward(self, inputs: dict) -> dict[str, torch.Tensor]:
        """Run a full Teacher forward pass (thread-safe, no grad)."""
        with self._teacher_lock:
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast("cuda"):
                        return self.teacher(**inputs)
                else:
                    return self.teacher(**inputs)

    def _stitch_cached(
        self,
        cached: list[dict[str, torch.Tensor]],
        device: str,
    ) -> dict[str, torch.Tensor]:
        """Reconstruct a batched output dict from per-sample cached tensors."""
        # Use key intersection — cached hits (from LRU) and freshly-computed
        # miss entries may carry slightly different tensor keys (e.g. 0-dim
        # scalars filtered during caching).  Intersection avoids KeyError.
        key_sets = [set(c.keys()) for c in cached]
        keys = key_sets[0]
        for ks in key_sets[1:]:
            keys = keys & ks

        out: dict[str, torch.Tensor] = {}
        for k in keys:
            tensors = [c[k] for c in cached]
            try:
                out[k] = torch.stack(tensors).to(device)
            except RuntimeError:
                pass  # skip keys with mismatched shapes
        return out

    @property
    def step_count(self) -> int:
        return self._step
