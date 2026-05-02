"""
Learner thread — trains the Teacher model asynchronously.

Runs on the GPU in the background, consuming transitions from the
replay buffer, computing losses, and taking gradient steps.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_ai.config import TrainingConfig, RuntimeConfig, REBELConfig, DEFAULT_CONFIG
from baby_ai.learning.rewards import RewardComposer
from baby_ai.memory.replay_buffer import PrioritizedReplayBuffer
from baby_ai.learning.intrinsic import JEPACuriosity, LearningProgressEstimator
from baby_ai.learning.rebel import REBELLoss, GRPOLoss
from baby_ai.learning.successor import SuccessorLoss
from baby_ai.core.goals import HERGoalSampler
from baby_ai.memory.consolidation import Consolidator
from baby_ai.utils.logging import get_logger

log = get_logger("learner", log_file="learner.log")


class LearnerThread:
    """
    Background Teacher training thread.

    Continuously:
    1. Samples prioritized minibatches from replay
    2. Computes policy + JEPA curiosity + consolidation losses
    3. Updates Teacher weights
    4. Updates transition priorities based on learning progress

    Args:
        teacher: Teacher model to train.
        replay: Prioritized replay buffer.
        curiosity: JEPA curiosity module.
        consolidator: EWC consolidation module.
        config: Training hyperparameters.
        device: Device for training.
    """

    def __init__(
        self,
        teacher: nn.Module,
        replay: PrioritizedReplayBuffer,
        curiosity: JEPACuriosity,
        consolidator: Consolidator,
        config: TrainingConfig | None = None,
        device: str = "cuda",
        teacher_lock: threading.Lock | None = None,
        runtime_config: RuntimeConfig | None = None,
        distill_ready_callback=None,
        curiosity_proj: nn.Linear | None = None,
    ):
        self.teacher = teacher
        self.replay = replay
        self.curiosity = curiosity
        self.consolidator = consolidator
        self.config = config or DEFAULT_CONFIG.training
        self.device = device
        self._teacher_lock = teacher_lock or threading.Lock()
        self._runtime_config = runtime_config or RuntimeConfig()
        self._distill_ready_callback = distill_ready_callback
        self._last_distill_signal_at = 0
        self._distill_engine = None  # set by orchestrator for cache version bumps

        # REBEL RL loss (Phase D)
        rebel_cfg = getattr(teacher, '_rebel_config', None)
        if rebel_cfg is None:
            rebel_cfg = REBELConfig()
        self._rebel_enabled = rebel_cfg.enabled
        self._rebel_loss_fn = REBELLoss(
            beta=rebel_cfg.beta,
            reward_clip=rebel_cfg.reward_clip,
            entropy_weight=getattr(rebel_cfg, 'entropy_weight', 0.0),
        ).to(device) if rebel_cfg.enabled else None
        self._rebel_value_loss_weight = rebel_cfg.value_loss_weight

        # GRPO loss (alternative to REBEL pairing — group-relative rewards)
        self._use_grpo = getattr(rebel_cfg, 'use_grpo', False)
        self._grpo_loss_fn = GRPOLoss(
            beta=rebel_cfg.beta,
            entropy_weight=getattr(rebel_cfg, 'entropy_weight', 0.01),
        ).to(device) if rebel_cfg.enabled else None

        # Successor features loss (if teacher has a successor_head attribute)
        _shead = getattr(teacher, 'successor_head', None)
        self._successor_loss_fn = SuccessorLoss().to(device) if _shead is not None else None

        # HER goal relabeling sampler
        self._her_sampler = HERGoalSampler(
            her_ratio=0.5,
            strategy='future',
        )

        # PCGrad: project conflicting task gradients to reduce interference
        # Only active when grad accumulation = 1 (multi-backward incompatible)
        self._use_pcgrad = getattr(self.config, 'use_pcgrad', False)

        # Optimizer with parameter-group-specific learning rates
        encoder_params = []
        core_params = []
        policy_params = []
        for name, param in teacher.named_parameters():
            if "encoder" in name or "fusion" in name:
                encoder_params.append(param)
            elif "temporal" in name:
                core_params.append(param)
            else:
                policy_params.append(param)

        # AdamW: weight decay prevents weight explosion from correlated RL replay
        # inputs (value heads see near-identical states during on-policy stretches).
        # Encoders use a lower decay since their features should be stable.
        #
        # Try Schedule-Free AdamW (Defazio et al., 2024): eliminates LR scheduling
        # by maintaining a dual iterate.  Falls back to standard AdamW if not installed.
        try:
            from schedulefree import AdamWScheduleFree
            self.optimizer = AdamWScheduleFree([
                {"params": encoder_params, "lr": self.config.encoder_lr, "initial_lr": self.config.encoder_lr, "weight_decay": 1e-4},
                {"params": core_params, "lr": self.config.core_lr, "initial_lr": self.config.core_lr, "weight_decay": 0.0},
                {"params": policy_params, "lr": self.config.policy_lr, "initial_lr": self.config.policy_lr, "weight_decay": 0.01},
            ])
            self._optimizer_is_sf = True
            log.info("Using Schedule-Free AdamW optimizer.")
        except ImportError:
            self.optimizer = torch.optim.AdamW([
                {"params": encoder_params, "lr": self.config.encoder_lr, "initial_lr": self.config.encoder_lr, "weight_decay": 1e-4},
                {"params": core_params, "lr": self.config.core_lr, "initial_lr": self.config.core_lr, "weight_decay": 0.0},
                {"params": policy_params, "lr": self.config.policy_lr, "initial_lr": self.config.policy_lr, "weight_decay": 0.01},
            ])
            self._optimizer_is_sf = False

        # Remember the *configured* base LRs so enable_cosine_schedule()
        # can always compute the correct peak, even after a checkpoint
        # load has overwritten initial_lr with a peak value from a
        # previous offline run.
        self._config_base_lrs = [
            self.config.encoder_lr,
            self.config.core_lr,
            self.config.policy_lr,
        ]

        # No warmup scheduler — constant LR controlled by the GUI.
        # The live LR is read from the control panel each optimizer step.
        # For offline training, enable_cosine_schedule() adds a proper
        # warmup + cosine decay schedule.
        self._lr_scheduler = None

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp else None

        # Learning progress tracker
        self.lp_estimator = LearningProgressEstimator()

        # Projection: Student fused (512-d) → Teacher fused (1024-d).
        # Replay stores fused/next_fused from the Student model, but the
        # Teacher's world-model target_encoder expects Teacher-sized inputs.
        # This is the SAME nn.Linear instance created by the Orchestrator
        # (orchestrator.curiosity_proj) so that both the main loop's
        # real-time curiosity computation and the learner's training
        # share the same weights.  Without this, the main loop would
        # use an untrained random projection for intrinsic reward.
        self._fused_proj = curiosity_proj

        # Add _fused_proj parameters to optimizer so they actually train
        if self._fused_proj is not None:
            self.optimizer.add_param_group({
                "params": list(self._fused_proj.parameters()),
                "lr": self.config.core_lr,
                "initial_lr": self.config.core_lr,
                "weight_decay": 0.0,
            })
            self._config_base_lrs.append(self.config.core_lr)

        # ── Muon optimizer for 2-D SSM weight matrices ───────────────────
        # Applied only to teacher.temporal's 2-D weight matrices.  These
        # are removed from the main AdamW optimizer to avoid double updates.
        # Falls back gracefully if muon.py is unavailable.
        self.muon: object | None = None
        try:
            from baby_ai.utils.muon import Muon
            _muon_params: list[torch.Tensor] = []
            _muon_ids: set[int] = set()
            for _n, _p in self.teacher.temporal.named_parameters():
                if _p.dim() >= 2 and _p.requires_grad:
                    _muon_params.append(_p)
                    _muon_ids.add(id(_p))
            if _muon_params:
                # Remove these params from main optimizer groups (avoid double update)
                for _pg in self.optimizer.param_groups:
                    _pg["params"] = [_p for _p in _pg["params"] if id(_p) not in _muon_ids]
                self.muon = Muon(_muon_params, lr=self.config.core_lr, momentum=0.95)
                log.info("Muon optimizer active for %d SSM weight matrices.", len(_muon_params))
        except Exception as _e:
            log.debug("Muon optimizer not initialized: %s", _e)

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._step = 0
        self._accum_count = 0
        self._consecutive_nan = 0  # steps with NaN loss in a row

        # Live reward recomposition — when set, sampled transitions are
        # re-weighted with the current UI weights/toggles so that user
        # changes take effect immediately instead of after a full buffer
        # turnover (~14 hours at 500 K capacity / 10 steps per second).
        self._reward_weights = None   # RewardWeightsState (set by main.py)
        self._toggle_state = None     # RewardToggleState  (set by main.py)

        # Event-driven coordination: the main loop or the distill thread
        # can signal that new data is available or a threshold is crossed,
        # eliminating fixed-interval sleep polling.
        self._data_ready = threading.Event()
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the learner thread."""
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self.teacher.to(self.device).train()
        # Schedule-Free AdamW requires train() to be called on the optimizer
        if self._optimizer_is_sf:
            self.optimizer.train()

        # Phase A: optional torch.compile for Teacher
        if self._runtime_config.compile_teacher and hasattr(torch, 'compile'):
            # Teacher uses 'default' mode to avoid CUDA graph conflicts with
            # dynamic replay shapes.  Config compile_mode is used for Student;
            # Teacher always gets 'default' for safety.
            log.info("Compiling Teacher model with torch.compile(mode='default')...")
            self.teacher = torch.compile(self.teacher, mode="default", fullgraph=False)

        self._thread = threading.Thread(target=self._loop, daemon=True, name="LearnerThread")
        self._thread.start()
        log.info("Learner thread started.")

    def stop(self) -> None:
        """Stop the learner thread."""
        self._running = False
        self._stop_event.set()      # wake the loop immediately
        self._data_ready.set()      # unblock any wait
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        log.info("Learner thread stopped after %d steps.", self._step)

    def notify_data_ready(self) -> None:
        """Signal that new transitions are available in the replay buffer.

        Called by the main loop after adding experience.  Wakes the
        learner thread immediately instead of waiting for the next
        sleep-poll cycle.
        """
        self._data_ready.set()

    def _loop(self) -> None:
        """Main training loop (event-driven)."""
        while self._running:
            # Pause training while record-only or disable-learning mode is active.
            try:
                from baby_ai.ui.control_panel import get_record_only, get_learning_disabled
                if get_record_only() or get_learning_disabled():
                    self._stop_event.wait(timeout=1.0)
                    continue
            except ImportError:
                pass

            # Wait until we have enough transitions
            if self.replay.size < self.config.micro_batch_size * 2:
                # Block until signalled (with timeout so we can check _running)
                self._data_ready.wait(timeout=1.0)
                self._data_ready.clear()
                continue

            try:
                self._train_step()
            except Exception as e:
                log.error("Learner error at step %d: %s", self._step, e, exc_info=True)
                # Free memory after OOM or allocation failures.
                # torch.cuda.empty_cache() releases cached blocks and
                # gc.collect() frees Python-side dead objects that hold
                # tensor references (e.g. stale closures, exception
                # tracebacks holding onto frame locals).
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                self._step += 1  # always advance to avoid infinite retry loops
                self._stop_event.wait(timeout=1.0)

            # Yield to other threads (short, bounded sleep)
            self._stop_event.wait(timeout=max(self.config.micro_batch_size * 0.001, 0.001))

    def _train_step(self) -> None:
        """Perform a single training step."""
        # Every 4 steps, use HER-enriched batch for goal-conditioned training;
        # otherwise use standard prioritized sampling.
        if self._step % 4 == 0:
            try:
                transitions, weights, indices = self.replay.sample_her(
                    self.config.micro_batch_size, device=self.device
                )
                # Relabel goals using the HER sampler (future-strategy)
                transitions = self._her_sampler.relabel(
                    transitions,
                    goal_proj=lambda s: s,  # fused is already goal-compatible
                )
            except Exception:
                transitions, weights, indices = self.replay.sample(
                    self.config.micro_batch_size, device=self.device
                )
        else:
            transitions, weights, indices = self.replay.sample(
                self.config.micro_batch_size, device=self.device
            )

        # Recompose rewards with current UI weights/toggles so that
        # slider and checkbox changes take effect immediately.
        self._recompose_rewards(transitions)

        # Build batch tensors from transitions
        batch = self._collate_transitions(transitions)
        weights_t = torch.from_numpy(weights).float().to(self.device)

        # _collate_transitions may drop shape-inconsistent samples, making the
        # actual batch smaller than micro_batch_size.  Trim weights and indices
        # to match so downstream losses (value_loss, rebel_loss) don't crash
        # with "size of tensor a != tensor b" runtime errors on dim 0.
        actual_B = next(
            (v.shape[0] for v in batch.values() if isinstance(v, torch.Tensor)),
            len(weights),
        )
        if actual_B < len(weights):
            weights_t = weights_t[:actual_B]
            indices = indices[:actual_B]

        # Guard: skip step if collate dropped ALL samples
        if actual_B == 0:
            log.warning("Empty batch at step %d after collate; skipping.", self._step)
            self._step += 1
            return

        # Forward pass with AMP
        # Acquire the teacher lock to prevent the distillation thread
        # from running a concurrent forward on the same SSM
        # (whose internal reserve buffers are not thread-safe).
        with self._teacher_lock:
            if self.config.use_amp:
                with torch.amp.autocast("cuda"):
                    loss_dict = self._compute_loss(batch, weights_t)
            else:
                loss_dict = self._compute_loss(batch, weights_t)

            total_loss = loss_dict["total"]

            # Guard: skip backward when the loss has no computation graph.
            # This happens when the batch lacks "action"/"reward" keys
            # (e.g. incomplete transitions) and EWC isn't initialized,
            # leaving loss as a detached scalar 0.
            if total_loss.grad_fn is None and not total_loss.requires_grad:
                log.debug(
                    "Skipping backward at step %d: loss has no grad_fn "
                    "(batch keys: %s)", self._step, list(batch.keys()),
                )
                self._step += 1
                return

            # Guard: skip backward when loss is NaN or Inf to prevent
            # poisoning the optimizer state (Adam momentum/variance).
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                self._consecutive_nan += 1
                log.warning(
                    "Skipping backward at step %d: loss is %s "
                    "(batch keys: %s) [consecutive=%d]",
                    self._step,
                    "NaN" if torch.isnan(total_loss) else "Inf",
                    list(batch.keys()),
                    self._consecutive_nan,
                )
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_count = 0
                # After 10 consecutive NaN/Inf steps, the optimizer's Adam
                # momentum terms likely contain NaN propagated from a corrupt
                # forward pass stored in replay.  Reset state to break the
                # cascade — weights stay intact but momentum is re-zeroed.
                if self._consecutive_nan >= 10:
                    log.warning(
                        "NaN cascade detected (%d consecutive) — resetting "
                        "optimizer state to break the loop",
                        self._consecutive_nan,
                    )
                    for pg in self.optimizer.param_groups:
                        for p in pg["params"]:
                            state = self.optimizer.state.get(p)
                            if state:
                                for k, v in state.items():
                                    if isinstance(v, torch.Tensor) and (
                                        torch.isnan(v).any() or torch.isinf(v).any()
                                    ):
                                        state[k] = torch.zeros_like(v)
                    if self.scaler is not None:
                        self.scaler = torch.amp.GradScaler("cuda")
                    self._consecutive_nan = 0
                self._step += 1
                return
            else:
                self._consecutive_nan = 0

            # Backward with gradient accumulation
            # PCGrad applies multiple per-task backward passes and projects
            # conflicting gradients.  Only active when:
            #   1. use_pcgrad is set in training config
            #   2. gradient_accumulation_steps == 1 (multi-backward incompatible)
            #   3. There are ≥2 separate task losses in the dict
            _task_losses = loss_dict.get("task_losses", [])
            _use_pcgrad_now = (
                self._use_pcgrad
                and self.config.gradient_accumulation_steps == 1
                and len(_task_losses) >= 2
            )
            if _use_pcgrad_now:
                _trainable_params = [p for p in self.teacher.parameters() if p.requires_grad]
                self._pcgrad_backward(_task_losses, _trainable_params, self.scaler)
                # Also backward the regularizer terms (MoE aux + EWC) on top
                _reg = total_loss - sum(_task_losses)
                if _reg.grad_fn is not None:
                    if self.scaler is not None:
                        self.scaler.scale(_reg).backward()
                    else:
                        _reg.backward()
            elif self.scaler is not None:
                self.scaler.scale(total_loss / self.config.gradient_accumulation_steps).backward()
            else:
                (total_loss / self.config.gradient_accumulation_steps).backward()

        self._accum_count += 1

        if self._accum_count >= self.config.gradient_accumulation_steps:
            # Gradient clipping — essential for RL stability.
            # Without this, rare reward spikes create enormous
            # gradients that destabilise all model weights.
            # Clip gradients and perform optimizer step.  Use a guarded
            # path for AMP's GradScaler so that `update()` is always
            # called even if `step()` raises, preventing repeated
            # `unscale_()` errors on subsequent iterations.
            if self.scaler is not None:
                _step_ok = False
                try:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.teacher.parameters(), max_norm=1.0
                    )
                    try:
                        self.scaler.step(self.optimizer)
                        # Muon uses unscaled gradients (already unscaled above)
                        if self.muon is not None:
                            try:
                                self.muon.step()
                            except Exception as _me:
                                log.debug("Muon step (scaler path) failed: %s", _me)
                        _step_ok = True
                    except Exception as e:
                        log.error("GradScaler.step failed at learner step %d: %s", self._step, e)
                except RuntimeError as e:
                    log.error("GradScaler.unscale_ failed at learner step %d: %s", self._step, e)
                finally:
                    try:
                        self.scaler.update()
                    except Exception as e:
                        log.error("GradScaler.update failed at learner step %d: %s", self._step, e)
                    if not _step_ok:
                        # The scaler is in a corrupt state (e.g. device
                        # mismatch or double-unscale).  Replace it with a
                        # fresh one so the next step starts clean instead
                        # of entering an infinite "unscale_() already called"
                        # error loop.
                        log.warning("Resetting GradScaler after failure at step %d", self._step)
                        self.scaler = torch.amp.GradScaler("cuda")
                        self.optimizer.zero_grad(set_to_none=True)
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.teacher.parameters(), max_norm=1.0
                )
                self.optimizer.step()
                # Muon optimizer steps independently (handles its own params)
                if self.muon is not None:
                    try:
                        self.muon.step()
                    except Exception as _me:
                        log.debug("Muon step failed: %s", _me)

            # ── Sync learning rate ─────────────────────────────────
            if self._lr_scheduler is not None:
                # Offline mode: use the cosine schedule.
                self._lr_scheduler.step()
            else:
                # Online mode: sync LR from GUI slider.
                # The GUI provides a single LR value.  To preserve the
                # per-group LR ratios (encoder_lr : core_lr : policy_lr),
                # scale each group proportionally.
                try:
                    from baby_ai.ui.control_panel import get_live_lr
                    gui_lr = get_live_lr()
                    base_lr = self.config.core_lr or 1e-4
                    scale = gui_lr / base_lr
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = pg.get("initial_lr", base_lr) * scale
                except Exception:
                    pass  # GUI not available yet

            self.optimizer.zero_grad()
            self._accum_count = 0

            # Update JEPA target encoder via EMA after each optimizer step
            if hasattr(self.teacher, 'predictive') and hasattr(self.teacher.predictive, 'update_target_encoder'):
                self.teacher.predictive.update_target_encoder()

            # Bump teacher version so the distill engine's soft-label
            # cache knows that cached outputs are now stale.
            if self._distill_engine is not None:
                self._distill_engine.bump_teacher_version()

        # Update replay priorities with per-sample value loss
        _psl = loss_dict.get("per_sample_loss")
        if _psl is not None and isinstance(_psl, torch.Tensor) and _psl.numel() >= len(indices):
            new_priorities = [max(float(_psl[i]), 1e-6) for i in range(len(indices))]
        else:
            new_priorities = [0.1] * len(indices)
        self.replay.update_priorities(indices, new_priorities)

        # Periodic consolidation -- recompute Fisher Information so the
        # EWC penalty (which is already part of _compute_loss every step)
        # actually protects important weights.
        if self._step > 0 and self._step % self.config.consolidation_every_n_steps == 0:
            # Save & restore GradScaler state so that Fisher computation
            # (which runs its own forward/backward passes) cannot corrupt
            # the scaler's internal bookkeeping and cause "backward through
            # the graph a second time" on the next training step.
            _scaler_snapshot = (
                self.scaler.state_dict() if self.scaler is not None else None
            )

            try:
                fisher_loader = self._make_fisher_loader(num_samples=200)
                self.consolidator.update_fisher(
                    self.teacher, fisher_loader,
                    num_samples=200, device=self.device,
                )
                log.info("Fisher Information recomputed at step %d.", self._step)
            except Exception as e:
                log.warning("Fisher computation failed at step %d: %s",
                            self._step, e, exc_info=True)
            finally:
                # ── Unconditional cleanup after Fisher computation ──
                # 1. Force-restore training mode.  compute_fisher() sets
                #    model.eval(); if it throws before model.train(), the
                #    MoE aux_loss guard ("if self.training") will skip
                #    updates on subsequent steps, leaving a stale tensor
                #    whose graph was already freed → double-backward.
                self.teacher.train()

                # 2. Clear MoE _aux_loss cache.  Even if model.train() is
                #    restored, a Fisher eval-mode forward wouldn't have
                #    overwritten _aux_loss.  The stale tensor still holds
                #    a reference to a freed computation graph.
                for m in self.teacher.modules():
                    if hasattr(m, '_aux_loss'):
                        m._aux_loss = None

            # Restore scaler to a known-good state and wipe gradients
            # so the next _train_step starts completely clean.
            if self.scaler is not None and _scaler_snapshot is not None:
                self.scaler.load_state_dict(_scaler_snapshot)
            self.optimizer.zero_grad(set_to_none=True)

            # Log the current EWC penalty (under no_grad -- we only need
            # the scalar value, not a computation graph).
            with torch.no_grad():
                ewc_snapshot = self.consolidator.consolidation_loss(self.teacher)
            log.info("Consolidation at step %d, EWC loss snapshot: %.4f",
                     self._step, ewc_snapshot.item())

        self._step += 1

        # ── Signal distill thread when step threshold crossed ────
        distill_interval = getattr(self.config, 'distill_every_n_steps', 100)
        if (self._distill_ready_callback is not None
                and self._step - self._last_distill_signal_at >= distill_interval):
            self._distill_ready_callback()
            self._last_distill_signal_at = self._step

        # ── Interleave n-step sequence training every 4 steps ────
        # This supplements the single-transition loss with GAE-based
        # multi-step value targets, giving the value head temporal
        # credit assignment without slowing down normal training.
        if self._step % 4 == 0 and self.replay.size >= self.config.micro_batch_size:
            try:
                sequences, seq_weights, seq_indices = self.replay.sample_sequence(
                    batch_size=max(self.config.micro_batch_size // 4, 2),
                    seq_len=8,
                    device=self.device,
                )
                for seq, w in zip(sequences, seq_weights):
                    self.train_on_sequence(seq, weight=float(w))
            except Exception as e:
                log.debug("N-step sequence training skipped: %s", e)

        # Log periodically
        if self._step % 100 == 0:
            # Gather learning rates for diagnostics
            current_lrs = [pg['lr'] for pg in self.optimizer.param_groups]
            lr_str = '/'.join(f'{lr:.2e}' for lr in current_lrs)
            log.info(
                "Step %d | loss=%.4f | replay=%d | priority=%.2f | lr=%s",
                self._step, total_loss.item(), self.replay.size,
                self.replay.tree.total, lr_str,
            )

    def _compute_loss(self, batch: dict, weights: torch.Tensor) -> dict:
        """Compute combined training loss."""
        # Teacher forward — pass actions so the policy head can compute
        # its denoising loss (applies to both Diffusion and Flow Matching).
        _modality_keys = {"vision", "audio", "code_x", "code_edge_index", "code_batch", "sensor"}
        forward_kwargs = {
            k: v for k, v in batch.items()
            if k in _modality_keys
        }
        if "action" in batch:
            forward_kwargs["actions"] = batch["action"]

        # Pass goal_embedding from replay for goal-conditioned training
        if "goal_embedding" in batch:
            forward_kwargs["goal"] = batch["goal_embedding"]

        # Guard: at least one modality tensor must be present for encode()
        if not (forward_kwargs.keys() & _modality_keys):
            log.warning("Step %d: no modality tensors in batch (keys=%s); skipping.",
                        self._step, list(batch.keys()))
            return {"total": torch.tensor(0.0, device=self.device), "task_losses": {}}

        outputs = self.teacher(**forward_kwargs)

        value = outputs["value"]

        # Soft-clamp value predictions via tanh squashing to [-7, 7].
        value = 7.0 * torch.tanh(value / 7.0)

        # Anchor loss to the computation graph via a zero derived from
        # model output.  This ensures loss.grad_fn is set even when no
        # policy branch fires, so backward() won't crash.
        loss = (value.sum() * 0.0).squeeze()
        per_sample_value_loss: torch.Tensor | None = None

        if "action" in batch and "reward" in batch:
            actions = batch["action"]
            rewards = batch["reward"]
            v_pred = value.squeeze(-1)

            # ── V-trace value targets (IMPALA-style off-policy correction) ──
            # If we stored log_prob + value from the behavior policy at
            # collection time, compute importance-weighted V-trace targets.
            # rho = clip(pi / mu, rho_bar=1.0) — single-step V-trace.
            # v_target = V_mu + rho * (r - V_mu)
            # This corrects for the off-policy gap between the current
            # policy π and the behavior policy μ that collected the data.
            if "log_prob" in batch and "value" in batch:
                mu_log_prob = batch["log_prob"].float()
                mu_value = batch["value"].float().squeeze(-1)
                with torch.no_grad():
                    try:
                        # Get current pi_log_prob from the live policy
                        pi_lp, _, _ = self.teacher.policy.evaluate(
                            outputs["core_state"], actions
                        )
                        rho = (pi_lp.detach() - mu_log_prob).exp().clamp(max=1.0)
                    except Exception:
                        rho = torch.ones_like(rewards)
                    v_target = (mu_value + rho * (rewards - mu_value)).clamp(-7.0, 7.0)
            else:
                # No behavior log_prob stored — fall back to raw reward target
                v_target = rewards.clamp(-7.0, 7.0)

            # Value loss against V-trace targets
            per_sample_value_loss = F.smooth_l1_loss(v_pred, v_target, reduction='none').detach()
            value_loss = (F.smooth_l1_loss(v_pred, v_target, reduction='none') * weights).mean()

            if self._rebel_enabled and self._rebel_loss_fn is not None:
                # ── REBEL / GRPO paired loss ──
                rebel_loss = self._compute_rebel_loss(outputs, batch, weights)
                loss = rebel_loss + self._rebel_value_loss_weight * value_loss
            else:
                # ── Legacy: reward-weighted denoising loss ──
                advantage = (v_target - v_pred.detach())
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
                advantage = torch.clamp(advantage, -3.0, 3.0)
                denoising_loss = outputs.get("denoising_loss", torch.tensor(0.0, device=value.device))
                policy_loss = torch.clamp(denoising_loss, max=10.0)
                loss = policy_loss + 0.5 * value_loss

        # JEPA world-model loss
        if "next_fused" in batch and "action" in batch:
            # Project Student-dim fused → Teacher-dim before world model
            next_fused = batch["next_fused"]
            if self._fused_proj is not None:
                next_fused = self._fused_proj(next_fused)
            wm_out = self.teacher.predictive(
                outputs["core_state"], next_fused, batch["action"],
            )
            dynamics_loss = torch.clamp(wm_out["dynamics_loss"], max=10.0)
            kl_loss = torch.clamp(wm_out["kl_loss"], max=10.0)
            loss = loss + 0.1 * dynamics_loss + 0.01 * kl_loss

            # ── Counterfactual causal auxiliary loss ─────────────────────
            cf_loss = wm_out.get("counterfactual_loss")
            if cf_loss is not None:
                loss = loss + 0.01 * torch.clamp(cf_loss, max=5.0)

            # ── Dyna imagined rollouts (every 8 steps to limit overhead) ──
            if (self._step % 8 == 0 and "action" in batch
                    and getattr(self.config, "save_imagined_rollouts", True)):
                try:
                    with torch.no_grad():
                        start_lat = self.teacher.predictive.encode_state(
                            outputs["core_state"][:4]  # small imagined batch
                        )
                    imagined = self.teacher.predictive.imagine_rollout(
                        start_lat, self.teacher.policy, horizon=3
                    )
                    # Add imagined transitions to replay at low priority
                    for it in imagined:
                        self.replay.add(it, priority=0.1)
                except Exception:
                    pass  # non-critical; skip on any error

        # ── CoT communication signal ─────────────────────────────────────
        # Train comm head to predict next subgoal token conditioned on
        # comm_logits from the current step.  Acts as a chain-of-thought
        # scratchpad: forces the comm head to encode useful intermediate
        # state about the current goal.
        if "comm_logits" in outputs and "action" in batch:
            comm_logits = outputs["comm_logits"]  # (B, vocab_size)
            # Self-prediction: next comm token ≈ argmax of current logits
            # Use stop-grad target to avoid trivial collapse
            with torch.no_grad():
                pseudo_target = comm_logits.argmax(dim=-1)  # (B,)
            # Only keep predictions where pseudo-confidence is high
            conf = comm_logits.softmax(dim=-1).max(dim=-1).values  # (B,)
            high_conf = conf > 0.3
            if high_conf.any():
                cot_loss = F.cross_entropy(
                    comm_logits[high_conf], pseudo_target[high_conf]
                )
                loss = loss + 0.01 * torch.clamp(cot_loss, max=5.0)

        # ── Successor features loss ──────────────────────────────────────
        # NOTE: Successor TD requires (s_t, s_{t+1}) from the same episode.
        # Random replay batches have no temporal ordering.  We gate on
        # episode_id and step metadata stored by sample_her(); when those
        # are absent we skip, since random adjacent indices are meaningless.
        if self._successor_loss_fn is not None:
            shead = getattr(self.teacher, 'successor_head', None)
            if shead is not None and "action" in batch:
                cs = outputs["core_state"]
                episode_ids = batch.get("episode_id")
                steps = batch.get("step")
                if episode_ids is not None and steps is not None:
                    # Build valid (t, t+1) pairs within the same episode
                    ep = episode_ids.long() if isinstance(episode_ids, torch.Tensor) else torch.tensor(episode_ids, device=self.device)
                    st = steps.long() if isinstance(steps, torch.Tensor) else torch.tensor(steps, device=self.device)
                    B_sf = cs.size(0)
                    # Check adjacent items: same episode AND consecutive steps
                    same_ep = ep[:-1] == ep[1:]
                    consecutive = st[1:] - st[:-1] == 1
                    valid = same_ep & consecutive
                    if valid.any():
                        idx = valid.nonzero(as_tuple=True)[0]
                        sf_out_t = shead(cs[idx])
                        sf_out_next = shead(cs[idx + 1].detach())
                        dones = torch.zeros(idx.size(0), device=self.device)
                        sf_loss = self._successor_loss_fn(
                            sf_out_t["psi"], sf_out_t["phi"],
                            sf_out_next["psi"], dones,
                        )
                        loss = loss + 0.05 * torch.clamp(sf_loss, max=5.0)

        # Track task losses for PCGrad (before regularizers are added)
        task_losses: list[torch.Tensor] = []
        if loss.grad_fn is not None:
            task_losses.append(loss)  # policy + value loss

        # MoE load-balancing auxiliary loss (from Jamba temporal core)
        if hasattr(self.teacher, 'temporal') and hasattr(self.teacher.temporal, 'aux_loss'):
            moe_loss = self.teacher.temporal.aux_loss
            if moe_loss.requires_grad or moe_loss.item() > 0:
                loss = loss + moe_loss

        # EWC penalty (regularizer — excluded from PCGrad task set)
        ewc_penalty = self.consolidator.consolidation_loss(self.teacher)
        ewc_penalty = torch.clamp(ewc_penalty, max=5.0)
        loss = loss + ewc_penalty

        return {"total": loss, "task_losses": task_losses, "per_sample_loss": per_sample_value_loss}

    @staticmethod
    def _pcgrad_backward(
        task_losses: list,
        params: list,
        scaler=None,
    ) -> None:
        """
        PCGrad: Project Conflicting Gradients (Yu et al., 2020).

        For each pair of task gradients whose cosine similarity is negative
        (i.e. they conflict), project each gradient onto the plane orthogonal
        to the other.  Sum the projected gradients and assign to param.grad.

        Skips multiple backward passes when gradient accumulation or AMP
        scaler is active (requires retain_graph which is incompatible).
        Falls back to normal summed backward in those cases.
        """
        n = len(task_losses)
        if n <= 1:
            loss = task_losses[0] if n == 1 else None
            if loss is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            return

        # ── Per-task backward with retain_graph ───────────────────────────
        # Each backward accumulates into p.grad; scaler-unscaling is done
        # once at the end so the raw scale factor is consistent.
        task_grad_vecs: list[torch.Tensor] = []
        for idx, t_loss in enumerate(task_losses):
            # Zero grad before each task backward so we capture only this task
            for p in params:
                p.grad = None
            retain = (idx < n - 1)  # last loss doesn't need retain
            if scaler is not None:
                scaler.scale(t_loss).backward(retain_graph=retain)
                # Temporarily unscale to work in true-gradient space
                inv_scale = 1.0 / (scaler.get_scale() + 1e-8)
                flat = torch.cat([
                    (p.grad * inv_scale).detach().flatten() if p.grad is not None
                    else torch.zeros(p.numel(), device=p.device)
                    for p in params
                ])
            else:
                t_loss.backward(retain_graph=retain)
                flat = torch.cat([
                    p.grad.detach().flatten() if p.grad is not None
                    else torch.zeros(p.numel(), device=p.device)
                    for p in params
                ])
            task_grad_vecs.append(flat)

        # ── PCGrad projection ─────────────────────────────────────────────
        projected = [g.clone() for g in task_grad_vecs]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                gj_norm_sq = task_grad_vecs[j].dot(task_grad_vecs[j]) + 1e-12
                dot = projected[i].dot(task_grad_vecs[j])
                if dot < 0:
                    projected[i] = projected[i] - (dot / gj_norm_sq) * task_grad_vecs[j]

        # ── Assign summed projected gradients back to params ──────────────
        summed = sum(projected)
        offset = 0
        inv_s = 1.0 / (scaler.get_scale() + 1e-8) if scaler is not None else 1.0
        for p in params:
            n_elem = p.numel()
            chunk = summed[offset: offset + n_elem].view_as(p)
            if scaler is not None:
                # Re-apply scale factor so the scaler.unscale_() call in the
                # training loop produces the correct unscaled gradient.
                p.grad = (chunk / inv_s).contiguous()
            else:
                p.grad = chunk.contiguous()
            offset += n_elem

    def _compute_rebel_loss(
        self, outputs: dict, batch: dict, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reinforcement learning policy loss.

        Supports two modes:
          • GRPO: Group-relative policy optimisation (all transitions,
            group-normalised advantages).  Enabled when rebel_cfg.use_grpo=True.
          • REBEL: Paired preference loss (half-batch pairing).  Default.

        Falls back to denoising_loss when the batch is too small.
        """
        actions = batch["action"]
        rewards = batch["reward"]
        core_state = outputs["core_state"]
        B = actions.size(0)

        if B < 4:
            return outputs.get("denoising_loss", torch.tensor(0.0, device=actions.device))

        # ── GRPO mode (group-relative reward normalisation) ───────────────
        if self._use_grpo and self._grpo_loss_fn is not None:
            return self._grpo_loss_fn(core_state, actions, rewards, self.teacher.policy)

        # ── REBEL paired mode ─────────────────────────────────────────────
        half = B // 2
        s1, s2 = core_state[:half], core_state[half:half * 2]
        a1, a2 = actions[:half], actions[half:half * 2]
        r1, r2 = rewards[:half], rewards[half:half * 2]

        w_mask = r1 >= r2
        state_pairs = s1
        action_w = torch.where(w_mask.unsqueeze(-1), a1, a2)
        action_l = torch.where(w_mask.unsqueeze(-1), a2, a1)
        reward_w = torch.where(w_mask, r1, r2)
        reward_l = torch.where(w_mask, r2, r1)

        rebel_loss = self._rebel_loss_fn(
            state=state_pairs,
            action_w=action_w,
            action_l=action_l,
            reward_w=reward_w,
            reward_l=reward_l,
            policy=self.teacher.policy,
        )
        return rebel_loss

    def enable_cosine_schedule(
        self,
        total_steps: int,
        warmup_steps: int = 500,
        peak_lr_mult: float = 6.0,
        eta_min_mult: float = 0.1,
    ) -> None:
        """Enable cosine-annealing LR schedule for offline training.

        Ramps LR from 10% → peak over ``warmup_steps``, then decays
        via cosine annealing to ``eta_min_mult * base_lr``.

        Uses a single LambdaLR to avoid SequentialLR double-stepping
        bugs that cause the warmup phase to overshoot the peak LR.

        Args:
            total_steps: Total optimizer steps expected (epochs * steps_per_epoch).
            warmup_steps: Linear warmup phase length.
            peak_lr_mult: Multiplier on the configured LR for peak value.
            eta_min_mult: Fraction of peak LR for the cosine floor.
        """
        import math as _math

        # Reset optimizer state (momentum/variance) so stale buffers
        # from online training at a much lower LR don't cause NaN when
        # the cosine schedule ramps up to peak_lr_mult × base_lr.
        self.optimizer.state.clear()

        # Also reset GradScaler — the loaded scale factor may be tuned
        # for the old LR regime and cause immediate overflow at the
        # higher offline peak LR.
        if self.scaler is not None:
            self.scaler = torch.amp.GradScaler("cuda")

        # Set each param group's LR to peak (config_base * mult).
        # Use the originally-configured base LRs, NOT initial_lr from
        # the param group (which may have been overwritten by a previous
        # offline run and persisted in the checkpoint).
        peak_lrs: list[float] = []
        for i, pg in enumerate(self.optimizer.param_groups):
            base = self._config_base_lrs[i] if i < len(self._config_base_lrs) else pg.get("initial_lr", pg["lr"])
            peak = base * peak_lr_mult
            pg["lr"] = peak
            pg["initial_lr"] = peak
            peak_lrs.append(peak)

        # Build a single lambda that implements warmup + cosine decay
        # so there are no SequentialLR init-stepping quirks.
        _warmup = warmup_steps
        _total = total_steps
        _eta_min = eta_min_mult  # fraction of peak

        def _lr_lambda(step: int) -> float:
            if step < _warmup:
                # Linear warmup: 10% → 100% of peak
                return 0.1 + 0.9 * step / max(_warmup, 1)
            # Cosine decay: 100% → eta_min_mult of peak
            progress = (step - _warmup) / max(_total - _warmup, 1)
            return _eta_min + (1.0 - _eta_min) * 0.5 * (
                1.0 + _math.cos(_math.pi * progress)
            )

        self._lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=_lr_lambda, last_epoch=-1,
        )

        log.info(
            "Cosine LR schedule enabled: warmup=%d, total=%d, peak_mult=%.1f, "
            "peak_lrs=%s",
            warmup_steps, total_steps, peak_lr_mult,
            [f"{lr:.2e}" for lr in peak_lrs],
        )

    # ── Live reward recomposition ────────────────────────────────────

    def _recompose_rewards(self, transitions: list[dict]) -> None:
        """Re-weight replay rewards using current UI weights and toggles.

        Each transition that carries a ``reward_channels`` dict (the raw
        per-channel breakdown saved at collection time) gets its scalar
        ``reward`` recomputed with the **current** weight multipliers
        and channel toggles.  Transitions without ``reward_channels``
        (legacy data) keep their original reward unchanged.
        """
        if self._reward_weights is None:
            return

        w = self._reward_weights.snapshot()

        # Read toggle state once (or default to all-enabled).
        if self._toggle_state is not None:
            enabled = self._toggle_state.snapshot()
        else:
            enabled = None

        _PENALTY = RewardComposer._PENALTY_CHANNELS
        _PASSTHROUGH = frozenset({"survival", "extrinsic"})

        for t in transitions:
            channels = t.get("reward_channels")
            if channels is None:
                continue

            total = 0.0
            for ch, raw_val in channels.items():
                if ch == "total" or raw_val == 0.0:
                    continue
                # Apply toggle (disabled → 0).  Always pass through
                # survival and extrinsic (matching filter_channels).
                if enabled is not None and ch not in _PASSTHROUGH:
                    if not enabled.get(ch, True):
                        continue
                weight = w.get(ch, 1.0)
                if ch in _PENALTY:
                    total -= weight * raw_val
                else:
                    total += weight * raw_val

            t["reward"] = torch.tensor(max(-5.0, min(5.0, total)))

    def _collate_transitions(self, transitions: list[dict]) -> dict:
        """Collate a list of transition dicts into batched tensors.

        For tensor-valued keys, only transitions whose tensors match
        the shape of the first transition's tensor are included.  This
        prevents ``torch.stack`` from failing on variable-shape data.
        """
        batch = {}
        if not transitions:
            return batch

        # Pre-filter: drop any transition where a tensor key is NaN/Inf.
        # This prevents replay-buffer contamination from a corrupt inference
        # forward pass from causing an unrecoverable NaN cascade.
        _CRITICAL_KEYS = {"fused", "next_fused", "log_prob", "value", "reward"}
        clean = []
        for t in transitions:
            bad = False
            for k in _CRITICAL_KEYS:
                v = t.get(k)
                if isinstance(v, torch.Tensor) and (
                    torch.isnan(v).any() or torch.isinf(v).any()
                ):
                    bad = True
                    break
            if not bad:
                clean.append(t)
        if len(clean) < len(transitions):
            log.debug(
                "_collate_transitions: dropped %d/%d transitions with NaN/Inf tensors",
                len(transitions) - len(clean), len(transitions),
            )
        transitions = clean if clean else transitions

        keys = transitions[0].keys()
        for key in keys:
            values = [t[key] for t in transitions if key in t]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                # Filter to consistent shapes before stacking.
                ref_shape = values[0].shape
                consistent = [v for v in values if v.shape == ref_shape]
                if len(consistent) < max(len(values) // 2, 1):
                    log.debug("Skipping key '%s' in collate: only %d/%d tensors "
                              "match reference shape %s",
                              key, len(consistent), len(values), ref_shape)
                    continue
                try:
                    batch[key] = torch.stack(consistent).to(self.device)
                except RuntimeError as e:
                    log.debug("Skipping key '%s' in collate: %s", key, e)
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values, dtype=torch.float32, device=self.device)

        return batch

    def _make_fisher_loader(self, num_samples: int = 200):
        """Create an iterable of batched dicts from replay for Fisher estimation.

        Yields small batches (size = micro_batch_size) of collated transitions
        until *num_samples* total transitions have been yielded.  The Teacher's
        ``compute_fisher`` consumes these to estimate the diagonal Fisher
        Information matrix.
        """
        bs = self.config.micro_batch_size
        yielded = 0
        while yielded < num_samples:
            if self.replay.size < bs:
                break
            transitions, _w, _idx = self.replay.sample(bs, device=self.device)
            batch = self._collate_transitions(transitions)
            yield batch
            yielded += bs

    # ── N-step return helpers ────────────────────────────────────────

    @staticmethod
    def compute_nstep_returns(
        rewards: torch.Tensor,
        values: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> torch.Tensor:
        """Compute GAE-style n-step returns for a sequence of transitions.

        Uses Generalized Advantage Estimation (Schulman et al., 2016)
        for variance-reduced temporal difference targets:

        .. math::

            \\hat{A}_t = \\sum_{l=0}^{T-t-1} (\\gamma\\lambda)^l \\delta_{t+l}

        where :math:`\\delta_t = r_t + \\gamma V(s_{t+1}) - V(s_t)`.

        The returned tensor is the **target value**, not the advantage:
        ``target_t = V(s_t) + A_t``.

        Args:
            rewards: (T,) reward at each timestep.
            values:  (T+1,) value estimates.  ``values[-1]`` is the
                     bootstrap value at the end of the sequence.
            gamma:   Discount factor.
            lam:     GAE lambda (1.0 = Monte-Carlo, 0.0 = 1-step TD).

        Returns:
            (T,) target values.
        """
        T = rewards.shape[0]
        advantages = torch.zeros(T, device=rewards.device, dtype=rewards.dtype)
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        return values[:T] + advantages

    def train_on_sequence(
        self,
        sequence: list[dict],
        weight: float = 1.0,
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> Optional[dict]:
        """Train the Teacher on a contiguous sub-sequence with n-step returns.

        Unlike :meth:`_train_step` which treats each transition
        independently, this method:
        1. Runs the Teacher forward on each transition in order
        2. Collects value predictions
        3. Computes GAE targets from the reward sequence
        4. Uses the n-step targets as the value regression target
           (replacing the single-step reward)

        This gives the value head **temporal credit assignment** —
        credit for a reward can propagate back through up to
        ``seq_len`` steps, drastically improving value estimation.

        Args:
            sequence: List of transition dicts (length >= 2).
            weight: Importance-sampling weight for this sequence.
            gamma: Discount factor for n-step returns.
            lam: GAE lambda.

        Returns:
            Loss dict, or None if the sequence is too short.
        """
        if len(sequence) < 2:
            return None

        # Recompose rewards with current UI weights/toggles.
        self._recompose_rewards(sequence)

        # Collate the full sequence
        batch = self._collate_transitions(sequence)
        if "reward" not in batch or "action" not in batch:
            return None

        T = len(sequence)
        rewards = batch["reward"]  # (T,)

        # Forward pass to get value predictions for each step
        with self._teacher_lock:
            forward_kwargs = {
                k: v for k, v in batch.items()
                if k in ("vision", "audio", "code_x", "code_edge_index",
                         "code_batch", "sensor")
            }
            forward_kwargs["actions"] = batch["action"]
            if "goal_embedding" in batch:
                forward_kwargs["goal"] = batch["goal_embedding"]

            if self.config.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.teacher(**forward_kwargs)
            else:
                outputs = self.teacher(**forward_kwargs)

            values_pred = outputs["value"].squeeze(-1)  # (T,)

            # Bootstrap: use the last value prediction as V(s_{T})
            # (terminal bootstrap — we don't have the next observation)
            bootstrap = values_pred[-1].detach()
            values_extended = torch.cat(
                [values_pred.detach(), bootstrap.unsqueeze(0)]
            )  # (T+1,)

            # Compute GAE targets
            nstep_targets = self.compute_nstep_returns(
                rewards, values_extended, gamma, lam,
            )  # (T,)

            # Value loss with n-step targets
            value_loss = F.smooth_l1_loss(values_pred, nstep_targets.detach())

            # Policy loss (flow denoising) — unchanged
            denoising_loss = outputs.get(
                "denoising_loss", torch.tensor(0.0, device=self.device),
            )
            policy_loss = torch.clamp(denoising_loss, max=10.0)

            # Total loss (weighted)
            loss = policy_loss + 0.5 * value_loss
            loss = loss * weight

            if loss.grad_fn is None and not loss.requires_grad:
                return None

            # Backward
            if self.scaler is not None:
                self.scaler.scale(
                    loss / self.config.gradient_accumulation_steps
                ).backward()
            else:
                (loss / self.config.gradient_accumulation_steps).backward()

        return {"total": loss.item(), "value_loss": value_loss.item()}

    @property
    def step_count(self) -> int:
        return self._step

    @property
    def stats(self) -> dict:
        return {
            "step": self._step,
            "running": self._running,
            "replay_size": self.replay.size,
        }
