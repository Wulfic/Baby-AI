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
from baby_ai.memory.replay_buffer import PrioritizedReplayBuffer
from baby_ai.learning.intrinsic import JEPACuriosity, LearningProgressEstimator
from baby_ai.learning.rebel import REBELLoss
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
        ).to(device) if rebel_cfg.enabled else None
        self._rebel_value_loss_weight = rebel_cfg.value_loss_weight

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

        self.optimizer = torch.optim.Adam([
            {"params": encoder_params, "lr": self.config.encoder_lr, "initial_lr": self.config.encoder_lr},
            {"params": core_params, "lr": self.config.core_lr, "initial_lr": self.config.core_lr},
            {"params": policy_params, "lr": self.config.policy_lr, "initial_lr": self.config.policy_lr},
        ])

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
            })

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._step = 0
        self._accum_count = 0

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
        # Sample from replay
        transitions, weights, indices = self.replay.sample(
            self.config.micro_batch_size, device=self.device
        )

        # Build batch tensors from transitions
        batch = self._collate_transitions(transitions)
        weights_t = torch.from_numpy(weights).float().to(self.device)

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

            # Backward with gradient accumulation
            if self.scaler is not None:
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

        # Update replay priorities
        new_priorities = [
            max(loss_dict.get("per_sample_loss", [0.1])[i] if i < len(loss_dict.get("per_sample_loss", [])) else 0.1, 1e-6)
            for i in range(len(indices))
        ]
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
        forward_kwargs = {
            k: v for k, v in batch.items()
            if k in ("vision", "audio", "code_x", "code_edge_index", "code_batch", "sensor")
        }
        if "action" in batch:
            forward_kwargs["actions"] = batch["action"]

        # Pass goal_embedding from replay for goal-conditioned training
        if "goal_embedding" in batch:
            forward_kwargs["goal"] = batch["goal_embedding"]

        outputs = self.teacher(**forward_kwargs)

        value = outputs["value"]

        # Soft-clamp value predictions via tanh squashing to [-7, 7].
        value = 7.0 * torch.tanh(value / 7.0)

        # Anchor loss to the computation graph via a zero derived from
        # model output.  This ensures loss.grad_fn is set even when no
        # policy branch fires, so backward() won't crash.
        loss = (value.sum() * 0.0).squeeze()

        if "action" in batch and "reward" in batch:
            actions = batch["action"]
            rewards = batch["reward"]

            # Value loss — always trained (needed for System 2 trajectory scoring)
            value_loss = (F.smooth_l1_loss(value.squeeze(-1), rewards, reduction='none') * weights).mean()

            if self._rebel_enabled and self._rebel_loss_fn is not None:
                # ── REBEL paired loss (Phase D) ──
                rebel_loss = self._compute_rebel_loss(outputs, batch, weights)
                loss = rebel_loss + self._rebel_value_loss_weight * value_loss
            else:
                # ── Legacy: reward-weighted denoising loss ──
                # NOTE: "value" is never stored in replay transitions,
                # so old_values is always zeros.  Advantage thus equals
                # raw rewards.  This is a known limitation — a proper
                # implementation would store the value estimate at
                # collection time for off-policy correction.
                old_values = batch.get("value", torch.zeros_like(rewards))
                advantage = rewards - old_values.squeeze(-1)
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

        # MoE load-balancing auxiliary loss (from Jamba temporal core)
        if hasattr(self.teacher, 'temporal') and hasattr(self.teacher.temporal, 'aux_loss'):
            moe_loss = self.teacher.temporal.aux_loss
            if moe_loss.requires_grad or moe_loss.item() > 0:
                loss = loss + moe_loss

        # EWC penalty
        ewc_penalty = self.consolidator.consolidation_loss(self.teacher)
        ewc_penalty = torch.clamp(ewc_penalty, max=5.0)
        loss = loss + ewc_penalty

        return {"total": loss}

    def _compute_rebel_loss(
        self, outputs: dict, batch: dict, weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute REBEL paired preference loss (Phase D).

        Splits the batch into two halves and designates the higher-reward
        action in each pair as the "winner".  Falls back to the flow-loss
        from the forward pass if the batch is too small for pairing.
        """
        actions = batch["action"]
        rewards = batch["reward"]
        core_state = outputs["core_state"]
        B = actions.size(0)

        if B < 4:
            # Batch too small for meaningful pairing — use forward flow loss
            return outputs.get("denoising_loss", torch.tensor(0.0, device=actions.device))

        half = B // 2
        s1, s2 = core_state[:half], core_state[half:half * 2]
        a1, a2 = actions[:half], actions[half:half * 2]
        r1, r2 = rewards[:half], rewards[half:half * 2]

        # Determine winner/loser per pair
        w_mask = r1 >= r2  # True where first element is winner
        state_pairs = s1  # Use first-half states as conditioning
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

        Args:
            total_steps: Total optimizer steps expected (epochs * steps_per_epoch).
            warmup_steps: Linear warmup phase length.
            peak_lr_mult: Multiplier on the configured LR for peak value.
            eta_min_mult: Fraction of peak LR for the cosine floor.
        """
        # Scale each param group's LR up to the peak
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg["initial_lr"] * peak_lr_mult
            pg["initial_lr"] = pg["lr"]  # cosine scheduler reads initial_lr

        self._lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    self.optimizer,
                    start_factor=0.1,
                    total_iters=warmup_steps,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=max(total_steps - warmup_steps, 1),
                    eta_min=pg["lr"] * eta_min_mult,
                ),
            ],
            milestones=[warmup_steps],
        )
        log.info(
            "Cosine LR schedule enabled: warmup=%d, total=%d, peak_mult=%.1f, "
            "peak_lrs=%s",
            warmup_steps, total_steps, peak_lr_mult,
            [f"{pg['lr']:.2e}" for pg in self.optimizer.param_groups],
        )

    def _collate_transitions(self, transitions: list[dict]) -> dict:
        """Collate a list of transition dicts into batched tensors.

        For tensor-valued keys, only transitions whose tensors match
        the shape of the first transition's tensor are included.  This
        prevents ``torch.stack`` from failing on variable-shape data.
        """
        batch = {}
        if not transitions:
            return batch

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
