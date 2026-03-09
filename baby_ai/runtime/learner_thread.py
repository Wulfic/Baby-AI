"""
Learner thread — trains the Teacher model asynchronously.

Runs on the GPU in the background, consuming transitions from the
replay buffer, computing losses, and taking gradient steps.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_ai.config import TrainingConfig, DEFAULT_CONFIG
from baby_ai.memory.replay_buffer import PrioritizedReplayBuffer
from baby_ai.learning.intrinsic import ICM, LearningProgressEstimator
from baby_ai.learning.rewards import RewardComposer
from baby_ai.memory.consolidation import Consolidator
from baby_ai.utils.logging import get_logger

log = get_logger("learner", log_file="learner.log")


class LearnerThread:
    """
    Background Teacher training thread.

    Continuously:
    1. Samples prioritized minibatches from replay
    2. Computes policy + ICM + consolidation losses
    3. Updates Teacher weights
    4. Updates transition priorities based on learning progress

    Args:
        teacher: Teacher model to train.
        replay: Prioritized replay buffer.
        icm: Intrinsic Curiosity Module.
        consolidator: EWC consolidation module.
        config: Training hyperparameters.
        device: Device for training.
    """

    def __init__(
        self,
        teacher: nn.Module,
        replay: PrioritizedReplayBuffer,
        icm: ICM,
        consolidator: Consolidator,
        config: TrainingConfig | None = None,
        device: str = "cuda",
        teacher_lock: threading.Lock | None = None,
    ):
        self.teacher = teacher
        self.replay = replay
        self.icm = icm
        self.consolidator = consolidator
        self.config = config or DEFAULT_CONFIG.training
        self.device = device
        self._teacher_lock = teacher_lock or threading.Lock()

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
            {"params": encoder_params, "lr": self.config.encoder_lr},
            {"params": core_params, "lr": self.config.core_lr},
            {"params": policy_params, "lr": self.config.policy_lr},
        ])

        # No warmup scheduler — constant LR controlled by the GUI.
        # The live LR is read from the control panel each optimizer step.

        # AMP scaler
        self.scaler = torch.amp.GradScaler("cuda") if self.config.use_amp else None

        # Reward composer
        self.reward_composer = RewardComposer(
            intrinsic_weight_start=self.config.intrinsic_weight_start,
            intrinsic_weight_end=self.config.intrinsic_weight_end,
            intrinsic_decay_steps=self.config.intrinsic_decay_steps,
        )

        # Learning progress tracker
        self.lp_estimator = LearningProgressEstimator()

        # Thread control
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._step = 0
        self._accum_count = 0

    def start(self) -> None:
        """Start the learner thread."""
        if self._running:
            return
        self._running = True
        self.teacher.to(self.device).train()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="LearnerThread")
        self._thread.start()
        log.info("Learner thread started.")

    def stop(self) -> None:
        """Stop the learner thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        log.info("Learner thread stopped after %d steps.", self._step)

    def _loop(self) -> None:
        """Main training loop."""
        while self._running:
            # Wait until we have enough transitions
            if self.replay.size < self.config.micro_batch_size * 2:
                time.sleep(0.5)
                continue

            try:
                self._train_step()
            except Exception as e:
                log.error("Learner error at step %d: %s", self._step, e, exc_info=True)
                time.sleep(1.0)

            # Yield to other threads
            time.sleep(self.config.micro_batch_size * 0.001)

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
        # from running a concurrent forward on the same cuDNN GRU
        # (whose internal reserve buffers are not thread-safe).
        with self._teacher_lock:
            if self.config.use_amp:
                with torch.amp.autocast("cuda"):
                    loss_dict = self._compute_loss(batch, weights_t)
            else:
                loss_dict = self._compute_loss(batch, weights_t)

            total_loss = loss_dict["total"]

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
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.teacher.parameters(), max_norm=1.0
            )
            if self.scaler is not None:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            # ── Sync learning rate from GUI ───────────────────────
            try:
                from baby_ai.ui.control_panel import get_live_lr
                gui_lr = get_live_lr()
                for pg in self.optimizer.param_groups:
                    pg["lr"] = gui_lr
            except Exception:
                pass  # GUI not available yet

            self.optimizer.zero_grad()
            self._accum_count = 0

        # Update replay priorities
        new_priorities = [
            max(loss_dict.get("per_sample_loss", [0.1])[i] if i < len(loss_dict.get("per_sample_loss", [])) else 0.1, 1e-6)
            for i in range(len(indices))
        ]
        self.replay.update_priorities(indices, new_priorities)

        # Periodic consolidation
        if self._step > 0 and self._step % self.config.consolidation_every_n_steps == 0:
            ewc_loss = self.consolidator.consolidation_loss(self.teacher)
            if ewc_loss.item() > 0:
                ewc_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            log.info("Consolidation at step %d, EWC loss: %.4f", self._step, ewc_loss.item())

        self._step += 1

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
        # Teacher forward
        outputs = self.teacher(**{
            k: v for k, v in batch.items()
            if k in ("vision", "audio", "code_x", "code_edge_index", "code_batch", "sensor")
        })

        action_logits = outputs["action_logits"]
        # Use moderately clamped bounds to prevent overflow without causing a massive dead
        # gradient zone, allowing gradients to recover if logits briefly spike.
        action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=100.0, neginf=-100.0)
        action_logits = torch.clamp(action_logits, min=-100.0, max=100.0)
        
        value = outputs["value"]

        # Soft-clamp value predictions via tanh squashing to [-7, 7].
        # Rewards are now clamped to [-5, 5], so values should stay in
        # a similar range.  tanh preserves gradients everywhere (no dead
        # zones), but smoothly compresses extreme predictions back toward
        # the valid range.  The 7.0 scale gives headroom above the reward
        # clamp so the value head can slightly overshoot without penalty.
        value = 7.0 * torch.tanh(value / 7.0)

        # Policy loss (if we have targets)
        loss = torch.tensor(0.0, device=self.device)

        if "action" in batch and "reward" in batch:
            # Actor-critic loss
            actions = batch["action"]
            rewards = batch["reward"]
            old_values = batch.get("value", torch.zeros_like(rewards))

            advantage = rewards - old_values.squeeze(-1)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            # Clamp advantage to prevent catastrophic policy updates.
            # Tighter [-3, 3] range aligns with the reduced reward scale
            # and prevents any single sample from dominating the batch.
            advantage = torch.clamp(advantage, -3.0, 3.0)

            dist = torch.distributions.Categorical(logits=action_logits)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            policy_loss = -(log_prob * advantage.detach() * weights).mean()
            # Use Huber loss (SmoothL1) instead of MSE to reduce
            # sensitivity to outlier value predictions.
            # Weight value loss by replay priorities.
            value_loss = (F.smooth_l1_loss(value.squeeze(-1), rewards, reduction='none') * weights).mean()

            # Entropy bonus — encourages exploration but must NOT be
            # weighted by replay priorities (which would make high-
            # priority samples produce disproportionate entropy signal).
            entropy_bonus = 0.01 * entropy.mean()

            loss = policy_loss + 0.5 * value_loss - entropy_bonus

        # ICM loss (if we have next-state info)
        if "next_fused" in batch and "action" in batch:
            icm_out = self.icm(batch["fused"], batch["next_fused"], batch["action"])
            # Clamp the ICM losses to avoid wild gradient swings from unpredictable states
            fwd_loss_clamped = torch.clamp(icm_out["forward_loss"], max=10.0)
            inv_loss_clamped = torch.clamp(icm_out["inverse_loss"], max=10.0)
            icm_loss = fwd_loss_clamped + inv_loss_clamped
            loss = loss + 0.1 * icm_loss

        # EWC penalty
        ewc_penalty = self.consolidator.consolidation_loss(self.teacher)
        ewc_penalty = torch.clamp(ewc_penalty, max=5.0)
        loss = loss + ewc_penalty

        return {"total": loss}

    def _collate_transitions(self, transitions: list[dict]) -> dict:
        """Collate a list of transition dicts into batched tensors."""
        batch = {}
        if not transitions:
            return batch

        keys = transitions[0].keys()
        for key in keys:
            values = [t[key] for t in transitions if key in t]
            if not values:
                continue
            if isinstance(values[0], torch.Tensor):
                try:
                    batch[key] = torch.stack(values).to(self.device)
                except RuntimeError:
                    # Can't stack (variable sizes) — skip
                    pass
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values, dtype=torch.float32, device=self.device)

        return batch

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
