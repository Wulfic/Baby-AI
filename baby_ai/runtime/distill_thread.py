"""
Distillation thread — periodic Teacher → Student knowledge transfer.

Runs on a background schedule: every N Teacher steps, sample
prioritized replay, compute distillation loss, update Student.
"""

from __future__ import annotations

import threading
from typing import Optional

import torch

from baby_ai.config import TrainingConfig, DEFAULT_CONFIG
from baby_ai.learning.distillation import DistillationEngine
from baby_ai.memory.replay_buffer import PrioritizedReplayBuffer
from baby_ai.utils.logging import get_logger

log = get_logger("distill_thread", log_file="distill.log")


class DistillThread:
    """
    Periodic distillation: Teacher → Student.

    Watches the Teacher's step count, and every `distill_every_n_steps`
    performs a round of distillation on prioritized replay data,
    then atomically swaps the Student weights.

    Args:
        distill_engine: The DistillationEngine handling loss + optimization.
        replay: Shared replay buffer for sampling.
        learner_step_fn: Callable returning current Teacher step count.
        config: Training hyperparameters.
        device: Device for distillation computation.
    """

    def __init__(
        self,
        distill_engine: DistillationEngine,
        replay: PrioritizedReplayBuffer,
        learner_step_fn,
        config: TrainingConfig | None = None,
        device: str = "cuda",
    ):
        self.engine = distill_engine
        self.replay = replay
        self.learner_step_fn = learner_step_fn
        self.config = config or DEFAULT_CONFIG.training
        self.device = device

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._last_distill_at = 0
        self._distill_count = 0

        # Event-driven coordination: the learner thread sets this event
        # when it crosses the ``distill_every_n_steps`` threshold,
        # waking the distill loop immediately instead of polling.
        self._distill_ready = threading.Event()
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._distill_ready.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DistillThread"
        )
        self._thread.start()
        log.info("Distillation thread started (every %d steps).", self.config.distill_every_n_steps)

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()       # unblock waits immediately
        self._distill_ready.set()    # unblock any wait
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        log.info("Distillation thread stopped after %d rounds.", self._distill_count)

    def notify_distill_ready(self) -> None:
        """Signal that enough learner steps have passed to trigger distill.

        Called by the learner thread or orchestrator when the step
        threshold is crossed.  Wakes the distill loop immediately.
        """
        self._distill_ready.set()

    def _loop(self) -> None:
        while self._running:
            # Pause distillation while record-only or disable-learning mode is active.
            try:
                from baby_ai.ui.control_panel import get_record_only, get_learning_disabled
                if get_record_only() or get_learning_disabled():
                    self._stop_event.wait(timeout=1.0)
                    continue
            except ImportError:
                pass

            # Event-driven: wait until signalled or timeout (fallback poll)
            self._distill_ready.wait(timeout=2.0)
            self._distill_ready.clear()

            if not self._running:
                break

            learner_step = self.learner_step_fn()

            # Check if it's time to distill
            if learner_step - self._last_distill_at >= self.config.distill_every_n_steps:
                if self.replay.size >= self.config.micro_batch_size:
                    try:
                        self._distill_round()
                        self._last_distill_at = learner_step
                    except Exception as e:
                        log.error("Distillation error: %s", e, exc_info=True)

    def _distill_round(self) -> None:
        """
        Run an adaptive round of distillation steps, then swap.

        Steps continue until one of:
          - Loss plateaus (no relative improvement > threshold for
            ``patience`` consecutive steps)
          - ``max_steps`` is reached
          - Replay buffer runs dry

        At minimum ``min_steps`` steps are always taken so the
        Student gets meaningful gradient signal even when a single
        batch has low loss variance.
        """
        min_steps = getattr(self.config, "distill_min_steps", 5)
        max_steps = getattr(self.config, "distill_max_steps", 40)
        patience  = getattr(self.config, "distill_plateau_patience", 4)
        threshold = getattr(self.config, "distill_plateau_threshold", 0.01)

        log.info("Starting distillation round %d (min=%d, max=%d, patience=%d)...",
                 self._distill_count, min_steps, max_steps, patience)

        total_loss = 0.0
        best_loss = float("inf")
        stale_count = 0
        steps_done = 0

        for step_i in range(max_steps):
            if self.replay.size < self.config.micro_batch_size:
                break

            # Sample prioritized batch
            transitions, weights, indices = self.replay.sample(
                self.config.micro_batch_size, device=self.device
            )

            # Collate into batch
            batch = self._collate(transitions)

            # Distill step (pass indices for teacher soft-label cache)
            loss_info = self.engine.distill_step(
                batch, device=self.device, replay_indices=list(indices),
            )
            step_loss = loss_info.get("total", 0.0)
            total_loss += step_loss
            steps_done += 1

            # ── Adaptive early stopping (after min_steps) ───────
            if step_i >= min_steps:
                rel_improvement = (best_loss - step_loss) / max(abs(best_loss), 1e-8)
                if rel_improvement < threshold:
                    stale_count += 1
                else:
                    stale_count = 0
                if stale_count >= patience:
                    log.info("Early stop at step %d/%d (loss plateaued at %.4f).",
                             step_i + 1, max_steps, step_loss)
                    break

            if step_loss < best_loss:
                best_loss = step_loss

        # Atomic swap: staging Student → live Student
        self.engine.swap_to_live()
        self._distill_count += 1

        avg_loss = total_loss / max(steps_done, 1)
        log.info(
            "Distillation round %d complete | steps=%d | avg_loss=%.4f | best_loss=%.4f",
            self._distill_count, steps_done, avg_loss, best_loss,
        )

    def _collate(self, transitions: list[dict]) -> dict:
        """Collate transitions into batched tensors."""
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
                except RuntimeError as e:
                    log.debug("Dropped key '%s' in collate (shape mismatch): %s", key, e)
            elif isinstance(values[0], (int, float)):
                batch[key] = torch.tensor(values, dtype=torch.float32, device=self.device)

        return batch

    @property
    def stats(self) -> dict:
        return {
            "distill_count": self._distill_count,
            "last_distill_at": self._last_distill_at,
            "running": self._running,
        }
