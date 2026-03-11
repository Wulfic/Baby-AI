"""
Distillation thread — periodic Teacher → Student knowledge transfer.

Runs on a background schedule: every N Teacher steps, sample
prioritized replay, compute distillation loss, update Student.
"""

from __future__ import annotations

import threading
import time
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

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="DistillThread"
        )
        self._thread.start()
        log.info("Distillation thread started (every %d steps).", self.config.distill_every_n_steps)

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=10.0)
        log.info("Distillation thread stopped after %d rounds.", self._distill_count)

    def _loop(self) -> None:
        while self._running:
            learner_step = self.learner_step_fn()

            # Check if it's time to distill
            if learner_step - self._last_distill_at >= self.config.distill_every_n_steps:
                if self.replay.size >= self.config.micro_batch_size:
                    try:
                        self._distill_round()
                        self._last_distill_at = learner_step
                    except Exception as e:
                        log.error("Distillation error: %s", e, exc_info=True)

            time.sleep(1.0)  # check every second

    def _distill_round(self, num_steps: int = 20) -> None:
        """
        Run a round of distillation steps, then swap.

        Args:
            num_steps: Number of gradient steps per distillation round.
        """
        log.info("Starting distillation round %d...", self._distill_count)
        total_loss = 0.0

        for _ in range(num_steps):
            if self.replay.size < self.config.micro_batch_size:
                break

            # Sample prioritized batch
            transitions, weights, indices = self.replay.sample(
                self.config.micro_batch_size, device=self.device
            )

            # Collate into batch
            batch = self._collate(transitions)

            # Distill step
            loss_info = self.engine.distill_step(batch, device=self.device)
            total_loss += loss_info.get("total", 0.0)

        # Atomic swap: staging Student → live Student
        self.engine.swap_to_live()
        self._distill_count += 1

        avg_loss = total_loss / max(num_steps, 1)
        log.info(
            "Distillation round %d complete | avg_loss=%.4f",
            self._distill_count, avg_loss,
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
                except RuntimeError:
                    pass
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
