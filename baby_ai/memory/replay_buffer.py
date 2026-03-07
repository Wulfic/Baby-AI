"""
Prioritized compressed replay buffer.

Stores transitions with priority scores (based on learning progress / novelty).
Transitions are compressed (FP16 + LZ4) and stored on disk at Z:\\Baby_AI\\replay.
Supports sampling prioritized minibatches for training and distillation.
"""

from __future__ import annotations

import os
import random
import struct
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baby_ai.config import REPLAY_DIR, ensure_dirs
from baby_ai.utils.compression import compress_transition, decompress_transition
from baby_ai.utils.logging import get_logger

log = get_logger("replay_buffer", log_file="replay.log")


class SumTree:
    """
    Binary sum tree for efficient O(log n) prioritized sampling.

    Each leaf stores the priority; internal nodes store the sum of children.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity)
        self.data_pointer = 0
        self.n_entries = 0

    @property
    def total(self) -> float:
        return self.tree[1]

    def update(self, tree_idx: int, priority: float) -> None:
        delta = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 1:
            tree_idx //= 2
            self.tree[tree_idx] += delta

    def add(self, priority: float) -> int:
        """Add a new entry and return its tree index."""
        tree_idx = self.data_pointer + self.capacity
        self.update(tree_idx, priority)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return tree_idx

    def get(self, s: float) -> Tuple[int, float, int]:
        """
        Sample by cumulative priority value s.

        Returns: (tree_idx, priority, data_idx)
        """
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity
        return idx, self.tree[idx], data_idx


class PrioritizedReplayBuffer:
    """
    Compressed prioritized experience replay.

    Transitions are compressed and stored on disk; metadata and priorities live in RAM.
    Supports disk-backed storage on network drive (Z:\\Baby_AI\\replay).

    Args:
        capacity: Maximum number of transitions.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        beta_start: Initial importance-sampling correction exponent.
        beta_end: Final beta after annealing.
        beta_anneal_steps: Steps to anneal beta from start to end.
        disk_cap_gb: Maximum on-disk storage in GB.
        compression: Compression method ('lz4' or 'gzip').
    """

    def __init__(
        self,
        capacity: int = 50_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 200_000,
        disk_cap_gb: float = 4.0,
        compression: str = "lz4",
    ):
        ensure_dirs()

        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.disk_cap_gb = disk_cap_gb
        self.compression = compression

        self.tree = SumTree(capacity)
        self.min_priority = 1e-6
        self.max_priority = 1.0

        # On-disk storage
        self._storage_dir = REPLAY_DIR / "transitions"
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory metadata index
        self._meta: list[Optional[dict]] = [None] * capacity

        # Thread safety
        self._lock = threading.Lock()
        self._step = 0
        self._disk_bytes = 0

    @property
    def size(self) -> int:
        return self.tree.n_entries

    @property
    def beta(self) -> float:
        """Annealed importance-sampling exponent."""
        frac = min(1.0, self._step / max(1, self.beta_anneal_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    def _file_path(self, data_idx: int) -> Path:
        return self._storage_dir / f"{data_idx:06d}.bin"

    def add(
        self,
        transition: Dict[str, Any],
        priority: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Store a transition with given priority.

        Args:
            transition: Dict with tensors and scalars (state, action, reward, etc.).
            priority: Priority score. If None, uses max seen priority.
            metadata: Optional metadata dict (timestamps, episode info, etc.).
        """
        if priority is None:
            priority = self.max_priority

        priority = max(priority, self.min_priority) ** self.alpha

        # Compress and write to disk
        blob = compress_transition(transition, method=self.compression)

        with self._lock:
            tree_idx = self.tree.add(priority)
            data_idx = tree_idx - self.tree.capacity

            # Remove old file if overwriting
            old_path = self._file_path(data_idx)
            if old_path.exists():
                self._disk_bytes -= old_path.stat().st_size
                old_path.unlink()

            # Write new
            with open(self._file_path(data_idx), "wb") as f:
                f.write(blob)
            self._disk_bytes += len(blob)

            self._meta[data_idx] = metadata or {}
            self.max_priority = max(self.max_priority, priority)
            self._step += 1

        # Check disk cap
        if self._disk_bytes > self.disk_cap_gb * 1e9:
            self._prune_oldest()

    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[int]]:
        """
        Sample a prioritized minibatch.

        Returns:
            transitions: List of decompressed transition dicts.
            weights: (batch_size,) importance-sampling weights.
            indices: List of tree indices (for priority updates).
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer has {self.size} items, need {batch_size}")

        segment = self.tree.total / batch_size
        beta = self.beta

        transitions = []
        weights = []
        indices = []

        with self._lock:
            min_prob = self.tree.tree[self.tree.capacity:self.tree.capacity + self.size].min()
            min_prob = max(min_prob, self.min_priority)

            for i in range(batch_size):
                lo = segment * i
                hi = segment * (i + 1)
                s = random.uniform(lo, hi)
                tree_idx, priority, data_idx = self.tree.get(s)

                # Load from disk
                path = self._file_path(data_idx)
                if not path.exists():
                    # Fallback: random re-sample
                    s = random.uniform(0, self.tree.total)
                    tree_idx, priority, data_idx = self.tree.get(s)
                    path = self._file_path(data_idx)

                with open(path, "rb") as f:
                    blob = f.read()
                trans = decompress_transition(blob, method=self.compression, device=device)
                transitions.append(trans)

                # Importance-sampling weight
                prob = priority / self.tree.total
                w = (prob * self.size) ** (-beta)
                weights.append(w)
                indices.append(tree_idx)

        weights = np.array(weights, dtype=np.float32)
        weights /= weights.max()  # normalize

        return transitions, weights, indices

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled transitions (after learning)."""
        with self._lock:
            for idx, p in zip(indices, priorities):
                p = max(p, self.min_priority) ** self.alpha
                self.tree.update(idx, p)
                self.max_priority = max(self.max_priority, p)

    def _prune_oldest(self) -> None:
        """Remove oldest transitions to stay under disk cap."""
        files = sorted(self._storage_dir.glob("*.bin"), key=lambda f: f.stat().st_mtime)
        removed = 0
        target = int(self.disk_cap_gb * 1e9 * 0.8)  # prune to 80% of cap
        for f in files:
            if self._disk_bytes <= target:
                break
            size = f.stat().st_size
            f.unlink()
            self._disk_bytes -= size
            removed += 1
        if removed:
            log.info("Pruned %d old transitions to fit disk cap.", removed)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "disk_mb": self._disk_bytes / (1024 ** 2),
            "disk_cap_gb": self.disk_cap_gb,
            "total_priority": float(self.tree.total),
            "step": self._step,
        }
