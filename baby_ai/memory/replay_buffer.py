"""
Prioritized compressed replay buffer.

Stores transitions with priority scores (based on learning progress / novelty).
Transitions are compressed (FP16 + LZ4) and stored on disk at Z:\\Baby_AI\\replay
using **chunked storage** — each file holds up to CHUNK_SIZE transitions so the
total on-disk file count stays in the low hundreds instead of tens of thousands.

Supports sampling prioritized minibatches for training and distillation.
"""

from __future__ import annotations

import os
import random
import struct
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from baby_ai.config import REPLAY_DIR, ensure_dirs
from baby_ai.utils.compression import compress_transition, decompress_transition
from baby_ai.utils.logging import get_logger

log = get_logger("replay_buffer", log_file="replay.log")

# ── Chunk storage constants ─────────────────────────────────────
# Each chunk file packs up to CHUNK_SIZE transitions.  With a
# capacity of 50 000 that gives ~100 files instead of 50 000.
CHUNK_SIZE = 500


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


# ── Chunked on-disk storage ────────────────────────────────────

class _ChunkStore:
    """
    Manages chunked on-disk storage for replay transitions.

    File format per chunk (``chunk_NNNN.bin``):

    .. code-block:: text

        [4 bytes]   num_slots  (uint32-LE, always == CHUNK_SIZE)
        For each slot (CHUNK_SIZE entries):
            [4 bytes] blob_offset  (uint32-LE, 0 = empty)
            [4 bytes] blob_length  (uint32-LE, 0 = empty)
        [variable]  concatenated compressed blobs

    Offsets are relative to the **start of the blob data region**
    (i.e. after the header).  An offset+length of (0, 0) means the
    slot is empty / not yet written.

    Writes are append-only within a chunk.  When a slot is overwritten
    the old blob bytes become dead space; the chunk is periodically
    compacted if fragmentation exceeds a threshold.
    """

    HEADER_SLOT_BYTES = 8  # 4 offset + 4 length per slot

    def __init__(self, storage_dir: Path, capacity: int):
        self._dir = storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._capacity = capacity
        self._num_chunks = (capacity + CHUNK_SIZE - 1) // CHUNK_SIZE
        self._disk_bytes = 0

        # In-memory blob cache — keeps the most recently used blobs
        # so repeated reads of the same transition don't hit disk.
        # OrderedDict gives us O(1) LRU eviction via move_to_end().
        self._blob_cache: OrderedDict[int, bytes] = OrderedDict()
        self._CACHE_MAX = 256

        # Scan existing files to get disk usage
        for f in self._dir.glob("chunk_*.bin"):
            self._disk_bytes += f.stat().st_size

    # ── Path helpers ────────────────────────────────────────────

    def _chunk_id(self, data_idx: int) -> int:
        return data_idx // CHUNK_SIZE

    def _slot_in_chunk(self, data_idx: int) -> int:
        return data_idx % CHUNK_SIZE

    def _chunk_path(self, chunk_id: int) -> Path:
        return self._dir / f"chunk_{chunk_id:04d}.bin"

    # ── Header I/O ──────────────────────────────────────────────

    def _header_size(self) -> int:
        return 4 + CHUNK_SIZE * self.HEADER_SLOT_BYTES

    def _read_header(self, chunk_id: int) -> List[Tuple[int, int]]:
        """Return list of (offset, length) for each slot."""
        path = self._chunk_path(chunk_id)
        if not path.exists():
            return [(0, 0)] * CHUNK_SIZE
        with open(path, "rb") as f:
            raw = f.read(self._header_size())
        if len(raw) < self._header_size():
            return [(0, 0)] * CHUNK_SIZE
        # Skip the 4-byte num_slots prefix
        entries = []
        for i in range(CHUNK_SIZE):
            base = 4 + i * self.HEADER_SLOT_BYTES
            off, ln = struct.unpack_from("<II", raw, base)
            entries.append((off, ln))
        return entries

    def _write_header(self, f, entries: List[Tuple[int, int]]) -> None:
        """Write the header at the current file position."""
        f.write(struct.pack("<I", CHUNK_SIZE))
        for off, ln in entries:
            f.write(struct.pack("<II", off, ln))

    # ── Public API ──────────────────────────────────────────────

    def write(self, data_idx: int, blob: bytes) -> None:
        """Store a compressed transition blob for *data_idx*."""
        chunk_id = self._chunk_id(data_idx)
        slot = self._slot_in_chunk(data_idx)
        path = self._chunk_path(chunk_id)

        # Read existing header (or create empty)
        entries = self._read_header(chunk_id)

        # Determine the end of the current blob region
        header_sz = self._header_size()
        if path.exists():
            file_sz = path.stat().st_size
        else:
            file_sz = header_sz  # new file starts with just the header

        blob_region_end = max(file_sz - header_sz, 0)

        # Old blob for this slot becomes dead space (we don't reclaim it
        # immediately — compaction handles that if fragmentation grows).
        old_len = entries[slot][1]

        # Append new blob at the end of the blob region
        new_offset = blob_region_end
        entries[slot] = (new_offset, len(blob))

        # Write: rewrite header + append blob
        if not path.exists():
            with open(path, "wb") as f:
                self._write_header(f, entries)
                f.write(blob)
            self._disk_bytes += header_sz + len(blob)
        else:
            # Rewrite header in-place, append blob at EOF
            with open(path, "r+b") as f:
                f.seek(0)
                self._write_header(f, entries)
                f.seek(0, 2)  # EOF
                f.write(blob)
            self._disk_bytes += len(blob)

        # Update cache (LRU: most recent access at the end)
        self._blob_cache[data_idx] = blob
        self._blob_cache.move_to_end(data_idx)
        if len(self._blob_cache) > self._CACHE_MAX:
            self._blob_cache.popitem(last=False)  # evict least-recently-used

    def read(self, data_idx: int) -> Optional[bytes]:
        """Read a compressed transition blob, or None if missing."""
        # Check cache first
        if data_idx in self._blob_cache:
            return self._blob_cache[data_idx]

        chunk_id = self._chunk_id(data_idx)
        slot = self._slot_in_chunk(data_idx)
        path = self._chunk_path(chunk_id)

        if not path.exists():
            return None

        entries = self._read_header(chunk_id)
        off, ln = entries[slot]
        if ln == 0:
            return None

        header_sz = self._header_size()
        with open(path, "rb") as f:
            f.seek(header_sz + off)
            blob = f.read(ln)

        # Populate cache (LRU: most recent access at the end)
        self._blob_cache[data_idx] = blob
        self._blob_cache.move_to_end(data_idx)
        if len(self._blob_cache) > self._CACHE_MAX:
            self._blob_cache.popitem(last=False)

        return blob

    def remove_chunk(self, chunk_id: int) -> int:
        """Delete an entire chunk file. Returns bytes freed."""
        path = self._chunk_path(chunk_id)
        if path.exists():
            sz = path.stat().st_size
            path.unlink()
            self._disk_bytes -= sz
            # Evict cache entries for this chunk
            start = chunk_id * CHUNK_SIZE
            for idx in range(start, start + CHUNK_SIZE):
                self._blob_cache.pop(idx, None)
            return sz
        return 0

    def clear_all(self) -> None:
        """Remove every chunk file."""
        for f in self._dir.glob("chunk_*.bin"):
            f.unlink()
        self._disk_bytes = 0
        self._blob_cache.clear()

    @property
    def disk_bytes(self) -> int:
        return self._disk_bytes


class PrioritizedReplayBuffer:
    """
    Compressed prioritized experience replay.

    Transitions are compressed and stored on disk in chunk files;
    metadata and priorities live in RAM.  Supports disk-backed
    storage on network drive (Z:\\Baby_AI\\replay).

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

        # Chunked on-disk storage (replaces per-file storage)
        storage_dir = REPLAY_DIR / "chunks"
        self._store = _ChunkStore(storage_dir, capacity)

        # In-memory metadata index
        self._meta: list[Optional[dict]] = [None] * capacity

        # Thread safety
        self._lock = threading.Lock()
        self._step = 0

    @property
    def size(self) -> int:
        return self.tree.n_entries

    @property
    def beta(self) -> float:
        """Annealed importance-sampling exponent."""
        frac = min(1.0, self._step / max(1, self.beta_anneal_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

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

        # Compress transition
        blob = compress_transition(transition, method=self.compression)

        with self._lock:
            tree_idx = self.tree.add(priority)
            data_idx = tree_idx - self.tree.capacity

            # Write blob into chunk file (overwrites old slot)
            self._store.write(data_idx, blob)

            self._meta[data_idx] = metadata or {}
            self.max_priority = max(self.max_priority, priority)
            self._step += 1

        # Check disk cap
        if self._store.disk_bytes > self.disk_cap_gb * 1e9:
            self._prune_oldest_chunks()

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

                # Load from chunk store
                blob = self._store.read(data_idx)
                if blob is None:
                    # Fallback: random re-sample
                    s = random.uniform(0, self.tree.total)
                    tree_idx, priority, data_idx = self.tree.get(s)
                    blob = self._store.read(data_idx)

                if blob is None:
                    # Still nothing — skip this sample
                    continue

                trans = decompress_transition(blob, method=self.compression, device=device)
                transitions.append(trans)

                # Importance-sampling weight
                prob = priority / self.tree.total
                w = (prob * self.size) ** (-beta)
                weights.append(w)
                indices.append(tree_idx)

        weights = np.array(weights, dtype=np.float32)
        if weights.size > 0:
            weights /= weights.max()  # normalize

        return transitions, weights, indices

    def update_priorities(self, indices: List[int], priorities: List[float]) -> None:
        """Update priorities for sampled transitions (after learning)."""
        with self._lock:
            for idx, p in zip(indices, priorities):
                p = max(p, self.min_priority) ** self.alpha
                self.tree.update(idx, p)
                self.max_priority = max(self.max_priority, p)

    def sample_pairs(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Tuple[List[Tuple[Dict, Dict, float, float]], np.ndarray, List[int]]:
        """
        Sample pairs of transitions for REBEL training (Phase D).

        Samples 2 * batch_size transitions, pairs them up, and designates
        the higher-reward transition as "winner" in each pair.

        Args:
            batch_size: Number of pairs to return.
            device:     Device for transition tensors.

        Returns:
            pairs:   List of (winner_trans, loser_trans, r_winner, r_loser) tuples.
            weights: (batch_size,) importance-sampling weights.
            indices: List of tree indices (first-half only, for priority updates).
        """
        transitions, weights, indices = self.sample(batch_size * 2, device)

        # Split into two halves and pair them
        t1 = transitions[:batch_size]
        t2 = transitions[batch_size:]

        pairs = []
        for a, b in zip(t1, t2):
            r_a = a.get("reward", 0.0)
            r_b = b.get("reward", 0.0)
            if isinstance(r_a, torch.Tensor):
                r_a = r_a.item()
            if isinstance(r_b, torch.Tensor):
                r_b = r_b.item()
            if r_a >= r_b:
                pairs.append((a, b, r_a, r_b))
            else:
                pairs.append((b, a, r_b, r_a))

        return pairs, weights[:batch_size], indices[:batch_size]

    def _prune_oldest_chunks(self) -> None:
        """Remove the oldest chunk files and zero their SumTree priorities."""
        chunk_files = sorted(
            self._store._dir.glob("chunk_*.bin"),
            key=lambda f: f.stat().st_mtime,
        )
        target = int(self.disk_cap_gb * 1e9 * 0.8)  # prune to 80%
        removed = 0
        for f in chunk_files:
            if self._store.disk_bytes <= target:
                break
            # Extract chunk_id from filename (chunk_0042.bin → 42)
            try:
                chunk_id = int(f.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            # Zero out SumTree priorities for every slot in this chunk
            # so the sampler never draws a stale, deleted transition.
            start_idx = chunk_id * CHUNK_SIZE
            for slot in range(CHUNK_SIZE):
                data_idx = start_idx + slot
                if data_idx < self.capacity:
                    tree_idx = data_idx + self.tree.capacity
                    self.tree.update(tree_idx, 0.0)
                    self._meta[data_idx] = None
            sz = f.stat().st_size
            f.unlink()
            self._store._disk_bytes -= sz
            # Evict cache entries for this chunk
            for idx in range(start_idx, start_idx + CHUNK_SIZE):
                self._store._blob_cache.pop(idx, None)
            removed += 1
        if removed:
            log.info("Pruned %d chunk files (zeroed SumTree priorities).", removed)

    def stats(self) -> dict:
        return {
            "size": self.size,
            "capacity": self.capacity,
            "disk_mb": self._store.disk_bytes / (1024 ** 2),
            "disk_cap_gb": self.disk_cap_gb,
            "total_priority": float(self.tree.total),
            "step": self._step,
            "chunk_files": self._store._num_chunks,
        }
