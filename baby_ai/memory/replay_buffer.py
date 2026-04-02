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

    Each leaf stores a transition's priority value.  Internal nodes
    store the sum of their children, enabling proportional sampling
    (i.e. transitions with higher priority are drawn more often).

    The tree array has 2 * capacity elements:
      - Index 0 is unused (sentinel).
      - Index 1 is the root (total sum of all priorities).
      - Indices [capacity, 2*capacity) are the leaves (one per transition).
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

        # Track which chunks are fully loaded into the blob cache
        # so ensure_chunks() can skip re-reads and evict old ones.
        self._loaded_chunks: set[int] = set()

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

    def load_chunk(self, chunk_id: int) -> int:
        """Load all blobs from one chunk file into the blob cache.

        Reads the chunk file **once** (a single sequential I/O
        operation) and inserts every non-empty slot into
        ``_blob_cache``.  If the chunk is already loaded this is a
        no-op.

        Returns:
            Number of blobs loaded (0 if already cached or missing).
        """
        if chunk_id in self._loaded_chunks:
            return 0

        path = self._chunk_path(chunk_id)
        if not path.exists():
            return 0

        header = self._read_header(chunk_id)
        header_sz = self._header_size()

        base_idx = chunk_id * CHUNK_SIZE
        loaded = 0

        # Try reading the entire blob region in one I/O operation
        # (fastest on network drives).  Fall back to per-blob seeks
        # if the single large allocation triggers a MemoryError.
        try:
            with open(path, "rb") as f:
                f.seek(header_sz)
                blob_region = f.read()

            for slot, (off, ln) in enumerate(header):
                if ln == 0:
                    continue
                data_idx = base_idx + slot
                self._blob_cache[data_idx] = blob_region[off : off + ln]
                loaded += 1
            del blob_region  # release bulk buffer immediately
        except MemoryError:
            # Not enough contiguous RAM for the whole chunk.
            # Free what we can, then read each blob individually.
            import gc; gc.collect()
            log.warning("Bulk read for chunk_%04d failed (MemoryError); "
                        "falling back to per-blob reads.", chunk_id)
            with open(path, "rb") as f:
                for slot, (off, ln) in enumerate(header):
                    if ln == 0:
                        continue
                    f.seek(header_sz + off)
                    data_idx = base_idx + slot
                    self._blob_cache[data_idx] = f.read(ln)
                    loaded += 1

        self._loaded_chunks.add(chunk_id)
        return loaded

    def evict_chunk(self, chunk_id: int) -> None:
        """Remove all cached blobs belonging to *chunk_id*."""
        if chunk_id not in self._loaded_chunks:
            return
        base = chunk_id * CHUNK_SIZE
        for idx in range(base, base + CHUNK_SIZE):
            self._blob_cache.pop(idx, None)
        self._loaded_chunks.discard(chunk_id)

    def ensure_chunks(self, needed: set[int]) -> None:
        """Load *needed* chunk IDs and evict all others.

        This is the main entry-point used by the sequential sampler
        to keep a rolling prefetch window of a few chunks in RAM
        while discarding chunks that have already been consumed.
        """
        # Evict chunks no longer needed
        for cid in list(self._loaded_chunks - needed):
            self.evict_chunk(cid)

        # Load new chunks
        for cid in sorted(needed - self._loaded_chunks):
            loaded = self.load_chunk(cid)
            if loaded > 0:
                log.info("Prefetched chunk_%04d (%d blobs)", cid, loaded)

        # Keep CACHE_MAX large enough so LRU eviction never
        # kicks out prefetched blobs during normal reads.
        self._CACHE_MAX = max(len(self._blob_cache) + CHUNK_SIZE, self._CACHE_MAX)

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

        # Chunked on-disk storage — AI transitions
        ai_dir = REPLAY_DIR / "ai"

        # Backward compat: migrate old replay/chunks/ → replay/ai/
        old_chunks_dir = REPLAY_DIR / "chunks"
        if old_chunks_dir.exists() and any(old_chunks_dir.glob("chunk_*.bin")):
            if not ai_dir.exists() or not any(ai_dir.glob("chunk_*.bin")):
                log.info(
                    "Migrating replay/chunks/ → replay/ai/ ..."
                )
                ai_dir.mkdir(parents=True, exist_ok=True)
                import shutil
                for f in old_chunks_dir.glob("chunk_*.bin"):
                    shutil.move(str(f), str(ai_dir / f.name))
                log.info("Migration complete.")

        self._store = _ChunkStore(ai_dir, capacity)

        # Separate append-only archive for recorded demos
        demo_dir = REPLAY_DIR / "demos"
        demo_dir.mkdir(parents=True, exist_ok=True)
        self._demo_store = _ChunkStore(demo_dir, capacity)
        self._demo_write_idx: int = 0  # independent append cursor
        # Scan existing demo chunks to resume append position
        for f in demo_dir.glob("chunk_*.bin"):
            header = self._demo_store._read_header(
                int(f.stem.split("_")[1])
            )
            for _, (_, ln) in enumerate(header):
                if ln > 0:
                    self._demo_write_idx += 1

        # In-memory metadata index
        self._meta: list[Optional[dict]] = [None] * capacity

        # Episode boundary tracking — stores the episode_id for each slot.
        # Used by sample_sequence() to avoid crossing episode boundaries.
        self._episode_ids: list[int] = [-1] * capacity
        self._current_episode_id: int = 0

        # Thread safety
        self._lock = threading.Lock()
        self._step = 0

        # ── Sequential-mode state (for offline training) ────────
        self._sequential_mode: bool = False
        self._seq_cursor: int = 0
        self._seq_indices: list[int] = []  # populated indices in order

    @property
    def size(self) -> int:
        return self.tree.n_entries

    @property
    def beta(self) -> float:
        """Annealed importance-sampling exponent."""
        frac = min(1.0, self._step / max(1, self.beta_anneal_steps))
        return self.beta_start + frac * (self.beta_end - self.beta_start)

    # ── Sequential mode (offline training) ──────────────────────

    def enable_sequential_mode(self) -> None:
        """Switch ``sample()`` to return transitions in temporal order.

        Builds an ordered index of all populated slots so each call to
        ``sample(batch_size)`` yields the *next* batch in data-index
        order (i.e. the order transitions were originally recorded).
        This preserves temporal coherence for Jamba/JEPA models.

        Call :meth:`reset_sequential` at the start of each epoch.
        """
        with self._lock:
            self._seq_indices = [
                i for i in range(self.capacity) if self._meta[i] is not None
            ]
            self._seq_cursor = 0
            self._sequential_mode = True
        log.info(
            "Sequential mode enabled — %d transitions in temporal order.",
            len(self._seq_indices),
        )

    def disable_sequential_mode(self) -> None:
        """Revert ``sample()`` to prioritized random sampling."""
        self._sequential_mode = False
        self._seq_cursor = 0
        self._seq_indices = []

    def reset_sequential(self) -> None:
        """Reset the sequential cursor to the beginning (new epoch)."""
        self._seq_cursor = 0

    @property
    def sequential_exhausted(self) -> bool:
        """True when the sequential cursor has reached the end."""
        return self._seq_cursor >= len(self._seq_indices)

    @property
    def sequential_remaining(self) -> int:
        """Number of transitions left in the current sequential pass."""
        return max(0, len(self._seq_indices) - self._seq_cursor)

    _PREFETCH_AHEAD: int = 2  # number of chunks to keep loaded ahead

    def _prefetch_sequential(self) -> None:
        """Ensure the current and next few chunks are in memory.

        Looks at upcoming indices (up to ``_PREFETCH_AHEAD`` chunks
        worth) and calls :meth:`_ChunkStore.ensure_chunks` so only
        a small rolling window of data lives in RAM at any time.
        Chunks that the cursor has passed are evicted automatically.
        """
        if not self._seq_indices:
            return
        lookahead_end = min(
            self._seq_cursor + self._PREFETCH_AHEAD * CHUNK_SIZE,
            len(self._seq_indices),
        )
        upcoming = self._seq_indices[self._seq_cursor : lookahead_end]
        if not upcoming:
            return
        needed: set[int] = {idx // CHUNK_SIZE for idx in upcoming}
        self._store.ensure_chunks(needed)

    def _sample_sequential(
        self, batch_size: int, device: str = "cpu",
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[int]]:
        """Return the next *batch_size* transitions in index order.

        Weights are uniform (1.0) — importance sampling is irrelevant
        for ordered full-pass training.
        """
        # Rolling prefetch: load upcoming chunks, evict consumed ones
        self._prefetch_sequential()

        transitions: List[Dict[str, Any]] = []
        indices: List[int] = []

        with self._lock:
            while len(transitions) < batch_size and self._seq_cursor < len(self._seq_indices):
                data_idx = self._seq_indices[self._seq_cursor]
                self._seq_cursor += 1

                blob = self._store.read(data_idx)
                if blob is None:
                    continue

                try:
                    trans = decompress_transition(
                        blob, method=self.compression, device=device,
                    )
                except (RuntimeError, Exception):
                    continue
                transitions.append(trans)
                indices.append(data_idx + self.tree.capacity)

        weights = np.ones(len(transitions), dtype=np.float32)
        return transitions, weights, indices

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

        # Check if this is a demo transition
        is_demo = False
        demo_val = transition.get("is_demo")
        if demo_val is not None:
            v = demo_val.item() if hasattr(demo_val, "item") else float(demo_val)
            is_demo = v > 0.5

        with self._lock:
            tree_idx = self.tree.add(priority)
            data_idx = tree_idx - self.tree.capacity

            if is_demo:
                # Archive demo transitions separately (append-only, never pruned)
                self._demo_store.write(self._demo_write_idx, blob)
                self._demo_write_idx += 1
            else:
                # Write blob into chunk file (overwrites old slot)
                self._store.write(data_idx, blob)

            self._meta[data_idx] = metadata or {}
            self._episode_ids[data_idx] = self._current_episode_id
            self.max_priority = max(self.max_priority, priority)
            self._step += 1

        # Check disk cap (only prunes AI store, never demos)
        if self._store.disk_bytes > self.disk_cap_gb * 1e9:
            self._prune_oldest_chunks()

    def mark_episode_boundary(self) -> None:
        """Signal that the current episode has ended.

        Call this when the environment resets.  Subsequent calls to
        :meth:`add` will tag transitions with a new episode id so
        :meth:`sample_sequence` never crosses episode boundaries.
        """
        with self._lock:
            self._current_episode_id += 1

    def sample(
        self,
        batch_size: int,
        device: str = "cpu",
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[int]]:
        """
        Sample a prioritized minibatch.

        If sequential mode is enabled (see :meth:`enable_sequential_mode`),
        returns the next ``batch_size`` transitions in temporal order
        instead of random prioritized sampling.

        Returns:
            transitions: List of decompressed transition dicts.
            weights: (batch_size,) importance-sampling weights.
            indices: List of tree indices (for priority updates).
        """
        # ── Sequential mode: delegate to temporal-order sampler ──
        if self._sequential_mode:
            return self._sample_sequential(batch_size, device)

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

                try:
                    trans = decompress_transition(blob, method=self.compression, device=device)
                except (RuntimeError, Exception):
                    continue
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

        NOTE: Currently unused — LearnerThread._compute_rebel_loss()
        does inline half-batch pairing instead of calling this method.
        Kept as a public API for external / offline REBEL training.

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
        """Remove the oldest AI chunk files and zero their SumTree priorities.

        Only prunes from the AI store — demo archive is never touched.
        """
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
            "demo_disk_mb": self._demo_store.disk_bytes / (1024 ** 2),
            "demo_count": self._demo_write_idx,
            "disk_cap_gb": self.disk_cap_gb,
            "total_priority": float(self.tree.total),
            "step": self._step,
            "chunk_files": self._store._num_chunks,
            "episode_id": self._current_episode_id,
        }

    def rebuild_from_disk(self, default_priority: float = 1.0) -> int:
        """Rebuild the SumTree and metadata from existing chunk files.

        Scans every **existing** chunk file on disk, reads slot headers,
        and inserts a uniform priority for each non-empty slot.  This
        makes the buffer usable for offline training without first
        collecting new data.

        Call this **once** after construction when you want to re-use
        replay data from a previous session (e.g. ``--offline`` mode).

        Args:
            default_priority: Priority assigned to every recovered
                transition (uniform sampling is usually fine for
                offline epochs).

        Returns:
            Number of transitions recovered.
        """
        recovered = 0
        alpha_priority = max(default_priority, self.min_priority) ** self.alpha

        # Discover which chunk files actually exist on disk so we only
        # read those instead of probing all capacity/CHUNK_SIZE paths
        # (which is extremely slow on network drives).
        existing_chunks: list[int] = []
        for f in self._store._dir.glob("chunk_*.bin"):
            try:
                chunk_id = int(f.stem.split("_")[1])
                existing_chunks.append(chunk_id)
            except (IndexError, ValueError):
                continue
        existing_chunks.sort()

        # Also discover demo chunks to inject into the training buffer
        demo_chunks: list[int] = []
        for f in self._demo_store._dir.glob("chunk_*.bin"):
            try:
                demo_chunks.append(int(f.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        demo_chunks.sort()

        log.info(
            "rebuild_from_disk: found %d AI + %d demo chunk files, scanning...",
            len(existing_chunks), len(demo_chunks),
        )

        max_data_idx = 0

        with self._lock:
            for chunk_id in existing_chunks:
                # Read the chunk header to find populated slots
                header = self._store._read_header(chunk_id)
                base_idx = chunk_id * CHUNK_SIZE
                chunk_recovered = 0

                for slot, (off, ln) in enumerate(header):
                    if ln == 0:
                        continue  # empty slot

                    data_idx = base_idx + slot
                    if data_idx >= self.capacity:
                        break

                    tree_idx = data_idx + self.tree.capacity
                    self.tree.update(tree_idx, alpha_priority)
                    self.tree.n_entries = min(
                        self.tree.n_entries + 1, self.capacity,
                    )
                    if data_idx >= max_data_idx:
                        max_data_idx = data_idx + 1

                    self._meta[data_idx] = {}
                    self._episode_ids[data_idx] = data_idx
                    recovered += 1
                    chunk_recovered += 1

                if chunk_recovered > 0:
                    log.info(
                        "  ai/chunk_%04d: %d transitions recovered",
                        chunk_id, chunk_recovered,
                    )

            # Inject demo archive transitions into the training buffer
            # with boosted priority so they're sampled more often.
            # A marker file tracks how many demos were previously
            # injected so we never duplicate them across rebuilds.
            _injection_marker = self._store._dir / "_demo_injection_count.txt"
            _prev_injected = 0
            if _injection_marker.exists():
                try:
                    _prev_injected = int(_injection_marker.read_text().strip())
                except (ValueError, OSError):
                    _prev_injected = 0

            demo_priority = max(default_priority * 5.0, self.min_priority) ** self.alpha
            demo_injected = 0
            demo_cursor = 0  # absolute index across all demo chunks
            for chunk_id in demo_chunks:
                header = self._demo_store._read_header(chunk_id)
                base_idx = chunk_id * CHUNK_SIZE
                with open(self._demo_store._chunk_path(chunk_id), "rb") as f:
                    f.seek(self._demo_store._header_size())
                    blob_region = f.read()

                for slot, (off, ln) in enumerate(header):
                    if ln == 0:
                        continue
                    # Skip demos that were already injected in a
                    # previous rebuild (they're already in the AI
                    # store's chunk files on disk).
                    if demo_cursor < _prev_injected:
                        demo_cursor += 1
                        continue
                    demo_cursor += 1

                    blob = blob_region[off : off + ln]
                    # Write into next available slot in the AI store
                    data_idx = (max_data_idx + demo_injected) % self.capacity
                    self._store.write(data_idx, blob)
                    tree_idx = data_idx + self.tree.capacity
                    self.tree.update(tree_idx, demo_priority)
                    self.tree.n_entries = min(
                        self.tree.n_entries + 1, self.capacity,
                    )
                    self._meta[data_idx] = {"is_demo": True}
                    self._episode_ids[data_idx] = data_idx
                    demo_injected += 1

                if demo_injected > 0:
                    log.info(
                        "  demos/chunk_%04d: injected transitions",
                        chunk_id,
                    )

            # Persist the total number of demos now in the AI store
            # so the next rebuild skips them.
            try:
                _injection_marker.write_text(str(_prev_injected + demo_injected))
            except OSError:
                log.warning("Could not write demo injection marker.")

            if demo_injected > 0:
                max_data_idx = max_data_idx + demo_injected
                recovered += demo_injected
                log.info(
                    "Injected %d new demo transitions (skipped %d already "
                    "present) with %.1fx priority boost.",
                    demo_injected, _prev_injected, 5.0,
                )
            elif _prev_injected > 0:
                log.info(
                    "All %d demo transitions already in AI store — "
                    "skipped injection.", _prev_injected,
                )

            # Set data_pointer past the highest recovered index so new
            # add() calls don't overwrite recovered data prematurely.
            self.tree.data_pointer = max_data_idx % self.capacity

            # Assign contiguous episode ids in index order so that
            # neighbouring slots are likely from the same episode.
            # We mark a boundary every time there is a gap (empty slot)
            # in the index sequence.
            if recovered > 0:
                ep_id = 0
                prev_was_populated = False
                for data_idx in range(self.capacity):
                    if self._episode_ids[data_idx] >= 0 and self._meta[data_idx] is not None:
                        if not prev_was_populated:
                            ep_id += 1
                        self._episode_ids[data_idx] = ep_id
                        prev_was_populated = True
                    else:
                        prev_was_populated = False
                self._current_episode_id = ep_id + 1

        log.info(
            "Rebuilt replay buffer from disk: %d transitions recovered "
            "(AI: %.1f MB, demos: %.1f MB on disk).",
            recovered,
            self._store.disk_bytes / (1024 ** 2),
            self._demo_store.disk_bytes / (1024 ** 2),
        )
        return recovered

    # ── Sequence sampling ────────────────────────────────────────

    def sample_sequence(
        self,
        batch_size: int,
        seq_len: int = 8,
        device: str = "cpu",
    ) -> Tuple[List[List[Dict[str, Any]]], np.ndarray, List[int]]:
        """Sample contiguous sub-sequences from the replay buffer.

        Each returned item is a **list** of ``seq_len`` consecutive
        transitions belonging to the **same episode**.  This enables
        n-step return computation and temporal-credit assignment.

        Sampling strategy:
        1. Draw ``batch_size`` anchor indices via the same prioritized
           sampling as :meth:`sample`.
        2. For each anchor, extend forward up to ``seq_len`` steps,
           stopping if an episode boundary is crossed or the slot is
           empty.
        3. Sequences shorter than 2 are discarded and re-sampled (up
           to 3 retries).

        Args:
            batch_size: Number of sequences to return.
            seq_len: Maximum length of each sub-sequence.
            device: Device for decompressed tensors.

        Returns:
            sequences:  List[List[dict]] — each inner list has up to
                        ``seq_len`` consecutive transition dicts.
            weights:    (batch_size,) importance-sampling weights
                        (based on the anchor transition).
            indices:    Tree indices of the anchor transitions.
        """
        if self.size < batch_size:
            raise ValueError(f"Buffer has {self.size} items, need {batch_size}")

        segment = self.tree.total / batch_size
        beta = self.beta

        sequences: List[List[Dict[str, Any]]] = []
        weights: List[float] = []
        indices: List[int] = []

        max_retries = 3

        with self._lock:
            for i in range(batch_size):
                for _retry in range(max_retries):
                    lo = segment * i + segment * _retry * 0.01  # slight jitter on retry
                    hi = segment * (i + 1)
                    s = random.uniform(lo, hi)
                    tree_idx, priority, data_idx = self.tree.get(s)

                    anchor_ep = self._episode_ids[data_idx]
                    if anchor_ep < 0:
                        continue  # empty slot

                    seq: List[Dict[str, Any]] = []
                    for offset in range(seq_len):
                        idx = (data_idx + offset) % self.capacity
                        # Stop at episode boundary
                        if self._episode_ids[idx] != anchor_ep:
                            break
                        blob = self._store.read(idx)
                        if blob is None:
                            break
                        try:
                            trans = decompress_transition(
                                blob, method=self.compression, device=device,
                            )
                        except (RuntimeError, Exception):
                            break
                        seq.append(trans)

                    if len(seq) >= 2:
                        sequences.append(seq)
                        prob = priority / max(self.tree.total, 1e-8)
                        w = (prob * self.size) ** (-beta)
                        weights.append(w)
                        indices.append(tree_idx)
                        break
                else:
                    # All retries exhausted — fall back to single-step
                    s = random.uniform(0, self.tree.total)
                    tree_idx, priority, data_idx = self.tree.get(s)
                    blob = self._store.read(data_idx)
                    if blob is not None:
                        try:
                            trans = decompress_transition(
                                blob, method=self.compression, device=device,
                            )
                        except (RuntimeError, Exception):
                            continue
                        sequences.append([trans])
                        prob = priority / max(self.tree.total, 1e-8)
                        w = (prob * self.size) ** (-beta)
                        weights.append(w)
                        indices.append(tree_idx)

        w_arr = np.array(weights, dtype=np.float32)
        if w_arr.size > 0:
            w_arr /= w_arr.max()

        return sequences, w_arr, indices
