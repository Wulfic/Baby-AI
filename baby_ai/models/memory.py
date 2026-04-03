"""
Titans-inspired differentiable K-V episodic memory.

A lightweight fixed-size ring buffer with:
  • Key   = projected observation latent  (key_dim)
  • Value = observation embedding         (value_dim)
  • Retrieval via scaled dot-product attention over the stored keys

This gives the agent a persistent external memory that survives across
Mamba's recurrent window, allowing it to recall distant events within an
episode (e.g. "I put iron ore in chest at position X two minutes ago").

Usage::

    mem = EpisodicMemory(mem_slots=64, key_dim=128, value_dim=512)
    mem.reset()                   # call at episode start
    mem.write(key, value)         # after each step
    out = mem.read(query)         # (B, value_dim) retrieved context
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EpisodicMemory(nn.Module):
    """
    Fixed-size ring-buffer K-V episodic memory with soft attention read.

    Args:
        mem_slots:  Number of slots in the ring buffer.
        key_dim:    Dimension of keys and queries.
        value_dim:  Dimension of stored values (= agent hidden_dim).
        input_dim:  Dimension of raw input to project to key/value.
                    If None, caller must supply pre-projected key/value.
    """

    def __init__(
        self,
        mem_slots: int = 64,
        key_dim: int = 128,
        value_dim: int = 512,
        input_dim: int | None = None,
    ):
        super().__init__()
        self.mem_slots = mem_slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.scale = math.sqrt(key_dim)

        if input_dim is not None:
            self.key_proj = nn.Linear(input_dim, key_dim)
            self.val_proj = nn.Linear(input_dim, value_dim)
        else:
            self.key_proj = None
            self.val_proj = None

        # Query projection: maps agent hidden state → query
        self.query_proj = nn.Linear(value_dim, key_dim)

        # Output gate: blend retrieved memory with current hidden state
        self.out_gate = nn.Sequential(
            nn.Linear(value_dim * 2, value_dim),
            nn.SiLU(),
            nn.Linear(value_dim, value_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(value_dim, value_dim)

        # Ring buffer state (registered as buffers so they move with .to())
        self.register_buffer(
            "_keys", torch.zeros(1, mem_slots, key_dim), persistent=False
        )
        self.register_buffer(
            "_values", torch.zeros(1, mem_slots, value_dim), persistent=False
        )
        self.register_buffer(
            "_write_ptr", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_filled", torch.zeros(1, dtype=torch.long), persistent=False
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def reset(self, batch_size: int = 1) -> None:
        """Clear the memory at the start of a new episode."""
        dev = self._keys.device
        self._keys = torch.zeros(batch_size, self.mem_slots, self.key_dim, device=dev)
        self._values = torch.zeros(batch_size, self.mem_slots, self.value_dim, device=dev)
        self._write_ptr = torch.zeros(batch_size, dtype=torch.long, device=dev)
        self._filled = torch.zeros(batch_size, dtype=torch.long, device=dev)

    # ── Write ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Write one entry per batch element into the ring buffer.

        Args:
            key:   (B, key_dim)
            value: (B, value_dim)
        """
        B = key.size(0)
        # Resize buffers if batch size changed (e.g. first write after reset)
        if self._keys.size(0) != B:
            self.reset(B)

        for b in range(B):
            ptr = self._write_ptr[b].item()
            self._keys[b, ptr] = key[b]
            self._values[b, ptr] = value[b]
            self._write_ptr[b] = (ptr + 1) % self.mem_slots
            self._filled[b] = min(self._filled[b].item() + 1, self.mem_slots)

    # ── Read ───────────────────────────────────────────────────────────────

    def read(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Retrieve memory conditioned on the agent's current hidden state.

        Args:
            hidden: (B, value_dim) current agent hidden state.

        Returns:
            (B, value_dim) memory-augmented representation (same dim).
        """
        B = hidden.size(0)
        if self._keys.size(0) != B:
            # Memory not initialised for this batch — return zeros
            return torch.zeros_like(hidden)

        query = self.query_proj(hidden)  # (B, key_dim)

        # Masked attention: only attend to filled slots
        # _filled is (B,); build mask (B, mem_slots)
        mask = torch.arange(self.mem_slots, device=hidden.device).unsqueeze(0)  # (1, S)
        filled = self._filled.unsqueeze(1)  # (B, 1)
        valid_mask = mask < filled  # (B, S) True where slot has data

        # Scaled dot-product attention
        # keys: (B, S, K), query: (B, K)
        scores = torch.bmm(
            self._keys,                 # (B, S, K)
            query.unsqueeze(-1),        # (B, K, 1)
        ).squeeze(-1) / self.scale     # (B, S)

        # Mask out unfilled slots
        scores = scores.masked_fill(~valid_mask, float("-inf"))

        # Softmax (returns zeros if all masked)
        attn = torch.softmax(scores, dim=-1)  # (B, S)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle all-inf edge case

        # Weighted sum of values
        retrieved = torch.bmm(
            attn.unsqueeze(1),  # (B, 1, S)
            self._values,       # (B, S, V)
        ).squeeze(1)            # (B, V)

        # Gated blend with current hidden state
        gate = self.out_gate(torch.cat([hidden, retrieved], dim=-1))  # (B, V)
        out = self.out_proj(hidden + gate * retrieved)                 # (B, V)
        return out

    # ── Convenience: project-then-write ───────────────────────────────────

    @torch.no_grad()
    def encode_and_write(self, x: torch.Tensor) -> None:
        """
        Project raw input x → (key, value) and write to memory.

        Only available when EpisodicMemory was constructed with input_dim.
        """
        if self.key_proj is None or self.val_proj is None:
            raise RuntimeError(
                "encode_and_write() requires input_dim to be set at construction."
            )
        key = self.key_proj(x)
        value = self.val_proj(x)
        self.write(key, value)
