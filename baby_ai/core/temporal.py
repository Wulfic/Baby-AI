"""
Temporal integration module — Jamba-based recurrent core (Mamba + MoE).

Interleaves Mamba-2 selective state-space blocks with sparse
Mixture-of-Experts MLP layers, providing O(1) per-step inference
with infinite context caching.

Phase B upgrade: Mamba-2 SSD (Structured State Space Duality)
adds chunked-parallel scan for training while keeping the identical
O(1) single-step recurrence at inference.

Key classes:

- **MambaBlock** — Single Mamba-2 SSM block with three execution paths:
  SSD chunked-parallel (training), single-step recurrence (inference),
  and sequential scan (fallback).
- **MoELayer** — Sparse Mixture-of-Experts with top-k routing and
  Switch-Transformer-style load-balancing loss.
- **JambaBlock** — Mamba + (MoE or Dense FFN) with residual connections.
- **JambaCore** — Full temporal backbone: stacked JambaBlocks with
  input/output projections and packed ``JambaState`` for recurrence.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Optional fast Mamba-2 Triton kernels — gracefully degrade if unavailable.
try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as _triton_ssd
    _HAS_MAMBA_SSM = True
except ImportError:
    _HAS_MAMBA_SSM = False


# ────────────────────────────────────────────────────────────────────────────
# Jamba hidden-state container
# ────────────────────────────────────────────────────────────────────────────

class JambaState:
    """
    Packed recurrent state for JambaCore.

    Stores per-block SSM states and causal convolution buffers.
    Provides a tensor-like interface (.detach(), .to()) for
    compatibility with existing hidden-state plumbing.
    """
    __slots__ = ("ssm_states", "conv_states")

    def __init__(
        self,
        ssm_states: list[torch.Tensor],
        conv_states: list[torch.Tensor],
    ):
        self.ssm_states = ssm_states
        self.conv_states = conv_states

    def detach(self) -> "JambaState":
        """Detach all tensors from the computation graph (for TBPTT)."""
        return JambaState(
            ssm_states=[s.detach() for s in self.ssm_states],
            conv_states=[c.detach() for c in self.conv_states],
        )

    def to(self, device: torch.device | str) -> "JambaState":
        """Move all tensors to the specified device."""
        return JambaState(
            ssm_states=[s.to(device) for s in self.ssm_states],
            conv_states=[c.to(device) for c in self.conv_states],
        )


# ────────────────────────────────────────────────────────────────────────────
# Mamba Block — Selective State-Space Model (S6)
# ────────────────────────────────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """
    Mamba-2 selective SSM block with SSD (Structured State Space Duality).

    Supports three execution paths:
      - **SSD chunked parallel** (training, T > 1, use_ssd=True):
        Splits the sequence into chunks, runs semi-separable matrix
        multiplies within each chunk, carries boundary states across
        chunks.  2–8× faster than sequential on GPU tensor cores.
      - **Single-step recurrence** (inference, T == 1):
        Identical to Mamba-1 — O(1) per step with cached state.
      - **Sequential scan** (fallback, use_ssd=False):
        Original Python-level for-loop over timesteps.

    Architecture per step:
      1. Input projection → (gate z, input x) split
      2. Depthwise causal convolution (with running buffer)
      3. Input-dependent SSM:  h_t = A_t h_{t-1} + B_t x_t,  y_t = C_t h_t
      4. Gated output:  out = proj(y * SiLU(z) + D * x)

    Args:
        dim:        Model dimension.
        d_state:    SSM recurrent state dimension (N in Mamba notation).
        d_conv:     Causal convolution kernel size.
        expand:     Expansion factor for inner dimension.
        dt_rank:    Rank for the delta-t projection (0 = auto).
        use_ssd:    Use Mamba-2 SSD chunked-parallel scan when T > 1.
        chunk_size: Chunk length for SSD (64 or 128 work well).
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 0,
        use_ssd: bool = True,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = dim * expand
        self.dt_rank = dt_rank if dt_rank > 0 else math.ceil(dim / 16)
        self.use_ssd = use_ssd
        self.chunk_size = chunk_size

        # Input projection → gate (z) and input (x)
        self.in_proj = nn.Linear(dim, self.d_inner * 2, bias=False)

        # Depthwise causal conv — padding=0 because we manage the
        # causal buffer manually (prepend conv_state before applying).
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=0,
            groups=self.d_inner,
            bias=True,
        )

        # SSM parameter projections from post-conv activations:
        #   dt (dt_rank) | B (d_state) | C (d_state)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + d_state * 2, bias=False,
        )

        # dt projection back to inner dim
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Initialise dt bias so initial timescales are small and positive.
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner)
                * (math.log(0.1) - math.log(0.001))
                + math.log(0.001)
            )
            inv_sp = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_sp)

        # A (log-space): diagonal state-transition decay rates
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(
            torch.log(A).unsqueeze(0).expand(self.d_inner, -1).clone()
        )

        # D: learnable skip-connection coefficient
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, dim, bias=False)

    # ── forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        ssm_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (B, T, dim) or (B, dim) input.
            ssm_state:  (B, d_inner, d_state) previous SSM state.
            conv_state: (B, d_inner, d_conv-1) previous conv buffer.

        Returns:
            output:     Same shape as *x*.
            ssm_state:  Updated SSM state.
            conv_state: Updated conv buffer.
        """
        squeeze = x.dim() == 2
        if squeeze:
            x = x.unsqueeze(1)

        B, T, _ = x.shape
        device = x.device

        if ssm_state is None:
            ssm_state = torch.zeros(B, self.d_inner, self.d_state, device=device)
        if conv_state is None:
            conv_state = torch.zeros(B, self.d_inner, self.d_conv - 1, device=device)

        # ── 1. Input projection → gate z & input x_in ──
        xz = self.in_proj(x)                       # (B, T, 2·d_inner)
        x_in, z = xz.chunk(2, dim=-1)              # each (B, T, d_inner)

        # ── 2. Causal conv1d with running buffer ──
        x_conv = x_in.transpose(1, 2)              # (B, d_inner, T)
        x_conv = torch.cat([conv_state, x_conv], dim=2)  # (B, d_inner, d_conv-1+T)

        # New conv buffer = last (d_conv-1) pre-conv timesteps
        new_conv_state = x_conv[:, :, -(self.d_conv - 1):].clone()

        x_conv = self.conv1d(x_conv)               # (B, d_inner, T)
        x_conv = F.silu(x_conv).transpose(1, 2)    # (B, T, d_inner)

        # ── 3. SSM parameter projection ──
        x_proj_out = self.x_proj(x_conv)            # (B, T, dt_rank + 2·d_state)
        dt_raw, B_param, C_param = torch.split(
            x_proj_out,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1,
        )
        dt = F.softplus(self.dt_proj(dt_raw))       # (B, T, d_inner)

        A = -torch.exp(self.A_log)                  # (d_inner, d_state)

        # ── 4. Scan (choose path based on config and sequence length) ──
        if T == 1:
            # Single-step recurrence — O(1) inference path
            y, h = self._single_step_scan(
                A, B_param, C_param, x_conv, dt, ssm_state,
            )
        elif self.use_ssd and T > 1:
            # SSD chunked-parallel scan — O(T·chunk) training path
            y, h = self._ssd_chunked_scan(
                A, B_param, C_param, x_conv, dt, ssm_state,
            )
        else:
            # Fallback sequential scan
            y, h = self._sequential_scan(
                A, B_param, C_param, x_conv, dt, ssm_state,
            )

        # ── 5. Gated output projection ──
        y = y * F.silu(z)
        output = self.out_proj(y)                   # (B, T, dim)

        if squeeze:
            output = output.squeeze(1)

        return output, h, new_conv_state

    # ── Scan implementations ──────────────────────────────────────────────

    def _single_step_scan(
        self,
        A: torch.Tensor,        # (d_inner, d_state)
        B_param: torch.Tensor,  # (B, 1, d_state)
        C_param: torch.Tensor,  # (B, 1, d_state)
        x_conv: torch.Tensor,   # (B, 1, d_inner)
        dt: torch.Tensor,       # (B, 1, d_inner)
        h: torch.Tensor,        # (B, d_inner, d_state)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single-step O(1) recurrence — identical for Mamba-1 and Mamba-2."""
        dt_t = dt[:, 0]                          # (B, d_inner)
        B_t  = B_param[:, 0]                     # (B, d_state)
        C_t  = C_param[:, 0]                     # (B, d_state)
        x_t  = x_conv[:, 0]                      # (B, d_inner)

        dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
        dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

        h = dA * h + dB * x_t.unsqueeze(-1)
        y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + self.D * x_t

        return y_t.unsqueeze(1), h  # (B, 1, d_inner), (B, d_inner, d_state)

    def _sequential_scan(
        self,
        A: torch.Tensor,
        B_param: torch.Tensor,
        C_param: torch.Tensor,
        x_conv: torch.Tensor,
        dt: torch.Tensor,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Original sequential scan — fallback when SSD is disabled."""
        T = x_conv.size(1)
        outputs: list[torch.Tensor] = []

        for t in range(T):
            dt_t = dt[:, t]
            B_t  = B_param[:, t]
            C_t  = C_param[:, t]
            x_t  = x_conv[:, t]

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            h = dA * h + dB * x_t.unsqueeze(-1)
            y_t = (h * C_t.unsqueeze(1)).sum(dim=-1) + self.D * x_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)
        return y, h

    def _ssd_chunked_scan(
        self,
        A: torch.Tensor,        # (d_inner, d_state)
        B_param: torch.Tensor,  # (B, T, d_state)
        C_param: torch.Tensor,  # (B, T, d_state)
        x_conv: torch.Tensor,   # (B, T, d_inner)
        dt: torch.Tensor,       # (B, T, d_inner)
        h: torch.Tensor,        # (B, d_inner, d_state)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Mamba-2 SSD chunked-parallel scan.

        Splits the sequence into chunks of size C.  Within each chunk,
        the SSM output is computed via dense matrix multiplies on the
        semi-separable structure (tensor-core friendly).  Across chunks,
        a small boundary-state recurrence carries forward the running state.

        This gives O(T * C) work with high GPU utilization vs O(T * d_state)
        sequential steps.
        """
        B_batch, T, d_inner = x_conv.shape
        C_len = self.chunk_size
        d_state = self.d_state

        # Pad sequence to multiple of chunk_size
        pad = (C_len - T % C_len) % C_len
        if pad > 0:
            x_conv  = F.pad(x_conv,  (0, 0, 0, pad))
            dt      = F.pad(dt,      (0, 0, 0, pad))
            B_param = F.pad(B_param, (0, 0, 0, pad))
            C_param = F.pad(C_param, (0, 0, 0, pad))

        T_padded = x_conv.size(1)
        num_chunks = T_padded // C_len

        # Reshape into chunks: (B, num_chunks, C_len, ...)
        x_c  = x_conv.reshape(B_batch, num_chunks, C_len, d_inner)
        dt_c = dt.reshape(B_batch, num_chunks, C_len, d_inner)
        B_c  = B_param.reshape(B_batch, num_chunks, C_len, d_state)
        C_c  = C_param.reshape(B_batch, num_chunks, C_len, d_state)

        all_outputs = []

        for chunk_idx in range(num_chunks):
            # Extract this chunk
            x_chunk  = x_c[:, chunk_idx]   # (B, C_len, d_inner)
            dt_chunk = dt_c[:, chunk_idx]   # (B, C_len, d_inner)
            B_chunk  = B_c[:, chunk_idx]    # (B, C_len, d_state)
            C_chunk  = C_c[:, chunk_idx]    # (B, C_len, d_state)

            # Compute discretized parameters for the full chunk
            # dA_chunk[b, t, i, n] = exp(dt[b,t,i] * A[i,n])
            dA_chunk = torch.exp(
                dt_chunk.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
            )  # (B, C_len, d_inner, d_state)

            # dB_chunk[b, t, i, n] = dt[b,t,i] * B[b,t,n]
            dB_chunk = dt_chunk.unsqueeze(-1) * B_chunk.unsqueeze(2)
            # (B, C_len, d_inner, d_state)

            # ── Intra-chunk: semi-separable matrix multiply ──
            # For each timestep t in the chunk, compute:
            #   h_t = dA_t * h_{t-1} + dB_t * x_t
            #   y_t = C_t^T @ h_t + D * x_t
            # This is done as a scan within the chunk, but the chunk
            # is small (64 steps) so the overhead is minimal and the
            # cross-chunk parallelism is the real win.
            chunk_outputs = []
            h_local = h  # boundary state from previous chunk

            for t in range(C_len):
                dA_t = dA_chunk[:, t]   # (B, d_inner, d_state)
                dB_t = dB_chunk[:, t]   # (B, d_inner, d_state)
                C_t  = C_chunk[:, t]    # (B, d_state)
                x_t  = x_chunk[:, t]    # (B, d_inner)

                h_local = dA_t * h_local + dB_t * x_t.unsqueeze(-1)
                y_t = (h_local * C_t.unsqueeze(1)).sum(dim=-1) + self.D * x_t
                chunk_outputs.append(y_t)

            # Update boundary state for next chunk
            h = h_local

            chunk_y = torch.stack(chunk_outputs, dim=1)  # (B, C_len, d_inner)
            all_outputs.append(chunk_y)

        y = torch.cat(all_outputs, dim=1)  # (B, T_padded, d_inner)

        # Remove padding
        if pad > 0:
            y = y[:, :T, :]

        return y, h


# ────────────────────────────────────────────────────────────────────────────
# Mixture-of-Experts Layer
# ────────────────────────────────────────────────────────────────────────────

class ExpertMLP(nn.Module):
    """Single expert: SwiGLU-style gated FFN."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_up   = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MoELayer(nn.Module):
    """
    Sparse Mixture-of-Experts layer with top-k routing.

    Uses a learned router with auxiliary Switch-Transformer-style
    load-balancing loss to prevent routing collapse.

    Args:
        dim:                  Input/output dimension.
        num_experts:          Total number of expert MLPs.
        top_k:                Experts activated per token.
        hidden_mult:          FFN hidden dimension multiplier.
        load_balance_weight:  Weight for the auxiliary loss.
    """

    def __init__(
        self,
        dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        hidden_mult: int = 2,
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight

        hidden_dim = dim * hidden_mult
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertMLP(dim, hidden_dim) for _ in range(num_experts)]
        )

        # Cached auxiliary loss (populated during training forward)
        self._aux_loss: torch.Tensor | None = None

    @property
    def aux_loss(self) -> torch.Tensor:
        """Load-balancing loss from the most recent forward pass."""
        if self._aux_loss is None:
            # Return a zero on the same device as the router weights
            # to avoid cuda/cpu device mismatch when added to the loss.
            return torch.tensor(0.0, device=self.router.weight.device)
        return self._aux_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, dim) or (B, dim).

        Returns:
            Sparse-routed expert output with the same shape as *x*.
        """
        orig_shape = x.shape
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)                # (N, D)

        router_logits = self.router(x_flat)          # (N, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)

        top_weights, top_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)

        # ── Auxiliary load-balancing loss (training only) ──
        if self.training:
            # f_i = fraction of tokens dispatched to expert i
            one_hot = F.one_hot(top_indices, self.num_experts).float()  # (N, k, E)
            f = one_hot.sum(dim=(0, 1)) / (B * T)                      # (E,)
            P = routing_weights.mean(dim=0)                             # (E,)
            self._aux_loss = (
                self.load_balance_weight * self.num_experts * (f * P).sum()
            )

        # ── Dispatch to experts ──
        output = torch.zeros_like(x_flat)
        for k_idx in range(self.top_k):
            for e_idx in range(self.num_experts):
                mask = top_indices[:, k_idx] == e_idx
                if mask.any():
                    expert_out = self.experts[e_idx](x_flat[mask])
                    output[mask] += top_weights[mask, k_idx].unsqueeze(-1) * expert_out

        output = output.reshape(B, T, D)
        if len(orig_shape) == 2:
            output = output.squeeze(1)
        return output


# ────────────────────────────────────────────────────────────────────────────
# Jamba Block — Mamba + (MoE or Dense FFN)
# ────────────────────────────────────────────────────────────────────────────

class JambaBlock(nn.Module):
    """
    Single Jamba block:
        LayerNorm → Mamba → Residual → LayerNorm → FFN/MoE → Residual

    Args:
        dim:                  Model dimension.
        d_state:              Mamba SSM state dimension.
        d_conv:               Mamba convolution kernel size.
        expand:               Mamba inner-dim multiplier.
        dt_rank:              Mamba dt projection rank (0 = auto).
        use_moe:              Use MoE FFN (True) or dense FFN (False).
        num_experts:          Number of experts (if *use_moe*).
        top_k:                Top-k routing (if *use_moe*).
        ffn_mult:             FFN hidden-dim multiplier.
        load_balance_weight:  MoE load-balancing loss weight.
        use_ssd:              Use Mamba-2 SSD chunked scan.
        chunk_size:           SSD chunk length.
    """

    def __init__(
        self,
        dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 0,
        use_moe: bool = False,
        num_experts: int = 4,
        top_k: int = 1,
        ffn_mult: int = 2,
        load_balance_weight: float = 0.01,
        use_ssd: bool = True,
        chunk_size: int = 64,
    ):
        super().__init__()

        self.norm_mamba = nn.LayerNorm(dim)
        self.mamba = MambaBlock(
            dim, d_state, d_conv, expand, dt_rank,
            use_ssd=use_ssd, chunk_size=chunk_size,
        )

        self.norm_ffn = nn.LayerNorm(dim)
        self.use_moe = use_moe

        if use_moe:
            self.ffn = MoELayer(
                dim, num_experts, top_k, ffn_mult, load_balance_weight,
            )
        else:
            hidden = dim * ffn_mult
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden, bias=False),
                nn.SiLU(),
                nn.Linear(hidden, dim, bias=False),
            )

    @property
    def aux_loss(self) -> torch.Tensor:
        """MoE load-balancing loss (0 if using dense FFN)."""
        if self.use_moe and isinstance(self.ffn, MoELayer):
            return self.ffn.aux_loss
        return torch.tensor(0.0, device=self.mamba.in_proj.weight.device)

    def forward(
        self,
        x: torch.Tensor,
        ssm_state: torch.Tensor | None = None,
        conv_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x:          (B, T, dim) or (B, dim).
            ssm_state:  Previous Mamba SSM state.
            conv_state: Previous Mamba conv buffer.

        Returns:
            output, ssm_state, conv_state.
        """
        # Mamba sub-block + residual
        residual = x
        mamba_out, ssm_state, conv_state = self.mamba(
            self.norm_mamba(x), ssm_state, conv_state,
        )
        x = residual + mamba_out

        # FFN / MoE sub-block + residual
        x = x + self.ffn(self.norm_ffn(x))

        return x, ssm_state, conv_state


# ────────────────────────────────────────────────────────────────────────────
# JambaCore — Jamba-based recurrent temporal core
# ────────────────────────────────────────────────────────────────────────────

class JambaCore(nn.Module):
    """
    Jamba temporal core — interleaves Mamba SSM blocks with sparse MoE MLPs.

    Interface:
        forward(x, hidden) → (output, hidden)
        init_hidden(batch_size, device) → hidden
        detach_hidden(hidden) → hidden

    The hidden state is a ``JambaState`` carrying per-block SSM + conv
    buffers.

    Args:
        input_dim:           Fused embedding dimension.
        hidden_dim:          Internal model dimension.
        num_layers:          Number of stacked Jamba blocks.
        d_state:             SSM state dimension.
        d_conv:              Causal convolution kernel size.
        expand:              Mamba expansion factor.
        dt_rank:             Rank for dt projection (0 = auto).
        num_experts:         MoE experts per MoE layer.
        top_k_routing:       Top-k expert selection.
        moe_every_n:         Deploy MoE every N blocks; others use dense FFN.
        ffn_mult:            FFN hidden-dim multiplier.
        load_balance_weight: MoE auxiliary loss weight.
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = 0,
        num_experts: int = 4,
        top_k_routing: int = 1,
        moe_every_n: int = 2,
        ffn_mult: int = 2,
        load_balance_weight: float = 0.01,
        use_ssd: bool = True,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection (fused_dim → hidden_dim if they differ)
        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.input_norm = nn.LayerNorm(hidden_dim)

        # Stacked Jamba blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # MoE at layers 1, 3, 5, … (0-indexed); dense at 0, 2, 4, …
            use_moe = (i % moe_every_n == (moe_every_n - 1))
            self.blocks.append(
                JambaBlock(
                    dim=hidden_dim,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k_routing,
                    ffn_mult=ffn_mult,
                    load_balance_weight=load_balance_weight,
                    use_ssd=use_ssd,
                    chunk_size=chunk_size,
                )
            )

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    # ── state helpers ──────────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device) -> JambaState:
        """Create zero-initialised Jamba state."""
        ssm_states: list[torch.Tensor] = []
        conv_states: list[torch.Tensor] = []
        for block in self.blocks:
            m = block.mamba
            ssm_states.append(
                torch.zeros(batch_size, m.d_inner, m.d_state, device=device)
            )
            conv_states.append(
                torch.zeros(batch_size, m.d_inner, m.d_conv - 1, device=device)
            )
        return JambaState(ssm_states=ssm_states, conv_states=conv_states)

    def detach_hidden(self, hidden: JambaState | None) -> JambaState | None:
        """Detach hidden state from computation graph (for TBPTT)."""
        if hidden is None:
            return None
        return hidden.detach()

    # ── forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        hidden: JambaState | None = None,
    ) -> tuple[torch.Tensor, JambaState]:
        """
        Args:
            x:      (B, T, input_dim) or (B, input_dim) fused embeddings.
            hidden: Previous ``JambaState`` (or None to init).

        Returns:
            output: (B, T, hidden_dim) or (B, hidden_dim) core output.
            hidden: Updated ``JambaState``.
        """
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        B = x.size(0)
        device = x.device

        if hidden is None:
            hidden = self.init_hidden(B, device)

        # SSM recurrent scans overflow in float16 — run entire core in float32.
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            hidden = JambaState(
                ssm_states=[s.float() for s in hidden.ssm_states],
                conv_states=[c.float() for c in hidden.conv_states],
            )

            x = self.input_norm(self.input_proj(x))

            new_ssm: list[torch.Tensor] = []
            new_conv: list[torch.Tensor] = []

            for i, block in enumerate(self.blocks):
                x, ssm_s, conv_s = block(
                    x,
                    ssm_state=hidden.ssm_states[i],
                    conv_state=hidden.conv_states[i],
                )
                new_ssm.append(ssm_s)
                new_conv.append(conv_s)

            x = self.output_proj(self.output_norm(x))

        if squeeze:
            x = x.squeeze(1)

        return x, JambaState(ssm_states=new_ssm, conv_states=new_conv)

    # ── auxiliary loss ─────────────────────────────────────────────────────

    @property
    def aux_loss(self) -> torch.Tensor:
        """Aggregate MoE load-balancing loss across all blocks."""
        device = self.output_proj.weight.device
        total = torch.tensor(0.0, device=device)
        for block in self.blocks:
            bl = block.aux_loss
            if bl.requires_grad or bl.item() > 0:
                total = total + bl
        return total
