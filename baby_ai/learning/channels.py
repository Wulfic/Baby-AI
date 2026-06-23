"""
Canonical reward-channel registry for Grounded Successor Features.

This module is the single source of truth for the ordered **cumulant
vector** ``φ ∈ ℝ^C`` used by the decomposed (successor-feature) value
head.  Each entry of ``φ`` is one raw reward channel (block_break,
crafting, idle_penalty, …) exactly as produced by the environment's
``RewardComputer`` and stored per transition in ``reward_channels``.

The ordering is derived from the UI weight sliders
(:data:`baby_ai.ui.reward_weights.TOP_LEVEL_WEIGHTS`) so that the
scalarisation ``φ · w`` reproduces the scalar reward **bit-for-bit**,
where ``w`` is the *signed* weight vector (penalty channels negated).

Why this matters
----------------
With grounded successor features the agent learns ``ψ(s) ∈ ℝ^C`` — the
expected discounted *future* value of each channel — instead of a single
opaque scalar.  The scalar value is recovered on demand as
``V(s) = ψ(s) · w``.  Because ``w`` is just the live UI weights, changing
a slider re-scalarises every state instantly (zero-shot) with no
retraining, and ``ψ(s)[i] · w[i]`` is a direct, readable answer to
"what is the agent being rewarded for".

.. warning::
   Checkpoints depend on :data:`REWARD_CHANNELS` ordering.  Only ever
   **append** new channels; never reorder or delete, or the ``ψ`` head's
   output dimension semantics will silently shift under loaded weights.
"""

from __future__ import annotations

import torch

from baby_ai.ui.reward_weights import TOP_LEVEL_WEIGHTS

# ── Canonical ordering ────────────────────────────────────────────────
# Derived from the top-level UI weight sliders (excludes sub-weights like
# mv_forward / int_impact, which are internal multipliers already folded
# into a channel's value by the RewardComputer).
REWARD_CHANNELS: tuple[str, ...] = tuple(w.key for w in TOP_LEVEL_WEIGHTS)
NUM_CHANNELS: int = len(REWARD_CHANNELS)
CHANNEL_INDEX: dict[str, int] = {k: i for i, k in enumerate(REWARD_CHANNELS)}

# Channels whose contribution is *subtracted* in the scalar reward.
PENALTY_CHANNELS: frozenset[str] = frozenset(
    w.key for w in TOP_LEVEL_WEIGHTS if w.is_penalty
)

# Channels that are never toggled off (low-magnitude base signals).
_PASSTHROUGH: frozenset[str] = frozenset({"survival", "extrinsic"})


def channels_to_vector(
    channels: dict[str, float],
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map a raw ``reward_channels`` dict → ordered cumulant vector φ ∈ ℝ^C.

    Penalty channels keep their stored (non-negative) magnitude here — the
    sign lives in the weight vector, so ``φ · w`` matches the scalar
    reward.  Unknown keys in *channels* are ignored; missing canonical
    channels default to 0.
    """
    v = torch.zeros(NUM_CHANNELS, dtype=dtype, device=device)
    for key, idx in CHANNEL_INDEX.items():
        val = channels.get(key)
        if val:  # skips None and 0.0
            v[idx] = float(val)
    return v


def weights_to_vector(
    weights: dict[str, float],
    toggles: dict[str, bool] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map a UI weight snapshot (+ optional toggle snapshot) → signed w ∈ ℝ^C.

    Penalty channels are negated so that ``φ · w == Σ ±wᵢ·φᵢ``, matching
    :meth:`RewardComposer._combine_rewards` / ``_recompose_rewards``.
    Disabled channels (per *toggles*) are zeroed, except the always-on
    passthrough channels (survival / extrinsic).
    """
    v = torch.zeros(NUM_CHANNELS, dtype=dtype, device=device)
    for key, idx in CHANNEL_INDEX.items():
        w = float(weights.get(key, 0.0))
        if (
            toggles is not None
            and key not in _PASSTHROUGH
            and not toggles.get(key, True)
        ):
            w = 0.0
        if key in PENALTY_CHANNELS:
            w = -w
        v[idx] = w
    return v


def default_weight_vector(
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Signed w ∈ ℝ^C built from the slider *default* weights.

    Fallback for when no live :class:`RewardWeightsState` is available
    (e.g. offline tooling or early startup before the UI is wired).
    """
    defaults = {w.key: w.default for w in TOP_LEVEL_WEIGHTS}
    return weights_to_vector(defaults, None, device=device, dtype=dtype)


def scalarize(psi: torch.Tensor, w_vec: torch.Tensor) -> torch.Tensor:
    """Project successor features onto the weight vector: ``V = ψ · w``.

    Args:
        psi:   ``(B, C)`` or ``(C,)`` successor features.
        w_vec: ``(C,)`` signed scalarisation weights.

    Returns:
        ``(B,)`` scalar values, or a 0-d scalar if *psi* is 1-D.
    """
    w_vec = w_vec.to(psi.dtype)
    if psi.dim() == 1:
        return (psi * w_vec).sum()
    return psi.matmul(w_vec)


def attribution(psi: torch.Tensor, w_vec: torch.Tensor) -> torch.Tensor:
    """Per-channel value contribution ``ψ[i] · w[i]`` → ``(B, C)`` (or ``(C,)``).

    This is the explainability read-out: each entry is the expected
    discounted future reward the agent attributes to that channel under
    the current weights.  Summing over the last dim recovers ``V(s)``.
    """
    return psi * w_vec.to(psi.dtype)
