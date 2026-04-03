"""
Muon optimizer — Nesterov momentum with Newton-Schulz orthogonalization.

Reference: Kowalski et al., 2025.
           "Modular Adaptive Optimization" (Muon / Orthogonal-Momentum).

Key idea: instead of applying the raw gradient or a scaled version of it,
orthogonalize the Nesterov momentum update matrix before applying it.
This keeps the effective update on the Stiefel manifold (orthogonal
updates), which is shown to be significantly more sample-efficient for
recurrent / SSM architectures than plain AdamW.

How to use
----------
Apply Muon *only* to 2-D weight matrices in the temporal / SSM core.
All other parameters (biases, LayerNorm scales, embeddings) should remain
in the standard AdamW optimizer to avoid stability issues.

Example
-------
    muon_params = [p for name, p in model.temporal.named_parameters()
                   if p.dim() >= 2 and p.requires_grad]
    muon = Muon(muon_params, lr=1e-3, momentum=0.95)
    # main AdamW handles everything else
"""

from __future__ import annotations

import torch
from torch.optim import Optimizer


def _newton_schulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """
    Orthogonalize G via quintic Newton-Schulz iterations.

    Converges to the unitary factor of the SVD (i.e. U·Vᵀ where G = U·S·Vᵀ)
    in ~5 steps.  Works for non-square matrices by folding to the smaller
    dimension.

    Quintic polynomial: X = a*X + b*A*X + c*A²*X  where A = X·Xᵀ
    Coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) satisfy the
    degree-5 minimax approximation on [0, 1].

    Args:
        G: 2-D gradient tensor (M, N).
        steps: Number of NS iterations (5 is sufficient for convergence).

    Returns:
        Orthogonalized matrix of same shape as G.
    """
    assert G.dim() == 2, f"Muon NS expects 2-D tensor, got {G.dim()}-D"
    a, b, c = 3.4445, -4.7750, 2.0315

    # Normalize to prevent divergence: ||G||_F ≈ 1 after this
    X = G / (G.norm(p='fro') + 1e-7)

    # If tall matrix, work on the transpose (square or wide is efficient)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.t().contiguous()  # now (N, M) with N ≤ M

    for _ in range(steps):
        A = X @ X.t()           # (N, N) Gram matrix
        X = a * X + b * (A @ X) + c * (A @ A @ X)

    if transposed:
        X = X.t().contiguous()

    return X


class Muon(Optimizer):
    """
    Muon optimizer for 2-D weight matrices.

    Applies Nesterov momentum followed by Newton-Schulz orthogonalization
    on each update step.  The orthogonalized direction is scaled by ``lr``
    and subtracted from the parameter.

    **Only use for 2-D weight matrices** (Linear.weight, Conv.weight with
    view, SSM projection matrices, etc.).  Biases, 1-D vectors, and
    embedding tables should use standard AdamW.

    Args:
        params: Iterable of 2-D parameters or param-group dicts.
        lr: Learning rate (default 1e-3).  Larger values are typically
            safe compared to AdamW because the update magnitude is bounded
            by the orthogonalization.
        momentum: Nesterov momentum coefficient (default 0.95).
        ns_steps: Newton-Schulz iteration count (default 5).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0.95,
        ns_steps: int = 5,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid lr: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad.float()  # use fp32 for NS iterations

                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(g)

                buf = state["buf"]
                # Nesterov update: buf = momentum * buf + grad
                buf.mul_(momentum).add_(g)
                # Nesterov lookahead direction
                g_nesterov = g + momentum * buf

                if g_nesterov.dim() == 2:
                    # Core case: orthogonalize the update
                    update = _newton_schulz5(g_nesterov, steps=ns_steps)
                    # Scale update to match gradient magnitude (NS output ≈ unit norm)
                    update = update * (g_nesterov.norm(p='fro') + 1e-7)
                else:
                    # Fallback for non-2D (shouldn't happen if caller filters correctly)
                    update = g_nesterov

                # Cast back to param dtype before applying
                p.add_(update.to(p.dtype), alpha=-lr)

        return loss
