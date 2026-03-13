"""
Consolidation module — EWC + Rehearsal for continual learning.

Implements Elastic Weight Consolidation to protect important weights
from catastrophic forgetting, combined with replay-based rehearsal.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from baby_ai.utils.logging import get_logger

log = get_logger("consolidation")


class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Computes Fisher Information diagonal to penalize changes
    to parameters that are important for previously learned tasks.

    Args:
        model: The model to protect.
        ewc_lambda: Strength of the consolidation penalty.
    """

    def __init__(self, model: nn.Module, ewc_lambda: float = 100.0):
        self.ewc_lambda = ewc_lambda
        self._fisher: Dict[str, torch.Tensor] = {}
        self._params_star: Dict[str, torch.Tensor] = {}
        self._initialized = False

    # Keys that the Teacher model's forward() actually accepts.
    _VALID_FORWARD_KEYS = frozenset({
        "vision", "audio", "sensor",
        "code_x", "code_edge_index", "code_batch",
        "actions", "goal",
    })

    def compute_fisher(
        self,
        model: nn.Module,
        data_loader,
        num_samples: int = 200,
        device: str = "cuda",
    ) -> None:
        """
        Estimate diagonal Fisher Information from a data sample.

        Uses ``torch.autograd.grad`` instead of ``.backward()`` so that
        no gradients are accumulated on parameter ``.grad`` attributes.
        This prevents corruption of the AMP GradScaler / optimizer state
        in the calling training loop.

        Args:
            model: Current model.
            data_loader: Iterable yielding dicts of batched tensors.
            num_samples: Number of samples for estimation.
            device: Device for computation.
        """
        was_training = model.training
        model.eval()

        params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        param_tensors = [p for _, p in params]
        fisher = {
            n: torch.zeros_like(p, device=device)
            for n, p in params
        }

        count = 0
        try:
            for batch in data_loader:
                if count >= num_samples:
                    break

                # Filter to only keys the Teacher forward() accepts.
                # Replay batches contain extra keys (reward, fused, is_demo, ...)
                # that would cause a TypeError if passed through.
                forward_kwargs = {}
                for k, v in batch.items():
                    # Map 'action' -> 'actions' (Teacher kwarg name)
                    fwd_key = "actions" if k == "action" else k
                    if fwd_key not in self._VALID_FORWARD_KEYS:
                        continue
                    forward_kwargs[fwd_key] = v.to(device) if isinstance(v, torch.Tensor) else v

                outputs = model(**forward_kwargs)

                # Squared-norm of continuous action output -> scalar for Fisher
                action = outputs["action"]
                loss = action.pow(2).sum(dim=-1).mean()
                batch_size = action.size(0)

                # Use autograd.grad to avoid touching .grad attributes.
                grads = torch.autograd.grad(
                    loss, param_tensors,
                    retain_graph=False, create_graph=False,
                    allow_unused=True,
                )
                for (n, _p), g in zip(params, grads):
                    if g is not None:
                        fisher[n] += g.detach().pow(2)

                count += batch_size

            # Average
            for n in fisher:
                fisher[n] /= max(count, 1)
        finally:
            # ALWAYS restore training mode, even if an exception occurred.
            # Leaving the model in eval mode causes _aux_loss (MoE) and
            # other training-only state to go stale, which triggers
            # "backward through the graph a second time" on the next step.
            if was_training:
                model.train()

        # Store snapshot
        self._fisher = fisher
        self._params_star = {
            n: p.data.clone()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self._initialized = True
        log.info("Fisher Information computed over %d samples.", count)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Compute EWC penalty: lambda/2 * sum(F_i * (theta_i - theta*_i)^2).

        Returns:
            Scalar penalty tensor (add to loss).
        """
        if not self._initialized:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self._fisher:
                fisher = self._fisher[n].to(p.device)
                star = self._params_star[n].to(p.device)
                loss += (fisher * (p - star).pow(2)).sum()

        return self.ewc_lambda / 2.0 * loss


class Consolidator:
    """
    Combined consolidation: EWC + rehearsal from replay buffer.

    Periodically updates the Fisher snapshot and computes
    consolidation loss during Teacher training.

    Args:
        ewc_lambda: EWC penalty strength.
        rehearsal_batch_size: Number of replay samples for rehearsal.
    """

    def __init__(
        self,
        ewc_lambda: float = 100.0,
        rehearsal_batch_size: int = 16,
    ):
        self.ewc = EWC(nn.Module(), ewc_lambda=ewc_lambda)
        self.rehearsal_batch_size = rehearsal_batch_size
        self._step = 0

    def update_fisher(
        self,
        model: nn.Module,
        data_loader,
        num_samples: int = 200,
        device: str = "cuda",
    ) -> None:
        """Recompute Fisher Information from recent data."""
        self.ewc.compute_fisher(model, data_loader, num_samples, device)

    def consolidation_loss(self, model: nn.Module) -> torch.Tensor:
        """Get EWC penalty for the current model."""
        return self.ewc.penalty(model)

    def step(self) -> None:
        self._step += 1
