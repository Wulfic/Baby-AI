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

    def compute_fisher(
        self,
        model: nn.Module,
        data_loader,
        num_samples: int = 200,
        device: str = "cuda",
    ) -> None:
        """
        Estimate diagonal Fisher Information from a data sample.

        Args:
            model: Current model.
            data_loader: Iterable yielding (inputs_dict, targets_dict).
            num_samples: Number of samples for estimation.
            device: Device for computation.
        """
        model.eval()

        fisher = {
            n: torch.zeros_like(p, device=device)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        count = 0
        for batch in data_loader:
            if count >= num_samples:
                break

            # Forward pass — use policy logits as the "output"
            model.zero_grad()
            outputs = model(**{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
            logits = outputs["action_logits"]

            # Compute log-likelihood of most likely action
            log_probs = F.log_softmax(logits, dim=-1)
            pseudo_label = logits.argmax(dim=-1)
            loss = F.nll_loss(log_probs, pseudo_label)
            loss.backward()

            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

            count += logits.size(0)

        # Average
        for n in fisher:
            fisher[n] /= max(count, 1)

        # Restore training mode — callers (e.g. the learner thread)
        # expect the model to remain in .train() mode.
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
