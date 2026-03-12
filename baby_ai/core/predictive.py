"""
Latent World Model — JEPA / RSSM-style forward dynamics in latent space.

Predicts *transitions in the latent embedding space* conditioned on
continuous actions from the diffusion policy. The agent never decodes
back to pixels; curiosity reward is the latent prediction error
(JEPA-style), which naturally filters stochastic noise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Latent World Model (RSSM-inspired)
# ────────────────────────────────────────────────────────────────────────────

class LatentWorldModel(nn.Module):
    """
    Recurrent State-Space Model operating entirely in latent space.

    Architecture:
      1. **Deterministic path** (prior):
         (core_state, action) → predicted next latent via MLP.
      2. **Stochastic path** (posterior — training only):
         Encodes the *actual* next observation embedding for KL
         regularisation against the prior, stabilising learning.
      3. **Latent predictor**:
         Maps the deterministic prediction to a target latent space
         that is compared against a momentum-EMA target encoder
         (JEPA-style), avoiding representation collapse.

    The model exposes two losses:
      • ``dynamics_loss``:  MSE between predicted and actual latent.
      • ``kl_loss``:        KL(posterior || prior) for stochastic states.

    Args:
        state_dim:     Dimension of the temporal core output (core_state).
        action_dim:    Continuous action dimensionality (e.g. 23).
        latent_dim:    Dimension of the latent world-model representation.
        hidden_dim:    MLP hidden layer size.
        stochastic_dim: Dimension of stochastic state (0 to disable).
        ema_decay:     Target-encoder EMA decay for JEPA collapse prevention.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 23,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        stochastic_dim: int = 32,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.stochastic_dim = stochastic_dim
        self.ema_decay = ema_decay

        # Continuous action projection
        action_embed_dim = min(action_dim, 64)
        self.action_proj = nn.Linear(action_dim, action_embed_dim)

        # ── Deterministic dynamics (prior): core_state + action → next latent ──
        self.dynamics_net = nn.Sequential(
            nn.Linear(state_dim + action_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ── Stochastic prior: predict mean/logvar of stochastic state ──
        if stochastic_dim > 0:
            self.prior_net = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, stochastic_dim * 2),  # mean + logvar
            )
            # Posterior: uses the actual next fused embedding
            self.posterior_net = nn.Sequential(
                nn.Linear(latent_dim + state_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, stochastic_dim * 2),
            )

        # ── Latent predictor (JEPA projection head) ──
        # Maps dynamics output to the target space for comparison
        pred_input_dim = latent_dim + (stochastic_dim if stochastic_dim > 0 else 0)
        self.predictor = nn.Sequential(
            nn.Linear(pred_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ── Online encoder: fused embedding → latent ──
        self.online_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ── Target encoder (EMA copy — not trained via gradients) ──
        self.target_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Initialise target = online
        self._sync_target_encoder()
        # Freeze target encoder (updated only via EMA)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    # ── Action encoding helper ──────────────────────────────────────────

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        """Encode a continuous action vector into an embedding.

        Args:
            action: (B, action_dim) continuous action vector.
        """
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.action_proj(action.float())

    # ── EMA target encoder update ──────────────────────────────────────────

    @torch.no_grad()
    def _sync_target_encoder(self) -> None:
        """Hard-copy online encoder weights to target encoder."""
        for tp, op in zip(self.target_encoder.parameters(),
                          self.online_encoder.parameters()):
            tp.data.copy_(op.data)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        """Soft-update target encoder via EMA (call after each training step)."""
        for tp, op in zip(self.target_encoder.parameters(),
                          self.online_encoder.parameters()):
            tp.data.mul_(self.ema_decay).add_(op.data, alpha=1.0 - self.ema_decay)

    # ── forward dynamics ───────────────────────────────────────────────────

    def predict_next_latent(
        self,
        core_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the next latent state (deterministic path only).

        Used at inference time for curiosity computation and planning.

        Args:
            core_state: (B, state_dim) from temporal core.
            action:     (B, action_dim) continuous action vector.

        Returns:
            (B, latent_dim) predicted next latent.
        """
        a_emb = self._encode_action(action)          # (B, action_embed_dim)
        x = torch.cat([core_state, a_emb], dim=-1)
        det_pred = self.dynamics_net(x)              # (B, latent_dim)

        if self.stochastic_dim > 0:
            prior_params = self.prior_net(det_pred)
            mean, _ = prior_params.chunk(2, dim=-1)
            pred_input = torch.cat([det_pred, mean], dim=-1)
        else:
            pred_input = det_pred

        return self.predictor(pred_input)            # (B, latent_dim)

    def encode_state(self, fused: torch.Tensor) -> torch.Tensor:
        """Encode a fused embedding into the online latent space."""
        return self.online_encoder(fused)

    @torch.no_grad()
    def encode_target(self, fused: torch.Tensor) -> torch.Tensor:
        """Encode a fused embedding via the target (EMA) encoder."""
        return self.target_encoder(fused)

    # ── training forward ───────────────────────────────────────────────────

    def forward(
        self,
        core_state: torch.Tensor,
        next_fused: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full training forward pass.

        Args:
            core_state: (B, state_dim) temporal core output at time t.
            next_fused: (B, state_dim) fused embedding at time t+1.
            action:     (B, action_dim) continuous action vector.

        Returns:
            dict with:
                predicted_latent:  (B, latent_dim) predicted next latent.
                target_latent:     (B, latent_dim) target (EMA-encoded) next latent.
                dynamics_loss:     Scalar MSE in latent space.
                kl_loss:           Scalar KL divergence (0 if no stochastic dim).
                curiosity_reward:  (B,) per-sample latent prediction error.
        """
        # ── Target: EMA-encoded actual next state ──
        target_latent = self.encode_target(next_fused)  # (B, latent_dim)

        # ── Deterministic dynamics prediction ──
        a_emb = self._encode_action(action)
        x = torch.cat([core_state, a_emb], dim=-1)
        det_pred = self.dynamics_net(x)                 # (B, latent_dim)

        kl_loss = torch.tensor(0.0, device=core_state.device)

        if self.stochastic_dim > 0:
            # Prior
            prior_params = self.prior_net(det_pred)
            prior_mean, prior_logvar = prior_params.chunk(2, dim=-1)

            # Posterior (uses knowledge of actual next state)
            online_next = self.online_encoder(next_fused)
            post_input = torch.cat([det_pred, online_next], dim=-1)
            post_params = self.posterior_net(post_input)
            post_mean, post_logvar = post_params.chunk(2, dim=-1)

            # Sample from posterior (reparameterisation trick)
            std = torch.exp(0.5 * post_logvar)
            eps = torch.randn_like(std)
            z = post_mean + eps * std

            pred_input = torch.cat([det_pred, z], dim=-1)

            # KL(posterior || prior) — free nats clamp to avoid overregularisation
            kl_loss = _kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)
            kl_loss = torch.clamp(kl_loss, min=1.0).mean()
        else:
            pred_input = det_pred

        # ── Predicted latent in target space ──
        predicted_latent = self.predictor(pred_input)   # (B, latent_dim)

        # ── Dynamics loss: distance in latent space ──
        # Use smooth-L1 (Huber) to be robust to outliers
        dynamics_loss = F.smooth_l1_loss(
            predicted_latent, target_latent.detach(), reduction="mean",
        )

        # ── Curiosity reward: per-sample prediction error ──
        with torch.no_grad():
            curiosity_reward = (predicted_latent - target_latent).pow(2).mean(dim=-1)

        return {
            "predicted_latent": predicted_latent,
            "target_latent": target_latent,
            "dynamics_loss": dynamics_loss,
            "kl_loss": kl_loss,
            "curiosity_reward": curiosity_reward.detach(),
        }


# ── KL divergence utility ─────────────────────────────────────────────────

def _kl_divergence(
    mu_q: torch.Tensor,
    logvar_q: torch.Tensor,
    mu_p: torch.Tensor,
    logvar_p: torch.Tensor,
) -> torch.Tensor:
    """Analytic KL(N(mu_q, sigma_q) || N(mu_p, sigma_p)), per-sample."""
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        logvar_p - logvar_q
        + var_q / var_p
        + (mu_q - mu_p).pow(2) / var_p
        - 1.0
    )
    return kl.sum(dim=-1)  # (B,)
