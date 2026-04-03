"""
Latent World Model — Dreamer-V3-inspired RSSM in latent space.

Predicts *transitions in the latent embedding space* conditioned on
continuous actions from the policy head.  The agent never decodes
back to pixels; curiosity reward is the latent prediction error
(JEPA-style), which naturally filters stochastic noise.

Key design choices:

- **Deterministic GRU path**: GRU encodes the sequence history
  h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1})).  This replaces the
  pure feedforward dynamics MLP and gives the world model its own
  recurrent memory, separate from JambaCore.
- **Categorical stochastic state (Dreamer-V3 trick)**: Instead of a
  Gaussian latent, use 32 categorical distributions each with 32
  classes.  The latent z ∈ ℝ^{32×32} is the concatenation of all
  one-hot samples (straight-through gradient).  This prevents posterior
  collapse and gives more stable gradients than Gaussian KL.
- **EMA target encoder**: The comparison target for the predictor is
  produced by an exponential-moving-average copy of the online encoder,
  following JEPA / BYOL to prevent representation collapse.
- **Dyna imagined rollouts**: expose `imagine_rollout()` to generate
  synthetic multi-step transitions for augmenting the replay buffer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Categorical straight-through utility ─────────────────────────────────

def _straight_through_softmax(logits: torch.Tensor) -> torch.Tensor:
    """
    Straight-through gradient for categorical argmax.

    Forward:  argmax one-hot (discrete)
    Backward: gradient flows through softmax (continuous)

    Args:
        logits: (B, num_cats, num_classes)
    Returns:
        (B, num_cats, num_classes) one-hot with ST gradient
    """
    soft = F.softmax(logits, dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), soft.size(-1)).float()
    return (hard - soft).detach() + soft  # straight-through estimator


# ────────────────────────────────────────────────────────────────────────────
# Latent World Model (Dreamer-V3 RSSM)
# ────────────────────────────────────────────────────────────────────────────

class LatentWorldModel(nn.Module):
    """
    Dreamer-V3-inspired Recurrent State-Space Model in latent space.

    Architecture:
      1. **GRU deterministic path**:
         h_t = GRU(h_{t-1}, concat(z_{t-1}_flat, a_emb))
         Gives the world model its own recurrent state, separate from
         JambaCore.  At t=0 or when no prior h is supplied, h is
         initialised from the current core_state via a linear projection.
      2. **Categorical stochastic state** (Dreamer-V3 trick):
         Prior / posterior predict logits over num_cats × num_classes
         categories.  Samples use straight-through argmax.  Avoids
         posterior collapse and vanishing Gaussian KL gradients.
      3. **JEPA predictor**:
         Maps (h_t, z_t) → target latent space, compared against the
         EMA-encoded actual next state.

    The model exposes:
      • ``dynamics_loss``: MSE between predicted and actual latent.
      • ``kl_loss``:       Categorical KL(posterior || prior).
      • ``imagine_rollout()``: Dyna-style imagined multi-step sequence.

    Args:
        state_dim:     Core hidden state dimension.
        action_dim:    Continuous action dimensionality.
        latent_dim:    Dimension of the latent world-model representation.
        hidden_dim:    GRU and MLP hidden dim.
        num_cats:      Number of categorical variables (Dreamer-V3: 32).
        num_classes:   Classes per categorical variable (Dreamer-V3: 32).
        ema_decay:     Target-encoder EMA decay.
    """

    def __init__(
        self,
        state_dim: int = 256,
        action_dim: int = 23,
        latent_dim: int = 256,
        hidden_dim: int = 256,
        stochastic_dim: int = 32,   # kept for API compatibility; maps to num_cats
        ema_decay: float = 0.99,
        num_cats: int = 32,
        num_classes: int = 32,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.ema_decay = ema_decay
        self.num_cats = num_cats
        self.num_classes = num_classes
        # stochastic_dim for API compatibility (used by planner + curiosity)
        self.stochastic_dim = num_cats * num_classes

        # ── Action projection ──
        action_embed_dim = min(action_dim, 64)
        self.action_proj = nn.Linear(action_dim, action_embed_dim)

        # ── State initializer: project core_state → GRU h_0 ──
        self.h_init = nn.Linear(state_dim, hidden_dim)

        # ── GRU deterministic core ──
        # Input: concat(z_flat, a_emb) where z_flat = num_cats * num_classes
        stoch_flat = num_cats * num_classes
        self.gru = nn.GRUCell(input_size=stoch_flat + action_embed_dim,
                              hidden_size=hidden_dim)

        # ── Prior net: h → categorical logits ──
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_cats * num_classes),
        )

        # ── Posterior net: (h, obs_latent) → categorical logits ──
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, num_cats * num_classes),
        )

        # ── Dynamics projection: (h, z_flat) → latent_dim ──
        self.dynamics_net = nn.Sequential(
            nn.Linear(hidden_dim + stoch_flat, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # ── JEPA predictor (projection head) ──
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
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
        self._sync_target_encoder()
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # Running GRU hidden state (single-step inference)
        self._h: torch.Tensor | None = None
        self._z_flat: torch.Tensor | None = None

    # ── Action encoding helper ──────────────────────────────────────────

    def _encode_action(self, action: torch.Tensor) -> torch.Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.action_proj(action.float())

    # ── EMA target encoder ──────────────────────────────────────────────

    @torch.no_grad()
    def _sync_target_encoder(self) -> None:
        for tp, op in zip(self.target_encoder.parameters(),
                          self.online_encoder.parameters()):
            tp.data.copy_(op.data)

    @torch.no_grad()
    def update_target_encoder(self) -> None:
        for tp, op in zip(self.target_encoder.parameters(),
                          self.online_encoder.parameters()):
            tp.data.mul_(self.ema_decay).add_(op.data, alpha=1.0 - self.ema_decay)

    # ── Categorical KL ──────────────────────────────────────────────────

    @staticmethod
    def _categorical_kl(
        post_logits: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL(posterior || prior) for factored categoricals.

        Args:
            post_logits:  (B, num_cats * num_classes)
            prior_logits: (B, num_cats * num_classes)
        Returns:
            (B,) per-sample KL (summed over all categoricals).
        """
        B = post_logits.size(0)
        num_cats = post_logits.size(-1) // prior_logits.size(-1) if post_logits.size(-1) != prior_logits.size(-1) else 1
        # Reshape into (B, num_cats, num_classes)
        n_dim = post_logits.size(-1)
        # Assume square: num_cats = num_classes = sqrt(n_dim)
        side = int(n_dim ** 0.5)
        if side * side != n_dim:
            # fallback: treat as single categorical
            q = F.log_softmax(post_logits, dim=-1)
            p = F.softmax(prior_logits, dim=-1)
            return F.kl_div(q, p, reduction='none').sum(dim=-1)

        q = post_logits.view(B, side, side)
        p = prior_logits.view(B, side, side)
        log_q = F.log_softmax(q, dim=-1)
        log_p = F.log_softmax(p, dim=-1)
        prob_q = log_q.exp()
        kl = (prob_q * (log_q - log_p)).sum(dim=-1).sum(dim=-1)  # (B,)
        return kl

    # ── Encoder helpers ─────────────────────────────────────────────────

    def encode_state(self, fused: torch.Tensor) -> torch.Tensor:
        return self.online_encoder(fused)

    @torch.no_grad()
    def encode_target(self, fused: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(fused)

    # ── Inference: predict next latent (single step) ────────────────────

    def predict_next_latent(
        self,
        core_state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict the next latent state (deterministic path with prior mean).

        Used at inference time for curiosity computation and planning.

        Args:
            core_state: (B, state_dim) from temporal/JambaCore output.
            action:     (B, action_dim) continuous action.

        Returns:
            (B, latent_dim) predicted next latent.
        """
        B = core_state.size(0)
        device = core_state.device

        # Initialise or validate GRU state
        if self._h is None or self._h.size(0) != B or self._h.device != device:
            self._h = torch.zeros(B, self.hidden_dim, device=device)
            self._z_flat = torch.zeros(B, self.num_cats * self.num_classes, device=device)

        a_emb = self._encode_action(action)
        gru_input = torch.cat([self._z_flat, a_emb], dim=-1)
        h = self.gru(gru_input, self._h)

        # Prior categorical logits → straight-through sample
        prior_logits = self.prior_net(h).view(B, self.num_cats, self.num_classes)
        z = _straight_through_softmax(prior_logits)
        z_flat = z.view(B, -1)

        # Update state buffers
        self._h = h.detach()
        self._z_flat = z_flat.detach()

        # Compose deterministic + stochastic representation
        rep = self.dynamics_net(torch.cat([h, z_flat], dim=-1))
        return self.predictor(rep)

    def reset_state(self) -> None:
        """Reset the single-step inference state (call at episode start)."""
        self._h = None
        self._z_flat = None

    # ── Training forward pass ────────────────────────────────────────────

    def forward(
        self,
        core_state: torch.Tensor,
        next_fused: torch.Tensor,
        action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Full training forward pass.

        Runs the GRU with action + previous stochastic state, then trains
        both the prior (from h alone) and posterior (from h + observed
        next embedding) to produce calibrated categorical latents.

        Args:
            core_state: (B, state_dim) temporal core output at time t.
            next_fused: (B, state_dim) fused embedding at time t+1.
            action:     (B, action_dim) continuous action.

        Returns:
            dict with predicted_latent, target_latent, dynamics_loss,
            kl_loss, curiosity_reward.
        """
        B = core_state.size(0)
        device = core_state.device

        # ── Target: EMA-encoded actual next state ──
        target_latent = self.encode_target(next_fused)  # (B, latent_dim)

        # ── Initialise GRU hidden state from current core_state ──
        h = torch.tanh(self.h_init(core_state))  # (B, hidden_dim)

        # Prior z (from h alone — no look-ahead)
        prior_logits = self.prior_net(h).view(B, self.num_cats, self.num_classes)
        z_prior_st = _straight_through_softmax(prior_logits)
        z_prior_flat = z_prior_st.view(B, -1)

        # Posterior z (from h + observed online-encoded next state)
        obs_latent = self.online_encoder(next_fused)   # (B, latent_dim)
        post_input = torch.cat([h, obs_latent], dim=-1)
        post_logits = self.posterior_net(post_input).view(B, self.num_cats, self.num_classes)
        z_post_st = _straight_through_softmax(post_logits)
        z_post_flat = z_post_st.view(B, -1)

        # GRU step: advance h using action + posterior z
        a_emb = self._encode_action(action)
        gru_input = torch.cat([z_post_flat.detach(), a_emb], dim=-1)
        h_next = self.gru(gru_input, h)

        # ── Predicted latent from posterior (training uses posterior for q) ──
        rep = self.dynamics_net(torch.cat([h_next, z_post_flat], dim=-1))
        predicted_latent = self.predictor(rep)  # (B, latent_dim)

        # ── Dynamics loss ──
        dynamics_loss = F.smooth_l1_loss(predicted_latent, target_latent.detach())

        # ── Categorical KL(posterior || prior) ──
        # Use straight-through logits (not sampled z) for KL
        prior_flat = prior_logits.view(B, -1)
        post_flat = post_logits.view(B, -1)
        kl_per_sample = self._categorical_kl(post_flat, prior_flat)
        # Free-bits: clamp at 1.0 nat to avoid overregularisation
        kl_loss = kl_per_sample.clamp(min=1.0).mean()

        # ── Curiosity reward ──
        with torch.no_grad():
            curiosity_reward = (predicted_latent - target_latent).pow(2).mean(dim=-1)

        # ── Counterfactual causal augmentation (CDL-inspired) ────────────
        # Randomly zero ~25% of core_state dims and re-run the forward.
        # The model must still predict the target — this forces it to learn
        # representations that are robust to masking non-causal dims,
        # implicitly penalizing spurious correlational shortcuts.
        counterfactual_loss = None
        if self.training:
            with torch.no_grad():
                # Bernoulli mask: keep 75% of dims
                mask = torch.bernoulli(
                    torch.full_like(core_state, 0.75)
                )
            masked_state = core_state * mask
            h_cf = torch.tanh(self.h_init(masked_state))
            prior_cf = self.prior_net(h_cf).view(B, self.num_cats, self.num_classes)
            z_cf_st = _straight_through_softmax(prior_cf)
            z_cf_flat = z_cf_st.view(B, -1)
            rep_cf = self.dynamics_net(torch.cat([h_cf, z_cf_flat], dim=-1))
            pred_cf = self.predictor(rep_cf)
            counterfactual_loss = F.smooth_l1_loss(pred_cf, target_latent.detach())

        return {
            "predicted_latent": predicted_latent,
            "target_latent": target_latent,
            "dynamics_loss": dynamics_loss,
            "kl_loss": kl_loss,
            "curiosity_reward": curiosity_reward.detach(),
            "counterfactual_loss": counterfactual_loss,
        }

    # ── Dyna imagined rollouts ───────────────────────────────────────────

    @torch.no_grad()
    def imagine_rollout(
        self,
        start_latent: torch.Tensor,
        policy: nn.Module,
        horizon: int = 5,
        gamma: float = 0.99,
    ) -> list[dict[str, torch.Tensor]]:
        """
        Dyna-style: generate synthetic transitions in imagination.

        Uses the world model as a simulator: starting from a real latent
        state, rolls out `horizon` steps of imagined experience under
        the current policy.  The resulting synthetic transitions can be
        added to the replay buffer (at a down-weighted priority).

        Args:
            start_latent: (B, latent_dim) starting latent (from encode_state).
            policy:       Policy module with .act(state) → (action, log_prob, value).
            horizon:      Number of imagined steps.
            gamma:        Discount factor for imagined rewards.

        Returns:
            List of `horizon` synthetic transition dicts, each with:
                state, action, reward (from value head), next_state.
        """
        B = start_latent.size(0)
        device = start_latent.device
        state = start_latent

        # Initialise GRU hidden from start
        # (Use zeros since we're starting from a latent, not core_state)
        h = torch.zeros(B, self.hidden_dim, device=device)
        z_flat = torch.zeros(B, self.num_cats * self.num_classes, device=device)

        transitions = []
        for t in range(horizon):
            # Get action from policy
            action, _, value = policy.act(state, deterministic=False)

            # Advance GRU
            a_emb = self._encode_action(action)
            gru_input = torch.cat([z_flat, a_emb], dim=-1)
            h = self.gru(gru_input, h)

            # Sample next z from prior
            prior_logits = self.prior_net(h).view(B, self.num_cats, self.num_classes)
            z_st = _straight_through_softmax(prior_logits)
            z_flat_next = z_st.view(B, -1)

            # Next latent
            rep = self.dynamics_net(torch.cat([h, z_flat_next], dim=-1))
            next_state = self.predictor(rep)

            # Reward proxy: discounted value (imagined reward)
            reward = (value.squeeze(-1) * (gamma ** t)).detach()

            transitions.append({
                "state": state.detach(),
                "action": action.detach(),
                "reward": reward,
                "next_state": next_state.detach(),
                "_imagined": True,  # flag so replay buffer can down-weight
            })

            state = next_state
            z_flat = z_flat_next

        return transitions
