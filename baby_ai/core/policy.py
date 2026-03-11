"""
Policy head — action selection from core hidden state.

**DiffusionPolicyHead** — Conditional DDIM diffusion for continuous
compound actions (pitch, yaw, movement, attack, hotbar, …).
Generates a unified continuous action vector in 3-5 denoising steps
conditioned on the JambaCore hidden state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────────────────────
# Noise schedule utilities
# ────────────────────────────────────────────────────────────────────────────

def _linear_beta_schedule(
    num_steps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Linear variance schedule — returns (num_steps,) beta values."""
    return torch.linspace(beta_start, beta_end, num_steps)


def _cosine_alpha_bar(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021) — returns alpha_bar_t."""
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos((steps / num_steps + s) / (1 + s) * (math.pi / 2)) ** 2
    alpha_bar = f / f[0]
    return alpha_bar[:num_steps].float()


# ────────────────────────────────────────────────────────────────────────────
# Sinusoidal timestep embedding
# ────────────────────────────────────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """Learned-free positional embedding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) integer or float timesteps.

        Returns:
            (B, dim) sinusoidal embeddings.
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ────────────────────────────────────────────────────────────────────────────
# Noise prediction network (MLP with residual blocks)
# ────────────────────────────────────────────────────────────────────────────

class _ResidualBlock(nn.Module):
    """Simple MLP residual block with SiLU."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))


class NoisePredictor(nn.Module):
    """
    MLP noise predictor conditioned on state and timestep.

    Input:  concat(noisy_action, time_embed, state)
    Output: predicted noise (same dim as action).
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        time_embed_dim: int = 64,
        hidden_dim: int = 256,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimestepEmbedding(time_embed_dim)

        input_dim = action_dim + time_embed_dim + state_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.ModuleList(
            [_ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            noisy_action: (B, action_dim).
            timestep:     (B,) diffusion timestep indices.
            state:        (B, state_dim) conditioning signal.

        Returns:
            (B, action_dim) predicted noise.
        """
        t_emb = self.time_embed(timestep)
        x = torch.cat([noisy_action, t_emb, state], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


# ────────────────────────────────────────────────────────────────────────────
# Diffusion Policy Head
# ────────────────────────────────────────────────────────────────────────────

class DiffusionPolicyHead(nn.Module):
    """
    Conditional denoising diffusion policy for continuous actions.

    Generates a bounded continuous action vector via DDIM deterministic
    sampling conditioned on the temporal core's hidden state.

    Action vector layout (default 20-dim):
        [0:2]   camera (yaw_delta, pitch_delta) in [-1, 1]
        [2:6]   movement (forward, back, left, right) in [0, 1]
        [6:11]  actions (attack, use, jump, sneak, sprint) in [0, 1]
        [11:20] hotbar (9 slots, softmax-normalised)

    Uses **reward-weighted denoising** for RL training:
        loss = advantage * ||noise_pred - noise_true||²

    Args:
        input_dim:        Core hidden state dimension.
        action_dim:       Continuous action vector size.
        hidden_dim:       MLP hidden dimension.
        num_train_steps:  Number of diffusion timesteps during training.
        num_infer_steps:  DDIM sampling steps for fast inference.
        time_embed_dim:   Sinusoidal timestep embedding dimension.
        beta_start:       Linear noise schedule start.
        beta_end:         Linear noise schedule end.
    """

    def __init__(
        self,
        input_dim: int = 256,
        action_dim: int = 20,
        hidden_dim: int = 256,
        num_train_steps: int = 100,
        num_infer_steps: int = 4,
        time_embed_dim: int = 64,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_train_steps = num_train_steps
        self.num_infer_steps = num_infer_steps

        # ── Noise schedule (precomputed, not learnable) ──
        betas = _linear_beta_schedule(num_train_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer(
            "sqrt_alpha_bar", torch.sqrt(alpha_bar),
        )
        self.register_buffer(
            "sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar),
        )

        # ── Noise predictor ──
        self.noise_net = NoisePredictor(
            action_dim=action_dim,
            state_dim=input_dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=3,
        )

        # ── Value head (state → scalar, independent of diffusion) ──
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Pre-compute DDIM step indices (evenly spaced)
        self._ddim_steps: torch.Tensor | None = None

    # ── helpers ────────────────────────────────────────────────────────────

    def _get_ddim_steps(self, device: torch.device) -> torch.Tensor:
        """Evenly-spaced DDIM timestep subsequence."""
        if self._ddim_steps is not None and self._ddim_steps.device == device:
            return self._ddim_steps
        step_size = self.num_train_steps // self.num_infer_steps
        steps = torch.arange(0, self.num_train_steps, step_size, device=device)
        self._ddim_steps = steps.flip(0)  # descending order
        return self._ddim_steps

    def _q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Forward diffusion: add noise to x_0 at timestep t."""
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(-1)      # (B, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(-1)
        return sqrt_ab * x_0 + sqrt_omab * noise

    # ── training forward ───────────────────────────────────────────────────

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass.

        Args:
            state:   (B, input_dim) from temporal core.
            actions: (B, action_dim) ground-truth continuous actions.
                     If None, returns zero loss + value.

        Returns:
            denoising_loss: Scalar MSE denoising loss.
            value:          (B, 1) state value.
        """
        value = self.value_head(state)

        if actions is None:
            return torch.tensor(0.0, device=state.device), value

        # Guard: squeeze stale replay data stored with batch dim (B,1,D) → (B,D)
        if actions.dim() == 3 and actions.size(1) == 1:
            actions = actions.squeeze(1)

        B = state.size(0)
        device = state.device

        # Random timestep per sample
        t = torch.randint(0, self.num_train_steps, (B,), device=device)

        # Add noise
        noise = torch.randn_like(actions)
        noisy_actions = self._q_sample(actions, t, noise)

        # Predict noise
        noise_pred = self.noise_net(noisy_actions, t, state)
        denoising_loss = F.mse_loss(noise_pred, noise)

        return denoising_loss, value

    # ── inference (DDIM sampling) ──────────────────────────────────────────

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate continuous actions via DDIM denoising.

        Args:
            state: (B, input_dim).
            deterministic: If True, use eta=0 (fully deterministic DDIM).

        Returns:
            action:   (B, action_dim) bounded continuous action vector.
            log_prob: (B,) approximate log-probability (negative denoising loss).
            value:    (B, 1) state value.
        """
        B = state.size(0)
        device = state.device

        # Start from pure noise
        x = torch.randn(B, self.action_dim, device=device)

        # DDIM denoising loop
        ddim_steps = self._get_ddim_steps(device)

        for i, t_cur in enumerate(ddim_steps):
            t_batch = t_cur.expand(B)

            noise_pred = self.noise_net(x, t_batch, state)

            # DDIM update (deterministic variant, eta=0)
            alpha_bar_t = self.alpha_bar[t_cur]
            sqrt_ab_t = self.sqrt_alpha_bar[t_cur]
            sqrt_omab_t = self.sqrt_one_minus_alpha_bar[t_cur]

            # Predicted x_0 from current noisy sample
            x0_pred = (x - sqrt_omab_t * noise_pred) / sqrt_ab_t.clamp(min=1e-8)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            if i < len(ddim_steps) - 1:
                # Next timestep
                t_next = ddim_steps[i + 1]
                alpha_bar_next = self.alpha_bar[t_next]
                sqrt_ab_next = torch.sqrt(alpha_bar_next)
                sqrt_omab_next = torch.sqrt(1.0 - alpha_bar_next)

                # DDIM formula: x_{t-1} = sqrt(α̅_{t-1}) * x̂_0 + sqrt(1-α̅_{t-1}) * ε_θ
                x = sqrt_ab_next * x0_pred + sqrt_omab_next * noise_pred
            else:
                x = x0_pred

        # Bound the output: camera dims to [-1,1], others to [0,1]
        action = self._bound_action(x)

        # Approximate log_prob: negative denoising score
        # (lower reconstruction error ≈ higher probability)
        log_prob = -((action - x0_pred).pow(2).sum(dim=-1))

        value = self.value_head(state)
        return action, log_prob, value

    def _bound_action(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply action-space bounds: camera in [-1,1], discrete-like in [0,1]."""
        action = raw.clone()
        # Camera (first 2 dims): tanh to [-1, 1]
        action[:, :2] = torch.tanh(action[:, :2])
        # Everything else: sigmoid to [0, 1]
        if action.size(-1) > 2:
            action[:, 2:] = torch.sigmoid(action[:, 2:])
        return action

    # ── PPO-compatible evaluation ──────────────────────────────────────────

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a given action for PPO-style training.

        For diffusion, log_prob is approximated via denoising loss and
        entropy is estimated from the noise schedule.

        Returns:
            log_prob: (B,) approximate log probability.
            entropy:  (B,) estimated entropy.
            value:    (B, 1) state value.
        """
        B = state.size(0)
        device = state.device

        # Estimate log_prob: how well the noise net can reconstruct
        # the action at a medium noise level
        t_eval = torch.full((B,), self.num_train_steps // 2, device=device)
        noise = torch.randn_like(action)
        noisy = self._q_sample(action, t_eval, noise)
        noise_pred = self.noise_net(noisy, t_eval, state)
        reconstruction_error = (noise_pred - noise).pow(2).mean(dim=-1)

        # log_prob proxy: lower error → higher probability, scaled
        log_prob = -reconstruction_error

        # Entropy proxy: average noise prediction variance across timesteps
        # (higher variance = more uncertain = higher entropy)
        entropy = torch.ones(B, device=device) * 0.5  # placeholder constant

        value = self.value_head(state)
        return log_prob, entropy, value
