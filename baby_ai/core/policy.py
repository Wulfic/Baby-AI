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

    Action vector layout (default 23-dim):
        [0:2]   camera (yaw_delta, pitch_delta) in [-1, 1]
        [2:6]   movement (forward, back, left, right) in [0, 1]
        [6:14]  actions (attack, use, jump, sneak, sprint, inventory, drop, pick_block) in [0, 1]
        [14:23] hotbar (9 slots, softmax-normalised)

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
        action_dim: int = 23,
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


# ────────────────────────────────────────────────────────────────────────────
# Velocity prediction network (Flow Matching)
# ────────────────────────────────────────────────────────────────────────────

class VelocityPredictor(nn.Module):
    """
    MLP velocity-field predictor conditioned on state and time.

    Input:  concat(x_t, time_embed, state)
    Output: predicted velocity v(x_t, t) — same dim as action.

    Architecture is identical to NoisePredictor — 3 residual blocks + SiLU.
    The semantic difference is in how the output is used during training
    (predicts velocity v = x_1 - x_0 instead of noise ε).
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
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:      (B, action_dim) interpolated sample at time t.
            timestep: (B,) continuous time values in [0, 1].
            state:    (B, state_dim) conditioning signal.

        Returns:
            (B, action_dim) predicted velocity field v(x_t, t).
        """
        t_emb = self.time_embed(timestep)
        x = torch.cat([x_t, t_emb, state], dim=-1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


# ────────────────────────────────────────────────────────────────────────────
# Flow Matching Policy Head
# ────────────────────────────────────────────────────────────────────────────

class FlowMatchingPolicyHead(nn.Module):
    """
    Flow Matching policy: generates actions via ODE integration
    of a learned velocity field.

    Training:
        1. Sample t ~ Uniform(0, 1)
        2. Construct x_t = (1 - t) * noise + t * action  (OT interpolation)
        3. Target velocity: v_target = action - noise
        4. Loss = MSE(v_predicted, v_target)

    Inference:
        1. Start from x_0 ~ N(0, I)
        2. Euler integrate: x_{t+dt} = x_t + dt * v(x_t, t, state)
        3. 1–2 steps suffice (straight OT paths!)

    Args:
        input_dim:       Core hidden state dimension.
        action_dim:      Continuous action vector size.
        hidden_dim:      MLP hidden dimension.
        num_infer_steps: Euler ODE steps for inference.
        time_embed_dim:  Sinusoidal timestep embedding dimension.
        sigma_min:       Minimum noise floor for numerical stability.
    """

    def __init__(
        self,
        input_dim: int = 256,
        action_dim: int = 23,
        hidden_dim: int = 256,
        num_infer_steps: int = 2,
        time_embed_dim: int = 64,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_infer_steps = num_infer_steps
        self.sigma_min = sigma_min

        # ── Velocity predictor (same arch as NoisePredictor) ──
        self.velocity_net = VelocityPredictor(
            action_dim=action_dim,
            state_dim=input_dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=3,
        )

        # ── Value head (state → scalar, independent of flow matching) ──
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    # ── training forward ───────────────────────────────────────────────────

    def forward(
        self,
        state: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training forward pass — compute flow matching loss + value.

        Args:
            state:   (B, input_dim) from temporal core.
            actions: (B, action_dim) ground-truth continuous actions.
                     If None, returns zero loss + value.

        Returns:
            flow_loss: Scalar MSE velocity-field loss.
            value:     (B, 1) state value.
        """
        value = self.value_head(state)

        if actions is None:
            return torch.tensor(0.0, device=state.device), value

        # Guard: squeeze stale replay data stored with batch dim (B,1,D) → (B,D)
        if actions.dim() == 3 and actions.size(1) == 1:
            actions = actions.squeeze(1)

        B = state.size(0)
        device = state.device

        # Sample random time t ∈ [0, 1]
        t = torch.rand(B, device=device)

        # Sample source noise x_0 ~ N(0, I)
        x_0 = torch.randn_like(actions)

        # OT interpolation: x_t = (1 - t) * x_0 + t * actions
        t_expand = t.unsqueeze(-1)  # (B, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * actions

        # Target velocity: v = actions - x_0  (straight line noise → data)
        v_target = actions - x_0

        # Predict velocity
        v_pred = self.velocity_net(x_t, t, state)

        # Loss: MSE on velocity field
        flow_loss = F.mse_loss(v_pred, v_target)

        return flow_loss, value

    # ── inference (Euler ODE integration) ──────────────────────────────────

    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate continuous actions via Euler ODE integration.

        Args:
            state: (B, input_dim).
            deterministic: If True, uses fixed seed noise (not yet implemented;
                           flow matching is inherently stochastic via initial noise).

        Returns:
            action:   (B, action_dim) bounded continuous action vector.
            log_prob: (B,) approximate log-probability.
            value:    (B, 1) state value.
        """
        B = state.size(0)
        device = state.device

        # Start from pure noise x_0 ~ N(0, I)
        x = torch.randn(B, self.action_dim, device=device)

        # Euler integration with uniform time steps: t goes from 0 to 1
        dt = 1.0 / self.num_infer_steps
        for i in range(self.num_infer_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(x, t, state)
            x = x + dt * v

        # Bound the output
        action = self._bound_action(x)

        # Log-prob approximation: use velocity magnitude at final step
        # Low velocity near t≈1 means the sample is near the data manifold
        v_final = self.velocity_net(x, torch.ones(B, device=device), state)
        log_prob = -(v_final.pow(2).sum(dim=-1))

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

    # ── REBEL-compatible evaluation ────────────────────────────────────────

    def evaluate(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate a given action for REBEL-style training.

        For flow matching, log_prob is approximated by how well the velocity
        network can reconstruct the velocity field pointing to this action.

        Returns:
            log_prob: (B,) approximate log probability.
            entropy:  (B,) estimated entropy (placeholder).
            value:    (B, 1) state value.
        """
        B = state.size(0)
        device = state.device

        # Reconstruct: how well can we predict velocity for this action?
        t_eval = torch.full((B,), 0.5, device=device)
        x_0 = torch.randn_like(action)
        t_expand = t_eval.unsqueeze(-1)
        x_t = (1 - t_expand) * x_0 + t_expand * action
        v_target = action - x_0
        v_pred = self.velocity_net(x_t, t_eval, state)

        reconstruction_error = (v_pred - v_target).pow(2).mean(dim=-1)
        log_prob = -reconstruction_error

        # Entropy proxy: proportional to velocity field divergence
        entropy = torch.ones(B, device=device) * 0.5

        value = self.value_head(state)
        return log_prob, entropy, value
