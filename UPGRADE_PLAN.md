# Baby-AI Upgrade Plan: Bleeding-Edge Technology Integration

> **Created:** 2026-03-11
> **Scope:** 5 upgrades — Mamba-2 SSD, Flow Matching, VQ-BeT, torch.compile, REBEL
> **Order:** 7a → 1 → 2 → 8 → 4 (dependency-driven; simplest first, compounding gains)

---

## Execution Order Rationale

| Phase | Upgrade | Why This Order |
|-------|---------|----------------|
| **A** | `torch.compile()` | Zero-risk, zero-refactor; immediate 30–50% speedup; baseline for benchmarking all later changes |
| **B** | Mamba-2 SSD | Core backbone upgrade; all downstream modules (policy, planner, learner) consume its output — do this before changing what feeds into it |
| **C** | Flow Matching | Replaces the policy head's diffusion loop; depends on core output shape being stable (Phase B) |
| **D** | REBEL | Replaces the RL training loss in the learner; depends on the new policy head's `evaluate()` interface (Phase C) |
| **E** | VQ-BeT Action Tokenization | End-to-end change from policy output → env decoder → distillation; do last because it changes the action representation contract |

---

## Phase A: `torch.compile()` Integration

**Risk:** Low | **Effort:** ~1 hour | **Impact:** 30–50% inference latency reduction

### What Changes

Wrap the Student model in `torch.compile` for inference, and optionally the Teacher for training.

### Files Modified

| File | Change |
|------|--------|
| `baby_ai/runtime/inference_thread.py` | Compile Student model at init |
| `baby_ai/runtime/learner_thread.py` | Compile Teacher model at init (optional) |
| `baby_ai/runtime/orchestrator.py` | Pass compilation flag from config |
| `baby_ai/config.py` | Add `compile_student: bool` and `compile_teacher: bool` to `RuntimeConfig` |

### Implementation Details

#### A.1 — Config addition (`config.py`)

Add to `RuntimeConfig`:

```python
@dataclass
class RuntimeConfig:
    # ... existing fields ...
    compile_student: bool = True    # torch.compile Student for inference
    compile_teacher: bool = False   # torch.compile Teacher for training (experimental)
    compile_mode: str = "reduce-overhead"  # "default", "reduce-overhead", or "max-autotune"
```

#### A.2 — Student compilation (`inference_thread.py`)

In `InferenceThread.__init__`, after `self.student = student`:

```python
if compile_student:
    log.info("Compiling Student model with torch.compile(mode='%s')...", compile_mode)
    # Compile the full model — torch.compile handles SSM scan, conv1d, etc.
    # Use reduce-overhead mode for real-time inference (CUDA graphs).
    self.student = torch.compile(self.student, mode=compile_mode)
    # Warmup: run one dummy forward to trigger compilation before live inference
    self._warmup_compile()
```

Add a warmup method:

```python
def _warmup_compile(self):
    """Trigger torch.compile graph capture with a dummy forward pass."""
    device = self.device
    dummy_vision = torch.randn(1, 3, 360, 640, device=device)
    dummy_audio = torch.randn(1, 64, 100, device=device)
    dummy_sensor = torch.randn(1, 32, device=device)
    with torch.no_grad():
        _ = self.student(vision=dummy_vision, audio=dummy_audio, sensor=dummy_sensor)
    log.info("torch.compile warmup complete.")
```

#### A.3 — Teacher compilation (`learner_thread.py`, optional)

Wrap the Teacher similarly, but with `mode="default"` (CUDA graphs conflict with dynamic shapes in replay batching):

```python
if compile_teacher:
    self.teacher = torch.compile(self.teacher, mode="default")
```

### Gotchas & Mitigations

| Issue | Mitigation |
|-------|------------|
| `torch.compile` breaks on data-dependent control flow (e.g., System 2 trigger checks) | Only compile the core forward pass, not the planning path; fencing with `torch.compiler.disable` on System 2/3 methods |
| CUDA Graph replay requires static shapes | Use `reduce-overhead` with a fixed batch dim of 1 for inference; training uses `default` mode |
| `_model_swap_lock` and compiled model | After distillation weight swap, call `student._orig_mod.load_state_dict(...)` then re-assign `self.student = torch.compile(self.student._orig_mod, ...)` — OR use `fullgraph=False` to avoid recompilation |
| First inference is slow (compilation) | `_warmup_compile()` runs at startup, before Minecraft connects |

### Validation

- [ ] Benchmark inference latency before/after (expect 30–50% drop)
- [ ] Verify actions are numerically identical (deterministic DDIM should produce same outputs)
- [ ] Confirm distillation weight swap still works under compiled model
- [ ] Run 5-minute Minecraft session with no crashes

---

## Phase B: Mamba-2 SSD (State Space Duality)

**Risk:** Medium | **Effort:** ~6–10 hours | **Impact:** 2–8x faster temporal core

### What Changes

Replace the sequential Python SSM scan in `MambaBlock` with Mamba-2's **Structured State Space Duality (SSD)** chunked-parallel algorithm. The SSD formulation unifies SSMs and linear attention through semiseparable matrices, enabling tensor-core-friendly chunk-level parallelism during training and identical O(1) recurrence at inference.

### Files Modified

| File | Change |
|------|--------|
| `baby_ai/core/temporal.py` | Rewrite `MambaBlock` internals; new `SSDKernel` class |
| `baby_ai/config.py` | Add `chunk_size` and `ssd_mode: bool` to `JambaConfig` |
| `requirements.txt` / `pyproject.toml` | Add `mamba-ssm>=2.0` or vendor the SSD kernel |

### Implementation Details

#### B.1 — New config fields (`config.py`)

```python
@dataclass
class JambaConfig:
    # ... existing fields ...
    use_ssd: bool = True          # use Mamba-2 SSD kernel (False = original scan)
    chunk_size: int = 64          # SSD chunk length (64 or 128 work well)
    ssd_head_dim: int = 64        # per-head state dimension for SSD multi-head
```

#### B.2 — SSD kernel implementation (`temporal.py`)

The key algorithm change. The current `MambaBlock.forward` has a sequential `for t in range(T)` loop (lines 199–217). Replace with:

**Option A (preferred): Use `mamba_ssm` library**

```python
# pip install mamba-ssm>=2.0
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

# During training (T > 1): chunked parallel scan
# During inference (T == 1): falls back to single-step recurrence automatically
```

**Option B (vendored pure-PyTorch fallback):**

Implement the SSD algorithm directly — this is the principled approach for maximum control:

```python
class SSDScan(torch.autograd.Function):
    """
    Structured State Space Duality: semi-separable matrix decomposition.

    Training: chunk-parallel scan in O(T * chunk_size) with tensor-core matmuls
    Inference: O(1) per-step recurrence (identical to current)
    """

    @staticmethod
    def forward(ctx, A, B, C, x, dt, chunk_size=64):
        # 1. Split sequence into chunks of size `chunk_size`
        # 2. Within each chunk: compute via matrix multiply (semiseparable structure)
        # 3. Across chunks: carry forward the boundary state (recurrence on chunk boundaries)
        # This gives O(T/chunk * chunk^2) = O(T * chunk) ≈ O(T) vs O(T * d_state) sequential
        ...
```

**The actual MambaBlock changes:**

```python
class MambaBlock(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, dt_rank=0, use_ssd=True, chunk_size=64):
        super().__init__()
        # ... existing projections stay the same ...
        self.use_ssd = use_ssd
        self.chunk_size = chunk_size

        if use_ssd:
            # Mamba-2 uses multi-head SSM instead of per-channel SSM
            # d_inner is split into nheads groups, each with head_dim state
            self.nheads = max(1, self.d_inner // 64)  # one SSM head per 64 channels
            # SSD needs A as a scalar per-head decay (not per-channel)
            self.A_log = nn.Parameter(torch.log(torch.ones(self.nheads) * 0.5))
            # dt is per-head, not per-channel
            self.dt_proj = nn.Linear(self.dt_rank, self.nheads, bias=True)
        else:
            # Original Mamba-1 path (kept for ablation / compatibility)
            self.A_log = nn.Parameter(...)  # existing code

    def forward(self, x, ssm_state=None, conv_state=None):
        # Steps 1-2 (in_proj, conv1d) are IDENTICAL to current code

        if self.use_ssd and x.size(1) > 1:
            # ── Training path: SSD chunked parallel scan ──
            y, h = ssd_chunk_scan(A, B, C, x_conv, dt, self.chunk_size, ssm_state)
        elif x.size(1) == 1:
            # ── Inference path: single-step recurrence (unchanged!) ──
            # This is identical to the current code — O(1) per step
            y, h = self._single_step(A, B, C, x_conv, dt, ssm_state)
        else:
            # ── Fallback: sequential scan (original code) ──
            y, h = self._sequential_scan(A, B, C, x_conv, dt, ssm_state)

        # Step 5 (gated output) is IDENTICAL to current code
```

#### B.3 — JambaState update

`JambaState` format stays the same — the SSM state tensor shape is `(B, d_inner, d_state)` for both Mamba-1 and Mamba-2 single-step inference. The SSD kernel just fills it differently during training.

#### B.4 — Checkpoint compatibility

Add a migration path in model loading:

```python
def load_state_dict(self, state_dict, strict=False):
    # If loading Mamba-1 checkpoint into Mamba-2 model:
    # - A_log shape changes from (d_inner, d_state) to (nheads,)
    # - dt_proj output dim changes from d_inner to nheads
    # Handle gracefully with shape-adaptive loading
    ...
```

### Gotchas & Mitigations

| Issue | Mitigation |
|-------|------------|
| `mamba_ssm` package requires Triton + CUDA compilation | Provide pure-PyTorch fallback (`use_ssd=False` in config) |
| A_log shape changes between Mamba-1 and Mamba-2 | Checkpoint migration utility; keep `use_ssd` as a toggleable config flag |
| Multi-head SSM changes total parameter count | Adjust `expand` factor to match target param budget |
| Windows Triton support is spotty | Test with `TRITON_PTXAS_PATH` override; fallback to sequential scan on Windows |

### Validation

- [ ] Unit test: Mamba-2 and Mamba-1 produce identical outputs on single-step forward (inference path)
- [ ] Benchmark: training throughput (tokens/sec) with SSD vs sequential scan
- [ ] End-to-end: load old checkpoint → fine-tune 1000 steps → verify no divergence
- [ ] Profile: GPU utilization during training (should show improved SM occupancy)

---

## Phase C: Flow Matching Policy Head

**Risk:** Medium | **Effort:** ~8–12 hours | **Impact:** 2–4x faster action generation, cleaner gradients

### What Changes

Replace the DDIM denoising diffusion policy with a **Flow Matching** (Continuous Normalizing Flow) policy using Optimal Transport conditional paths. The NoisePredictor MLP architecture stays nearly identical — only the loss, sampling, and parameterization change.

### Core Conceptual Swap

| Aspect | Current (DDIM Diffusion) | New (Flow Matching) |
|--------|--------------------------|---------------------|
| **What the network predicts** | Noise ε added to action | Velocity field v(x_t, t) pointing from noise to data |
| **Training loss** | `MSE(ε_pred, ε_true)` | `MSE(v_pred, v_true)` where `v_true = x_1 - x_0` |
| **Sampling** | DDIM: iterative denoise x_T → x_0 | Euler ODE: `x_{t+dt} = x_t + dt * v(x_t, t)` |
| **Interpolation** | Curved path (√ᾱ_t x₀ + √(1-ᾱ_t) ε) | Straight OT path: `x_t = (1-t) x_0 + t x_1` |
| **Inference steps** | 4 (Student), 20 (Teacher) | **1–2** (Student), **4** (Teacher) |
| **Schedule** | Beta schedule (100 train steps) | No schedule — uniform t ∈ [0, 1] |

### Files Modified

| File | Change |
|------|--------|
| `baby_ai/core/policy.py` | New `FlowMatchingPolicyHead` class; keep `DiffusionPolicyHead` for ablation |
| `baby_ai/config.py` | New `FlowMatchingConfig` dataclass; add `policy_type` selector |
| `baby_ai/models/base.py` | Conditional construction: diffusion or flow matching |
| `baby_ai/core/planner.py` | Update System 2 action sampling to use flow matching `.act()` |
| `baby_ai/learning/distillation.py` | Update action MSE target (no change needed if distilling raw actions) |

### Implementation Details

#### C.1 — Config (`config.py`)

```python
@dataclass
class FlowMatchingConfig:
    """Flow Matching policy hyperparameters."""
    action_continuous_dim: int = 20   # same as diffusion
    time_embed_dim: int = 64
    num_infer_steps: int = 2          # Euler steps for inference (Student)
    sigma_min: float = 1e-4           # minimum noise floor
    ot_method: str = "linear"         # "linear" (OT displacement) or "cosine"

@dataclass
class StudentConfig:
    # ... existing ...
    policy_type: str = "flow_matching"  # "diffusion" or "flow_matching"
    diffusion: DiffusionPolicyConfig = ...
    flow_matching: FlowMatchingConfig = field(default_factory=FlowMatchingConfig)
```

#### C.2 — VelocityPredictor (`policy.py`)

Rename/modify the `NoisePredictor` — the architecture is identical, only the semantics change:

```python
class VelocityPredictor(nn.Module):
    """
    MLP velocity-field predictor conditioned on state and time.

    Input:  concat(x_t, time_embed, state)
    Output: predicted velocity v(x_t, t) — same dim as action.

    Architecture is identical to NoisePredictor — 3 residual blocks + SiLU.
    """
    # Exact same code as NoisePredictor — just the name/docstring changes.
    # The semantic difference is in how the output is used during training & sampling.
    def __init__(self, action_dim, state_dim, time_embed_dim=64, hidden_dim=256, num_blocks=3):
        # ... identical to NoisePredictor ...

    def forward(self, x_t, timestep, state):
        # ... identical to NoisePredictor ...
        return self.output_proj(x)  # returns velocity, not noise
```

#### C.3 — FlowMatchingPolicyHead (`policy.py`)

```python
class FlowMatchingPolicyHead(nn.Module):
    """
    Flow Matching policy: generates actions via ODE integration of a learned velocity field.

    Training:
        1. Sample t ~ Uniform(0, 1)
        2. Construct x_t = (1 - t) * noise + t * action   (OT interpolation)
        3. Target velocity: v_target = action - noise
        4. Loss = MSE(v_predicted, v_target)

    Inference:
        1. Start from x_0 ~ N(0, I)
        2. Euler integrate: x_{t+dt} = x_t + dt * v(x_t, t, state)
        3. 1–2 steps suffice (straight OT paths!)
    """

    def __init__(self, input_dim, action_dim=20, hidden_dim=256,
                 num_infer_steps=2, time_embed_dim=64, sigma_min=1e-4):
        super().__init__()
        self.action_dim = action_dim
        self.num_infer_steps = num_infer_steps
        self.sigma_min = sigma_min

        # Velocity predictor (same architecture as NoisePredictor)
        self.velocity_net = VelocityPredictor(
            action_dim=action_dim,
            state_dim=input_dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
            num_blocks=3,
        )

        # Value head (unchanged from diffusion)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, actions=None):
        """Training: compute flow matching loss + value."""
        value = self.value_head(state)
        if actions is None:
            return torch.tensor(0.0, device=state.device), value

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

        # Target velocity: v = actions - x_0 (straight line from noise to data)
        v_target = actions - x_0

        # Predict velocity
        v_pred = self.velocity_net(x_t, t, state)

        # Loss: MSE on velocity field
        flow_loss = F.mse_loss(v_pred, v_target)

        return flow_loss, value

    @torch.no_grad()
    def act(self, state, deterministic=False):
        """Generate actions via Euler ODE integration."""
        B = state.size(0)
        device = state.device

        # Start from noise
        x = torch.randn(B, self.action_dim, device=device)

        # Euler integration with uniform time steps
        dt = 1.0 / self.num_infer_steps
        for i in range(self.num_infer_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.velocity_net(x, t, state)
            x = x + dt * v

        action = self._bound_action(x)

        # Log-prob approximation: use velocity magnitude at final step
        # Low velocity at t≈1 means the sample is near the data manifold → high prob
        v_final = self.velocity_net(x, torch.ones(B, device=device), state)
        log_prob = -(v_final.pow(2).sum(dim=-1))

        value = self.value_head(state)
        return action, log_prob, value

    def evaluate(self, state, action):
        """REBEL-compatible evaluation (Phase D will use this)."""
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

        # Entropy: proportional to velocity field divergence (placeholder for now)
        entropy = torch.ones(B, device=device) * 0.5

        value = self.value_head(state)
        return log_prob, entropy, value

    def _bound_action(self, raw):
        """Same bounding as DiffusionPolicyHead."""
        action = raw.clone()
        action[:, :2] = torch.tanh(action[:, :2])
        if action.size(-1) > 2:
            action[:, 2:] = torch.sigmoid(action[:, 2:])
        return action
```

#### C.4 — Model construction (`models/base.py`)

In `BabyAgentBase.__init__`, conditionally build the policy head:

```python
if policy_type == "flow_matching":
    self.policy = FlowMatchingPolicyHead(
        input_dim=hidden_dim,
        action_dim=diffusion_config.action_continuous_dim,
        hidden_dim=policy_hidden,
        num_infer_steps=flow_config.num_infer_steps,
        time_embed_dim=flow_config.time_embed_dim,
    )
else:
    self.policy = DiffusionPolicyHead(...)  # existing code
```

### Gotchas & Mitigations

| Issue | Mitigation |
|-------|------------|
| Old checkpoints have `noise_net.*` keys, new model has `velocity_net.*` | Key-remapping in `load_state_dict`: `noise_net → velocity_net` (architectures are identical) |
| System 2 planner calls `policy.act(stochastic=True)` | Flow matching act() inherently supports stochastic sampling (just use different initial noise) |
| Distillation uses action MSE from Teacher | No change needed — distillation matches raw action vectors, not internal network outputs |
| Teacher still uses Diffusion during transition | Support mixed: Teacher=diffusion, Student=flow_matching; distill on raw actions |

### Validation

- [ ] Unit test: FlowMatchingPolicyHead produces valid bounded actions (camera in [-1,1], rest in [0,1])
- [ ] Benchmark: latency of 2-step Euler vs 4-step DDIM (expect ~2x speedup)
- [ ] Training: flow loss converges on dummy data within 1000 steps
- [ ] Minecraft: 5-minute session with flow matching produces reasonable behavior
- [ ] A/B: compare reward accumulation over 10K steps between diffusion and flow matching

---

## Phase D: REBEL (Regressing Relative Rewards)

**Risk:** Medium | **Effort:** ~6–8 hours | **Impact:** Fixes broken RL training; proper policy optimization

### What Changes

Replace the current reward-weighted denoising + placeholder PPO with **REBEL** — a minimalist RL algorithm that reduces policy optimization to regressing the relative reward between two trajectory completions. This completely removes the need for:
- Value network (for REBEL itself; keep for System 2 trajectory scoring)
- Clipped surrogate objective
- GAE / advantage estimation
- Real entropy computation
- log_prob approximation hacks

### Core Algorithm

```
For each training step:
    1. Sample two transitions (s, a₁, r₁) and (s, a₂, r₂) from replay
    2. Compute relative reward: Δr = r₁ - r₂
    3. Compute policy score: score(a|s) = -||v_pred(x_t, t, s) - v_target||²  (flow matching reconstruction)
    4. REBEL loss = -Δr * (score(a₁|s) - score(a₂|s))
    5. Essentially: if a₁ has higher reward, push policy toward a₁ relative to a₂
```

### Files Modified

| File | Change |
|------|--------|
| `baby_ai/learning/rebel.py` | **New file** — REBEL loss computation |
| `baby_ai/runtime/learner_thread.py` | Replace `_compute_loss` policy gradient section with REBEL |
| `baby_ai/config.py` | Add `REBELConfig` dataclass |
| `baby_ai/memory/replay_buffer.py` | Add paired sampling method |

### Implementation Details

#### D.1 — Config (`config.py`)

```python
@dataclass
class REBELConfig:
    """REBEL RL training hyperparameters."""
    enabled: bool = True
    beta: float = 0.1             # KL regularization weight (toward reference policy)
    pair_sampling: str = "same_state"  # "same_state" or "random"
    reward_clip: float = 5.0      # clip relative rewards
    value_loss_weight: float = 0.5  # keep training value head for System 2
```

#### D.2 — REBEL module (`baby_ai/learning/rebel.py`)

```python
"""
REBEL: Reinforcement Learning via Regressing Relative Rewards.

Reduces policy optimization to regression:
    L = -Δr · (π_score(a₁|s) - π_score(a₂|s))

Where π_score is the policy's log-probability proxy (negative reconstruction error
from the flow matching velocity field).

Reference: arXiv 2404.16767
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class REBELLoss(nn.Module):
    """
    Computes the REBEL policy optimization loss.

    Given pairs of (state, action, reward):
        (s, a_w, r_w)  — the "winner" (higher reward)
        (s, a_l, r_l)  — the "loser"  (lower reward)

    Loss = -log σ(β · (score(a_w|s) - score(a_l|s)))

    This is equivalent to a Bradley-Terry preference model
    and reduces to Natural Policy Gradient in the tabular setting.
    """

    def __init__(self, beta: float = 0.1, reward_clip: float = 5.0):
        super().__init__()
        self.beta = beta
        self.reward_clip = reward_clip

    def forward(
        self,
        state: torch.Tensor,        # (B, D) core hidden state
        action_w: torch.Tensor,      # (B, action_dim) winner action
        action_l: torch.Tensor,      # (B, action_dim) loser action
        reward_w: torch.Tensor,      # (B,) winner reward
        reward_l: torch.Tensor,      # (B,) loser reward
        policy: nn.Module,           # policy head with .evaluate()
    ) -> torch.Tensor:
        """
        Compute REBEL loss.

        Returns:
            Scalar loss (lower = better policy).
        """
        # Get policy scores for both actions
        log_prob_w, _, _ = policy.evaluate(state, action_w)  # (B,)
        log_prob_l, _, _ = policy.evaluate(state, action_l)  # (B,)

        # Relative reward (clipped for stability)
        delta_r = torch.clamp(reward_w - reward_l, -self.reward_clip, self.reward_clip)

        # REBEL: sigmoid cross-entropy on score difference weighted by reward difference
        score_diff = self.beta * (log_prob_w - log_prob_l)

        # When delta_r > 0: push score_diff positive (prefer winner)
        # When delta_r < 0: push score_diff negative (prefer loser)
        # When delta_r ≈ 0: no gradient (actions equally good)
        loss = -F.logsigmoid(delta_r * score_diff).mean()

        return loss
```

#### D.3 — Paired sampling (`replay_buffer.py`)

Add a method to sample pairs:

```python
def sample_pairs(self, batch_size: int, device: str = "cpu"):
    """
    Sample pairs of transitions for REBEL training.

    Strategy: sample 2*batch_size transitions, pair them up.
    Within each pair, the one with higher reward is the "winner".
    """
    transitions, weights, indices = self.sample(batch_size * 2, device)

    # Split into two halves
    t1 = transitions[:batch_size]
    t2 = transitions[batch_size:]

    # Determine winner/loser by reward
    pairs = []
    for a, b in zip(t1, t2):
        r_a = a.get("reward", 0.0)
        r_b = b.get("reward", 0.0)
        if isinstance(r_a, torch.Tensor): r_a = r_a.item()
        if isinstance(r_b, torch.Tensor): r_b = r_b.item()
        if r_a >= r_b:
            pairs.append((a, b, r_a, r_b))
        else:
            pairs.append((b, a, r_b, r_a))

    return pairs, weights[:batch_size], indices[:batch_size]
```

#### D.4 — Learner integration (`learner_thread.py`)

Replace the reward-weighted denoising section in `_compute_loss`:

```python
from baby_ai.learning.rebel import REBELLoss

# In __init__:
self.rebel_loss = REBELLoss(beta=rebel_config.beta, reward_clip=rebel_config.reward_clip)

# In _compute_loss, replace the PPO/advantage section:
if self.rebel_config.enabled and "action" in batch and "reward" in batch:
    # REBEL training — pair-based loss
    rebel_loss = self._compute_rebel_loss(batch)
    loss = loss + rebel_loss

    # Still train value head for System 2 trajectory scoring
    if "reward" in batch:
        value_loss = F.smooth_l1_loss(value.squeeze(-1), batch["reward"], reduction='none')
        value_loss = (value_loss * weights).mean()
        loss = loss + self.rebel_config.value_loss_weight * value_loss
```

### Gotchas & Mitigations

| Issue | Mitigation |
|-------|------------|
| Pairing transitions from different states is noisy | Start with random pairing; later implement same-state pairing by storing state hashes in replay |
| REBEL needs decent policy initialization first | Keep distillation as primary Student improvement; REBEL supplements Teacher training |
| Value head still needed for System 2 scoring | Keep value head training with simple L1 loss (separate from REBEL) |
| β hyperparameter sensitivity | Start with β=0.1, sweep [0.01, 0.5]; log the score_diff distribution |

### Validation

- [ ] Unit test: REBEL loss decreases when model correctly ranks winner > loser
- [ ] Compare: Teacher reward accumulation REBEL vs reward-weighted denoising over 5K steps
- [ ] Verify: System 2 still works (value head remains well-calibrated)
- [ ] Profile: REBEL adds <10% overhead vs current training loop

---

## Phase E: VQ-BeT Action Tokenization

**Risk:** High | **Effort:** ~12–16 hours | **Impact:** Multimodal action capture, faster distillation, action chunking

### What Changes

Replace the raw 20-dim continuous action vector with a **hierarchical VQ codebook** that tokenizes action sequences into discrete codes. This enables:
1. **Multimodal action distributions** — different codebook entries capture distinct behavior modes (mining, building, exploring)
2. **Action chunking** — predict 4–8 actions in one forward pass for temporal consistency
3. **Simpler distillation** — distill codebook indices (cross-entropy) instead of continuous vectors (MSE)

### Architecture Addition

```
Current:  core_state → DiffusionPolicyHead → 20-dim continuous → ActionDecoder → keys/mouse
New:      core_state → FlowMatchingHead → 20-dim continuous
                                             ↓
                                         VQ-BeT Tokenizer
                                             ↓
                                      Codebook index (1 of K)
                                             ↓
                                       Decoded action chunk
                                             ↓
                                        ActionDecoder → keys/mouse
```

### Files Modified/Created

| File | Change |
|------|--------|
| `baby_ai/core/action_tokenizer.py` | **New file** — VQ-BeT codebook module |
| `baby_ai/core/policy.py` | Integrate VQ after flow matching output |
| `baby_ai/config.py` | Add `VQConfig` dataclass |
| `baby_ai/environments/minecraft/action_decoder.py` | Accept both raw continuous and VQ-decoded actions |
| `baby_ai/learning/distillation.py` | Add codebook index distillation (cross-entropy) |
| `baby_ai/models/base.py` | Instantiate and wire up the tokenizer |

### Implementation Details

#### E.1 — Config (`config.py`)

```python
@dataclass
class VQConfig:
    """VQ-BeT action tokenizer hyperparameters."""
    enabled: bool = True
    num_codes: int = 512          # codebook size (K)
    code_dim: int = 64            # codebook embedding dimension
    num_residual: int = 2         # hierarchical VQ levels (residual VQ)
    commitment_weight: float = 0.25  # VQ commitment loss weight
    action_chunk_size: int = 1    # predict N actions at once (1 = no chunking initially)
    ema_update: bool = True       # EMA codebook updates (vs gradient)
    ema_decay: float = 0.99       # codebook EMA decay rate
```

#### E.2 — VQ-BeT module (`baby_ai/core/action_tokenizer.py`)

```python
"""
VQ-BeT: Vector-Quantized Behavior Transformer action tokenizer.

Converts continuous action vectors into discrete codebook indices
using Residual Vector Quantization, capturing multimodal action
distributions (mining, building, exploring, combat, etc.).

Reference: arXiv 2403.03181
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    Single VQ layer with EMA codebook updates.

    Encoder: continuous action → nearest codebook vector
    Decoder: codebook vector → reconstructed action
    """

    def __init__(self, num_codes=512, code_dim=64, ema_decay=0.99,
                 commitment_weight=0.25, ema_update=True):
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_weight = commitment_weight
        self.ema_update = ema_update
        self.ema_decay = ema_decay

        # Codebook
        self.embedding = nn.Embedding(num_codes, code_dim)
        nn.init.uniform_(self.embedding.weight, -1/num_codes, 1/num_codes)

        if ema_update:
            self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
            self.register_buffer("_ema_embed_sum", self.embedding.weight.clone())

    def forward(self, z):
        """
        Args:
            z: (B, code_dim) continuous vectors to quantize.

        Returns:
            z_q: (B, code_dim) quantized vectors.
            indices: (B,) codebook indices.
            vq_loss: scalar commitment + codebook loss.
        """
        # Find nearest codebook vector
        distances = torch.cdist(z.unsqueeze(0), self.embedding.weight.unsqueeze(0)).squeeze(0)
        indices = distances.argmin(dim=-1)  # (B,)
        z_q = self.embedding(indices)       # (B, code_dim)

        if self.training:
            if self.ema_update:
                self._ema_update(z, indices)
                vq_loss = self.commitment_weight * F.mse_loss(z.detach(), z_q)
            else:
                codebook_loss = F.mse_loss(z_q, z.detach())
                commitment_loss = F.mse_loss(z_q.detach(), z)
                vq_loss = codebook_loss + self.commitment_weight * commitment_loss
        else:
            vq_loss = torch.tensor(0.0, device=z.device)

        # Straight-through estimator: gradient flows through z_q as if it were z
        z_q = z + (z_q - z).detach()

        return z_q, indices, vq_loss

    def _ema_update(self, z, indices):
        """Update codebook via exponential moving averages."""
        one_hot = F.one_hot(indices, self.num_codes).float()
        cluster_size = one_hot.sum(dim=0)
        embed_sum = one_hot.T @ z

        self._ema_cluster_size.mul_(self.ema_decay).add_(cluster_size, alpha=1 - self.ema_decay)
        self._ema_embed_sum.mul_(self.ema_decay).add_(embed_sum, alpha=1 - self.ema_decay)

        n = self._ema_cluster_size.sum()
        cluster_size = (self._ema_cluster_size + 1e-5) / (n + self.num_codes * 1e-5) * n
        self.embedding.weight.data.copy_(self._ema_embed_sum / cluster_size.unsqueeze(1))


class ResidualVQ(nn.Module):
    """
    Hierarchical Residual Vector Quantization.

    Applies multiple VQ layers in sequence, each quantizing
    the residual from the previous layer. This captures both
    coarse behavior modes (level 0) and fine adjustments (level 1+).
    """

    def __init__(self, num_levels=2, num_codes=512, code_dim=64, **kwargs):
        super().__init__()
        self.levels = nn.ModuleList([
            VectorQuantizer(num_codes=num_codes, code_dim=code_dim, **kwargs)
            for _ in range(num_levels)
        ])

    def forward(self, z):
        """
        Returns:
            z_q: (B, code_dim) final quantized vector (sum of all levels).
            all_indices: list of (B,) index tensors per level.
            total_vq_loss: scalar sum of VQ losses.
        """
        residual = z
        z_q = torch.zeros_like(z)
        all_indices = []
        total_loss = torch.tensor(0.0, device=z.device)

        for vq in self.levels:
            z_q_level, indices, vq_loss = vq(residual)
            z_q = z_q + z_q_level
            residual = residual - z_q_level.detach()
            all_indices.append(indices)
            total_loss = total_loss + vq_loss

        return z_q, all_indices, total_loss


class ActionTokenizer(nn.Module):
    """
    Full VQ-BeT action tokenizer.

    Encoder: 20-dim action → project → VQ → codebook indices
    Decoder: codebook indices → reconstruct → 20-dim action

    Wraps around the policy: the policy generates continuous actions,
    the tokenizer discretizes them, and the decoder maps back.
    """

    def __init__(self, action_dim=20, code_dim=64, num_codes=512,
                 num_residual=2, **vq_kwargs):
        super().__init__()
        self.action_dim = action_dim
        self.code_dim = code_dim

        # Encoder: action → latent
        self.encoder = nn.Sequential(
            nn.Linear(action_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, code_dim),
        )

        # Residual VQ
        self.rvq = ResidualVQ(
            num_levels=num_residual,
            num_codes=num_codes,
            code_dim=code_dim,
            **vq_kwargs,
        )

        # Decoder: latent → action
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, code_dim),
            nn.SiLU(),
            nn.Linear(code_dim, action_dim),
        )

    def encode(self, action):
        """Encode continuous action → codebook indices."""
        z = self.encoder(action)
        z_q, indices, vq_loss = self.rvq(z)
        return z_q, indices, vq_loss

    def decode(self, z_q):
        """Decode quantized latent → continuous action."""
        return self.decoder(z_q)

    def forward(self, action):
        """Full encode-decode pass (for training the tokenizer)."""
        z_q, indices, vq_loss = self.encode(action)
        reconstructed = self.decode(z_q)
        recon_loss = F.mse_loss(reconstructed, action)
        return reconstructed, indices, vq_loss + recon_loss

    def decode_from_indices(self, indices_list):
        """Decode directly from codebook index lists (for inference)."""
        z_q = torch.zeros(indices_list[0].size(0), self.code_dim, device=indices_list[0].device)
        for indices, vq in zip(indices_list, self.rvq.levels):
            z_q = z_q + vq.embedding(indices)
        return self.decoder(z_q)
```

#### E.3 — Integration into base model (`models/base.py`)

```python
# In BabyAgentBase.__init__:
if vq_config and vq_config.enabled:
    self.action_tokenizer = ActionTokenizer(
        action_dim=diffusion_config.action_continuous_dim,
        code_dim=vq_config.code_dim,
        num_codes=vq_config.num_codes,
        num_residual=vq_config.num_residual,
        commitment_weight=vq_config.commitment_weight,
        ema_update=vq_config.ema_update,
        ema_decay=vq_config.ema_decay,
    )
else:
    self.action_tokenizer = None

# In forward(), after policy generates actions:
if self.action_tokenizer is not None and actions is not None:
    _, vq_indices, vq_loss = self.action_tokenizer.encode(actions)
    outputs["vq_indices"] = vq_indices
    outputs["vq_loss"] = vq_loss
```

#### E.4 — Distillation with codebook indices (`distillation.py`)

Add codebook-level distillation alongside existing continuous action MSE:

```python
# In distillation loss computation:
if hasattr(teacher_model, 'action_tokenizer') and teacher_model.action_tokenizer is not None:
    # Teacher's codebook indices are the target
    _, teacher_indices, _ = teacher_model.action_tokenizer.encode(teacher_actions)
    _, student_indices, _ = student_model.action_tokenizer.encode(student_actions)

    # Cross-entropy on level-0 (coarse behavior mode) indices
    # This is optional and supplements continuous action MSE
    codebook_loss = F.cross_entropy(
        student_logits_for_indices,  # need router logits, not just argmax
        teacher_indices[0],
    )
    distill_loss += 0.1 * codebook_loss
```

### Gotchas & Mitigations

| Issue | Mitigation |
|-------|------------|
| Codebook collapse (only a few codes used) | EMA updates + commitment loss; monitor code usage in logs; reset dead codes periodically |
| Reconstruction error degrades action quality | Start with VQ disabled (`enabled: false`); pre-train tokenizer on 10K replay transitions before going live |
| Action chunking breaks real-time step loop | Start with `chunk_size=1`; only enable chunking after tokenizer is stable |
| VQ loss must not interfere with policy gradient | Detach VQ loss from policy backward; train tokenizer as a separate auxiliary loss |

### Validation

- [ ] Unit test: encode → VQ → decode produces actions within 5% of originals
- [ ] Codebook usage: >50% of codes used after 1K training steps (no collapse)
- [ ] End-to-end: VQ actions produce similar rewards as raw continuous actions
- [ ] Distillation: codebook index cross-entropy converges
- [ ] Profile: VQ adds <5ms to per-step inference

---

## Cross-Phase Dependencies

```
Phase A (torch.compile)
    │
    ▼
Phase B (Mamba-2 SSD)  ──── temporal.py rewrite
    │
    ▼
Phase C (Flow Matching) ──── policy.py rewrite, depends on stable core output
    │
    ▼
Phase D (REBEL) ──────────── learner_thread.py, depends on policy.evaluate()
    │
    ▼
Phase E (VQ-BeT) ─────────── action_tokenizer.py, touches policy + decoder + distillation
```

## Checkpoint Migration Strategy

Each phase should maintain backward compatibility:

1. **Phase A**: No checkpoint changes
2. **Phase B**: Save `use_ssd` flag in checkpoint; load gracefully with shape remapping
3. **Phase C**: Save `policy_type` in checkpoint; support loading diffusion weights into flow matching
4. **Phase D**: No checkpoint changes (REBEL only affects loss computation)
5. **Phase E**: Save codebook state; load without VQ for older checkpoints

## Rollback Strategy

Each phase can be individually disabled via config:

```yaml
runtime:
  compile_student: false       # Phase A off
jamba:
  use_ssd: false               # Phase B off → falls back to sequential scan
student:
  policy_type: "diffusion"     # Phase C off → uses old DDIM
rebel:
  enabled: false               # Phase D off → uses reward-weighted denoising
vq:
  enabled: false               # Phase E off → uses raw continuous actions
```

## Estimated Total Timeline

| Phase | Effort | Cumulative |
|-------|--------|------------|
| A: torch.compile | 1 hour | 1 hour |
| B: Mamba-2 SSD | 6–10 hours | 7–11 hours |
| C: Flow Matching | 8–12 hours | 15–23 hours |
| D: REBEL | 6–8 hours | 21–31 hours |
| E: VQ-BeT | 12–16 hours | 33–47 hours |
| **Total** | **33–47 hours** | ~1–2 weeks of focused work |
