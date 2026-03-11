"""
Central configuration for Baby-AI.

All hyperparameters, resource budgets, and paths are defined here.
Modify this file to tune the system for your hardware.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import torch
import yaml

# Load .env file from project root (if it exists) — must happen
# BEFORE any os.environ.get() calls below.
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # python-dotenv not installed — rely on real env vars


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------

# Local fast storage (SSD) for code and checkpoints
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Network storage for replay data and large artefacts
NETWORK_STORAGE = Path(os.environ.get("BABY_AI_STORAGE", "./storage"))

# Derived paths
CHECKPOINT_DIR = NETWORK_STORAGE / "checkpoints"
REPLAY_DIR = NETWORK_STORAGE / "replay"
LOG_DIR = NETWORK_STORAGE / "logs"
TENSORBOARD_DIR = NETWORK_STORAGE / "tensorboard"
RAW_DATA_DIR = NETWORK_STORAGE / "raw_data"
SCREENSHOT_DIR = NETWORK_STORAGE / "screenshots"


def ensure_dirs() -> None:
    """Create all required directories if they do not exist."""
    for d in (CHECKPOINT_DIR, REPLAY_DIR, LOG_DIR, TENSORBOARD_DIR, RAW_DATA_DIR, SCREENSHOT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CPU_CORES = os.cpu_count() or 4
GPU_MEM_GB = (
    torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if torch.cuda.is_available()
    else 0.0
)


# ---------------------------------------------------------------------------
# Model dimensions  — Student (small) vs Teacher (large)
# ---------------------------------------------------------------------------

@dataclass
class System2Config:
    """System 2 test-time compute — threshold-triggered latent planning."""
    enabled: bool = True
    uncertainty_threshold: float = 0.5   # trigger planning when uncertainty > this
    num_trajectories: int = 8            # parallel trajectories to evaluate
    planning_horizon: int = 5            # latent rollout steps per trajectory
    planning_budget_ms: float = 150.0    # max time for a single MCTS pass
    discount: float = 0.99              # discount factor for trajectory scoring
    pause_game: bool = True             # send pause command to MC mod during planning
    warmup_steps: int = 0             # min inference steps before System 2 can trigger
    cooldown_steps: int = 20            # min steps between consecutive triggers
    deliberation_rounds: int = 3         # number of MCTS passes (keep best across rounds)
    min_deliberation_ms: float = 300.0   # minimum pause duration (ms) so game visibly freezes
    pause_settle_ms: float = 50.0        # delay after sending pause before thinking (let mod process)


@dataclass
class System3Config:
    """System 3 hierarchical long-horizon planning via latent goals."""
    enabled: bool = True               # enable goal-conditioned hierarchy
    goal_dim: int = 64                 # latent goal embedding size
    num_goal_candidates: int = 8       # GoalProposer outputs K candidates
    max_subgoals: int = 12             # max subgoal sequence length
    planner_layers: int = 2            # SubgoalPlanner transformer layers
    planner_heads: int = 4             # attention heads
    proposer_hidden_dim: int = 256     # GoalProposer MLP width
    achieve_threshold: float = 0.85    # cosine sim to mark subgoal done
    patience_steps: int = 200          # steps before stuck → replan
    min_replan_interval: int = 50      # cooldown between replans
    replan_on_death: bool = True       # full replan after player death
    pause_game_on_replan: bool = True  # freeze MC during full System 3 replan
    warmup_steps: int = 0            # min inference steps before System 3 activates
    propose_every_n: int = 50          # when no goal active, propose one every N steps
    deliberation_rounds: int = 5       # iterative refinement passes during replan
    min_deliberation_ms: float = 500.0 # minimum pause duration (ms) so game visibly freezes
    pause_settle_ms: float = 50.0      # delay after sending pause before thinking (let mod process)


@dataclass
class DiffusionPolicyConfig:
    """Diffusion policy hyperparameters for continuous action generation."""
    action_continuous_dim: int = 20   # continuous action vector size
    num_train_steps: int = 100        # diffusion timesteps during training
    num_infer_steps: int = 4          # DDIM sampling steps for inference (<200ms)
    time_embed_dim: int = 64          # sinusoidal timestep embedding dim
    beta_start: float = 0.0001        # noise schedule start
    beta_end: float = 0.02            # noise schedule end


@dataclass
class JambaConfig:
    """Jamba architecture hyperparameters (Mamba SSM + Mixture of Experts).

    Interleaves Mamba-2 selective state-space blocks with sparse MoE FFNs
    for O(1) per-step inference with infinite context caching.
    """
    num_layers: int = 4           # number of stacked Jamba blocks
    d_state: int = 16             # SSM state dimension (N in Mamba notation)
    d_conv: int = 4               # causal convolution kernel size
    expand: int = 2               # Mamba inner dimension multiplier
    dt_rank: int = 0              # dt projection rank (0 = auto: ceil(dim/16))
    num_experts: int = 4          # total MoE experts per MoE layer
    top_k_routing: int = 1        # experts activated per token
    moe_every_n: int = 2          # MoE every N blocks (others use dense FFN)
    ffn_mult: int = 2             # FFN hidden dimension multiplier
    load_balance_weight: float = 0.01  # auxiliary load-balancing loss weight


@dataclass
class EncoderConfig:
    """Shared encoder dimensionality."""
    vision_embed_dim: int = 256
    audio_embed_dim: int = 256
    code_embed_dim: int = 256
    sensor_embed_dim: int = 128
    fused_dim: int = 512  # after multimodal fusion


@dataclass
class StudentConfig:
    """Student model: 10-30M params, <200ms inference."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    hidden_dim: int = 512        # Jamba hidden state dim
    policy_hidden: int = 512
    comm_vocab_size: int = 4096  # small vocabulary for utterances
    comm_max_len: int = 32
    action_dim: int = 128        # discrete action count (used by env reward computer)
    total_target_params: str = "10-30M"

    # Jamba temporal core
    jamba: JambaConfig = field(default_factory=JambaConfig)

    # Diffusion policy (20-dim continuous actions)
    diffusion: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)

    # System 2 test-time search
    system2: System2Config = field(default_factory=System2Config)

    # System 3 hierarchical goal planning
    system3: System3Config = field(default_factory=System3Config)


@dataclass
class TeacherConfig:
    """Teacher model: 50-100M params, async training."""
    encoder: EncoderConfig = field(default_factory=lambda: EncoderConfig(
        vision_embed_dim=512,
        audio_embed_dim=512,
        code_embed_dim=512,
        sensor_embed_dim=256,
        fused_dim=1024,
    ))
    hidden_dim: int = 1024       # Jamba hidden state dim
    policy_hidden: int = 1024
    comm_vocab_size: int = 4096
    comm_max_len: int = 64
    action_dim: int = 128        # discrete action count (used by env reward computer)
    total_target_params: str = "50-100M"

    # Jamba temporal core — scaled up for Teacher
    jamba: JambaConfig = field(default_factory=lambda: JambaConfig(
        num_layers=4,
        d_state=16,
        expand=1,           # keep inner dim = hidden_dim to control total param count
        num_experts=8,
        top_k_routing=2,
        ffn_mult=1,         # slim experts — 8 experts compensate for narrower FFN
    ))

    # Diffusion policy — Teacher uses more refinement steps
    diffusion: DiffusionPolicyConfig = field(default_factory=lambda: DiffusionPolicyConfig(
        num_infer_steps=20,
    ))

    # System 2 — disabled for Teacher (Teacher trains, doesn't plan)
    system2: System2Config = field(default_factory=lambda: System2Config(enabled=False))

    # System 3 — disabled for Teacher
    system3: System3Config = field(default_factory=lambda: System3Config(enabled=False))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    fps: int = 4                     # sample rate from raw video
    resolution: Tuple[int, int] = (360, 640)  # H, W
    grayscale: bool = False
    clip_length_sec: float = 1.0     # seconds per clip
    channels: int = 3


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_mels: int = 64
    hop_length: int = 160            # 10 ms @ 16 kHz
    win_length: int = 400            # 25 ms @ 16 kHz
    context_sec: float = 1.0         # context window


@dataclass
class CodeConfig:
    max_nodes: int = 256
    max_edges: int = 512
    node_feature_dim: int = 64


@dataclass
class SensorConfig:
    max_channels: int = 16
    frame_rate: int = 30  # fixed-rate normalised frames/sec


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Learning rates (overridden by GUI slider at runtime)
    encoder_lr: float = 5e-5
    core_lr: float = 5e-5
    policy_lr: float = 5e-5
    distill_lr: float = 5e-5

    # Batching
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 2

    # Replay
    replay_capacity: int = 50_000
    replay_disk_cap_gb: float = 4.0  # compressed on-disk cap

    # Distillation
    distill_every_n_steps: int = 100
    distill_kl_weight: float = 1.0
    distill_feature_weight: float = 0.5

    # Intrinsic reward
    intrinsic_weight_start: float = 0.5
    intrinsic_weight_end: float = 0.05
    intrinsic_decay_steps: int = 10_000

    # Consolidation (EWC + rehearsal)
    consolidation_every_n_steps: int = 2000
    ewc_lambda: float = 10.0

    # Mixed precision
    use_amp: bool = True

    # Checkpointing
    checkpoint_every_n_steps: int = 5000


# ---------------------------------------------------------------------------
# Runtime / latency
# ---------------------------------------------------------------------------

@dataclass
class RuntimeConfig:
    inference_target_ms: float = 200.0
    learner_sleep_ms: float = 10.0   # yield between training steps
    distill_sleep_ms: float = 50.0
    max_ram_gb: float = 32.0
    max_gpu_mem_gb: float = 11.0


# ---------------------------------------------------------------------------
# Minecraft environment
# ---------------------------------------------------------------------------

@dataclass
class MinecraftConfig:
    """Settings for the Minecraft game environment."""
    # ── Window / capture ───────────────────────────────────────
    window_title: str = "Minecraft"         # substring matched against window titles
    input_mode: str = "active"              # "background" (safe, no cursor) or "active" (camera look, moves cursor)
    capture_resolution: Tuple[int, int] = (360, 640)  # (H, W) for screen captures
    step_delay_ms: float = 100.0            # minimum ms between consecutive actions
    look_pixels_small: int = 40             # small camera rotation (pixels per step)
    look_pixels_large: int = 160            # large camera rotation
    camera_smooth_steps: int = 8            # sub-steps for mouse interpolation (1 = disabled)
    max_episode_steps: int = 0              # 0 = unlimited
    initial_pause_sec: float = 3.0          # pause before starting to let user alt-tab

    # ── Auto-launcher ──────────────────────────────────────────
    auto_launch: bool = True                # launch MC automatically
    mc_dir: str = field(default_factory=lambda: os.environ.get("MC_DIR", ""))
    mc_version: str = field(default_factory=lambda: os.environ.get("MC_VERSION", "1.21.11"))
    world_name: str = field(default_factory=lambda: os.environ.get("MC_WORLD_NAME", ""))
    player_name: str = field(default_factory=lambda: os.environ.get("MC_PLAYER_NAME", ""))
    player_uuid: str = field(default_factory=lambda: os.environ.get("MC_PLAYER_UUID", ""))
    max_memory_mb: int = field(default_factory=lambda: int(os.environ.get("MC_MAX_MEMORY_MB", "4096")))  # JVM -Xmx (4 GB default; 2 GB causes GC thrashing on 1.21+)
    launch_timeout_sec: float = 120.0       # max wait for MC window to appear
    window_width: int = 1920                # Game window width matching standard 1080p maximize
    window_height: int = 1080               # Game window height matching standard 1080p maximize

    # ── Input guard ────────────────────────────────────────────
    block_user_input: bool = True           # block user KB/mouse when MC focused

    # ── Mod bridge (Fabric mod TCP event stream) ───────────────
    mod_bridge_port: int = 5556             # TCP port the Baby-AI Bridge mod listens on


# ---------------------------------------------------------------------------
# Top-level config bundle
# ---------------------------------------------------------------------------

@dataclass
class BabyAIConfig:
    student: StudentConfig = field(default_factory=StudentConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    code: CodeConfig = field(default_factory=CodeConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    minecraft: MinecraftConfig = field(default_factory=MinecraftConfig)

    # Convenience
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def save(self, path: Path | str) -> None:
        """Serialize config to YAML."""
        import dataclasses
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_dict(v) for v in obj]
            return obj

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path | str) -> "BabyAIConfig":
        """Load config from YAML, recursively merging nested dataclass fields."""
        import dataclasses
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()

        def _merge(target, overlay: dict):
            """Recursively merge a dict overlay into a dataclass instance."""
            for k, v in overlay.items():
                if not hasattr(target, k):
                    continue
                current = getattr(target, k)
                if isinstance(v, dict) and dataclasses.is_dataclass(current):
                    # Recursively merge into the nested dataclass
                    _merge(current, v)
                else:
                    setattr(target, k, v)

        if isinstance(data, dict):
            _merge(cfg, data)
        return cfg


# Singleton default config — importable everywhere
DEFAULT_CONFIG = BabyAIConfig()
