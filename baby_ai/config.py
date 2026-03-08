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


def ensure_dirs() -> None:
    """Create all required directories if they do not exist."""
    for d in (CHECKPOINT_DIR, REPLAY_DIR, LOG_DIR, TENSORBOARD_DIR, RAW_DATA_DIR):
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
    gru_hidden: int = 512
    gru_layers: int = 2
    policy_hidden: int = 512
    comm_vocab_size: int = 4096  # small vocabulary for utterances
    comm_max_len: int = 32
    action_dim: int = 128  # discrete action space size
    total_target_params: str = "10-30M"


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
    gru_hidden: int = 1024
    gru_layers: int = 3
    policy_hidden: int = 1024
    comm_vocab_size: int = 4096
    comm_max_len: int = 64
    action_dim: int = 128
    total_target_params: str = "50-100M"


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    fps: int = 4                     # sample rate from raw video
    resolution: Tuple[int, int] = (160, 160)  # H, W
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
    # Learning rates
    encoder_lr: float = 1e-4
    core_lr: float = 1e-4
    policy_lr: float = 3e-4
    distill_lr: float = 5e-5

    # Batching
    micro_batch_size: int = 16
    gradient_accumulation_steps: int = 2

    # Replay
    replay_capacity: int = 50_000
    replay_disk_cap_gb: float = 4.0  # compressed on-disk cap

    # Distillation
    distill_every_n_steps: int = 250
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

    # Learning Rate Warmup
    warmup_steps: int = 1000

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
    capture_resolution: Tuple[int, int] = (160, 160)  # (H, W) for screen captures
    step_delay_ms: float = 100.0            # minimum ms between consecutive actions
    look_pixels_small: int = 40             # small camera rotation (pixels per step)
    look_pixels_large: int = 160            # large camera rotation
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
            return obj

        with open(path, "w") as f:
            yaml.dump(_to_dict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path | str) -> "BabyAIConfig":
        """Load config from YAML (flat dict merge)."""
        with open(path) as f:
            data = yaml.safe_load(f)
        cfg = cls()
        # Simple overlay — only handles top-level sub-config keys
        for section_name, section_data in data.items():
            if hasattr(cfg, section_name) and isinstance(section_data, dict):
                sub = getattr(cfg, section_name)
                for k, v in section_data.items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
        return cfg


# Singleton default config — importable everywhere
DEFAULT_CONFIG = BabyAIConfig()
