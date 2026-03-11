"""
Base agent model — shared architecture for Student and Teacher.

Composes: modality encoders → multimodal fusion → Jamba core →
diffusion policy head + communication head + latent world model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from baby_ai.encoders.vision import VisionEncoder
from baby_ai.encoders.audio import AudioEncoder
from baby_ai.encoders.code import CodeEncoder
from baby_ai.encoders.multimodal import MultimodalFusion
from baby_ai.core.temporal import JambaCore
from baby_ai.core.policy import DiffusionPolicyHead
from baby_ai.core.communication import CommunicationHead
from baby_ai.core.predictive import LatentWorldModel
from baby_ai.core.goals import GoalConditioner
from baby_ai.config import JambaConfig, DiffusionPolicyConfig


class BabyAgentBase(nn.Module):
    """
    Complete agent model composing all components.

    This is the base class used by both Student and Teacher.
    Scale is controlled via configuration parameters.

    Args:
        vision_embed_dim: Vision encoder output dim.
        audio_embed_dim: Audio encoder output dim.
        code_embed_dim: Code encoder output dim.
        sensor_embed_dim: Sensor encoder output dim (simple linear).
        fused_dim: Multimodal fusion output dim.
        hidden_dim: Jamba hidden state dim.
        policy_hidden: Policy MLP hidden dim.
        action_dim: Discrete action count (used only by env reward computer).
        comm_vocab_size: Communication vocabulary size.
        comm_max_len: Max utterance length.
        vision_width_mult: Width multiplier for vision encoder.
        audio_width_mult: Width multiplier for audio encoder.
        code_hidden_dim: GNN hidden dim.
        code_num_layers: Number of GNN layers.
        n_mels: Number of mel frequency bins.
        code_node_feat_dim: Node feature dimension for code GNN.
        sensor_channels: Number of sensor input channels.
    """

    def __init__(
        self,
        vision_embed_dim: int = 128,
        audio_embed_dim: int = 128,
        code_embed_dim: int = 128,
        sensor_embed_dim: int = 64,
        fused_dim: int = 256,
        hidden_dim: int = 256,
        policy_hidden: int = 256,
        action_dim: int = 64,
        comm_vocab_size: int = 4096,
        comm_max_len: int = 32,
        vision_width_mult: float = 0.5,
        audio_width_mult: float = 1.0,
        code_hidden_dim: int = 128,
        code_num_layers: int = 3,
        n_mels: int = 64,
        code_node_feat_dim: int = 64,
        sensor_channels: int = 16,
        jamba_config: JambaConfig | None = None,
        diffusion_config: DiffusionPolicyConfig | None = None,
        goal_dim: int = 0,
    ):
        super().__init__()

        # --- Modality encoders ---
        self.vision_encoder = VisionEncoder(
            in_channels=3,
            embed_dim=vision_embed_dim,
            width_mult=vision_width_mult,
        )
        self.audio_encoder = AudioEncoder(
            n_mels=n_mels,
            embed_dim=audio_embed_dim,
            width_mult=audio_width_mult,
        )
        self.code_encoder = CodeEncoder(
            node_feature_dim=code_node_feat_dim,
            hidden_dim=code_hidden_dim,
            embed_dim=code_embed_dim,
            num_layers=code_num_layers,
        )

        # Simple sensor encoder (MLP on normalized sensor vector)
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_channels, sensor_embed_dim),
            nn.ReLU(),
            nn.Linear(sensor_embed_dim, sensor_embed_dim),
            nn.LayerNorm(sensor_embed_dim),
        )

        # --- Multimodal fusion ---
        self.fusion = MultimodalFusion(
            modality_dims={
                "vision": vision_embed_dim,
                "audio": audio_embed_dim,
                "code": code_embed_dim,
                "sensor": sensor_embed_dim,
            },
            fused_dim=fused_dim,
        )

        # --- Temporal core ---
        if jamba_config is None:
            jamba_config = JambaConfig()
        self.temporal = JambaCore(
            input_dim=fused_dim,
            hidden_dim=hidden_dim,
            num_layers=jamba_config.num_layers,
            d_state=jamba_config.d_state,
            d_conv=jamba_config.d_conv,
            expand=jamba_config.expand,
            dt_rank=jamba_config.dt_rank,
            num_experts=jamba_config.num_experts,
            top_k_routing=jamba_config.top_k_routing,
            moe_every_n=jamba_config.moe_every_n,
            ffn_mult=jamba_config.ffn_mult,
            load_balance_weight=jamba_config.load_balance_weight,
        )

        # --- Output heads ---
        if diffusion_config is None:
            diffusion_config = DiffusionPolicyConfig()
        self.policy = DiffusionPolicyHead(
            input_dim=hidden_dim,
            action_dim=diffusion_config.action_continuous_dim,
            hidden_dim=policy_hidden,
            num_train_steps=diffusion_config.num_train_steps,
            num_infer_steps=diffusion_config.num_infer_steps,
            time_embed_dim=diffusion_config.time_embed_dim,
            beta_start=diffusion_config.beta_start,
            beta_end=diffusion_config.beta_end,
        )
        self.communication = CommunicationHead(
            input_dim=hidden_dim,
            vocab_size=comm_vocab_size,
            hidden_dim=hidden_dim,
            max_len=comm_max_len,
        )
        self.predictive = LatentWorldModel(
            state_dim=hidden_dim,      # core output dim (world model observes core state)
            action_dim=diffusion_config.action_continuous_dim,
            latent_dim=hidden_dim,
            hidden_dim=hidden_dim,
            stochastic_dim=32,
        )

        # --- Goal conditioning (System 3) ---
        self.goal_dim = goal_dim
        if goal_dim > 0:
            self.goal_conditioner = GoalConditioner(
                state_dim=hidden_dim,
                goal_dim=goal_dim,
            )
        else:
            self.goal_conditioner = None

        # Store dims for external access
        self.fused_dim = fused_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

    def encode(
        self,
        vision: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        code_x: Optional[torch.Tensor] = None,
        code_edge_index: Optional[torch.Tensor] = None,
        code_batch: Optional[torch.Tensor] = None,
        sensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode raw modalities into a fused embedding.

        Returns:
            (B, fused_dim) fused multimodal embedding.
        """
        embeddings = {}

        if vision is not None:
            embeddings["vision"] = self.vision_encoder(vision)

        if audio is not None:
            embeddings["audio"] = self.audio_encoder(audio)

        if code_x is not None and code_edge_index is not None:
            embeddings["code"] = self.code_encoder(code_x, code_edge_index, code_batch)

        if sensor is not None:
            embeddings["sensor"] = self.sensor_encoder(sensor)

        return self.fusion(embeddings)

    def forward(
        self,
        vision: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        code_x: Optional[torch.Tensor] = None,
        code_edge_index: Optional[torch.Tensor] = None,
        code_batch: Optional[torch.Tensor] = None,
        sensor: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        goal: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Full forward pass: encode → fuse → temporal → policy + comm.

        Returns dict with:
            fused, core_state, hidden, value, comm_logits,
            denoising_loss (scalar), action (B, action_dim).
        """
        fused = self.encode(
            vision=vision, audio=audio,
            code_x=code_x, code_edge_index=code_edge_index, code_batch=code_batch,
            sensor=sensor,
        )

        core_state, hidden = self.temporal(fused, hidden)

        # Apply goal conditioning (System 3) — FiLM modulation
        if self.goal_conditioner is not None:
            core_state = self.goal_conditioner(core_state, goal)

        comm_logits = self.communication.get_logits(core_state)

        denoising_loss, value = self.policy(core_state, actions=actions)

        # Generate action for distillation / evaluation (no grad noise)
        with torch.no_grad():
            action, _, _ = self.policy.act(core_state, deterministic=True)

        return {
            "fused": fused,
            "core_state": core_state,
            "hidden": hidden,
            "comm_logits": comm_logits,
            "denoising_loss": denoising_loss,
            "value": value,
            "action": action,
        }

    def act(
        self,
        vision: Optional[torch.Tensor] = None,
        audio: Optional[torch.Tensor] = None,
        code_x: Optional[torch.Tensor] = None,
        code_edge_index: Optional[torch.Tensor] = None,
        code_batch: Optional[torch.Tensor] = None,
        sensor: Optional[torch.Tensor] = None,
        hidden: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        goal: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Single inference step for the agent.

        Returns dict with action, log_prob, value, hidden, utterance.
        """
        fused = self.encode(
            vision=vision, audio=audio,
            code_x=code_x, code_edge_index=code_edge_index, code_batch=code_batch,
            sensor=sensor,
        )
        core_state, hidden = self.temporal(fused, hidden)

        # Apply goal conditioning (System 3) — FiLM modulation
        if self.goal_conditioner is not None:
            core_state = self.goal_conditioner(core_state, goal)

        action, log_prob, value = self.policy.act(core_state, deterministic=deterministic)
        utterance = self.communication(core_state)  # autoregressive generation

        return {
            "fused": fused,
            "core_state": core_state,
            "hidden": hidden,
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "utterance": utterance,
        }
