"""
Base agent model — shared architecture for Student and Teacher.

Composes: modality encoders → multimodal fusion → temporal core →
policy head + communication head + predictive head.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from baby_ai.encoders.vision import VisionEncoder
from baby_ai.encoders.audio import AudioEncoder
from baby_ai.encoders.code import CodeEncoder
from baby_ai.encoders.multimodal import MultimodalFusion
from baby_ai.core.temporal import TemporalCore
from baby_ai.core.policy import PolicyHead
from baby_ai.core.communication import CommunicationHead
from baby_ai.core.predictive import PredictiveHead


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
        gru_hidden: GRU hidden state dim.
        gru_layers: Number of GRU layers.
        policy_hidden: Policy MLP hidden dim.
        action_dim: Number of discrete actions.
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
        gru_hidden: int = 256,
        gru_layers: int = 2,
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
        self.temporal = TemporalCore(
            input_dim=fused_dim,
            hidden_dim=gru_hidden,
            num_layers=gru_layers,
        )

        # --- Output heads ---
        self.policy = PolicyHead(
            input_dim=gru_hidden,
            hidden_dim=policy_hidden,
            action_dim=action_dim,
        )
        self.communication = CommunicationHead(
            input_dim=gru_hidden,
            vocab_size=comm_vocab_size,
            hidden_dim=gru_hidden,
            max_len=comm_max_len,
        )
        self.predictive = PredictiveHead(
            state_dim=fused_dim,
            action_dim=action_dim,
            hidden_dim=gru_hidden,
        )

        # Store dims for external access
        self.fused_dim = fused_dim
        self.gru_hidden = gru_hidden
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
    ) -> dict:
        """
        Full forward pass: encode → fuse → temporal → policy + comm.

        Returns dict with:
            fused: (B, fused_dim) fused embedding.
            core_state: (B, gru_hidden) temporal core output.
            hidden: (num_layers, B, gru_hidden) updated GRU hidden.
            action_logits: (B, action_dim) policy logits.
            value: (B, 1) state value.
            comm_logits: (B, vocab_size) first-step communication logits.
        """
        fused = self.encode(
            vision=vision, audio=audio,
            code_x=code_x, code_edge_index=code_edge_index, code_batch=code_batch,
            sensor=sensor,
        )

        core_state, hidden = self.temporal(fused, hidden)
        action_logits, value = self.policy(core_state)
        comm_logits = self.communication.get_logits(core_state)

        return {
            "fused": fused,
            "core_state": core_state,
            "hidden": hidden,
            "action_logits": action_logits,
            "value": value,
            "comm_logits": comm_logits,
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
