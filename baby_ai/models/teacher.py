"""
Teacher model — larger agent for continuous background learning (50-100M parameters).

Trains asynchronously on replay data and periodically distills
knowledge into the Student model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baby_ai.config import TeacherConfig, DEFAULT_CONFIG
from baby_ai.models.base import BabyAgentBase


class TeacherModel(BabyAgentBase):
    """
    Large agent model for background training.

    Target: 50-100M parameters, runs asynchronously on GPU.
    Uses wider encoders and deeper core than Student.
    """

    def __init__(self, config: TeacherConfig | None = None):
        if config is None:
            config = DEFAULT_CONFIG.teacher

        super().__init__(
            vision_embed_dim=config.encoder.vision_embed_dim,
            audio_embed_dim=config.encoder.audio_embed_dim,
            code_embed_dim=config.encoder.code_embed_dim,
            sensor_embed_dim=config.encoder.sensor_embed_dim,
            fused_dim=config.encoder.fused_dim,
            hidden_dim=config.hidden_dim,
            policy_hidden=config.policy_hidden,
            action_dim=config.action_dim,
            comm_vocab_size=config.comm_vocab_size,
            comm_max_len=config.comm_max_len,
            # Teacher uses wider encoders — tuned for 50-100M total
            vision_width_mult=2.0,
            audio_width_mult=2.5,
            code_hidden_dim=config.encoder.code_embed_dim * 2,
            code_num_layers=5,
            # Jamba temporal core (Top-2 MoE, 8 experts)
            jamba_config=config.jamba,
            # Diffusion policy (continuous actions, 20 DDIM steps)
            diffusion_config=config.diffusion,
        )

    def get_distillation_targets(
        self,
        **kwargs,
    ) -> dict:
        """
        Generate soft targets for distillation.

        Returns the Teacher's continuous action, communication logits,
        and intermediate fused features for the Student to match.
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(**kwargs)
        return {
            "action": outputs["action"],
            "comm_logits": outputs["comm_logits"],
            "fused_features": outputs["fused"],
            "core_state": outputs["core_state"],
        }
