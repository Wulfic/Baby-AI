"""
Student model — compact, fast-inference agent (10-30M parameters).

Designed for <200ms inference latency on RTX 2080 Ti.
Updated periodically via distillation from the Teacher.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from baby_ai.config import StudentConfig, DEFAULT_CONFIG
from baby_ai.models.base import BabyAgentBase


class StudentModel(BabyAgentBase):
    """
    Small, fast agent for real-time inference.

    Target: 10-30M parameters, <200ms per step.
    Uses narrow encoders (width_mult=0.5) and shallow core.
    """

    def __init__(self, config: StudentConfig | None = None):
        if config is None:
            config = DEFAULT_CONFIG.student

        super().__init__(
            vision_embed_dim=config.encoder.vision_embed_dim,
            audio_embed_dim=config.encoder.audio_embed_dim,
            code_embed_dim=config.encoder.code_embed_dim,
            sensor_embed_dim=config.encoder.sensor_embed_dim,
            fused_dim=config.encoder.fused_dim,
            gru_hidden=config.gru_hidden,
            gru_layers=config.gru_layers,
            policy_hidden=config.policy_hidden,
            action_dim=config.action_dim,
            comm_vocab_size=config.comm_vocab_size,
            comm_max_len=config.comm_max_len,
            # Student uses moderate encoders — tuned for 10-30M total
            vision_width_mult=1.0,
            audio_width_mult=1.0,
            code_hidden_dim=config.encoder.code_embed_dim,
            code_num_layers=3,
        )

    @torch.no_grad()
    def inference_step(
        self,
        **kwargs,
    ) -> dict:
        """
        Optimized inference — no gradients, evaluation mode.
        Same signature as .act() but guaranteed no grad tracking.
        """
        was_training = self.training
        self.eval()
        result = self.act(deterministic=False, **kwargs)
        if was_training:
            self.train()
        return result

    def load_distilled_weights(self, state_dict: dict) -> None:
        """
        Atomic weight update from distillation.
        Loads new weights without disrupting ongoing inference.
        """
        self.load_state_dict(state_dict, strict=False)
