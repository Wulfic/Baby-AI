"""
Base agent model — shared architecture for Student and Teacher.

Architecture flow::

    Raw modalities (vision, audio, code, sensor)
        ↓
    Per-modality encoders (VisionEncoder, AudioEncoder, CodeEncoder, SensorMLP)
        ↓
    MultimodalFusion (gated attention over available modalities)
        ↓ (B, fused_dim)
    JambaCore (Mamba-2 SSM + MoE temporal backbone)
        ↓ (B, hidden_dim) + recurrent hidden state
    GoalConditioner [optional] (FiLM modulation from System 3 goals)
        ↓
    Output heads:
        ├─ Policy (FlowMatchingPolicyHead or DiffusionPolicyHead) → 23-dim action
        ├─ CommunicationHead → token sequence
        └─ LatentWorldModel (JEPA/RSSM) → curiosity reward + dynamics loss

Scale is controlled by config: Student uses narrow encoders (10–30M params),
Teacher uses wider/deeper encoders (50–100M params).
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
from baby_ai.core.policy import DiffusionPolicyHead, FlowMatchingPolicyHead
from baby_ai.core.communication import CommunicationHead
from baby_ai.core.predictive import LatentWorldModel
from baby_ai.core.goals import GoalConditioner
from baby_ai.core.action_tokenizer import ActionTokenizer
from baby_ai.models.memory import EpisodicMemory
from baby_ai.config import JambaConfig, DiffusionPolicyConfig, FlowMatchingConfig, VQConfig


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
        sensor_channels: int = 32,
        jamba_config: JambaConfig | None = None,
        diffusion_config: DiffusionPolicyConfig | None = None,
        flow_matching_config: FlowMatchingConfig | None = None,
        vq_config: VQConfig | None = None,
        policy_type: str = "flow_matching",
        goal_dim: int = 0,
        use_slot_attention: bool = False,
        num_vision_slots: int = 8,
        slot_dim: int = 64,
        use_episodic_memory: bool = False,
        mem_slots: int = 64,
    ):
        super().__init__()

        # --- Modality encoders ---
        self.vision_encoder = VisionEncoder(
            in_channels=3,
            embed_dim=vision_embed_dim,
            width_mult=vision_width_mult,
            use_slot_attention=use_slot_attention,
            num_slots=num_vision_slots,
            slot_dim=slot_dim,
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
            use_ssd=jamba_config.use_ssd,
            chunk_size=jamba_config.chunk_size,
        )

        # --- Output heads ---
        if diffusion_config is None:
            diffusion_config = DiffusionPolicyConfig()
        if flow_matching_config is None:
            flow_matching_config = FlowMatchingConfig()

        # Policy head: select based on policy_type
        self.policy_type = policy_type
        if policy_type == "flow_matching":
            self.policy = FlowMatchingPolicyHead(
                input_dim=hidden_dim,
                action_dim=flow_matching_config.action_continuous_dim,
                hidden_dim=policy_hidden,
                num_infer_steps=flow_matching_config.num_infer_steps,
                time_embed_dim=flow_matching_config.time_embed_dim,
                sigma_min=flow_matching_config.sigma_min,
            )
        else:
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
        # NOTE: The communication head produces utterance logits but is
        # currently not directly trained by the RL loop (learner_thread).
        # It receives gradients only through distillation KL from Teacher
        # → Student.  For the Teacher, comm_logits have no training
        # signal at all (no ground-truth utterances or reward shaping).
        # This is a known gap — future work could add language grounding.
        self.communication = CommunicationHead(
            input_dim=hidden_dim,
            vocab_size=comm_vocab_size,
            hidden_dim=hidden_dim,
            max_len=comm_max_len,
        )
        # Determine the active action dimension based on policy type
        active_action_dim = (
            flow_matching_config.action_continuous_dim
            if policy_type == "flow_matching"
            else diffusion_config.action_continuous_dim
        )

        self.predictive = LatentWorldModel(
            state_dim=hidden_dim,      # core output dim (world model observes core state)
            action_dim=active_action_dim,
            latent_dim=hidden_dim,
            hidden_dim=hidden_dim,
            stochastic_dim=32,
        )

        # --- Episodic K-V memory (Titans-style) ---
        if use_episodic_memory:
            self.episodic_memory = EpisodicMemory(
                mem_slots=mem_slots,
                key_dim=hidden_dim // 4,
                value_dim=hidden_dim,
                input_dim=hidden_dim,
            )
        else:
            self.episodic_memory = None

        # --- Goal conditioning (System 3) ---
        self.goal_dim = goal_dim
        if goal_dim > 0:
            self.goal_conditioner = GoalConditioner(
                state_dim=hidden_dim,
                goal_dim=goal_dim,
            )
        else:
            self.goal_conditioner = None

        # --- VQ-BeT Action Tokenizer (Phase E) ---
        if vq_config is None:
            vq_config = VQConfig()
        if vq_config.enabled:
            action_dim_for_vq = (
                flow_matching_config.action_continuous_dim
                if policy_type == "flow_matching"
                else diffusion_config.action_continuous_dim
            )
            self.action_tokenizer = ActionTokenizer(
                action_dim=action_dim_for_vq,
                code_dim=vq_config.code_dim,
                num_codes=vq_config.num_codes,
                num_residual=vq_config.num_residual,
                commitment_weight=vq_config.commitment_weight,
                ema_update=vq_config.ema_update,
                ema_decay=vq_config.ema_decay,
            )
        else:
            self.action_tokenizer = None

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
            fused:           (B, fused_dim) multimodal embedding
            core_state:      (B, hidden_dim) temporal core output
            hidden:          Recurrent state for next step
            value:           (B, 1) or scalar state-value estimate
            comm_logits:     (B, vocab_size) communication head logits
            denoising_loss:  Scalar policy training loss (named for
                             backward compat — applies to both diffusion
                             and flow matching policy heads)
            action:          (B, 23) deterministic action vector
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

        # Episodic K-V memory augmentation
        if self.episodic_memory is not None:
            core_state = self.episodic_memory.read(core_state)
            self.episodic_memory.encode_and_write(core_state.detach())

        comm_logits = self.communication.get_logits(core_state)

        policy_loss, value = self.policy(core_state, actions=actions)

        # Generate action for distillation / evaluation (no grad noise)
        with torch.no_grad():
            action, _, _ = self.policy.act(core_state, deterministic=True)

        result = {
            "fused": fused,
            "core_state": core_state,
            "hidden": hidden,
            "comm_logits": comm_logits,
            "denoising_loss": policy_loss,  # keep key name for backward compat
            "value": value,
            "action": action,
        }

        # VQ-BeT tokenization (Phase E)
        if self.action_tokenizer is not None and actions is not None:
            _, vq_indices, vq_loss = self.action_tokenizer.encode(
                actions.squeeze(1) if actions.dim() == 3 and actions.size(1) == 1 else actions
            )
            result["vq_indices"] = vq_indices
            result["vq_loss"] = vq_loss

        return result

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

        # Episodic K-V memory augmentation
        if self.episodic_memory is not None:
            core_state = self.episodic_memory.read(core_state)
            self.episodic_memory.encode_and_write(core_state.detach())

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
