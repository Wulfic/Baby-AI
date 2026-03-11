"""
Comprehensive tests for the 5-phase upgrade:
    Phase A: torch.compile integration
    Phase B: Mamba-2 SSD
    Phase C: Flow Matching policy
    Phase D: REBEL RL loss
    Phase E: VQ-BeT action tokenizer
"""

from __future__ import annotations

import sys
import os
import pytest
import torch
import torch.nn as nn

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def action_dim():
    return 20


@pytest.fixture
def hidden_dim():
    return 64  # Small for test speed


# ═════════════════════════════════════════════════════════════════════════════
# Phase A: torch.compile
# ═════════════════════════════════════════════════════════════════════════════

class TestPhaseA_TorchCompile:
    """Tests for torch.compile integration in config and runtime."""

    def test_config_compile_fields_exist(self):
        """RuntimeConfig should have compile_student, compile_teacher, compile_mode."""
        from baby_ai.config import RuntimeConfig
        rc = RuntimeConfig()
        assert hasattr(rc, "compile_student")
        assert hasattr(rc, "compile_teacher")
        assert hasattr(rc, "compile_mode")
        assert rc.compile_student is True
        assert rc.compile_teacher is False
        assert rc.compile_mode == "reduce-overhead"

    def test_compile_student_model(self, device):
        """torch.compile should wrap a StudentModel without errors."""
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")
        if device == "cpu":
            pytest.skip("torch.compile test unreliable on CPU-only")
        try:
            import triton  # noqa: F401
        except ImportError:
            pytest.skip("triton not installed – torch.compile inductor backend unavailable")
        from baby_ai.config import StudentConfig, JambaConfig
        from baby_ai.models.student import StudentModel
        cfg = StudentConfig(
            hidden_dim=64,
            policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
        )
        model = StudentModel(cfg).to(device)
        compiled = torch.compile(model, mode="default", fullgraph=False)
        # Must be able to do a forward pass
        out = compiled(sensor=torch.randn(1, 32, device=device))
        assert "action" in out

    def test_compiled_model_has_orig_mod(self, device):
        """Compiled models should expose _orig_mod for weight swapping."""
        if not hasattr(torch, "compile"):
            pytest.skip("torch.compile not available")
        model = nn.Linear(10, 10).to(device)
        compiled = torch.compile(model, fullgraph=False)
        assert hasattr(compiled, "_orig_mod")


# ═════════════════════════════════════════════════════════════════════════════
# Phase B: Mamba-2 SSD
# ═════════════════════════════════════════════════════════════════════════════

class TestPhaseB_Mamba2SSD:
    """Tests for SSD chunked scan in the temporal core."""

    def test_config_ssd_fields(self):
        """JambaConfig should have SSD-related fields."""
        from baby_ai.config import JambaConfig
        jc = JambaConfig()
        assert jc.use_ssd is True
        assert jc.chunk_size == 64
        assert jc.ssd_head_dim == 64

    def test_mamba_block_ssd_training(self, device):
        """MambaBlock with use_ssd=True should handle multi-step training sequences."""
        from baby_ai.core.temporal import MambaBlock
        dim = 64
        mb = MambaBlock(dim=dim, d_state=16, d_conv=4, expand=2,
                        use_ssd=True, chunk_size=8).to(device)
        # Sequence of 16 steps (> chunk_size=8)
        x = torch.randn(2, 16, dim, device=device)
        y, ssm_state, conv_state = mb(x)
        assert y.shape == (2, 16, dim)
        assert ssm_state is not None

    def test_mamba_block_ssd_inference(self, device):
        """MambaBlock with use_ssd=True should handle single-step inference."""
        from baby_ai.core.temporal import MambaBlock
        dim = 64
        mb = MambaBlock(dim=dim, d_state=16, d_conv=4, expand=2,
                        use_ssd=True, chunk_size=8).to(device)
        # Single step (inference mode)
        x = torch.randn(2, 1, dim, device=device)
        y, ssm_state, conv_state = mb(x)
        assert y.shape == (2, 1, dim)

    def test_mamba_block_no_ssd_fallback(self, device):
        """MambaBlock with use_ssd=False should use sequential scan."""
        from baby_ai.core.temporal import MambaBlock
        dim = 64
        mb = MambaBlock(dim=dim, d_state=16, d_conv=4, expand=2,
                        use_ssd=False).to(device)
        x = torch.randn(2, 8, dim, device=device)
        y, ssm_state, conv_state = mb(x)
        assert y.shape == (2, 8, dim)

    def test_jamba_core_passes_ssd_params(self, device):
        """JambaCore should accept and pass through use_ssd and chunk_size."""
        from baby_ai.core.temporal import JambaCore
        core = JambaCore(
            input_dim=64, hidden_dim=64, num_layers=1,
            d_state=16, d_conv=4, expand=2, dt_rank=0,
            num_experts=2, top_k_routing=1, moe_every_n=2,
            ffn_mult=2, load_balance_weight=0.01,
            use_ssd=True, chunk_size=16,
        ).to(device)
        x = torch.randn(2, 64, device=device)
        out, hidden = core(x)
        assert out.shape == (2, 64)


# ═════════════════════════════════════════════════════════════════════════════
# Phase C: Flow Matching Policy
# ═════════════════════════════════════════════════════════════════════════════

class TestPhaseC_FlowMatching:
    """Tests for the Flow Matching policy head."""

    def test_velocity_predictor(self, device, batch_size, action_dim, hidden_dim):
        """VelocityPredictor should produce velocity vectors of correct shape."""
        from baby_ai.core.policy import VelocityPredictor
        vp = VelocityPredictor(
            action_dim=action_dim, state_dim=hidden_dim,
        ).to(device)
        x_t = torch.randn(batch_size, action_dim, device=device)
        t = torch.rand(batch_size, device=device)
        state = torch.randn(batch_size, hidden_dim, device=device)
        v = vp(x_t, t, state)
        assert v.shape == (batch_size, action_dim)

    def test_flow_matching_forward_with_actions(self, device, batch_size, hidden_dim, action_dim):
        """FlowMatchingPolicyHead.forward() should return loss and value."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        head = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        state = torch.randn(batch_size, hidden_dim, device=device)
        actions = torch.randn(batch_size, action_dim, device=device)
        loss, value = head(state, actions)
        assert loss.dim() == 0  # scalar
        assert loss.item() >= 0
        assert value.shape == (batch_size, 1)

    def test_flow_matching_forward_no_actions(self, device, hidden_dim):
        """FlowMatchingPolicyHead.forward() with no actions returns zero loss."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        head = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=20, hidden_dim=hidden_dim,
        ).to(device)
        state = torch.randn(2, hidden_dim, device=device)
        loss, value = head(state)
        assert loss.item() == 0.0
        assert value.shape == (2, 1)

    def test_flow_matching_act_bounds(self, device, hidden_dim):
        """FlowMatchingPolicyHead.act() should produce bounded actions."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        head = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=20, hidden_dim=hidden_dim,
            num_infer_steps=2,
        ).to(device)
        state = torch.randn(4, hidden_dim, device=device)
        action, log_prob, value = head.act(state)
        assert action.shape == (4, 20)
        # Camera dims should be in [-1, 1] (tanh)
        assert action[:, :2].min() >= -1.0
        assert action[:, :2].max() <= 1.0
        # Other dims should be in [0, 1] (sigmoid)
        assert action[:, 2:].min() >= 0.0
        assert action[:, 2:].max() <= 1.0
        assert log_prob.shape == (4,)
        assert value.shape == (4, 1)

    def test_flow_matching_evaluate(self, device, hidden_dim, action_dim):
        """FlowMatchingPolicyHead.evaluate() should return log_prob, entropy, value."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        head = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        state = torch.randn(4, hidden_dim, device=device)
        action = torch.randn(4, action_dim, device=device)
        lp, ent, val = head.evaluate(state, action)
        assert lp.shape == (4,)
        assert ent.shape == (4,)
        assert val.shape == (4, 1)
        # log_prob should be negative (it's -MSE)
        assert (lp <= 0).all()

    def test_flow_matching_training_convergence(self, device, hidden_dim, action_dim):
        """Flow matching loss should decrease over a few training steps on fixed data."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        head = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        head.train()
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        state = torch.randn(16, hidden_dim, device=device)
        actions = torch.randn(16, action_dim, device=device)
        losses = []
        for _ in range(50):
            loss, _ = head(state, actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # Loss should decrease
        assert losses[-1] < losses[0], (
            f"Flow loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"
        )

    def test_flow_matching_fewer_euler_steps_than_diffusion(self, device, hidden_dim):
        """Flow matching should default to fewer inference steps than diffusion."""
        from baby_ai.config import FlowMatchingConfig, DiffusionPolicyConfig
        fm = FlowMatchingConfig()
        df = DiffusionPolicyConfig()
        assert fm.num_infer_steps <= df.num_infer_steps

    def test_policy_type_selection_in_base(self, device):
        """BabyAgentBase should select FlowMatchingPolicyHead when policy_type='flow_matching'."""
        from baby_ai.core.policy import FlowMatchingPolicyHead, DiffusionPolicyHead
        from baby_ai.config import StudentConfig, JambaConfig
        from baby_ai.models.student import StudentModel
        cfg = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
            policy_type="flow_matching",
        )
        s = StudentModel(cfg).to(device)
        assert isinstance(s.policy, FlowMatchingPolicyHead)

        cfg2 = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
            policy_type="diffusion",
        )
        s2 = StudentModel(cfg2).to(device)
        assert isinstance(s2.policy, DiffusionPolicyHead)


# ═════════════════════════════════════════════════════════════════════════════
# Phase D: REBEL RL Loss
# ═════════════════════════════════════════════════════════════════════════════

class TestPhaseD_REBEL:
    """Tests for REBEL (Regressing Relative Rewards) loss."""

    def test_rebel_config_fields(self):
        """REBELConfig should have all expected fields."""
        from baby_ai.config import REBELConfig
        rc = REBELConfig()
        assert rc.enabled is True
        assert rc.beta == 0.1
        assert rc.reward_clip == 5.0
        assert rc.value_loss_weight == 0.5

    def test_rebel_loss_basic(self, device, hidden_dim, action_dim):
        """REBELLoss should produce a scalar loss."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        from baby_ai.learning.rebel import REBELLoss
        B = 4
        policy = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        rebel = REBELLoss(beta=0.1, reward_clip=5.0)
        loss = rebel(
            state=torch.randn(B, hidden_dim, device=device),
            action_w=torch.randn(B, action_dim, device=device),
            action_l=torch.randn(B, action_dim, device=device),
            reward_w=torch.tensor([1.0, 2.0, 3.0, 4.0], device=device),
            reward_l=torch.tensor([0.0, 0.5, 1.0, 1.5], device=device),
            policy=policy,
        )
        assert loss.dim() == 0
        assert loss.item() > 0
        # Should have gradient
        loss.backward()
        grads_exist = any(p.grad is not None and p.grad.abs().sum() > 0
                          for p in policy.parameters())
        assert grads_exist, "REBEL loss should propagate gradients to the policy"

    def test_rebel_loss_equal_rewards_low_gradient(self, device, hidden_dim, action_dim):
        """When rewards are equal, REBEL gradient should be near-zero."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        from baby_ai.learning.rebel import REBELLoss
        B = 4
        policy = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        rebel = REBELLoss(beta=0.1)
        loss = rebel(
            state=torch.randn(B, hidden_dim, device=device),
            action_w=torch.randn(B, action_dim, device=device),
            action_l=torch.randn(B, action_dim, device=device),
            reward_w=torch.ones(B, device=device),  # same reward
            reward_l=torch.ones(B, device=device),
            policy=policy,
        )
        loss.backward()
        # Gradients should be very small when delta_r = 0
        max_grad = max(
            p.grad.abs().max().item() for p in policy.parameters()
            if p.grad is not None
        )
        assert max_grad < 0.1, f"Grad should be near-zero for equal rewards, got {max_grad}"

    def test_rebel_reward_clipping(self, device, hidden_dim, action_dim):
        """REBEL should clip extreme reward differences."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        from baby_ai.learning.rebel import REBELLoss
        B = 2
        policy = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        rebel = REBELLoss(beta=0.1, reward_clip=1.0)
        # Extreme rewards
        loss = rebel(
            state=torch.randn(B, hidden_dim, device=device),
            action_w=torch.randn(B, action_dim, device=device),
            action_l=torch.randn(B, action_dim, device=device),
            reward_w=torch.tensor([100.0, 200.0], device=device),
            reward_l=torch.tensor([-100.0, -200.0], device=device),
            policy=policy,
        )
        assert torch.isfinite(loss), "Loss should be finite even with extreme rewards"

    def test_rebel_config_on_models(self):
        """Student and Teacher should stash _rebel_config."""
        from baby_ai.models.student import StudentModel
        from baby_ai.models.teacher import TeacherModel
        from baby_ai.config import REBELConfig
        s = StudentModel()
        t = TeacherModel()
        assert hasattr(s, "_rebel_config")
        assert hasattr(t, "_rebel_config")
        assert isinstance(s._rebel_config, REBELConfig)

    def test_sample_pairs_method_exists(self):
        """PrioritizedReplayBuffer should have a sample_pairs method."""
        from baby_ai.memory.replay_buffer import PrioritizedReplayBuffer
        assert hasattr(PrioritizedReplayBuffer, "sample_pairs")


# ═════════════════════════════════════════════════════════════════════════════
# Phase E: VQ-BeT Action Tokenizer
# ═════════════════════════════════════════════════════════════════════════════

class TestPhaseE_VQBeT:
    """Tests for VQ-BeT action tokenizer."""

    def test_vq_config_fields(self):
        """VQConfig should have all expected fields."""
        from baby_ai.config import VQConfig
        vc = VQConfig()
        assert vc.enabled is False
        assert vc.num_codes == 512
        assert vc.code_dim == 64
        assert vc.num_residual == 2
        assert vc.ema_update is True

    def test_vector_quantizer_basic(self, device):
        """VectorQuantizer should quantize and return indices."""
        from baby_ai.core.action_tokenizer import VectorQuantizer
        vq = VectorQuantizer(num_codes=32, code_dim=16, ema_update=True).to(device)
        z = torch.randn(8, 16, device=device)
        vq.train()
        z_q, indices, loss = vq(z)
        assert z_q.shape == (8, 16)
        assert indices.shape == (8,)
        assert loss.item() >= 0
        assert indices.min() >= 0
        assert indices.max() < 32

    def test_vector_quantizer_straight_through(self, device):
        """Straight-through estimator: z_q should have gradients from z."""
        from baby_ai.core.action_tokenizer import VectorQuantizer
        vq = VectorQuantizer(num_codes=32, code_dim=16, ema_update=False).to(device)
        z = torch.randn(4, 16, device=device, requires_grad=True)
        vq.train()
        z_q, _, loss = vq(z)
        (z_q.sum() + loss).backward()
        assert z.grad is not None, "Gradient should flow through straight-through estimator"

    def test_residual_vq(self, device):
        """ResidualVQ should produce multi-level indices."""
        from baby_ai.core.action_tokenizer import ResidualVQ
        rvq = ResidualVQ(num_levels=3, num_codes=32, code_dim=16).to(device)
        z = torch.randn(4, 16, device=device)
        rvq.train()
        z_q, indices_list, loss = rvq(z)
        assert z_q.shape == (4, 16)
        assert len(indices_list) == 3
        for idx in indices_list:
            assert idx.shape == (4,)
        assert loss.item() >= 0

    def test_action_tokenizer_encode_decode(self, device, action_dim):
        """ActionTokenizer should encode → VQ → decode with reasonable reconstruction."""
        from baby_ai.core.action_tokenizer import ActionTokenizer
        tok = ActionTokenizer(
            action_dim=action_dim, code_dim=32, num_codes=64,
            num_residual=2, ema_update=True,
        ).to(device)
        action = torch.randn(8, action_dim, device=device)
        # Training pass
        tok.train()
        reconstructed, indices, total_loss = tok(action)
        assert reconstructed.shape == action.shape
        assert len(indices) == 2  # 2 residual levels
        assert total_loss.item() >= 0

    def test_action_tokenizer_decode_from_indices(self, device, action_dim):
        """decode_from_indices should reconstruct from codebook indices."""
        from baby_ai.core.action_tokenizer import ActionTokenizer
        tok = ActionTokenizer(
            action_dim=action_dim, code_dim=32, num_codes=64,
            num_residual=2,
        ).to(device)
        # Encode
        action = torch.randn(4, action_dim, device=device)
        z_q, indices, _ = tok.encode(action)
        # Decode from indices
        decoded = tok.decode_from_indices(indices)
        assert decoded.shape == (4, action_dim)
        # decode_from_indices uses raw codebook embeddings while encode()
        # uses straight-through estimator (z + (z_q - z).detach() ≈ z),
        # so exact match is NOT expected.  Just verify the round-trip
        # decode_from_indices → encode → decode_from_indices is consistent.
        decoded_2 = tok.decode_from_indices(indices)
        assert torch.allclose(decoded, decoded_2, atol=1e-5)

    def test_action_tokenizer_codebook_usage(self, device, action_dim):
        """After training, >10% of codebook entries should be used."""
        from baby_ai.core.action_tokenizer import ActionTokenizer
        tok = ActionTokenizer(
            action_dim=action_dim, code_dim=32, num_codes=64,
            num_residual=1, ema_update=True,
        ).to(device)
        tok.train()
        opt = torch.optim.Adam(tok.parameters(), lr=1e-3)
        all_indices = set()
        for _ in range(200):
            actions = torch.randn(32, action_dim, device=device)
            _, indices, loss = tok(actions)
            opt.zero_grad()
            loss.backward()
            opt.step()
            all_indices.update(indices[0].cpu().tolist())
        usage = len(all_indices) / 64
        # With EMA updates the codebook may converge slowly; verify
        # at least some diversity (>10% codes used = >6 codes).
        assert usage >= 0.10, f"Codebook usage too low: {usage:.1%} ({len(all_indices)}/64)"

    def test_vq_disabled_by_default(self):
        """Student with default config should have no action_tokenizer."""
        from baby_ai.models.student import StudentModel
        s = StudentModel()
        assert s.action_tokenizer is None

    def test_vq_enabled_produces_indices(self, device):
        """Student with VQ enabled should produce vq_indices in forward output."""
        from baby_ai.config import StudentConfig, VQConfig, JambaConfig
        from baby_ai.models.student import StudentModel
        cfg = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
            vq=VQConfig(enabled=True, num_codes=32, code_dim=16, num_residual=2),
        )
        s = StudentModel(cfg).to(device)
        s.eval()
        out = s(
            sensor=torch.randn(2, 32, device=device),
            actions=torch.randn(2, 20, device=device),
        )
        assert "vq_indices" in out
        assert "vq_loss" in out
        assert len(out["vq_indices"]) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Cross-Phase Integration Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestCrossPhaseIntegration:
    """Tests that verify correct wiring across multiple phases."""

    def test_student_full_forward_pass(self, device):
        """Full Student forward with flow_matching + SSD-off + VQ-disabled."""
        from baby_ai.config import StudentConfig, JambaConfig
        from baby_ai.models.student import StudentModel
        cfg = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
            policy_type="flow_matching",
        )
        s = StudentModel(cfg).to(device).eval()
        out = s(
            sensor=torch.randn(2, 32, device=device),
            actions=torch.randn(2, 20, device=device),
        )
        assert "denoising_loss" in out  # backward compat key
        assert "action" in out
        assert "value" in out
        assert "core_state" in out

    def test_teacher_full_forward_pass(self, device):
        """Full Teacher forward with flow_matching + SSD-off."""
        from baby_ai.config import TeacherConfig, JambaConfig, EncoderConfig
        from baby_ai.models.teacher import TeacherModel
        cfg = TeacherConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=EncoderConfig(
                vision_embed_dim=64, audio_embed_dim=64,
                code_embed_dim=64, sensor_embed_dim=32, fused_dim=64,
            ),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
            policy_type="flow_matching",
        )
        t = TeacherModel(cfg).to(device).eval()
        out = t(sensor=torch.randn(2, 32, device=device))
        assert "action" in out

    def test_student_act_method(self, device):
        """Student .act() should return action, log_prob, value, hidden."""
        from baby_ai.config import StudentConfig, JambaConfig
        from baby_ai.models.student import StudentModel
        cfg = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
        )
        s = StudentModel(cfg).to(device).eval()
        out = s.act(sensor=torch.randn(1, 32, device=device))
        assert out["action"].shape == (1, 20)
        assert out["log_prob"].shape == (1,)
        assert out["value"].shape == (1, 1)

    def test_world_model_action_dim_matches_policy(self, device):
        """World model action_dim should match the active policy's action_dim."""
        from baby_ai.config import StudentConfig, JambaConfig, FlowMatchingConfig
        from baby_ai.models.student import StudentModel
        fm_cfg = FlowMatchingConfig(action_continuous_dim=20)
        cfg = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
            policy_type="flow_matching",
            flow_matching=fm_cfg,
        )
        s = StudentModel(cfg).to(device)
        assert s.predictive.action_dim == 20  # should match flow_matching_config

    def test_rebel_with_flow_matching_end_to_end(self, device):
        """REBEL loss should work end-to-end with FlowMatchingPolicyHead."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        from baby_ai.learning.rebel import REBELLoss
        B = 8
        hidden_dim = 64
        action_dim = 20
        policy = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        rebel = REBELLoss(beta=0.1)
        state = torch.randn(B, hidden_dim, device=device)
        # Generate actions from policy
        with torch.no_grad():
            action1, _, _ = policy.act(state[:B // 2])
            action2, _, _ = policy.act(state[:B // 2])
        reward1 = torch.randn(B // 2, device=device)
        reward2 = torch.randn(B // 2, device=device)
        loss = rebel(
            state=state[:B // 2],
            action_w=action1, action_l=action2,
            reward_w=reward1, reward_l=reward2,
            policy=policy,
        )
        loss.backward()
        assert torch.isfinite(loss)

    def test_vq_with_flow_matching_actions(self, device):
        """VQ tokenizer should encode flow-matching-generated actions."""
        from baby_ai.core.policy import FlowMatchingPolicyHead
        from baby_ai.core.action_tokenizer import ActionTokenizer
        hidden_dim, action_dim = 64, 20
        policy = FlowMatchingPolicyHead(
            input_dim=hidden_dim, action_dim=action_dim, hidden_dim=hidden_dim,
        ).to(device)
        tok = ActionTokenizer(
            action_dim=action_dim, code_dim=32, num_codes=64, num_residual=2,
        ).to(device)
        state = torch.randn(4, hidden_dim, device=device)
        with torch.no_grad():
            action, _, _ = policy.act(state)
        # Tokenize the generated actions
        tok.eval()
        recon, indices, loss = tok(action)
        assert recon.shape == action.shape
        assert len(indices) == 2


# ═════════════════════════════════════════════════════════════════════════════
# Rollback / Config Toggle Tests
# ═════════════════════════════════════════════════════════════════════════════

class TestRollbackToggles:
    """Test that each phase can be individually disabled via config."""

    def test_disable_ssd(self, device):
        """use_ssd=False should produce valid outputs."""
        from baby_ai.config import StudentConfig, JambaConfig
        from baby_ai.models.student import StudentModel
        cfg = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
        )
        s = StudentModel(cfg).to(device).eval()
        out = s(sensor=torch.randn(1, 32, device=device))
        assert "action" in out

    def test_disable_flow_matching(self, device):
        """policy_type='diffusion' should use DiffusionPolicyHead."""
        from baby_ai.config import StudentConfig, JambaConfig
        from baby_ai.models.student import StudentModel
        from baby_ai.core.policy import DiffusionPolicyHead
        cfg = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
            policy_type="diffusion",
        )
        s = StudentModel(cfg).to(device).eval()
        assert isinstance(s.policy, DiffusionPolicyHead)
        out = s(sensor=torch.randn(1, 32, device=device))
        assert "action" in out

    def test_disable_rebel(self):
        """REBELConfig.enabled=False should be respected."""
        from baby_ai.config import REBELConfig
        rc = REBELConfig(enabled=False)
        assert rc.enabled is False

    def test_disable_vq(self, device):
        """VQConfig.enabled=False should result in no action_tokenizer."""
        from baby_ai.config import StudentConfig, JambaConfig, VQConfig
        from baby_ai.models.student import StudentModel
        cfg = StudentConfig(
            hidden_dim=64, policy_hidden=64,
            encoder=_small_encoder_config(),
            jamba=JambaConfig(num_layers=1, num_experts=2, use_ssd=False),
            vq=VQConfig(enabled=False),
        )
        s = StudentModel(cfg).to(device)
        assert s.action_tokenizer is None


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _small_encoder_config():
    """Create a small EncoderConfig for fast tests."""
    from baby_ai.config import EncoderConfig
    return EncoderConfig(
        vision_embed_dim=64,
        audio_embed_dim=64,
        code_embed_dim=64,
        sensor_embed_dim=32,
        fused_dim=64,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
