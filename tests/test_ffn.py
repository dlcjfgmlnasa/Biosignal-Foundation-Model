# -*- coding:utf-8 -*-
"""module/ffn.py 테스트: FeedForward, GatedLinearUnitFeedForward, MoEFeedForward."""
import pytest
import torch
import torch.nn.functional as F

from module.ffn import FeedForward, GatedLinearUnitFeedForward, MoEFeedForward


# ── FeedForward ───────────────────────────────────────────────────


class TestFeedForward:
    def test_default_dims(self):
        """기본값: hidden_dim=4*in_dim, out_dim=in_dim."""
        ff = FeedForward(in_dim=64)
        assert ff.hidden_dim == 256
        assert ff.out_dim == 64

    def test_custom_dims(self):
        ff = FeedForward(in_dim=64, hidden_dim=128, out_dim=32)
        assert ff.hidden_dim == 128
        assert ff.out_dim == 32

    def test_forward_shape(self):
        ff = FeedForward(in_dim=64)
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == (2, 10, 64)

    def test_forward_custom_out_dim(self):
        ff = FeedForward(in_dim=64, out_dim=32)
        x = torch.randn(2, 10, 64)
        out = ff(x)
        assert out.shape == (2, 10, 32)

    def test_no_centroid_param(self):
        """FeedForward.forward는 centroid 파라미터를 받지 않는다."""
        ff = FeedForward(in_dim=32)
        x = torch.randn(1, 5, 32)
        out = ff(x)
        assert out.shape == (1, 5, 32)

    def test_no_bias(self):
        ff = FeedForward(in_dim=32, bias=False)
        assert ff.fc1.bias is None
        assert ff.fc2.bias is None

    def test_gradient_flow(self):
        ff = FeedForward(in_dim=32)
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = ff(x)
        out.sum().backward()
        assert x.grad is not None


# ── GatedLinearUnitFeedForward ────────────────────────────────────


class TestGLUFeedForward:
    def test_adjust_hidden_dim(self):
        """2/3 * dim, 8의 배수로 반올림."""
        # 256 * 2/3 = 170.666, int = 170, (170+7)//8*8 = 176
        assert GatedLinearUnitFeedForward.adjust_hidden_dim(256) == 176
        # 64 * 2/3 = 42.666, int = 42, (42+7)//8*8 = 48
        assert GatedLinearUnitFeedForward.adjust_hidden_dim(64) == 48

    def test_default_hidden_dim(self):
        """기본 hidden_dim = adjust(4 * in_dim)."""
        glu = GatedLinearUnitFeedForward(in_dim=64)
        expected = GatedLinearUnitFeedForward.adjust_hidden_dim(4 * 64)
        assert glu.hidden_dim == expected

    def test_forward_shape(self):
        glu = GatedLinearUnitFeedForward(in_dim=64)
        x = torch.randn(2, 10, 64)
        out = glu(x)
        assert out.shape == (2, 10, 64)

    def test_has_gate_linear(self):
        """fc_gate가 존재하고 올바른 shape."""
        glu = GatedLinearUnitFeedForward(in_dim=64)
        assert hasattr(glu, "fc_gate")
        assert glu.fc_gate.in_features == 64
        assert glu.fc_gate.out_features == glu.hidden_dim

    def test_silu_activation_default(self):
        """기본 activation은 F.silu."""
        glu = GatedLinearUnitFeedForward(in_dim=32)
        assert glu.activation is F.silu

    def test_3d_4d_input(self):
        """3D, 4D 입력 모두 처리 가능."""
        glu = GatedLinearUnitFeedForward(in_dim=32)
        x_3d = torch.randn(2, 8, 32)
        x_4d = torch.randn(2, 4, 8, 32)
        assert glu(x_3d).shape == (2, 8, 32)
        assert glu(x_4d).shape == (2, 4, 8, 32)


# ── MoEFeedForward ────────────────────────────────────────────────


class TestMoEFeedForward:
    def test_num_experts(self):
        moe = MoEFeedForward(num_experts=4, num_experts_per_token=2, in_dim=32)
        assert len(moe.experts) == 4
        assert moe.num_experts_per_token == 2

    def test_gate_weight_shape(self):
        """gate는 (num_experts, in_dim) shape의 Linear."""
        moe = MoEFeedForward(num_experts=4, num_experts_per_token=2, in_dim=32)
        assert moe.gate.weight.shape == (4, 32)
        assert moe.gate.bias is None

    def test_forward_shape(self):
        moe = MoEFeedForward(num_experts=4, num_experts_per_token=2, in_dim=32)
        x = torch.randn(2, 8, 32)
        out = moe(x)
        assert out.shape == (2, 8, 32)

    def test_forward_4d(self):
        """4D 입력도 처리 가능."""
        moe = MoEFeedForward(num_experts=4, num_experts_per_token=1, in_dim=16)
        x = torch.randn(2, 3, 8, 16)
        out = moe(x)
        assert out.shape == (2, 3, 8, 16)

    def test_experts_are_glu(self):
        """각 expert는 GatedLinearUnitFeedForward 인스턴스."""
        moe = MoEFeedForward(num_experts=3, num_experts_per_token=1, in_dim=32)
        for expert in moe.experts:
            assert isinstance(expert, GatedLinearUnitFeedForward)

    def test_gradient_flow(self):
        moe = MoEFeedForward(num_experts=4, num_experts_per_token=2, in_dim=32)
        moe.train()
        x = torch.randn(1, 4, 32, requires_grad=True)
        out = moe(x)
        out.sum().backward()
        assert x.grad is not None

    def test_output_finite(self):
        """출력에 NaN이나 Inf가 없는지 확인."""
        moe = MoEFeedForward(num_experts=4, num_experts_per_token=2, in_dim=32)
        x = torch.randn(2, 8, 32)
        out = moe(x)
        assert torch.isfinite(out).all()

    def test_aux_loss_training(self):
        """Training mode에서 aux_loss가 생성되고 finite한 양수."""
        moe = MoEFeedForward(num_experts=4, num_experts_per_token=2, in_dim=32)
        moe.train()
        x = torch.randn(2, 8, 32)
        moe(x)
        assert moe.aux_loss is not None
        assert torch.isfinite(moe.aux_loss)
        assert moe.aux_loss.item() >= 0

    def test_aux_loss_eval(self):
        """Eval mode에서는 aux_loss가 None."""
        moe = MoEFeedForward(num_experts=4, num_experts_per_token=2, in_dim=32)
        moe.eval()
        x = torch.randn(2, 8, 32)
        moe(x)
        assert moe.aux_loss is None
