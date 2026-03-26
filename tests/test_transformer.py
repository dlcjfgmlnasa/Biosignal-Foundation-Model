# -*- coding:utf-8 -*-
"""module/transformer.py 테스트: TransformerEncoderLayer, TransformerEncoder."""
from functools import partial

import pytest
import torch

from module.norm import RMSNorm
from module.transformer import TransformerEncoder, TransformerEncoderLayer
from module.attention import GroupedQueryAttention
from module.ffn import FeedForward, GatedLinearUnitFeedForward
from module.position import BinaryAttentionBias, QueryKeyProjection, RotaryProjection


# ── TransformerEncoderLayer ───────────────────────────────────────


class TestTransformerEncoderLayer:
    def _make_layer(self, dim=64, num_heads=4, pre_norm=True):
        attn = GroupedQueryAttention(dim=dim, num_heads=num_heads, num_groups=num_heads)
        ffn = GatedLinearUnitFeedForward(in_dim=dim)
        norm1 = RMSNorm(dim)
        norm2 = RMSNorm(dim)
        return TransformerEncoderLayer(
            self_attn=attn, ffn=ffn, norm1=norm1, norm2=norm2, pre_norm=pre_norm
        )

    def test_pre_norm_forward(self):
        layer = self._make_layer(pre_norm=True)
        layer.eval()
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_post_norm_forward(self):
        layer = self._make_layer(pre_norm=False)
        layer.eval()
        x = torch.randn(2, 16, 64)
        out = layer(x)
        assert out.shape == (2, 16, 64)

    def test_residual_connection(self):
        """Pre-norm에서 residual connection: 출력이 입력과 완전히 같지 않아야 함."""
        layer = self._make_layer(pre_norm=True)
        layer.eval()
        x = torch.randn(1, 8, 64)
        out = layer(x)
        assert not torch.allclose(out, x)

    def test_with_var_id_and_time_id(self):
        """var_id, time_id를 전달해도 정상 동작."""
        layer = self._make_layer()
        layer.eval()
        x = torch.randn(2, 10, 64)
        var_id = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3, 3]] * 2)
        time_id = torch.arange(10).unsqueeze(0).expand(2, -1)
        out = layer(x, var_id=var_id, time_id=time_id)
        assert out.shape == (2, 10, 64)

    def test_with_attn_mask(self):
        layer = self._make_layer()
        layer.eval()
        x = torch.randn(1, 6, 64)
        mask = torch.tril(torch.ones(6, 6, dtype=torch.bool)).unsqueeze(0)
        out = layer(x, attn_mask=mask)
        assert out.shape == (1, 6, 64)


# ── TransformerEncoder ────────────────────────────────────────────


class TestTransformerEncoder:
    def test_basic_forward(self):
        """기본 설정으로 forward pass."""
        encoder = TransformerEncoder(d_model=128, num_layers=2)
        encoder.eval()
        x = torch.randn(2, 32, 128)
        out = encoder(x)
        assert out.shape == (2, 32, 128)

    def test_default_num_heads(self):
        """num_heads 기본값 = d_model // 64."""
        encoder = TransformerEncoder(d_model=256, num_layers=1)
        layer = encoder.layers[0]
        assert layer.self_attn.num_heads == 4  # 256 // 64

    def test_custom_heads_and_groups(self):
        encoder = TransformerEncoder(d_model=128, num_layers=1, num_heads=8, num_groups=2)
        layer = encoder.layers[0]
        assert layer.self_attn.num_heads == 8
        assert layer.self_attn.num_groups == 2

    def test_rmsnorm_default(self):
        """기본 norm_layer가 RMSNorm인지 확인."""
        encoder = TransformerEncoder(d_model=64, num_layers=1)
        assert isinstance(encoder.norm, RMSNorm)
        layer = encoder.layers[0]
        assert isinstance(layer.norm1, RMSNorm)

    def test_use_glu_true(self):
        """use_glu=True면 GatedLinearUnitFeedForward 사용."""
        encoder = TransformerEncoder(d_model=64, num_layers=1, use_glu=True)
        layer = encoder.layers[0]
        assert isinstance(layer.ffn, GatedLinearUnitFeedForward)

    def test_use_glu_false(self):
        """use_glu=False면 FeedForward 사용."""
        encoder = TransformerEncoder(d_model=64, num_layers=1, use_glu=False)
        layer = encoder.layers[0]
        assert isinstance(layer.ffn, FeedForward)
        assert not isinstance(layer.ffn, GatedLinearUnitFeedForward)

    def test_num_layers(self):
        encoder = TransformerEncoder(d_model=64, num_layers=6)
        assert len(encoder.layers) == 6

    def test_with_var_attn_bias(self):
        """BinaryAttentionBias를 var_attn_bias_layer로 전달."""
        encoder = TransformerEncoder(
            d_model=128, num_layers=2,
            var_attn_bias_layer=BinaryAttentionBias,
        )
        encoder.eval()
        x = torch.randn(2, 16, 128)
        var_id = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]] * 2)
        out = encoder(x, var_id=var_id)
        assert out.shape == (2, 16, 128)

    def test_with_time_qk_proj_rope(self):
        """RoPE를 time_qk_proj_layer로 전달."""
        encoder = TransformerEncoder(
            d_model=128, num_layers=2,
            time_qk_proj_layer=partial(
                QueryKeyProjection,
                proj_layer=RotaryProjection,
            ),
        )
        encoder.eval()
        x = torch.randn(1, 20, 128)
        time_id = torch.arange(20).unsqueeze(0)
        out = encoder(x, time_id=time_id)
        assert out.shape == (1, 20, 128)

    def test_shared_layers(self):
        """shared_var_attn_bias=True: 모든 layer가 같은 bias 모듈 공유."""
        encoder = TransformerEncoder(
            d_model=128, num_layers=3,
            var_attn_bias_layer=BinaryAttentionBias,
            shared_var_attn_bias=True,
        )
        bias0 = encoder.layers[0].self_attn.var_attn_bias
        bias1 = encoder.layers[1].self_attn.var_attn_bias
        bias2 = encoder.layers[2].self_attn.var_attn_bias
        assert bias0 is bias1
        assert bias1 is bias2

    def test_unshared_layers(self):
        """shared=False: 각 layer가 독립적인 bias 모듈."""
        encoder = TransformerEncoder(
            d_model=128, num_layers=2,
            var_attn_bias_layer=BinaryAttentionBias,
            shared_var_attn_bias=False,
        )
        bias0 = encoder.layers[0].self_attn.var_attn_bias
        bias1 = encoder.layers[1].self_attn.var_attn_bias
        assert bias0 is not bias1

    def test_moe_forward(self):
        """MoE 모드 forward pass."""
        encoder = TransformerEncoder(
            d_model=64, num_layers=2, use_moe=True
        )
        encoder.eval()
        x = torch.randn(1, 8, 64)
        out = encoder(x)
        assert out.shape == (1, 8, 64)

    def test_gradient_flow(self):
        """전체 Encoder를 통한 gradient flow."""
        encoder = TransformerEncoder(d_model=64, num_layers=2)
        x = torch.randn(1, 8, 64, requires_grad=True)
        out = encoder(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_use_qk_norm_false(self):
        """use_qk_norm=False면 q_norm/k_norm이 Identity."""
        encoder = TransformerEncoder(d_model=64, num_layers=1, use_qk_norm=False)
        layer = encoder.layers[0]
        assert isinstance(layer.self_attn.q_norm, torch.nn.Identity)

    def test_custom_d_ff(self):
        """d_ff로 FFN hidden dim 직접 지정."""
        encoder = TransformerEncoder(d_model=64, num_layers=1, d_ff=256)
        layer = encoder.layers[0]
        assert layer.ffn.hidden_dim == 256
