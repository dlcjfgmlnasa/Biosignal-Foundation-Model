# -*- coding:utf-8 -*-
"""module/attention.py 테스트: GQA, MHA, MQA, native_sdpa."""
import math
from functools import partial

import pytest
import torch

from module.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
    native_scaled_dot_product_attention,
)
from module.position import BinaryAttentionBias, QueryKeyProjection, RotaryProjection


# ── native_scaled_dot_product_attention ───────────────────────────


class TestNativeSDPA:
    def test_output_shape(self):
        q = torch.randn(2, 4, 1, 8, 16)
        k = torch.randn(2, 4, 1, 8, 16)
        v = torch.randn(2, 4, 1, 8, 16)
        out = native_scaled_dot_product_attention(q, k, v)
        assert out.shape == (2, 4, 1, 8, 16)

    def test_with_bool_mask(self):
        """Bool mask로 특정 위치를 마스킹."""
        q = torch.randn(1, 1, 1, 3, 8)
        k = torch.randn(1, 1, 1, 3, 8)
        v = torch.randn(1, 1, 1, 3, 8)
        # causal mask: 상삼각 차단
        mask = torch.tril(torch.ones(3, 3, dtype=torch.bool)).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        out = native_scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert out.shape == (1, 1, 1, 3, 8)

    def test_with_float_mask(self):
        """Float bias mask 적용."""
        q = torch.randn(1, 2, 1, 4, 8)
        k = torch.randn(1, 2, 1, 4, 8)
        v = torch.randn(1, 2, 1, 4, 8)
        bias = torch.zeros(1, 1, 1, 4, 4)
        out = native_scaled_dot_product_attention(q, k, v, attn_mask=bias)
        assert out.shape == (1, 2, 1, 4, 8)

    def test_identity_attention(self):
        """Q=K일 때 self-attention이 잘 작동하는지 확인."""
        x = torch.randn(1, 1, 1, 4, 8)
        out = native_scaled_dot_product_attention(x, x, x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


# ── GroupedQueryAttention ─────────────────────────────────────────


class TestGroupedQueryAttention:
    def test_basic_forward(self):
        """기본 GQA forward pass shape 확인."""
        dim, num_heads, num_groups = 64, 4, 2
        gqa = GroupedQueryAttention(dim=dim, num_heads=num_heads, num_groups=num_groups)
        gqa.eval()

        x = torch.randn(2, 16, dim)
        out = gqa(x, x, x)
        assert out.shape == (2, 16, dim)

    def test_different_qkv_lengths(self):
        """Query와 Key/Value의 sequence length가 다른 경우."""
        dim, num_heads, num_groups = 64, 4, 2
        gqa = GroupedQueryAttention(dim=dim, num_heads=num_heads, num_groups=num_groups)
        gqa.eval()

        q = torch.randn(1, 8, dim)
        kv = torch.randn(1, 20, dim)
        out = gqa(q, kv, kv)
        assert out.shape == (1, 8, dim)

    def test_with_attn_mask(self):
        """Attention mask 적용."""
        dim, num_heads = 64, 4
        gqa = GroupedQueryAttention(dim=dim, num_heads=num_heads, num_groups=num_heads)
        gqa.eval()

        x = torch.randn(2, 10, dim)
        mask = torch.tril(torch.ones(10, 10, dtype=torch.bool)).unsqueeze(0).expand(2, -1, -1)
        out = gqa(x, x, x, attn_mask=mask)
        assert out.shape == (2, 10, dim)

    def test_with_var_attn_bias(self):
        """BinaryAttentionBias를 var_attn_bias로 사용."""
        dim, num_heads, num_groups = 64, 4, 2
        gqa = GroupedQueryAttention(
            dim=dim, num_heads=num_heads, num_groups=num_groups,
            var_attn_bias=partial(BinaryAttentionBias, dim=dim, num_heads=num_heads, num_groups=num_groups),
        )
        gqa.eval()

        x = torch.randn(2, 12, dim)
        var_id = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]] * 2)
        out = gqa(x, x, x, query_var_id=var_id, kv_var_id=var_id)
        assert out.shape == (2, 12, dim)

    def test_with_time_qk_proj(self):
        """RotaryProjection을 time_qk_proj로 사용."""
        dim, num_heads, num_groups = 64, 4, 2
        gqa = GroupedQueryAttention(
            dim=dim, num_heads=num_heads, num_groups=num_groups,
            time_qk_proj=partial(
                QueryKeyProjection,
                dim=dim, num_heads=num_heads, num_groups=num_groups,
                proj_layer=RotaryProjection,
            ),
        )
        gqa.eval()

        x = torch.randn(2, 16, dim)
        time_id = torch.arange(16).unsqueeze(0).expand(2, -1)
        out = gqa(x, x, x, query_time_id=time_id, kv_time_id=time_id)
        assert out.shape == (2, 16, dim)

    def test_no_bias_linear(self):
        """bias=False로 설정 시 Linear에 bias가 없는지 확인."""
        gqa = GroupedQueryAttention(dim=32, num_heads=2, num_groups=2, bias=False)
        assert gqa.q_proj.bias is None
        assert gqa.k_proj.bias is None
        assert gqa.v_proj.bias is None
        assert gqa.out_proj.bias is None

    def test_no_norm(self):
        """norm_layer=None이면 q_norm/k_norm이 Identity."""
        gqa = GroupedQueryAttention(dim=32, num_heads=2, num_groups=2, norm_layer=None)
        assert isinstance(gqa.q_norm, torch.nn.Identity)
        assert isinstance(gqa.k_norm, torch.nn.Identity)

    def test_kv_proj_dim_gqa(self):
        """GQA에서 k_proj/v_proj의 출력 차원 = head_dim * num_groups (< dim)."""
        dim, num_heads, num_groups = 128, 8, 2
        gqa = GroupedQueryAttention(dim=dim, num_heads=num_heads, num_groups=num_groups)
        head_dim = dim // num_heads  # 16
        assert gqa.k_proj.out_features == head_dim * num_groups  # 16 * 2 = 32
        assert gqa.v_proj.out_features == head_dim * num_groups

    def test_gradient_flow(self):
        """Gradient가 잘 흐르는지 확인."""
        dim = 32
        gqa = GroupedQueryAttention(dim=dim, num_heads=2, num_groups=2)
        x = torch.randn(1, 4, dim, requires_grad=True)
        out = gqa(x, x, x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


# ── MultiHeadAttention ────────────────────────────────────────────


class TestMultiHeadAttention:
    def test_num_groups_equals_num_heads(self):
        """MHA는 num_groups == num_heads."""
        mha = MultiHeadAttention(dim=64, num_heads=4)
        assert mha.num_groups == 4
        assert mha.heads_per_group == 1

    def test_forward(self):
        mha = MultiHeadAttention(dim=64, num_heads=4)
        mha.eval()
        x = torch.randn(2, 10, 64)
        out = mha(x, x, x)
        assert out.shape == (2, 10, 64)

    def test_kv_proj_full_dim(self):
        """MHA에서 k_proj/v_proj 출력 = head_dim * num_heads = dim."""
        mha = MultiHeadAttention(dim=64, num_heads=4)
        assert mha.k_proj.out_features == 64


# ── MultiQueryAttention ───────────────────────────────────────────


class TestMultiQueryAttention:
    def test_num_groups_is_one(self):
        """MQA는 num_groups == 1."""
        mqa = MultiQueryAttention(dim=64, num_heads=4)
        assert mqa.num_groups == 1
        assert mqa.heads_per_group == 4

    def test_forward(self):
        mqa = MultiQueryAttention(dim=64, num_heads=4)
        mqa.eval()
        x = torch.randn(2, 10, 64)
        out = mqa(x, x, x)
        assert out.shape == (2, 10, 64)

    def test_kv_proj_single_group(self):
        """MQA에서 k_proj/v_proj 출력 = head_dim * 1."""
        dim, num_heads = 64, 4
        mqa = MultiQueryAttention(dim=dim, num_heads=num_heads)
        head_dim = dim // num_heads  # 16
        assert mqa.k_proj.out_features == head_dim  # 16


# ── Validation ────────────────────────────────────────────────────


class TestValidation:
    def test_invalid_dim_heads(self):
        """dim이 num_heads로 나누어지지 않으면 AssertionError."""
        with pytest.raises(AssertionError):
            GroupedQueryAttention(dim=30, num_heads=4, num_groups=2)

    def test_invalid_groups(self):
        """num_heads가 num_groups로 나누어지지 않으면 AssertionError."""
        with pytest.raises(AssertionError):
            GroupedQueryAttention(dim=64, num_heads=4, num_groups=3)
