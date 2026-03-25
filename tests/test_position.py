# -*- coding:utf-8 -*-
"""module/position/ 테스트: AttentionBias, BinaryAttentionBias, RotaryProjection, QueryKeyProjection."""
import pytest
import torch

from module.position import (
    AttentionBias,
    BinaryAttentionBias,
    Projection,
    QueryKeyProjection,
    RotaryProjection,
)


# ── BinaryAttentionBias ──────────────────────────────────────────


class TestBinaryAttentionBias:
    def test_output_shape(self):
        """출력 shape이 (batch, group, hpg, q_len, kv_len)인지 확인."""
        dim, num_heads, num_groups = 64, 4, 2
        bias = BinaryAttentionBias(dim=dim, num_heads=num_heads, num_groups=num_groups)

        batch, q_len, kv_len = 2, 10, 12
        hpg = num_heads // num_groups
        head_dim = dim // num_heads

        query = torch.randn(batch, num_groups, hpg, q_len, head_dim)
        key = torch.randn(batch, num_groups, hpg, kv_len, head_dim)
        query_id = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2]]).unsqueeze(1).unsqueeze(1)  # (1, 1, 1, 10)
        kv_id = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]]).unsqueeze(1).unsqueeze(1)  # (1, 1, 1, 12)
        query_id = query_id.expand(batch, -1, -1, -1)
        kv_id = kv_id.expand(batch, -1, -1, -1)

        out = bias(query, key, query_id=query_id, kv_id=kv_id)
        assert out.shape == (batch, num_groups, hpg, q_len, kv_len)

    def test_same_id_different_bias(self):
        """같은 ID vs 다른 ID에 대해 서로 다른 bias 값이 생성되는지 확인."""
        dim, num_heads = 32, 2
        bias = BinaryAttentionBias(dim=dim, num_heads=num_heads, num_groups=num_heads)

        head_dim = dim // num_heads
        query = torch.randn(1, num_heads, 1, 4, head_dim)
        key = torch.randn(1, num_heads, 1, 4, head_dim)
        # token 0,1은 id=0, token 2,3은 id=1
        ids = torch.tensor([0, 0, 1, 1]).view(1, 1, 1, 4)

        out = bias(query, key, query_id=ids, kv_id=ids)
        # (0,0)은 same-id, (0,2)는 diff-id → 값이 달라야 함
        same_val = out[0, 0, 0, 0, 0].item()
        diff_val = out[0, 0, 0, 0, 2].item()
        assert same_val != diff_val, "같은 ID와 다른 ID의 bias가 같으면 안 됨"

    def test_symmetric_for_same_ids(self):
        """모든 ID가 동일할 때 bias가 균일한지 확인."""
        dim, num_heads = 16, 2
        bias = BinaryAttentionBias(dim=dim, num_heads=num_heads, num_groups=num_heads)

        head_dim = dim // num_heads
        query = torch.randn(1, num_heads, 1, 3, head_dim)
        key = torch.randn(1, num_heads, 1, 3, head_dim)
        ids = torch.zeros(1, 1, 1, 3, dtype=torch.long)

        out = bias(query, key, query_id=ids, kv_id=ids)
        # 모든 쌍이 same-id → 모든 값이 동일
        head0 = out[0, 0, 0]
        assert torch.allclose(head0, head0[0, 0].expand_as(head0))


# ── RotaryProjection ─────────────────────────────────────────────


class TestRotaryProjection:
    def test_output_shape(self):
        """RoPE 적용 후 shape 보존 확인."""
        proj_width = 16
        rope = RotaryProjection(proj_width=proj_width, num_heads=4, num_groups=2)

        x = torch.randn(2, 2, 2, 10, proj_width)
        seq_id = torch.arange(10).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(2, 1, 1, -1)

        out = rope(x, seq_id=seq_id)
        assert out.shape == x.shape

    def test_different_positions_different_outputs(self):
        """다른 position에 대해 다른 출력이 나오는지 확인."""
        proj_width = 8
        rope = RotaryProjection(proj_width=proj_width, num_heads=2, num_groups=2)

        x = torch.ones(1, 2, 1, 3, proj_width)
        seq_id = torch.tensor([0, 1, 2]).view(1, 1, 1, 3)

        out = rope(x, seq_id=seq_id)
        # 같은 입력이라도 position이 다르면 출력이 달라야 함
        assert not torch.allclose(out[0, 0, 0, 0], out[0, 0, 0, 1])

    def test_same_position_same_output(self):
        """같은 position에 대해 같은 출력이 나오는지 확인."""
        proj_width = 8
        rope = RotaryProjection(proj_width=proj_width, num_heads=2, num_groups=2)

        x = torch.ones(1, 2, 1, 3, proj_width)
        seq_id = torch.tensor([5, 5, 5]).view(1, 1, 1, 3)

        out = rope(x, seq_id=seq_id)
        assert torch.allclose(out[0, 0, 0, 0], out[0, 0, 0, 1])

    def test_norm_preserving(self):
        """RoPE는 norm을 보존해야 함 (rotation이므로)."""
        proj_width = 16
        rope = RotaryProjection(proj_width=proj_width, num_heads=2, num_groups=2)

        x = torch.randn(1, 2, 1, 5, proj_width)
        seq_id = torch.arange(5).view(1, 1, 1, 5)

        out = rope(x, seq_id=seq_id)
        # 각 token의 norm이 보존되어야 함
        x_norm = x.norm(dim=-1)
        out_norm = out.norm(dim=-1)
        assert torch.allclose(x_norm, out_norm, atol=1e-5)

    def test_auto_extend_max_len(self):
        """max_len보다 큰 seq_id가 들어와도 자동 확장되는지 확인."""
        proj_width = 8
        rope = RotaryProjection(proj_width=proj_width, num_heads=2, num_groups=2, max_len=10)

        x = torch.randn(1, 2, 1, 1, proj_width)
        seq_id = torch.tensor([50]).view(1, 1, 1, 1)  # max_len=10보다 큼

        out = rope(x, seq_id=seq_id)
        assert out.shape == x.shape


# ── QueryKeyProjection ────────────────────────────────────────────


class TestQueryKeyProjection:
    def test_full_projection(self):
        """partial_factor=None: 전체 head_dim에 projection 적용."""
        dim, num_heads, num_groups = 32, 4, 2
        qk_proj = QueryKeyProjection(
            dim=dim, num_heads=num_heads, num_groups=num_groups,
            proj_layer=RotaryProjection,
        )

        head_dim = dim // num_heads
        hpg = num_heads // num_groups
        query = torch.randn(2, num_groups, hpg, 8, head_dim)
        key = torch.randn(2, num_groups, hpg, 8, head_dim)
        seq_id = torch.arange(8).view(1, 1, 1, 8).expand(2, -1, -1, -1)

        q_out, k_out = qk_proj(query, key, query_id=seq_id, kv_id=seq_id)
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape

    def test_partial_factor(self):
        """partial_factor=(0.0, 0.5): head_dim의 절반에만 projection 적용."""
        dim, num_heads, num_groups = 64, 4, 4
        qk_proj = QueryKeyProjection(
            dim=dim, num_heads=num_heads, num_groups=num_groups,
            proj_layer=RotaryProjection,
            partial_factor=(0.0, 0.5),
        )

        head_dim = dim // num_heads  # 16
        assert qk_proj.proj_width == 8  # 0.5 * 16
        assert qk_proj.split_sizes == (0, 8, 8)

        query = torch.randn(1, num_groups, 1, 6, head_dim)
        key = torch.randn(1, num_groups, 1, 6, head_dim)
        seq_id = torch.arange(6).view(1, 1, 1, 6)

        q_out, k_out = qk_proj(query, key, query_id=seq_id, kv_id=seq_id)
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape

    def test_shared_query_key_proj(self):
        """key_proj_layer=None이면 query_proj와 key_proj가 같은 객체."""
        dim, num_heads = 32, 4
        qk_proj = QueryKeyProjection(
            dim=dim, num_heads=num_heads, num_groups=num_heads,
            proj_layer=RotaryProjection,
        )
        assert qk_proj.query_proj is qk_proj.key_proj


# ── ABC 테스트 ────────────────────────────────────────────────────


class TestABCs:
    def test_attention_bias_is_abstract(self):
        """AttentionBias를 직접 인스턴스화하면 에러."""
        with pytest.raises(TypeError):
            AttentionBias(dim=32, num_heads=4, num_groups=2)

    def test_projection_is_abstract(self):
        """Projection을 직접 인스턴스화하면 에러."""
        with pytest.raises(TypeError):
            Projection(proj_width=16, num_heads=4, num_groups=2)
