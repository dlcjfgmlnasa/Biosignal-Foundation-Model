# -*- coding:utf-8 -*-
"""Transformer Encoder (GQA, GLU FFN, MoE, position encoding 지원).

Salesforce uni2ts (Apache 2.0)에서 포팅.
RMSNorm을 기본 norm_layer로 사용하도록 수정.
주의: forward의 var_id/time_id는 PackedBatch의 sample_id/variate_id에 대응.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from .attention import GroupedQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward, MoEFeedForward
from .norm import RMSNorm
from .position import AttentionBias, QueryKeyProjection


class TransformerEncoderLayer(nn.Module):
    """단일 Transformer Encoder 레이어.

    Pre-norm 또는 Post-norm 구조를 지원하며,
    Self-Attention + FFN (또는 MoE FFN) 구성.

    Parameters
    ----------
    self_attn:
        Self-attention 모듈.
    ffn:
        Feed-forward 모듈.
    norm1:
        첫 번째 정규화 레이어.
    norm2:
        두 번째 정규화 레이어.
    post_attn_dropout_p:
        어텐션 출력 드롭아웃 확률.
    pre_norm:
        ``True``이면 Pre-norm, ``False``이면 Post-norm.
    """

    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,
        norm1: nn.Module | None,
        norm2: nn.Module | None,
        post_attn_dropout_p: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = post_attn_dropout_p

        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1 or nn.Identity()
        self.norm2 = norm2 or nn.Identity()
        self.dropout = nn.Dropout(post_attn_dropout_p)

    def forward(
        self,
        x: torch.Tensor,  # (*batch, time_len, dim)
        attn_mask: torch.Tensor | None = None,  # (*batch, time_len, time_len) bool
        var_id: torch.Tensor | None = None,  # (*batch, time_len) long
        time_id: torch.Tensor | None = None,  # (*batch, time_len) long
    ) -> torch.Tensor:  # (*batch, time_len, dim)
        if self.pre_norm:
            x = x + self._sa_block(
                self.norm1(x), attn_mask, var_id=var_id, time_id=time_id
            )
            x = x + self.ffn(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, attn_mask, var_id=var_id, time_id=time_id)
            )
            x = self.norm2(x + self.ffn(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,  # (*batch, time_len, dim)
        attn_mask: torch.Tensor | None,  # (*batch, time_len, time_len) bool
        var_id: torch.Tensor | None = None,  # (*batch, time_len) long
        time_id: torch.Tensor | None = None,  # (*batch, time_len) long
    ) -> torch.Tensor:  # (*batch, time_len, dim)
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_var_id=var_id,
            kv_var_id=var_id,
            query_time_id=time_id,
            kv_time_id=time_id,
        )
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Stacked Transformer Encoder.

    GQA, GLU FFN, MoE, RoPE, variate bias 등을 조합하여
    다중 레이어 인코더를 구성한다.

    Parameters
    ----------
    d_model:
        임베딩 차원.
    num_layers:
        레이어 수.
    num_heads:
        어텐션 헤드 수. ``None``이면 ``d_model // 64``.
    num_groups:
        GQA 그룹 수. ``None``이면 ``num_heads`` (MHA).
    pre_norm:
        Pre-norm 사용 여부.
    attn_dropout_p:
        어텐션 드롭아웃 확률.
    dropout_p:
        일반 드롭아웃 확률.
    norm_layer:
        정규화 레이어 팩토리.
    activation:
        FFN 활성화 함수.
    use_moe:
        MoE FFN 사용 여부.
    use_glu:
        GLU FFN 사용 여부.
    use_qk_norm:
        Q/K norm 사용 여부.
    d_ff:
        FFN 은닉 차원. ``None``이면 기본값 사용.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int | None = None,
        num_groups: int | None = None,
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Callable[[int], nn.Module] | None = RMSNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_moe: bool = False,
        use_glu: bool = True,
        use_qk_norm: bool = True,
        var_attn_bias_layer: Callable[[int, int, int], AttentionBias] | None = None,
        time_attn_bias_layer: Callable[[int, int, int], AttentionBias] | None = None,
        var_qk_proj_layer: Callable[[int, int, int], QueryKeyProjection] | None = None,
        time_qk_proj_layer: Callable[[int, int, int], QueryKeyProjection] | None = None,
        shared_var_attn_bias: bool = False,
        shared_time_attn_bias: bool = False,
        shared_var_qk_proj: bool = False,
        shared_time_qk_proj: bool = False,
        d_ff: int | None = None,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
    ):
        super().__init__()
        self.use_moe = use_moe
        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads  # 기본 MHA

        var_attn_bias = self.get_layer(
            d_model, num_heads, num_groups, var_attn_bias_layer, shared_var_attn_bias
        )
        time_attn_bias = self.get_layer(
            d_model, num_heads, num_groups, time_attn_bias_layer, shared_time_attn_bias
        )
        var_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )

        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=False,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
        if not use_moe:
            get_ffn = partial(
                GatedLinearUnitFeedForward if use_glu else FeedForward,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=False,
                ffn_dropout_p=dropout_p,
            )
        else:
            get_ffn = partial(
                MoEFeedForward,
                num_experts=num_experts,
                num_experts_per_token=num_experts_per_token,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=False,
                ffn_dropout_p=dropout_p,
            )
        get_encoder_layer_norm = partial(norm_layer, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self_attn=get_self_attn(),
                    ffn=get_ffn(),
                    norm1=get_encoder_layer_norm(),
                    norm2=get_encoder_layer_norm(),
                    pre_norm=pre_norm,
                    post_attn_dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm_layer(d_model)

    @staticmethod
    def get_layer(
        dim: int,
        num_heads: int,
        num_groups: int,
        layer: Callable | None,
        shared_layer: bool,
    ) -> Callable[[], nn.Module] | None:
        """레이어 팩토리를 반환한다. shared면 동일 인스턴스를 재사용."""
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)

    def forward(
        self,
        x: torch.Tensor,  # (*batch, time_len, dim)
        attn_mask: torch.Tensor | None = None,  # (*batch, time_len, time_len) bool
        var_id: torch.Tensor | None = None,  # (*batch, time_len) long
        time_id: torch.Tensor | None = None,  # (*batch, time_len) long
    ) -> torch.Tensor:  # (*batch, time_len, dim)
        for layer in self.layers:
            x = layer(x, attn_mask, var_id=var_id, time_id=time_id)
        return self.norm(x)
