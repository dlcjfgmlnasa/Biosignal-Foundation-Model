# -*- coding:utf-8 -*-
"""Feed-Forward Network 모듈: 표준 FFN, GLU FFN, Mixture of Experts.

Salesforce uni2ts (Apache 2.0)에서 포팅.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    """표준 Feed-Forward Network (fc1 → activation → fc2).

    Parameters
    ----------
    in_dim:
        입력 차원.
    hidden_dim:
        은닉 차원. ``None``이면 ``4 * in_dim``.
    out_dim:
        출력 차원. ``None``이면 ``in_dim``.
    activation:
        활성화 함수.
    bias:
        Linear bias 사용 여부.
    ffn_dropout_p:
        드롭아웃 확률.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or 4 * in_dim
        out_dim = out_dim or in_dim

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.bias = bias
        self.ffn_dropout_p = ffn_dropout_p

        self.fc1 = nn.Linear(in_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, out_dim, bias=bias)
        self.dropout1 = nn.Dropout(ffn_dropout_p)
        self.dropout2 = nn.Dropout(ffn_dropout_p)
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,  # (..., in_dim)
        centroid: torch.Tensor | None = None,  # (expert, in_dim)
    ) -> torch.Tensor:  # (..., out_dim)
        x = self._in_proj(x)
        return self.dropout2(self.fc2(self.dropout1(x)))

    def _in_proj(
        self, x: torch.Tensor,  # (..., in_dim)
    ) -> torch.Tensor:  # (..., out_dim)
        return self.activation(self.fc1(x))


class GatedLinearUnitFeedForward(FeedForward):
    """SiLU-gated FFN (hidden_dim = 2/3 * 4d, 8의 배수로 반올림).

    Parameters
    ----------
    in_dim:
        입력 차원.
    hidden_dim:
        은닉 차원. ``None``이면 ``adjust_hidden_dim(4 * in_dim)``.
    out_dim:
        출력 차원. ``None``이면 ``in_dim``.
    activation:
        게이트 활성화 함수.
    bias:
        Linear bias 사용 여부.
    ffn_dropout_p:
        드롭아웃 확률.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__(
            in_dim,
            hidden_dim=hidden_dim or self.adjust_hidden_dim(4 * in_dim),
            out_dim=out_dim,
            activation=activation,
            bias=bias,
            ffn_dropout_p=ffn_dropout_p,
        )
        self.fc_gate = nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias)

    @staticmethod
    def adjust_hidden_dim(dim: int) -> int:
        return (int(dim * 2 / 3) + 7) // 8 * 8

    def _in_proj(
        self, x: torch.Tensor,  # (..., in_dim)
    ) -> torch.Tensor:  # (..., out_dim)
        return self.activation(self.fc_gate(x)) * self.fc1(x)


class MoEFeedForward(nn.Module):
    """Mixture of Experts FFN (centroid 기반 라우팅, top-k expert 선택).

    Parameters
    ----------
    num_experts:
        전문가 수.
    num_experts_per_token:
        토큰당 활성화되는 전문가 수.
    in_dim:
        입력 차원.
    hidden_dim:
        각 expert의 은닉 차원.
    out_dim:
        출력 차원.
    activation:
        활성화 함수.
    bias:
        Linear bias 사용 여부.
    ffn_dropout_p:
        드롭아웃 확률.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_token: int,
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token

        self.experts = nn.ModuleList(
            [
                GatedLinearUnitFeedForward(
                    in_dim=in_dim,
                    hidden_dim=hidden_dim,
                    out_dim=out_dim,
                    activation=activation,
                    bias=bias,
                    ffn_dropout_p=ffn_dropout_p,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,  # (..., in_dim)
        centroid: torch.Tensor | None = None,  # (expert, in_dim)
    ) -> torch.Tensor:  # (..., dim)
        x_squashed = x.view(-1, x.shape[-1])

        centroid = centroid.to(x.device).type_as(x)
        if len(x.shape) > 3:
            x_temp = x.view(-1, x.shape[-2], x.shape[-1])
        else:
            x_temp = x
        centroid = centroid.unsqueeze(0).repeat(x_temp.shape[0], 1, 1)
        cdist = torch.cdist(x_temp, centroid)
        gate_logits = cdist.view(-1, cdist.shape[-1])

        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_token)
        weights = F.softmax(
            weights,
            dim=1,
            dtype=torch.float,
        ).type_as(x)

        results = torch.zeros_like(x_squashed)
        # expert별 인덱스를 1회 sync로 사전 그룹핑 (expert 수만큼 torch.where 방지)
        flat_experts = selected_experts.reshape(-1)               # (T * K,)
        flat_batch = torch.arange(
            selected_experts.shape[0], device=x.device,
        ).unsqueeze(-1).expand_as(selected_experts).reshape(-1)   # (T * K,)
        flat_slot = torch.arange(
            selected_experts.shape[1], device=x.device,
        ).unsqueeze(0).expand_as(selected_experts).reshape(-1)    # (T * K,)
        order = flat_experts.argsort()                            # GPU 1회 sort
        sorted_experts = flat_experts[order]
        sorted_batch = flat_batch[order]
        sorted_slot = flat_slot[order]
        # expert 경계를 한 번에 계산
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        counts.scatter_add_(0, sorted_experts, torch.ones_like(sorted_experts, dtype=torch.long))
        splits = counts.cumsum(0).tolist()                        # CPU sync 1회 (불가피)
        start = 0
        for i, expert in enumerate(self.experts):
            end = splits[i]
            if start < end:
                idx = sorted_batch[start:end]
                slot = sorted_slot[start:end]
                results[idx] += weights[idx, slot, None] * expert(x_squashed[idx])
            start = end

        results = results.view_as(x)
        return results
