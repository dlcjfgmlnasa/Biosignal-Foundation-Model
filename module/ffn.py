# -*- coding:utf-8 -*-
"""Feed-Forward Network 모듈: 표준 FFN, GLU FFN, Mixture of Experts.

Salesforce uni2ts (Apache 2.0)에서 포팅.
"""

from __future__ import annotations

import math
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
        token_mask: torch.Tensor | None = None,  # noqa: ARG002 — MoE 인터페이스 호환
    ) -> torch.Tensor:  # (..., out_dim)
        # token_mask는 MoE FFN과의 인터페이스 통일용; 일반 FFN은 무시.
        x = self._in_proj(x)
        return self.dropout2(self.fc2(self.dropout1(x)))

    def _in_proj(
        self,
        x: torch.Tensor,  # (..., in_dim)
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
        self,
        x: torch.Tensor,  # (..., in_dim)
    ) -> torch.Tensor:  # (..., out_dim)
        return self.activation(self.fc_gate(x)) * self.fc1(x)


class MoEFeedForward(nn.Module):
    """Mixture of Experts FFN (Linear gate 기반 라우팅, top-k expert 선택).

    Switch Transformer 방식의 learned linear gate를 사용하여 토큰별로
    top-k expert를 선택하고, softmax 가중치로 expert 출력을 합산한다.
    학습 시 load balancing auxiliary loss를 ``self.aux_loss``에 저장한다.

    라우팅 모니터링:
        ``get_routing_stats()``를 호출하면 expert별 할당 비율, 라우팅 엔트로피,
        최대/최소 expert 할당 비율을 반환한다.

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

        self.gate = nn.Linear(in_dim, num_experts, bias=False)
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
        self.aux_loss: torch.Tensor | None = None
        # Routing 모니터링: forward마다 갱신 (detached, no grad)
        self._expert_counts: torch.Tensor | None = None  # (E,) — expert별 할당 토큰 수
        self._routing_entropy: float | None = None  # 라우팅 엔트로피 (균등=log(E))

    def forward(
        self,
        x: torch.Tensor,  # (..., in_dim)
        token_mask: torch.Tensor | None = None,  # (..., ) bool — True=유효, False=padded
    ) -> torch.Tensor:  # (..., dim)
        x_squashed = x.view(-1, x.shape[-1])  # (T, in_dim)

        # 유효 토큰 마스크: 패딩 토큰은 라우팅에서 제외해야 load balance가 왜곡되지
        # 않음. 마스크 미제공 시 모든 토큰 유효로 가정 (하위 호환).
        if token_mask is not None:
            valid_flat = token_mask.reshape(-1).bool()  # (T,)
        else:
            valid_flat = torch.ones(
                x_squashed.shape[0], dtype=torch.bool, device=x.device
            )

        # ── Gate: linear → softmax → topk ──
        gate_logits = self.gate(x_squashed)  # (T, num_experts)
        gate_probs = F.softmax(gate_logits, dim=1, dtype=torch.float).type_as(
            x
        )  # (T, E)

        weights, selected_experts = torch.topk(
            gate_logits,
            self.num_experts_per_token,
            dim=1,
        )  # (T, K), (T, K)
        weights = F.softmax(weights, dim=1, dtype=torch.float).type_as(x)  # (T, K)

        # ── Load balancing auxiliary loss (Switch Transformer) ──
        # f_i / probs는 유효 토큰만으로 계산해야 padded 토큰이 expert 부하를 왜곡
        # 시키지 않음.
        valid_count = int(valid_flat.sum().item())
        if valid_count > 0:
            valid_one_hot = F.one_hot(
                selected_experts[valid_flat].reshape(-1),
                self.num_experts,
            ).float()  # (V*K, E)
            tokens_per_expert = valid_one_hot.sum(dim=0)  # (E,)
            f = tokens_per_expert / (valid_count * self.num_experts_per_token)  # (E,)
            probs = gate_probs[valid_flat].mean(dim=0)  # (E,)
        else:
            f = gate_logits.new_zeros(self.num_experts)
            probs = gate_logits.new_zeros(self.num_experts)
            tokens_per_expert = f

        if self.training:
            self.aux_loss = self.num_experts * (f * probs).sum()
        else:
            self.aux_loss = None

        # ── Routing 모니터링 통계 (no grad) ──
        with torch.no_grad():
            self._expert_counts = tokens_per_expert.detach()  # (E,)
            # 라우팅 엔트로피: H = -sum(f_i * log(f_i)), 균등 분포 시 log(E)
            f_clamped = f.detach().clamp(min=1e-8)
            self._routing_entropy = -(f_clamped * f_clamped.log()).sum().item()

        # ── Expert dispatch (argsort 기반 그룹핑) ──
        results = torch.zeros_like(x_squashed)
        flat_experts = selected_experts.reshape(-1)  # (T * K,)
        flat_batch = (
            torch.arange(
                selected_experts.shape[0],
                device=x.device,
            )
            .unsqueeze(-1)
            .expand_as(selected_experts)
            .reshape(-1)
        )  # (T * K,)
        flat_slot = (
            torch.arange(
                selected_experts.shape[1],
                device=x.device,
            )
            .unsqueeze(0)
            .expand_as(selected_experts)
            .reshape(-1)
        )  # (T * K,)
        order = flat_experts.argsort()  # GPU 1회 sort
        sorted_experts = flat_experts[order]
        sorted_batch = flat_batch[order]
        sorted_slot = flat_slot[order]
        # expert 경계를 한 번에 계산
        counts = torch.zeros(self.num_experts, dtype=torch.long, device=x.device)
        counts.scatter_add_(
            0, sorted_experts, torch.ones_like(sorted_experts, dtype=torch.long)
        )
        splits = counts.cumsum(0).tolist()  # CPU sync 1회 (불가피)
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

    def get_routing_stats(self) -> dict[str, object]:
        """최근 forward의 라우팅 통계 반환.

        Returns
        -------
        dict with keys:
            ``expert_load``: list[float] — expert별 할당 비율 (합=1).
            ``routing_entropy``: float — 라우팅 엔트로피 (균등=log(E)).
            ``max_entropy``: float — 최대 엔트로피 (log(num_experts)).
            ``max_min_ratio``: float — 최다/최소 expert 할당 비율 (1.0이 이상적).
        """
        if self._expert_counts is None:
            return {}
        counts = self._expert_counts.float()
        total = counts.sum().clamp(min=1.0)
        load = counts / total  # (E,)
        max_count = counts.max().item()
        min_count = counts.min().item()
        return {
            "expert_load": [round(v, 4) for v in load.tolist()],
            "routing_entropy": round(self._routing_entropy or 0.0, 4),
            "max_entropy": round(math.log(self.num_experts), 4),
            "max_min_ratio": round(max_count / max(min_count, 1.0), 2),
        }
