# -*- coding:utf-8 -*-
"""Packed 시계열 배치 정규화 스케일러.

Salesforce uni2ts (Apache 2.0)에서 포팅.
scatter_add 기반 O(L) 구현 (기존 O(L^2) pairwise mask 대체).
"""

from __future__ import annotations

import torch
from torch import nn

from ._util import safe_div


class PackedScaler(nn.Module):
    """Packed batch 정규화 스케일러 기본 클래스."""

    def forward(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor | None = None,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor | None = None,  # (*batch, seq_len) long
        variate_id: torch.Tensor | None = None,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )
        if variate_id is None:
            variate_id = torch.zeros(
                target.shape[:-1], dtype=torch.long, device=target.device
            )

        loc, scale = self._get_loc_scale(target, observed_mask, sample_id, variate_id)
        return loc, scale

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        raise NotImplementedError


def _make_group_key(
    sample_id: torch.Tensor,  # (B, L)
    variate_id: torch.Tensor,  # (B, L)
) -> tuple[torch.Tensor, int]:  # (B, L), n_groups
    """(sample_id, variate_id) 쌍을 단일 정수 group key로 변환한다.

    Returns
    -------
    group_key:
        ``(B, L)`` — 그룹 키.
    n_groups:
        그룹 수 (scatter_add 대상 텐서 크기).
    """
    max_vid = variate_id.max()  # 0-dim tensor (GPU)
    group_key = sample_id * (max_vid + 1) + variate_id
    # n_groups: scatter_add 대상 크기로 Python int 필요 — 단일 sync로 통합
    n_groups: int = group_key.max().item() + 1
    return group_key, n_groups


class PackedNOPScaler(PackedScaler):
    """No-op 스케일러: loc=0, scale=1."""

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
    ]:
        loc = torch.zeros_like(target, dtype=target.dtype)
        scale = torch.ones_like(target, dtype=target.dtype)
        return loc, scale


class PackedStdScaler(PackedScaler):
    """Z-score 정규화 (Bessel 보정, sample_id/variate_id 그룹별).

    scatter_add 기반 O(L) 구현.

    Parameters
    ----------
    correction:
        분산 계산 시 Bessel 보정값.
    minimum_scale:
        최소 스케일 (수치 안정성).
    """

    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (B, L, D)
        observed_mask: torch.Tensor,  # (B, L, D) bool
        sample_id: torch.Tensor,  # (B, L) long
        variate_id: torch.Tensor,  # (B, L) long
    ) -> tuple[
        torch.Tensor,  # (B, L, D) — loc
        torch.Tensor,  # (B, L, D) — scale
    ]:
        B, L, D = target.shape
        group_key, n_groups = _make_group_key(sample_id, variate_id)  # (B, L), int

        # group_key를 D 차원으로 확장
        gk = group_key.unsqueeze(-1).expand(B, L, D)  # (B, L, D)
        obs_float = observed_mask.to(target.dtype)  # (B, L, D)

        # 그룹별 관측 수
        group_count = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_count.scatter_add_(1, gk, obs_float)  # (B, n_groups, D)

        # 그룹별 합
        group_sum = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_sum.scatter_add_(1, gk, target * obs_float)  # (B, n_groups, D)

        # 그룹별 평균 → per-timestep loc
        group_loc = safe_div(group_sum, group_count)  # (B, n_groups, D)
        loc = group_loc.gather(1, gk)  # (B, L, D)

        # 그룹별 분산
        diff_sq = ((target - loc) ** 2) * obs_float  # (B, L, D)
        group_var_sum = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_var_sum.scatter_add_(1, gk, diff_sq)  # (B, n_groups, D)

        group_var = safe_div(
            group_var_sum, (group_count - self.correction).clamp(min=0)
        )
        group_scale = torch.sqrt(group_var + self.minimum_scale)  # (B, n_groups, D)
        scale = group_scale.gather(1, gk)  # (B, L, D)

        # 패딩 위치(sample_id==0) 초기화
        padding = (sample_id == 0).unsqueeze(-1)  # (B, L, 1)
        loc = loc.masked_fill(padding, 0.0)
        scale = scale.masked_fill(padding, 1.0)

        return loc, scale


class PackedAbsMeanScaler(PackedScaler):
    """절대 평균 스케일링 (sample_id/variate_id 그룹별).

    scatter_add 기반 O(L) 구현.

    Parameters
    ----------
    minimum_scale:
        최소 스케일 (수치 안정성).
    """

    def __init__(self, minimum_scale: float = 1e-5):
        super().__init__()
        self.minimum_scale = minimum_scale

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (B, L, D)
        observed_mask: torch.Tensor,  # (B, L, D) bool
        sample_id: torch.Tensor,  # (B, L) long
        variate_id: torch.Tensor,  # (B, L) long
    ) -> tuple[
        torch.Tensor,  # (B, L, D) — loc
        torch.Tensor,  # (B, L, D) — scale
    ]:
        B, L, D = target.shape
        group_key, n_groups = _make_group_key(sample_id, variate_id)  # (B, L), int

        gk = group_key.unsqueeze(-1).expand(B, L, D)  # (B, L, D)
        obs_float = observed_mask.to(target.dtype)  # (B, L, D)

        # 그룹별 관측 수
        group_count = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_count.scatter_add_(1, gk, obs_float)

        # 그룹별 절대값 합
        group_abs_sum = torch.zeros(
            B, n_groups, D, dtype=target.dtype, device=target.device
        )
        group_abs_sum.scatter_add_(1, gk, target.abs() * obs_float)

        # 그룹별 절대 평균 → scale
        group_scale = safe_div(group_abs_sum, group_count)
        group_scale = torch.clamp(group_scale, min=self.minimum_scale)
        scale = group_scale.gather(1, gk)  # (B, L, D)

        loc = torch.zeros_like(scale)

        # 패딩 위치 초기화
        padding = (sample_id == 0).unsqueeze(-1)
        loc = loc.masked_fill(padding, 0.0)
        scale = scale.masked_fill(padding, 1.0)

        return loc, scale
