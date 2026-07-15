# -*- coding:utf-8 -*-
"""Packed 시계열 배치 정규화 스케일러.

Salesforce uni2ts (Apache 2.0)에서 포팅.
scatter_add 기반 O(L) 구현 (기존 O(L^2) pairwise mask 대체).

conditioning enrichment (2026-07-15): loc/scale 외에 그룹별 winsorized max/min 도
반환한다 (4-tuple). max/min 은 정규화에는 쓰이지 않고 AdaLN conditioning 입력으로만
사용된다 (calibrated 신호의 SBP/DBP·ETCO2·PIP/PEEP 같은 절대 peak 정보). unitless
신호(PPG·RESP_Imp)는 model 단 게이팅으로 conditioning 에서 제외되므로 여기서는 신호
종류 구분 없이 전 그룹에 대해 균일 계산한다.
"""

from __future__ import annotations

import torch
from torch import nn

from ._util import safe_div


class PackedScaler(nn.Module):
    """Packed batch 정규화 스케일러 기본 클래스.

    forward 는 ``(loc, scale, max, min)`` 4-tuple 을 반환한다. loc/scale 은 정규화용,
    max/min 은 conditioning enrichment 용(winsorized 그룹 극값).
    """

    def forward(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor | None = None,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor | None = None,  # (*batch, seq_len) long
        variate_id: torch.Tensor | None = None,  # (*batch, seq_len) long
    ) -> tuple[
        torch.Tensor,  # (*batch, seq_len, #dim) — loc
        torch.Tensor,  # (*batch, seq_len, #dim) — scale
        torch.Tensor,  # (*batch, seq_len, #dim) — winsorized max
        torch.Tensor,  # (*batch, seq_len, #dim) — winsorized min
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

        return self._get_loc_scale(target, observed_mask, sample_id, variate_id)

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _grouped_winsor_minmax(
    target: torch.Tensor,  # (B, L, D)
    gk: torch.Tensor,  # (B, L, D) group key (expanded)
    n_groups: int,
    loc: torch.Tensor,  # (B, L, D) per-timestep group mean
    scale: torch.Tensor,  # (B, L, D) per-timestep group std
    observed_mask: torch.Tensor,  # (B, L, D) bool
    k: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """그룹별 winsorized max/min — loc±k·scale 로 clamp 후 scatter_reduce amax/amin.

    극값 outlier(스파이크)가 max/min 을 지배하지 않게 하면서 정렬 없이 O(L) 로 robust
    peak 를 얻는다 (분위수 대비 ~130× 저렴). include_self=False + ±inf 초기화로 빈 그룹은
    gather 되지 않는다. 미관측(observed_mask=False) 위치는 ±inf 로 채워 극값 산정에서 제외.
    """
    B, L, D = target.shape
    lo = loc - k * scale
    hi = loc + k * scale
    capped = torch.minimum(torch.maximum(target, lo), hi)  # (B, L, D)
    not_obs = ~observed_mask
    src_max = capped.masked_fill(not_obs, float("-inf"))
    src_min = capped.masked_fill(not_obs, float("inf"))

    group_max = torch.full(
        (B, n_groups, D), float("-inf"), dtype=target.dtype, device=target.device
    )
    group_max.scatter_reduce_(1, gk, src_max, reduce="amax", include_self=False)
    group_min = torch.full(
        (B, n_groups, D), float("inf"), dtype=target.dtype, device=target.device
    )
    group_min.scatter_reduce_(1, gk, src_min, reduce="amin", include_self=False)

    max_ = group_max.gather(1, gk)  # (B, L, D)
    min_ = group_min.gather(1, gk)  # (B, L, D)
    return max_, min_


class PackedNOPScaler(PackedScaler):
    """No-op 스케일러: loc=0, scale=1. max/min 도 0 (conditioning 정보 없음)."""

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (*batch, seq_len, #dim)
        observed_mask: torch.Tensor,  # (*batch, seq_len, #dim) bool
        sample_id: torch.Tensor,  # (*batch, seq_len) long
        variate_id: torch.Tensor,  # (*batch, seq_len) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loc = torch.zeros_like(target, dtype=target.dtype)
        scale = torch.ones_like(target, dtype=target.dtype)
        max_ = torch.zeros_like(target, dtype=target.dtype)
        min_ = torch.zeros_like(target, dtype=target.dtype)
        return loc, scale, max_, min_


class PackedStdScaler(PackedScaler):
    """Z-score 정규화 (Bessel 보정, sample_id/variate_id 그룹별).

    scatter_add 기반 O(L) 구현.

    Parameters
    ----------
    correction:
        분산 계산 시 Bessel 보정값.
    minimum_scale:
        최소 스케일 (수치 안정성).
    winsor_k:
        max/min conditioning 계산 시 loc±k·scale clamp 배수 (outlier 억제).
    """

    def __init__(
        self, correction: int = 1, minimum_scale: float = 1e-5, winsor_k: float = 5.0
    ):
        super().__init__()
        self.correction = correction
        self.minimum_scale = minimum_scale
        self.winsor_k = winsor_k

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (B, L, D)
        observed_mask: torch.Tensor,  # (B, L, D) bool
        sample_id: torch.Tensor,  # (B, L) long
        variate_id: torch.Tensor,  # (B, L) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # 그룹별 winsorized max/min (conditioning enrichment 전용; 정규화엔 미사용)
        max_, min_ = _grouped_winsor_minmax(
            target, gk, n_groups, loc, scale, observed_mask, k=self.winsor_k
        )

        # 패딩 위치(sample_id==0) 초기화
        padding = (sample_id == 0).unsqueeze(-1)  # (B, L, 1)
        loc = loc.masked_fill(padding, 0.0)
        scale = scale.masked_fill(padding, 1.0)
        max_ = max_.masked_fill(padding, 0.0)
        min_ = min_.masked_fill(padding, 0.0)

        return loc, scale, max_, min_


class PackedAbsMeanScaler(PackedScaler):
    """절대 평균 스케일링 (sample_id/variate_id 그룹별).

    scatter_add 기반 O(L) 구현.

    Parameters
    ----------
    minimum_scale:
        최소 스케일 (수치 안정성).
    winsor_k:
        max/min conditioning 계산 시 loc±k·scale clamp 배수.
    """

    def __init__(self, minimum_scale: float = 1e-5, winsor_k: float = 5.0):
        super().__init__()
        self.minimum_scale = minimum_scale
        self.winsor_k = winsor_k

    def _get_loc_scale(
        self,
        target: torch.Tensor,  # (B, L, D)
        observed_mask: torch.Tensor,  # (B, L, D) bool
        sample_id: torch.Tensor,  # (B, L) long
        variate_id: torch.Tensor,  # (B, L) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # 그룹별 winsorized max/min (loc=0 기준 ±k·scale clamp)
        max_, min_ = _grouped_winsor_minmax(
            target, gk, n_groups, loc, scale, observed_mask, k=self.winsor_k
        )

        # 패딩 위치 초기화
        padding = (sample_id == 0).unsqueeze(-1)
        loc = loc.masked_fill(padding, 0.0)
        scale = scale.masked_fill(padding, 1.0)
        max_ = max_.masked_fill(padding, 0.0)
        min_ = min_.masked_fill(padding, 0.0)

        return loc, scale, max_, min_
