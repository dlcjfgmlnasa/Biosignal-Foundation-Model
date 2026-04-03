# -*- coding:utf-8 -*-
"""Packed 생체신호 시퀀스의 패치 기반 토큰화.

연속 신호를 고정 크기 패치 단위로 나누어 트랜스포머 입력 토큰으로 변환한다.
PackCollate의 patch_size/stride 정렬과 함께 사용한다.
"""
from __future__ import annotations

import torch
from torch import nn


class ResidualMLP(nn.Module):
    """Residual MLP block for patch embedding (TimesFM style).

    MLP(1 hidden layer) + skip connection으로 패치를 d_model로 변환한다.
    MLP가 비선형 feature를 추출하고, skip path가 원본 정보를 보존한다.

    Parameters
    ----------
    in_dim:
        입력 차원 (patch_size).
    out_dim:
        출력 차원 (d_model).
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        hidden = in_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (*, in_dim) → (*, out_dim)
        return self.mlp(x) + self.skip(x)


class PatchEmbedding(nn.Module):
    """패치 임베딩 (non-overlapping 및 overlapping 지원).

    PackCollate(patch_size=P, stride=S)로 생성된 PackedBatch를 입력받아,
    각 variate를 patch_size 단위로 분할하고 Residual MLP로 투영한다.

    Parameters
    ----------
    patch_size:
        패치 하나의 시간 길이 (time-step 수).
    d_model:
        출력 임베딩 차원.
    stride:
        패치 간 이동 보폭. ``None``이면 ``patch_size``와 동일 (non-overlapping).
        ``stride < patch_size``이면 overlapping. ``patch_size % stride == 0`` 필수.
    bias:
        선형 투영의 bias 사용 여부.
    stem:
        CNN stem 모듈. ``None``이면 Residual MLP 사용.
    """

    def __init__(
        self,
        patch_size: int,
        d_model: int,
        stride: int | None = None,
        bias: bool = True,
        stem: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.d_model = d_model
        assert patch_size % self.stride == 0, (
            f"patch_size({patch_size})는 stride({self.stride})의 배수여야 합니다."
        )
        if stem is not None:
            self.stem = stem
            self.proj = None
        else:
            self.stem = None
            self.proj = ResidualMLP(patch_size, d_model)

    # ── Public API ────────────────────────────────────────────────

    def patchify(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
    ) -> tuple[
        torch.Tensor,  # (batch, num_patches, patch_size) — raw patches
        torch.Tensor,  # (batch, num_patches) long — patch-level sample_id
        torch.Tensor,  # (batch, num_patches) long — patch-level variate_id
        torch.Tensor,  # (batch, num_patches) long — patch-level time_id
        torch.Tensor,  # (batch, num_patches) bool — patch_mask (True=유효)
    ]:
        """패치 추출 + 메타데이터 (projection 미적용)."""
        p = self.patch_size
        s = self.stride
        if s == p:
            return self._patchify_non_overlapping(values, sample_id, variate_id)
        else:
            return self._patchify_overlapping(values, sample_id, variate_id)

    def project(
        self,
        patches: torch.Tensor,  # (batch, num_patches, patch_size)
        patch_signal_types: torch.Tensor | None = None,  # (batch, num_patches) long
    ) -> torch.Tensor:  # (batch, num_patches, d_model)
        """Raw patches를 d_model 임베딩으로 투영한다 (linear 또는 CNN stem)."""
        if self.stem is not None and patch_signal_types is not None:
            return self.stem(patches, patch_signal_types)
        return self.proj(patches)

    def forward(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
        patch_signal_types: torch.Tensor | None = None,  # (batch, num_patches) long
    ) -> tuple[
        torch.Tensor,  # (batch, num_patches, d_model) — 패치 임베딩
        torch.Tensor,  # (batch, num_patches) long — patch-level sample_id
        torch.Tensor,  # (batch, num_patches) long — patch-level variate_id
        torch.Tensor,  # (batch, num_patches) long — patch-level time_id
        torch.Tensor,  # (batch, num_patches) bool — patch_mask (True=유효)
    ]:
        patches, p_sid, p_vid, time_id, patch_mask = self.patchify(
            values, sample_id, variate_id,
        )
        embedded = self.project(patches, patch_signal_types)
        return embedded, p_sid, p_vid, time_id, patch_mask

    # ── Internal patchify methods ──────────────────────────────

    def _patchify_non_overlapping(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.patch_size
        b, l = values.shape
        assert l % p == 0, (
            f"max_length({l})가 patch_size({p})의 배수가 아닙니다. "
            f"PackCollate(patch_size={p})를 사용하세요."
        )
        n = l // p

        patches = values.reshape(b, n, p)  # (B, N, P)

        # 메타데이터 다운샘플
        patch_sample_id = sample_id[:, ::p]  # (B, N)
        patch_variate_id = variate_id[:, ::p]  # (B, N)
        patch_mask = patch_sample_id != 0  # (B, N)

        # time_id
        time_id = self._compute_time_id(patch_sample_id, patch_variate_id)
        time_id[~patch_mask] = 0

        return patches, patch_sample_id, patch_variate_id, time_id, patch_mask

    def _patchify_overlapping(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        p = self.patch_size
        s = self.stride
        b, l = values.shape
        assert l >= p, (
            f"max_length({l})가 patch_size({p})보다 작습니다."
        )
        assert (l - p) % s == 0, (
            f"(max_length({l}) - patch_size({p})) % stride({s}) != 0. "
            f"PackCollate(patch_size={p}, stride={s})를 사용하세요."
        )
        n = (l - p) // s + 1

        patches = values.unfold(-1, p, s)  # (B, N, P)

        # unfold로 패치 유효성 검사
        sid_unfold = sample_id.unfold(-1, p, s)  # (B, N, P)
        vid_unfold = variate_id.unfold(-1, p, s)  # (B, N, P)

        # 각 패치 첫 위치의 메타데이터
        patch_sample_id = sid_unfold[:, :, 0]  # (B, N)
        patch_variate_id = vid_unfold[:, :, 0]  # (B, N)

        # 패치 유효 조건: p개 위치 모두 동일 (sid, vid) & sid != 0
        sid_ok = (sid_unfold == sid_unfold[:, :, :1]).all(dim=-1)  # (B, N)
        vid_ok = (vid_unfold == vid_unfold[:, :, :1]).all(dim=-1)  # (B, N)
        patch_mask = sid_ok & vid_ok & (patch_sample_id != 0)  # (B, N)

        # time_id
        time_id = self._compute_time_id(patch_sample_id, patch_variate_id)
        time_id[~patch_mask] = 0

        return patches, patch_sample_id, patch_variate_id, time_id, patch_mask

    @staticmethod
    def _compute_time_id(
        sample_id: torch.Tensor,  # (batch, num_patches) long
        variate_id: torch.Tensor,  # (batch, num_patches) long
    ) -> torch.Tensor:  # (batch, num_patches) long
        """각 variate 내에서 패치의 순서 인덱스를 계산한다.

        (sample_id, variate_id) 조합이 같은 연속 패치들에 대해
        0부터 시작하는 순차 인덱스를 부여한다.
        """
        b, n = sample_id.shape
        device = sample_id.device

        # 고유 variate 키: sample_id와 variate_id를 조합
        combined = sample_id * (variate_id.max() + 1) + variate_id  # (B, N)

        # 경계 감지: combined가 이전과 다르면 새 variate 시작
        boundary = torch.ones(b, n, dtype=torch.bool, device=device)
        boundary[:, 1:] = combined[:, 1:] != combined[:, :-1]

        # arange와 cummax로 각 위치의 그룹 시작 인덱스 계산
        arange = torch.arange(n, device=device).unsqueeze(0).expand(b, -1)  # (B, N)
        boundary_pos = torch.where(boundary, arange, torch.zeros_like(arange))
        group_start, _ = boundary_pos.cummax(dim=-1)  # (B, N)

        # time_id = 현재 위치 - 그룹 시작 위치
        time_id = arange - group_start  # (B, N)

        return time_id
