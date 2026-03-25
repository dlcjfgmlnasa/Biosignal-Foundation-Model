# -*- coding:utf-8 -*-
"""Patch-based tokenization for packed biosignal sequences.

연속 신호를 고정 크기 패치 단위로 나누어 트랜스포머 입력 토큰으로 변환한다.
PackCollate의 patch_size/stride 정렬과 함께 사용한다.
"""
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """패치 임베딩 (non-overlapping 및 overlapping 지원).

    PackCollate(patch_size=P, stride=S)로 생성된 PackedBatch를 입력받아,
    각 variate를 patch_size 단위로 분할하고 선형 투영한다.

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
            self.proj = nn.Linear(patch_size, d_model, bias=bias)

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
        P = self.patch_size
        S = self.stride
        if S == P:
            return self._patchify_non_overlapping(values, sample_id, variate_id)
        else:
            return self._patchify_overlapping(values, sample_id, variate_id)

    def project(
        self,
        patches: torch.Tensor,  # (batch, num_patches, patch_size)
        patch_signal_types: torch.Tensor | None = None,  # (batch, num_patches) long
    ) -> torch.Tensor:  # (batch, num_patches, d_model)
        """Raw patches → d_model embedding (linear 또는 CNN stem)."""
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
        P = self.patch_size
        B, L = values.shape
        assert L % P == 0, (
            f"max_length({L})가 patch_size({P})의 배수가 아닙니다. "
            f"PackCollate(patch_size={P})를 사용하세요."
        )
        N = L // P

        patches = values.reshape(B, N, P)  # (B, N, P)

        # Downsample metadata
        patch_sample_id = sample_id[:, ::P]  # (B, N)
        patch_variate_id = variate_id[:, ::P]  # (B, N)
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
        P = self.patch_size
        S = self.stride
        B, L = values.shape
        assert L >= P, (
            f"max_length({L})가 patch_size({P})보다 작습니다."
        )
        assert (L - P) % S == 0, (
            f"(max_length({L}) - patch_size({P})) % stride({S}) != 0. "
            f"PackCollate(patch_size={P}, stride={S})를 사용하세요."
        )
        N = (L - P) // S + 1

        patches = values.unfold(-1, P, S)  # (B, N, P)

        # Unfold metadata to check patch validity
        sid_unfold = sample_id.unfold(-1, P, S)  # (B, N, P)
        vid_unfold = variate_id.unfold(-1, P, S)  # (B, N, P)

        # Metadata from first position of each patch
        patch_sample_id = sid_unfold[:, :, 0]  # (B, N)
        patch_variate_id = vid_unfold[:, :, 0]  # (B, N)

        # Patch is valid iff all P positions have same (sid, vid) and sid != 0
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
        B, N = sample_id.shape
        device = sample_id.device

        # 고유 variate 키: sample_id와 variate_id를 조합
        combined = sample_id * (variate_id.max().item() + 1) + variate_id  # (B, N)

        # 경계 감지: combined가 이전과 다르면 새 variate 시작
        boundary = torch.ones(B, N, dtype=torch.bool, device=device)
        boundary[:, 1:] = combined[:, 1:] != combined[:, :-1]

        # arange와 cummax로 각 위치의 그룹 시작 인덱스 계산
        arange = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)  # (B, N)
        boundary_pos = torch.where(boundary, arange, torch.zeros_like(arange))
        group_start, _ = boundary_pos.cummax(dim=-1)  # (B, N)

        # time_id = 현재 위치 - 그룹 시작 위치
        time_id = arange - group_start  # (B, N)

        return time_id


class MultiResolutionPatchEmbedding(nn.Module):
    """다중 해상도 패치 임베딩 (MOIRAI 스타일).

    sampling_rate별로 서로 다른 patch_size를 적용하여,
    모든 패치가 대략 같은 물리적 시간을 커버하도록 한다.

    Parameters
    ----------
    patch_sizes:
        사용 가능한 패치 크기 목록 (e.g., ``[8, 16, 32, 64]``).
    d_model:
        출력 임베딩 차원.
    target_patch_duration_ms:
        목표 패치 물리적 지속 시간 (ms). sampling_rate와 함께
        가장 가까운 patch_size를 선택한다.
    bias:
        선형 투영의 bias 사용 여부.
    """

    def __init__(
        self,
        patch_sizes: list[int],
        d_model: int,
        target_patch_duration_ms: float,
        bias: bool = True,
        stem: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.patch_sizes = sorted(patch_sizes)
        self.d_model = d_model
        self.target_patch_duration_ms = target_patch_duration_ms
        self.stem = stem
        if stem is None:
            self.projections = nn.ModuleDict({
                str(ps): nn.Linear(ps, d_model, bias=bias) for ps in self.patch_sizes
            })
        else:
            self.projections = None

    def _resolve_patch_size(self, sampling_rate: float) -> int:
        """sampling_rate에서 가장 적합한 patch_size를 결정한다."""
        ideal = sampling_rate * self.target_patch_duration_ms / 1000.0
        return min(self.patch_sizes, key=lambda ps: abs(ps - ideal))

    def forward(
        self,
        values: torch.Tensor,  # (batch, max_length)
        sample_id: torch.Tensor,  # (batch, max_length) long
        variate_id: torch.Tensor,  # (batch, max_length) long
        sampling_rates: torch.Tensor,  # (total_variates,) float
        padded_lengths: torch.Tensor,  # (total_variates,) long
        variate_patch_sizes: torch.Tensor,  # (total_variates,) long
        signal_types: torch.Tensor | None = None,  # (total_variates,) long
    ) -> tuple[
        torch.Tensor,  # (batch, max_patches, d_model) — 패치 임베딩
        torch.Tensor,  # (batch, max_patches) long — patch-level sample_id
        torch.Tensor,  # (batch, max_patches) long — patch-level variate_id
        torch.Tensor,  # (batch, max_patches) long — patch-level time_id
        torch.Tensor,  # (batch, max_patches) bool — patch_mask (True=유효)
    ]:
        B, L = values.shape
        device = values.device

        # 각 행의 variate 세그먼트를 검출하고, per-variate 패칭 수행
        row_embeds: list[torch.Tensor] = []
        row_sids: list[torch.Tensor] = []
        row_vids: list[torch.Tensor] = []
        flat_var_idx = 0

        for b in range(B):
            segments = _detect_segments(sample_id[b], variate_id[b])
            embed_parts: list[torch.Tensor] = []
            sid_parts: list[torch.Tensor] = []
            vid_parts: list[torch.Tensor] = []

            for seg_start, seg_end, sid_val, vid_val in segments:
                ps = variate_patch_sizes[flat_var_idx].item()
                seg_len = seg_end - seg_start
                n_patches = seg_len // ps

                if n_patches > 0:
                    seg_values = values[b, seg_start:seg_start + n_patches * ps]
                    patches = seg_values.reshape(n_patches, ps)

                    # CNN stem 또는 linear projection
                    if self.stem is not None and signal_types is not None:
                        st = signal_types[flat_var_idx].item()
                        embedded = self.stem.stems[st](patches)  # (n_patches, d_model)
                    else:
                        proj = self.projections[str(ps)]
                        embedded = proj(patches)  # (n_patches, d_model)

                    embed_parts.append(embedded)
                    sid_parts.append(
                        torch.full((n_patches,), sid_val, dtype=torch.long, device=device)
                    )
                    vid_parts.append(
                        torch.full((n_patches,), vid_val, dtype=torch.long, device=device)
                    )

                flat_var_idx += 1

            if embed_parts:
                row_embeds.append(torch.cat(embed_parts, dim=0))
                row_sids.append(torch.cat(sid_parts, dim=0))
                row_vids.append(torch.cat(vid_parts, dim=0))
            else:
                row_embeds.append(torch.zeros(0, self.d_model, device=device))
                row_sids.append(torch.zeros(0, dtype=torch.long, device=device))
                row_vids.append(torch.zeros(0, dtype=torch.long, device=device))

        # 최대 패치 수로 패딩하여 (B, max_patches, d_model) 텐서 생성
        max_patches = max(e.shape[0] for e in row_embeds) if row_embeds else 0
        if max_patches == 0:
            max_patches = 1  # 최소 1

        out_embeds = torch.zeros(B, max_patches, self.d_model, device=device)
        out_sids = torch.zeros(B, max_patches, dtype=torch.long, device=device)
        out_vids = torch.zeros(B, max_patches, dtype=torch.long, device=device)
        out_mask = torch.zeros(B, max_patches, dtype=torch.bool, device=device)

        for b in range(B):
            n = row_embeds[b].shape[0]
            if n > 0:
                out_embeds[b, :n] = row_embeds[b]
                out_sids[b, :n] = row_sids[b]
                out_vids[b, :n] = row_vids[b]
                out_mask[b, :n] = True

        # time_id 계산
        time_id = PatchEmbedding._compute_time_id(out_sids, out_vids)
        time_id[~out_mask] = 0

        return out_embeds, out_sids, out_vids, time_id, out_mask


def _detect_segments(
    sample_id_row: torch.Tensor,  # (max_length,) long
    variate_id_row: torch.Tensor,  # (max_length,) long
) -> list[tuple[int, int, int, int]]:
    """한 행에서 연속 variate 세그먼트를 검출한다.

    Returns
    -------
    list of (start, end, sample_id, variate_id) tuples.
    Padding 영역 (sample_id == 0)은 제외한다.
    """
    combined = sample_id_row * 10000 + variate_id_row
    L = combined.shape[0]

    segments: list[tuple[int, int, int, int]] = []
    if L == 0:
        return segments

    changes = torch.where(combined[1:] != combined[:-1])[0] + 1
    starts = torch.cat([torch.tensor([0], device=combined.device), changes])
    ends = torch.cat([changes, torch.tensor([L], device=combined.device)])

    for s, e in zip(starts, ends):
        sid = sample_id_row[s].item()
        vid = variate_id_row[s].item()
        if sid != 0:
            segments.append((s.item(), e.item(), sid, vid))

    return segments
