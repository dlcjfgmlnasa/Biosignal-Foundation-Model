# -*- coding:utf-8 -*-
from __future__ import annotations

"""Masked Patch Modeling Loss.

Phase 1 (CI): 랜덤 패치 마스킹 → 같은 variate 내 형태학 복원.
Phase 2 (Any-variate): variate-level 마스킹 → 다른 모달리티로부터 복원 (Virtual Sensing).
"""
import torch
from torch import nn


class MaskedPatchLoss(nn.Module):
    """마스킹된 패치 위치만 MSE를 계산하는 손실 함수.

    pred_mask=True인 패치에 대해 (reconstructed - original)^2 평균을 반환한다.
    마스킹된 위치가 없으면 0을 반환한다.
    """

    def forward(
        self,
        reconstructed: torch.Tensor,     # (B, N, P)
        original_patches: torch.Tensor,  # (B, N, P)
        pred_mask: torch.Tensor,         # (B, N) bool
    ) -> torch.Tensor:  # scalar
        n_masked = pred_mask.float().sum()
        if n_masked > 0:
            loss = (
                (reconstructed[pred_mask] - original_patches[pred_mask]) ** 2
            ).mean()
        else:
            loss = reconstructed.new_tensor(0.0)
        return loss


def create_patch_mask(
    patch_mask: torch.Tensor,          # (B, N) — 유효 패치 (True=유효)
    mask_ratio: float = 0.15,
    patch_variate_id: torch.Tensor | None = None,  # (B, N)
    variate_mask_prob: float = 0.0,    # Phase 2: 전체 variate 마스킹 확률
    block_mask: bool = False,          # True면 연속 블록 마스킹
    block_size_min: int = 3,           # 블록 최소 크기 (패치 수)
    block_size_max: int = 8,           # 블록 최대 크기 (패치 수)
) -> torch.Tensor:  # (B, N) bool — 마스킹 대상 (True=마스킹)
    """패치 마스킹 생성.

    Parameters
    ----------
    patch_mask:
        유효 패치 마스크. True=유효 패치.
    mask_ratio:
        랜덤 패치 마스킹 비율.
    patch_variate_id:
        패치별 variate_id. variate-level 마스킹에 필요.
    variate_mask_prob:
        variate-level 마스킹 확률. 0이면 비활성 (Phase 1 동작).
        > 0이면 해당 확률로 랜덤 variate를 선택하여 모든 패치를 마스킹.
    block_mask:
        True이면 연속 블록 단위로 마스킹. 보간 기반 복원을 방지하여
        장기 시간적 의존성 학습을 강제한다.
    block_size_min:
        블록 최소 크기 (패치 수). 기본 3 (3초).
    block_size_max:
        블록 최대 크기 (패치 수). 기본 8 (8초).

    Returns
    -------
    torch.Tensor
        (B, N) bool. True=마스킹 대상.
    """
    b, n = patch_mask.shape
    device = patch_mask.device

    pred_mask = torch.zeros(b, n, dtype=torch.bool, device=device)

    for bi in range(b):
        valid_idx = patch_mask[bi].nonzero(as_tuple=True)[0]  # 유효 패치 인덱스
        if len(valid_idx) == 0:
            continue

        # variate-level 마스킹 (Phase 2)
        if (
            variate_mask_prob > 0
            and patch_variate_id is not None
            and torch.rand(1).item() < variate_mask_prob
        ):
            valid_var_ids = patch_variate_id[bi, valid_idx]
            unique_vars = valid_var_ids[valid_var_ids > 0].unique()
            if len(unique_vars) > 1:
                chosen_var = unique_vars[torch.randint(len(unique_vars), (1,)).item()]
                var_mask = (patch_variate_id[bi] == chosen_var)
                pred_mask[bi] = var_mask & patch_mask[bi]
                continue

        n_valid = len(valid_idx)
        n_mask = max(1, int(n_valid * mask_ratio))

        if block_mask and n_valid >= block_size_min:
            # ── Block Masking ──
            # variate별로 연속 구간을 찾아 블록 단위로 마스킹
            masked_count = 0
            # valid_idx는 정렬되어 있으므로 연속 구간(run) 추출
            runs = _find_contiguous_runs(valid_idx)
            # 랜덤 순서로 run을 순회하며 블록 배치
            run_order = torch.randperm(len(runs)).tolist()
            while masked_count < n_mask:
                placed = False
                for ri in run_order:
                    run_start, run_len = runs[ri]
                    if run_len < block_size_min:
                        continue
                    bs = torch.randint(block_size_min, min(block_size_max, run_len) + 1, (1,)).item()
                    bs = min(bs, n_mask - masked_count)  # 초과 방지
                    if bs < 1:
                        break
                    max_start = run_len - bs
                    offset = torch.randint(0, max_start + 1, (1,)).item()
                    start_idx = run_start + offset
                    pred_mask[bi, start_idx:start_idx + bs] = True
                    masked_count += bs
                    placed = True
                if not placed:
                    break
            # 목표 미달 시 랜덤으로 나머지 채움
            if masked_count < n_mask:
                remaining = valid_idx[~pred_mask[bi, valid_idx]]
                if len(remaining) > 0:
                    extra = min(n_mask - masked_count, len(remaining))
                    perm = torch.randperm(len(remaining), device=device)[:extra]
                    pred_mask[bi, remaining[perm]] = True
        else:
            # ── Random Masking (기본) ──
            perm = torch.randperm(n_valid, device=device)[:n_mask]
            pred_mask[bi, valid_idx[perm]] = True

    return pred_mask


def _find_contiguous_runs(
    indices: torch.Tensor,  # (K,) sorted 인덱스
) -> list[tuple[int, int]]:
    """정렬된 인덱스에서 연속 구간 (start, length) 리스트를 반환한다."""
    if len(indices) == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = indices[0].item()
    prev = start
    for i in range(1, len(indices)):
        cur = indices[i].item()
        if cur == prev + 1:
            prev = cur
        else:
            runs.append((start, prev - start + 1))
            start = cur
            prev = cur
    runs.append((start, prev - start + 1))
    return runs
