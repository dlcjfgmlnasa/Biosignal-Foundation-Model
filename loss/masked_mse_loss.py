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
            # 유효 패치의 variate_id에서 0 제외 (패딩)
            valid_var_ids = patch_variate_id[bi, valid_idx]
            unique_vars = valid_var_ids[valid_var_ids > 0].unique()
            if len(unique_vars) > 1:
                # 랜덤 variate 선택 → 해당 variate 전체 마스킹
                chosen_var = unique_vars[torch.randint(len(unique_vars), (1,)).item()]
                var_mask = (patch_variate_id[bi] == chosen_var)
                pred_mask[bi] = var_mask & patch_mask[bi]
                continue

        # 랜덤 패치 마스킹 (기본 / Phase 1)
        n_valid = len(valid_idx)
        n_mask = max(1, int(n_valid * mask_ratio))
        perm = torch.randperm(n_valid, device=device)[:n_mask]
        pred_mask[bi, valid_idx[perm]] = True

    return pred_mask
