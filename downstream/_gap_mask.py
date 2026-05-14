# -*- coding: utf-8 -*-
"""Downstream task 공용 gap masking utilities.

정책 (memory: project_downstream_gap_window_policy):
  Step 1 — Window 의 valid_ratio (non-NaN 비율) < threshold 이면 window 자체 drop
  Step 2 — 통과 window 의 NaN 위치는 0 으로 채우고, gap_mask boolean 별도 저장.
           학습 시 model 이 gap_mask 를 patch-level 로 변환 후 [MASK] 토큰 교체
           (사전학습 mechanism 재사용 — model/biosignal_model.py:237).

[MASK] 토큰 동작:
  - raw waveform 단계: NaN 위치 = 0 fill (PatchEmbedding NaN propagation 회피)
  - PatchEmbedding 이후: gap_mask → patch-level 다운샘플 → mask_token 으로 교체
  - Conditioning (signal_type/spatial_id/loc/scale) 은 mask 위치도 유지됨
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_VALID_RATIO_THRESHOLD = 0.8  # 80% non-NaN 이상 통과


def compute_valid_ratio(arrs: Sequence[np.ndarray]) -> float:
    """Multi-channel window 의 valid sample ratio.

    한 sample 이라도 어떤 channel 에서 NaN 이면 그 sample 은 invalid 로 카운트.
    채널 전체에서 non-NaN 인 sample 비율을 반환.

    Args:
        arrs: 같은 길이 1-D array 리스트 (각 channel)

    Returns:
        valid_ratio ∈ [0, 1]. arrs 비었으면 0.0.
    """
    if not arrs:
        return 0.0
    n_samples = arrs[0].shape[0]
    if n_samples == 0:
        return 0.0
    nan_any = np.zeros(n_samples, dtype=bool)
    for a in arrs:
        nan_any |= np.isnan(a)
    return float(np.mean(~nan_any))


def apply_gap_mask(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN 위치를 0 으로 채우고 boolean gap mask 반환.

    Args:
        arr: 1-D 또는 N-D array

    Returns:
        (filled, gap_mask):
            filled — NaN 위치가 0 으로 교체된 array (원본 dtype 유지)
            gap_mask — bool array, True 면 원본이 NaN 이었음
    """
    gap_mask = np.isnan(arr)
    filled = np.where(gap_mask, 0.0, arr).astype(arr.dtype, copy=False)
    return filled, gap_mask


def apply_gap_mask_multichannel(
    signals: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Multi-channel dict 단위로 gap mask 적용.

    Returns:
        (filled_signals, gap_masks) — 같은 key set 의 dict 쌍.
    """
    filled: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for k, arr in signals.items():
        f, m = apply_gap_mask(arr)
        filled[k] = f
        masks[k] = m
    return filled, masks


def sample_to_patch_mask(
    sample_mask,  # torch.Tensor (B, T) bool, or (T,) for unbatched
    patch_size: int,
    gap_threshold: float = 0.5,
):
    """Sample-level gap_mask → patch-level mask.

    Patch 내 gap 비율 ≥ gap_threshold 면 그 patch 는 mask 됨 (True).
    학습 시점에서 dataloader/model 이 patch_size 알 때 호출.

    Args:
        sample_mask: (B, T) bool tensor 또는 (T,) bool. True=gap.
        patch_size: patch 당 sample 수.
        gap_threshold: patch 마스크 임계 (0~1). 기본 0.5 — 절반 이상 gap 이면 mask.

    Returns:
        torch.Tensor (B, n_patches) bool — patch-level mask. True=gap (mask 대상).
        T 가 patch_size 로 안 나누어떨어지면 끝 부분 잘림.
    """
    import torch  # 지연 import — numpy-only 환경에서도 _gap_mask import 가능

    is_1d = sample_mask.dim() == 1
    if is_1d:
        sample_mask = sample_mask.unsqueeze(0)
    b, t = sample_mask.shape
    n_patches = t // patch_size
    if n_patches == 0:
        return torch.zeros(b, 0, dtype=torch.bool, device=sample_mask.device)
    truncated = sample_mask[:, : n_patches * patch_size]
    reshaped = truncated.reshape(b, n_patches, patch_size).float()
    patch_mask = reshaped.mean(dim=-1) >= gap_threshold
    if is_1d:
        patch_mask = patch_mask.squeeze(0)
    return patch_mask


class GapStats:
    """Window drop / gap masking 통계 누적기.

    prepare_data 의 sweep / fold loop 에서 reuse — 끝에서 summary() 호출.
    """

    def __init__(self):
        self.n_windows_total: int = 0
        self.n_windows_dropped: int = 0
        self.n_samples_total: int = 0
        self.n_samples_masked: int = 0

    def add_window(self, n_samples: int, n_gap_samples: int) -> None:
        """통과한 window 의 통계 추가."""
        self.n_windows_total += 1
        self.n_samples_total += n_samples
        self.n_samples_masked += n_gap_samples

    def add_drop(self) -> None:
        """drop 된 window 한 개 count."""
        self.n_windows_total += 1
        self.n_windows_dropped += 1

    def summary(self) -> str:
        if self.n_windows_total == 0:
            return "  gap policy: 0 windows processed"
        drop_rate = 100.0 * self.n_windows_dropped / self.n_windows_total
        mask_rate = (
            100.0 * self.n_samples_masked / max(1, self.n_samples_total)
        )
        return (
            f"  gap policy: dropped {self.n_windows_dropped}/{self.n_windows_total} "
            f"windows ({drop_rate:.1f}%), "
            f"masked {self.n_samples_masked}/{self.n_samples_total} "
            f"samples ({mask_rate:.2f}%) in retained windows"
        )
