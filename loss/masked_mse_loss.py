# -*- coding:utf-8 -*-
from __future__ import annotations

"""Masked Patch Modeling Loss.

Phase 1 (CI): 랜덤 패치 마스킹 → 같은 variate 내 형태학 복원.
Phase 2 (Any-variate): variate-level 마스킹 → 다른 모달리티로부터 복원 (Virtual Sensing).
"""
import torch
from torch import nn


def _multi_resolution_stft_loss(
    pred: torch.Tensor,  # (M, P)
    target: torch.Tensor,  # (M, P)
    n_ffts: tuple[int, ...] = (16, 32, 64),
) -> torch.Tensor:
    """Multi-Resolution STFT Loss.

    여러 n_fft 크기로 STFT를 수행하여 시간-주파수 구조를 다중 스케일로 비교한다.
    각 스케일에서 log-magnitude L1 + spectral convergence를 합산.

    Parameters
    ----------
    pred:
        예측 패치. (M, P).
    target:
        원본 패치. (M, P).
    n_ffts:
        STFT window 크기들. hop_length = n_fft // 4.
    """
    loss = pred.new_tensor(0.0)
    # cuFFT half precision은 power-of-2만 지원 → float32로 캐스팅
    pred_f = pred.float()
    target_f = target.float()

    patch_len = pred_f.shape[-1]
    # n_fft가 patch_size보다 크면 스킵
    valid_ffts = [n for n in n_ffts if n <= patch_len]
    if not valid_ffts:
        return loss

    for n_fft in valid_ffts:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=pred.device)
        pred_stft = torch.stft(
            pred_f,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )  # (M, n_fft//2+1, T)
        target_stft = torch.stft(
            target_f,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=window,
            return_complex=True,
        )  # (M, n_fft//2+1, T)

        pred_mag = pred_stft.abs()  # (M, F, T)
        target_mag = target_stft.abs()  # (M, F, T)

        # Spectral Convergence: Frobenius norm ratio
        sc = torch.norm(target_mag - pred_mag, p="fro") / (
            torch.norm(target_mag, p="fro") + 1e-8
        )

        # Log-magnitude L1
        log_mag = (torch.log1p(pred_mag) - torch.log1p(target_mag)).abs().mean()

        loss = loss + sc + log_mag

    return loss / len(valid_ffts)


def compute_peak_weighted_mse(
    pred: torch.Tensor,  # (M, P)
    target: torch.Tensor,  # (M, P)
    peak_alpha: float = 0.0,
) -> torch.Tensor:
    """Peak-Weighted MSE 계산.

    진폭이 큰 sample에 자동으로 높은 가중치를 부여하여
    R-peak, systolic peak 등 임상적으로 중요한 peak 복원을 강화한다.
    peak_alpha=0이면 일반 MSE와 동일.

    Parameters
    ----------
    pred:
        예측 패치. ``(M, P)``.
    target:
        원본 패치. ``(M, P)``.
    peak_alpha:
        Peak 가중치 강도. 0이면 일반 MSE.
    """
    if peak_alpha > 0:
        abs_target = target.abs()  # (M, P)
        max_abs = abs_target.amax(dim=-1, keepdim=True).clamp(min=1e-8)  # (M, 1)
        weight = 1.0 + peak_alpha * (abs_target / max_abs)  # (M, P)
        return (weight * (pred - target) ** 2).mean()
    return ((pred - target) ** 2).mean()


def compute_patch_loss(
    pred: torch.Tensor,  # (M, P)
    target: torch.Tensor,  # (M, P)
    peak_alpha: float = 0.0,
    lambda_spec: float = 0.0,
    spec_n_ffts: tuple[int, ...] = (16, 32, 64),
) -> dict[str, torch.Tensor]:
    """Peak-Weighted MSE + Multi-Resolution STFT Loss 계산.

    Parameters
    ----------
    pred:
        예측 패치. ``(M, P)``.
    target:
        원본 패치. ``(M, P)``.
    peak_alpha:
        Peak 가중치 강도. 0이면 일반 MSE.
    lambda_spec:
        STFT loss 가중치. 0이면 비활성.
    spec_n_ffts:
        STFT window 크기들.

    Returns
    -------
    dict with keys: ``mse``, ``spec``, ``total``.
    """
    mse = compute_peak_weighted_mse(pred, target, peak_alpha)

    if lambda_spec > 0:
        spec_loss = _multi_resolution_stft_loss(pred, target, spec_n_ffts)
        total = mse + lambda_spec * spec_loss
    else:
        spec_loss = mse.new_tensor(0.0)
        total = mse

    return {"mse": mse, "spec": spec_loss, "total": total}


class MaskedPatchLoss(nn.Module):
    """마스킹된 패치 위치만 Peak-Weighted MSE를 계산하는 손실 함수.

    진폭이 큰 sample에 자동으로 높은 가중치를 부여하여
    R-peak, systolic peak 등 임상적으로 중요한 peak 복원을 강화한다.
    peak_alpha=0이면 일반 MSE와 동일.

    pred_mask=True인 패치에 대해 loss를 반환한다.
    마스킹된 위치가 없으면 0을 반환한다.

    Parameters
    ----------
    peak_alpha:
        Peak 가중치 강도. 0이면 일반 MSE. 높을수록 peak에 집중.
    """

    def __init__(
        self,
        peak_alpha: float = 0.0,
        lambda_spec: float = 0.0,
        spec_n_ffts: tuple[int, ...] = (16, 32, 64),
    ) -> None:
        super().__init__()
        self.peak_alpha = peak_alpha
        self.lambda_spec = lambda_spec
        self.spec_n_ffts = spec_n_ffts

    def forward(
        self,
        reconstructed: torch.Tensor,  # (B, N, P)
        original_patches: torch.Tensor,  # (B, N, P)
        pred_mask: torch.Tensor,  # (B, N) bool
    ) -> dict[str, torch.Tensor]:
        n_masked = pred_mask.float().sum()
        if n_masked == 0:
            zero = reconstructed.new_tensor(0.0)
            return {"mse": zero, "spec": zero, "total": zero}

        pred_m = reconstructed[pred_mask]  # (M, P)
        target_m = original_patches[pred_mask]  # (M, P)

        return compute_patch_loss(
            pred_m,
            target_m,
            peak_alpha=self.peak_alpha,
            lambda_spec=self.lambda_spec,
            spec_n_ffts=self.spec_n_ffts,
        )


def create_patch_mask(
    patch_mask: torch.Tensor,  # (B, N) — 유효 패치 (True=유효)
    mask_ratio: float = 0.15,
    patch_variate_id: torch.Tensor | None = None,  # (B, N)
    variate_mask_prob: float = 0.0,  # Phase 2: 전체 variate 마스킹 확률
    block_mask: bool = False,  # True면 연속 블록 마스킹
    block_size_min: int = 3,  # 블록 최소 크기 (패치 수)
    block_size_max: int = 8,  # 블록 최대 크기 (패치 수)
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
                var_mask = patch_variate_id[bi] == chosen_var
                pred_mask[bi] = var_mask & patch_mask[bi]
                continue

        n_valid = len(valid_idx)
        n_mask = max(1, int(n_valid * mask_ratio))

        if block_mask and n_valid >= block_size_min:
            # ── Block Masking ──
            # variate별로 연속 구간을 찾아 블록 단위로 마스킹.
            # 블록 배치 후 해당 영역을 run에서 제외하여 중복/인접 배치를 방지한다.
            masked_count = 0
            # valid_idx는 정렬되어 있으므로 연속 구간(run) 추출
            runs = _find_contiguous_runs(valid_idx)

            while masked_count < n_mask and runs:
                # 배치 가능한 run만 필터링
                eligible = [
                    (i, s, l) for i, (s, l) in enumerate(runs) if l >= block_size_min
                ]
                if not eligible:
                    break

                # 랜덤 run 선택
                pick = torch.randint(0, len(eligible), (1,)).item()
                _, run_start, run_len = eligible[pick]

                bs = torch.randint(
                    block_size_min, min(block_size_max, run_len) + 1, (1,)
                ).item()
                bs = min(bs, n_mask - masked_count)  # 초과 방지
                if bs < 1:
                    break

                max_start = run_len - bs
                offset = torch.randint(0, max_start + 1, (1,)).item()
                start_idx = run_start + offset
                pred_mask[bi, start_idx : start_idx + bs] = True
                masked_count += bs

                # 배치된 블록 영역 + 양옆 1패치 gap을 run에서 제거
                ri = eligible[pick][0]
                old_start, old_len = runs[ri]
                old_end = old_start + old_len
                new_runs: list[tuple[int, int]] = []
                # 왼쪽 sub-run (1패치 gap 확보)
                left_len = start_idx - old_start - 1
                if left_len > 0:
                    new_runs.append((old_start, left_len))
                # 오른쪽 sub-run (1패치 gap 확보)
                right_start = start_idx + bs + 1
                right_len = old_end - right_start
                if right_len > 0:
                    new_runs.append((right_start, right_len))
                # 기존 run을 교체 (block_size_min 미만 run도 보존 — eligible 필터가 걸러줌)
                runs = runs[:ri] + new_runs + runs[ri + 1 :]

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
