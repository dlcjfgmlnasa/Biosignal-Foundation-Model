"""파서 공통 유틸리티.

모든 데이터셋 파서가 공유하는 resampling, 품질 검증, 저장 헬퍼를 제공한다.
"""

from __future__ import annotations

import math

import numpy as np
import torch


def resample_to_target(
    signal: np.ndarray,
    orig_sr: float,
    target_sr: float = 100.0,
) -> np.ndarray:
    """신호를 target sampling rate로 resampling한다.

    Parameters
    ----------
    signal:
        (n_channels, n_timesteps) 또는 (n_timesteps,) 형태의 배열.
    orig_sr:
        원본 sampling rate (Hz).
    target_sr:
        목표 sampling rate (Hz). 기본값 100.0.

    Returns
    -------
    resampling된 배열. 입력과 동일한 차원 구조를 유지한다.
    """
    if orig_sr == target_sr:
        return signal

    from scipy.signal import resample_poly

    # up/down 비율 계산 — 정수 비율로 변환
    gcd = math.gcd(int(target_sr), int(orig_sr))
    up = int(target_sr) // gcd
    down = int(orig_sr) // gcd

    # 1D 입력 처리
    if signal.ndim == 1:
        return resample_poly(signal, up, down, axis=0).astype(signal.dtype)

    # 2D (n_channels, n_timesteps) — axis=1로 resampling
    return resample_poly(signal, up, down, axis=1).astype(signal.dtype)


def quality_gate(signal: np.ndarray, min_duration_s: float, sr: float) -> bool:
    """최소 길이 검증.

    Parameters
    ----------
    signal:
        (n_channels, n_timesteps) 또는 (n_timesteps,) 형태의 배열.
    min_duration_s:
        최소 신호 길이(초).
    sr:
        sampling rate (Hz).

    Returns
    -------
    True이면 신호가 최소 길이 이상이다.
    """
    n_timesteps = signal.shape[-1]
    return n_timesteps >= min_duration_s * sr


def save_recording(tensor: torch.Tensor, out_path: str) -> None:
    """float32로 강제 변환 후 .pt 파일로 저장한다.

    Parameters
    ----------
    tensor:
        저장할 텐서. shape은 일반적으로 (n_channels, n_timesteps).
    out_path:
        저장 경로 (.pt).
    """
    torch.save(tensor.to(torch.float32), out_path)
