"""파서 공통 유틸리티.

모든 데이터셋 파서가 공유하는 resampling, 품질 검증, 저장 헬퍼를 제공한다.
"""

from __future__ import annotations

import math
from pathlib import Path

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


def segment_quality_score(
    segment: np.ndarray,
    max_flatline_ratio: float = 0.5,
    max_clip_ratio: float = 0.1,
    max_high_freq_ratio: float = 2.0,
    min_amplitude: float = 0.0,
    max_amplitude: float = 0.0,
    min_high_freq_ratio: float = 0.0,
) -> dict[str, float]:
    """세그먼트 품질 점수를 계산한다.

    Parameters
    ----------
    segment:
        (n_timesteps,) 1D 배열.
    max_flatline_ratio:
        flatline 비율 상한 (이상이면 불량). 기본 0.5.
    max_clip_ratio:
        clipping 비율 상한 (이상이면 불량). 기본 0.1.
    max_high_freq_ratio:
        고주파 에너지 비율 상한 (이상이면 불량). 기본 2.0.
        신호 타입별로 다른 값을 전달하여 적절한 품질 판정 가능.
    min_amplitude:
        최소 peak-to-peak 진폭. 미만이면 불량. 기본 0.0 (비활성).
    max_amplitude:
        최대 peak-to-peak 진폭. 초과하면 불량 (artifact). 기본 0.0 (비활성).
    min_high_freq_ratio:
        최소 고주파 에너지 비율. 미만이면 불량. 기본 0.0 (비활성).
        ECG처럼 QRS spike가 있어야 정상인 신호에서, hf가 너무 낮으면 spike 없는 것.

    Returns
    -------
    dict with keys:
        ``flatline_ratio``: 연속 동일 값 비율 (0~1).
        ``clip_ratio``: min/max 값에 고정된 비율 (0~1).
        ``high_freq_ratio``: 고주파 에너지 비율 (높으면 노이즈).
        ``amplitude``: peak-to-peak 진폭.
        ``pass``: True이면 품질 통과.
    """
    n = len(segment)
    if n < 2:
        return {"flatline_ratio": 1.0, "clip_ratio": 1.0, "high_freq_ratio": 1.0, "amplitude": 0.0, "pass": False}

    # 1. Flatline: 연속 동일 값 비율
    diffs = np.diff(segment)
    flatline_ratio = float(np.sum(np.abs(diffs) < 1e-4)) / max(len(diffs), 1)

    # 2. Clipping/Saturation: min/max 값에 고정된 비율
    smin, smax = segment.min(), segment.max()
    if smax - smin < 1e-10:
        clip_ratio = 1.0
    else:
        at_min = np.sum(np.abs(segment - smin) < 1e-8)
        at_max = np.sum(np.abs(segment - smax) < 1e-8)
        clip_ratio = float(at_min + at_max) / n

    # 3. High-frequency noise ratio (1차 미분 에너지 / 신호 에너지)
    sig_energy = float(np.mean(segment ** 2))
    diff_energy = float(np.mean(diffs ** 2))
    if sig_energy < 1e-10:
        high_freq_ratio = 1.0
    else:
        high_freq_ratio = diff_energy / sig_energy

    # 4. Peak-to-peak amplitude
    amplitude = float(smax - smin)

    # 통과 기준 — 호출자가 신호별 threshold를 전달
    passed = (
        flatline_ratio < max_flatline_ratio
        and clip_ratio < max_clip_ratio
        and high_freq_ratio < max_high_freq_ratio
        and amplitude >= min_amplitude
        and (max_amplitude <= 0 or amplitude <= max_amplitude)
        and high_freq_ratio >= min_high_freq_ratio
    )

    return {
        "flatline_ratio": flatline_ratio,
        "clip_ratio": clip_ratio,
        "high_freq_ratio": high_freq_ratio,
        "amplitude": amplitude,
        "pass": passed,
    }


# ── Domain-specific quality checks ────────────────────────────
# _quality_checks.py로 분리됨. 하위 호환을 위해 re-export.

from data.parser._quality_checks import (  # noqa: F401, E402
    awp_quality_check,
    abp_quality_check,
    co2_quality_check,
    cvp_quality_check,
    ecg_quality_check,
    eeg_quality_check,
    ppg_quality_check,
    domain_quality_check,
)


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


def save_recording_zarr(
    data: np.ndarray,
    out_path: str | Path,
    compressor=None,
) -> None:
    """(C, T) float32 배열을 zarr 압축 포맷으로 저장한다.

    Parameters
    ----------
    data:
        (n_channels, n_timesteps) float32 배열.
    out_path:
        저장 경로 (.zarr 디렉토리).
    compressor:
        zarr compressor. None이면 blosc(zstd, clevel=3)을 사용한다.
    """
    import zarr

    # float16으로 저장하여 용량 절반 절감 (학습 시 scaler가 정규화하므로 정밀도 충분)
    store_dtype = "float16"

    # zarr v3: codecs 파라미터 사용, v2: compressor 파라미터 사용
    try:
        from zarr.codecs import BloscCodec, BytesCodec
        z = zarr.open(
            str(out_path), mode="w", shape=data.shape, dtype=store_dtype,
            chunks=(data.shape[0], min(data.shape[1], 100_000)),
            codecs=[BytesCodec(), BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")],
        )
    except ImportError:
        # zarr v2 fallback
        import numcodecs
        if compressor is None:
            compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
        z = zarr.open(
            str(out_path), mode="w", shape=data.shape, dtype=store_dtype,
            chunks=(data.shape[0], min(data.shape[1], 100_000)),
            compressor=compressor,
        )
    z[:] = data.astype(np.float16)
