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
    flatline_ratio = float(np.sum(np.abs(diffs) < 1e-10)) / max(len(diffs), 1)

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


def _bandpass_for_peaks(
    data: np.ndarray,
    lo: float,
    hi: float,
    sr: float,
) -> np.ndarray:
    """Peak detection용 임시 bandpass filter. 원본을 변경하지 않는다."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= lo:
        return data
    sos = butter(3, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def ecg_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.5,
) -> dict:
    """ECG 세그먼트의 QRS peak 기반 심박수 품질 검사.

    Parameters
    ----------
    segment:
        (n_timesteps,) 1D ECG 배열 (bandpass + resample 후 100Hz 기준).
    sr:
        sampling rate (Hz).
    min_hr, max_hr:
        정상 심박수 범위 (bpm).
    regularity_threshold:
        R-R interval의 변이계수(std/mean) 상한. 초과하면 불규칙.

    Returns
    -------
    {"hr": float, "hr_valid": bool, "n_peaks": int, "regularity": float, "pass": bool}
    """
    from scipy.signal import find_peaks

    if len(segment) < int(sr * 2):
        return {"hr": 0.0, "hr_valid": False, "n_peaks": 0, "regularity": 1.0, "pass": False}

    # R-peak 검출: prominence 기반 (amplitude 대비 상대적 높이)
    amp = np.max(segment) - np.min(segment)
    if amp < 1e-6:
        return {"hr": 0.0, "hr_valid": False, "n_peaks": 0, "regularity": 1.0, "pass": False}

    # R-R interval 최소 거리: max_hr=200bpm → 0.3s → sr*0.3 samples
    min_distance = int(sr * 60.0 / max_hr * 0.8)  # 약간 여유
    min_distance = max(min_distance, 1)

    peaks, properties = find_peaks(
        segment,
        prominence=amp * 0.2,  # peak-to-peak의 20% 이상 prominence
        distance=min_distance,
    )

    n_peaks = len(peaks)
    if n_peaks < 2:
        return {"hr": 0.0, "hr_valid": False, "n_peaks": n_peaks, "regularity": 1.0, "pass": False}

    # R-R intervals → 심박수
    rr_intervals = np.diff(peaks) / sr  # seconds
    rr_mean = float(np.mean(rr_intervals))
    if rr_mean < 1e-6:
        return {"hr": 0.0, "hr_valid": False, "n_peaks": n_peaks, "regularity": 1.0, "pass": False}

    hr = 60.0 / rr_mean
    hr_valid = min_hr <= hr <= max_hr

    # Regularity: 변이계수 (CV = std/mean)
    rr_std = float(np.std(rr_intervals))
    regularity = rr_std / rr_mean

    passed = hr_valid and regularity < regularity_threshold

    return {
        "hr": round(hr, 1),
        "hr_valid": hr_valid,
        "n_peaks": n_peaks,
        "regularity": round(regularity, 4),
        "pass": passed,
    }


def abp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.5,
) -> dict:
    """ABP 세그먼트의 pulse peak regularity 기반 품질 검사.

    데이터가 z-score 정규화되어 있으므로 절대 mmHg 범위 대신
    pulse 존재 여부와 regularity를 검사한다.

    Parameters
    ----------
    segment:
        (n_timesteps,) 1D ABP 배열 (bandpass + resample 후 100Hz 기준).
    sr:
        sampling rate (Hz).
    min_hr, max_hr:
        정상 맥박수 범위 (bpm).
    regularity_threshold:
        peak-to-peak interval 변이계수 상한.

    Returns
    -------
    {"hr": float, "n_peaks": int, "regularity": float, "pass": bool}
    """
    from scipy.signal import find_peaks

    if len(segment) < int(sr * 2):
        return {"hr": 0.0, "n_peaks": 0, "regularity": 1.0, "pass": False}

    amp = np.max(segment) - np.min(segment)
    if amp < 1e-6:
        return {"hr": 0.0, "n_peaks": 0, "regularity": 1.0, "pass": False}

    # Systolic peaks: 최소 0.4s 간격 (max ~150bpm)
    min_distance = int(sr * 0.4)
    peaks, _ = find_peaks(
        segment,
        prominence=amp * 0.15,
        distance=min_distance,
    )

    if len(peaks) < 2:
        return {"hr": 0.0, "n_peaks": len(peaks), "regularity": 1.0, "pass": False}

    pp_intervals = np.diff(peaks) / sr
    pp_mean = float(np.mean(pp_intervals))
    if pp_mean < 1e-6:
        return {"hr": 0.0, "n_peaks": len(peaks), "regularity": 1.0, "pass": False}

    hr = 60.0 / pp_mean
    hr_valid = min_hr <= hr <= max_hr

    pp_std = float(np.std(pp_intervals))
    regularity = pp_std / pp_mean

    passed = hr_valid and regularity < regularity_threshold

    return {
        "hr": round(hr, 1),
        "n_peaks": len(peaks),
        "regularity": round(regularity, 4),
        "pass": passed,
    }


def ppg_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.5,
) -> dict:
    """PPG 세그먼트의 pulse peak regularity 기반 품질 검사.

    Parameters
    ----------
    segment:
        (n_timesteps,) 1D PPG 배열 (bandpass + resample 후 100Hz 기준).
    sr:
        sampling rate (Hz).
    min_hr, max_hr:
        정상 심박수 범위 (bpm).
    regularity_threshold:
        peak-to-peak interval 변이계수 상한.

    Returns
    -------
    {"hr": float, "regularity": float, "pass": bool}
    """
    from scipy.signal import find_peaks

    if len(segment) < int(sr * 2):
        return {"hr": 0.0, "regularity": 1.0, "pass": False}

    amp = np.max(segment) - np.min(segment)
    if amp < 1e-6:
        return {"hr": 0.0, "regularity": 1.0, "pass": False}

    min_distance = int(sr * 60.0 / max_hr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=amp * 0.15,
        distance=min_distance,
    )

    if len(peaks) < 2:
        return {"hr": 0.0, "regularity": 1.0, "pass": False}

    pp_intervals = np.diff(peaks) / sr
    pp_mean = float(np.mean(pp_intervals))
    if pp_mean < 1e-6:
        return {"hr": 0.0, "regularity": 1.0, "pass": False}

    hr = 60.0 / pp_mean
    hr_valid = min_hr <= hr <= max_hr

    pp_std = float(np.std(pp_intervals))
    regularity = pp_std / pp_mean

    passed = hr_valid and regularity < regularity_threshold

    return {
        "hr": round(hr, 1),
        "regularity": round(regularity, 4),
        "pass": passed,
    }


def co2_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_rr: float = 4.0,
    max_rr: float = 40.0,
) -> dict:
    """CO2 (capnogram) 세그먼트의 호흡 사이클 품질 검사.

    Parameters
    ----------
    segment:
        (n_timesteps,) 1D CO2 배열 (resample 후 100Hz 기준).
    sr:
        sampling rate (Hz).
    min_rr, max_rr:
        정상 호흡수 범위 (breaths/min).

    Returns
    -------
    {"resp_rate": float, "pass": bool}
    """
    from scipy.signal import find_peaks

    duration_s = len(segment) / sr
    if duration_s < 5.0:
        return {"resp_rate": 0.0, "pass": False}

    amp = np.max(segment) - np.min(segment)
    if amp < 1.0:  # CO2 최소 진폭 ~1 mmHg 이상이어야 호흡 존재
        return {"resp_rate": 0.0, "pass": False}

    # End-tidal CO2 peaks: 호흡 주기 최소 60/max_rr 초
    min_distance = int(sr * 60.0 / max_rr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=amp * 0.2,
        distance=min_distance,
    )

    if len(peaks) < 2:
        return {"resp_rate": 0.0, "pass": False}

    # 호흡수 계산: peak 간격 기반
    peak_intervals = np.diff(peaks) / sr
    mean_interval = float(np.mean(peak_intervals))
    if mean_interval < 1e-6:
        return {"resp_rate": 0.0, "pass": False}

    resp_rate = 60.0 / mean_interval
    rr_valid = min_rr <= resp_rate <= max_rr

    return {
        "resp_rate": round(resp_rate, 1),
        "pass": rr_valid,
    }


def awp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_rr: float = 4.0,
    max_rr: float = 40.0,
) -> dict:
    """AWP (기도압) 세그먼트의 환기 사이클 품질 검사.

    Parameters
    ----------
    segment:
        (n_timesteps,) 1D AWP 배열 (resample 후 100Hz 기준).
    sr:
        sampling rate (Hz).
    min_rr, max_rr:
        정상 호흡수 범위 (breaths/min).

    Returns
    -------
    {"resp_rate": float, "pass": bool}
    """
    from scipy.signal import find_peaks

    duration_s = len(segment) / sr
    if duration_s < 5.0:
        return {"resp_rate": 0.0, "pass": False}

    amp = np.max(segment) - np.min(segment)
    if amp < 1.0:  # AWP 최소 진폭 ~1 cmH2O
        return {"resp_rate": 0.0, "pass": False}

    min_distance = int(sr * 60.0 / max_rr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=amp * 0.2,
        distance=min_distance,
    )

    if len(peaks) < 2:
        return {"resp_rate": 0.0, "pass": False}

    peak_intervals = np.diff(peaks) / sr
    mean_interval = float(np.mean(peak_intervals))
    if mean_interval < 1e-6:
        return {"resp_rate": 0.0, "pass": False}

    resp_rate = 60.0 / mean_interval
    rr_valid = min_rr <= resp_rate <= max_rr

    return {
        "resp_rate": round(resp_rate, 1),
        "pass": rr_valid,
    }


# ── Dispatcher: signal type → domain check function ──────────

DOMAIN_QUALITY_CHECKS: dict[str, callable] = {
    "ecg": ecg_quality_check,
    "abp": abp_quality_check,
    "ppg": ppg_quality_check,
    "co2": co2_quality_check,
    "awp": awp_quality_check,
}


def domain_quality_check(stype_key: str, segment: np.ndarray, sr: float = 100.0) -> dict:
    """Signal type에 해당하는 domain-specific 품질 검사를 실행한다.

    Parameters
    ----------
    stype_key:
        신호 타입 키 ("ecg", "abp", "ppg", "co2", "awp").
        "eeg", "cvp" 등 미지원 타입은 항상 {"pass": True}를 반환한다.
    segment:
        (n_timesteps,) 1D 배열.
    sr:
        sampling rate (Hz).

    Returns
    -------
    dict with at least "pass" key.
    """
    check_fn = DOMAIN_QUALITY_CHECKS.get(stype_key)
    if check_fn is None:
        return {"pass": True}
    try:
        return check_fn(segment, sr)
    except Exception:
        # 검출기 자체 에러 시 보수적으로 통과 처리 (데이터 유실 방지)
        return {"pass": True}


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
