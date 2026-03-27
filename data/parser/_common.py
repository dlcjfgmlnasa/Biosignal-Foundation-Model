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
    regularity_threshold: float = 0.7,
    min_autocorr: float = 0.10,
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
    min_autocorr:
        HR lag 범위 autocorrelation 최소값. R-peak가 sharp하여 0.25로 설정.

    Returns
    -------
    {"hr": float, "hr_valid": bool, "n_peaks": int, "regularity": float,
     "autocorr_peak": float, "pass": bool}
    """
    from scipy.signal import find_peaks

    _fail = {"hr": 0.0, "hr_valid": False, "n_peaks": 0, "regularity": 1.0, "autocorr_peak": 0.0, "pass": False}

    if len(segment) < int(sr * 2):
        return _fail

    # R-peak 검출: IQR 기반 prominence (outlier spike에 robust)
    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        return _fail

    # R-R interval 최소 거리: max_hr=200bpm → 0.3s → sr*0.3 samples
    min_distance = int(sr * 60.0 / max_hr * 0.8)  # 약간 여유
    min_distance = max(min_distance, 1)

    peaks, properties = find_peaks(
        segment,
        prominence=iqr * 0.5,  # IQR의 50% — outlier에 robust
        distance=min_distance,
    )

    n_peaks = len(peaks)
    if n_peaks < 2:
        _fail["n_peaks"] = n_peaks
        return _fail

    # R-R intervals → 심박수
    rr_intervals = np.diff(peaks) / sr  # seconds
    rr_mean = float(np.mean(rr_intervals))
    if rr_mean < 1e-6:
        _fail["n_peaks"] = n_peaks
        return _fail

    hr = 60.0 / rr_mean
    hr_valid = min_hr <= hr <= max_hr

    # Regularity: 변이계수 (CV = std/mean)
    rr_std = float(np.std(rr_intervals))
    regularity = rr_std / rr_mean

    # Autocorrelation 주기성 체크
    min_lag_s = 60.0 / max_hr
    max_lag_s = 60.0 / min_hr
    autocorr_peak = _autocorrelation_peak(segment, sr, min_lag_s, max_lag_s)

    passed = hr_valid and regularity < regularity_threshold and autocorr_peak >= min_autocorr

    return {
        "hr": round(hr, 1),
        "hr_valid": hr_valid,
        "n_peaks": n_peaks,
        "regularity": round(regularity, 4),
        "autocorr_peak": round(autocorr_peak, 4),
        "pass": passed,
    }


def abp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.5,
    min_autocorr: float = 0.10,
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
    min_autocorr:
        HR lag 범위 autocorrelation 최소값.

    Returns
    -------
    {"hr": float, "n_peaks": int, "regularity": float, "autocorr_peak": float, "pass": bool}
    """
    from scipy.signal import find_peaks

    _fail = {"hr": 0.0, "n_peaks": 0, "regularity": 1.0, "autocorr_peak": 0.0, "pass": False}

    if len(segment) < int(sr * 2):
        return _fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        return _fail

    # Systolic peaks: 최소 0.4s 간격 (max ~150bpm)
    min_distance = int(sr * 0.4)
    peaks, _ = find_peaks(
        segment,
        prominence=iqr * 0.5,
        distance=min_distance,
    )

    if len(peaks) < 2:
        _fail["n_peaks"] = len(peaks)
        return _fail

    pp_intervals = np.diff(peaks) / sr
    pp_mean = float(np.mean(pp_intervals))
    if pp_mean < 1e-6:
        _fail["n_peaks"] = len(peaks)
        return _fail

    hr = 60.0 / pp_mean
    hr_valid = min_hr <= hr <= max_hr

    pp_std = float(np.std(pp_intervals))
    regularity = pp_std / pp_mean

    # Autocorrelation 주기성 체크
    min_lag_s = 60.0 / max_hr
    max_lag_s = 60.0 / min_hr
    autocorr_peak = _autocorrelation_peak(segment, sr, min_lag_s, max_lag_s)

    passed = hr_valid and regularity < regularity_threshold and autocorr_peak >= min_autocorr

    return {
        "hr": round(hr, 1),
        "n_peaks": len(peaks),
        "regularity": round(regularity, 4),
        "autocorr_peak": round(autocorr_peak, 4),
        "pass": passed,
    }


def ppg_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.5,
    min_autocorr: float = 0.10,
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
    min_autocorr:
        HR lag 범위 autocorrelation 최소값.

    Returns
    -------
    {"hr": float, "regularity": float, "autocorr_peak": float, "pass": bool}
    """
    from scipy.signal import find_peaks

    _fail = {"hr": 0.0, "regularity": 1.0, "autocorr_peak": 0.0, "pass": False}

    if len(segment) < int(sr * 2):
        return _fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        return _fail

    min_distance = int(sr * 60.0 / max_hr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=iqr * 0.5,
        distance=min_distance,
    )

    if len(peaks) < 2:
        return _fail

    pp_intervals = np.diff(peaks) / sr
    pp_mean = float(np.mean(pp_intervals))
    if pp_mean < 1e-6:
        return _fail

    hr = 60.0 / pp_mean
    hr_valid = min_hr <= hr <= max_hr

    pp_std = float(np.std(pp_intervals))
    regularity = pp_std / pp_mean

    # Autocorrelation 주기성 체크
    min_lag_s = 60.0 / max_hr
    max_lag_s = 60.0 / min_hr
    autocorr_peak = _autocorrelation_peak(segment, sr, min_lag_s, max_lag_s)

    passed = hr_valid and regularity < regularity_threshold and autocorr_peak >= min_autocorr

    return {
        "hr": round(hr, 1),
        "regularity": round(regularity, 4),
        "autocorr_peak": round(autocorr_peak, 4),
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

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 0.5:  # CO2 최소 IQR ~0.5 이상이어야 호흡 존재
        return {"resp_rate": 0.0, "pass": False}

    # End-tidal CO2 peaks: 호흡 주기 최소 60/max_rr 초
    min_distance = int(sr * 60.0 / max_rr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=iqr * 0.5,
        distance=min_distance,
    )

    if len(peaks) < 2:
        # 15초 윈도우에서 peak < 2 = 호흡수 추정 불가 (interval 없음)
        # ECG/ABP와 동일 기준: 최소 2개 peak 필요
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

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 0.5:  # AWP 최소 IQR ~0.5 cmH2O
        return {"resp_rate": 0.0, "pass": False}

    min_distance = int(sr * 60.0 / max_rr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=iqr * 0.5,
        distance=min_distance,
    )

    if len(peaks) < 2:
        # 15초 윈도우에서 peak < 2 = 호흡수 추정 불가 (interval 없음)
        # ECG/ABP와 동일 기준: 최소 2개 peak 필요
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


def eeg_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_band_ratio: float = 0.1,
) -> dict:
    """EEG 세그먼트의 주파수 대역 파워 비율 기반 품질 검사.

    정상 EEG는 delta/theta/alpha/beta 대역에 에너지가 분포.
    artifact나 flatline은 특정 대역에 에너지가 편중되거나 전체 에너지가 없음.

    Parameters
    ----------
    segment:
        (n_timesteps,) 1D EEG 배열 (bandpass + resample 후 100Hz 기준).
    sr:
        sampling rate (Hz).
    min_band_ratio:
        정상 대역(1-30Hz) 에너지가 전체의 이 비율 이상이어야 통과.

    Returns
    -------
    {"band_ratio": float, "pass": bool}
    """
    if len(segment) < int(sr * 2):
        return {"band_ratio": 0.0, "pass": False}

    # FFT 기반 파워 스펙트럼
    n = len(segment)
    fft_vals = np.fft.rfft(segment)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    total_power = float(power.sum())
    if total_power < 1e-10:
        return {"band_ratio": 0.0, "pass": False}

    # 정상 EEG 대역: 1-30Hz (delta~beta)
    band_mask = (freqs >= 1.0) & (freqs <= 30.0)
    band_power = float(power[band_mask].sum())
    band_ratio = band_power / total_power

    # 스펙트럼 엔트로피 — 너무 낮으면 단일 주파수(artifact), 너무 높으면 white noise
    power_norm = power / total_power
    power_norm = power_norm[power_norm > 1e-12]
    entropy = float(-np.sum(power_norm * np.log(power_norm)))
    max_entropy = np.log(len(power))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    # 통과 조건:
    # 1. 1-30Hz 대역이 전체의 min_band_ratio 이상
    # 2. 정규화 엔트로피가 0.3~0.95 (너무 규칙적이지도, 완전 white noise도 아닌)
    passed = band_ratio >= min_band_ratio and 0.3 <= norm_entropy <= 0.95

    return {
        "band_ratio": round(band_ratio, 4),
        "entropy": round(norm_entropy, 4),
        "pass": passed,
    }


def _autocorrelation_peak(
    segment: np.ndarray,
    sr: float,
    min_lag_s: float,
    max_lag_s: float,
) -> float:
    """HR 범위 내 lag에서 정규화 autocorrelation의 최대값을 반환한다.

    정상 주기 신호는 해당 lag에서 뚜렷한 peak(>0.3)를 보이고,
    랜덤 노이즈는 빠르게 0으로 수렴하여 peak가 없다.

    Parameters
    ----------
    segment: 1D 신호 (mean-subtracted 권장).
    sr: sampling rate (Hz).
    min_lag_s: 탐색할 최소 lag (초). HR 200bpm → 0.3s.
    max_lag_s: 탐색할 최대 lag (초). HR 30bpm → 2.0s.

    Returns
    -------
    HR 범위 lag에서의 최대 정규화 autocorrelation 값 (0~1).
    """
    x = segment - np.mean(segment)
    n = len(x)
    autocorr_full = np.correlate(x, x, mode="full")
    # 정규화: lag=0의 autocorrelation(= 에너지)으로 나눔
    zero_lag = autocorr_full[n - 1]
    if zero_lag < 1e-10:
        return 0.0
    autocorr = autocorr_full[n - 1:] / zero_lag  # lag >= 0 부분만

    min_lag = max(1, int(min_lag_s * sr))
    max_lag = min(len(autocorr) - 1, int(max_lag_s * sr))
    if min_lag >= max_lag:
        return 0.0

    return float(np.max(autocorr[min_lag:max_lag + 1]))


def _respiratory_band_power(
    segment: np.ndarray,
    sr: float,
    lo: float = 0.1,
    hi: float = 0.5,
) -> float:
    """호흡 대역(0.1~0.5Hz) 파워 비율을 반환한다.

    기계환기 환자는 거의 항상 respiratory variation이 존재.
    정상 CVP: resp_power_ratio > 0.05 정도.

    Returns
    -------
    호흡 대역 파워 / 전체 파워 (0~1).
    """
    n = len(segment)
    fft_vals = np.fft.rfft(segment - np.mean(segment))
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    total_power = float(power.sum())
    if total_power < 1e-10:
        return 0.0

    resp_mask = (freqs >= lo) & (freqs <= hi)
    resp_power = float(power[resp_mask].sum())
    return resp_power / total_power


def cvp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.7,
    max_flatline_ratio: float = 0.3,
    min_autocorr: float = 0.15,
) -> dict:
    """CVP 세그먼트의 정맥파(a/c/v wave) 기반 품질 검사.

    CVP는 저압 정맥 파형으로 ABP보다 진폭이 작고 불규칙하다.
    심박과 동기화된 a/c/v wave가 존재하며, 호흡에 의한 저주파 변동이 있다.

    검증 항목:
        1. Flatline 체크 (센서 분리)
        2. Peak detection (a/c/v wave) + HR/regularity
        3. Autocorrelation 주기성 (랜덤 노이즈 배제)
        4. Respiratory variation (호흡 저주파 변동 존재 여부, 정보 제공용)

    Parameters
    ----------
    segment:
        (n_timesteps,) 1D CVP 배열 (lowpass + resample 후 100Hz 기준).
    sr:
        sampling rate (Hz).
    min_hr, max_hr:
        정상 맥박수 범위 (bpm).
    regularity_threshold:
        peak-to-peak interval 변이계수 상한. ABP(0.5)보다 관대하게 0.7.
    max_flatline_ratio:
        flatline 비율 상한. 초과하면 센서 분리로 판단.
    min_autocorr:
        HR lag 범위 autocorrelation 최소값. 미만이면 주기성 없음(노이즈).

    Returns
    -------
    dict with keys: hr, n_peaks, regularity, flatline_ratio, autocorr_peak,
                    resp_power_ratio, pass.
    """
    from scipy.signal import find_peaks

    _fail = {
        "hr": 0.0, "n_peaks": 0, "regularity": 1.0,
        "flatline_ratio": 1.0, "autocorr_peak": 0.0,
        "resp_power_ratio": 0.0, "pass": False,
    }

    if len(segment) < int(sr * 2):
        return _fail

    # 1. Flatline 체크: 센서 분리 감지
    diffs = np.diff(segment)
    flatline_ratio = float(np.sum(np.abs(diffs) < 1e-4)) / max(len(diffs), 1)
    if flatline_ratio >= max_flatline_ratio:
        _fail["flatline_ratio"] = round(flatline_ratio, 4)
        return _fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 0.1:  # CVP 진폭이 작으므로 ABP보다 낮은 IQR 기준
        _fail["flatline_ratio"] = round(flatline_ratio, 4)
        return _fail

    # 2. Peak detection: a/c/v wave
    min_distance = int(sr * 60.0 / max_hr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=iqr * 0.3,
        distance=min_distance,
    )

    if len(peaks) < 2:
        _fail["flatline_ratio"] = round(flatline_ratio, 4)
        _fail["n_peaks"] = len(peaks)
        return _fail

    pp_intervals = np.diff(peaks) / sr
    pp_mean = float(np.mean(pp_intervals))
    if pp_mean < 1e-6:
        _fail["flatline_ratio"] = round(flatline_ratio, 4)
        _fail["n_peaks"] = len(peaks)
        return _fail

    hr = 60.0 / pp_mean
    hr_valid = min_hr <= hr <= max_hr

    pp_std = float(np.std(pp_intervals))
    regularity = pp_std / pp_mean

    # 3. Autocorrelation 주기성 체크
    #    HR 30~200bpm → lag 0.3~2.0초
    min_lag_s = 60.0 / max_hr  # 0.3s at 200bpm
    max_lag_s = 60.0 / min_hr  # 2.0s at 30bpm
    autocorr_peak = _autocorrelation_peak(segment, sr, min_lag_s, max_lag_s)

    # 4. Respiratory variation (정보 제공 + 보조 지표)
    resp_power_ratio = _respiratory_band_power(segment, sr)

    # 통과 조건: peak + HR + regularity + autocorrelation 주기성
    passed = (
        hr_valid
        and regularity < regularity_threshold
        and autocorr_peak >= min_autocorr
    )

    return {
        "hr": round(hr, 1),
        "n_peaks": len(peaks),
        "regularity": round(regularity, 4),
        "flatline_ratio": round(flatline_ratio, 4),
        "autocorr_peak": round(autocorr_peak, 4),
        "resp_power_ratio": round(resp_power_ratio, 4),
        "pass": passed,
    }


# ── Dispatcher: signal type → domain check function ──────────

DOMAIN_QUALITY_CHECKS: dict[str, callable] = {
    "ecg": ecg_quality_check,
    "eeg": eeg_quality_check,
    "abp": abp_quality_check,
    "ppg": ppg_quality_check,
    "cvp": cvp_quality_check,
    "co2": co2_quality_check,
    "awp": awp_quality_check,
}


def domain_quality_check(stype_key: str, segment: np.ndarray, sr: float = 100.0) -> dict:
    """Signal type에 해당하는 domain-specific 품질 검사를 실행한다.

    Parameters
    ----------
    stype_key:
        신호 타입 키 ("ecg", "eeg", "abp", "ppg", "cvp", "co2", "awp").
        미등록 타입은 항상 {"pass": True}를 반환한다.
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
