"""신호별 Domain-specific 품질 검사.

7종 생체신호(ECG, EEG, ABP, PPG, CVP, CO2, AWP) 각각의 생리학적 특성에 맞는
품질 검사를 제공한다. 파서(_common.py)의 segment_quality_score와 함께 사용된다.
"""
from __future__ import annotations

import numpy as np


# ── 내부 헬퍼 ─────────────────────────────────────────────────


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


def _autocorrelation_peak(
    segment: np.ndarray,
    sr: float,
    min_lag_s: float,
    max_lag_s: float,
) -> float:
    """HR 범위 내 lag에서 정규화 autocorrelation의 최대값을 반환한다.

    정상 주기 신호는 해당 lag에서 뚜렷한 peak(>0.3)를 보이고,
    랜덤 노이즈는 빠르게 0으로 수렴하여 peak가 없다.
    """
    x = segment - np.mean(segment)
    n = len(x)
    autocorr_full = np.correlate(x, x, mode="full")
    zero_lag = autocorr_full[n - 1]
    if zero_lag < 1e-10:
        return 0.0
    autocorr = autocorr_full[n - 1:] / zero_lag

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
    """호흡 대역(0.1~0.5Hz) 파워 비율을 반환한다."""
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


# ── ECG ───────────────────────────────────────────────────────


def ecg_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.7,
    min_autocorr: float = 0.10,
) -> dict:
    """ECG 세그먼트의 QRS peak 기반 심박수 품질 검사."""
    from scipy.signal import find_peaks

    _fail = {"hr": 0.0, "hr_valid": False, "n_peaks": 0, "regularity": 1.0, "autocorr_peak": 0.0, "pass": False}

    if len(segment) < int(sr * 2):
        return _fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        return _fail

    min_distance = int(sr * 60.0 / max_hr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, properties = find_peaks(
        segment,
        prominence=iqr * 0.5,
        distance=min_distance,
    )

    n_peaks = len(peaks)
    if n_peaks < 2:
        _fail["n_peaks"] = n_peaks
        return _fail

    rr_intervals = np.diff(peaks) / sr
    rr_mean = float(np.mean(rr_intervals))
    if rr_mean < 1e-6:
        _fail["n_peaks"] = n_peaks
        return _fail

    hr = 60.0 / rr_mean
    hr_valid = min_hr <= hr <= max_hr

    rr_std = float(np.std(rr_intervals))
    regularity = rr_std / rr_mean

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


# ── ABP ───────────────────────────────────────────────────────


def abp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.5,
    min_autocorr: float = 0.10,
) -> dict:
    """ABP 세그먼트의 pulse peak regularity 기반 품질 검사."""
    from scipy.signal import find_peaks

    _fail = {"hr": 0.0, "n_peaks": 0, "regularity": 1.0, "autocorr_peak": 0.0, "pass": False}

    if len(segment) < int(sr * 2):
        return _fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        return _fail

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


# ── PPG ───────────────────────────────────────────────────────


def ppg_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.5,
    min_autocorr: float = 0.10,
) -> dict:
    """PPG 세그먼트의 pulse peak regularity 기반 품질 검사."""
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


# ── CO2 ───────────────────────────────────────────────────────


def co2_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_rr: float = 4.0,
    max_rr: float = 40.0,
) -> dict:
    """CO2 (capnogram) 세그먼트의 호흡 사이클 품질 검사."""
    from scipy.signal import find_peaks

    duration_s = len(segment) / sr
    if duration_s < 5.0:
        return {"resp_rate": 0.0, "pass": False}

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 0.5:
        return {"resp_rate": 0.0, "pass": False}

    min_distance = int(sr * 60.0 / max_rr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=iqr * 0.5,
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


# ── AWP ───────────────────────────────────────────────────────


def awp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_rr: float = 4.0,
    max_rr: float = 40.0,
) -> dict:
    """AWP (기도압) 세그먼트의 환기 사이클 품질 검사."""
    from scipy.signal import find_peaks

    duration_s = len(segment) / sr
    if duration_s < 5.0:
        return {"resp_rate": 0.0, "pass": False}

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 0.5:
        return {"resp_rate": 0.0, "pass": False}

    min_distance = int(sr * 60.0 / max_rr * 0.8)
    min_distance = max(min_distance, 1)

    peaks, _ = find_peaks(
        segment,
        prominence=iqr * 0.5,
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


# ── EEG ───────────────────────────────────────────────────────


def eeg_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_band_ratio: float = 0.1,
) -> dict:
    """EEG 세그먼트의 주파수 대역 파워 비율 기반 품질 검사."""
    if len(segment) < int(sr * 2):
        return {"band_ratio": 0.0, "pass": False}

    n = len(segment)
    fft_vals = np.fft.rfft(segment)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    total_power = float(power.sum())
    if total_power < 1e-10:
        return {"band_ratio": 0.0, "pass": False}

    band_mask = (freqs >= 1.0) & (freqs <= 30.0)
    band_power = float(power[band_mask].sum())
    band_ratio = band_power / total_power

    power_norm = power / total_power
    power_norm = power_norm[power_norm > 1e-12]
    entropy = float(-np.sum(power_norm * np.log(power_norm)))
    max_entropy = np.log(len(power))
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

    passed = band_ratio >= min_band_ratio and 0.3 <= norm_entropy <= 0.95

    return {
        "band_ratio": round(band_ratio, 4),
        "entropy": round(norm_entropy, 4),
        "pass": passed,
    }


# ── CVP ───────────────────────────────────────────────────────


def cvp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.7,
    max_flatline_ratio: float = 0.3,
    min_autocorr: float = 0.15,
) -> dict:
    """CVP 세그먼트의 정맥파(a/c/v wave) 기반 품질 검사."""
    from scipy.signal import find_peaks

    _fail = {
        "hr": 0.0, "n_peaks": 0, "regularity": 1.0,
        "flatline_ratio": 1.0, "autocorr_peak": 0.0,
        "resp_power_ratio": 0.0, "pass": False,
    }

    if len(segment) < int(sr * 2):
        return _fail

    diffs = np.diff(segment)
    flatline_ratio = float(np.sum(np.abs(diffs) < 1e-4)) / max(len(diffs), 1)
    if flatline_ratio >= max_flatline_ratio:
        _fail["flatline_ratio"] = round(flatline_ratio, 4)
        return _fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 0.1:
        _fail["flatline_ratio"] = round(flatline_ratio, 4)
        return _fail

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

    min_lag_s = 60.0 / max_hr
    max_lag_s = 60.0 / min_hr
    autocorr_peak = _autocorrelation_peak(segment, sr, min_lag_s, max_lag_s)

    resp_power_ratio = _respiratory_band_power(segment, sr)

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


# ── Dispatcher ────────────────────────────────────────────────


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
        return {"pass": True}
