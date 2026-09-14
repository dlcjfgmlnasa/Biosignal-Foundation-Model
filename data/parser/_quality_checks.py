"""신호별 Domain-specific 품질 검사.

생체신호(ECG, ABP, PPG, CVP, CO2, AWP, PAP, ICP) 각각의 생리학적 특성에 맞는
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
    autocorr = autocorr_full[n - 1 :] / zero_lag

    min_lag = max(1, int(min_lag_s * sr))
    max_lag = min(len(autocorr) - 1, int(max_lag_s * sr))
    if min_lag >= max_lag:
        return 0.0

    return float(np.max(autocorr[min_lag : max_lag + 1]))


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

    _fail = {
        "hr": 0.0,
        "hr_valid": False,
        "n_peaks": 0,
        "regularity": 1.0,
        "autocorr_peak": 0.0,
        "pass": False,
    }

    if len(segment) < int(sr * 2):
        return _fail

    q75, q25 = np.percentile(segment, [75, 25])
    iqr = q75 - q25
    if iqr < 1e-6:
        return _fail

    # ── R-peak 검출 (2026-08-22 수정) ────────────────────────
    # 예전에는 필터를 거친 원 파형에 곧장 find_peaks 를 걸었다. 그러면 **T파가
    # R파로 같이 세어져 심박수가 정확히 2배로 나온다.**
    #
    #   실측: 모니터 자체 HR 트랙(Intellivue/ECG_HR)과 1,064쌍 대조 결과
    #     파서 추정 중앙 141.9 bpm  vs  모니터 중앙 71.0 bpm
    #     비율 중앙 2.03배, **94.6%가 ≈2배 구간**에 몰림
    #
    #   파급이 셋이다.
    #     1. regularity(=RR 변동계수)가 QRS-T-QRS-T 로 짧고-길고 번갈아 나와
    #        구조적으로 부풀고, 리듬과 무관하게 domain check 를 탈락시킨다.
    #     2. max_hr=200 이 실질적으로 100 bpm 제한이 된다 — 실제로 탈락 창의
    #        모니터 HR 중앙이 106, 통과 창은 71이었다(빈맥이 선택적으로 잘림).
    #     3. 저장된 hr 메타데이터가 전부 2배라 품질 리포트를 믿을 수 없다.
    #
    # 고치는 방법: **T파는 저주파(1~3Hz), QRS는 고주파(10~25Hz)** 라는 성질을 쓴다.
    # 5~15Hz 대역으로 강조한 뒤 제곱·이동평균으로 포락선을 만들면(Pan-Tompkins의
    # 축약형) QRS마다 봉우리가 하나씩 생기고 T파는 눌린다. 검출에만 쓰고 저장되는
    # 파형은 건드리지 않는다.
    nyq = sr / 2.0
    try:
        det = _bandpass_for_peaks(segment, 5.0, min(15.0, nyq - 1.0), sr)
    except Exception:  # noqa: BLE001
        det = segment
    env = np.convolve(
        np.asarray(det, dtype=np.float64) ** 2,
        np.ones(max(1, int(0.08 * sr))) / max(1, int(0.08 * sr)),
        mode="same",
    )
    # 문턱은 **IQR이 아니라 봉우리 높이 기준**이다. 제곱 포락선은 QRS 에너지가
    # 압도적이라 IQR이 바닥값에 눌리고, IQR 기준으로 잡으면 T파 잔재가 그대로
    # 걸러지지 않는다(1차 수정에서 중복계수가 94.6%→51.9%로만 줄었던 이유).
    e_hi = float(np.percentile(env, 99))
    e_base = float(np.median(env))
    span = e_hi - e_base
    if span < 1e-12:
        return _fail
    # 불응기 250ms — 생리적으로 연속 QRS가 이보다 가까울 수 없다(최대 240 bpm).
    # 예전 값 0.24초는 T파(R 이후 0.2~0.4초)를 배제하지 못했다.
    min_distance = max(1, int(0.25 * sr))

    peaks, properties = find_peaks(
        env,
        height=e_base + span * 0.25,
        prominence=span * 0.25,
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
    # 주기성도 포락선에서 잰다 — 원 파형은 T파 때문에 절반 lag에서도 봉우리가 선다.
    autocorr_peak = _autocorrelation_peak(env, sr, min_lag_s, max_lag_s)

    passed = (
        hr_valid
        and regularity < regularity_threshold
        and autocorr_peak >= min_autocorr
    )

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
    min_hr: float = 25.0,
    max_hr: float = 220.0,
    regularity_threshold: float = 0.8,
    min_autocorr: float = 0.05,
) -> dict:
    """ABP 세그먼트의 pulse peak regularity 기반 품질 검사.

    ⚠️ 임계 완화 (2026-08-22) — **부정맥은 아티팩트가 아니다.**

    regularity(peak 간격의 변동계수) 0.5는 규칙적인 리듬을 전제한 값이라
    심방세동·빈번한 조기박동을 그대로 탈락시킨다. 서맥·빈맥도 min_hr 30 /
    max_hr 200 밖으로 나가면 창이 통째로 버려진다. 이들은 급성 사건
    다운스트림에서 오히려 신호다.

      regularity_threshold 0.5 → 0.8   (부정맥 허용)
      min_hr 30 → 25, max_hr 200 → 220 (심한 서맥·빈맥 허용)
      min_autocorr 0.10 → 0.05         (주기성 약한 구간 허용)

    라인 분리·완전 감쇠는 이 검사가 아니라 진폭·평탄선·포화 축에서 잡힌다
    (peak가 2개 미만이거나 IQR이 0이면 여기서도 여전히 탈락한다).

    ⚠️ **앞으로 파싱하는 데이터에만 적용된다.** 기존 processed/ 산출물은
    옛 기준으로 걸러진 상태다.
    """
    from scipy.signal import find_peaks

    _fail = {
        "hr": 0.0,
        "n_peaks": 0,
        "regularity": 1.0,
        "autocorr_peak": 0.0,
        "pass": False,
    }

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

    passed = (
        hr_valid
        and regularity < regularity_threshold
        and autocorr_peak >= min_autocorr
    )

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

    passed = (
        hr_valid
        and regularity < regularity_threshold
        and autocorr_peak >= min_autocorr
    )

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


# ── CVP ───────────────────────────────────────────────────────


def cvp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,  # 자기상관 lag 범위에만 쓴다(판정에는 안 쓴다)
    max_hr: float = 200.0,
    regularity_threshold: float = 1.0,  # a/c/v 혼입 → 간격이 고르지 않은 게 정상
    max_flatline_ratio: float = 0.3,
    min_autocorr: float = 0.15,
) -> dict:
    """CVP 세그먼트의 정맥파(a/c/v wave) 기반 품질 검사.

    ⚠️ 2026-08-22 수정 — **정맥파에서 심박수를 세지 않는다.**

    예전에는 ``find_peaks``로 센 봉우리 간격을 그대로 심박수로 환산하고
    ``min_hr``/``max_hr`` 범위로 창을 버렸다. 그런데 정맥파는 **심주기 한 번에
    봉우리가 여러 개**다 — a파(심방 수축), c파(삼첨판 융기), v파(심방 충만).
    동맥압처럼 박동당 봉우리 하나가 아니다.

      실측: 모니터 ECG_HR과 701쌍 대조 결과 비율 중앙 **1.66**
            (≈1배 23.3% / 1.2~1.7배 26.4% / **≈2배 44.7%**)
            — a·c·v 중 몇 개가 잡히느냐가 구간마다 달라 배수가 흩어진다.

    ECG의 T파 중복계수(정확히 2배)와 달리 이건 **검출 버그가 아니라 생리다.**
    문제는 그것을 "심박수"라 부르고 범위로 판정한 것이다. 그래서 심박수 판정을
    빼고, 이 함수의 본래 목적인 **"쓸 만한 파형인가"** 만 본다:

      평탄선 / 진폭(IQR) / 봉우리 존재 / 주기성(자기상관) / 간격 규칙성

    자기상관 lag 범위는 심박수 범위를 그대로 쓴다 — 정맥파도 심주기로 반복하므로
    주 주기는 심박에 선다. 봉우리 간격 규칙성은 a/c/v가 섞여 들어오므로 임계를
    느슨하게(0.7 → 1.0) 둔다.

    ``hr`` 필드는 **``peak_rate``로 이름을 바꿨다** — 심박수로 오해하면 안 된다.
    하위 호환을 위해 ``hr`` 키도 같은 값으로 남기되 의미는 "봉우리 빈도"다.
    """
    from scipy.signal import find_peaks

    _fail = {
        "hr": 0.0,
        "peak_rate": 0.0,
        "n_peaks": 0,
        "regularity": 1.0,
        "flatline_ratio": 1.0,
        "autocorr_peak": 0.0,
        "pass": False,
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

    # 봉우리 빈도. **심박수가 아니다** — a/c/v가 섞여 들어온다.
    peak_rate = 60.0 / pp_mean

    pp_std = float(np.std(pp_intervals))
    regularity = pp_std / pp_mean

    # 주기성은 심박수 범위 lag에서 본다 — 정맥파도 심주기로 반복한다.
    min_lag_s = 60.0 / max_hr
    max_lag_s = 60.0 / min_hr
    autocorr_peak = _autocorrelation_peak(segment, sr, min_lag_s, max_lag_s)

    # 심박수 범위 판정을 뺐다(위 docstring). 남은 것은 파형 자체의 쓸모다.
    passed = (
        regularity < regularity_threshold and autocorr_peak >= min_autocorr
    )

    return {
        "hr": round(peak_rate, 1),  # 하위 호환 — 의미는 peak_rate다
        "peak_rate": round(peak_rate, 1),
        "n_peaks": len(peaks),
        "regularity": round(regularity, 4),
        "flatline_ratio": round(flatline_ratio, 4),
        "autocorr_peak": round(autocorr_peak, 4),
        "pass": passed,
    }


# ── PAP ──────────────────────────────────────────────────────


def pap_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.7,
    min_autocorr: float = 0.10,
) -> dict:
    """PAP 세그먼트의 pulse peak regularity 기반 품질 검사."""
    return abp_quality_check(
        segment, sr, min_hr, max_hr, regularity_threshold, min_autocorr
    )


# ── ICP ──────────────────────────────────────────────────────


def icp_quality_check(
    segment: np.ndarray,
    sr: float = 100.0,
    min_hr: float = 30.0,
    max_hr: float = 200.0,
    regularity_threshold: float = 0.7,
    max_flatline_ratio: float = 0.3,
    min_autocorr: float = 0.15,
) -> dict:
    """ICP 세그먼트의 맥동 기반 품질 검사. CVP와 동일 로직."""
    return cvp_quality_check(
        segment,
        sr,
        min_hr,
        max_hr,
        regularity_threshold,
        max_flatline_ratio,
        min_autocorr,
    )


# ── Dispatcher ────────────────────────────────────────────────


DOMAIN_QUALITY_CHECKS: dict[str, callable] = {
    "ecg": ecg_quality_check,
    "abp": abp_quality_check,
    "ppg": ppg_quality_check,
    "cvp": cvp_quality_check,
    "co2": co2_quality_check,
    "awp": awp_quality_check,
    "pap": pap_quality_check,
    "icp": icp_quality_check,
}


def domain_quality_check(
    stype_key: str, segment: np.ndarray, sr: float = 100.0
) -> dict:
    """Signal type에 해당하는 domain-specific 품질 검사를 실행한다.

    Parameters
    ----------
    stype_key:
        신호 타입 키 ("ecg", "abp", "ppg", "cvp", "co2", "awp", "pap", "icp").
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
