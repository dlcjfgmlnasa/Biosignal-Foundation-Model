# -*- coding:utf-8 -*-
"""Downstream pilot test 데이터 로딩, 라벨링, 분할 유틸리티.

VitalDB 뒤쪽 케이스를 pilot test 전용으로 사용한다.
기존 data/parser/ 파이프라인을 재사용하여 필터링/리샘플링을 수행한다.

사용법:
    from downstream.data_utils import (
        load_pilot_cases, extract_windows, apply_pipeline,
        split_by_subject, create_labeled_dataset_hypotension,
        create_labeled_dataset_bradytachy,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.parser._common import (
    domain_quality_check,
    resample_to_target,
    segment_quality_score,
)
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    TARGET_SR,
    _apply_filter,
    _apply_median_filter,
    _apply_notch_filter,
    _apply_range_check,
    _detect_electrocautery,
    _detect_motion_artifact,
    _extract_nan_free_segments,
)


# ── 트랙 선택 (signal type -> 대표 트랙명) ─────────────────────


# signal_type_key -> 우선순위 트랙 리스트 (첫 번째 존재하는 트랙 사용)
PREFERRED_TRACKS: dict[str, list[str]] = {
    "ecg": ["SNUADC/ECG_II", "SNUADC/ECG_V5", "Solar8000/ECG_II"],
    "abp": ["SNUADC/ART", "SNUADC/FEM"],
    "ppg": ["SNUADC/PLETH", "Solar8000/PLETH"],
    "cvp": ["SNUADC/CVP"],
    "co2": ["Primus/CO2", "Solar8000/CO2"],
    "awp": ["Primus/AWP", "Solar8000/AWP"],
}

# 트랙별 native sampling rate (Hz)
NATIVE_SR: dict[str, float] = {
    "ecg": 500.0,
    "abp": 500.0,
    "ppg": 500.0,
    "cvp": 500.0,
    "co2": 62.5,
    "awp": 62.5,
}


# ── 데이터 구조 ────────────────────────────────────────────────


@dataclass
class CaseData:
    """단일 VitalDB 케이스에서 로드된 데이터."""

    case_id: int
    tracks: dict[str, np.ndarray] = field(default_factory=dict)
    # tracks: {signal_type_key: (n_timesteps,) resampled 100Hz array}


@dataclass
class Window:
    """단일 시간 윈도우."""

    signal: np.ndarray  # (win_samples,) at TARGET_SR
    signal_type: str  # "ecg", "abp", etc.
    case_id: int
    win_start: int  # sample index in resampled signal
    quality_passed: bool


@dataclass
class LabeledWindow:
    """라벨이 부여된 윈도우."""

    signal: np.ndarray  # (win_samples,)
    signal_type: str
    case_id: int
    label: int  # task-specific label
    label_value: float  # 연속값 (MAP, HR 등)


# ── Pilot 케이스 로딩 ──────────────────────────────────────────


def load_pilot_cases(
    n_cases: int = 50,
    offset_from_end: int = 0,
    signal_types: list[str] | None = None,
) -> list[CaseData]:
    """VitalDB 뒤쪽 케이스 N개를 로드하여 파이프라인 필터링 후 반환한다.

    Parameters
    ----------
    n_cases:
        로드할 케이스 수.
    offset_from_end:
        VitalDB 전체 케이스 목록의 뒤에서 offset만큼 건너뛴 위치부터 시작.
        예: offset_from_end=0이면 마지막 n_cases개, =100이면 뒤에서 100번째부터.
    signal_types:
        로드할 signal type 키 목록. None이면 ["ecg", "abp", "ppg", "cvp"].

    Returns
    -------
    list[CaseData]
    """
    import vitaldb

    if signal_types is None:
        signal_types = ["ecg", "abp", "ppg", "cvp"]

    # ECG가 있는 케이스를 기준으로 전체 목록 구성
    all_cases = sorted(vitaldb.find_cases(["SNUADC/ECG_II"]))
    total = len(all_cases)

    start_idx = max(0, total - offset_from_end - n_cases)
    end_idx = max(0, total - offset_from_end)
    pilot_ids = all_cases[start_idx:end_idx]

    print(f"Pilot cases: {len(pilot_ids)} (IDs {pilot_ids[0]}~{pilot_ids[-1]})")

    results: list[CaseData] = []
    for case_id in pilot_ids:
        case_data = _load_single_case(case_id, signal_types)
        if case_data.tracks:
            results.append(case_data)

    print(f"Loaded {len(results)}/{len(pilot_ids)} cases with data")
    return results


def _load_single_case(
    case_id: int,
    signal_types: list[str],
) -> CaseData:
    """단일 케이스의 지정된 signal type들을 로드하고 파이프라인 적용."""
    import vitaldb

    case = CaseData(case_id=case_id)

    for stype_key in signal_types:
        if stype_key not in PREFERRED_TRACKS:
            continue

        native_sr = NATIVE_SR.get(stype_key, 500.0)
        cfg = SIGNAL_CONFIGS.get(stype_key)
        if cfg is None:
            continue

        # 우선순위 트랙 중 첫 번째 유효한 것 사용
        data = None
        for track_name in PREFERRED_TRACKS[stype_key]:
            try:
                raw = vitaldb.load_case(case_id, [track_name], interval=1.0 / native_sr)
                if raw is not None and len(raw) > 0:
                    col = raw[:, 0].flatten()
                    if (~np.isnan(col)).sum() > int(60 * native_sr):  # 최소 60초 유효
                        data = col
                        break
            except Exception:
                continue

        if data is None:
            continue

        # 파이프라인 적용
        processed = _apply_full_pipeline(data, stype_key, cfg, native_sr)
        if processed is not None and len(processed) >= int(10 * TARGET_SR):
            case.tracks[stype_key] = processed

    return case


def _apply_full_pipeline(
    data: np.ndarray,
    stype_key: str,
    cfg,
    native_sr: float,
) -> np.ndarray | None:
    """단일 트랙에 전체 전처리 파이프라인을 적용한다.

    Range check -> Spike detection -> Motion artifact (PPG) ->
    NaN segment extraction -> Median -> Notch -> Filter -> Resample.
    가장 긴 유효 세그먼트를 반환한다.
    """
    # Step 1: Range check
    if cfg.valid_range is not None:
        data, _ = _apply_range_check(data, cfg.valid_range)

    # Step 2: Spike detection
    if cfg.spike_detection:
        data, _ = _detect_electrocautery(
            data, native_sr, threshold_std=cfg.spike_threshold_std
        )

    # Step 2b: PPG motion artifact
    if stype_key == "ppg":
        data, _ = _detect_motion_artifact(data, native_sr)

    # Step 3: NaN-free segments (가장 긴 것 사용)
    min_samples = int(60.0 * native_sr)
    segments = _extract_nan_free_segments(data, min_samples)
    if not segments:
        return None

    # 가장 긴 세그먼트 선택
    segment = max(segments, key=len)

    # Step 4: Median -> Notch -> Filter
    if cfg.median_kernel > 0:
        segment = _apply_median_filter(segment, kernel_size=cfg.median_kernel)
    if cfg.notch_freq is not None:
        segment = _apply_notch_filter(segment, freq=cfg.notch_freq, sr=native_sr)
    segment = _apply_filter(segment, cfg, native_sr)

    # Step 5: Resample to TARGET_SR
    if native_sr != TARGET_SR:
        segment = resample_to_target(segment, orig_sr=native_sr, target_sr=TARGET_SR)

    return segment


# ── 윈도우 추출 ────────────────────────────────────────────────


def extract_windows(
    case_data: CaseData,
    signal_type: str,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    sr: float = TARGET_SR,
    quality_check: bool = True,
) -> list[Window]:
    """CaseData에서 슬라이딩 윈도우를 추출한다.

    Parameters
    ----------
    case_data:
        로드된 케이스 데이터.
    signal_type:
        추출할 signal type 키.
    window_sec:
        윈도우 길이 (초).
    stride_sec:
        슬라이드 보폭 (초).
    sr:
        sampling rate (Hz). 기본 100.0.
    quality_check:
        True이면 각 윈도우에 domain quality check를 수행.

    Returns
    -------
    list[Window]
    """
    if signal_type not in case_data.tracks:
        return []

    signal = case_data.tracks[signal_type]
    win_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)

    windows: list[Window] = []
    cfg = SIGNAL_CONFIGS.get(signal_type)

    for start in range(0, len(signal) - win_samples + 1, stride_samples):
        win = signal[start : start + win_samples]

        passed = True
        if quality_check and cfg is not None:
            basic = segment_quality_score(
                win,
                max_flatline_ratio=cfg.max_flatline_ratio,
                max_clip_ratio=cfg.max_clip_ratio,
                max_high_freq_ratio=cfg.max_high_freq_ratio,
                min_amplitude=cfg.min_amplitude,
                max_amplitude=cfg.max_amplitude,
                min_high_freq_ratio=cfg.min_high_freq_ratio,
            )
            if basic["pass"]:
                domain = domain_quality_check(signal_type, win, sr=sr)
                passed = domain["pass"]
            else:
                passed = False

        windows.append(
            Window(
                signal=win,
                signal_type=signal_type,
                case_id=case_data.case_id,
                win_start=start,
                quality_passed=passed,
            )
        )

    return windows


def apply_pipeline(
    windows: list[Window],
    keep_only_passed: bool = True,
) -> list[Window]:
    """윈도우 리스트에서 품질 통과한 것만 필터링한다.

    Parameters
    ----------
    windows:
        extract_windows()의 출력.
    keep_only_passed:
        True이면 quality_passed=True인 윈도우만 반환.
    """
    if keep_only_passed:
        return [w for w in windows if w.quality_passed]
    return windows


# ── Subject 단위 분할 ──────────────────────────────────────────


def split_by_subject(
    cases: list[CaseData],
    train_ratio: float = 0.7,
    seed: int = 42,
) -> tuple[list[CaseData], list[CaseData]]:
    """Subject(case) 단위로 train/test를 분할한다.

    같은 case_id의 모든 데이터가 같은 split에 배정된다.

    Parameters
    ----------
    cases:
        CaseData 리스트.
    train_ratio:
        train 비율 (0~1).
    seed:
        랜덤 시드.

    Returns
    -------
    (train_cases, test_cases)
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(len(cases))
    rng.shuffle(indices)

    n_train = max(1, int(len(cases) * train_ratio))
    train_idx = set(indices[:n_train].tolist())

    train_cases = [cases[i] for i in range(len(cases)) if i in train_idx]
    test_cases = [cases[i] for i in range(len(cases)) if i not in train_idx]

    return train_cases, test_cases


# ── Hypotension 라벨링 ────────────────────────────────────────


def create_labeled_dataset_hypotension(
    windows: list[Window],
    map_threshold: float = 65.0,
) -> list[LabeledWindow]:
    """ABP 윈도우에서 MAP을 계산하고 저혈압 라벨을 부여한다.

    MAP 계산: 윈도우 전체의 평균값으로 근사 (ABP 원본 스케일 mmHg).
    ABP 파형의 평균은 MAP에 가까운 근사치이다.

    Parameters
    ----------
    windows:
        ABP signal type의 Window 리스트.
    map_threshold:
        MAP이 이 미만이면 hypotension (label=1).

    Returns
    -------
    list[LabeledWindow] -- label: 0=normal, 1=hypotension.
    """
    labeled: list[LabeledWindow] = []

    for w in windows:
        if w.signal_type != "abp":
            continue

        # MAP 근사: ABP 파형의 평균
        map_value = float(np.mean(w.signal))
        label = 1 if map_value < map_threshold else 0

        labeled.append(
            LabeledWindow(
                signal=w.signal,
                signal_type=w.signal_type,
                case_id=w.case_id,
                label=label,
                label_value=map_value,
            )
        )

    return labeled


# ── Bradycardia/Tachycardia 라벨링 ────────────────────────────


def create_labeled_dataset_bradytachy(
    windows: list[Window],
    brady_threshold: float = 60.0,
    tachy_threshold: float = 100.0,
    sr: float = TARGET_SR,
) -> list[LabeledWindow]:
    """ECG 윈도우에서 HR을 추출하고 서맥/정상/빈맥 라벨을 부여한다.

    HR 추출: R-peak detection (find_peaks) -> R-R interval -> HR(bpm).

    Parameters
    ----------
    windows:
        ECG signal type의 Window 리스트.
    brady_threshold:
        HR < 이 값이면 서맥 (label=0).
    tachy_threshold:
        HR > 이 값이면 빈맥 (label=2).
    sr:
        sampling rate (Hz).

    Returns
    -------
    list[LabeledWindow] -- label: 0=bradycardia, 1=normal, 2=tachycardia.
    """
    from scipy.signal import find_peaks

    labeled: list[LabeledWindow] = []

    for w in windows:
        if w.signal_type != "ecg":
            continue

        segment = w.signal
        if len(segment) < int(sr * 2):
            continue

        # R-peak detection
        q75, q25 = np.percentile(segment, [75, 25])
        iqr = q75 - q25
        if iqr < 1e-6:
            continue

        min_distance = max(1, int(sr * 0.3))
        peaks, _ = find_peaks(segment, prominence=iqr * 0.5, distance=min_distance)

        if len(peaks) < 2:
            continue

        # HR 계산
        rr_intervals = np.diff(peaks) / sr
        rr_mean = float(np.mean(rr_intervals))
        if rr_mean < 1e-6:
            continue

        hr = 60.0 / rr_mean

        # 3-class 라벨링
        if hr < brady_threshold:
            label = 0  # bradycardia
        elif hr > tachy_threshold:
            label = 2  # tachycardia
        else:
            label = 1  # normal

        labeled.append(
            LabeledWindow(
                signal=segment,
                signal_type=w.signal_type,
                case_id=w.case_id,
                label=label,
                label_value=hr,
            )
        )

    return labeled
