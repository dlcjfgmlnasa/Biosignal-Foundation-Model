# -*- coding:utf-8 -*-
"""VitalDB (.vital) → datasets/processed/ 변환 스크립트.

VitalDB(https://vitaldb.net)의 수술 중 모니터링 데이터를 파싱하여
zarr 압축 포맷으로 저장한다. 신호별 physiological range check와
bandpass filtering을 적용하고, 모든 유효 세그먼트를 개별 zarr로 저장한다.

신호 타입 매핑:
  ECG(0), ABP(1), EEG(2), PPG(3), CVP(4), CO2(5), AWP(6)

사용법:
  # 트랙 탐색
  python -m data.parser.vitaldb --raw datasets/raw/vitaldb --discover --max-files 3

  # 단일 파일 테스트
  python -m data.parser.vitaldb --raw datasets/raw/vitaldb --out datasets/processed --max-files 1

  # 전체 파싱
  python -m data.parser.vitaldb --raw datasets/raw/vitaldb --out datasets/processed

  # 병렬 파싱 (4 workers)
  python -m data.parser.vitaldb --raw datasets/raw/vitaldb --out datasets/processed --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data.parser._common import (
    domain_quality_check,
    resample_to_target,
    save_recording_h5,
    save_recording_zarr,
    segment_quality_score,
)

# 목표 sampling rate (Hz)
TARGET_SR: float = 100.0


# ── 신호별 전처리 설정 ──────────────────────────────────────────


@dataclass
class SignalConfig:
    """신호 타입별 전처리 파라미터.

    품질 기준 (segment_quality_score 연동):
        max_flatline_ratio: 연속 동일 값 비율 상한 (초과 시 불량)
        max_clip_ratio: min/max 고정 비율 상한
        max_high_freq_ratio: 고주파 에너지 비율 상한 (신호별 특성에 맞게 설정)

    high_freq_ratio 근거 (실측 + 시각 검증, 2026-03-26):
        ECG: QRS spike → 정상도 ~0.4, P99=1.97 → 3.0
        ABP: 매우 부드러운 파형, P99=0.03 → 0.5, valid_range 하한 20mmHg
        EEG: alpha/beta 고주파 → 합성 clean=0.38 → 2.0
        PPG: 부드러운 파형, hf>0.05부터 노이즈 → 0.05 (시각 검증)
        CVP: 저주파, 합성 clean=0.0004 → 0.5
        CO2: 느린 capnogram, flatline 구간 → hf=1.0, flatline=0.3
        AWP: P95=0.54, 1.0 이상 스파이크 → 1.0
    """
    valid_range: tuple[float, float] | None  # (min, max) — None이면 range check 안 함
    filter_type: str = "none"                # "bandpass" | "lowpass" | "none"
    filter_freq: tuple[float, float] | None = None  # bandpass=(lo,hi), lowpass=(hi,) → (0, hi)로 저장
    max_flatline_ratio: float = 0.5          # 50% 이상 flat이면 불량
    max_clip_ratio: float = 0.1              # 10% 이상 clipping이면 불량
    max_high_freq_ratio: float = 2.0         # 기본값; 신호별로 아래에서 재정의
    min_amplitude: float = 0.0               # 최소 peak-to-peak 진폭 (0=비활성)
    max_amplitude: float = 0.0               # 최대 peak-to-peak 진폭 (0=비활성, EEG artifact용)
    min_high_freq_ratio: float = 0.0         # 최소 hf ratio (0=비활성, ECG용: QRS 없으면 불량)
    notch_freq: float | None = None          # 50 또는 60Hz notch filter (None=비활성)
    spike_detection: bool = False            # 스파이크/아티팩트 검출 적용 여부
    spike_threshold_std: float = 10.0        # spike 검출 threshold (MAD 배수)
    median_kernel: int = 0                   # median filter kernel size (0=비활성, 홀수만)
    quality_window_s: float = 5.0            # 품질 검사 윈도우 크기 (초). 호흡 신호는 10초 권장


SIGNAL_CONFIGS: dict[str, SignalConfig] = {
    # ECG/EEG: bandpass — baseline wander(저주파) + 고주파 노이즈 동시 제거
    #   notch 60Hz (VitalDB=한국, 60Hz 전원), spike detection 활성
    "ecg": SignalConfig(valid_range=(-5.0, 5.0),       filter_type="bandpass", filter_freq=(0.5, 40.0),  max_high_freq_ratio=1.0, min_amplitude=0.3, min_high_freq_ratio=0.05, notch_freq=60.0, spike_detection=True, spike_threshold_std=10.0),
    "eeg": SignalConfig(valid_range=(-500.0, 500.0),   filter_type="bandpass", filter_freq=(0.5, 45.0),  max_high_freq_ratio=2.0, min_amplitude=2.0, max_amplitude=500.0, notch_freq=60.0, spike_detection=True, spike_threshold_std=10.0),
    # ABP/PPG/CVP: lowpass — DC(절대값) 보존, 고주파 노이즈만 제거
    #   PPG/ABP: median filter로 임펄스 노이즈 제거, spike detection 활성
    "abp": SignalConfig(valid_range=(20.0, 300.0),     filter_type="lowpass",  filter_freq=(0.0, 15.0),  max_high_freq_ratio=0.5, max_flatline_ratio=0.3, min_amplitude=10.0, spike_detection=True, spike_threshold_std=6.0, median_kernel=5),
    "ppg": SignalConfig(valid_range=(0.0, 2000.0),     filter_type="lowpass",  filter_freq=(0.0, 8.0),   max_high_freq_ratio=0.05, max_flatline_ratio=0.3, min_amplitude=5.0, notch_freq=60.0, spike_detection=True, spike_threshold_std=6.0, median_kernel=5),
    "cvp": SignalConfig(valid_range=(-5.0, 40.0),      filter_type="lowpass",  filter_freq=(0.0, 10.0),  max_high_freq_ratio=0.5, spike_detection=True, spike_threshold_std=8.0),
    # CO2/AWP: lowpass — 느린 호흡 신호, DC 보존
    "co2": SignalConfig(valid_range=(0.0, 100.0),      filter_type="lowpass",  filter_freq=(0.0, 5.0),   max_high_freq_ratio=1.0, max_flatline_ratio=0.3, min_amplitude=5.0, quality_window_s=15.0),
    "awp": SignalConfig(valid_range=(-20.0, 80.0),     filter_type="lowpass",  filter_freq=(0.0, 20.0),  max_high_freq_ratio=1.0, min_amplitude=2.0, quality_window_s=15.0),
}


# ── VitalDB 트랙 매핑 ──────────────────────────────────────────


# VitalDB Waveform 트랙 → (signal_type_key, local_spatial_id)
# 공식: https://vitaldb.net/dataset/ — Hemodynamic Parameters (W=waveform)
# 비공식: 일부 파일에 존재하는 대체 장비 트랙 (같은 신호, 다른 소스)
TRACK_MAP: dict[str, tuple[str, int]] = {
    # ECG (0) — 500Hz, mV
    "SNUADC/ECG_II": ("ecg", 1),    # Lead II (공식)
    "SNUADC/ECG_V5": ("ecg", 2),    # Lead V5 (공식)
    "SNUADC/ECG_I": ("ecg", 0),     # Lead I (비공식)
    "SNUADC/ECG_III": ("ecg", 0),   # Lead III (비공식)
    "Solar8000/ECG_II": ("ecg", 1), # Lead II — Solar8000 대체 (비공식)
    # ABP (1) — 500Hz, mmHg
    "SNUADC/ART": ("abp", 1),       # Radial artery (공식)
    "SNUADC/FEM": ("abp", 2),       # Femoral artery (공식)
    # EEG (2) — BIS 128Hz, μV
    "BIS/EEG1_WAV": ("eeg", 0),     # BIS channel 1 (공식)
    "BIS/EEG2_WAV": ("eeg", 0),     # BIS channel 2 (공식)
    "SNUADC/EEG_BIS": ("eeg", 0),   # BIS EEG (비공식)
    "SNUADC/EEG1": ("eeg", 0),      # EEG ch1 (비공식)
    "SNUADC/EEG2": ("eeg", 0),      # EEG ch2 (비공식)
    # PPG (3) — 500Hz, unitless
    "SNUADC/PLETH": ("ppg", 1),     # Finger (공식)
    "Solar8000/PLETH": ("ppg", 1),  # Finger — Solar8000 대체 (비공식)
    # CVP (4) — 500Hz, mmHg
    "SNUADC/CVP": ("cvp", 0),       # Central venous pressure (공식)
    # CO2 (5) — Primus 62.5Hz, mmHg
    "Primus/CO2": ("co2", 0),       # Capnography (공식)
    "Solar8000/CO2": ("co2", 0),    # Capnography — Solar8000 대체 (비공식)
    # AWP (6) — Primus 62.5Hz, hPa
    "Primus/AWP": ("awp", 0),       # Airway pressure (공식)
    "Solar8000/AWP": ("awp", 0),    # Airway pressure — Solar8000 대체 (비공식)
}

SIGNAL_TYPES: dict[str, int] = {
    "ecg": 0, "abp": 1, "eeg": 2, "ppg": 3, "cvp": 4, "co2": 5, "awp": 6,
}


# ── 전처리 함수 ────────────────────────────────────────────────


def _apply_notch_filter(data: np.ndarray, freq: float, sr: float, Q: float = 30.0) -> np.ndarray:
    """전원 간섭(50/60Hz) 제거를 위한 notch filter (1D)."""
    from scipy.signal import filtfilt, iirnotch
    nyq = sr / 2.0
    if freq >= nyq:
        return data
    b, a = iirnotch(freq / nyq, Q)
    return filtfilt(b, a, data).astype(data.dtype)


def _apply_median_filter(data: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """임펄스 노이즈 제거를 위한 median filter (1D)."""
    from scipy.signal import medfilt
    if kernel_size < 3:
        return data
    if kernel_size % 2 == 0:
        kernel_size += 1
    return medfilt(data, kernel_size=kernel_size).astype(data.dtype)


def _detect_step_change(
    data: np.ndarray, sr: float,
    threshold_std: float = 6.0, blank_ms: float = 300.0,
) -> tuple[np.ndarray, int]:
    """ABP/PPG 계단형 아티팩트 검출: 1차 미분 기반 급격한 level shift를 NaN으로 마킹.

    사각파(센서 탈락/클리핑)처럼 값이 급격히 전환되는 구간을 감지한다.
    기존 spike detection(MAD 기반)은 단일 피크만 잡지만, 이 함수는
    level shift(계단 함수)를 잡는다.
    """
    out = data.copy()
    diff1 = np.abs(np.diff(out, prepend=out[0]))
    med = np.median(diff1)
    mad = np.median(np.abs(diff1 - med)) * 1.4826
    if mad < 1e-10:
        return out, 0
    step_mask = diff1 > (med + threshold_std * mad)
    if not step_mask.any():
        return out, 0
    blank_samples = int(blank_ms / 1000.0 * sr)
    step_idx = np.where(step_mask)[0]
    for idx in step_idx:
        start = max(0, idx - blank_samples)
        end = min(len(out), idx + blank_samples + 1)
        out[start:end] = np.nan
    n_blanked = int(np.isnan(out).sum() - np.isnan(data).sum())
    return out, n_blanked


def _detect_motion_artifact(
    data: np.ndarray, sr: float,
    threshold_std: float = 5.0, blank_ms: float = 200.0,
) -> tuple[np.ndarray, int]:
    """PPG motion artifact 검출: 2차 미분 기반 baseline shift를 NaN으로 마킹."""
    out = data.copy()
    diff2 = np.abs(np.diff(out, n=2, prepend=[out[0], out[0]]))
    med = np.median(diff2)
    mad = np.median(np.abs(diff2 - med)) * 1.4826
    if mad < 1e-10:
        return out, 0
    spike_mask = diff2 > (med + threshold_std * mad)
    if not spike_mask.any():
        return out, 0
    blank_samples = int(blank_ms / 1000.0 * sr)
    spike_idx = np.where(spike_mask)[0]
    for idx in spike_idx:
        start = max(0, idx - blank_samples)
        end = min(len(out), idx + blank_samples + 1)
        out[start:end] = np.nan
    n_blanked = int(np.isnan(out).sum() - np.isnan(data).sum())
    return out, n_blanked


def _apply_range_check(data: np.ndarray, valid_range: tuple[float, float]) -> tuple[np.ndarray, int]:
    """범위 밖 값을 NaN으로 마킹한다 (1D, 원본 비파괴 복사)."""
    lo, hi = valid_range
    out = data.copy()
    mask = (out < lo) | (out > hi)
    n_bad = int(mask.sum())
    if n_bad > 0:
        out[mask] = np.nan
    return out, n_bad


def _apply_bandpass(data: np.ndarray, lo: float, hi: float, sr: float) -> np.ndarray:
    """Butterworth 대역통과 필터 (1D)."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= lo:
        return data

    sos = butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def _apply_lowpass(data: np.ndarray, hi: float, sr: float) -> np.ndarray:
    """Butterworth 저역통과 필터 (1D). DC 성분(절대값)을 보존한다."""
    from scipy.signal import butter, sosfiltfilt

    nyq = sr / 2.0
    if hi >= nyq:
        hi = nyq - 1.0
    if hi <= 0:
        return data

    sos = butter(4, hi / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, data).astype(data.dtype)


def _apply_filter(data: np.ndarray, cfg: SignalConfig, sr: float) -> np.ndarray:
    """SignalConfig의 filter_type에 따라 적절한 필터를 적용한다."""
    if cfg.filter_type == "bandpass" and cfg.filter_freq is not None:
        return _apply_bandpass(data, cfg.filter_freq[0], cfg.filter_freq[1], sr)
    elif cfg.filter_type == "lowpass" and cfg.filter_freq is not None:
        return _apply_lowpass(data, cfg.filter_freq[1], sr)
    return data


def _detect_electrocautery(data: np.ndarray, sr: float, threshold_std: float = 10.0,
                           blank_ms: float = 100.0) -> tuple[np.ndarray, int]:
    """전기소작기 아티팩트 구간을 NaN으로 마킹한다. (ECG/EEG용)

    급격한 진폭 변화(미분의 절대값)가 threshold_std배 이상인 구간을
    전후 blank_ms만큼 확장하여 NaN 처리한다.
    """
    out = data.copy()
    diff = np.abs(np.diff(out, prepend=out[0]))
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) * 1.4826  # MAD → std 추정
    if mad < 1e-10:
        return out, 0

    spike_mask = diff > (med + threshold_std * mad)
    if not spike_mask.any():
        return out, 0

    # blank_ms만큼 전후 확장
    blank_samples = int(blank_ms / 1000.0 * sr)
    spike_idx = np.where(spike_mask)[0]
    for idx in spike_idx:
        start = max(0, idx - blank_samples)
        end = min(len(out), idx + blank_samples + 1)
        out[start:end] = np.nan

    n_blanked = np.isnan(out).sum() - np.isnan(data).sum()
    return out, int(n_blanked)


def _extract_nan_free_segments(
    data: np.ndarray,  # (T,) float
    min_samples: int,
) -> list[np.ndarray]:
    """NaN 구간을 제거하고 연속 유효 세그먼트를 반환한다."""
    valid = ~np.isnan(data)
    segments: list[np.ndarray] = []

    diff = np.diff(valid.astype(np.int8), prepend=0, append=0)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    for s, e in zip(starts, ends):
        if e - s >= min_samples:
            segments.append(data[s:e])

    return segments


# ── 탐색 / 파싱 ────────────────────────────────────────────────


def discover_tracks(vital_path: Path) -> list[str]:
    """파일 내 트랙명을 출력한다 (로컬 탐색용)."""
    import vitaldb

    vf = vitaldb.VitalFile(str(vital_path))
    tracks = vf.get_track_names()
    return tracks


def _parse_subject_id(vital_path: Path) -> tuple[str, str]:
    """파일명에서 (subject_id, session_id)를 추출한다."""
    stem = vital_path.stem
    digits = "".join(c for c in stem if c.isdigit())
    if not digits:
        digits = stem
    subject_id = f"VDB_{int(digits):04d}"
    session_id = f"{subject_id}_S0"
    return subject_id, session_id


def _save_subject_manifest(
    subj_dir: Path,
    subject_id: str,
    session_id: str,
    recordings: list[dict],
) -> None:
    """subject의 manifest.json을 즉시 갱신한다.

    기존 manifest가 있으면 세션/레코딩을 병합하고,
    없으면 새로 생성한다. zarr 저장 직후 호출하여
    중단 시에도 manifest 유실을 방지한다.
    """
    manifest_path = subj_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
        # 같은 session_id가 있으면 recordings 병합, 없으면 세션 추가
        existing_session = None
        for s in manifest["sessions"]:
            if s["session_id"] == session_id:
                existing_session = s
                break
        if existing_session is not None:
            existing_files = {r["file"] for r in existing_session["recordings"]}
            for rec in recordings:
                if rec["file"] not in existing_files:
                    existing_session["recordings"].append(rec)
        else:
            manifest["sessions"].append(
                {"session_id": session_id, "recordings": recordings}
            )
    else:
        manifest = {
            "subject_id": subject_id,
            "source": "vitaldb",
            "sessions": [{"session_id": session_id, "recordings": recordings}],
        }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def process_vital(
    vital_path: Path,
    out_dir: Path,
    min_duration_s: float = 60.0,
    signal_types: set[int] | None = None,
) -> tuple[str, str, list[dict]]:
    """단일 .vital 파일을 처리하여 zarr 파일들을 저장한다.

    각 트랙의 zarr 저장 직후 manifest.json을 갱신하여,
    중간 중단 시에도 이미 저장된 데이터의 manifest가 유지된다.

    Returns
    -------
    (subject_id, session_id, recordings)
    """
    import vitaldb

    subject_id, session_id = _parse_subject_id(vital_path)
    subj_out = out_dir / subject_id
    subj_out.mkdir(parents=True, exist_ok=True)

    vf = vitaldb.VitalFile(str(vital_path))
    available_tracks = vf.get_track_names()

    recordings: list[dict] = []
    processed_keys: set[tuple[int, int]] = set()  # (signal_type, spatial_id) 중복 방지

    for track_name in available_tracks:
        if track_name not in TRACK_MAP:
            continue

        stype_key, spatial_id = TRACK_MAP[track_name]
        signal_type = SIGNAL_TYPES[stype_key]
        cfg = SIGNAL_CONFIGS[stype_key]

        # signal_types 필터: 지정된 타입만 파싱
        if signal_types is not None and signal_type not in signal_types:
            continue

        # 동일 (signal_type, spatial_id) 중복 시 첫 번째만 처리
        key = (signal_type, spatial_id)
        if key in processed_keys:
            continue

        # 원본 sampling rate로 데이터 추출
        try:
            trk = vf.find_track(track_name)
            native_sr = trk.srate if trk is not None and trk.srate > 0 else 0
            if native_sr <= 0:
                native_sr = 500.0
            data = vf.to_numpy(track_name, interval=1.0 / native_sr)
        except Exception as exc:
            print(f"    [WARN] {track_name} 로드 실패: {exc}", file=sys.stderr)
            continue

        if data is None or len(data) == 0:
            continue

        data = data.flatten()

        # ── Step 1: Physiological range check ──
        if cfg.valid_range is not None:
            data, n_bad = _apply_range_check(data, cfg.valid_range)
            if n_bad > 0:
                pct = n_bad / len(data) * 100
                print(f"    [RANGE] {track_name}: {n_bad} samples ({pct:.1f}%) out of range", file=sys.stderr)

        # ── Step 2: 스파이크/아티팩트 제거 (SignalConfig.spike_detection 기반) ──
        if cfg.spike_detection:
            data, n_blanked = _detect_electrocautery(data, native_sr, threshold_std=cfg.spike_threshold_std)
            if n_blanked > 0:
                pct = n_blanked / len(data) * 100
                print(f"    [SPIKE] {track_name}: {n_blanked} samples ({pct:.1f}%) blanked", file=sys.stderr)

        # ── Step 2b: ABP/PPG step change (계단형 아티팩트) 제거 ──
        if stype_key in ("abp", "ppg"):
            data, n_step = _detect_step_change(data, native_sr)
            if n_step > 0:
                pct = n_step / len(data) * 100
                print(f"    [STEP] {track_name}: {n_step} samples ({pct:.1f}%) blanked", file=sys.stderr)

        # ── Step 2c: PPG motion artifact 제거 ──
        if stype_key == "ppg":
            data, n_motion = _detect_motion_artifact(data, native_sr)
            if n_motion > 0:
                pct = n_motion / len(data) * 100
                print(f"    [MOTION] {track_name}: {n_motion} samples ({pct:.1f}%) blanked", file=sys.stderr)

        # ── Step 3: NaN-free 세그먼트 추출 ──
        min_samples = int(min_duration_s * native_sr)
        segments = _extract_nan_free_segments(data, min_samples)
        if not segments:
            print(f"    [SKIP] {track_name}: 유효 세그먼트 없음 (min={min_duration_s}s)", file=sys.stderr)
            continue

        # ── Step 4: 각 세그먼트 → 필터/리샘플 → 윈도우 단위 품질 검사 → 저장 ──
        seg_count = 0
        track_recordings: list[dict] = []
        quality_window_s = cfg.quality_window_s

        for seg_idx, segment in enumerate(segments):
            # Median filter → Notch filter → Bandpass/Lowpass 순서
            if cfg.median_kernel > 0:
                segment = _apply_median_filter(segment, kernel_size=cfg.median_kernel)
            if cfg.notch_freq is not None:
                segment = _apply_notch_filter(segment, freq=cfg.notch_freq, sr=native_sr)
            segment = _apply_filter(segment, cfg, native_sr)

            # 리샘플링 → TARGET_SR (100Hz)
            if native_sr != TARGET_SR:
                segment = resample_to_target(segment, orig_sr=native_sr, target_sr=TARGET_SR)

            # 윈도우 단위 품질 검사 → 연속 통과 윈도우를 그룹으로 수집
            win_samples = int(quality_window_s * TARGET_SR)
            # pass/fail 결과를 윈도우 인덱스와 함께 기록
            window_results: list[tuple[int, np.ndarray]] = []  # (win_idx, win_data)
            n_windows = 0
            n_fail_basic = 0
            n_fail_domain = 0

            for win_idx, win_start in enumerate(
                range(0, len(segment) - win_samples + 1, win_samples)
            ):
                win = segment[win_start:win_start + win_samples]
                n_windows += 1

                # 기본 품질 검사
                qscore = segment_quality_score(
                    win,
                    max_flatline_ratio=cfg.max_flatline_ratio,
                    max_clip_ratio=cfg.max_clip_ratio,
                    max_high_freq_ratio=cfg.max_high_freq_ratio,
                    min_amplitude=cfg.min_amplitude,
                    max_amplitude=cfg.max_amplitude,
                    min_high_freq_ratio=cfg.min_high_freq_ratio,
                )
                if not qscore["pass"]:
                    n_fail_basic += 1
                    window_results.append((win_idx, None))
                    continue

                # Domain-specific 품질 검사
                domain_result = domain_quality_check(stype_key, win, sr=TARGET_SR)
                if not domain_result["pass"]:
                    n_fail_domain += 1
                    window_results.append((win_idx, None))
                    continue

                window_results.append((win_idx, win))

            # 연속 통과 윈도우를 그룹으로 묶기 (불연속 경계에서 분리)
            contiguous_groups: list[list[np.ndarray]] = []
            current_group: list[np.ndarray] = []
            prev_idx = -2  # 불가능한 초기값

            for win_idx, win_data in window_results:
                if win_data is not None:
                    if win_idx == prev_idx + 1:
                        # 이전 통과 윈도우와 연속
                        current_group.append(win_data)
                    else:
                        # 새 연속 그룹 시작
                        if current_group:
                            contiguous_groups.append(current_group)
                        current_group = [win_data]
                    prev_idx = win_idx
                # win_data is None (fail) → prev_idx 갱신하지 않음

            if current_group:
                contiguous_groups.append(current_group)

            if not contiguous_groups:
                if n_windows > 0:
                    print(
                        f"    [SKIP] {track_name} seg{seg_idx}: "
                        f"모든 윈도우 불량 ({n_windows}개 중 basic={n_fail_basic}, domain={n_fail_domain})",
                        file=sys.stderr,
                    )
                continue

            # 각 연속 그룹을 별도 세그먼트로 저장
            n_good_total = sum(len(g) for g in contiguous_groups)
            for group_idx, group in enumerate(contiguous_groups):
                clean_segment = np.concatenate(group)
                channel_data = clean_segment.reshape(1, -1).astype(np.float32)

                # 최소 길이 체크 (min_duration_s)
                duration_s = channel_data.shape[1] / TARGET_SR
                if duration_s < min_duration_s:
                    continue

                ds_name = f"{session_id}_{stype_key}_{spatial_id}_seg{seg_idx}_{group_idx}"
                h5_path = subj_out / f"{subject_id}.h5"
                save_recording_h5(channel_data, h5_path, ds_name)

                rec = {
                    "signal_type": signal_type,
                    "file": f"{subject_id}.h5#{ds_name}",
                    "n_channels": 1,
                    "sampling_rate": TARGET_SR,
                    "n_timesteps": channel_data.shape[1],
                    "spatial_ids": [spatial_id],
                }
                track_recordings.append(rec)
                recordings.append(rec)
                seg_count += 1

            pct = n_good_total / n_windows * 100 if n_windows > 0 else 0
            n_groups = len(contiguous_groups)
            total_dur = sum(
                np.concatenate(g).shape[0] for g in contiguous_groups
            ) / TARGET_SR
            print(
                f"    saved {stype_key} seg{seg_idx}: {n_groups} contiguous group(s), "
                f"{total_dur:.0f}s total"
                f"  ({n_good_total}/{n_windows} windows, {pct:.0f}% pass)"
            )

        if seg_count > 0:
            processed_keys.add(key)
            _save_subject_manifest(subj_out, subject_id, session_id, track_recordings)
            if seg_count > 1:
                print(f"    [{track_name}] {seg_count}개 세그먼트 저장")

    return subject_id, session_id, recordings


def _process_one_worker(
    vf_path: Path,
    out_dir: Path,
    min_duration_s: float,
    signal_types: set[int] | None,
) -> tuple[str, str, list[dict]] | None:
    """단일 vital 파일 처리 (multiprocessing worker 호환)."""
    try:
        return process_vital(
            vf_path, out_dir, min_duration_s=min_duration_s,
            signal_types=signal_types,
        )
    except Exception as exc:
        print(f"    [{vf_path.name}] [ERROR] {exc}", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VitalDB (.vital) → datasets/processed/ zarr 변환"
    )
    parser.add_argument(
        "--raw", required=True,
        help="VitalDB .vital 파일이 있는 디렉토리",
    )
    parser.add_argument(
        "--out", default=None,
        help="처리 결과를 저장할 루트 디렉토리 (--discover 시 불필요)",
    )
    parser.add_argument(
        "--min-duration", type=float, default=60.0,
        help="최소 유효 신호 길이 (초, 기본 60)",
    )
    parser.add_argument(
        "--discover", action="store_true",
        help="트랙 탐색 모드: 파일 내 트랙명만 출력",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="처리할 최대 파일 수 (테스트용)",
    )
    parser.add_argument(
        "--signal-types", type=int, nargs="+", default=None,
        help="파싱할 signal type IDs (0=ECG,1=ABP,2=EEG,3=PPG,4=CVP,5=CO2,6=AWP). 미지정 시 전부.",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="병렬 처리 worker 수 (기본 1=순차 처리)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    vital_files = sorted(raw_dir.glob("*.vital"))
    if not vital_files:
        print(f"ERROR: {raw_dir}에 .vital 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    if args.max_files is not None:
        vital_files = vital_files[: args.max_files]

    print(f".vital 파일 {len(vital_files)}개 발견\n")

    # ── Discover 모드 ──
    if args.discover:
        for vf_path in vital_files:
            print(f"[{vf_path.name}]")
            try:
                tracks = discover_tracks(vf_path)
                for t in tracks:
                    mapped = TRACK_MAP.get(t)
                    tag = f" → {mapped[0]}({mapped[1]})" if mapped else ""
                    print(f"    {t}{tag}")
            except Exception as exc:
                print(f"    [ERROR] {exc}", file=sys.stderr)
            print()
        return

    # ── 파싱 모드 ──
    if args.out is None:
        print("ERROR: --out 경로를 지정하세요.", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # manifest.jsonl 기존 항목 로드
    jsonl_path = out_dir / "manifest.jsonl"
    existing_subjects: set[str] = set()
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_subjects.add(json.loads(line)["subject_id"])

    sig_filter = set(args.signal_types) if args.signal_types else None
    min_dur = args.min_duration

    n_processed = 0

    if args.workers > 1:
        from multiprocessing import Pool
        from functools import partial

        _worker = partial(
            _process_one_worker,
            out_dir=out_dir, min_duration_s=min_dur, signal_types=sig_filter,
        )
        print(f"병렬 처리: {args.workers} workers\n")
        with Pool(processes=args.workers) as pool:
            for result in pool.imap_unordered(_worker, vital_files):
                if result is None:
                    continue
                subject_id, session_id, recordings = result
                if not recordings:
                    continue
                if subject_id not in existing_subjects:
                    prev = ""
                    if jsonl_path.exists():
                        prev = jsonl_path.read_text(encoding="utf-8")
                    new_line = json.dumps(
                        {"subject_id": subject_id, "manifest": f"{subject_id}/manifest.json"},
                        ensure_ascii=False,
                    ) + "\n"
                    jsonl_path.write_text(prev + new_line, encoding="utf-8")
                    existing_subjects.add(subject_id)
                n_processed += 1
    else:
        for vf_path in vital_files:
            print(f"[{vf_path.name}]")
            result = _process_one_worker(
                vf_path, out_dir=out_dir, min_duration_s=min_dur,
                signal_types=sig_filter,
            )
            if result is None:
                continue
            subject_id, session_id, recordings = result
            if not recordings:
                print(f"    [SKIP] 유효 레코딩 없음")
                continue
            if subject_id not in existing_subjects:
                prev = ""
                if jsonl_path.exists():
                    prev = jsonl_path.read_text(encoding="utf-8")
                new_line = json.dumps(
                    {"subject_id": subject_id, "manifest": f"{subject_id}/manifest.json"},
                    ensure_ascii=False,
                ) + "\n"
                jsonl_path.write_text(prev + new_line, encoding="utf-8")
                existing_subjects.add(subject_id)
            n_processed += 1

    print(
        f"\n완료: {n_processed}명 처리 → {out_dir}"
        f"\n인덱스: {jsonl_path}"
    )


if __name__ == "__main__":
    main()
