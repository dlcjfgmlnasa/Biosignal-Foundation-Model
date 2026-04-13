# -*- coding:utf-8 -*-
"""Atrial Fibrillation Detection - 데이터 준비 스크립트.

맥파(pulse) 기반 beat interval 불규칙성으로 AF pseudo-label을 생성한다.
ECG (R-peak), PPG (systolic peak), ABP (systolic peak) 등 다양한 신호 지원.

AF 판별 기준:
  1. Peak 검출 (scipy find_peaks) — 신호 타입별 파라미터 자동 설정
  2. Beat interval coefficient of variation (CV) 계산
  3. CV > threshold → AF (불규칙 리듬)
  4. 추가: 연속 beat interval 차이의 RMSSD

데이터 소스: 로컬 전처리된 .pt 파일 (vitaldb_pt_test/)

사용법:
    # ECG (기본)
    python -m downstream.classification.atrial_fibrillation.prepare_data \
        --data-dir vitaldb_pt_test --signal-type ecg

    # PPG
    python -m downstream.classification.atrial_fibrillation.prepare_data \
        --data-dir vitaldb_pt_test --signal-type ppg

    # ABP
    python -m downstream.classification.atrial_fibrillation.prepare_data \
        --data-dir vitaldb_pt_test --signal-type abp
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.signal import find_peaks

TARGET_SR: float = 100.0

# 신호 타입별 peak 검출 파라미터
PEAK_PARAMS: dict[str, dict] = {
    "ecg": {
        "min_distance_sec": 0.3,
        "height_percentile": 70.0,
        "description": "R-peak",
    },
    "ppg": {
        "min_distance_sec": 0.4,
        "height_percentile": 60.0,
        "description": "systolic peak",
    },
    "abp": {
        "min_distance_sec": 0.4,
        "height_percentile": 60.0,
        "description": "systolic peak",
    },
}

# 지원 신호 타입
SUPPORTED_SIGNAL_TYPES = list(PEAK_PARAMS.keys())


@dataclass
class AFSample:
    """AF 분류 샘플."""

    signal: np.ndarray  # (win_samples,) at TARGET_SR
    signal_type: str  # "ecg", "ppg", "abp"
    label: int  # 0=Normal Sinus Rhythm, 1=AF
    beat_cv: float  # beat interval coefficient of variation
    rmssd: float  # successive beat interval difference RMSSD (ms)
    mean_hr: float  # 평균 심박수 (bpm)
    case_id: str
    win_start_sec: float


# ---- Peak 검출 ----


def detect_peaks(
    signal: np.ndarray,
    signal_type: str,
    sr: float = 100.0,
) -> np.ndarray:
    """신호에서 beat peak 위치를 검출한다.

    Parameters
    ----------
    signal : (n_samples,) 신호.
    signal_type : "ecg", "ppg", "abp".
    sr : sampling rate (Hz).

    Returns
    -------
    Peak sample indices.
    """
    params = PEAK_PARAMS.get(signal_type, PEAK_PARAMS["ecg"])
    min_distance = int(params["min_distance_sec"] * sr)
    height_thresh = np.percentile(signal, params["height_percentile"])
    peaks, _ = find_peaks(signal, distance=min_distance, height=height_thresh)
    return peaks


# ---- Beat interval feature ----


def compute_beat_features(
    peaks: np.ndarray,
    sr: float = 100.0,
) -> dict[str, float] | None:
    """Peak 배열에서 beat interval 기반 feature를 계산한다.

    Returns
    -------
    None if insufficient peaks (< 4).
    dict with: beat_cv, rmssd_ms, mean_hr_bpm
    """
    if len(peaks) < 4:
        return None

    intervals = np.diff(peaks) / sr  # 초 단위

    # 비정상 interval 필터링 (0.3~2.0초, 30~200 bpm)
    valid = (intervals >= 0.3) & (intervals <= 2.0)
    intervals_valid = intervals[valid]

    if len(intervals_valid) < 3:
        return None

    mean_bi = float(np.mean(intervals_valid))
    std_bi = float(np.std(intervals_valid))
    beat_cv = std_bi / mean_bi if mean_bi > 0 else 0.0

    # RMSSD: successive beat interval difference
    bi_diff = np.diff(intervals_valid)
    rmssd_ms = float(np.sqrt(np.mean(bi_diff**2))) * 1000  # ms

    mean_hr = 60.0 / mean_bi if mean_bi > 0 else 0.0

    return {
        "beat_cv": beat_cv,
        "rmssd_ms": rmssd_ms,
        "mean_hr_bpm": mean_hr,
    }


def classify_af(
    beat_cv: float,
    rmssd_ms: float,
    cv_threshold: float = 0.15,
    rmssd_threshold: float = 80.0,
) -> int:
    """Beat interval feature 기반 AF pseudo-label.

    AF 판별 조건 (둘 다 충족):
      - beat CV > cv_threshold (리듬 불규칙)
      - RMSSD > rmssd_threshold (beat-to-beat 변동 큼)

    Returns
    -------
    1 if AF, 0 if Normal Sinus Rhythm.
    """
    if beat_cv > cv_threshold and rmssd_ms > rmssd_threshold:
        return 1
    return 0


# ---- 로컬 .pt 로더 ----


def _parse_pt_filename(name: str) -> dict | None:
    """파일명���서 메타데이터를 추출한다."""
    m = re.match(
        r"^(.+?)_S(\d+)_([a-z0-9]+)_(\d+)_seg(\d+)_(\d+)\.pt$",
        name,
    )
    if m is None:
        return None
    return {
        "subject_id": m.group(1),
        "session_id": int(m.group(2)),
        "signal_type": m.group(3),
        "spatial_id": int(m.group(4)),
        "seg_i": int(m.group(5)),
        "seg_j": int(m.group(6)),
    }


def _load_signal_segments(
    data_dir: str,
    signal_type: str,
    min_duration_sec: float = 60.0,
    max_subjects: int | None = None,
) -> list[dict]:
    """로컬 .pt 디렉토리에서 지정 신호 타입의 세그먼트를 로드한다.

    Returns
    -------
    list of {"case_id": str, "patient_id": str, "signal": np.ndarray, "signal_type": str}
    """
    root = Path(data_dir)
    if not root.is_dir():
        print(f"  ERROR: Data directory not found: {root}")
        return []

    subject_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]

    print(f"  Scanning {len(subject_dirs)} subjects in {root}...")
    segments: list[dict] = []

    for subj_dir in subject_dirs:
        subject_id = subj_dir.name

        for pt_file in subj_dir.glob("*.pt"):
            meta = _parse_pt_filename(pt_file.name)
            if meta is None:
                continue
            if meta["signal_type"] != signal_type:
                continue

            t = torch.load(pt_file, weights_only=True)  # (1, T)
            sig = t.squeeze(0).numpy()  # (T,)

            if len(sig) < int(min_duration_sec * TARGET_SR):
                continue

            seg_key = (
                f"{subject_id}_s{meta['session_id']}"
                f"_seg{meta['seg_i']}_{meta['seg_j']}"
            )
            segments.append(
                {
                    "case_id": seg_key,
                    "patient_id": subject_id,
                    "signal": sig,
                    "signal_type": signal_type,
                }
            )

    print(f"  Loaded {len(segments)} {signal_type.upper()} segments")
    return segments


# ---- 윈도우 추출 + AF 라벨링 ----


def extract_af_samples(
    segments: list[dict],
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    cv_threshold: float = 0.15,
    rmssd_threshold: float = 80.0,
) -> list[AFSample]:
    """세그먼트에서 윈도우를 추출하고 AF pseudo-label을 부여한다."""
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    samples: list[AFSample] = []

    for seg in segments:
        sig = seg["signal"]
        stype = seg["signal_type"]
        n_total = len(sig)

        for start in range(0, n_total - win_samples + 1, stride_samples):
            window = sig[start : start + win_samples]

            peaks = detect_peaks(window, stype, sr=TARGET_SR)
            beat_feat = compute_beat_features(peaks, sr=TARGET_SR)

            if beat_feat is None:
                continue

            label = classify_af(
                beat_feat["beat_cv"],
                beat_feat["rmssd_ms"],
                cv_threshold,
                rmssd_threshold,
            )

            samples.append(
                AFSample(
                    signal=window,
                    signal_type=stype,
                    label=label,
                    beat_cv=beat_feat["beat_cv"],
                    rmssd=beat_feat["rmssd_ms"],
                    mean_hr=beat_feat["mean_hr_bpm"],
                    case_id=seg["case_id"],
                    win_start_sec=start / TARGET_SR,
                )
            )

    return samples


# ---- 저장 ----


def save_dataset(
    train_samples: list[AFSample],
    test_samples: list[AFSample],
    signal_type: str,
    window_sec: float,
    out_dir: str,
) -> Path:
    """AFSample 리스트를 .pt로 저장한다."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples: list[AFSample]) -> dict:
        if not samples:
            return {
                "signals": torch.tensor([]),
                "labels": torch.tensor([]),
                "beat_cvs": torch.tensor([]),
                "rmssds": torch.tensor([]),
                "mean_hrs": torch.tensor([]),
            }

        signals = torch.stack([torch.from_numpy(s.signal).float() for s in samples])
        labels = torch.tensor([s.label for s in samples], dtype=torch.long)
        beat_cvs = torch.tensor([s.beat_cv for s in samples], dtype=torch.float32)
        rmssds = torch.tensor([s.rmssd for s in samples], dtype=torch.float32)
        mean_hrs = torch.tensor([s.mean_hr for s in samples], dtype=torch.float32)
        case_ids = [s.case_id for s in samples]

        return {
            "signals": signals,
            "labels": labels,
            "beat_cvs": beat_cvs,
            "rmssds": rmssds,
            "mean_hrs": mean_hrs,
            "case_ids": case_ids,
        }

    save_dict = {
        "train": _to_tensors(train_samples),
        "test": _to_tensors(test_samples),
        "metadata": {
            "task": "atrial_fibrillation_detection",
            "source": "vitaldb_pt",
            "input_signal": signal_type,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "label_method": "beat_interval_irregularity_pseudo_label",
            "n_train": len(train_samples),
            "n_test": len(test_samples),
        },
    }

    filename = f"af_detection_{signal_type}.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 통계 출력 ----


def print_stats(name: str, samples: list[AFSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n_total = len(samples)
    n_af = sum(1 for s in samples if s.label == 1)
    n_nsr = n_total - n_af
    hrs = [s.mean_hr for s in samples]
    cvs = [s.beat_cv for s in samples]

    print(f"  {name}: {n_total} samples")
    print(f"    NSR: {n_nsr} ({n_nsr / n_total * 100:.1f}%)")
    print(f"    AF:  {n_af} ({n_af / n_total * 100:.1f}%)")
    print(
        f"    HR:  [{min(hrs):.0f}, {max(hrs):.0f}] bpm, "
        f"mean={np.mean(hrs):.0f} +/- {np.std(hrs):.0f}"
    )
    print(
        f"    Beat CV: [{min(cvs):.3f}, {max(cvs):.3f}], "
        f"mean={np.mean(cvs):.3f} +/- {np.std(cvs):.3f}"
    )


# ---- 메인 ----


def prepare_af_detection(
    data_dir: str,
    signal_type: str = "ecg",
    max_subjects: int | None = None,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    cv_threshold: float = 0.15,
    rmssd_threshold: float = 80.0,
    out_dir: str = "outputs/downstream/atrial_fibrillation",
) -> Path:
    """AF detection 데이터를 준비한다."""
    if signal_type not in SUPPORTED_SIGNAL_TYPES:
        print(
            f"ERROR: Unsupported signal type '{signal_type}'. "
            f"Choose from {SUPPORTED_SIGNAL_TYPES}",
            file=sys.stderr,
        )
        sys.exit(1)

    peak_desc = PEAK_PARAMS[signal_type]["description"]
    min_duration_sec = window_sec + stride_sec

    print(f"{'=' * 60}")
    print("  Atrial Fibrillation Detection")
    print(f"  Data:      {data_dir}")
    print(f"  Signal:    {signal_type.upper()} ({peak_desc} detection)")
    print(f"  Window:    {window_sec}s, Stride: {stride_sec}s")
    print(f"  AF criteria: beat CV > {cv_threshold}, RMSSD > {rmssd_threshold}ms")
    print(f"{'=' * 60}")

    # 1. 신호 로드
    print(f"\n[1/4] Loading {signal_type.upper()} segments...")
    segments = _load_signal_segments(
        data_dir, signal_type, min_duration_sec, max_subjects
    )

    if not segments:
        print(
            f"ERROR: No valid {signal_type.upper()} segments loaded.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Patient-level split
    print(f"\n[2/4] Splitting by patient (ratio={train_ratio})...")
    rng = np.random.default_rng(42)
    patient_ids = list({s["patient_id"] for s in segments})
    rng.shuffle(patient_ids)
    n_train_patients = max(1, int(len(patient_ids) * train_ratio))
    train_patients = set(patient_ids[:n_train_patients])

    train_segs = [s for s in segments if s["patient_id"] in train_patients]
    test_segs = [s for s in segments if s["patient_id"] not in train_patients]
    print(f"  Train: {len(train_segs)} segments ({len(train_patients)} patients)")
    print(
        f"  Test:  {len(test_segs)} segments "
        f"({len(patient_ids) - len(train_patients)} patients)"
    )

    # 3. 윈도우 추출 + AF 라벨링
    print("\n[3/4] Extracting windows + AF labeling...")
    train_samples = extract_af_samples(
        train_segs, window_sec, stride_sec, cv_threshold, rmssd_threshold
    )
    test_samples = extract_af_samples(
        test_segs, window_sec, stride_sec, cv_threshold, rmssd_threshold
    )

    print_stats("Train", train_samples)
    print_stats("Test", test_samples)

    if not train_samples and not test_samples:
        print("ERROR: No samples extracted.", file=sys.stderr)
        sys.exit(1)

    # 4. 저장
    print("\n[4/4] Saving...")
    save_path = save_dataset(
        train_samples, test_samples, signal_type, window_sec, out_dir
    )

    print(f"\n{'=' * 60}")
    print(f"  Done! {save_path}")
    print(f"{'=' * 60}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atrial Fibrillation Detection - Data Preparation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Local .pt data directory (e.g. vitaldb_pt_test/)",
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        default="ecg",
        choices=SUPPORTED_SIGNAL_TYPES,
        help="Input signal type for peak detection",
    )
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--cv-threshold", type=float, default=0.15)
    parser.add_argument("--rmssd-threshold", type=float, default=80.0)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/downstream/atrial_fibrillation",
    )
    args = parser.parse_args()

    prepare_af_detection(
        data_dir=args.data_dir,
        signal_type=args.signal_type,
        max_subjects=args.max_subjects,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        cv_threshold=args.cv_threshold,
        rmssd_threshold=args.rmssd_threshold,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
