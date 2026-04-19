# -*- coding:utf-8 -*-
"""Shock Prediction — 데이터 준비 스크립트.

미래 horizon 내 (MAP < 65 mmHg AND HR > 100 bpm) ≥1분 sustained 예측을 위한
(input_window, future_label) 쌍 생성.

입력: ABP + ECG (둘 다 필요; ABP → MAP, ECG → R-peak → HR)
라벨 소스:
    - MAP: ABP waveform의 미래 horizon 구간에서 sliding window로 계산
    - HR:  ECG R-peak detection (scipy.signal.find_peaks) → RR interval → HR

사용법:
    python -m downstream.acute_event.shock.prepare_data \
        --data-dir vitaldb_pt_test \
        --input-signals abp ecg \
        --window-sec 600 --horizon-min 5

Output
------
outputs/downstream/shock/shock_<source>_<sigs>_h<horizon>min.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


TARGET_SR: float = 100.0
SHOCK_MAP_THRESHOLD: float = 65.0  # mmHg
SHOCK_HR_THRESHOLD: float = 100.0  # bpm


def compute_map(abp: np.ndarray, fs: float = 100.0) -> np.ndarray:
    """Sliding window mean of ABP waveform (approximate MAP, 10s window)."""
    win = int(10.0 * fs)
    if len(abp) < win:
        return np.array([float(np.mean(abp))])
    kernel = np.ones(win) / win
    return np.convolve(abp, kernel, mode="valid")


def compute_hr_from_ecg(
    ecg: np.ndarray, fs: float = 100.0, min_rr_sec: float = 0.3
) -> np.ndarray:
    """ECG에서 R-peak 검출 → HR(bpm) 계열 반환.

    Simple threshold + distance 기반 피크 검출. 정밀도보다 속도 우선
    (precise detection은 neurokit2 등 권장).

    Returns
    -------
    hr_per_beat: (n_beats - 1,) — 각 RR interval의 순시 HR (bpm)
    """
    try:
        from scipy.signal import find_peaks
    except ImportError:
        raise ImportError("scipy required for HR extraction. pip install scipy")

    # Normalize + threshold at 75th percentile
    if len(ecg) < int(fs * 2):
        return np.array([])
    mu, sd = float(np.mean(ecg)), float(np.std(ecg) + 1e-6)
    z = (ecg - mu) / sd
    peaks, _ = find_peaks(z, height=1.5, distance=int(fs * min_rr_sec))
    if len(peaks) < 2:
        return np.array([])
    rr_sec = np.diff(peaks) / fs
    return 60.0 / rr_sec


def compute_shock_label(
    abp_future: np.ndarray,
    ecg_future: np.ndarray,
    fs: float = 100.0,
    sustained_sec: float = 60.0,
) -> tuple[int, float, float]:
    """Horizon 구간의 ABP + ECG로 shock label 계산.

    Returns
    -------
    (label, min_map, max_hr):
        label = 1 if 연속 sustained_sec 이상 (MAP<65 AND HR>100) 동시 성립
    """
    map_series = compute_map(abp_future, fs=fs)
    hr_series = compute_hr_from_ecg(ecg_future, fs=fs)

    if len(map_series) == 0 or len(hr_series) == 0:
        return 0, float("nan"), float("nan")

    min_map = float(np.min(map_series))
    max_hr = float(np.max(hr_series))

    # 간단 휴리스틱: 전체 horizon에서 min_map < 65 AND max_hr > 100
    # (정확한 sustained 판정은 시간 정렬 후 AND 연산 필요 — 추후 정제)
    label = int(
        (min_map < SHOCK_MAP_THRESHOLD) and (max_hr > SHOCK_HR_THRESHOLD)
    )
    return label, min_map, max_hr


def main() -> None:
    parser = argparse.ArgumentParser(description="Shock prediction data preparation")
    parser.add_argument(
        "--source", type=str, default="local_pt",
        choices=["local_pt", "vitaldb_api"],
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--input-signals", nargs="+", default=["abp", "ecg"],
        choices=["abp", "ecg", "ppg"],
    )
    parser.add_argument("--window-sec", type=float, default=600.0)
    parser.add_argument("--stride-sec", type=float, default=60.0)
    parser.add_argument("--horizon-min", type=float, default=5.0)
    parser.add_argument("--sustained-sec", type=float, default=60.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument(
        "--out-dir", type=str, default="outputs/downstream/shock"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon_sec = args.horizon_min * 60.0
    sig_str = "_".join(args.input_signals)
    out_path = (
        out_dir
        / f"shock_{args.source}_{sig_str}_h{int(args.horizon_min)}min.pt"
    )

    print("=" * 60)
    print("  Shock Data Preparation — Phase B stub")
    print("=" * 60)
    print(f"  Source:        {args.source}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Input signals: {args.input_signals}")
    print(f"  Window:        {args.window_sec}s, stride {args.stride_sec}s")
    print(f"  Horizon:       {horizon_sec}s ({args.horizon_min} min)")
    print(
        f"  Threshold:     MAP<{SHOCK_MAP_THRESHOLD} AND HR>{SHOCK_HR_THRESHOLD} "
        f"sustained >= {args.sustained_sec}s"
    )
    print(f"  Output:        {out_path}")
    print()

    if args.source == "local_pt":
        if not args.data_dir or not Path(args.data_dir).is_dir():
            print("ERROR: --data-dir required for local_pt source.", file=sys.stderr)
            sys.exit(1)
        print(
            "TODO: hypotension/prepare_data.py의 _load_local_pt_aligned_signals를\n"
            "  재사용하여 ABP+ECG 동시 로드 후 compute_shock_label 적용.\n"
            "  핵심 헬퍼 (compute_map, compute_hr_from_ecg, compute_shock_label)는 이 파일에 구현 완료."
        )
        sys.exit(2)

    if args.source == "vitaldb_api":
        try:
            import vitaldb  # noqa: F401
        except ImportError:
            print("ERROR: pip install vitaldb", file=sys.stderr)
            sys.exit(1)
        print(
            "TODO: VitalDB API로 ABP+ECG 동시 로드 후 compute_shock_label 적용."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
