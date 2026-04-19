# -*- coding:utf-8 -*-
"""Hypoxemia Prediction — 데이터 준비 스크립트.

미래 5/10/15분 후 SpO2 < 90% (≥1분 sustained) 예측을 위한 (input_window, future_label) 쌍 생성.

입력: PPG waveform (100Hz, 30~600초 윈도우)
라벨 소스: VitalDB numeric track `Solar8000/PLETH_SPO2` 또는 `Solar8000/SPO2`
    → 미래 horizon 구간에서 SpO2 < 90% 비율 >= 1분/horizon 이면 label=1

참고:
- VitalDB API를 통해 numeric track을 직접 로드하거나, 사전 파싱된 .pt (SpO2 numeric 포함) 필요.
- MIMIC-III Waveform Matched Subset은 SpO2 numeric 미포함 → 이 task는 VitalDB 전용 (Phase B).

사용법:
    # 로컬 .pt (사전 파싱된 VitalDB)
    python -m downstream.acute_event.hypoxemia.prepare_data \
        --data-dir /path/to/vitaldb_with_spo2/ \
        --input-signals ppg --horizon-min 5 --window-sec 600

    # VitalDB API 직접 (느림, pilot용)
    python -m downstream.acute_event.hypoxemia.prepare_data \
        --source vitaldb_api --n-cases 10 \
        --input-signals ppg --horizon-min 5

Output
------
outputs/downstream/hypoxemia/hypoxemia_<source>_<sigs>_h<horizon>min.pt
    {
        "metadata": {"task": "hypoxemia_prediction", "horizon_sec": N, ...},
        "train": {"signals": {st: (N, win)}, "labels": (N,), "label_values": (N,), "case_ids": list},
        "test":  {...}
    }
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0
HYPOXEMIA_THRESHOLD: float = 90.0  # SpO2 %


@dataclass
class HypoxemiaSample:
    input_signals: dict[str, np.ndarray]
    label: int
    label_value: float  # min SpO2 in horizon (또는 % time below 90)
    case_id: str


def compute_hypoxemia_label(
    spo2_track: np.ndarray,  # (T,) numeric SpO2 samples (arbitrary sampling rate)
    spo2_sr: float,
    sustained_sec: float = 60.0,
) -> tuple[int, float]:
    """Horizon 구간의 SpO2 track으로 label 계산.

    Returns
    -------
    (label, min_spo2_in_horizon):
        label = 1 if 연속 `sustained_sec` 이상 SpO2 < 90% 구간 존재
    """
    if len(spo2_track) == 0:
        return 0, float("nan")

    valid = np.isfinite(spo2_track) & (spo2_track > 0)
    if not valid.any():
        return 0, float("nan")

    min_val = float(np.nanmin(spo2_track[valid]))
    below = (spo2_track < HYPOXEMIA_THRESHOLD) & valid

    # Sustained: consecutive samples below threshold for >= sustained_sec
    sustained_samples = int(sustained_sec * spo2_sr)
    if sustained_samples <= 0:
        sustained_samples = 1

    run_lengths = []
    current = 0
    for v in below:
        if v:
            current += 1
        else:
            if current > 0:
                run_lengths.append(current)
            current = 0
    if current > 0:
        run_lengths.append(current)

    label = 1 if (run_lengths and max(run_lengths) >= sustained_samples) else 0
    return label, min_val


def main() -> None:
    parser = argparse.ArgumentParser(description="Hypoxemia data preparation")
    parser.add_argument(
        "--source",
        type=str,
        default="local_pt",
        choices=["local_pt", "vitaldb_api"],
        help="local_pt: 사전 파싱된 .pt 디렉토리. vitaldb_api: VitalDB API 직접 로드.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="local_pt source 사용 시 디렉토리 경로",
    )
    parser.add_argument(
        "--input-signals", nargs="+", default=["ppg"], choices=["ppg", "ecg"]
    )
    parser.add_argument("--window-sec", type=float, default=600.0)
    parser.add_argument("--stride-sec", type=float, default=60.0)
    parser.add_argument("--horizon-min", type=float, default=5.0)
    parser.add_argument("--sustained-sec", type=float, default=60.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--n-cases", type=int, default=10, help="vitaldb_api 사용 시 케이스 수")
    parser.add_argument(
        "--out-dir", type=str, default="outputs/downstream/hypoxemia"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    horizon_sec = args.horizon_min * 60.0
    sig_str = "_".join(args.input_signals)
    out_path = (
        out_dir
        / f"hypoxemia_{args.source}_{sig_str}_h{int(args.horizon_min)}min.pt"
    )

    # NOTE: 실제 데이터 추출 로직은 VitalDB API 혹은 사전 파싱 .pt 구조에 의존.
    # Phase B 구현은 파이프라인 스텁 제공 — 실제 SpO2 numeric track 파싱은
    # 사용 가능한 데이터 구조 확인 후 채움 (TODO).
    print("=" * 60)
    print("  Hypoxemia Data Preparation — Phase B stub")
    print("=" * 60)
    print(f"  Source:        {args.source}")
    print(f"  Data dir:      {args.data_dir}")
    print(f"  Input signals: {args.input_signals}")
    print(f"  Window:        {args.window_sec}s, stride {args.stride_sec}s")
    print(f"  Horizon:       {horizon_sec}s ({args.horizon_min} min)")
    print(f"  Threshold:     SpO2 < {HYPOXEMIA_THRESHOLD}% sustained >= "
          f"{args.sustained_sec}s")
    print(f"  Output:        {out_path}")
    print()

    if args.source == "local_pt":
        if not args.data_dir or not Path(args.data_dir).is_dir():
            print(
                "ERROR: --data-dir is required and must exist for local_pt source.",
                file=sys.stderr,
            )
            print(
                "  (VitalDB SpO2 numeric track 파싱이 선행되어야 함 — vitaldb parser 확장 필요)",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            "TODO: SpO2 numeric track이 포함된 파싱 스키마 확정 후 구현.\n"
            "  현재 data/parser/vitaldb.py는 waveform만 처리하므로 numeric track 확장 필요."
        )
        sys.exit(2)

    if args.source == "vitaldb_api":
        try:
            import vitaldb  # noqa: F401
        except ImportError:
            print(
                "ERROR: vitaldb package not installed. pip install vitaldb",
                file=sys.stderr,
            )
            sys.exit(1)
        print(
            "TODO: VitalDB API로 직접 SpO2 + PPG 동시 로드 구현.\n"
            "  compute_hypoxemia_label() 헬퍼는 이미 이 파일에 있음."
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
