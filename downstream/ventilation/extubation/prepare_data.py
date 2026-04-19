# -*- coding:utf-8 -*-
"""Extubation Success Prediction — 데이터 준비 스크립트.

VitalDB case metadata에서 발관 성공/실패를 라벨링:
    Success = (48h 이내 reintubation 없음) AND (48h 이내 사망 없음)

입력: 발관 전 30분 waveform (CO2 + AWP + ECG + ABP)
    → 5분 윈도우 × 6개 슬라이딩 → Aggregator에 투입

VitalDB case 필드 사용:
    - `extubation_time` (발관 시각)
    - `reintubation_time` (재삽관 시각, null 가능)
    - `death_time` (사망 시각, null 가능)

사용법:
    python -m downstream.ventilation.extubation.prepare_data \
        --n-cases 20 --window-sec 300 --n-windows 6 \
        --out-dir outputs/downstream/extubation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extubation Success data preparation"
    )
    parser.add_argument(
        "--source", type=str, default="vitaldb_api",
        choices=["vitaldb_api", "local_pt"],
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--window-sec", type=float, default=300.0)
    parser.add_argument("--n-windows", type=int, default=6)
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--reintubation-window-h", type=float, default=48.0)
    parser.add_argument(
        "--signals", nargs="+", default=["co2", "awp", "ecg", "abp"]
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--out-dir", type=str, default="outputs/downstream/extubation"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Extubation Success Data Preparation — Phase B stub")
    print("=" * 60)
    print(f"  Source:      {args.source}")
    print(f"  N cases:     {args.n_cases}")
    print(f"  Window:      {args.window_sec}s × {args.n_windows} (pre-extubation)")
    print(f"  Signals:     {args.signals}")
    print(f"  Outcome window: {args.reintubation_window_h}h")
    print(f"  Output:      {out_dir}/extubation_<source>_w{int(args.window_sec)}s.pt")
    print()
    print("TODO:")
    print("  1) VitalDB API로 case metadata (extubation_time, reintubation_time,")
    print("     death_time) 조회")
    print("  2) 각 case에서 발관 직전 (window_sec × n_windows)초 구간 waveform 로드")
    print("  3) label 계산: (reintubation_time - extubation_time) > 48h AND alive")
    print("  4) .pt 저장 — Mortality와 동일 스키마:")
    print("     {'train': [{'signals': {st: (n_windows, win_samples)}, 'n_windows': N,")
    print("                  'label': 0/1, 'case_id': ...}, ...], 'test': [...]}")
    sys.exit(2)


if __name__ == "__main__":
    main()
