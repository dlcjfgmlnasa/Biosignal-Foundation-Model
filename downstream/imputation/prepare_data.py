# -*- coding:utf-8 -*-
"""Imputation - 데이터 준비 스크립트.

Temporal Imputation: 같은 채널의 시간 구간을 마스킹 -> 앞뒤 context로 복원.
센서 일시 탈락, 아티팩트 구간 등 시간적 결측 상황을 시뮬레이션한다.

- 입력: 단일 신호 (CI 모드)
- forward(task="masked") -> reconstructed
- 평가: 마스킹 구간의 MSE/MAE/Pearson r

데이터 소스: VitalDB, MIMIC-III

사용법:
    # ECG temporal imputation
    python -m downstream.imputation.prepare_data \
        --signal-type ecg --n-cases 5

    # MIMIC-III ABP
    python -m downstream.imputation.prepare_data \
        --source mimic3 --signal-type abp --n-cases 5

    # 모든 신호
    python -m downstream.imputation.prepare_data \
        --signal-type all --n-cases 10

    # 시각화 포함
    python -m downstream.imputation.prepare_data \
        --signal-type ecg --n-cases 5 --visualize
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0

ALL_SIGNAL_TYPES = ["ecg", "abp", "ppg", "eeg", "co2", "awp", "cvp"]


# ---- 데이터 구조 ----


from dataclasses import dataclass


@dataclass
class TemporalSample:
    """Temporal imputation 샘플."""
    signal: np.ndarray       # (win_samples,) 전체 waveform (마스킹 전)
    signal_type: str
    case_id: str
    win_start_sec: float


# ---- VitalDB 로더 ----


def _load_vitaldb_signals(
    n_cases: int,
    signal_type: str,
    offset_from_end: int = 200,
) -> list[dict]:
    from downstream._common.data_utils import load_pilot_cases

    cases = load_pilot_cases(
        n_cases=n_cases,
        offset_from_end=offset_from_end,
        signal_types=[signal_type],
    )

    results = []
    for c in cases:
        if signal_type in c.tracks and len(c.tracks[signal_type]) >= int(60 * TARGET_SR):
            results.append({
                "case_id": f"vitaldb_{c.case_id}",
                "signal": c.tracks[signal_type],
            })

    print(f"  Loaded {len(results)} cases with {signal_type.upper()}")
    return results


# ---- MIMIC-III 로더 ----


def _load_mimic3_signals(
    n_cases: int,
    signal_type: str,
) -> list[dict]:
    from data.parser.mimic3_waveform import (
        scan_abp_records, load_and_preprocess_record,
    )

    if signal_type not in ("abp", "ecg", "ppg"):
        print(f"  WARNING: MIMIC-III only supports abp/ecg/ppg, got {signal_type}")
        return []

    records = scan_abp_records(max_records=n_cases * 5, verbose=False)
    filtered = []
    for r in records:
        if signal_type == "abp" and r.has_abp:
            filtered.append(r)
        elif signal_type == "ecg" and r.has_ecg:
            filtered.append(r)
        elif signal_type == "ppg" and r.has_ppg:
            filtered.append(r)

    filtered = filtered[:n_cases]
    results = []
    for info in filtered:
        case = load_and_preprocess_record(info, [signal_type])
        if case and signal_type in case.signals:
            sig = case.signals[signal_type]
            if len(sig) >= int(60 * TARGET_SR):
                results.append({
                    "case_id": info.record_name,
                    "signal": sig,
                })

    print(f"  Loaded {len(results)} cases with {signal_type.upper()} from MIMIC-III")
    return results


# ---- 윈도우 추출 ----


def extract_temporal_samples(
    cases: list[dict],
    signal_type: str,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
) -> list[TemporalSample]:
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)

    samples = []
    for case in cases:
        sig = case["signal"]
        for start in range(0, len(sig) - win_samples + 1, stride_samples):
            win = sig[start:start + win_samples]
            if np.isnan(win).any():
                continue
            samples.append(TemporalSample(
                signal=win,
                signal_type=signal_type,
                case_id=case["case_id"],
                win_start_sec=start / TARGET_SR,
            ))

    return samples


# ---- 저장 ----


def save_dataset(
    train: list[TemporalSample],
    test: list[TemporalSample],
    signal_type: str,
    window_sec: float,
    source: str,
    out_dir: str,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples):
        if not samples:
            return {}
        return {
            "signals": torch.stack([torch.from_numpy(s.signal).float() for s in samples]),
            "case_ids": [s.case_id for s in samples],
        }

    save_dict = {
        "train": _to_tensors(train),
        "test": _to_tensors(test),
        "metadata": {
            "task": "temporal_imputation",
            "source": source,
            "signal_type": signal_type,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "n_train": len(train),
            "n_test": len(test),
        },
    }

    filename = f"imputation_{source}_{signal_type}.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 시각화 ----


def _visualize(samples: list[TemporalSample], signal_type: str, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    print("\nGenerating visualization...")
    n_show = min(3, len(samples))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), squeeze=False)

    for i in range(n_show):
        ax = axes[i, 0]
        s = samples[i]
        t = np.arange(len(s.signal)) / TARGET_SR

        n = len(s.signal)
        mask_start = int(n * 0.35)
        mask_end = int(n * 0.65)

        ax.plot(t, s.signal, color="steelblue", linewidth=0.6)
        ax.axvspan(t[mask_start], t[mask_end], alpha=0.2, color="red", label="Masked region (30%)")
        ax.set_ylabel(signal_type.upper())
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(f"Temporal Imputation - {signal_type.upper()}", fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / f"imputation_{signal_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- 메인 ----


def prepare_imputation(
    source: str = "vitaldb",
    signal_type: str = "ecg",
    n_cases: int = 10,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/imputation",
    visualize: bool = False,
) -> list[Path]:
    if signal_type == "all":
        stypes = ALL_SIGNAL_TYPES
    else:
        stypes = [signal_type]

    saved_paths = []
    for stype in stypes:
        print(f"\n{'='*60}")
        print(f"  Temporal Imputation: {stype.upper()}")
        print(f"  Source: {source}, Window: {window_sec}s")
        print(f"{'='*60}")

        # 1. 로드
        print(f"\n[1/4] Loading data...")
        if source == "vitaldb":
            cases = _load_vitaldb_signals(n_cases, stype)
        elif source == "mimic3":
            cases = _load_mimic3_signals(n_cases, stype)
        else:
            print(f"ERROR: Unknown source '{source}'", file=sys.stderr)
            continue

        if not cases:
            print(f"  SKIP: No valid {stype.upper()} data")
            continue

        # 2. 분할
        print(f"\n[2/4] Splitting (ratio={train_ratio})...")
        rng = np.random.default_rng(42)
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        n_train = max(1, int(len(cases) * train_ratio))
        train_cases = [cases[i] for i in indices[:n_train]]
        test_cases = [cases[i] for i in indices[n_train:]]
        print(f"  Train: {len(train_cases)} cases, Test: {len(test_cases)} cases")

        # 3. 추출
        print(f"\n[3/4] Extracting samples...")
        train_samples = extract_temporal_samples(train_cases, stype, window_sec, stride_sec)
        test_samples = extract_temporal_samples(test_cases, stype, window_sec, stride_sec)
        print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

        if not train_samples and not test_samples:
            print(f"  SKIP: No samples for {stype.upper()}")
            continue

        # 4. 저장
        print(f"\n[4/4] Saving...")
        path = save_dataset(train_samples, test_samples, stype, window_sec, source, out_dir)
        saved_paths.append(path)

        if visualize:
            _visualize(train_samples + test_samples, stype, Path(out_dir))

    if saved_paths:
        print(f"\n{'='*60}")
        print(f"  Imputation data ready: {len(saved_paths)} signal type(s)")
        for p in saved_paths:
            print(f"    {p}")
        print(f"{'='*60}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Temporal Imputation - Data Preparation",
    )
    parser.add_argument("--source", type=str, default="vitaldb",
                        choices=["vitaldb", "mimic3"])
    parser.add_argument("--signal-type", type=str, default="ecg",
                        choices=ALL_SIGNAL_TYPES + ["all"],
                        help="Signal type ('all' for all types)")
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/imputation")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    prepare_imputation(
        source=args.source,
        signal_type=args.signal_type,
        n_cases=args.n_cases,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
