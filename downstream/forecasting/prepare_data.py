# -*- coding:utf-8 -*-
"""Vital Sign Forecasting - 데이터 준비 스크립트.

임의 신호의 과거 waveform으로 미래 waveform을 예측하는 task.
model.generate() (autoregressive) API를 평가한다.

지원 신호: ecg, abp, ppg, eeg, co2, awp, cvp (모두 가능)
데이터 소스: VitalDB, MIMIC-III

사용법:
    # VitalDB ECG forecasting (5 cases)
    python -m downstream.forecasting.prepare_data \
        --source vitaldb --signal-type ecg --n-cases 5

    # VitalDB 모든 신호 (개별 .pt 생성)
    python -m downstream.forecasting.prepare_data \
        --source vitaldb --signal-type all --n-cases 10

    # MIMIC-III ABP forecasting
    python -m downstream.forecasting.prepare_data \
        --source mimic3 --signal-type abp --n-cases 5

    # 시각화 포함
    python -m downstream.forecasting.prepare_data \
        --source vitaldb --signal-type ecg --n-cases 5 --visualize
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0

ALL_SIGNAL_TYPES = ["ecg", "abp", "ppg", "eeg", "co2", "awp", "cvp"]


# ---- 데이터 구조 ----


@dataclass
class ForecastSample:
    """Forecasting 샘플."""
    context: np.ndarray     # (context_samples,) 과거 waveform
    target: np.ndarray      # (target_samples,) 미래 waveform (ground truth)
    signal_type: str
    case_id: str
    win_start_sec: float


# ---- VitalDB 로더 ----


def _load_vitaldb_signal(
    n_cases: int,
    signal_type: str,
    offset_from_end: int = 200,
) -> list[dict]:
    """VitalDB에서 단일 signal type 데이터를 로드한다."""
    from downstream._common.data_utils import load_pilot_cases

    cases = load_pilot_cases(
        n_cases=n_cases,
        offset_from_end=offset_from_end,
        signal_types=[signal_type],
    )

    results = []
    for c in cases:
        if signal_type in c.tracks:
            sig = c.tracks[signal_type]
            if len(sig) >= int(60 * TARGET_SR):
                results.append({
                    "case_id": f"vitaldb_{c.case_id}",
                    "signal": sig,
                })

    print(f"  Loaded {len(results)} cases with {signal_type.upper()}")
    return results


# ---- MIMIC-III 로더 ----


def _load_mimic3_signal(
    n_cases: int,
    signal_type: str,
) -> list[dict]:
    """MIMIC-III에서 단일 signal type 데이터를 로드한다."""
    from data.parser.mimic3_waveform import (
        scan_abp_records, load_and_preprocess_record,
    )

    if signal_type not in ("abp", "ecg", "ppg"):
        print(f"  WARNING: MIMIC-III only supports abp/ecg/ppg, got {signal_type}")
        return []

    records = scan_abp_records(max_records=n_cases * 5, verbose=False)

    # 필요 채널 필터
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
    for i, info in enumerate(filtered):
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


def extract_forecast_samples(
    cases: list[dict],
    signal_type: str,
    context_sec: float = 30.0,
    target_sec: float = 10.0,
    stride_sec: float = 15.0,
) -> list[ForecastSample]:
    """과거(context) + 미래(target) 윈도우 쌍을 추출한다."""
    ctx_samples = int(context_sec * TARGET_SR)
    tgt_samples = int(target_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)
    total_needed = ctx_samples + tgt_samples

    samples: list[ForecastSample] = []

    for case in cases:
        sig = case["signal"]
        if len(sig) < total_needed:
            continue

        for start in range(0, len(sig) - total_needed + 1, stride_samples):
            context = sig[start:start + ctx_samples]
            target = sig[start + ctx_samples:start + ctx_samples + tgt_samples]

            if np.isnan(context).any() or np.isnan(target).any():
                continue

            samples.append(ForecastSample(
                context=context,
                target=target,
                signal_type=signal_type,
                case_id=case["case_id"],
                win_start_sec=start / TARGET_SR,
            ))

    return samples


# ---- 저장 ----


def save_dataset(
    train_samples: list[ForecastSample],
    test_samples: list[ForecastSample],
    signal_type: str,
    context_sec: float,
    target_sec: float,
    source: str,
    out_dir: str,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples: list[ForecastSample]) -> dict:
        if not samples:
            return {}
        return {
            "context": torch.stack([torch.from_numpy(s.context).float() for s in samples]),
            "target": torch.stack([torch.from_numpy(s.target).float() for s in samples]),
            "case_ids": [s.case_id for s in samples],
        }

    save_dict = {
        "train": _to_tensors(train_samples),
        "test": _to_tensors(test_samples),
        "metadata": {
            "task": "vital_sign_forecasting",
            "source": source,
            "signal_type": signal_type,
            "context_sec": context_sec,
            "target_sec": target_sec,
            "sampling_rate": TARGET_SR,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
        },
    }

    filename = f"forecasting_{source}_{signal_type}_ctx{int(context_sec)}s_tgt{int(target_sec)}s.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 통계 ----


def print_stats(name: str, samples: list[ForecastSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n = len(samples)
    ctx_dur = len(samples[0].context) / TARGET_SR
    tgt_dur = len(samples[0].target) / TARGET_SR
    print(f"  {name}: {n} samples (context={ctx_dur:.0f}s, target={tgt_dur:.0f}s)")


# ---- 시각화 ----


def _visualize(
    samples: list[ForecastSample],
    signal_type: str,
    context_sec: float,
    target_sec: float,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping.")
        return

    print("\nGenerating visualizations...")

    n_show = min(4, len(samples))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), squeeze=False)

    for i in range(n_show):
        ax = axes[i, 0]
        s = samples[i]

        ctx_t = np.arange(len(s.context)) / TARGET_SR
        tgt_t = np.arange(len(s.target)) / TARGET_SR + context_sec

        ax.plot(ctx_t, s.context, color="steelblue", linewidth=0.6, label="Context (input)")
        ax.plot(tgt_t, s.target, color="tab:red", linewidth=0.6, label="Target (to predict)")
        ax.axvline(context_sec, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_ylabel(f"{signal_type.upper()}")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(0, context_sec + target_sec)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(f"Vital Sign Forecasting - {signal_type.upper()} "
                 f"(context={context_sec:.0f}s -> predict {target_sec:.0f}s)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / f"forecasting_{signal_type}_examples.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- 메인 ----


def prepare_forecasting(
    source: str = "vitaldb",
    signal_type: str = "ecg",
    n_cases: int = 10,
    context_sec: float = 30.0,
    target_sec: float = 10.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/forecasting",
    visualize: bool = False,
) -> list[Path]:
    # "all"이면 모든 signal type에 대해 반복
    if signal_type == "all":
        stypes = ALL_SIGNAL_TYPES
    else:
        stypes = [signal_type]

    saved_paths = []
    for stype in stypes:
        print(f"\n{'='*60}")
        print(f"  Vital Sign Forecasting: {stype.upper()}")
        print(f"  Source: {source}")
        print(f"  Context: {context_sec}s -> Predict: {target_sec}s")
        print(f"{'='*60}")

        # 1. 데이터 로드
        print(f"\n[1/4] Loading {stype.upper()} data...")
        if source == "vitaldb":
            cases = _load_vitaldb_signal(n_cases, stype)
        elif source == "mimic3":
            cases = _load_mimic3_signal(n_cases, stype)
        else:
            print(f"ERROR: Unknown source '{source}'", file=sys.stderr)
            continue

        if not cases:
            print(f"  SKIP: No valid {stype.upper()} data")
            continue

        # 2. Train/Test 분할
        print(f"\n[2/4] Splitting (ratio={train_ratio})...")
        rng = np.random.default_rng(42)
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        n_train = max(1, int(len(cases) * train_ratio))
        train_cases = [cases[i] for i in indices[:n_train]]
        test_cases = [cases[i] for i in indices[n_train:]]
        print(f"  Train: {len(train_cases)} cases, Test: {len(test_cases)} cases")

        # 3. 윈도우 추출
        print(f"\n[3/4] Extracting forecast samples...")
        train_samples = extract_forecast_samples(
            train_cases, stype, context_sec, target_sec, stride_sec,
        )
        test_samples = extract_forecast_samples(
            test_cases, stype, context_sec, target_sec, stride_sec,
        )
        print_stats("Train", train_samples)
        print_stats("Test", test_samples)

        if not train_samples and not test_samples:
            print(f"  SKIP: No samples for {stype.upper()}")
            continue

        # 4. 저장
        print(f"\n[4/4] Saving...")
        path = save_dataset(
            train_samples, test_samples,
            stype, context_sec, target_sec, source, out_dir,
        )
        saved_paths.append(path)

        if visualize:
            all_samples = train_samples + test_samples
            _visualize(all_samples, stype, context_sec, target_sec, Path(out_dir))

    if saved_paths:
        print(f"\n{'='*60}")
        print(f"  Forecasting data ready: {len(saved_paths)} signal type(s)")
        for p in saved_paths:
            print(f"    {p}")
        print(f"{'='*60}")

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vital Sign Forecasting - Data Preparation",
    )
    parser.add_argument("--source", type=str, default="vitaldb",
                        choices=["vitaldb", "mimic3"])
    parser.add_argument("--signal-type", type=str, default="ecg",
                        choices=ALL_SIGNAL_TYPES + ["all"],
                        help="Signal type to forecast ('all' for all types)")
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--context-sec", type=float, default=30.0,
                        help="Context window length (seconds)")
    parser.add_argument("--target-sec", type=float, default=10.0,
                        help="Target prediction length (seconds)")
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/forecasting")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    prepare_forecasting(
        source=args.source,
        signal_type=args.signal_type,
        n_cases=args.n_cases,
        context_sec=args.context_sec,
        target_sec=args.target_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
