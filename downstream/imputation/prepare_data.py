# -*- coding:utf-8 -*-
"""Imputation - 데이터 준비 스크립트.

결측 신호 복원 능력 평가. 두 가지 모드:

1. Temporal Imputation: 같은 채널의 시간 구간을 마스킹 -> 앞뒤 context로 복원
   - 입력: 단일 신호 (CI 모드)
   - forward(task="masked") -> reconstructed

2. Cross-modal Imputation: 특정 채널 전체를 제거 -> 다른 채널로 복원
   - 입력: 다변량 (any_variate 모드)
   - forward(task="masked") -> cross_pred
   - = "Virtual Sensing" (센서 고장 시 대체)

데이터 소스: VitalDB, MIMIC-III

사용법:
    # Temporal: ECG 시간 구간 복원
    python -m downstream.imputation.prepare_data \
        --mode temporal --signal-type ecg --n-cases 5

    # Cross-modal: ECG+PPG로 ABP 복원
    python -m downstream.imputation.prepare_data \
        --mode cross_modal --input-signals ecg ppg --target-signal abp --n-cases 5

    # 시각화 포함
    python -m downstream.imputation.prepare_data \
        --mode temporal --signal-type abp --n-cases 5 --visualize
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

# Cross-modal 시나리오 (입력 -> 타겟)
DEFAULT_CROSS_MODAL_SCENARIOS = [
    # Cardiovascular intra-group
    {"inputs": ["ecg", "ppg"], "target": "abp", "name": "ECG+PPG->ABP"},
    {"inputs": ["ecg", "abp"], "target": "ppg", "name": "ECG+ABP->PPG"},
    {"inputs": ["ecg"],        "target": "abp", "name": "ECG->ABP"},
    {"inputs": ["ppg"],        "target": "abp", "name": "PPG->ABP"},
    # Inter-group
    {"inputs": ["ecg"],        "target": "eeg", "name": "ECG->EEG"},
    # Respiratory
    {"inputs": ["co2"],        "target": "awp", "name": "CO2->AWP"},
]


# ---- 데이터 구조 ----


@dataclass
class TemporalSample:
    """Temporal imputation 샘플."""
    signal: np.ndarray       # (win_samples,) 전체 waveform (마스킹 전)
    signal_type: str
    case_id: str
    win_start_sec: float


@dataclass
class CrossModalSample:
    """Cross-modal imputation 샘플."""
    input_signals: dict[str, np.ndarray]   # {"ecg": (win,), "ppg": (win,)}
    target_signal: np.ndarray              # (win_samples,) 복원 타겟
    target_type: str
    case_id: str
    win_start_sec: float


# ---- VitalDB 로더 ----


def _load_vitaldb_signals(
    n_cases: int,
    signal_types: list[str],
    offset_from_end: int = 200,
) -> list[dict]:
    from downstream._common.data_utils import load_pilot_cases

    cases = load_pilot_cases(
        n_cases=n_cases,
        offset_from_end=offset_from_end,
        signal_types=signal_types,
    )

    results = []
    for c in cases:
        signals = {}
        for st in signal_types:
            if st in c.tracks and len(c.tracks[st]) >= int(60 * TARGET_SR):
                signals[st] = c.tracks[st]
        if signals:
            results.append({
                "case_id": f"vitaldb_{c.case_id}",
                "signals": signals,
            })

    print(f"  Loaded {len(results)} cases")
    return results


# ---- MIMIC-III 로더 ----


def _load_mimic3_signals(
    n_cases: int,
    signal_types: list[str],
) -> list[dict]:
    from data.parser.mimic3_waveform import (
        scan_abp_records, load_and_preprocess_record,
    )

    mimic_types = [s for s in signal_types if s in ("abp", "ecg", "ppg")]
    if not mimic_types:
        print(f"  WARNING: MIMIC-III only supports abp/ecg/ppg")
        return []

    records = scan_abp_records(max_records=n_cases * 5, verbose=False)
    records = records[:n_cases]

    results = []
    for info in records:
        case = load_and_preprocess_record(info, mimic_types)
        if case and case.signals:
            results.append({
                "case_id": info.record_name,
                "signals": case.signals,
            })

    print(f"  Loaded {len(results)} cases from MIMIC-III")
    return results


# ---- Temporal 윈도우 추출 ----


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
        if signal_type not in case["signals"]:
            continue
        sig = case["signals"][signal_type]

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


# ---- Cross-modal 윈도우 추출 ----


def extract_cross_modal_samples(
    cases: list[dict],
    input_types: list[str],
    target_type: str,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
) -> list[CrossModalSample]:
    all_types = set(input_types) | {target_type}
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)

    samples = []
    for case in cases:
        # 모든 필요 채널 있는지 확인
        if not all(st in case["signals"] for st in all_types):
            continue

        min_len = min(len(case["signals"][st]) for st in all_types)

        for start in range(0, min_len - win_samples + 1, stride_samples):
            end = start + win_samples

            inputs = {}
            valid = True
            for st in input_types:
                win = case["signals"][st][start:end]
                if np.isnan(win).any():
                    valid = False
                    break
                inputs[st] = win

            target = case["signals"][target_type][start:end]
            if np.isnan(target).any():
                valid = False

            if not valid:
                continue

            samples.append(CrossModalSample(
                input_signals=inputs,
                target_signal=target,
                target_type=target_type,
                case_id=case["case_id"],
                win_start_sec=start / TARGET_SR,
            ))

    return samples


# ---- 저장 ----


def save_temporal_dataset(
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

    filename = f"imputation_temporal_{source}_{signal_type}.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


def save_cross_modal_dataset(
    train: list[CrossModalSample],
    test: list[CrossModalSample],
    input_types: list[str],
    target_type: str,
    window_sec: float,
    source: str,
    out_dir: str,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples):
        if not samples:
            return {}
        sig_tensors = {}
        for st in input_types:
            sig_tensors[st] = torch.stack([
                torch.from_numpy(s.input_signals[st]).float() for s in samples
            ])
        return {
            "input_signals": sig_tensors,
            "target": torch.stack([torch.from_numpy(s.target_signal).float() for s in samples]),
            "case_ids": [s.case_id for s in samples],
        }

    mode_str = "_".join(input_types)
    save_dict = {
        "train": _to_tensors(train),
        "test": _to_tensors(test),
        "metadata": {
            "task": "cross_modal_imputation",
            "source": source,
            "input_signals": input_types,
            "target_signal": target_type,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "n_train": len(train),
            "n_test": len(test),
        },
    }

    filename = f"imputation_cross_{source}_{mode_str}_to_{target_type}.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 시각화 ----


def _visualize_temporal(samples: list[TemporalSample], signal_type: str, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    print("\nGenerating temporal imputation visualization...")
    n_show = min(3, len(samples))
    fig, axes = plt.subplots(n_show, 1, figsize=(14, 3 * n_show), squeeze=False)

    for i in range(n_show):
        ax = axes[i, 0]
        s = samples[i]
        t = np.arange(len(s.signal)) / TARGET_SR

        # 중앙 30% 구간을 마스킹 영역으로 표시
        n = len(s.signal)
        mask_start = int(n * 0.35)
        mask_end = int(n * 0.65)

        ax.plot(t, s.signal, color="steelblue", linewidth=0.6)
        ax.axvspan(t[mask_start], t[mask_end], alpha=0.2, color="red", label="Masked region")
        ax.set_ylabel(signal_type.upper())
        if i == 0:
            ax.legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    fig.suptitle(f"Temporal Imputation - {signal_type.upper()}", fontsize=12, y=1.01)
    plt.tight_layout()
    path = out_dir / f"imputation_temporal_{signal_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _visualize_cross_modal(
    samples: list[CrossModalSample],
    input_types: list[str],
    target_type: str,
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    print("\nGenerating cross-modal imputation visualization...")
    n_show = min(2, len(samples))
    n_rows = len(input_types) + 1  # inputs + target
    fig, axes = plt.subplots(n_rows, n_show, figsize=(7 * n_show, 3 * n_rows), squeeze=False)

    for col in range(n_show):
        s = samples[col]
        t = np.arange(len(s.target_signal)) / TARGET_SR

        for row, st in enumerate(input_types):
            ax = axes[row, col]
            ax.plot(t, s.input_signals[st], color="steelblue", linewidth=0.6)
            if col == 0:
                ax.set_ylabel(f"{st.upper()} (input)")

        ax = axes[-1, col]
        ax.plot(t, s.target_signal, color="tab:red", linewidth=0.6)
        if col == 0:
            ax.set_ylabel(f"{target_type.upper()} (target)")
        ax.set_xlabel("Time (s)")

    mode_str = "+".join(s.upper() for s in input_types)
    fig.suptitle(f"Cross-modal Imputation: {mode_str} -> {target_type.upper()}", fontsize=12, y=1.02)
    plt.tight_layout()
    path = out_dir / f"imputation_cross_{'_'.join(input_types)}_to_{target_type}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---- 메인 ----


def prepare_imputation(
    mode: str = "temporal",
    source: str = "vitaldb",
    signal_type: str = "ecg",
    input_signals: list[str] | None = None,
    target_signal: str = "abp",
    n_cases: int = 10,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/imputation",
    visualize: bool = False,
) -> Path | None:

    if mode == "temporal":
        print(f"\n{'='*60}")
        print(f"  Temporal Imputation: {signal_type.upper()}")
        print(f"  Source: {source}, Window: {window_sec}s")
        print(f"{'='*60}")

        # 로드
        print(f"\n[1/4] Loading data...")
        if source == "vitaldb":
            cases = _load_vitaldb_signals(n_cases, [signal_type])
        else:
            cases = _load_mimic3_signals(n_cases, [signal_type])

        if not cases:
            print("ERROR: No valid cases.", file=sys.stderr)
            return None

        # 분할
        print(f"\n[2/4] Splitting...")
        rng = np.random.default_rng(42)
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        n_train = max(1, int(len(cases) * train_ratio))
        train_cases = [cases[i] for i in indices[:n_train]]
        test_cases = [cases[i] for i in indices[n_train:]]

        # 추출
        print(f"\n[3/4] Extracting samples...")
        train_samples = extract_temporal_samples(train_cases, signal_type, window_sec, stride_sec)
        test_samples = extract_temporal_samples(test_cases, signal_type, window_sec, stride_sec)
        print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

        # 저장
        print(f"\n[4/4] Saving...")
        path = save_temporal_dataset(train_samples, test_samples, signal_type, window_sec, source, out_dir)

        if visualize:
            _visualize_temporal(train_samples + test_samples, signal_type, Path(out_dir))

        return path

    elif mode == "cross_modal":
        if input_signals is None:
            input_signals = ["ecg", "ppg"]

        all_types = list(set(input_signals) | {target_signal})
        mode_str = "+".join(s.upper() for s in input_signals)

        print(f"\n{'='*60}")
        print(f"  Cross-modal Imputation: {mode_str} -> {target_signal.upper()}")
        print(f"  Source: {source}, Window: {window_sec}s")
        print(f"{'='*60}")

        # 로드
        print(f"\n[1/4] Loading data...")
        if source == "vitaldb":
            cases = _load_vitaldb_signals(n_cases, all_types)
        else:
            cases = _load_mimic3_signals(n_cases, all_types)

        if not cases:
            print("ERROR: No valid cases.", file=sys.stderr)
            return None

        # 필요 채널 모두 있는 케이스만
        cases = [c for c in cases if all(st in c["signals"] for st in all_types)]
        if not cases:
            print("ERROR: No cases with all required signals.", file=sys.stderr)
            return None

        # 분할
        print(f"\n[2/4] Splitting...")
        rng = np.random.default_rng(42)
        indices = list(range(len(cases)))
        rng.shuffle(indices)
        n_train = max(1, int(len(cases) * train_ratio))
        train_cases = [cases[i] for i in indices[:n_train]]
        test_cases = [cases[i] for i in indices[n_train:]]

        # 추출
        print(f"\n[3/4] Extracting samples...")
        train_samples = extract_cross_modal_samples(
            train_cases, input_signals, target_signal, window_sec, stride_sec,
        )
        test_samples = extract_cross_modal_samples(
            test_cases, input_signals, target_signal, window_sec, stride_sec,
        )
        print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")

        # 저장
        print(f"\n[4/4] Saving...")
        path = save_cross_modal_dataset(
            train_samples, test_samples,
            input_signals, target_signal, window_sec, source, out_dir,
        )

        if visualize:
            _visualize_cross_modal(
                train_samples + test_samples, input_signals, target_signal, Path(out_dir),
            )

        return path

    else:
        print(f"ERROR: Unknown mode '{mode}'", file=sys.stderr)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Imputation - Data Preparation (Temporal + Cross-modal)",
    )
    parser.add_argument("--mode", type=str, default="temporal",
                        choices=["temporal", "cross_modal"])
    parser.add_argument("--source", type=str, default="vitaldb",
                        choices=["vitaldb", "mimic3"])
    # Temporal mode
    parser.add_argument("--signal-type", type=str, default="ecg",
                        choices=ALL_SIGNAL_TYPES)
    # Cross-modal mode
    parser.add_argument("--input-signals", nargs="+", default=None,
                        choices=ALL_SIGNAL_TYPES)
    parser.add_argument("--target-signal", type=str, default="abp",
                        choices=ALL_SIGNAL_TYPES)
    # Common
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/imputation")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    prepare_imputation(
        mode=args.mode,
        source=args.source,
        signal_type=args.signal_type,
        input_signals=args.input_signals,
        target_signal=args.target_signal,
        n_cases=args.n_cases,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
