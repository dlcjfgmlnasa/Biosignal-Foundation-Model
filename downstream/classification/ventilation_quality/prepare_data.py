# -*- coding:utf-8 -*-
"""Ventilation Quality - 데이터 준비 스크립트.

CO2 + AWP waveform 입력 -> ETCO2 기반 환기 품질 2-class 분류.
VitalDB internal evaluation (pretrain holdout split).

2-class:
    0: Hyperventilation (ETCO2 < 35 mmHg) - 과환기
    1: Normal          (ETCO2 >= 35 mmHg) - 정상

사용법:
    # 소수 테스트
    python -m downstream.classification.ventilation_quality.prepare_data --n-cases 5

    # 시각화 포함
    python -m downstream.classification.ventilation_quality.prepare_data --n-cases 10 --visualize

    # CO2 단독 입력
    python -m downstream.classification.ventilation_quality.prepare_data --input-signals co2

    # AWP 단독 입력
    python -m downstream.classification.ventilation_quality.prepare_data --input-signals awp
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0
NATIVE_SR_CO2: float = 62.5
NATIVE_SR_AWP: float = 62.5

# ETCO2 기반 2-class 라벨 기준
ETCO2_THRESHOLD = 35.0  # < 35: 과환기, >= 35: 정상
CLASS_NAMES = ["Hyperventilation", "Normal"]


# ---- 데이터 구조 ----


@dataclass
class VentilationSample:
    """환기 품질 샘플."""

    signals: dict[str, np.ndarray]  # {"co2": (win_samples,), "awp": (win_samples,)}
    label: int  # 0=hyper, 1=normal, 2=hypo
    etco2_value: float  # ETCO2 (mmHg)
    case_id: int
    win_start_sec: float


# ---- VitalDB 로더 ----


def _load_vitaldb_cases(
    n_cases: int,
    input_signals: list[str],
    offset_from_end: int = 0,
) -> list[dict]:
    """VitalDB에서 CO2/AWP waveform + ETCO2 numeric을 로드한다."""
    import vitaldb
    from data.parser._common import resample_to_target
    from data.parser.vitaldb import (
        SIGNAL_CONFIGS,
        _apply_filter,
        _apply_range_check,
        _extract_nan_free_segments,
    )

    # CO2+AWP+ETCO2 모두 있는 케이스
    required_tracks = ["Primus/CO2", "Primus/AWP", "Solar8000/ETCO2"]
    all_cases = sorted(vitaldb.find_cases(required_tracks))
    total = len(all_cases)

    start_idx = max(0, total - offset_from_end - n_cases)
    end_idx = max(0, total - offset_from_end)
    target_ids = all_cases[start_idx:end_idx]

    print(
        f"  Loading {len(target_ids)} cases (IDs {target_ids[0]}~{target_ids[-1]})..."
    )

    cases = []
    for i, case_id in enumerate(target_ids):
        if (i + 1) % 5 == 0:
            print(f"  [{i + 1}/{len(target_ids)}]...", end=" ")

        try:
            # Waveform 로드
            signals = {}
            for stype, track, native_sr in [
                ("co2", "Primus/CO2", NATIVE_SR_CO2),
                ("awp", "Primus/AWP", NATIVE_SR_AWP),
            ]:
                if stype not in input_signals:
                    continue

                raw = vitaldb.load_case(case_id, [track], interval=1.0 / native_sr)
                if raw is None or len(raw) == 0:
                    continue

                data = raw[:, 0].flatten().astype(np.float64)
                if (~np.isnan(data)).sum() < int(60 * native_sr):
                    continue

                # 전처리
                cfg = SIGNAL_CONFIGS.get(stype)
                if cfg is None:
                    continue

                if cfg.valid_range is not None:
                    data, _ = _apply_range_check(data, cfg.valid_range)

                min_samples = int(60.0 * native_sr)
                segments = _extract_nan_free_segments(data, min_samples)
                if not segments:
                    continue

                segment = max(segments, key=len)
                segment = _apply_filter(segment, cfg, native_sr)

                if native_sr != TARGET_SR:
                    segment = resample_to_target(segment, native_sr, TARGET_SR)

                if len(segment) >= int(60 * TARGET_SR):
                    signals[stype] = segment

            if not all(s in signals for s in input_signals):
                continue

            # ETCO2 numeric 로드 (1초 간격)
            etco2_raw = vitaldb.load_case(case_id, ["Solar8000/ETCO2"], interval=1.0)
            if etco2_raw is None or len(etco2_raw) == 0:
                continue

            etco2 = etco2_raw[:, 0].flatten().astype(np.float64)

            # 동일 길이로 자르기 (waveform 기준)
            min_wave_len = min(len(s) for s in signals.values())
            signals = {k: v[:min_wave_len] for k, v in signals.items()}

            cases.append(
                {
                    "case_id": case_id,
                    "signals": signals,
                    "etco2": etco2,
                }
            )

        except Exception:
            continue

    print(f"\n  Loaded {len(cases)} cases with all required signals")
    return cases


# ---- 윈도우 추출 + 라벨링 ----


def extract_ventilation_samples(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
) -> list[VentilationSample]:
    """CO2/AWP waveform 윈도우 + ETCO2 기반 라벨을 추출한다."""
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)

    samples: list[VentilationSample] = []

    for case in cases:
        signals = case["signals"]
        etco2 = case["etco2"]
        min_wave_len = min(len(s) for s in signals.values())

        for start in range(0, min_wave_len - win_samples + 1, stride_samples):
            end = start + win_samples

            # 입력 윈도우
            input_dict = {}
            valid = True
            for stype in input_signals:
                if stype in signals:
                    win = signals[stype][start:end]
                    if np.isnan(win).any():
                        valid = False
                        break
                    input_dict[stype] = win
                else:
                    valid = False
                    break

            if not valid or not input_dict:
                continue

            # ETCO2 라벨: 윈도우 중앙 시점의 ETCO2 값 (1초 간격 numeric)
            center_sec = int((start + end) / 2 / TARGET_SR)
            if center_sec >= len(etco2):
                continue

            # 중앙 +-5초 범위의 ETCO2 평균
            etco2_start = max(0, center_sec - 5)
            etco2_end = min(len(etco2), center_sec + 5)
            etco2_window = etco2[etco2_start:etco2_end]
            etco2_valid = etco2_window[~np.isnan(etco2_window)]

            if len(etco2_valid) == 0:
                continue

            etco2_val = float(np.mean(etco2_valid))

            # 비정상 범위 필터 (ETCO2 < 10 또는 > 80은 센서 오류)
            if etco2_val < 10 or etco2_val > 80:
                continue

            # 2-class 라벨링
            if etco2_val < ETCO2_THRESHOLD:
                label = 0  # Hyperventilation
            else:
                label = 1  # Normal

            samples.append(
                VentilationSample(
                    signals=input_dict,
                    label=label,
                    etco2_value=etco2_val,
                    case_id=case["case_id"],
                    win_start_sec=start / TARGET_SR,
                )
            )

    return samples


# ---- 저장 ----


def save_dataset(
    train_samples: list[VentilationSample],
    test_samples: list[VentilationSample],
    input_signals: list[str],
    window_sec: float,
    out_dir: str,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples: list[VentilationSample]) -> dict:
        if not samples:
            return {}

        sig_tensors = {}
        for stype in input_signals:
            arrs = [s.signals[stype] for s in samples if stype in s.signals]
            if arrs:
                sig_tensors[stype] = torch.stack(
                    [torch.from_numpy(a).float() for a in arrs]
                )

        return {
            "signals": sig_tensors,
            "labels": torch.tensor([s.label for s in samples], dtype=torch.long),
            "etco2_values": torch.tensor(
                [s.etco2_value for s in samples], dtype=torch.float32
            ),
            "case_ids": [s.case_id for s in samples],
        }

    save_dict = {
        "train": _to_tensors(train_samples),
        "test": _to_tensors(test_samples),
        "metadata": {
            "task": "ventilation_quality",
            "source": "VitalDB",
            "input_signals": input_signals,
            "n_classes": 2,
            "class_names": CLASS_NAMES,
            "etco2_threshold": ETCO2_THRESHOLD,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
        },
    }

    mode_str = "_".join(input_signals)
    filename = f"ventilation_quality_{mode_str}.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 통계 ----


def print_stats(name: str, samples: list[VentilationSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n = len(samples)
    from collections import Counter

    dist = Counter(s.label for s in samples)
    etco2s = [s.etco2_value for s in samples]

    print(f"  {name}: {n} samples")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cnt = dist.get(cls_id, 0)
        print(f"    {cls_name}({cls_id}): {cnt} ({cnt / n * 100:.1f}%)")
    print(
        f"    ETCO2: {np.mean(etco2s):.1f} +/- {np.std(etco2s):.1f} mmHg "
        f"[{np.min(etco2s):.1f}, {np.max(etco2s):.1f}]"
    )


# ---- 시각화 ----


def _visualize(
    samples: list[VentilationSample], input_signals: list[str], out_dir: Path
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping.")
        return

    print("\nGenerating visualizations...")

    # 1. 클래스별 waveform 예시
    from collections import defaultdict

    by_class = defaultdict(list)
    for s in samples:
        by_class[s.label].append(s)

    n_signals = len(input_signals)
    fig, axes = plt.subplots(2, n_signals, figsize=(6 * n_signals, 6), squeeze=False)
    colors = ["tab:blue", "tab:green"]

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        if cls_id not in by_class or not by_class[cls_id]:
            for col in range(n_signals):
                axes[cls_id, col].set_title(f"{cls_name} - no samples")
                axes[cls_id, col].axis("off")
            continue

        sample = by_class[cls_id][0]
        for col, stype in enumerate(input_signals):
            ax = axes[cls_id, col]
            sig = sample.signals[stype]
            t = np.arange(len(sig)) / TARGET_SR
            ax.plot(t, sig, linewidth=0.6, color=colors[cls_id])
            ax.set_title(f"{cls_name} (ETCO2={sample.etco2_value:.1f})")
            if col == 0:
                ax.set_ylabel(cls_name)
            if cls_id == 2:
                ax.set_xlabel("Time (s)")
            ax.set_xlim(0, t[-1])

    for col, stype in enumerate(input_signals):
        axes[0, col].text(
            0.5,
            1.15,
            stype.upper(),
            transform=axes[0, col].transAxes,
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    fig.suptitle("Ventilation Quality - Waveform Examples", fontsize=13, y=1.02)
    plt.tight_layout()
    path1 = out_dir / "ventilation_quality_examples.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # 2. ETCO2 분포
    etco2s = [s.etco2_value for s in samples]
    labels = [s.label for s in samples]

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        vals = [e for e, l in zip(etco2s, labels) if l == cls_id]
        if vals:
            ax.hist(
                vals,
                bins=30,
                alpha=0.6,
                color=colors[cls_id],
                label=f"{cls_name} (n={len(vals)})",
            )

    ax.axvline(
        ETCO2_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {ETCO2_THRESHOLD} mmHg",
    )
    ax.set_xlabel("ETCO2 (mmHg)")
    ax.set_ylabel("Count")
    ax.set_title("ETCO2 Distribution by Ventilation Quality Class")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path2 = out_dir / "ventilation_quality_etco2_dist.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path2}")


# ---- 메인 ----


def prepare_ventilation_quality(
    input_signals: list[str] | None = None,
    n_cases: int = 10,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    train_ratio: float = 0.7,
    offset_from_end: int = 200,
    out_dir: str = "outputs/downstream/ventilation_quality",
    visualize: bool = False,
) -> Path:
    if input_signals is None:
        input_signals = ["co2", "awp"]

    mode_str = " + ".join(s.upper() for s in input_signals)
    print(f"{'=' * 60}")
    print(f"  Ventilation Quality: {mode_str} -> 2-class")
    print(f"  Window: {window_sec}s, Stride: {stride_sec}s")
    print(f"{'=' * 60}")

    # 1. 데이터 로드
    print("\n[1/4] Loading VitalDB cases...")
    cases = _load_vitaldb_cases(n_cases, input_signals, offset_from_end)

    if not cases:
        print("ERROR: No valid cases.", file=sys.stderr)
        sys.exit(1)

    # 2. Train/Test 분할 (case 단위)
    print(f"\n[2/4] Splitting by case (ratio={train_ratio})...")
    rng = np.random.default_rng(42)
    case_ids = [c["case_id"] for c in cases]
    rng.shuffle(case_ids)
    n_train = max(1, int(len(case_ids) * train_ratio))
    train_ids = set(case_ids[:n_train])

    train_cases = [c for c in cases if c["case_id"] in train_ids]
    test_cases = [c for c in cases if c["case_id"] not in train_ids]
    print(f"  Train: {len(train_cases)} cases, Test: {len(test_cases)} cases")

    # 3. 윈도우 추출 + 라벨링
    print("\n[3/4] Extracting ventilation samples...")
    train_samples = extract_ventilation_samples(
        train_cases, input_signals, window_sec, stride_sec
    )
    test_samples = extract_ventilation_samples(
        test_cases, input_signals, window_sec, stride_sec
    )

    print_stats("Train", train_samples)
    print_stats("Test", test_samples)

    if not train_samples and not test_samples:
        print("ERROR: No samples extracted.", file=sys.stderr)
        sys.exit(1)

    # 4. 저장
    print("\n[4/4] Saving...")
    save_path = save_dataset(
        train_samples, test_samples, input_signals, window_sec, out_dir
    )

    if visualize:
        all_samples = train_samples + test_samples
        _visualize(all_samples, input_signals, Path(out_dir))

    total = len(train_samples) + len(test_samples)
    print(f"\n{'=' * 60}")
    print(f"  Ventilation Quality data ready: {total} samples, 2 classes")
    print(f"  File: {save_path}")
    print(f"{'=' * 60}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ventilation Quality - Data Preparation (CO2+AWP -> 3-class)",
    )
    parser.add_argument(
        "--input-signals",
        nargs="+",
        default=["co2", "awp"],
        choices=["co2", "awp"],
        help="Input signal types",
    )
    parser.add_argument("--n-cases", type=int, default=10)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--offset-from-end", type=int, default=200)
    parser.add_argument(
        "--out-dir", type=str, default="outputs/downstream/ventilation_quality"
    )
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    prepare_ventilation_quality(
        input_signals=args.input_signals,
        n_cases=args.n_cases,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        offset_from_end=args.offset_from_end,
        out_dir=args.out_dir,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
