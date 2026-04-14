# -*- coding:utf-8 -*-
"""Arrhythmia Classification — 데이터 준비 스크립트.

MIMIC-III-Ext-PPG 파싱된 데이터 + metadata.csv 리듬 라벨을 사용하여
multi-class arrhythmia classification용 (signal, label) 쌍을 생성한다.

5-class:
    0: SR    (Sinus Rhythm, 정상 동율동)
    1: AF    (Atrial Fibrillation, 심방세동)
    2: STACH (Sinus Tachycardia, 동성빈맥)
    3: SBRAD (Sinus Bradycardia, 동성서맥)
    4: AFLT  (Atrial Flutter, 심방조동)

입력 신호 조합:
    - ppg (기본)
    - ecg
    - ppg + ecg (multi-signal)

데이터 소스: data/parser/mimic3_ext_ppg.py로 파싱된 .pt + manifest.json

사용법:
    # PPG 단독
    python -m downstream.classification.arrhythmia.prepare_data \
        --data-dir datasets/processed/mimic3_ext_ppg \
        --input-signals ppg

    # ECG 단독
    python -m downstream.classification.arrhythmia.prepare_data \
        --data-dir datasets/processed/mimic3_ext_ppg \
        --input-signals ecg

    # PPG + ECG
    python -m downstream.classification.arrhythmia.prepare_data \
        --data-dir datasets/processed/mimic3_ext_ppg \
        --input-signals ppg ecg
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0

# 5-class 리듬 라벨
RHYTHM_TO_LABEL: dict[str, int] = {
    "SR": 0,
    "AF": 1,
    "STACH": 2,
    "SBRAD": 3,
    "AFLT": 4,
}
CLASS_NAMES = ["SR", "AF", "STACH", "SBRAD", "AFLT"]
N_CLASSES = len(CLASS_NAMES)


@dataclass
class ArrhythmiaSample:
    """Arrhythmia classification 샘플."""

    signals: dict[str, np.ndarray]  # {"ppg": (3000,), "ecg": (3000,)}
    label: int  # 0=SR, 1=AF, 2=STACH, 3=SBRAD, 4=AFLT
    rhythm: str  # 원본 리듬 라벨
    patient: str
    record: str


def load_manifest_and_signals(
    data_dir: str,
    input_signals: list[str],
) -> list[ArrhythmiaSample]:
    """manifest.json에서 레코드 목록 + 라벨을 읽고, .pt 신호를 로드한다."""
    root = Path(data_dir)
    manifest_path = root / "manifest.json"

    if not manifest_path.exists():
        print(f"ERROR: manifest.json not found at {manifest_path}", file=sys.stderr)
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    records = manifest.get("records", [])
    print(f"  Total records in manifest: {len(records)}")

    samples: list[ArrhythmiaSample] = []
    rhythm_counts: Counter = Counter()
    n_skip_rhythm = 0
    n_skip_signal = 0

    for rec in records:
        labels = rec.get("labels", {})
        rhythm = labels.get("rhythm", "").strip()

        if rhythm not in RHYTHM_TO_LABEL:
            n_skip_rhythm += 1
            continue

        rhythm_counts[rhythm] += 1
        patient = rec.get("patient", "")
        record_name = rec.get("record", "")
        patient_dir = root / patient

        # 필요한 신호 로드
        signal_dict: dict[str, np.ndarray] = {}
        all_found = True

        for sig_type in input_signals:
            pt_path = patient_dir / f"{record_name}_{sig_type}.pt"
            if not pt_path.exists():
                all_found = False
                break
            tensor = torch.load(pt_path, weights_only=True)  # (1, T)
            signal_dict[sig_type] = tensor.squeeze(0).numpy()

        if not all_found:
            n_skip_signal += 1
            continue

        label = RHYTHM_TO_LABEL[rhythm]

        samples.append(
            ArrhythmiaSample(
                signals=signal_dict,
                label=label,
                rhythm=rhythm,
                patient=patient,
                record=record_name,
            )
        )

    print(f"  Rhythm distribution: {dict(rhythm_counts)}")
    print(f"  Skip (rhythm not in target): {n_skip_rhythm}")
    print(f"  Skip (signal missing): {n_skip_signal}")
    print(f"  Valid samples: {len(samples)}")
    return samples


def save_dataset(
    train_samples: list[ArrhythmiaSample],
    test_samples: list[ArrhythmiaSample],
    input_signals: list[str],
    out_dir: str,
) -> Path:
    """ArrhythmiaSample 리스트를 .pt로 저장한다."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(samples: list[ArrhythmiaSample]) -> dict:
        if not samples:
            return {}

        sig_tensors = {}
        for stype in input_signals:
            arrs = [s.signals[stype] for s in samples]
            sig_tensors[stype] = torch.stack(
                [torch.from_numpy(a).float() for a in arrs]
            )

        return {
            "signals": sig_tensors,
            "labels": torch.tensor([s.label for s in samples], dtype=torch.long),
            "rhythms": [s.rhythm for s in samples],
            "patients": [s.patient for s in samples],
            "records": [s.record for s in samples],
        }

    save_dict = {
        "train": _to_tensors(train_samples),
        "test": _to_tensors(test_samples),
        "metadata": {
            "task": "arrhythmia_classification",
            "source": "MIMIC-III-Ext-PPG",
            "input_signals": input_signals,
            "n_classes": N_CLASSES,
            "class_names": CLASS_NAMES,
            "rhythm_to_label": RHYTHM_TO_LABEL,
            "sampling_rate": TARGET_SR,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
        },
    }

    mode_str = "_".join(input_signals)
    filename = f"arrhythmia_{mode_str}.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


def print_stats(name: str, samples: list[ArrhythmiaSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return

    n = len(samples)
    dist = Counter(s.label for s in samples)
    n_patients = len({s.patient for s in samples})

    print(f"  {name}: {n} samples ({n_patients} patients)")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cnt = dist.get(cls_id, 0)
        print(f"    {cls_name}({cls_id}): {cnt} ({cnt / n * 100:.1f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arrhythmia Classification — Data Preparation (MIMIC-III-Ext-PPG)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Parsed MIMIC-III-Ext-PPG directory (with manifest.json)",
    )
    parser.add_argument(
        "--input-signals",
        nargs="+",
        default=["ppg"],
        choices=["ppg", "ecg", "abp"],
        help="Input signal types",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Patient-level train/test split ratio",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/downstream/arrhythmia",
        help="Output directory",
    )
    args = parser.parse_args()

    mode_str = " + ".join(s.upper() for s in args.input_signals)
    print(f"{'=' * 60}")
    print(f"  Arrhythmia Classification: {mode_str} → {N_CLASSES}-class")
    print(f"  Classes: {CLASS_NAMES}")
    print(f"  Data: {args.data_dir}")
    print(f"{'=' * 60}")

    # 1. 데이터 로드
    print("\n[1/3] Loading samples...")
    samples = load_manifest_and_signals(args.data_dir, args.input_signals)

    if not samples:
        print("ERROR: No valid samples.", file=sys.stderr)
        sys.exit(1)

    # 2. Train/Test 분할 (patient 단위)
    print(f"\n[2/3] Splitting by patient (ratio={args.train_ratio})...")
    rng = np.random.default_rng(42)
    patient_ids = list({s.patient for s in samples})
    rng.shuffle(patient_ids)
    n_train = max(1, int(len(patient_ids) * args.train_ratio))
    train_patients = set(patient_ids[:n_train])

    train_samples = [s for s in samples if s.patient in train_patients]
    test_samples = [s for s in samples if s.patient not in train_patients]

    print_stats("Train", train_samples)
    print_stats("Test", test_samples)

    if not train_samples or not test_samples:
        print("ERROR: Insufficient data for split.", file=sys.stderr)
        sys.exit(1)

    # 3. 저장
    print("\n[3/3] Saving...")
    save_path = save_dataset(
        train_samples, test_samples, args.input_signals, args.out_dir,
    )

    print(f"\n{'=' * 60}")
    print(f"  Done! {save_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
