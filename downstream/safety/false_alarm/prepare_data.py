# -*- coding:utf-8 -*-
"""ICU False Alarm Reduction — 데이터 준비 스크립트.

PhysioNet 2015 Challenge 파싱 결과(manifest.json + 채널별 .pt)를 로드하여
window_sec별로 train/test split된 .pt 데이터셋을 생성한다.

각 .pt 파일 구조:
    {
        train: {signals: {ecg: (N, T), ppg: ...}, labels, alarm_types, records},
        test:  {signals: {ecg: (N, T), ppg: ...}, labels, alarm_types, records},
        metadata: {task, window_sec, input_signals, alarm_distribution, ...},
    }

사용법:
    # 단일 윈도우
    python -m downstream.safety.false_alarm.prepare_data \
        --data-dir datasets/processed/anomaly_detection \
        --input-signals ecg ppg abp --window-secs 60

    # Sweep: 여러 윈도우 크기
    python -m downstream.safety.false_alarm.prepare_data \
        --data-dir datasets/processed/anomaly_detection \
        --input-signals ecg ppg abp --window-secs 30 60 120 300
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0

# signal_name → signal_type 매핑
SIGNAL_NAME_TO_TYPE: dict[str, str] = {
    "II": "ecg", "I": "ecg", "III": "ecg",
    "V": "ecg", "V1": "ecg", "V2": "ecg", "V5": "ecg",
    "aVR": "ecg", "aVL": "ecg", "aVF": "ecg",
    "MCL": "ecg", "MCL1": "ecg",
    "ABP": "abp", "ART": "abp", "AOBP": "abp",
    "PLETH": "ppg",
}


def load_all_records(
    data_dir: Path,
    input_signals: list[str],
) -> list[dict]:
    """manifest.json에서 전체 레코드를 로드한다.

    Returns
    -------
    list of dict: {record, alarm_type, label, signals: {stype: ndarray(full)}}
    """
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    records = []
    for rec_info in manifest["records"]:
        label = 1 if rec_info["label"] else 0
        alarm_type = rec_info.get("alarm_type", "unknown")

        signals: dict[str, np.ndarray] = {}
        type_loaded: set[str] = set()

        for sig_info in rec_info.get("signals", []):
            sig_type = sig_info.get("signal_type")
            if sig_type is None:
                sig_name = sig_info.get("signal_name", "")
                sig_type = SIGNAL_NAME_TO_TYPE.get(sig_name)
            if sig_type is None or sig_type not in input_signals:
                continue
            if sig_type in type_loaded:
                continue

            pt_path = data_dir / sig_info.get("file", "")
            if not pt_path.exists():
                continue

            tensor = torch.load(pt_path, weights_only=True)
            signals[sig_type] = tensor.squeeze(0).numpy()
            type_loaded.add(sig_type)

        if not signals:
            continue

        records.append({
            "record": rec_info.get("record", ""),
            "alarm_type": alarm_type,
            "label": label,
            "signals": signals,
        })

    return records


def prepare_dataset(
    records: list[dict],
    input_signals: list[str],
    window_sec: float,
    train_ratio: float,
    out_dir: Path,
) -> Path:
    """window_sec에 맞게 잘라서 train/test .pt를 저장한다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    win_samples = int(window_sec * TARGET_SR)

    # 윈도우 트리밍 + 길이 통일
    trimmed = []
    for rec in records:
        sigs = {}
        for stype, signal in rec["signals"].items():
            if len(signal) >= win_samples:
                sigs[stype] = signal[-win_samples:]
            else:
                # 짧으면 앞에 zero-pad
                pad = np.zeros(win_samples - len(signal), dtype=signal.dtype)
                sigs[stype] = np.concatenate([pad, signal])
        trimmed.append({**rec, "signals": sigs})

    # Train/Test split (record-level, 고정 시드)
    rng = np.random.default_rng(42)
    indices = np.arange(len(trimmed))
    rng.shuffle(indices)

    n_train = max(1, int(len(trimmed) * train_ratio))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    def _to_tensors(idxs: np.ndarray) -> dict:
        subset = [trimmed[i] for i in idxs]
        if not subset:
            return {"signals": {}, "labels": torch.tensor([]),
                    "alarm_types": [], "records": []}

        sig_tensors = {}
        for stype in input_signals:
            arrs = [s["signals"][stype] for s in subset if stype in s["signals"]]
            if arrs:
                sig_tensors[stype] = torch.stack(
                    [torch.from_numpy(a).float() for a in arrs]
                )

        return {
            "signals": sig_tensors,
            "labels": torch.tensor([s["label"] for s in subset], dtype=torch.long),
            "alarm_types": [s["alarm_type"] for s in subset],
            "records": [s["record"] for s in subset],
        }

    train_data = _to_tensors(train_idx)
    test_data = _to_tensors(test_idx)

    # 알람 분포 통계
    train_alarm_dist = dict(Counter(train_data["alarm_types"]))
    test_alarm_dist = dict(Counter(test_data["alarm_types"]))

    save_dict = {
        "train": train_data,
        "test": test_data,
        "metadata": {
            "task": "icu_false_alarm_reduction",
            "source": "PhysioNet-Challenge-2015",
            "input_signals": input_signals,
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "train_ratio": train_ratio,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_train_true": int(train_data["labels"].sum().item()),
            "n_test_true": int(test_data["labels"].sum().item()),
            "train_alarm_dist": train_alarm_dist,
            "test_alarm_dist": test_alarm_dist,
        },
    }

    sig_str = "_".join(input_signals)
    win_int = int(window_sec)
    filename = f"false_alarm_{sig_str}_w{win_int}s.pt"
    save_path = out_dir / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.1f} MB)")
    return save_path


def print_stats(name: str, data: dict) -> None:
    labels = data["labels"]
    n = len(labels)
    if n == 0:
        print(f"  {name}: 0 samples")
        return
    n_true = int(labels.sum().item())
    n_false = n - n_true
    alarm_dist = dict(Counter(data["alarm_types"]))
    print(f"  {name}: {n} samples (True={n_true}, False={n_false})")
    for atype, cnt in sorted(alarm_dist.items()):
        print(f"    {atype}: {cnt}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICU False Alarm Reduction — Data Preparation"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Parsed data directory (manifest.json + .pt files)",
    )
    parser.add_argument(
        "--input-signals", nargs="+", default=["ecg", "ppg", "abp"],
        choices=["ecg", "ppg", "abp"],
    )
    parser.add_argument(
        "--window-secs", nargs="+", type=float, default=[60],
        help="Pre-alarm window lengths (seconds). 예: 30 60 120 300",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory (기본: data-dir과 동일)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir

    # 전체 레코드 한 번만 로드
    print(f"Loading records from: {data_dir}")
    print(f"  Input signals: {args.input_signals}")
    records = load_all_records(data_dir, args.input_signals)

    if not records:
        print("ERROR: No records loaded.", file=sys.stderr)
        sys.exit(1)

    n_true = sum(1 for r in records if r["label"] == 1)
    print(f"  Total: {len(records)} records (True={n_true}, False={len(records)-n_true})")

    # 신호 조합 분포
    sig_combos = Counter(tuple(sorted(r["signals"].keys())) for r in records)
    for combo, cnt in sig_combos.most_common():
        print(f"    {' + '.join(combo)}: {cnt}")

    # 각 window_sec에 대해 데이터셋 생성
    print(f"\nGenerating datasets for window_secs={args.window_secs}")
    for wsec in args.window_secs:
        print(f"\n--- window_sec={wsec}s ---")
        save_path = prepare_dataset(
            records, args.input_signals, wsec, args.train_ratio, out_dir,
        )

        # 저장된 데이터 검증
        saved = torch.load(save_path, weights_only=False)
        print_stats("Train", saved["train"])
        print_stats("Test", saved["test"])

    print(f"\nDone! Output: {out_dir}")


if __name__ == "__main__":
    main()
