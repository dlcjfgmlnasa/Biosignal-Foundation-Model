# -*- coding:utf-8 -*-
"""Arrhythmia Detection - 데이터 준비 스크립트.

PTB-XL 12-lead ECG를 다운로드하고 5-class superclass 분류용 데이터를 준비한다.
공식 10-fold split (train: fold 1-8, val: fold 9, test: fold 10) 사용.

5-class superclass:
    NORM(0) - Normal ECG
    MI(1)   - Myocardial Infarction
    STTC(2) - ST/T Change
    CD(3)   - Conduction Disturbance
    HYP(4)  - Hypertrophy

사용법:
    # 전체 다운로드 + 준비
    python -m downstream.arrhythmia.prepare_data --download --n-records 0

    # 일부만 다운로드 (테스트용)
    python -m downstream.arrhythmia.prepare_data --download --n-records 500

    # 이미 다운로드된 경우
    python -m downstream.arrhythmia.prepare_data
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch


# ---- 다운로드 ----


def download_ptbxl(
    out_dir: str = "datasets/ptb-xl/1.0.3",
    n_records: int = 0,
    verbose: bool = True,
) -> Path:
    """PTB-XL records100/ + 메타데이터를 다운로드한다.

    Parameters
    ----------
    out_dir : 저장 경로.
    n_records : 다운로드할 레코드 수. 0이면 전체 (21,799개).
    verbose : 진행 상황 출력.

    Returns
    -------
    데이터 디렉토리 경로.
    """
    BASE = "https://physionet.org/files/ptb-xl/1.0.3"
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. CSV 다운로드
    for fname in ["ptbxl_database.csv", "scp_statements.csv"]:
        dest = out_path / fname
        if not dest.exists():
            if verbose:
                print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(f"{BASE}/{fname}", str(dest))

    # 2. records100/ 다운로드
    csv_path = out_path / "ptbxl_database.csv"
    if not csv_path.exists():
        print("ERROR: ptbxl_database.csv not found.", file=sys.stderr)
        sys.exit(1)

    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        count = 0
        skipped = 0
        t0 = time.time()

        for row in reader:
            if 0 < n_records <= count:
                break

            path = row["filename_lr"]  # e.g. "records100/00000/00001_lr"
            for ext in [".hea", ".dat"]:
                url = f"{BASE}/{path}{ext}"
                dest = out_path / f"{path}{ext}"
                if dest.exists():
                    skipped += 1
                    continue
                os.makedirs(dest.parent, exist_ok=True)
                urllib.request.urlretrieve(url, str(dest))

            count += 1
            if verbose and count % 500 == 0:
                elapsed = time.time() - t0
                print(f"  Downloaded {count} records ({elapsed:.1f}s)...")

    if verbose:
        total = count
        print(f"  Done: {total} records ({skipped} already existed)")

    return out_path


# ---- 데이터 준비 ----


def prepare_arrhythmia_data(
    data_dir: str = "datasets/ptb-xl/1.0.3",
    lead: str = "II",
    out_dir: str = "outputs/downstream/arrhythmia",
    download: bool = False,
    n_records: int = 0,
    visualize: bool = False,
) -> Path:
    """PTB-XL을 로드하고 공식 split으로 .pt 저장한다.

    Parameters
    ----------
    data_dir : PTB-XL 데이터 디렉토리.
    lead : 사용할 ECG lead (기본 "II").
    out_dir : 저장 디렉토리.
    download : True면 다운로드 먼저 수행.
    n_records : 다운로드할 레코드 수. 0이면 전체.
    visualize : True면 시각화 생성.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. 다운로드
    if download:
        print("[1/4] Downloading PTB-XL...")
        download_ptbxl(data_dir, n_records)
    else:
        print("[1/4] Using existing PTB-XL data...")

    # 2. 공식 split으로 로드
    print("\n[2/4] Loading PTB-XL with official splits...")
    from data.parser.ptbxl import load_ptbxl_split, SUPERCLASS_NAMES

    splits = load_ptbxl_split(
        data_dir=data_dir,
        lead=lead,
        sampling_rate=100,
    )

    # 3. 통계 출력
    print(f"\n[3/4] Dataset statistics (Lead={lead}):")
    from collections import Counter

    for split_name in ["train", "val", "test"]:
        samples, labels = splits[split_name]
        n = len(samples)
        if n == 0:
            print(f"  {split_name}: 0 samples")
            continue

        dist = Counter(labels)
        print(f"  {split_name}: {n} samples")
        for cls_name in SUPERCLASS_NAMES:
            cls_id = SUPERCLASS_NAMES.index(cls_name)
            cnt = dist.get(cls_id, 0)
            pct = cnt / n * 100
            print(f"    {cls_name}({cls_id}): {cnt} ({pct:.1f}%)")

    # 4. .pt 저장
    print(f"\n[4/4] Saving to .pt...")

    save_dict = {
        "metadata": {
            "task": "arrhythmia_detection",
            "source": "PTB-XL v1.0.3",
            "lead": lead,
            "sampling_rate": 100,
            "n_classes": 5,
            "class_names": SUPERCLASS_NAMES,
            "signal_length": 1000,  # 10s x 100Hz
        },
    }

    for split_name in ["train", "val", "test"]:
        samples, labels = splits[split_name]
        if samples:
            save_dict[split_name] = {
                "signals": torch.stack(samples),       # (N, 1000)
                "labels": torch.tensor(labels, dtype=torch.long),  # (N,)
            }
        else:
            save_dict[split_name] = {
                "signals": torch.tensor([]),
                "labels": torch.tensor([], dtype=torch.long),
            }

    save_path = out_path / f"arrhythmia_ptbxl_{lead}.pt"
    torch.save(save_dict, save_path)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")

    # 시각화
    if visualize:
        _visualize(splits, SUPERCLASS_NAMES, out_path, lead)

    # 요약
    total = sum(len(splits[s][0]) for s in ["train", "val", "test"])
    print(f"\n{'='*50}")
    print(f"  Arrhythmia Detection data ready")
    print(f"  Total: {total} ECGs, 5 classes, Lead {lead}")
    print(f"  File: {save_path}")
    print(f"{'='*50}")

    return save_path


def _visualize(splits, class_names, out_dir, lead):
    """클래스별 ECG 예시 + 분포 시각화."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping.")
        return

    print("\nGenerating visualizations...")
    samples, labels = splits["train"]
    if not samples:
        return

    # 1. 클래스별 ECG 예시
    fig, axes = plt.subplots(5, 1, figsize=(12, 12), squeeze=False)
    fig.suptitle(f"PTB-XL ECG Examples (Lead {lead}, 100Hz)", fontsize=14)

    from collections import defaultdict
    by_class = defaultdict(list)
    for s, l in zip(samples, labels):
        by_class[l].append(s)

    for cls_id, cls_name in enumerate(class_names):
        ax = axes[cls_id, 0]
        if cls_id in by_class and by_class[cls_id]:
            ecg = by_class[cls_id][0].numpy()
            t = np.arange(len(ecg)) / 100.0
            ax.plot(t, ecg, linewidth=0.8, color="tab:blue")
        ax.set_ylabel("mV")
        ax.set_title(f"{cls_name} (class {cls_id})")
        ax.set_xlim(0, 10)

    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout()
    path1 = out_dir / "arrhythmia_ecg_examples.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # 2. 클래스 분포
    from collections import Counter
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for idx, split_name in enumerate(["train", "val", "test"]):
        ax = axes[idx]
        samps, labs = splits[split_name]
        if not labs:
            continue
        dist = Counter(labs)
        counts = [dist.get(i, 0) for i in range(5)]
        bars = ax.bar(class_names, counts, color=["green", "red", "orange", "purple", "brown"])
        ax.set_title(f"{split_name} (n={len(labs)})")
        ax.set_ylabel("Count")
        for bar, c in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(c), ha="center", fontsize=8)

    plt.suptitle("PTB-XL Class Distribution", fontsize=13)
    plt.tight_layout()
    path2 = out_dir / "arrhythmia_class_distribution.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path2}")


# ---- CLI ----


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arrhythmia Detection - PTB-XL Data Preparation",
    )
    parser.add_argument("--data-dir", type=str, default="datasets/ptb-xl/1.0.3")
    parser.add_argument("--lead", type=str, default="II")
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/arrhythmia")
    parser.add_argument("--download", action="store_true",
                        help="Download PTB-XL first")
    parser.add_argument("--n-records", type=int, default=0,
                        help="Number of records to download (0=all)")
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    prepare_arrhythmia_data(
        data_dir=args.data_dir,
        lead=args.lead,
        out_dir=args.out_dir,
        download=args.download,
        n_records=args.n_records,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
