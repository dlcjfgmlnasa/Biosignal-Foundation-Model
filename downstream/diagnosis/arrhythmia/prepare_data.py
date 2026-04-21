# -*- coding:utf-8 -*-
"""Arrhythmia Classification - 데이터 준비 스크립트 (MIMIC-III-Ext-PPG 원본).

30초 segment의 현재 rhythm class 분류 (5-class).
선별 다운로드된 wfdb 원본(.dat/.hea) + metadata.csv 라벨을 직접 읽는다.

5-class:
    0: SR    (Sinus Rhythm)
    1: AF    (Atrial Fibrillation)
    2: STACH (Sinus Tachycardia)
    3: SBRAD (Sinus Bradycardia)
    4: AFLT  (Atrial Flutter)

입력 신호: PPG(PLETH) + ECG(II) + ABP — MIMIC-III-Ext-PPG native 125Hz → 100Hz 리샘플.

분할 (strat_fold 0-9):
    fold 0-5 → train (60%)
    fold 6-7 → val   (20%)
    fold 8-9 → test  (20%)

사용법:
    python -m downstream.diagnosis.arrhythmia.prepare_data \
        --data-dir "C:/Users/SNUH_VitalLab_LEGION/Downloads/physionet.org/files/mimic-iii-ext-ppg/1.1.0" \
        --metadata "C:/.../metadata.csv" \
        --subset-csv downstream/diagnosis/arrhythmia/arrhythmia_subset_labels.csv \
        --out-dir outputs/downstream/arrhythmia \
        --max-segments-per-class 5000
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.parser._common import resample_to_target


TARGET_SR: float = 100.0
NATIVE_SR: float = 125.0
SEGMENT_SEC: float = 30.0

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

# SIGNAL_TYPE 정수 매핑 (foundation model과 정합)
SIGNAL_TYPE_MAP: dict[int, str] = {0: "ecg", 1: "abp", 2: "ppg"}

# wfdb 채널명 → 내부 signal type
CHANNEL_TO_STYPE: dict[str, str] = {
    "PLETH": "ppg",
    "II": "ecg",
    "II+": "ecg",
    "ABP": "abp",
    "ART": "abp",
}


@dataclass
class ArrhythmiaSample:
    """30초 segment arrhythmia classification 샘플."""

    signals: dict[str, np.ndarray]  # {"ecg": (3000,), "abp": (3000,), "ppg": (3000,)}
    label: int  # 0-4
    rhythm: str
    patient: str
    segment_id: str
    strat_fold: int


# ---- metadata 로딩 ----


def load_subset_patients(subset_csv: str) -> dict[str, str]:
    """arrhythmia_subset_labels.csv → {patient: primary_class}."""
    out: dict[str, str] = {}
    with open(subset_csv, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["patient"].strip()] = row["class"].strip()
    return out


def load_segment_metadata(
    metadata_csv: str,
    subset_patients: dict[str, str],
) -> list[dict]:
    """metadata.csv에서 subset 환자의 primary-class segment만 수집.

    각 환자의 primary rhythm과 일치하는 segment만 학습용으로 사용.

    Returns
    -------
    list of {"patient", "folder_path", "signal_file_name", "event_rhythm",
             "strat_fold", "segment_id"}
    """
    csv.field_size_limit(10**9)

    REQUIRED = (
        "patient", "folder_path", "signal_file_name", "event_rhythm",
        "strat_fold", "segment_id",
    )

    segments: list[dict] = []
    n_scanned = 0
    n_kept = 0
    with open(metadata_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: header.index(c) for c in REQUIRED}
        for row in reader:
            n_scanned += 1
            pt = row[idx["patient"]].strip()
            target_cls = subset_patients.get(pt)
            if target_cls is None:
                continue
            rh = row[idx["event_rhythm"]].strip()
            if rh != target_cls:
                continue
            try:
                fold = int(row[idx["strat_fold"]].strip())
            except ValueError:
                continue
            segments.append({
                "patient": pt,
                "folder_path": row[idx["folder_path"]].strip(),
                "signal_file_name": row[idx["signal_file_name"]].strip(),
                "event_rhythm": rh,
                "strat_fold": fold,
                "segment_id": row[idx["segment_id"]].strip(),
            })
            n_kept += 1
            if n_scanned % 1_000_000 == 0:
                print(f"  scanned {n_scanned:,}, kept {n_kept:,}...")

    print(f"  scanned {n_scanned:,} segments, kept {n_kept:,} for subset patients")
    return segments


# ---- signal 로딩 ----


def load_segment_signals(
    data_dir: Path,
    folder_path: str,
    signal_file_name: str,
    required_signals: tuple[str, ...] = ("ecg", "abp", "ppg"),
) -> dict[str, np.ndarray] | None:
    """wfdb로 30초 segment를 읽고 125Hz → 100Hz 리샘플.

    Returns
    -------
    {"ecg": (3000,), "abp": (3000,), "ppg": (3000,)} 또는 None(실패).
    """
    import wfdb

    rec_path = data_dir / folder_path
    rec_name = signal_file_name

    # record_base: folder_path는 pXX/pXXXXXX/signal_file_name 형태
    # wfdb.rdrecord는 .dat/.hea 없는 base path를 받음
    full_base = (rec_path.parent / rec_name) if rec_path.is_dir() else (
        data_dir / Path(folder_path).parent / rec_name
    )

    try:
        rec = wfdb.rdrecord(str(full_base))
    except Exception:
        return None

    if rec.p_signal is None or rec.sig_name is None:
        return None

    # 각 required signal type 추출
    signals: dict[str, np.ndarray] = {}
    for ch_idx, ch_name in enumerate(rec.sig_name):
        stype = CHANNEL_TO_STYPE.get(ch_name)
        if stype is None or stype in signals:
            continue
        sig = rec.p_signal[:, ch_idx].astype(np.float32)
        if np.isnan(sig).all():
            continue
        signals[stype] = sig

    if not all(s in signals for s in required_signals):
        return None

    # 30초 × 125Hz = 3750 samples 기대 → 100Hz 리샘플 시 3000 samples
    expected_native = int(SEGMENT_SEC * NATIVE_SR)
    min_native = int(SEGMENT_SEC * NATIVE_SR * 0.95)  # 허용 오차 5%
    out: dict[str, np.ndarray] = {}
    target_len = int(SEGMENT_SEC * TARGET_SR)

    for stype, sig in signals.items():
        if len(sig) < min_native:
            return None
        sig = sig[:expected_native] if len(sig) >= expected_native else sig

        # NaN 비율 체크
        nan_ratio = np.isnan(sig).mean() if len(sig) > 0 else 1.0
        if nan_ratio > 0.1:
            return None
        # NaN을 0으로 치환 후 리샘플
        sig = np.nan_to_num(sig, nan=0.0)

        sig = resample_to_target(sig, NATIVE_SR, TARGET_SR)
        if len(sig) < target_len:
            return None
        out[stype] = sig[:target_len].astype(np.float32)

    return out


# ---- 샘플 빌드 ----


def build_samples(
    data_dir: Path,
    segments: list[dict],
    required_signals: tuple[str, ...],
    max_per_class: int | None,
    verbose: bool = True,
) -> list[ArrhythmiaSample]:
    """segment metadata를 따라 신호를 로드해 ArrhythmiaSample 리스트 구축."""
    # class별 카운팅 (balance 상한)
    per_class_count: Counter = Counter()
    samples: list[ArrhythmiaSample] = []
    n_fail_load = 0
    n_cap_skip = 0

    total = len(segments)
    for i, seg in enumerate(segments, 1):
        cls = seg["event_rhythm"]
        if max_per_class is not None and per_class_count[cls] >= max_per_class:
            n_cap_skip += 1
            continue

        sig_dict = load_segment_signals(
            data_dir,
            seg["folder_path"],
            seg["signal_file_name"],
            required_signals=required_signals,
        )
        if sig_dict is None:
            n_fail_load += 1
            continue

        samples.append(ArrhythmiaSample(
            signals=sig_dict,
            label=RHYTHM_TO_LABEL[cls],
            rhythm=cls,
            patient=seg["patient"],
            segment_id=seg["segment_id"],
            strat_fold=seg["strat_fold"],
        ))
        per_class_count[cls] += 1

        if verbose and i % 500 == 0:
            print(
                f"  [{i}/{total}] loaded={len(samples)}, "
                f"fail={n_fail_load}, cap_skip={n_cap_skip}, "
                f"per-class={dict(per_class_count)}"
            )

    print(f"  Final: {len(samples)} samples, fail_load={n_fail_load}, cap_skip={n_cap_skip}")
    print(f"  Per-class: {dict(per_class_count)}")
    return samples


# ---- 분할 & 저장 ----


def split_by_fold(
    samples: list[ArrhythmiaSample],
) -> tuple[list, list, list]:
    """strat_fold 기반 분할: 0-5 train, 6-7 val, 8-9 test."""
    train, val, test = [], [], []
    for s in samples:
        if s.strat_fold <= 5:
            train.append(s)
        elif s.strat_fold <= 7:
            val.append(s)
        else:
            test.append(s)
    return train, val, test


def save_dataset(
    train: list[ArrhythmiaSample],
    val: list[ArrhythmiaSample],
    test: list[ArrhythmiaSample],
    input_signals: tuple[str, ...],
    out_dir: str,
) -> Path:
    """저장 포맷: spec대로 list[dict] (Sample마다 dict 하나)."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_list(split: list[ArrhythmiaSample]) -> list[dict]:
        return [
            {
                "signals": {st: s.signals[st] for st in input_signals},
                "label": s.label,
                "rhythm": s.rhythm,
                "patient": s.patient,
                "segment_id": s.segment_id,
                "strat_fold": s.strat_fold,
            }
            for s in split
        ]

    save_dict = {
        "train": _to_list(train),
        "val": _to_list(val),
        "test": _to_list(test),
        "metadata": {
            "task": "arrhythmia_5class",
            "source": "MIMIC-III-Ext-PPG",
            "classes": CLASS_NAMES,
            "rhythm_to_label": RHYTHM_TO_LABEL,
            "input_signals": list(input_signals),
            "signal_type_map": SIGNAL_TYPE_MAP,
            "sampling_rate": TARGET_SR,
            "segment_sec": SEGMENT_SEC,
            "split": "fold 0-5 train, 6-7 val, 8-9 test",
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
    }

    filename = "arrhythmia_mimic3extppg_5class.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)
    size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({size_mb:.2f} MB)")
    return save_path


def print_split_stats(name: str, samples: list[ArrhythmiaSample]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return
    dist = Counter(s.label for s in samples)
    n_pt = len({s.patient for s in samples})
    print(f"  {name}: {len(samples)} samples ({n_pt} patients)")
    for cid, cname in enumerate(CLASS_NAMES):
        cnt = dist.get(cid, 0)
        pct = 100 * cnt / len(samples) if samples else 0.0
        print(f"    {cname}({cid}): {cnt} ({pct:.1f}%)")


# ---- CLI ----


def main() -> None:
    default_data = (
        "C:/Users/SNUH_VitalLab_LEGION/Downloads/physionet.org/"
        "files/mimic-iii-ext-ppg/1.1.0"
    )
    default_meta = default_data + "/metadata.csv"

    parser = argparse.ArgumentParser(
        description="Arrhythmia 5-class Classification (MIMIC-III-Ext-PPG native)"
    )
    parser.add_argument("--data-dir", type=str, default=default_data,
                        help="MIMIC-III-Ext-PPG 원본 루트 (pXX/pXXXXXX/*.dat)")
    parser.add_argument("--metadata", type=str, default=default_meta,
                        help="metadata.csv 경로")
    parser.add_argument("--subset-csv", type=str,
                        default="downstream/diagnosis/arrhythmia/arrhythmia_subset_labels.csv",
                        help="build_subset.py가 생성한 subset labels CSV")
    parser.add_argument("--out-dir", type=str,
                        default="outputs/downstream/arrhythmia")
    parser.add_argument("--input-signals", nargs="+",
                        default=["ecg", "abp", "ppg"],
                        choices=["ecg", "abp", "ppg"],
                        help="사용할 신호 타입")
    parser.add_argument("--max-segments-per-class", type=int, default=None,
                        help="class당 segment 상한 (balance 강화)")
    args = parser.parse_args()

    input_signals = tuple(args.input_signals)
    mode_str = " + ".join(s.upper() for s in input_signals)
    print("=" * 60)
    print(f"  Arrhythmia 5-class: {mode_str}")
    print(f"  Data: {args.data_dir}")
    print(f"  Subset: {args.subset_csv}")
    print(f"  Classes: {CLASS_NAMES}")
    print("=" * 60)

    # 1. subset 로드
    print("\n[1/4] Loading subset patients...")
    subset = load_subset_patients(args.subset_csv)
    print(f"  Subset patients: {len(subset)}")
    subset_cls_dist = Counter(subset.values())
    print(f"  Subset class distribution: {dict(subset_cls_dist)}")

    # 2. metadata.csv 필터링
    print("\n[2/4] Filtering metadata.csv for subset segments...")
    segments = load_segment_metadata(args.metadata, subset)
    if not segments:
        print("ERROR: no segments matched subset.", file=sys.stderr)
        sys.exit(1)
    seg_class_dist = Counter(s["event_rhythm"] for s in segments)
    print(f"  Segment class distribution: {dict(seg_class_dist)}")

    # 3. wfdb 로딩 → sample build
    print("\n[3/4] Loading waveforms + resampling...")
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data_dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)
    samples = build_samples(
        data_dir, segments, input_signals,
        max_per_class=args.max_segments_per_class,
    )
    if not samples:
        print("ERROR: no samples loaded.", file=sys.stderr)
        sys.exit(1)

    # 4. 분할 + 저장
    print("\n[4/4] Splitting by strat_fold + saving...")
    train, val, test = split_by_fold(samples)
    print_split_stats("Train (folds 0-5)", train)
    print_split_stats("Val   (folds 6-7)", val)
    print_split_stats("Test  (folds 8-9)", test)

    save_path = save_dataset(train, val, test, input_signals, args.out_dir)
    print("\n" + "=" * 60)
    print(f"  Done: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
