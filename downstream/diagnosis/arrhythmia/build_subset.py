# -*- coding:utf-8 -*-
"""MIMIC-III-Ext-PPG Arrhythmia 5-class balanced subset 구축.

`metadata.csv`(~4.9GB)를 스트리밍 스캔하여 SR/AF/STACH/SBRAD/AFLT 5 클래스에
대해 환자 단위 balanced subset을 선정한다.

전체 다운로드(200GB)는 비현실적이므로, 환자 폴더 단위로 선별 다운로드할
대상만 리스트업한다.

환자-rhythm primary assignment 원칙:
  환자는 여러 rhythm segment를 가질 수 있으므로, 각 환자를 한 클래스에만
  배정한다. 우선순위는 희귀 클래스 → 흔한 클래스 (AFLT > SBRAD > AF > STACH > SR).
  이렇게 하면 희귀 클래스가 흔한 클래스에 흡수되지 않는다.

strat_fold(0-9)는 metadata 자체 5-fold(실제 10-fold) 분할이며, fold별
balance를 자연스럽게 유지하도록 각 클래스에서 fold 비율로 샘플링한다.

사용법:
    python -m downstream.diagnosis.arrhythmia.build_subset \
        --metadata "C:/.../physionet.org/files/mimic-iii-ext-ppg/1.1.0/metadata.csv" \
        --per-class 200 \
        --out-dir downstream/diagnosis/arrhythmia
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


# Arrhythmia 5-class (희귀 → 흔한 순으로 primary assign)
TARGET_CLASSES = ("AFLT", "SBRAD", "AF", "STACH", "SR")

# metadata.csv 필수 컬럼
REQUIRED_COLS = (
    "event_rhythm", "patient", "folder_path", "strat_fold", "subject_id",
)


def scan_metadata(
    metadata_csv: str,
) -> tuple[dict[str, dict], Counter, Counter]:
    """metadata.csv를 스트리밍 스캔하여 환자별 집계를 만든다.

    Returns
    -------
    patient_info : {patient_id: {"folder_path": str,
                                  "subject_id": str,
                                  "rhythm_counts": Counter,
                                  "folds": Counter}}
    rhythm_total : {rhythm: n_segments}
    fold_total : {fold: n_segments}
    """
    csv.field_size_limit(10**9)

    patient_info: dict[str, dict] = {}
    rhythm_total: Counter = Counter()
    fold_total: Counter = Counter()

    with open(metadata_csv, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {c: header.index(c) for c in REQUIRED_COLS}

        n = 0
        for row in reader:
            rh = row[idx["event_rhythm"]].strip()
            pt = row[idx["patient"]].strip()
            fp = row[idx["folder_path"]].strip()
            sf = row[idx["strat_fold"]].strip()
            sid = row[idx["subject_id"]].strip()

            rhythm_total[rh] += 1
            fold_total[sf] += 1

            info = patient_info.get(pt)
            if info is None:
                # folder_path에서 상위 두 디렉토리 추출 (pXX/pXXXXXX)
                parts = fp.split("/")
                folder = (
                    f"{parts[0]}/{parts[1]}" if len(parts) >= 2 else ""
                )
                info = {
                    "folder_path": folder,
                    "subject_id": sid,
                    "rhythm_counts": Counter(),
                    "folds": Counter(),
                }
                patient_info[pt] = info
            info["rhythm_counts"][rh] += 1
            info["folds"][sf] += 1

            n += 1
            if n % 1_000_000 == 0:
                print(f"  scanned {n:,} segments, {len(patient_info):,} patients...")

    print(f"  scanned {n:,} segments total")
    return patient_info, rhythm_total, fold_total


def assign_primary_class(
    patient_info: dict[str, dict],
    target_classes: tuple[str, ...] = TARGET_CLASSES,
) -> dict[str, str]:
    """각 환자의 primary rhythm class 1개 배정.

    우선순위 규칙: target_classes 순서대로(희귀→흔한), 해당 rhythm segment가
    하나라도 있으면 primary로 배정. 5-class 어디에도 속하지 않으면 제외.
    """
    primary: dict[str, str] = {}
    for pt, info in patient_info.items():
        rc = info["rhythm_counts"]
        for cls in target_classes:
            if rc.get(cls, 0) > 0:
                primary[pt] = cls
                break
    return primary


def sample_balanced(
    patient_info: dict[str, dict],
    primary_class: dict[str, str],
    per_class: int,
    target_classes: tuple[str, ...] = TARGET_CLASSES,
    seed: int = 42,
) -> dict[str, list[str]]:
    """클래스별 per_class 환자를 strat_fold 비례로 샘플링.

    Returns
    -------
    selected : {class_name: [patient_id, ...]}
    """
    rng = random.Random(seed)

    class_to_patients: dict[str, list[str]] = {c: [] for c in target_classes}
    for pt, cls in primary_class.items():
        class_to_patients[cls].append(pt)

    selected: dict[str, list[str]] = {}
    for cls in target_classes:
        pts = class_to_patients[cls]
        if len(pts) <= per_class:
            selected[cls] = sorted(pts)
            continue

        # fold별 환자 그룹
        by_fold: dict[str, list[str]] = defaultdict(list)
        for pt in pts:
            # 환자가 여러 fold segment를 가지면 가장 흔한 fold를 환자 fold로
            folds = patient_info[pt]["folds"]
            pt_fold = folds.most_common(1)[0][0]
            by_fold[pt_fold].append(pt)

        # fold별 할당량: per_class를 fold 크기 비례 분배
        total = sum(len(v) for v in by_fold.values())
        chosen: list[str] = []
        for fold, fold_pts in by_fold.items():
            quota = max(1, round(per_class * len(fold_pts) / total))
            if quota >= len(fold_pts):
                chosen.extend(fold_pts)
            else:
                chosen.extend(rng.sample(fold_pts, quota))

        # 정확히 per_class 만큼으로 trim/보충
        if len(chosen) > per_class:
            chosen = rng.sample(chosen, per_class)
        elif len(chosen) < per_class:
            remaining = [p for p in pts if p not in set(chosen)]
            need = per_class - len(chosen)
            if remaining:
                chosen.extend(rng.sample(remaining, min(need, len(remaining))))

        selected[cls] = sorted(chosen)

    return selected


def write_outputs(
    patient_info: dict[str, dict],
    selected: dict[str, list[str]],
    primary_class: dict[str, str],
    out_dir: Path,
) -> None:
    """RECORDS-arrhythmia-subset + label CSV 저장."""
    out_dir.mkdir(parents=True, exist_ok=True)

    all_patients: list[tuple[str, str, str]] = []  # (patient, cls, folder)
    for cls, pts in selected.items():
        for pt in pts:
            folder = patient_info[pt]["folder_path"]
            if folder:
                all_patients.append((pt, cls, folder))

    all_patients.sort(key=lambda x: x[2])

    records_path = out_dir / "RECORDS-arrhythmia-subset"
    with open(records_path, "w") as f:
        for _, _, folder in all_patients:
            f.write(folder + "\n")

    labels_path = out_dir / "arrhythmia_subset_labels.csv"
    with open(labels_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "class", "folder_path", "subject_id", "n_segments"])
        for pt, cls, folder in all_patients:
            info = patient_info[pt]
            w.writerow([
                pt, cls, folder, info["subject_id"],
                sum(info["rhythm_counts"].values()),
            ])

    print(f"  wrote {records_path} ({len(all_patients)} folders)")
    print(f"  wrote {labels_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-III-Ext-PPG arrhythmia 5-class subset builder"
    )
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="metadata.csv 경로 (~4.9GB)",
    )
    parser.add_argument(
        "--per-class", type=int, default=200,
        help="클래스당 환자 수 (기본 200, 5-class 총 ~1000명)",
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="downstream/diagnosis/arrhythmia",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)

    print(f"Scanning metadata.csv...")
    patient_info, rhythm_total, fold_total = scan_metadata(args.metadata)
    print(f"  total patients: {len(patient_info)}")

    print(f"\nAssigning primary rhythm class (priority: {TARGET_CLASSES})...")
    primary = assign_primary_class(patient_info)
    cls_cnt = Counter(primary.values())
    print(f"  Primary-class patients available:")
    for cls in TARGET_CLASSES:
        print(f"    {cls:6s}: {cls_cnt.get(cls, 0):>5}")
    print(f"  (not in 5-class): {len(patient_info) - sum(cls_cnt.values())}")

    print(f"\nSampling {args.per_class} patients per class (seed={args.seed})...")
    selected = sample_balanced(
        patient_info, primary, args.per_class, seed=args.seed,
    )
    for cls in TARGET_CLASSES:
        print(f"  {cls:6s}: {len(selected[cls])}")

    print(f"\nWriting outputs to {out_dir}...")
    write_outputs(patient_info, selected, primary, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
