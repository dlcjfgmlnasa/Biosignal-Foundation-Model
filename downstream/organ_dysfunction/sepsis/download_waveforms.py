# -*- coding:utf-8 -*-
"""MIMIC-III Waveform 선택적 다운로드 — Sepsis onset 주변만.

sepsis3_cohort CSV + RECORDS-waveforms → onset 주변 레코드만 다운로드.

각 환자에 대해:
  - Sepsis+: onset 전 24시간 ~ onset 후 6시간 범위의 레코드
  - Sepsis-: ICU stay 중 랜덤 24시간 구간의 레코드

사용법:
    # 파일럿 (sepsis+ 50명 + sepsis- 50명)
    python -m downstream.organ_dysfunction.sepsis.download_waveforms \
        --cohort-csv downstream/classification/sepsis/sepsis3_cohort.csv \
        --records-file downstream/classification/sepsis/RECORDS-waveforms \
        --out-dir datasets/raw/mimic3-waveform-sepsis \
        --max-sepsis 50 --max-nonsepsis 50

    # 전체 다운로드
    python -m downstream.organ_dysfunction.sepsis.download_waveforms \
        --cohort-csv downstream/classification/sepsis/sepsis3_cohort.csv \
        --records-file downstream/classification/sepsis/RECORDS-waveforms \
        --out-dir datasets/raw/mimic3-waveform-sepsis
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path


PHYSIONET_BASE = "https://physionet.org/files/mimic3wdb-matched/1.0"


def parse_record_datetime(record_path: str) -> tuple[int, datetime | None]:
    """레코드 경로에서 subject_id와 시작 시간을 추출한다.

    예: p00/p000020/p000020-2183-04-28-17-47
    → subject_id=20, datetime=2183-04-28 17:47
    """
    parts = record_path.strip().split("/")
    if len(parts) < 3:
        return 0, None

    folder = parts[1]  # p000020
    sid = int(folder[1:])

    # 파일명에서 날짜 추출: p000020-2183-04-28-17-47
    fname = parts[-1]
    m = re.match(r"p\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", fname)
    if m is None:
        return sid, None

    try:
        dt = datetime(
            int(m.group(1)), int(m.group(2)), int(m.group(3)),
            int(m.group(4)), int(m.group(5)),
        )
    except ValueError:
        return sid, None

    return sid, dt


def load_waveform_index(records_file: str) -> dict[int, list[tuple[str, datetime]]]:
    """RECORDS-waveforms → {subject_id: [(record_path, datetime), ...]}"""
    index: dict[int, list[tuple[str, datetime]]] = {}

    with open(records_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, dt = parse_record_datetime(line)
            if sid == 0 or dt is None:
                continue
            if sid not in index:
                index[sid] = []
            index[sid].append((line, dt))

    # 시간순 정렬
    for sid in index:
        index[sid].sort(key=lambda x: x[1])

    return index


def load_cohort(
    cohort_csv: str,
    wf_index: dict[int, list],
) -> tuple[list[dict], list[dict]]:
    """cohort CSV에서 waveform이 있는 sepsis+/sepsis- 환자를 분리한다."""
    sepsis_pos = []
    sepsis_neg = []

    with open(cohort_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["subject_id"])
            if sid not in wf_index:
                continue

            rec = {
                "subject_id": sid,
                "icustay_id": row["icustay_id"],
                "sepsis3": int(row["sepsis3"]),
                "icu_intime": row["icu_intime"],
                "icu_outtime": row["icu_outtime"],
                "suspected_infection_time": row.get("suspected_infection_time", ""),
                "sofa_total": row.get("sofa_total", ""),
                "hospital_expire_flag": row.get("hospital_expire_flag", "0"),
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
            }

            if rec["sepsis3"] == 1:
                sepsis_pos.append(rec)
            else:
                sepsis_neg.append(rec)

    return sepsis_pos, sepsis_neg


def parse_datetime_str(s: str) -> datetime | None:
    """다양한 포맷의 datetime 문자열을 파싱한다."""
    if not s or s.strip() == "":
        return None
    s = s.strip().replace("T", " ").split(".")[0].replace(" UTC", "")
    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def select_records_for_patient(
    patient: dict,
    wf_records: list[tuple[str, datetime]],
    pre_hours: float = 24.0,
    post_hours: float = 6.0,
) -> list[str]:
    """환자의 관심 시간 윈도우에 해당하는 waveform 레코드를 선택한다.

    Sepsis+: onset 전 pre_hours ~ onset 후 post_hours
    Sepsis-: ICU intime 기준 첫 24+6시간
    """
    if patient["sepsis3"] == 1:
        center = parse_datetime_str(patient["suspected_infection_time"])
    else:
        center = parse_datetime_str(patient["icu_intime"])

    if center is None:
        return []

    window_start = center - timedelta(hours=pre_hours)
    window_end = center + timedelta(hours=post_hours)

    selected = []
    for rec_path, rec_dt in wf_records:
        if window_start <= rec_dt <= window_end:
            selected.append(rec_path)

    return selected


def download_record(record_path: str, out_dir: Path) -> bool:
    """wfdb.dl_database로 multi-segment 레코드를 다운로드한다.

    MIMIC-III Waveform은 MultiRecord 포맷이라 세그먼트별 .hea/.dat가 있음.
    wfdb.dl_database가 자동으로 모든 세그먼트를 받아줌.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        return False

    # record_path: p00/p000052/p000052-2191-01-10-02-21
    parts = record_path.split("/")
    # wfdb.dl_database의 db 인자는 버전 없이: mimic3wdb-matched/p00/p000052
    db_subdir = f"mimic3wdb-matched/{parts[0]}/{parts[1]}"
    patient_dir = out_dir / parts[0] / parts[1]

    # 이미 다운된 경우 스킵
    rec_name = parts[-1]
    if (patient_dir / f"{rec_name}.hea").exists():
        return True

    try:
        wfdb.dl_database(
            db_subdir,
            dl_dir=str(out_dir),
            records=[rec_name],
            overwrite=False,
        )
        return True
    except Exception as e:
        print(f"  FAIL {record_path}: {e}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-III Waveform selective download for sepsis prediction"
    )
    parser.add_argument(
        "--cohort-csv", type=str, required=True,
        help="sepsis3_cohort CSV (BigQuery 결과)",
    )
    parser.add_argument(
        "--records-file", type=str, required=True,
        help="RECORDS-waveforms file",
    )
    parser.add_argument(
        "--out-dir", type=str, default="datasets/raw/mimic3-waveform-sepsis",
    )
    parser.add_argument(
        "--max-sepsis", type=int, default=None,
        help="최대 sepsis+ 환자 수 (None=전부)",
    )
    parser.add_argument(
        "--max-nonsepsis", type=int, default=None,
        help="최대 sepsis- 환자 수 (None=전부)",
    )
    parser.add_argument(
        "--pre-hours", type=float, default=24.0,
        help="관심 윈도우: center 전 N시간",
    )
    parser.add_argument(
        "--post-hours", type=float, default=6.0,
        help="관심 윈도우: center 후 N시간",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="병렬 다운로드 worker 수 (기본 8). bandwidth 한계로 16 이상은 비추천.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Waveform 인덱스 로드
    print("Loading waveform index...")
    wf_index = load_waveform_index(args.records_file)
    print(f"  {len(wf_index)} subjects with waveforms")

    # 2. Cohort 로드 + 매칭
    print("Loading cohort...")
    sepsis_pos, sepsis_neg = load_cohort(args.cohort_csv, wf_index)
    print(f"  Sepsis+ with waveform: {len(sepsis_pos)}")
    print(f"  Sepsis- with waveform: {len(sepsis_neg)}")

    # 3. 샘플 제한
    if args.max_sepsis is not None:
        sepsis_pos = sepsis_pos[: args.max_sepsis]
    if args.max_nonsepsis is not None:
        sepsis_neg = sepsis_neg[: args.max_nonsepsis]

    all_patients = sepsis_pos + sepsis_neg
    print(f"\nTarget: {len(sepsis_pos)} sepsis+ + {len(sepsis_neg)} sepsis- "
          f"= {len(all_patients)} patients")

    # 4. 환자별 레코드 선택 → 모든 (patient, rec_path) 쌍 평탄화
    total_records = 0
    skipped_patients = 0
    patient_to_selected: dict[int, list[str]] = {}
    patient_meta: dict[int, dict] = {}

    for patient in all_patients:
        sid = patient["subject_id"]
        wf_records = wf_index.get(sid, [])
        selected = select_records_for_patient(
            patient, wf_records,
            pre_hours=args.pre_hours,
            post_hours=args.post_hours,
        )
        if not selected:
            skipped_patients += 1
            continue
        total_records += len(selected)
        patient_to_selected[sid] = selected
        patient_meta[sid] = patient

    # 평탄화: [(sid, rec_path), ...]
    pending = [
        (sid, rec) for sid, recs in patient_to_selected.items() for rec in recs
    ]
    print(
        f"\nDownloading {len(pending)} records for {len(patient_to_selected)} "
        f"patients with {args.workers} workers (skipped {skipped_patients} "
        f"patients with no records in window)..."
    )

    downloaded = 0
    failed = 0
    n_done = 0
    success_by_patient: dict[int, list[str]] = {sid: [] for sid in patient_to_selected}

    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {
            executor.submit(download_record, rec, out_dir): (sid, rec)
            for sid, rec in pending
        }
        for fut in as_completed(futures):
            sid, rec = futures[fut]
            try:
                ok = fut.result()
            except Exception as e:
                ok = False
                print(f"  FAIL {rec}: {e}")
            if ok:
                downloaded += 1
                success_by_patient[sid].append(rec)
            else:
                failed += 1
            n_done += 1
            if n_done % 10 == 0 or n_done == 1:
                print(
                    f"  [{n_done}/{len(pending)}] downloaded={downloaded}, "
                    f"failed={failed}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down workers...", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True)

    skipped = skipped_patients
    manifest = []
    for sid, patient in patient_meta.items():
        recs = success_by_patient[sid]
        manifest.append({
            **patient,
            "waveform_records": recs,
            "n_records": len(recs),
        })

    # 5. Manifest 저장
    manifest_path = out_dir / "download_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "n_patients": len(manifest),
            "n_sepsis_pos": sum(1 for m in manifest if m["sepsis3"] == 1),
            "n_sepsis_neg": sum(1 for m in manifest if m["sepsis3"] == 0),
            "total_records": total_records,
            "downloaded": downloaded,
            "failed": failed,
            "skipped_no_records": skipped,
            "pre_hours": args.pre_hours,
            "post_hours": args.post_hours,
            "patients": manifest,
        }, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Download Complete")
    print(f"  Patients: {len(manifest)} ({sum(1 for m in manifest if m['sepsis3']==1)} sepsis+)")
    print(f"  Records: {downloaded} downloaded, {failed} failed, {skipped} skipped")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
