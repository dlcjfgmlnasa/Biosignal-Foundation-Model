# -*- coding:utf-8 -*-
"""MIMIC-III Waveform 다운로드 — ICU Mortality Prediction용.

mortality cohort CSV + RECORDS-waveforms → ICU stay 중 waveform 다운로드.

Sepsis task와의 차이:
  - Sepsis: onset ± 24h/6h 윈도우만 다운로드
  - Mortality: ICU stay 전체 기간의 waveform 다운로드
  - Mortality: 모든 ICU 유닛 포함 (sepsis는 5개 유닛만)

사용법:
    # 파일럿 (사망 50명 + 생존 50명)
    python -m downstream.safety.mortality.download_waveforms \
        --cohort-csv downstream/classification/mortality/icu_mortality_cohort.csv \
        --records-file downstream/classification/sepsis/RECORDS-waveforms \
        --out-dir datasets/raw/mimic3-waveform-mortality \
        --max-dead 50 --max-alive 50

    # 전체
    python -m downstream.safety.mortality.download_waveforms \
        --cohort-csv downstream/classification/mortality/icu_mortality_cohort.csv \
        --records-file downstream/classification/sepsis/RECORDS-waveforms \
        --out-dir datasets/raw/mimic3-waveform-mortality
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path


def parse_record_datetime(record_path: str) -> tuple[int, datetime | None]:
    """레코드 경로에서 subject_id와 시작 시간을 추출한다."""
    parts = record_path.strip().split("/")
    if len(parts) < 3:
        return 0, None

    folder = parts[1]  # p000020
    sid = int(folder[1:])

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

    for sid in index:
        index[sid].sort(key=lambda x: x[1])

    return index


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


def load_cohort(
    cohort_csv: str,
    wf_index: dict[int, list],
) -> tuple[list[dict], list[dict]]:
    """cohort CSV에서 waveform이 있는 사망/생존 환자를 분리한다."""
    dead = []
    alive = []

    with open(cohort_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = int(row["subject_id"])
            if sid not in wf_index:
                continue

            rec = {
                "subject_id": sid,
                "icustay_id": row["icustay_id"],
                "mortality": int(row.get("hospital_expire_flag", "0")),
                "icu_intime": row.get("icu_intime", ""),
                "icu_outtime": row.get("icu_outtime", ""),
                "first_careunit": row.get("first_careunit", ""),
                "age": row.get("age", ""),
                "gender": row.get("gender", ""),
            }

            if rec["mortality"] == 1:
                dead.append(rec)
            else:
                alive.append(rec)

    return dead, alive


def select_records_for_patient(
    patient: dict,
    wf_records: list[tuple[str, datetime]],
) -> list[str]:
    """ICU stay 기간 내의 waveform 레코드를 선택한다."""
    icu_in = parse_datetime_str(patient["icu_intime"])
    icu_out = parse_datetime_str(patient["icu_outtime"])

    if icu_in is None:
        return []

    # icu_outtime이 없으면 intime + 7일을 최대로
    if icu_out is None:
        icu_out = icu_in + timedelta(days=7)

    selected = []
    for rec_path, rec_dt in wf_records:
        if icu_in <= rec_dt <= icu_out:
            selected.append(rec_path)

    return selected


def download_record(record_path: str, out_dir: Path) -> bool:
    """wfdb.dl_database로 multi-segment 레코드를 다운로드한다."""
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        return False

    parts = record_path.split("/")
    db_subdir = f"mimic3wdb-matched/{parts[0]}/{parts[1]}"
    patient_dir = out_dir / parts[0] / parts[1]

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
        description="MIMIC-III Waveform download for ICU Mortality Prediction"
    )
    parser.add_argument(
        "--cohort-csv", type=str, required=True,
        help="Mortality cohort CSV (hospital_expire_flag 포함)",
    )
    parser.add_argument(
        "--records-file", type=str, required=True,
        help="RECORDS-waveforms file (sepsis task에서 공유)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="datasets/raw/mimic3-waveform-mortality",
    )
    parser.add_argument(
        "--max-dead", type=int, default=None,
        help="최대 사망 환자 수",
    )
    parser.add_argument(
        "--max-alive", type=int, default=None,
        help="최대 생존 환자 수",
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

    # 2. Cohort 로드
    print("Loading cohort...")
    dead, alive = load_cohort(args.cohort_csv, wf_index)
    print(f"  Dead with waveform: {len(dead)}")
    print(f"  Alive with waveform: {len(alive)}")

    # 3. 샘플 제한
    if args.max_dead is not None:
        dead = dead[: args.max_dead]
    if args.max_alive is not None:
        alive = alive[: args.max_alive]

    all_patients = dead + alive
    print(f"\nTarget: {len(dead)} dead + {len(alive)} alive "
          f"= {len(all_patients)} patients")

    # 4. 환자별 레코드 선택 → 평탄화하여 병렬 다운로드
    total_records = 0
    skipped_patients = 0
    patient_to_selected: dict[int, list[str]] = {}
    patient_meta: dict[int, dict] = {}

    for patient in all_patients:
        sid = patient["subject_id"]
        wf_records = wf_index.get(sid, [])
        selected = select_records_for_patient(patient, wf_records)
        if not selected:
            skipped_patients += 1
            continue
        total_records += len(selected)
        patient_to_selected[sid] = selected
        patient_meta[sid] = patient

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
            "task": "icu_mortality_prediction",
            "n_patients": len(manifest),
            "n_dead": sum(1 for m in manifest if m["mortality"] == 1),
            "n_alive": sum(1 for m in manifest if m["mortality"] == 0),
            "total_records": total_records,
            "downloaded": downloaded,
            "failed": failed,
            "skipped_no_records": skipped,
            "patients": manifest,
        }, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Download Complete")
    print(f"  Patients: {len(manifest)} "
          f"({sum(1 for m in manifest if m['mortality']==1)} dead)")
    print(f"  Records: {downloaded} downloaded, {failed} failed, "
          f"{skipped} skipped")
    print(f"  Manifest: {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
