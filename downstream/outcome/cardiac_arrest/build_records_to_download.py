# -*- coding:utf-8 -*-
"""Cardiac Arrest cohort + RECORDS-waveforms → RECORDS-to-download 생성.

download_waveforms.py의 레코드 선택 로직(onset ± pre/post hours)과
동일한 규칙을 dry-run으로 적용하여, 실제 다운로드 대상 record path
목록만 파일로 떨어뜨린다.

사용법:
    python -m downstream.outcome.cardiac_arrest.build_records_to_download \
        --cohort-csv downstream/outcome/cardiac_arrest/bquxjob_cardiac_arrest_TODO.csv \
        --records-file downstream/outcome/cardiac_arrest/RECORDS-waveforms \
        --out-file downstream/outcome/cardiac_arrest/RECORDS-to-download
"""
from __future__ import annotations

import argparse
from pathlib import Path

from downstream.outcome.cardiac_arrest.download_waveforms import (
    load_cohort,
    load_waveform_index,
    select_records_for_patient,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RECORDS-to-download for cardiac_arrest task (dry-run)."
    )
    parser.add_argument("--cohort-csv", type=str, required=True)
    parser.add_argument("--records-file", type=str, required=True)
    parser.add_argument(
        "--out-file", type=str,
        default="downstream/outcome/cardiac_arrest/RECORDS-to-download",
    )
    parser.add_argument("--pre-hours", type=float, default=24.0)
    parser.add_argument("--post-hours", type=float, default=6.0)
    args = parser.parse_args()

    print("Loading waveform index...")
    wf_index = load_waveform_index(args.records_file)
    print(f"  {len(wf_index)} subjects with waveforms")

    print("Loading cohort...")
    arrest_pos, arrest_neg = load_cohort(args.cohort_csv, wf_index)
    print(f"  Cardiac Arrest+ with waveform: {len(arrest_pos)}")
    print(f"  Cardiac Arrest- with waveform: {len(arrest_neg)}")

    selected_records: set[str] = set()
    n_patients_with_records = 0

    for patient in arrest_pos + arrest_neg:
        sid = patient["subject_id"]
        recs = select_records_for_patient(
            patient, wf_index.get(sid, []),
            pre_hours=args.pre_hours,
            post_hours=args.post_hours,
        )
        if recs:
            n_patients_with_records += 1
            selected_records.update(recs)

    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in sorted(selected_records):
            f.write(rec + "\n")

    print(f"\n{'=' * 60}")
    print(f"  Patients with selected records: {n_patients_with_records}")
    print(f"  Unique records to download: {len(selected_records)}")
    print(f"  Output: {out_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
