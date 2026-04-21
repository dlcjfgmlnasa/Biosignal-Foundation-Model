# -*- coding:utf-8 -*-
"""Mortality cohort + RECORDS-waveforms → RECORDS-to-download 생성.

download_waveforms.py의 ICU stay 기간 기반 선택 로직을 dry-run으로
적용하여, 실제 다운로드 대상 record path 목록만 파일로 생성한다.

사용법:
    python -m downstream.safety.mortality.build_records_to_download \
        --cohort-csv downstream/safety/mortality/bquxjob_6a8255f2_19d9042b214.csv \
        --records-file downstream/organ_dysfunction/sepsis/RECORDS-waveforms \
        --out-file downstream/safety/mortality/RECORDS-to-download
"""
from __future__ import annotations

import argparse
from pathlib import Path

from downstream.safety.mortality.download_waveforms import (
    load_cohort,
    load_waveform_index,
    select_records_for_patient,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RECORDS-to-download for mortality task (dry-run)."
    )
    parser.add_argument("--cohort-csv", type=str, required=True)
    parser.add_argument("--records-file", type=str, required=True)
    parser.add_argument(
        "--out-file", type=str,
        default="downstream/safety/mortality/RECORDS-to-download",
    )
    args = parser.parse_args()

    print("Loading waveform index...")
    wf_index = load_waveform_index(args.records_file)
    print(f"  {len(wf_index)} subjects with waveforms")

    print("Loading cohort...")
    dead, alive = load_cohort(args.cohort_csv, wf_index)
    print(f"  Dead with waveform: {len(dead)}")
    print(f"  Alive with waveform: {len(alive)}")

    selected_records: set[str] = set()
    n_patients_with_records = 0

    for patient in dead + alive:
        sid = patient["subject_id"]
        recs = select_records_for_patient(patient, wf_index.get(sid, []))
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
