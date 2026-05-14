# -*- coding:utf-8 -*-
"""Sepsis cohort + RECORDS-waveforms → RECORDS-to-download 생성.

download_waveforms.py의 레코드 선택 로직(onset ± pre/post hours)과
동일한 규칙을 dry-run으로 적용하여, 실제 다운로드 대상 record path
목록만 파일로 떨어뜨린다.

사용법:
    python -m downstream.outcome.sepsis.build_records_to_download \
        --cohort-csv downstream/outcome/sepsis/bquxjob_93e3c7c_19d8f609070.csv \
        --records-file downstream/outcome/sepsis/RECORDS-waveforms \
        --out-file downstream/outcome/sepsis/RECORDS-to-download
"""
from __future__ import annotations

import argparse
from pathlib import Path

from downstream.outcome.sepsis.download_waveforms import (
    load_cohort,
    load_waveform_index,
    select_records_for_patient,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build RECORDS-to-download for sepsis task (dry-run)."
    )
    parser.add_argument("--cohort-csv", type=str, required=True)
    parser.add_argument("--records-file", type=str, required=True)
    parser.add_argument(
        "--out-file", type=str,
        default="downstream/outcome/sepsis/RECORDS-to-download",
    )
    parser.add_argument("--pre-hours", type=float, default=24.0)
    parser.add_argument("--post-hours", type=float, default=0.0)
    parser.add_argument(
        "--horizons", type=float, nargs="+", default=[4, 6, 12, 24],
        help="Sepsis multi-horizon (paper 1 표준 4/6/12/24h)",
    )
    parser.add_argument(
        "--max-sepsis", type=int, default=None,
    )
    parser.add_argument(
        "--max-nonsepsis", type=int, default=None,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading waveform index...")
    wf_index = load_waveform_index(args.records_file)
    print(f"  {len(wf_index)} subjects with waveforms")

    print(f"Loading cohort (seed={args.seed})...")
    sepsis_pos, sepsis_neg = load_cohort(args.cohort_csv, wf_index, seed=args.seed)
    print(f"  Sepsis+ with waveform: {len(sepsis_pos)}")
    print(f"  Sepsis- with waveform: {len(sepsis_neg)}")

    if args.max_sepsis is not None:
        sepsis_pos = sepsis_pos[: args.max_sepsis]
    if args.max_nonsepsis is not None:
        sepsis_neg = sepsis_neg[: args.max_nonsepsis]

    horizons = sorted(set(args.horizons))
    print(f"Horizons: {horizons}")

    selected_records: set[str] = set()
    n_patients_with_records = 0

    for patient in sepsis_pos + sepsis_neg:
        sid = patient["subject_id"]
        recs = select_records_for_patient(
            patient, wf_index.get(sid, []),
            pre_hours=args.pre_hours,
            post_hours=args.post_hours,
            horizons=horizons,
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
