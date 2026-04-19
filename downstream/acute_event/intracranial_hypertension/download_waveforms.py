# -*- coding:utf-8 -*-
"""MIMIC-III Waveform 다운로드 — ICP 채널 포함 레코드만.

RECORDS-waveforms에서 각 레코드의 헤더를 읽어 ICP 채널 존재 여부를 확인하고,
ICP가 있는 레코드만 선택적으로 다운로드한다.

MIMIC-III에서 ICP는 매우 드물므로 (주로 신경외과/TBI 환자),
먼저 scan으로 ICP 레코드를 찾고, 그 후 다운로드한다.

사용법:
    # Step 1: ICP 레코드 스캔 (헤더만 읽어 목록 생성)
    python -m downstream.acute_event.intracranial_hypertension.download_waveforms \
        scan --records-file downstream/classification/sepsis/RECORDS-waveforms

    # Step 2: 다운로드
    python -m downstream.acute_event.intracranial_hypertension.download_waveforms \
        download --icp-records-file downstream/classification/intracranial_hypertension/ICP-RECORDS \
        --out-dir datasets/raw/mimic3-waveform-ich
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PHYSIONET_BASE = "https://physionet.org/files/mimic3wdb-matched/1.0"


def scan_icp_records(
    records_file: str,
    out_file: str = "downstream/classification/intracranial_hypertension/ICP-RECORDS",
    max_scan: int | None = None,
) -> None:
    """RECORDS-waveforms에서 ICP 채널이 있는 레코드를 찾는다.

    각 레코드의 .hea 파일을 원격으로 읽어 채널 목록을 확인한다.
    """
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    with open(records_file, "r") as f:
        all_records = [line.strip() for line in f if line.strip()]

    if max_scan is not None:
        all_records = all_records[:max_scan]

    print(f"Scanning {len(all_records)} records for ICP channels...")

    icp_records: list[str] = []
    n_scanned = 0
    n_errors = 0

    for i, record_path in enumerate(all_records):
        parts = record_path.split("/")
        if len(parts) < 3:
            continue

        # 헤더만 읽어서 채널 확인
        db_name = f"mimic3wdb-matched/{parts[0]}/{parts[1]}"
        rec_name = parts[-1]

        try:
            # rdheader로 헤더만 읽기 (데이터 다운로드 없음)
            hdr = wfdb.rdheader(rec_name, pn_dir=db_name)
            sig_names = hdr.sig_name if hdr.sig_name else []

            if "ICP" in sig_names:
                icp_records.append(record_path)
                print(f"  [{i + 1}] ICP found: {record_path} "
                      f"(signals: {sig_names})")

        except Exception:
            n_errors += 1

        n_scanned += 1
        if (i + 1) % 500 == 0:
            print(f"  Scanned {i + 1}/{len(all_records)}, "
                  f"ICP found: {len(icp_records)}, errors: {n_errors}")

    # 저장
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in icp_records:
            f.write(rec + "\n")

    print(f"\n{'=' * 60}")
    print(f"  ICP Record Scan Complete")
    print(f"  Scanned: {n_scanned}")
    print(f"  ICP records: {len(icp_records)}")
    print(f"  Errors: {n_errors}")
    print(f"  Output: {out_path}")
    print(f"{'=' * 60}")


def download_icp_records(
    icp_records_file: str,
    out_dir: str = "datasets/raw/mimic3-waveform-ich",
    max_records: int | None = None,
) -> None:
    """ICP 레코드 목록에서 waveform을 다운로드한다."""
    try:
        import wfdb
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    with open(icp_records_file, "r") as f:
        records = [line.strip() for line in f if line.strip()]

    if max_records is not None:
        records = records[:max_records]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(records)} ICP records to {out_dir}...")

    downloaded = 0
    failed = 0
    skipped = 0

    for i, record_path in enumerate(records):
        parts = record_path.split("/")
        db_subdir = f"mimic3wdb-matched/{parts[0]}/{parts[1]}"
        rec_name = parts[-1]
        patient_dir = out_path / parts[0] / parts[1]

        # 이미 다운된 경우 스킵
        if (patient_dir / f"{rec_name}.hea").exists():
            skipped += 1
            continue

        try:
            wfdb.dl_database(
                db_subdir,
                dl_dir=str(out_path),
                records=[rec_name],
                overwrite=False,
            )
            downloaded += 1
        except Exception as e:
            print(f"  FAIL {record_path}: {e}")
            failed += 1

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i + 1}/{len(records)}] downloaded={downloaded}, "
                  f"skipped={skipped}, failed={failed}")

    print(f"\n{'=' * 60}")
    print(f"  Download Complete")
    print(f"  Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")
    print(f"  Output: {out_dir}")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-III ICP Waveform Scanner & Downloader"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    scan_parser = subparsers.add_parser("scan", help="Scan for ICP records")
    scan_parser.add_argument(
        "--records-file", type=str, required=True,
        help="RECORDS-waveforms file",
    )
    scan_parser.add_argument(
        "--out-file", type=str,
        default="downstream/classification/intracranial_hypertension/ICP-RECORDS",
    )
    scan_parser.add_argument("--max-scan", type=int, default=None)

    # download
    dl_parser = subparsers.add_parser("download", help="Download ICP records")
    dl_parser.add_argument(
        "--icp-records-file", type=str, required=True,
        help="ICP-RECORDS file (scan 결과)",
    )
    dl_parser.add_argument(
        "--out-dir", type=str,
        default="datasets/raw/mimic3-waveform-ich",
    )
    dl_parser.add_argument("--max-records", type=int, default=None)

    args = parser.parse_args()

    if args.command == "scan":
        scan_icp_records(
            records_file=args.records_file,
            out_file=args.out_file,
            max_scan=args.max_scan,
        )
    elif args.command == "download":
        download_icp_records(
            icp_records_file=args.icp_records_file,
            out_dir=args.out_dir,
            max_records=args.max_records,
        )


if __name__ == "__main__":
    main()
