# -*- coding:utf-8 -*-
"""MIMIC-III Waveform 다운로드 — ICP 채널 포함 레코드만.

RECORDS-waveforms에서 각 레코드의 헤더를 읽어 ICP 채널 존재 여부를 확인하고,
ICP가 있는 레코드만 선택적으로 다운로드한다.

MIMIC-III에서 ICP는 매우 드물므로 (주로 신경외과/TBI 환자),
먼저 scan으로 ICP 레코드를 찾고, 그 후 다운로드한다.

사용법:
    # Step 1: ICP 레코드 스캔 (헤더만 읽어 목록 생성)
    python -m downstream.acute_event.intracranial_hypertension.download_waveforms \
        scan --records-file downstream/outcome/sepsis/RECORDS-waveforms

    # Step 2: 다운로드
    python -m downstream.acute_event.intracranial_hypertension.download_waveforms \
        download --icp-records-file downstream/acute_event/intracranial_hypertension/ICP-RECORDS \
        --out-dir datasets/raw/mimic3-waveform-ich
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PHYSIONET_BASE = "https://physionet.org/files/mimic3wdb-matched/1.0"


def get_all_channels(hdr, pn_dir: str) -> set:
    """Multi-segment record에서 모든 signal channel 이름 수집.

    MIMIC-III Waveform은 multi-segment 구조:
        - Master .hea: sig_name=None, seg_name에 하위 segments 목록
        - *_layout segment: 전체 channel list 포함
        - 개별 segment: 실제 데이터

    Layout segment가 있으면 거기서 채널 이름 추출.
    """
    import wfdb

    channels: set = set()

    # Master header 자체에 sig_name 있으면 우선 추가
    if hdr.sig_name:
        channels.update(hdr.sig_name)

    # Multi-segment인 경우 layout 읽기
    seg_names = getattr(hdr, "seg_name", None)
    if seg_names:
        for seg in seg_names:
            if seg.endswith("_layout"):
                try:
                    seg_hdr = wfdb.rdheader(seg, pn_dir=pn_dir)
                    if seg_hdr.sig_name:
                        channels.update(seg_hdr.sig_name)
                except Exception:
                    pass
                # Layout만 확인해도 전체 채널 커버 (MIMIC-III 관례)
                break

    return channels


def _scan_one_record(
    record_path: str,
    target_channel: str,
) -> tuple[str, bool, object]:
    """단일 레코드 스캔 worker. (record_path, has_target, channels_or_error) 반환.

    channels_or_error는 성공 시 sorted channel list, 실패 시 Exception.
    """
    import wfdb

    parts = record_path.split("/")
    if len(parts) < 3:
        return record_path, False, ValueError("invalid record path")

    db_name = f"mimic3wdb-matched/{parts[0]}/{parts[1]}"
    rec_name = parts[-1]

    try:
        hdr = wfdb.rdheader(rec_name, pn_dir=db_name)
        channels = get_all_channels(hdr, db_name)
        return record_path, target_channel in channels, sorted(channels)
    except Exception as e:
        return record_path, False, e


def scan_icp_records(
    records_file: str,
    out_file: str = "downstream/acute_event/intracranial_hypertension/ICP-RECORDS",
    max_scan: int | None = None,
    target_channel: str = "ICP",
    workers: int = 16,
) -> None:
    """RECORDS-waveforms에서 target_channel이 있는 레코드를 병렬 스캔한다.

    MIMIC-III multi-segment record 구조를 고려하여
    master header + layout segment 모두 확인.
    """
    try:
        import wfdb  # noqa: F401
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    with open(records_file, "r") as f:
        all_records = [line.strip() for line in f if line.strip()]

    if max_scan is not None:
        all_records = all_records[:max_scan]

    total = len(all_records)
    print(
        f"Scanning {total} records for '{target_channel}' channel "
        f"with {workers} workers..."
    )

    icp_records: list[str] = []
    n_scanned = 0
    n_errors = 0

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            executor.submit(_scan_one_record, rec, target_channel): rec
            for rec in all_records
        }

        for fut in as_completed(futures):
            record_path, has_target, payload = fut.result()

            if isinstance(payload, Exception):
                n_errors += 1
            elif has_target:
                icp_records.append(record_path)
                print(
                    f"  [{n_scanned + 1}] {target_channel} found: "
                    f"{record_path} (channels: {payload})"
                )

            n_scanned += 1
            if n_scanned % 500 == 0:
                print(
                    f"  Completed {n_scanned}/{total}, "
                    f"{target_channel} found: {len(icp_records)}, "
                    f"errors: {n_errors}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down workers...", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True)

    # 순서 보존 불필요 — 저장 시 정렬
    icp_records.sort()

    # 저장
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for rec in icp_records:
            f.write(rec + "\n")

    print(f"\n{'=' * 60}")
    print(f"  {target_channel} Record Scan Complete")
    print(f"  Scanned: {n_scanned}")
    print(f"  {target_channel} records: {len(icp_records)}")
    print(f"  Errors: {n_errors}")
    print(f"  Output: {out_path}")
    print(f"{'=' * 60}")


def _download_one_record(
    record_path: str,
    out_path: Path,
) -> tuple[str, str, str]:
    """단일 레코드 다운로드 worker.
    반환: (record_path, status, message). status ∈ {"downloaded","skipped","failed"}.
    """
    import wfdb

    parts = record_path.split("/")
    if len(parts) < 3:
        return record_path, "failed", "invalid record path"

    db_subdir = f"mimic3wdb-matched/{parts[0]}/{parts[1]}"
    rec_name = parts[-1]
    patient_dir = out_path / parts[0] / parts[1]

    if (patient_dir / f"{rec_name}.hea").exists():
        return record_path, "skipped", ""

    try:
        wfdb.dl_database(
            db_subdir,
            dl_dir=str(out_path),
            records=[rec_name],
            overwrite=False,
        )
        return record_path, "downloaded", ""
    except Exception as e:
        return record_path, "failed", str(e)


def download_icp_records(
    icp_records_file: str,
    out_dir: str = "datasets/raw/mimic3-waveform-ich",
    max_records: int | None = None,
    workers: int = 8,
) -> None:
    """ICP 레코드 목록에서 waveform을 병렬로 다운로드한다."""
    try:
        import wfdb  # noqa: F401
    except ImportError:
        print("ERROR: wfdb 패키지 필요. pip install wfdb", file=sys.stderr)
        sys.exit(1)

    with open(icp_records_file, "r") as f:
        records = [line.strip() for line in f if line.strip()]

    if max_records is not None:
        records = records[:max_records]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    total = len(records)
    print(
        f"Downloading {total} ICP records to {out_dir} "
        f"with {workers} workers..."
    )

    downloaded = 0
    failed = 0
    skipped = 0
    n_done = 0

    executor = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {
            executor.submit(_download_one_record, rec, out_path): rec
            for rec in records
        }

        for fut in as_completed(futures):
            record_path, status, msg = fut.result()
            if status == "downloaded":
                downloaded += 1
            elif status == "skipped":
                skipped += 1
            else:
                failed += 1
                print(f"  FAIL {record_path}: {msg}")

            n_done += 1
            if n_done % 10 == 0 or n_done == 1:
                print(
                    f"  [{n_done}/{total}] downloaded={downloaded}, "
                    f"skipped={skipped}, failed={failed}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down workers...", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        executor.shutdown(wait=True)

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
        default="downstream/acute_event/intracranial_hypertension/ICP-RECORDS",
    )
    scan_parser.add_argument("--max-scan", type=int, default=None)
    scan_parser.add_argument(
        "--workers", type=int, default=16,
        help="병렬 스캔 worker 수 (기본 16, 최대 32 권장)",
    )

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
    dl_parser.add_argument(
        "--workers", type=int, default=8,
        help="병렬 다운로드 worker 수 (기본 8). bandwidth 한계로 16 이상은 비추천.",
    )

    args = parser.parse_args()

    if args.command == "scan":
        scan_icp_records(
            records_file=args.records_file,
            out_file=args.out_file,
            max_scan=args.max_scan,
            workers=args.workers,
        )
    elif args.command == "download":
        download_icp_records(
            icp_records_file=args.icp_records_file,
            out_dir=args.out_dir,
            max_records=args.max_records,
            workers=args.workers,
        )


if __name__ == "__main__":
    main()
