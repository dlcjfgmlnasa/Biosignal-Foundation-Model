#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""Cardiac Arrest waveform 단독 다운로더 (다른 PC에서 바로 실행).

repo 의존성 없음. 아래 2개 파일만 다른 PC로 복사하면 됩니다:
  1. 이 스크립트 (standalone_download.py)
  2. RECORDS-to-download   (다운로드 대상 레코드 목록 — 이미 빌드됨)

RECORDS-to-download 에는 arrest onset 윈도우 + risk-set matched negative
선택이 모두 반영된 최종 레코드 경로(7,658개)가 들어 있어, cohort CSV 나
RECORDS-waveforms 없이도 그대로 받을 수 있습니다.

────────────────────────────────────────────────────────────────────
준비
────────────────────────────────────────────────────────────────────
  pip install wfdb            # 유일한 외부 의존성 (multi-segment 자동 처리)

  # PhysioNet 인증이 필요한 경우에만 (mimic3wdb-matched 가 credentialed 일 때).
  # 비밀번호를 명령줄/스크립트에 넣지 말고 ~/.netrc 에 두세요:
  #   cat > ~/.netrc <<'EOF'
  #   machine physionet.org
  #   login <PHYSIONET_ID>
  #   password <PHYSIONET_PW>
  #   EOF
  #   chmod 600 ~/.netrc        # (Windows 는 생략 가능)

────────────────────────────────────────────────────────────────────
사용법
────────────────────────────────────────────────────────────────────
  python standalone_download.py \
      --records-file RECORDS-to-download \
      --out-dir mimic3-waveform-cardiac-arrest \
      --workers 8

  # 소량 테스트 (앞 20개만)
  python standalone_download.py --limit 20

특징
  - Resumable    : 이미 받은 파일은 건너뜀 (중단 후 재실행 OK)
  - 병렬 다운로드 : ThreadPool (기본 8 worker)
  - 진행률 로깅
  - 완료 후 .dat 개수 검증 — 헤더(.hea)만 받고 신호(.dat)가 0 인 함정 탐지
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_record_list(records_file: Path, limit: int | None) -> list[str]:
    """RECORDS-to-download → 레코드 경로 리스트 (pXX/pXXXXXX/pXXXXXX-...)."""
    records: list[str] = []
    with open(records_file, "r") as f:
        for line in f:
            rec = line.strip()
            if rec:
                records.append(rec)
    if limit is not None:
        records = records[:limit]
    return records


def is_already_downloaded(record_path: str, out_dir: Path) -> bool:
    """master .hea + 세그먼트 .dat 가 최소 1개라도 있으면 완료로 간주.

    (.hea 만 있고 .dat 가 0 이면 미완료 — 헤더-only 함정 방지)
    """
    parts = record_path.split("/")
    patient_dir = out_dir / parts[0] / parts[1]
    rec_name = parts[-1]
    master_hea = patient_dir / f"{rec_name}.hea"
    if not master_hea.exists():
        return False
    # 세그먼트 .dat 가 하나라도 있어야 진짜 완료
    return any(patient_dir.glob("*.dat"))


def download_record(record_path: str, out_dir: Path) -> tuple[str, bool, str]:
    """wfdb.dl_database 로 multi-segment 레코드 다운로드.

    MIMIC-III matched 는 한 레코드가 layout .hea + 여러 segment(.hea/.dat)로
    쪼개져 있어 wfdb 가 자동으로 모든 세그먼트를 받아준다.
    """
    import wfdb  # worker 내 import (ThreadPool 안전)

    parts = record_path.split("/")  # ['p00', 'p000052', 'p000052-...']
    if len(parts) < 3:
        return record_path, False, "잘못된 경로 형식"

    db_subdir = f"mimic3wdb-matched/{parts[0]}/{parts[1]}"
    rec_name = parts[-1]

    if is_already_downloaded(record_path, out_dir):
        return record_path, True, "skip(이미 존재)"

    # 반드시 pXX/pXXXXXX/ 구조로 저장한다.
    # wfdb.dl_database 는 dl_dir 아래에 파일을 평면(flat)으로 떨어뜨리므로
    # (segment 파일명 3238451_0001.dat 에는 subject_id 가 없음),
    # dl_dir 을 환자별 디렉토리로 지정해야 prepare_data.py 가 경로의
    # pXXXXXX 부분으로 subject_id 를 식별할 수 있고, 환자 간 segment ID
    # 충돌도 방지된다.
    patient_dir = out_dir / parts[0] / parts[1]
    patient_dir.mkdir(parents=True, exist_ok=True)

    try:
        wfdb.dl_database(
            db_subdir,
            dl_dir=str(patient_dir),
            records=[rec_name],
            overwrite=False,  # 기존 파일 유지 → resumable
        )
        return record_path, True, "ok"
    except Exception as e:  # noqa: BLE001
        return record_path, False, f"FAIL: {e}"


def verify(out_dir: Path) -> None:
    """다운로드 결과 검증 — .hea/.dat 개수, .dat 0 인 환자 디렉토리 보고."""
    hea = list(out_dir.rglob("*.hea"))
    dat = list(out_dir.rglob("*.dat"))
    print(f"\n{'=' * 60}")
    print("  검증")
    print(f"    .hea (헤더) : {len(hea)}")
    print(f"    .dat (신호) : {len(dat)}")
    if not dat:
        print("    ⚠️  .dat 가 0 입니다 — 신호가 전혀 안 받아졌습니다 "
              "(인증/네트워크 확인 필요).")
    elif len(dat) < len(hea) // 4:
        print("    ⚠️  .dat 가 .hea 대비 비정상적으로 적습니다 — "
              "일부 레코드가 헤더만 받았을 수 있습니다.")
    print(f"{'=' * 60}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone cardiac arrest waveform downloader (no repo deps)."
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--records-file", type=str,
        default=str(script_dir / "RECORDS-to-download"),
        help="다운로드 대상 레코드 목록 (기본: 스크립트와 같은 폴더의 RECORDS-to-download)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="mimic3-waveform-cardiac-arrest",
        help="다운로드 저장 디렉토리",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="병렬 다운로드 worker 수 (bandwidth 한계로 16 이상 비추천)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="앞 N개만 받기 (테스트용)",
    )
    args = parser.parse_args()

    try:
        import wfdb  # noqa: F401
    except ImportError:
        print("ERROR: wfdb 패키지가 필요합니다.  pip install wfdb", file=sys.stderr)
        sys.exit(1)

    records_file = Path(args.records_file)
    if not records_file.exists():
        print(f"ERROR: 레코드 목록 파일이 없습니다: {records_file}", file=sys.stderr)
        print("       RECORDS-to-download 를 이 스크립트와 같은 폴더에 두세요.",
              file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_record_list(records_file, args.limit)
    print(f"대상 레코드: {len(records)}개")
    print(f"저장 위치  : {out_dir.resolve()}")
    print(f"Worker     : {args.workers}\n")

    downloaded = 0
    skipped = 0
    failed = 0
    n_done = 0
    total = len(records)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futures = {
            executor.submit(download_record, rec, out_dir): rec
            for rec in records
        }
        for fut in as_completed(futures):
            rec, ok, msg = fut.result()
            if ok and msg.startswith("skip"):
                skipped += 1
            elif ok:
                downloaded += 1
            else:
                failed += 1
                print(f"  {rec}: {msg}", file=sys.stderr)
            n_done += 1
            if n_done == 1 or n_done % 25 == 0 or n_done == total:
                print(f"  [{n_done}/{total}] 받음={downloaded} "
                      f"skip={skipped} 실패={failed}")
    except KeyboardInterrupt:
        print("\n중단됨 — worker 종료 중... (재실행하면 이어받기 됩니다)",
              file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(130)
    finally:
        executor.shutdown(wait=True)

    print(f"\n{'=' * 60}")
    print("  다운로드 완료")
    print(f"    새로 받음 : {downloaded}")
    print(f"    이미 존재 : {skipped}")
    print(f"    실패      : {failed}")
    print(f"{'=' * 60}")

    verify(out_dir)

    if failed:
        print(f"\n실패 {failed}건 — 네트워크/인증 확인 후 같은 명령으로 재실행하면 "
              "이미 받은 건 건너뛰고 실패분만 다시 받습니다.")


if __name__ == "__main__":
    main()
