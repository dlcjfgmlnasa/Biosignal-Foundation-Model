#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""In-Hospital Mortality (24h Landmark) — Pass 2: 파일 단위 다운로드.

`build_landmark_records.py` (Pass 1) 이 만든 ``FILES-to-download`` 를 읽어
필요한 파일만 받는다. Pass 1 이 이미 landmark 창에 겹치는 **세그먼트만** 골라
놨으므로, 90시간짜리 레코드에서 첫 24h 분량만 내려온다.

repo 의존성 없음 — 이 스크립트 + FILES-to-download (+ 검증용
landmark_manifest.json) 만 복사하면 다른 PC/서버에서 그대로 돈다.

기존 `download_waveforms.py` 대비 고친 것:
  - ``dl_dir`` 불일치 제거: 파일을 항상 ``out_dir/pXX/pXXXXXX/`` 에 쓴다.
    (구 코드는 wfdb 가 out_dir 루트에 평면 저장하는데 존재확인은 환자 디렉토리에서
     해서 resume 이 영원히 안 먹고, prepare_data 가 경로에서 subject_id 도 못 뽑았다.)
  - ``.hea`` 존재만으로 skip 하지 않는다. 파일 단위로 존재 + 크기>0 을 본다.
  - 검증을 **레코드 단위**로 한다. 환자 디렉토리에 ``.dat`` 가 하나라도 있으면
    통과시키던 방식은 같은 환자의 두 번째 레코드 누락을 놓친다.
  - 비원자적 쓰기 제거: ``.part`` 에 받고 완료 후 ``os.replace``.
    (중단된 다운로드가 "이미 존재" 로 오인되는 것을 막는다.)

사용법:
    python -m downstream.outcome.mortality.download_landmark \
        --files-list downstream/outcome/mortality/FILES-to-download \
        --manifest   downstream/outcome/mortality/landmark_manifest.json \
        --out-dir    datasets/raw/mimic3-waveform-mortality \
        --workers 8

    # 소량 테스트
    python -m downstream.outcome.mortality.download_landmark --limit 50

    # 검증만 (다운로드 없이)
    python -m downstream.outcome.mortality.download_landmark --verify-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PHYSIONET_BASE = "https://physionet.org/files/mimic3wdb-matched/1.0"


def load_file_list(files_list: Path, limit: int | None) -> list[str]:
    """FILES-to-download → 상대경로 목록 (pXX/pXXXXXX/<name>)."""
    files = [ln.strip() for ln in files_list.read_text().splitlines() if ln.strip()]
    return files[:limit] if limit else files


def is_present(dest: Path) -> bool:
    """존재 + 크기>0. 0바이트는 중단된 다운로드로 보고 다시 받는다."""
    return dest.exists() and dest.stat().st_size > 0


def download_file(rel_path: str, out_dir: Path, retries: int = 3) -> tuple[str, str, int]:
    """파일 하나를 원자적으로 받는다 → (rel_path, status, bytes).

    status ∈ {"ok", "skip", "fail:<msg>"}
    """
    import requests  # wfdb 의 의존성이라 항상 있다

    dest = out_dir / rel_path
    if is_present(dest):
        return rel_path, "skip", 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = f"{PHYSIONET_BASE}/{rel_path}"

    last = ""
    for _ in range(retries):
        try:
            n = 0
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
                        n += len(chunk)
            if n == 0:
                raise RuntimeError("0 bytes")
            os.replace(tmp, dest)          # 원자적 커밋
            return rel_path, "ok", n
        except Exception as e:             # noqa: BLE001 — 네트워크 오류는 재시도
            last = str(e)
            tmp.unlink(missing_ok=True)
    return rel_path, f"fail:{last}", 0


def verify(manifest_path: Path, out_dir: Path) -> int:
    """레코드 단위 검증 — 기대한 .dat 가 전부 존재하고 크기>0 인가.

    반환: 불완전한 레코드 수.
    """
    manifest = json.loads(manifest_path.read_text())
    incomplete: list[tuple[str, int, int]] = []
    n_records = 0
    n_patients_ok = 0

    for pt in manifest["patients"]:
        pt_ok = True
        for rec in pt["records"]:
            n_records += 1
            dats = [f for f in rec["files"] if f.endswith(".dat")]
            missing = [f for f in dats if not is_present(out_dir / f)]
            if missing:
                incomplete.append((rec["record"], len(missing), len(dats)))
                pt_ok = False
        if pt_ok:
            n_patients_ok += 1

    print(f"\n{'=' * 62}")
    print("  검증 (레코드 단위)")
    print(f"    환자      : {n_patients_ok}/{len(manifest['patients'])} 완전")
    print(f"    레코드    : {n_records - len(incomplete)}/{n_records} 완전")
    if incomplete:
        print(f"    ⚠ 불완전 레코드 {len(incomplete)}개 (앞 10개):")
        for rec, miss, tot in incomplete[:10]:
            print(f"        {rec}: .dat {miss}/{tot} 누락")
        print("      → 같은 명령으로 재실행하면 누락분만 다시 받습니다.")
    else:
        print("    ✓ 기대한 .dat 전부 존재")
    print(f"{'=' * 62}")
    return len(incomplete)


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Pass 2: landmark waveform 파일 다운로드 (resumable, 원자적)."
    )
    parser.add_argument("--files-list", type=str,
                        default=str(here / "FILES-to-download"))
    parser.add_argument("--manifest", type=str,
                        default=str(here / "landmark_manifest.json"),
                        help="검증용 (Pass 1 산출물)")
    parser.add_argument("--out-dir", type=str,
                        default="datasets/raw/mimic3-waveform-mortality")
    parser.add_argument("--workers", type=int, default=8,
                        help="병렬도. PhysioNet 예의상 16 초과 비권장.")
    parser.add_argument("--limit", type=int, default=None, help="앞 N개 파일만")
    parser.add_argument("--verify-only", action="store_true",
                        help="다운로드 없이 검증만")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)

    if args.verify_only:
        if not manifest_path.exists():
            sys.exit(f"ERROR: manifest 없음: {manifest_path}")
        sys.exit(1 if verify(manifest_path, out_dir) else 0)

    files_list = Path(args.files_list)
    if not files_list.exists():
        sys.exit(
            f"ERROR: {files_list} 가 없습니다.\n"
            f"       먼저 Pass 1 을 실행하세요:\n"
            f"       python -m downstream.outcome.mortality.build_landmark_records \\\n"
            f"           --cohort-csv <landmark cohort CSV>"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    files = load_file_list(files_list, args.limit)

    print(f"대상 파일 : {len(files)}")
    print(f"저장 위치 : {out_dir.resolve()}")
    print(f"Worker    : {args.workers}\n")

    ok = skip = fail = done = 0
    total_bytes = 0
    executor = ThreadPoolExecutor(max_workers=args.workers)
    try:
        futs = [executor.submit(download_file, f, out_dir) for f in files]
        for fut in as_completed(futs):
            rel, status, nbytes = fut.result()
            if status == "ok":
                ok += 1
                total_bytes += nbytes
            elif status == "skip":
                skip += 1
            else:
                fail += 1
                print(f"  {rel}: {status}", file=sys.stderr)
            done += 1
            if done == 1 or done % 100 == 0 or done == len(files):
                print(f"  [{done}/{len(files)}] 받음={ok} skip={skip} "
                      f"실패={fail} ({total_bytes / 1e9:.2f} GB)")
    except KeyboardInterrupt:
        print("\n중단됨 — 재실행하면 이어받습니다.", file=sys.stderr)
        executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(130)
    finally:
        executor.shutdown(wait=True)

    print(f"\n{'=' * 62}")
    print("  다운로드 완료")
    print(f"    새로 받음 : {ok}  ({total_bytes / 1e9:.2f} GB)")
    print(f"    이미 존재 : {skip}")
    print(f"    실패      : {fail}")
    print(f"{'=' * 62}")

    if manifest_path.exists() and not args.limit:
        verify(manifest_path, out_dir)
    if fail:
        print(f"\n실패 {fail}건 — 같은 명령으로 재실행하면 실패분만 다시 받습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
