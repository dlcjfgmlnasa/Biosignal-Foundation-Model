# -*- coding:utf-8 -*-
"""1-pass 데이터셋 변환: raw .vital → shards (intermediate .pt 자동 정리).

기존 워크플로:
    python -m data.parser.vitaldb --raw ... --out ...    # 100GB per-recording .pt
    python -m scripts.build_shards --manifest ... --out ...  # 100GB shards
    # 총 디스크: 200GB (intermediate + shard)

이 스크립트:
    python -m scripts.parse_to_shard --raw ... --processed-tmp ... --shard-out ...
    # 한 번에 끝남, intermediate 자동 정리 → 디스크 ~100GB (shard만)

핵심 동작:
  1. 기존 vitaldb.py 파서 호출 → per-recording .pt + manifest 생성
  2. 즉시 build_shards.py 호출 → shard 패킹
  3. --cleanup-intermediate 옵션 시 per-recording .pt 삭제
     (manifest_full.jsonl과 manifest 디렉토리는 보존 — 학습에 필수)

사용법 (전형적):
    python -m scripts.parse_to_shard \\
        --raw /path/to/raw_vital_dir/ \\
        --processed-tmp /path/to/processed/dataset_name/ \\
        --shard-out /path/to/sharded/dataset_name/ \\
        --workers 8 \\
        --target-shard-mb 1024 \\
        --cleanup-intermediate

이후 학습 yaml:
    data_dir: /path/to/processed/dataset_name/
    shard_index_path: /path/to/sharded/dataset_name/shard_index.json

NOTE: 진정한 1-pass (메모리 buffer → 직접 shard write, intermediate 0)는 process_vital
리팩토링 필요. 향후 데이터셋 규모가 압박되면 별도 작업.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_step(cmd: list[str], label: str) -> int:
    """서브프로세스 실행 + 시간 측정. 실패 시 exit code 그대로 반환."""
    print(f"\n{'='*60}\n[{label}] {' '.join(cmd)}\n{'='*60}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd)
    elapsed = time.time() - t0
    print(f"[{label}] done in {elapsed:.1f}s (rc={rc})", flush=True)
    return rc


def cleanup_pt_files(processed_dir: Path) -> None:
    """processed_dir 내의 per-recording .pt 파일 삭제. manifest는 보존."""
    print(f"\n=== Cleanup intermediate .pt files in {processed_dir} ===")
    n_files = 0
    n_bytes = 0
    for pt in processed_dir.rglob("*.pt"):
        try:
            n_bytes += pt.stat().st_size
            pt.unlink()
            n_files += 1
            if n_files % 10000 == 0:
                print(f"  removed {n_files} files...", flush=True)
        except OSError as e:
            print(f"  WARN: failed to remove {pt}: {e}", file=sys.stderr)
    print(f"  removed {n_files} .pt files ({n_bytes / 1024**3:.1f} GB freed)")


def main() -> None:
    p = argparse.ArgumentParser(
        description="1-pass parse + shard for new datasets",
    )
    p.add_argument("--raw", required=True, help="raw .vital 파일 디렉토리")
    p.add_argument(
        "--processed-tmp",
        required=True,
        help="중간 per-recording .pt + manifest 출력 디렉토리. "
             "--cleanup-intermediate 옵션 시 .pt만 삭제됨 (manifest 유지).",
    )
    p.add_argument(
        "--shard-out",
        required=True,
        help="shard 출력 디렉토리 (학습 시 shard_index_path의 부모)",
    )
    p.add_argument(
        "--target-shard-mb",
        type=int,
        default=1024,
        help="shard 크기 목표 (MB). 큰 dataset은 1024-2048 권장",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="parse + shard 단계 모두에 적용",
    )
    p.add_argument(
        "--signal-types",
        type=int,
        nargs="+",
        default=None,
        help="파싱할 signal type IDs (0=ECG ~ 7=ICP). 미지정 시 전부.",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="처리할 최대 .vital 파일 수 (테스트용)",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=60.0,
        help="최소 유효 신호 길이 (초)",
    )
    p.add_argument(
        "--subject-from-parent",
        type=int,
        default=0,
        help="K-MIMIC 같은 4단계 구조면 2 사용 (bucket/subject/stay/file)",
    )
    p.add_argument(
        "--cleanup-intermediate",
        action="store_true",
        help="shard 빌드 후 per-recording .pt 자동 삭제 (디스크 ~50%% 절약). "
             "manifest는 보존.",
    )
    p.add_argument(
        "--from-list",
        type=str,
        default=None,
        help="파일 경로 리스트(.txt) 사용 — os.walk 우회. "
             "NAS scan silent-fail 회피용. find $RAW -name '*.vital' > list.txt 후 지정.",
    )
    p.add_argument(
        "--skip-parse",
        action="store_true",
        help="parse 단계 건너뛰기 (이미 processed-tmp에 manifest 있을 때)",
    )
    p.add_argument(
        "--skip-shard",
        action="store_true",
        help="shard 단계 건너뛰기 (parse만)",
    )
    args = p.parse_args()

    raw_dir = Path(args.raw)
    processed_dir = Path(args.processed_tmp)
    shard_dir = Path(args.shard_out)

    if not raw_dir.exists():
        print(f"ERROR: raw dir not found: {raw_dir}")
        sys.exit(1)

    processed_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ── Step 1: Parse raw → per-recording .pt + manifest ──
    if not args.skip_parse:
        parse_cmd = [
            sys.executable, "-m", "data.parser.vitaldb",
            "--raw", str(raw_dir),
            "--out", str(processed_dir),
            "--workers", str(args.workers),
            "--min-duration", str(args.min_duration),
        ]
        if args.signal_types:
            parse_cmd += ["--signal-types"] + [str(s) for s in args.signal_types]
        if args.max_files:
            parse_cmd += ["--max-files", str(args.max_files)]
        if args.subject_from_parent:
            parse_cmd += ["--subject-from-parent", str(args.subject_from_parent)]
        if args.from_list:
            parse_cmd += ["--from-list", str(args.from_list)]

        rc = run_step(parse_cmd, "parse")
        if rc != 0:
            print(f"ERROR: parse step failed with rc={rc}")
            sys.exit(rc)
    else:
        print("[parse] skipped per --skip-parse")

    # ── manifest_full.jsonl 확인/생성 ──
    manifest_full = processed_dir / "manifest_full.jsonl"
    if not manifest_full.exists():
        print(f"\n[manifest] {manifest_full} not found, building...")
        rc = run_step(
            [
                sys.executable, "-m", "scripts.build_manifest_full",
                "--data-dir", str(processed_dir),
            ],
            "manifest_full",
        )
        if rc != 0:
            print(f"ERROR: manifest_full build failed with rc={rc}")
            sys.exit(rc)

    # ── Step 2: per-recording .pt → shards ──
    if not args.skip_shard:
        shard_cmd = [
            sys.executable, "-m", "scripts.build_shards",
            "--manifest", str(manifest_full),
            "--out", str(shard_dir),
            "--target-shard-mb", str(args.target_shard_mb),
            "--workers", str(args.workers),
        ]
        rc = run_step(shard_cmd, "shard")
        if rc != 0:
            print(f"ERROR: shard step failed with rc={rc}")
            sys.exit(rc)
    else:
        print("[shard] skipped per --skip-shard")

    # ── Step 3 (optional): cleanup intermediate .pt ──
    if args.cleanup_intermediate:
        if args.skip_shard:
            print(
                "WARNING: --cleanup-intermediate ignored when --skip-shard "
                "(shards 없으면 .pt 삭제하면 데이터 영구 손실)"
            )
        else:
            cleanup_pt_files(processed_dir)

    elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"All steps complete in {elapsed/60:.1f} min")
    print(f"Manifest:    {manifest_full}")
    print(f"Shards:      {shard_dir}/shard_*.pt + shard_index.json")
    print(f"\n학습 yaml에 추가:")
    print(f"  data_dir:")
    print(f"    - {processed_dir}/")
    print(f"  shard_index_path: {shard_dir}/shard_index.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
