# -*- coding:utf-8 -*-
"""processed 디렉토리에서 manifest_full.jsonl 생성.

vitaldb.py 파서가 처리 끝에 자동으로 만들지만, 누락되거나 K-MIMIC처럼
직접 수집한 경우 별도로 빌드 필요.

수집 우선순위:
  1. manifest.jsonl (경로 인덱스) — 빠름
  2. */manifest.json glob — 느리지만 안정

사용법:
    python -m scripts.build_manifest_full \\
        --data-dir /path/to/processed/k-mimic
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="processed 디렉토리")
    p.add_argument(
        "--out",
        default=None,
        help="출력 manifest_full.jsonl 경로. 기본값: data-dir/manifest_full.jsonl",
    )
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out) if args.out else data_dir / "manifest_full.jsonl"

    print(f"data_dir = {data_dir}", flush=True)
    print(f"out_path = {out_path}", flush=True)

    if not data_dir.exists():
        print(f"ERROR: {data_dir} not found")
        return

    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    # ── 우선순위 1: manifest.jsonl ──
    index_file = data_dir / "manifest.jsonl"
    manifest_paths: list[Path] = []
    source = ""

    if index_file.exists():
        source = "manifest.jsonl"
        print(f"Reading index: {index_file}", flush=True)
        t0 = time.time()
        with open(index_file, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        print(f"  {len(lines)} entries; verifying paths exist...", flush=True)

        line_iter = (
            tqdm(lines, desc="Checking manifests", unit="entry")
            if _have_tqdm else lines
        )
        for line in line_iter:
            meta = json.loads(line)
            mf = data_dir / meta["manifest"]
            if mf.exists():
                manifest_paths.append(mf)
        print(
            f"Using {source}: {len(manifest_paths)} subject manifests "
            f"(in {time.time() - t0:.1f}s)",
            flush=True,
        )

    # ── 우선순위 2: scandir */manifest.json (progress 표시) ──
    if not manifest_paths:
        source = "scandir */manifest.json"
        print(f"Falling back to {source}...")
        t0 = time.time()

        try:
            from tqdm import tqdm
            scan_pbar = tqdm(desc="Scanning subject dirs", unit="dir")
        except ImportError:
            scan_pbar = None

        with os.scandir(data_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                mf = Path(entry.path) / "manifest.json"
                if mf.exists():
                    manifest_paths.append(mf)
                if scan_pbar is not None:
                    scan_pbar.update(1)
                    scan_pbar.set_postfix_str(f"found {len(manifest_paths)}")

        if scan_pbar is not None:
            scan_pbar.close()

        manifest_paths.sort()
        print(
            f"Found {len(manifest_paths)} subject manifests in "
            f"{time.time() - t0:.1f}s"
        )

    if not manifest_paths:
        print(f"ERROR: no manifests found in {data_dir}")
        return

    # ── 합치기 ──
    print(f"Writing to {out_path}")
    t0 = time.time()
    count = 0
    skipped = 0

    try:
        from tqdm import tqdm
        iterator = tqdm(manifest_paths, desc="Merging", unit="subj")
    except ImportError:
        iterator = manifest_paths

    with open(out_path, "w", encoding="utf-8") as out:
        for mf_path in iterator:
            try:
                with open(mf_path, encoding="utf-8") as f:
                    data = json.load(f)
                out.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
            except (json.JSONDecodeError, OSError) as e:
                print(f"  SKIP {mf_path}: {e}")
                skipped += 1

    elapsed = time.time() - t0
    print(
        f"\nDone. Wrote {count} subjects ({skipped} skipped) in {elapsed:.1f}s"
        f"\nOutput: {out_path}"
    )

    # 간단 통계
    if count > 0:
        out_size_mb = out_path.stat().st_size / 1024**2
        print(f"manifest_full.jsonl size: {out_size_mb:.1f} MB")


if __name__ == "__main__":
    main()
