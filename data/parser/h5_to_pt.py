# -*- coding:utf-8 -*-
"""HDF5 → .pt (mmap) 변환 스크립트.

subject별 HDF5 파일의 각 dataset을 개별 .pt 파일로 저장한다.
.pt는 torch.load(mmap=True)로 디스크에서 직접 읽을 수 있어
preload 없이도 I/O 병목을 크게 줄인다.

manifest.json은 path를 .pt 경로로 업데이트한다.

Usage
-----
    python -m data.parser.h5_to_pt --data_dir ./datasets/vitaldb_h5 --workers 4

    # 별도 디렉토리에 출력
    python -m data.parser.h5_to_pt --data_dir ./datasets/vitaldb_h5 \
        --out_dir ./datasets/vitaldb_pt --workers 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import torch


def convert_subject(subject_dir: Path, out_dir: Path | None) -> tuple[str, int]:
    """subject 디렉토리의 h5 → pt 변환 + manifest 업데이트."""
    manifest_path = subject_dir / "manifest.json"
    if not manifest_path.exists():
        return subject_dir.name, 0

    with open(manifest_path, encoding="utf-8") as f:
        meta = json.load(f)

    dst_dir = (out_dir / subject_dir.name) if out_dir else subject_dir
    dst_dir.mkdir(parents=True, exist_ok=True)

    n_converted = 0
    for session in meta["sessions"]:
        for rec in session["recordings"]:
            file_ref = rec["file"]
            if "#" not in file_ref:
                continue  # 이미 pt 파일이면 스킵

            h5_file, ds_name = file_ref.split("#", 1)
            h5_path = subject_dir / h5_file

            if not h5_path.exists():
                continue

            # h5 dataset → tensor
            with h5py.File(h5_path, "r") as hf:
                if ds_name not in hf:
                    continue
                data = torch.from_numpy(hf[ds_name][:]).float()

            # .pt 저장 (ds_name의 / → _ 치환)
            pt_name = ds_name.replace("/", "_") + ".pt"
            pt_path = dst_dir / pt_name
            torch.save(data, pt_path)

            # manifest 업데이트: "subject.h5#dataset" → "dataset.pt"
            rec["file"] = pt_name
            n_converted += 1

    # 업데이트된 manifest 저장
    with open(dst_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return subject_dir.name, n_converted


def main():
    parser = argparse.ArgumentParser(description="HDF5 → .pt 변환")
    parser.add_argument("--data_dir", type=str, required=True, help="HDF5 데이터 디렉토리")
    parser.add_argument("--out_dir", type=str, default=None, help="출력 디렉토리 (없으면 in-place)")
    parser.add_argument("--workers", type=int, default=4, help="병렬 워커 수")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    # subject 디렉토리 목록
    subject_dirs = sorted([
        d for d in data_dir.iterdir()
        if d.is_dir() and (d / "manifest.json").exists()
    ])
    print(f"Found {len(subject_dirs)} subjects")

    total_converted = 0
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_subject, sd, out_dir): sd
            for sd in subject_dirs
        }
        for future in as_completed(futures):
            name, n = future.result()
            total_converted += n
            done += 1
            if done % 100 == 0:
                print(f"  [{done}/{len(subject_dirs)}] {name}: {n} recordings converted")

    print(f"Done. {total_converted} recordings converted from {len(subject_dirs)} subjects.")


if __name__ == "__main__":
    main()
