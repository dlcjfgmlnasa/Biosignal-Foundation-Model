"""Zarr → HDF5 변환 스크립트.

기존 subject별 zarr 파일들을 subject별 단일 HDF5 파일로 통합한다.
manifest.json은 그대로 유지하고, 파일 경로만 .zarr → .h5로 변경.

Usage
-----
    python -m data.parser.zarr_to_hdf5 --data_dir datasets/vitaldb --workers 4

변환 후 구조:
    datasets/vitaldb/
      VDB_0001/
        VDB_0001.h5              ← 단일 HDF5 (모든 seg 통합)
        manifest.json            ← 경로가 .h5#dataset_key로 변경
        VDB_0001_S0_ecg_1_seg0.zarr  ← 기존 파일 (삭제 가능)
        ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr


def convert_subject(subject_dir: Path) -> int:
    """subject 디렉토리의 zarr 파일들을 단일 HDF5로 변환한다.

    Returns
    -------
    변환된 recording 수.
    """
    manifest_path = subject_dir / "manifest.json"
    if not manifest_path.exists():
        return 0

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    subject_id = manifest["subject_id"]
    h5_path = subject_dir / f"{subject_id}.h5"

    # 이미 변환 완료된 경우 스킵
    if h5_path.exists():
        # manifest가 이미 .h5를 가리키는지 확인
        first_rec = manifest["sessions"][0]["recordings"][0] if manifest["sessions"] else None
        if first_rec and first_rec["file"].endswith(".h5"):
            return 0

    n_converted = 0

    with h5py.File(h5_path, "w") as hf:
        for session in manifest["sessions"]:
            for rec in session["recordings"]:
                zarr_file = rec["file"]
                zarr_path = subject_dir / zarr_file

                if not zarr_path.exists():
                    continue

                # zarr 읽기
                try:
                    z = zarr.open(str(zarr_path), mode="r")
                    data = np.array(z[:], dtype=np.float16)
                except Exception as e:
                    print(f"  [WARN] {zarr_file}: {e}", file=sys.stderr)
                    continue

                # HDF5 dataset 이름: zarr 파일명에서 .zarr 제거
                ds_name = zarr_file.replace(".zarr", "")
                hf.create_dataset(
                    ds_name, data=data,
                    compression="gzip", compression_opts=4,
                    chunks=(data.shape[0], min(data.shape[1], 100_000)),
                )

                # manifest 경로 업데이트
                rec["file"] = f"{subject_id}.h5#{ds_name}"
                n_converted += 1

    # manifest 저장 (경로 업데이트)
    if n_converted > 0:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return n_converted


def _convert_worker(sd: Path) -> tuple[str, int]:
    """multiprocessing worker."""
    n = convert_subject(sd)
    return sd.name, n


def main():
    parser = argparse.ArgumentParser(description="Zarr → HDF5 변환")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="processed 데이터 디렉토리 (subject별 하위 디렉토리)")
    parser.add_argument("--workers", type=int, default=1,
                        help="병렬 처리 worker 수")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()])

    print(f"Found {len(subject_dirs)} subjects in {data_dir}")

    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(processes=args.workers) as pool:
            results = pool.map(_convert_worker, subject_dirs)

        total = 0
        for name, n in results:
            if n > 0:
                total += n
        print(f"\nConverted {total} recordings from {len(subject_dirs)} subjects")
    else:
        total = 0
        for i, sd in enumerate(subject_dirs):
            n = convert_subject(sd)
            total += n
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(subject_dirs)} subjects processed ({total} recordings)")

        print(f"\nConverted {total} recordings from {len(subject_dirs)} subjects")

    print("Done! manifest.json 파일이 .h5 경로로 업데이트되었습니다.")
    print("기존 .zarr 파일은 수동으로 삭제할 수 있습니다:")
    print(f"  find {data_dir} -name '*.zarr' -type d -exec rm -rf {{}} +")


if __name__ == "__main__":
    main()
