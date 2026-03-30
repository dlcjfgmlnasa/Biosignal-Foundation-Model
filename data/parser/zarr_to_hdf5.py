"""Zarr → HDF5 변환 스크립트.

기존 subject별 zarr 파일들을 subject별 단일 HDF5 파일로 통합한다.
--out_dir 지정 시 별도 디렉토리에 HDF5 + manifest.json을 생성한다.

Usage
-----
    # 같은 디렉토리에 변환
    python -m data.parser.zarr_to_hdf5 --data_dir datasets/vitaldb --workers 4

    # 별도 디렉토리에 변환 (원본 유지)
    python -m data.parser.zarr_to_hdf5 --data_dir ../updown/datasets/vitaldb \
        --out_dir ./datasets/vitaldb_h5 --workers 4
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr

# 모듈 레벨 변수 (worker에서 접근)
_out_dir: Path | None = None


def convert_subject(subject_dir: Path, out_dir: Path | None = None) -> int:
    """subject 디렉토리의 zarr 파일들을 단일 HDF5로 변환한다.

    Parameters
    ----------
    subject_dir:
        원본 subject 디렉토리 (manifest.json + zarr 파일).
    out_dir:
        None이면 subject_dir에 저장. 지정 시 해당 경로에 저장.

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

    # 출력 경로 결정
    if out_dir is not None:
        target_dir = out_dir / subject_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)
    else:
        target_dir = subject_dir

    h5_path = target_dir / f"{subject_id}.h5"

    # 이미 변환 완료된 경우 스킵
    if h5_path.exists():
        target_manifest = target_dir / "manifest.json"
        if target_manifest.exists():
            with open(target_manifest, encoding="utf-8") as f:
                existing = json.load(f)
            first_rec = existing["sessions"][0]["recordings"][0] if existing["sessions"] else None
            if first_rec and "#" in first_rec["file"]:
                return 0

    n_converted = 0

    with h5py.File(h5_path, "w") as hf:
        for session in manifest["sessions"]:
            for rec in session["recordings"]:
                zarr_file = rec["file"]
                zarr_path = subject_dir / zarr_file

                if not zarr_path.exists():
                    continue

                try:
                    z = zarr.open(str(zarr_path), mode="r")
                    data = np.array(z[:], dtype=np.float16)
                except Exception as e:
                    print(f"  [WARN] {zarr_file}: {e}", file=sys.stderr)
                    continue

                ds_name = zarr_file.replace(".zarr", "")
                hf.create_dataset(
                    ds_name, data=data,
                    compression="gzip", compression_opts=4,
                    chunks=(data.shape[0], min(data.shape[1], 100_000)),
                )

                rec["file"] = f"{subject_id}.h5#{ds_name}"
                n_converted += 1

    # manifest 저장
    if n_converted > 0:
        out_manifest = target_dir / "manifest.json"
        with open(out_manifest, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        # out_dir 사용 시 manifest.jsonl에도 추가
        if out_dir is not None:
            jsonl_path = out_dir / "manifest.jsonl"
            line = json.dumps(
                {"subject_id": subject_id, "manifest": f"{subject_dir.name}/manifest.json"},
                ensure_ascii=False,
            ) + "\n"
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(line)

    return n_converted


def _convert_worker(sd: Path) -> tuple[str, int]:
    """multiprocessing worker."""
    n = convert_subject(sd, out_dir=_out_dir)
    return sd.name, n


def main():
    global _out_dir

    parser = argparse.ArgumentParser(description="Zarr → HDF5 변환")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="원본 데이터 디렉토리 (subject별 zarr)")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="출력 디렉토리 (미지정 시 data_dir에 저장)")
    parser.add_argument("--workers", type=int, default=1,
                        help="병렬 처리 worker 수")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    _out_dir = Path(args.out_dir) if args.out_dir else None

    if _out_dir is not None:
        _out_dir.mkdir(parents=True, exist_ok=True)
        # manifest.jsonl 초기화
        jsonl_path = _out_dir / "manifest.jsonl"
        jsonl_path.write_text("", encoding="utf-8")

    subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir() and (d / "manifest.json").exists()])
    print(f"Found {len(subject_dirs)} subjects in {data_dir}")
    if _out_dir:
        print(f"Output: {_out_dir}")

    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(processes=args.workers) as pool:
            results = pool.map(_convert_worker, subject_dirs)

        total = sum(n for _, n in results)
        print(f"\nConverted {total} recordings from {len(subject_dirs)} subjects")
    else:
        total = 0
        for i, sd in enumerate(subject_dirs):
            n = convert_subject(sd, out_dir=_out_dir)
            total += n
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(subject_dirs)} subjects processed ({total} recordings)")

        print(f"\nConverted {total} recordings from {len(subject_dirs)} subjects")

    out_path = _out_dir or data_dir
    print(f"Done! HDF5 파일이 {out_path}에 생성되었습니다.")


if __name__ == "__main__":
    main()
