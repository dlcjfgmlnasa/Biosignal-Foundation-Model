# -*- coding:utf-8 -*-
"""Manifest 기반 .pt 파일들을 큰 shard로 패킹 (Option D).

수만 개 작은 .pt 파일 → 수백 개 큰 .pt shard로 합쳐 file open() 횟수와
디스크 IO interrupt를 대폭 감소시킨다. 각 shard 안에 dict[rec_idx_str → tensor]
형태로 저장하고, shard_index.json에 rec_idx → shard_id 매핑을 둔다.

사용법:
    python -m scripts.build_shards \
        --manifest datasets/processed/vitaldb/manifest_full.jsonl \
        --out datasets/sharded/vitaldb \
        --target-shard-mb 1024
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch


def _load_manifest_paths_from_jsonl(jsonl_path: Path) -> list[dict]:
    """manifest_full.jsonl에서 모든 (subject_dir, recording_dict)를 평탄화하여 반환."""
    entries: list[dict] = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            subject = json.loads(line)
            subject_id = subject["subject_id"]
            for session in subject.get("sessions", []):
                session_id = session["session_id"]
                for rec in session.get("recordings", []):
                    file_ref = rec["file"]
                    abs_path = (
                        jsonl_path.parent / subject_id / file_ref
                        if "#" not in file_ref
                        else None
                    )
                    entries.append({
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "file": file_ref,
                        "abs_path": str(abs_path) if abs_path else None,
                        "n_channels": rec["n_channels"],
                        "n_timesteps": rec["n_timesteps"],
                        "sampling_rate": rec["sampling_rate"],
                        "signal_type": rec["signal_type"],
                        "spatial_ids": rec.get("spatial_ids"),
                        "start_sample": rec.get("start_sample", 0),
                    })
    return entries


def _estimate_size_bytes(rec: dict) -> int:
    """recording 텐서의 예상 크기 (float32 가정, 4B per sample)."""
    return rec["n_channels"] * rec["n_timesteps"] * 4


def _group_into_shards(
    entries: list[dict],
    target_shard_bytes: int,
) -> list[list[dict]]:
    """같은 subject의 recording들을 같은 shard에 모으면서, 누적 크기가
    target에 도달하면 새 shard 시작. 같은 subject가 너무 크면 분할 허용.
    """
    shards: list[list[dict]] = []
    current: list[dict] = []
    current_size = 0
    last_subject = None

    for rec in entries:
        rec_size = _estimate_size_bytes(rec)
        # subject가 바뀌었고 현재 shard가 target에 도달했으면 새 shard 시작
        if (
            last_subject is not None
            and rec["subject_id"] != last_subject
            and current_size >= target_shard_bytes
        ):
            shards.append(current)
            current = []
            current_size = 0
        current.append(rec)
        current_size += rec_size
        last_subject = rec["subject_id"]

    if current:
        shards.append(current)
    return shards


def main() -> None:
    p = argparse.ArgumentParser(description="Build .pt shards from per-recording files")
    p.add_argument(
        "--manifest", required=True, help="manifest_full.jsonl 경로"
    )
    p.add_argument(
        "--out", required=True, help="출력 shard 디렉토리"
    )
    p.add_argument(
        "--target-shard-mb",
        type=int,
        default=1024,
        help="shard 크기 목표 (MB). 큰 dataset은 1024-2048, 작은 것은 64-256.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="shard 그룹핑만 보고 실제 변환은 안 함.",
    )
    args = p.parse_args()

    manifest_path = Path(args.manifest)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = args.target_shard_mb * 1024 * 1024

    print(f"Loading manifest: {manifest_path}")
    entries = _load_manifest_paths_from_jsonl(manifest_path)
    print(f"  Total recordings: {len(entries)}")
    total_bytes = sum(_estimate_size_bytes(r) for r in entries)
    print(f"  Total estimated size: {total_bytes / 1024**3:.2f} GB")

    shards = _group_into_shards(entries, target_bytes)
    print(f"  Will produce {len(shards)} shards "
          f"(target {args.target_shard_mb} MB each)")

    if args.dry_run:
        for i, shard in enumerate(shards[:5]):
            sz = sum(_estimate_size_bytes(r) for r in shard) / 1024**2
            subjects = sorted({r["subject_id"] for r in shard})
            print(
                f"  shard {i}: {len(shard)} recordings, ~{sz:.1f} MB, "
                f"{len(subjects)} subjects ({subjects[0]}..{subjects[-1]})"
            )
        if len(shards) > 5:
            print(f"  ... ({len(shards) - 5} more)")
        return

    # 실제 빌드
    rec_to_shard: dict[str, int] = {}
    shard_meta: list[dict] = []
    rec_global_idx = 0
    t_start = time.time()

    for shard_id, shard in enumerate(shards):
        shard_dict: dict[str, dict] = {}
        for rec in shard:
            tensor = torch.load(rec["abs_path"], weights_only=True)
            # rec_global_idx를 key로 사용 (문자열, JSON 호환)
            key = str(rec_global_idx)
            shard_dict[key] = {
                "values": tensor,
                "subject_id": rec["subject_id"],
                "session_id": rec["session_id"],
                "n_channels": rec["n_channels"],
                "n_timesteps": rec["n_timesteps"],
                "sampling_rate": rec["sampling_rate"],
                "signal_type": rec["signal_type"],
                "spatial_ids": rec["spatial_ids"],
                "start_sample": rec["start_sample"],
            }
            rec_to_shard[key] = shard_id
            rec_global_idx += 1

        shard_path = out_dir / f"shard_{shard_id:05d}.pt"
        torch.save(shard_dict, shard_path)
        actual_size_mb = shard_path.stat().st_size / 1024**2
        shard_meta.append({
            "shard_id": shard_id,
            "n_recordings": len(shard),
            "size_mb": round(actual_size_mb, 2),
        })
        elapsed = time.time() - t_start
        print(
            f"  shard {shard_id+1}/{len(shards)}: "
            f"{len(shard)} recordings, {actual_size_mb:.1f} MB "
            f"[total elapsed {elapsed:.1f}s]"
        )

    # 인덱스 저장
    index = {
        "n_shards": len(shards),
        "n_recordings": rec_global_idx,
        "target_shard_mb": args.target_shard_mb,
        "source_manifest": str(manifest_path),
        "rec_to_shard": rec_to_shard,
        "shards": shard_meta,
        "build_seconds": round(time.time() - t_start, 2),
    }
    index_path = out_dir / "shard_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. Wrote {len(shards)} shards + {index_path}")
    print(f"Total time: {index['build_seconds']:.1f}s")


if __name__ == "__main__":
    main()
