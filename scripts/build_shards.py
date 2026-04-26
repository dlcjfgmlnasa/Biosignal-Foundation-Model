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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch


def _load_one_tensor(path: str) -> torch.Tensor:
    """Worker: 한 recording 파일을 로드. 별도 프로세스에서 실행됨."""
    return torch.load(path, weights_only=True)


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


def _load_existing_index(out_dir: Path) -> dict | None:
    """기존 shard_index.json이 있으면 로드. 없으면 None."""
    index_path = out_dir / "shard_index.json"
    if not index_path.exists():
        return None
    with open(index_path, encoding="utf-8") as f:
        return json.load(f)


def _extract_done_subjects(index: dict) -> set[str]:
    """기존 index에서 이미 sharded된 subject_id 집합 추출.

    각 shard meta에 'subjects' 필드가 있으면 사용. 없으면 shards 자체를
    스캔(과거 인덱스 호환). 후자는 느리므로 가능하면 'subjects' 필드 사용 권장.
    """
    done: set[str] = set()
    missing_subjects_field = False
    for sm in index.get("shards", []):
        if "subjects" in sm:
            done.update(sm["subjects"])
        else:
            missing_subjects_field = True
    if missing_subjects_field:
        print(
            "  WARN: 기존 shard_index.json에 'subjects' 필드 없음. "
            "incremental 동작 불완전 — 전체 재빌드 권장.",
            flush=True,
        )
    return done


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
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="병렬 torch.load worker 수 (기본 8). 0=직렬. "
             "디스크 IO bound면 4-16이 sweet spot, CPU 코어 수보다 더 안 빠름.",
    )
    p.add_argument(
        "--incremental",
        action="store_true",
        help="기존 shard_index.json이 있으면 done subjects는 skip하고 "
             "신규 entry만 새 shard로 추가 (기존 shard 무수정).",
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

    # ── Incremental: 기존 index 있으면 done subjects 제외 ──
    existing_index: dict | None = None
    shard_id_offset = 0
    rec_idx_offset = 0
    if args.incremental:
        existing_index = _load_existing_index(out_dir)
        if existing_index is None:
            print("  --incremental 지정됐으나 기존 shard_index.json 없음 → 전체 빌드 진행")
        else:
            done_subjects = _extract_done_subjects(existing_index)
            shard_id_offset = existing_index.get("n_shards", 0)
            rec_idx_offset = existing_index.get("n_recordings", 0)
            n_before = len(entries)
            entries = [e for e in entries if e["subject_id"] not in done_subjects]
            n_skipped = n_before - len(entries)
            print(
                f"  Incremental: {len(done_subjects)} subjects already sharded, "
                f"skipped {n_skipped} entries. Remaining: {len(entries)}",
                flush=True,
            )
            print(
                f"  New shards will start at id={shard_id_offset}, "
                f"rec_idx={rec_idx_offset}",
                flush=True,
            )
            if not entries:
                print("  Nothing new to shard. Exiting.")
                return

    shards = _group_into_shards(entries, target_bytes)
    print(f"  Will produce {len(shards)} new shards "
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

    # 실제 빌드 — tqdm 있으면 사용, 없으면 print fallback
    try:
        from tqdm import tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    rec_to_shard: dict[str, int] = {}
    shard_meta: list[dict] = []
    rec_global_idx = rec_idx_offset
    t_start = time.time()

    shard_pbar = (
        tqdm(total=len(shards), desc="Shards", unit="shard", position=0)
        if _have_tqdm else None
    )

    # workers > 0이면 ProcessPoolExecutor를 단일 pool로 유지 (shard마다
    # fork 오버헤드 회피). 단일 pool이 모든 shard에서 재사용됨.
    use_parallel = args.workers > 0
    pool: ProcessPoolExecutor | None = None
    if use_parallel:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        print(
            f"  Using {args.workers} parallel workers for torch.load (single pool)"
        )

    def _build_one_shard(shard_id: int, shard: list[dict]) -> None:
        nonlocal rec_global_idx
        local_idx = shard_id - shard_id_offset  # 진행률/ETA용 0-based 로컬 인덱스
        shard_dict: dict[str, dict] = {}
        paths = [rec["abs_path"] for rec in shard]

        if use_parallel:
            tensor_iter = pool.map(_load_one_tensor, paths, chunksize=4)
        else:
            tensor_iter = (torch.load(p, weights_only=True) for p in paths)

        if _have_tqdm:
            tensor_iter = tqdm(
                tensor_iter,
                total=len(shard),
                desc=f"  shard {local_idx+1}/{len(shards)} read",
                unit="rec",
                leave=False,
                position=1,
            )

        for rec, tensor in zip(shard, tensor_iter):
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
        if _have_tqdm:
            shard_pbar.set_postfix_str(f"writing {shard_path.name}")
        torch.save(shard_dict, shard_path)
        actual_size_mb = shard_path.stat().st_size / 1024**2
        shard_meta.append({
            "shard_id": shard_id,
            "n_recordings": len(shard),
            "size_mb": round(actual_size_mb, 2),
            "subjects": sorted({r["subject_id"] for r in shard}),
        })
        elapsed = time.time() - t_start
        avg_per_shard = elapsed / (local_idx + 1)
        eta_seconds = avg_per_shard * (len(shards) - local_idx - 1)
        if _have_tqdm:
            shard_pbar.set_postfix_str(
                f"{actual_size_mb:.0f}MB | avg {avg_per_shard:.0f}s/shard | "
                f"ETA {eta_seconds/60:.1f}min"
            )
            shard_pbar.update(1)
        else:
            print(
                f"  shard {local_idx+1}/{len(shards)} (id={shard_id}): "
                f"{len(shard)} recordings, {actual_size_mb:.1f} MB "
                f"[elapsed {elapsed:.1f}s, ETA {eta_seconds/60:.1f}min]",
                flush=True,
            )

    try:
        for local_idx, shard in enumerate(shards):
            _build_one_shard(local_idx + shard_id_offset, shard)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        if shard_pbar is not None:
            shard_pbar.close()

    # 인덱스 저장 — incremental이면 기존과 머지
    if existing_index is not None:
        merged_rec_to_shard = dict(existing_index.get("rec_to_shard", {}))
        merged_rec_to_shard.update(rec_to_shard)
        merged_shards = list(existing_index.get("shards", [])) + shard_meta
        index = {
            "n_shards": len(merged_shards),
            "n_recordings": rec_global_idx,
            "target_shard_mb": args.target_shard_mb,
            "source_manifest": str(manifest_path),
            "rec_to_shard": merged_rec_to_shard,
            "shards": merged_shards,
            "build_seconds": round(time.time() - t_start, 2),
            "incremental_runs": existing_index.get("incremental_runs", 0) + 1,
        }
    else:
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

    if existing_index is not None:
        print(
            f"\nDone. Wrote {len(shards)} new shards "
            f"(total now {index['n_shards']}) + {index_path}"
        )
    else:
        print(f"\nDone. Wrote {len(shards)} shards + {index_path}")
    print(f"Total time: {index['build_seconds']:.1f}s")


if __name__ == "__main__":
    main()
