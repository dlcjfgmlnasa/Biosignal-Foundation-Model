# -*- coding:utf-8 -*-
"""Shard 파일 빠른 로딩 테스트.

build_shards.py가 진행 중이거나 일부만 완료된 상태에서도 만들어진 shard
파일을 읽어 정합성과 속도를 확인할 수 있다.

사용법:
    # 단일 shard 로드 + 내용 확인
    python -m scripts.test_shard_load --shard /path/to/vitaldb_wbs/shard_00000.pt

    # 여러 shard 순차 로딩 시간 측정
    python -m scripts.test_shard_load --shard-dir /path/to/vitaldb_wbs --n 5

    # BiosignalDataset 통합 테스트 (sampler 포함)
    python -m scripts.test_shard_load \\
        --data-dir /path/to/processed/vitaldb \\
        --shard-index /path/to/vitaldb_wbs/shard_index.json \\
        --integration
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


def test_one_shard(shard_path: Path) -> dict:
    """단일 shard 로드 + 구조 검증."""
    print(f"\n=== Loading {shard_path.name} ===")
    print(f"  File size: {shard_path.stat().st_size / 1024**2:.1f} MB")

    t0 = time.time()
    shard = torch.load(shard_path, weights_only=True)
    t_load = time.time() - t0
    print(f"  torch.load: {t_load:.2f}s")

    if not isinstance(shard, dict):
        print(f"  ERROR: shard is {type(shard).__name__}, expected dict")
        return {"ok": False}

    print(f"  Recordings in shard: {len(shard)}")
    if not shard:
        return {"ok": True, "n_recordings": 0}

    # 첫 번째 entry 구조 확인
    first_key = next(iter(shard))
    first = shard[first_key]
    print(f"  First key: {first_key!r}")
    print(f"  First entry type: {type(first).__name__}")

    if isinstance(first, dict):
        print(f"  First entry fields: {list(first.keys())}")
        if "values" in first:
            v = first["values"]
            print(
                f"  values: shape={tuple(v.shape) if torch.is_tensor(v) else 'NOT TENSOR'}, "
                f"dtype={v.dtype if torch.is_tensor(v) else type(v).__name__}"
            )
        for k in ("subject_id", "session_id", "signal_type", "sampling_rate"):
            if k in first:
                print(f"  {k}: {first[k]!r}")

    # 평균 텐서 크기
    total_samples = 0
    n_with_values = 0
    for v in shard.values():
        if isinstance(v, dict) and "values" in v and torch.is_tensor(v["values"]):
            total_samples += v["values"].numel()
            n_with_values += 1
    if n_with_values:
        avg_samples = total_samples / n_with_values
        print(
            f"  Total tensor samples: {total_samples:,} "
            f"(avg {avg_samples:,.0f} per recording)"
        )

    # subject 다양성
    subjects = {
        v.get("subject_id") for v in shard.values()
        if isinstance(v, dict)
    }
    print(f"  Unique subjects in shard: {len(subjects)}")

    return {
        "ok": True,
        "n_recordings": len(shard),
        "load_seconds": t_load,
        "size_mb": shard_path.stat().st_size / 1024**2,
        "subjects": len(subjects),
    }


def test_sequential_loads(shard_dir: Path, n: int) -> None:
    """N개 shard 순차 로드 — sustained throughput 측정."""
    shards = sorted(shard_dir.glob("shard_*.pt"))[:n]
    if not shards:
        print(f"No shard_*.pt files in {shard_dir}")
        return

    print(f"\n=== Sequential load test: {len(shards)} shards ===")
    total_size = 0
    total_time = 0.0
    for sp in shards:
        t0 = time.time()
        shard = torch.load(sp, weights_only=True)
        t = time.time() - t0
        sz = sp.stat().st_size / 1024**2
        n_rec = len(shard) if isinstance(shard, dict) else 0
        print(f"  {sp.name}: {sz:.0f} MB, {n_rec} recordings, {t:.2f}s "
              f"({sz / max(t, 0.001):.0f} MB/s)")
        total_size += sz
        total_time += t
        del shard

    if total_time > 0:
        print(f"\n  Total: {total_size:.0f} MB in {total_time:.1f}s "
              f"= {total_size / total_time:.0f} MB/s sustained")


def test_integration(
    data_dir: str,
    shard_index: str,
    n_samples: int = 10000,
    shard_cache_size: int = 8,
) -> None:
    """BiosignalDataset + sampler 통합 테스트.

    n_samples는 충분히 커야 의미 있음 (shard cold load 비용 분산되는 단위).
    1 shard ≈ 100K sample window 보유. 권장 최소 5000+, 가능하면 50000+.
    """
    from data.dataset import BiosignalDataset
    from data.sampler import RecordingLocalitySampler
    from train.train_utils import load_manifest_from_processed

    print(f"\n=== Integration test (n_samples={n_samples}) ===")
    print(f"  Loading manifest from {data_dir}")
    manifest = load_manifest_from_processed(data_dir)
    print(f"  Manifest: {len(manifest)} recordings")

    print(f"  Creating shard-backed BiosignalDataset (cache_size={shard_cache_size})")
    ds = BiosignalDataset(
        manifest,
        window_seconds=30.0,
        patch_size=200,
        cache_size=16,
        shard_index_path=shard_index,
        shard_cache_size=shard_cache_size,
    )
    print(f"  Dataset: {len(ds)} windows")

    sampler = RecordingLocalitySampler(ds, shuffle=True, seed=0)
    sampler.set_epoch(0)

    print(f"  Iterating first {n_samples} samples via sampler...")
    try:
        from tqdm import tqdm
        progress = tqdm(total=n_samples, desc="  samples", unit="sample")
    except ImportError:
        progress = None

    t0 = time.time()
    t_first_batch = None
    n = 0
    for idx in sampler:
        s = ds[idx]
        n += 1
        if t_first_batch is None and n >= 100:
            t_first_batch = time.time() - t0
        if progress:
            progress.update(1)
        if n >= n_samples:
            break
    t = time.time() - t0
    if progress:
        progress.close()
    print(f"\n  Total: {n} samples in {t:.2f}s = {n/t:.0f} samples/sec")
    if t_first_batch:
        print(f"  First 100 samples (cold cache): {t_first_batch:.2f}s "
              f"= {100/t_first_batch:.0f} sps  (이후 cache warm)")
        if n > 100:
            warm_samples = n - 100
            warm_time = t - t_first_batch
            if warm_time > 0:
                print(f"  Remaining {warm_samples} samples (warm): "
                      f"{warm_time:.2f}s = {warm_samples/warm_time:.0f} sps")
    print(f"  Sample[last]: length={s.length}, signal_type={s.signal_type}, "
          f"shape={tuple(s.values.shape)}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shard", help="단일 shard 파일 경로")
    p.add_argument("--shard-dir", help="여러 shard가 있는 디렉토리")
    p.add_argument("--n", type=int, default=3, help="--shard-dir 사용 시 로드할 shard 수")
    p.add_argument("--data-dir", help="--integration 사용 시 manifest 디렉토리")
    p.add_argument("--shard-index", help="--integration 사용 시 shard_index.json 경로")
    p.add_argument(
        "--integration",
        action="store_true",
        help="BiosignalDataset+sampler 통합 테스트",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="--integration 시 측정 샘플 수 (기본 10000). "
             "5000 미만이면 shard cold load 비용 미분산 → 잘못된 결론 가능.",
    )
    p.add_argument(
        "--shard-cache-size",
        type=int,
        default=8,
        help="--integration 시 shard LRU cache 크기 (기본 8 = 8GB)",
    )
    args = p.parse_args()

    if args.shard:
        test_one_shard(Path(args.shard))

    if args.shard_dir:
        test_sequential_loads(Path(args.shard_dir), args.n)

    if args.integration:
        if not (args.data_dir and args.shard_index):
            print("--integration requires --data-dir and --shard-index")
            sys.exit(1)
        test_integration(
            args.data_dir,
            args.shard_index,
            n_samples=args.n_samples,
            shard_cache_size=args.shard_cache_size,
        )

    if not (args.shard or args.shard_dir or args.integration):
        print("아무 옵션도 안 줬음. --shard / --shard-dir / --integration 중 하나 사용.")
        p.print_help()


if __name__ == "__main__":
    main()
