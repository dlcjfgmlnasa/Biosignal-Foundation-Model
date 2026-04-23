# -*- coding:utf-8 -*-
"""DataLoader + shard backend 메모리 사용량 측정.

목적: worker가 fork되며 shard cache 복제로 RAM 폭증하는지 확인.
공식: 총 RAM ≈ num_workers × shard_cache_size × shard_size

사용법:
    python -m scripts.bench_dataloader_memory \\
        --data-dir datasets/processed/vitaldb \\
        --shard-index datasets/sharded/vitaldb_small/shard_index.json \\
        --num-workers 4 --shard-cache-size 2 --n-batches 50
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import psutil
import torch
from torch.utils.data import DataLoader

from data.dataset import BiosignalDataset
from data.sampler import RecordingLocalitySampler
from data.dataloader import create_dataloader
from train.train_utils import load_manifest_from_processed


def get_memory_info() -> dict:
    """현재 프로세스 + 모든 자식 프로세스의 RAM 사용량."""
    p = psutil.Process(os.getpid())
    rss_self = p.memory_info().rss
    children = p.children(recursive=True)
    rss_children = sum(c.memory_info().rss for c in children if c.is_running())
    total_system = psutil.virtual_memory()
    return {
        "main_mb": rss_self / 1024**2,
        "children_mb": rss_children / 1024**2,
        "total_mb": (rss_self + rss_children) / 1024**2,
        "n_children": len(children),
        "system_used_pct": total_system.percent,
        "system_avail_gb": total_system.available / 1024**3,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--shard-index", required=True)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--shard-cache-size", type=int, default=2)
    p.add_argument("--n-batches", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--max-length", type=int, default=64000)
    p.add_argument("--persistent-workers", action="store_true", default=True)
    args = p.parse_args()

    print(f"=== DataLoader memory benchmark ===")
    print(f"  num_workers={args.num_workers}")
    print(f"  shard_cache_size={args.shard_cache_size} per worker")
    print(f"  persistent_workers={args.persistent_workers}")
    print(f"  Theoretical max RAM: {args.num_workers} × {args.shard_cache_size} × ~1GB shard "
          f"= ~{args.num_workers * args.shard_cache_size}GB worker cache")
    print()

    mem0 = get_memory_info()
    print(f"Initial: main={mem0['main_mb']:.0f}MB, "
          f"system_used={mem0['system_used_pct']:.0f}%, "
          f"avail={mem0['system_avail_gb']:.1f}GB\n")

    manifest = load_manifest_from_processed(args.data_dir)
    print(f"Loaded {len(manifest)} recordings")

    ds = BiosignalDataset(
        manifest,
        window_seconds=600.0,
        patch_size=200,
        cache_size=16,
        shard_index_path=args.shard_index,
        shard_cache_size=args.shard_cache_size,
    )
    sampler = RecordingLocalitySampler(ds, shuffle=True, seed=42)
    sampler.set_epoch(0)

    dataloader = create_dataloader(
        ds,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_mode="ci",
        patch_size=200,
        pin_memory=False,  # 측정 일관성 위해 off
        prefetch_factor=2,  # 공격적 prefetch도 메모리 영향 ↑
        persistent_workers=args.persistent_workers,
        sampler=sampler,
    )

    mem1 = get_memory_info()
    print(f"After dataloader create: main={mem1['main_mb']:.0f}MB "
          f"(+{mem1['main_mb']-mem0['main_mb']:.0f})\n")

    print(f"Iterating {args.n_batches} batches...")
    t0 = time.time()
    peak_total = 0.0
    peak_children = 0.0
    n = 0
    for batch in dataloader:
        n += 1
        if n in (1, 5, 10, 20, 50) or n == args.n_batches:
            mem = get_memory_info()
            peak_total = max(peak_total, mem["total_mb"])
            peak_children = max(peak_children, mem["children_mb"])
            print(
                f"  batch {n:3d}: main={mem['main_mb']:.0f}MB, "
                f"children={mem['children_mb']:.0f}MB ({mem['n_children']} workers), "
                f"total={mem['total_mb']:.0f}MB, "
                f"avail={mem['system_avail_gb']:.1f}GB"
            )
        if n >= args.n_batches:
            break
    elapsed = time.time() - t0

    print(f"\n=== Summary ===")
    print(f"  Batches: {n}, time: {elapsed:.1f}s ({n/elapsed:.1f} batches/sec)")
    print(f"  Peak total RAM: {peak_total:.0f}MB ({peak_total/1024:.1f}GB)")
    print(f"  Peak children RAM: {peak_children:.0f}MB ({peak_children/1024:.1f}GB)")
    if args.num_workers > 0:
        per_worker = peak_children / args.num_workers
        print(f"  Per worker (avg): {per_worker:.0f}MB")
    print(f"  System available end: {get_memory_info()['system_avail_gb']:.1f}GB")


if __name__ == "__main__":
    main()
