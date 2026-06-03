# -*- coding:utf-8 -*-
"""any_variate cross-modal co-occurrence 진단.

GroupedBatchSampler/collate 와 동일한 (session_id, time_slot) 그룹핑을 manifest
레벨에서 재현하여, cross-modal pair 가 실제로 얼마나 생기는지 측정한다.
Phase 2 학습 로그에서 cross/contrastive loss 가 0 으로 자주 찍히는 원인이
(A) session_id 부재인지 (B) 같은 슬롯 multi-modal 부족인지 숫자로 확정한다.

Usage:
    python -m scripts.diagnose_cooccurrence --config configs/ablation/_phase2_base.yaml
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from data.dataset import BiosignalDataset
from data.spatial_map import SIGNAL_TYPE_NAMES
from train.train_utils import (
    TrainConfig,
    load_manifest_from_processed,
    split_manifest_by_subject,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="phase2 yaml (slot_size 등 일치)")
    ap.add_argument("--use-val-split", action="store_true",
                    help="train split 만 분석 (학습과 동일). 미지정 시 전체 manifest")
    args = ap.parse_args()

    config = TrainConfig.from_yaml(args.config)
    # GroupedBatchSampler/collate 와 동일: any_variate slot_size = max_length // 2
    slot_size = config.max_length // 2

    print(f"Loading manifest from {config.data_dir} ...")
    manifest = load_manifest_from_processed(
        config.data_dir,
        signal_types=config.signal_types,
        max_subjects=config.max_subjects,
        subject_ids_file=config.subject_ids_file,
    )
    if args.use_val_split and config.val_ratio > 0:
        manifest, _ = split_manifest_by_subject(
            manifest, val_ratio=config.val_ratio, seed=config.seed
        )

    ds = BiosignalDataset(
        manifest,
        window_seconds=config.window_seconds,
        cache_size=config.cache_size,
        patch_size=config.model_config.patch_size,
        min_patches=config.min_patches,
        shard_index_path=config.shard_index_path,
        shard_cache_size=config.shard_cache_size,
    )

    # ── (session_id, slot) → {signal_type} 그룹핑 (sampler.py 와 동일 로직) ──
    slot_sig: dict[tuple, set[int]] = defaultdict(set)
    no_session = 0
    total_rec = len(ds._manifest)
    win_total = 0

    for rec_idx, entry in enumerate(ds._manifest):
        n_win = ds._n_windows_per_rec[rec_idx]
        if n_win == 0:
            continue
        stride = ds._strides_per_rec[rec_idx]
        win_total += n_win
        if not entry.session_id:
            no_session += 1
            continue
        for w in range(n_win):
            slot = (entry.start_sample + w * stride) // slot_size
            slot_sig[(entry.session_id, slot)].add(entry.signal_type)

    n_groups = len(slot_sig)
    size_dist = Counter(len(v) for v in slot_sig.values())
    multi = sum(c for k, c in size_dist.items() if k >= 2)

    # multi-modal 그룹에서 어떤 signal_type 조합이 흔한지
    pair_counter: Counter = Counter()
    for sigset in slot_sig.values():
        if len(sigset) >= 2:
            for a in sorted(sigset):
                for b in sorted(sigset):
                    if a < b:
                        pair_counter[(a, b)] += 1

    print("=" * 64)
    print(f"slot_size          = {slot_size} samples ({slot_size/100:.0f}s @100Hz)")
    print(f"recordings         = {total_rec}, windows = {win_total}")
    print(f"session_id 비어있음 = {no_session} "
          f"({100*no_session/max(total_rec,1):.1f}% of recordings)")
    print(f"(session,slot) 그룹  = {n_groups}")
    print(f"  signal_type 개수 분포 = {dict(sorted(size_dist.items()))}")
    print(f"  multi-modal(2+) 그룹 = {multi} ({100*multi/max(n_groups,1):.1f}%)")
    print("-" * 64)
    print("  multi-modal 그룹의 흔한 signal_type 쌍 (top 12):")
    for (a, b), c in pair_counter.most_common(12):
        na = SIGNAL_TYPE_NAMES.get(a, str(a))
        nb = SIGNAL_TYPE_NAMES.get(b, str(b))
        print(f"    ({a},{b}) {na:>14}↔{nb:<14} {c:>8}")
    print("=" * 64)

    # ── 판정 ──
    if no_session > total_rec * 0.5:
        print("⚠️ session_id 가 대부분 비어있음 → (session,slot) 그룹화 불가가")
        print("   cross/contrastive 0 빈발의 근본 원인. sampler 의 시간 키를")
        print("   session_id 대신 다른 식별자(예: subject+절대시간)로 바꿔야 함.")
    elif n_groups > 0 and multi < n_groups * 0.2:
        print(f"⚠️ multi-modal 그룹 비율 {100*multi/n_groups:.1f}% 로 낮음 →")
        print("   같은 시간슬롯 co-occurrence 부족이 0 빈발 원인.")
        print("   slot_size 확대 / 시간정렬 manifest / sampling 보강 검토.")
    else:
        print("✓ co-occurrence 는 충분 → 0 빈발은 배치 단위 희석(개별 배치에")
        print("   multi-modal 그룹이 가끔만 들어옴) 또는 variate_id>0 조건 영향.")


if __name__ == "__main__":
    main()
