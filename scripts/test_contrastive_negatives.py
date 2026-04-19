"""Phase 2 Contrastive negative 구성 분석.

실제 VitalDB/K-MIMIC 데이터로 GroupedBatchSampler + PackCollate(any_variate)
배치 생성하고 각 anchor의 negative pool 분포 측정.

측정 카테고리:
  - Positive: same (sample_id, time_id), different variate_id
  - Pseudo-negative (risky): 같은 session의 다른 time (같은 환자, 시간만 다름)
  - True negative (good): 다른 session (다른 환자 or 다른 시간대 세션)

Usage:
  # 로컬 (processed_h5, 2 subjects)
  python -m scripts.test_contrastive_negatives

  # 서버 (VitalDB 실데이터)
  python -m scripts.test_contrastive_negatives \
      --data-dir /home/coder/workspace/updown/bio_fm/data/train/vitaldb/ \
      --batch-size 64 --max-length 120000 --n-batches 50

  # K-MIMIC 추가
  python -m scripts.test_contrastive_negatives \
      --data-dir /home/coder/workspace/updown/bio_fm/data/train/vitaldb/ \
                 /home/coder/workspace/updown/bio_fm/data/train/k-mimic/ \
      --batch-size 64 --max-length 120000 --n-batches 50
"""
from __future__ import annotations

import argparse
import random
import statistics as st
from collections import Counter
from pathlib import Path

import torch

from data.collate import PackCollate
from data.dataloader import create_dataloader
from data.dataset import BiosignalDataset
from train.train_utils import load_manifest_from_processed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        nargs="+",
        default=["datasets/processed_h5"],
        help="manifest 디렉토리 (여러 개 가능)",
    )
    p.add_argument("--signal-types", type=int, nargs="+",
                   default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--max-subjects", type=int, default=None,
                   help="최대 subject 수 (None=전체)")
    p.add_argument("--patch-size", type=int, default=200)
    p.add_argument("--min-patches", type=int, default=5)
    p.add_argument("--max-length", type=int, default=120000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--window-seconds", type=float, default=600.0)
    p.add_argument("--crop-ratio-min", type=float, default=0.01)
    p.add_argument("--crop-ratio-max", type=float, default=1.0)
    p.add_argument("--n-batches", type=int, default=20,
                   help="분석할 배치 수")
    p.add_argument("--n-anchors-per-row", type=int, default=50,
                   help="row당 샘플링할 anchor 수 (전체가 50 이하면 전부 사용)")
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    patch_size = args.patch_size
    min_patches = args.min_patches
    max_length = args.max_length
    batch_size = args.batch_size
    data_dir = args.data_dir if len(args.data_dir) > 1 else args.data_dir[0]

    print(f"Loading manifest from: {data_dir}")
    manifest = load_manifest_from_processed(
        data_dir,
        signal_types=args.signal_types,
        max_subjects=args.max_subjects,
    )
    print(f"Loaded {len(manifest)} recordings")

    crop_range = None
    if args.crop_ratio_min > 0 and args.crop_ratio_max > 0:
        crop_range = (args.crop_ratio_min, args.crop_ratio_max)

    dataset = BiosignalDataset(
        manifest,
        window_seconds=args.window_seconds,
        patch_size=patch_size,
        min_patches=min_patches,
        crop_ratio_range=crop_range,
    )
    print(f"Dataset size: {len(dataset)} windows")

    random.seed(42)
    torch.manual_seed(42)

    dataloader = create_dataloader(
        dataset,
        max_length=max_length,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_mode="any_variate",
        patch_size=patch_size,
        stride=patch_size,
        min_patches=min_patches,
    )

    # 배치 분석
    n_batches_to_analyze = args.n_batches
    stats_per_anchor = {
        "positives": [],
        "same_session_neg": [],   # pseudo-negative (risky)
        "cross_session_neg": [],  # true negative (good)
    }
    row_level_stats = []

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= n_batches_to_analyze:
            break

        # Extract metadata from PackedBatch
        sample_id = batch.sample_id              # (B, L)
        variate_id = batch.variate_id            # (B, L)
        B, L = sample_id.shape

        # Downsample to patch-level (take every patch_size-th sample)
        p_sid = sample_id[:, ::patch_size]       # (B, N)
        p_vid = variate_id[:, ::patch_size]
        # patch_mask: valid if variate_id > 0 and sample_id > 0
        patch_mask = (p_sid > 0) & (p_vid > 0)   # (B, N)

        # 실제 time_id: (sample_id, variate_id) 조합이 바뀔 때마다 0부터 리셋
        # -> ECG patch 0과 ABP patch 0이 같은 time_id 공유 -> positive pair 성립
        N = p_sid.shape[1]
        combined = p_sid * (p_vid.max() + 1) + p_vid
        boundary = torch.ones(B, N, dtype=torch.bool)
        boundary[:, 1:] = combined[:, 1:] != combined[:, :-1]
        arange = torch.arange(N).unsqueeze(0).expand(B, -1)
        boundary_pos = torch.where(boundary, arange, torch.zeros_like(arange))
        group_start, _ = boundary_pos.cummax(dim=-1)
        time_id = arange - group_start  # (B, N)

        # Row 단위 통계
        for b in range(B):
            valid_idx = torch.where(patch_mask[b])[0]
            n_valid = len(valid_idx)
            if n_valid < 2:
                continue

            # 한 row 내 unique sample_id (= 서로 다른 units)
            unique_sids = p_sid[b][patch_mask[b]].unique()
            row_level_stats.append({
                "n_valid_patches": n_valid,
                "n_units": len(unique_sids),
            })

            # 각 valid patch를 anchor로 삼아 분석
            # 너무 많으면 샘플링 (--n-anchors-per-row)
            anchor_indices = valid_idx.tolist()
            if len(anchor_indices) > args.n_anchors_per_row:
                anchor_indices = random.sample(
                    anchor_indices, args.n_anchors_per_row
                )

            for anchor_n in anchor_indices:
                a_sid = p_sid[b, anchor_n].item()
                a_vid = p_vid[b, anchor_n].item()
                a_tid = time_id[b, anchor_n].item()

                positives = 0
                same_session_neg = 0
                cross_session_neg = 0

                for other_n in valid_idx.tolist():
                    if other_n == anchor_n:
                        continue
                    o_sid = p_sid[b, other_n].item()
                    o_vid = p_vid[b, other_n].item()
                    o_tid = time_id[b, other_n].item()

                    # Positive: same sample_id + same time_id + different variate
                    is_pos = (o_sid == a_sid) and (o_tid == a_tid) and (o_vid != a_vid)

                    if is_pos:
                        positives += 1
                    elif o_sid == a_sid:
                        # same unit (same session/slot), not positive -> pseudo-negative
                        same_session_neg += 1
                    else:
                        # different sample_id -> truly different session/slot -> good negative
                        cross_session_neg += 1

                stats_per_anchor["positives"].append(positives)
                stats_per_anchor["same_session_neg"].append(same_session_neg)
                stats_per_anchor["cross_session_neg"].append(cross_session_neg)

    # ── 결과 출력 ──
    print(f"\n{'='*70}")
    print(f"Analyzed {n_batches_to_analyze} batches, {len(stats_per_anchor['positives'])} anchors")
    print(f"{'='*70}")

    def stats(values):
        if not values:
            return 0, 0, 0.0, 0, 0
        return min(values), st.median(values), st.mean(values), max(values), len(values)

    print(f"\n--- Row-level statistics ---")
    n_units_dist = Counter(r["n_units"] for r in row_level_stats)
    n_patches_dist = [r["n_valid_patches"] for r in row_level_stats]
    print(f"Rows analyzed: {len(row_level_stats)}")
    print(f"Units per row distribution: {dict(sorted(n_units_dist.items()))}")
    print(f"  (1 unit = only 1 session/slot -> no cross-session negatives)")
    print(f"Valid patches per row: min={min(n_patches_dist)}, "
          f"median={st.median(n_patches_dist):.0f}, "
          f"mean={st.mean(n_patches_dist):.0f}, "
          f"max={max(n_patches_dist)}")

    print(f"\n--- Per-anchor statistics ---")
    for key, label in [
        ("positives", "Positive pairs"),
        ("same_session_neg", "Same-session neg (pseudo)"),
        ("cross_session_neg", "Cross-session neg (true)"),
    ]:
        vals = stats_per_anchor[key]
        mn, med, mean, mx, n = stats(vals)
        print(f"  {label:30s}: min={mn:3d}, median={med:5.1f}, mean={mean:5.1f}, max={mx:3d}")

    # Negative pool 비율
    total_same = sum(stats_per_anchor["same_session_neg"])
    total_cross = sum(stats_per_anchor["cross_session_neg"])
    total_neg = total_same + total_cross
    if total_neg > 0:
        print(f"\n--- Negative pool composition ---")
        print(f"  Same-session (pseudo): {total_same:7d} ({100*total_same/total_neg:5.1f}%)")
        print(f"  Cross-session (true):  {total_cross:7d} ({100*total_cross/total_neg:5.1f}%)")
        print(f"  Total negatives:       {total_neg:7d}")

    # Anchor당 negative 수 vs positive 수 비율
    n_anchors_no_cross = sum(1 for v in stats_per_anchor["cross_session_neg"] if v == 0)
    print(f"\n--- Anchors with NO true negatives ---")
    print(f"  {n_anchors_no_cross} / {len(stats_per_anchor['cross_session_neg'])} "
          f"({100*n_anchors_no_cross/max(1,len(stats_per_anchor['cross_session_neg'])):.1f}%)")
    print(f"  (anchor's row has only 1 unit -> pure same-session negatives)")

    # ── Same-slot exclude 필터 시뮬레이션 ──
    # Filter: 같은 sample_id인 non-positive pair를 denominator에서 제외
    # -> within-slot pseudo-neg (같은 10분 slot, 시간만 다른) 완전 제거
    # -> anchor의 denom = positives + cross-session neg만
    # -> cross-session neg가 0이면 denom = numer -> InfoNCE loss=0 -> 자동 skip
    print(f"\n{'='*70}")
    print(f"AFTER FILTER (same-slot exclude in InfoNCE denominator)")
    print(f"{'='*70}")

    total_anchors = len(stats_per_anchor["positives"])

    # 필터 적용 후 effective anchor = cross_session_neg > 0
    contributing = [
        (p, c) for p, c in zip(
            stats_per_anchor["positives"],
            stats_per_anchor["cross_session_neg"],
        ) if c > 0
    ]
    n_contrib = len(contributing)
    n_skipped = total_anchors - n_contrib

    print(f"\n--- Anchor participation ---")
    print(f"  Contributing anchors: {n_contrib}/{total_anchors} "
          f"({100*n_contrib/max(1,total_anchors):.1f}%)")
    print(f"    (cross-session neg >= 1 -> clean InfoNCE 계산)")
    print(f"  Auto-skipped:         {n_skipped}/{total_anchors} "
          f"({100*n_skipped/max(1,total_anchors):.1f}%)")
    print(f"    (cross-session neg = 0 -> loss=0, gradient=0, 안 해침)")

    if contributing:
        pos_vals = [p for p, _ in contributing]
        neg_vals = [c for _, c in contributing]
        print(f"\n--- Contributing anchors only - clean InfoNCE stats ---")
        mn, med, mean, mx, _ = stats(pos_vals)
        print(f"  Positives  : min={mn:3d}, median={med:5.1f}, mean={mean:5.1f}, max={mx:3d}")
        mn, med, mean, mx, _ = stats(neg_vals)
        print(f"  True negs  : min={mn:3d}, median={med:5.1f}, mean={mean:5.1f}, max={mx:3d}")

        # InfoNCE ratio
        ratios = [c / max(1, p) for p, c in contributing]
        mn, med, mean, mx, _ = stats(ratios)
        print(f"  Neg/Pos ratio: min={mn:5.1f}, median={med:5.1f}, "
              f"mean={mean:5.1f}, max={mx:5.1f}")
        print(f"    (InfoNCE는 이 ratio가 클수록 강한 대조 학습 - 10+ 이상 권장)")

        # Gradient budget 효율
        total_contrib_pairs = sum(pos_vals) + sum(neg_vals)
        total_gradient_before = sum(stats_per_anchor["positives"]) + \
                                sum(stats_per_anchor["same_session_neg"]) + \
                                sum(stats_per_anchor["cross_session_neg"])
        clean_ratio = total_contrib_pairs / max(1, total_gradient_before) * 100
        print(f"\n--- Gradient budget 효율 ---")
        print(f"  필터 전 전체 pair 계산량: {total_gradient_before:,}")
        print(f"  필터 후 clean pair:       {total_contrib_pairs:,} "
              f"({clean_ratio:.1f}%)")
        print(f"  낭비된 pseudo pair:       {total_gradient_before - total_contrib_pairs:,} "
              f"({100-clean_ratio:.1f}%)")
        print(f"  -> 필터 적용 시 pseudo gradient 제거, 나머지 clean signal로 학습")

    # 최종 권고
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"  필터 전: {total_anchors} anchors, negative pool 중 "
          f"{100*sum(stats_per_anchor['cross_session_neg'])/max(1,total_neg):.1f}% true")
    print(f"  필터 후: {n_contrib} contributing ({100*n_contrib/max(1,total_anchors):.1f}%), "
          f"각각 평균 {st.mean([c for _,c in contributing]) if contributing else 0:.0f}개 clean negative")
    n_good = sum(1 for p, c in contributing if c >= 10)
    print(f"  그 중 neg >= 10 (건강한 InfoNCE): {n_good}/{n_contrib} "
          f"({100*n_good/max(1,n_contrib):.1f}% of contributing)")


if __name__ == "__main__":
    main()
