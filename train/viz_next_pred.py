# -*- coding:utf-8 -*-
"""Next-Patch Prediction 시각화.

에폭 종료 후 next-patch prediction의 원본 vs 예측 파형을 비교한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.collate import PackedBatch
from model import BiosignalFoundationModel

from ._viz_common import (
    SIGNAL_TYPE_NAMES,
    RowCandidate,
    batch_to_device,
    build_signal_map,
    plot_figure,
    select_diverse,
)


def _process_batch(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    horizon: int,
    device: torch.device | None,
) -> list[RowCandidate]:
    """Next-patch prediction 후보를 (sample_id, variate_id) 단위로 추출한다."""
    if device is not None:
        batch_to_device(batch, device)

    out = model(batch, task="next_pred", horizon=horizon)
    next_pred = out["next_pred"]      # (B, N, P)
    patch_mask = out["patch_mask"]    # (B, N)
    p_sid = out["patch_sample_id"]    # (B, N)
    p_vid = out["patch_variate_id"]   # (B, N)

    P = model.patch_size
    loc = out["loc"]    # (B, L, 1)
    scale = out["scale"]  # (B, L, 1)
    normalized = ((batch.values.unsqueeze(-1) - loc) / scale).squeeze(-1)
    B, L = normalized.shape
    N = L // P
    original_patches = batch.values[:, :N * P].reshape(B, N, P)  # 원본 스케일

    # denormalize용 per-patch loc/scale
    stride = model.patch_embed.stride
    patch_starts = torch.arange(N, device=loc.device) * stride
    patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
    patch_loc = loc[:, patch_starts, 0]      # (B, N)
    patch_scale = scale[:, patch_starts, 0]  # (B, N)

    sig_map = build_signal_map(batch, p_sid, p_vid, patch_mask, B)

    # V2 여부 감지
    is_v2 = "eeg_mask" in out or "eeg_reconstructed" in out

    candidates: list[RowCandidate] = []
    for b in range(B):
        valid = patch_mask[b]
        if not valid.any():
            continue

        combo = p_sid[b] * 10000 + p_vid[b]
        unique_combos = combo[valid].unique().tolist()
        for c in unique_combos:
            sid = int(c) // 10000
            vid = int(c) % 10000
            seg_mask = valid & (combo == c)
            seg_indices = seg_mask.nonzero(as_tuple=True)[0]
            n_seg = len(seg_indices)
            if n_seg <= horizon:
                continue

            seg_orig = original_patches[b, seg_indices].cpu().numpy()
            # next_pred는 정규화된 공간 → 원본 스케일로 denormalize
            seg_next_norm = next_pred[b, seg_indices]  # (n_seg, P)
            seg_loc = patch_loc[b, seg_indices].unsqueeze(-1)    # (n_seg, 1)
            seg_scale = patch_scale[b, seg_indices].unsqueeze(-1)  # (n_seg, 1)
            seg_next = (seg_next_norm * seg_scale + seg_loc).cpu().numpy()
            pred = np.full((n_seg, P), np.nan)
            pred[horizon:n_seg] = seg_next[:n_seg - horizon]

            sig_type = sig_map.get((b, sid, vid), -1)
            # V2: EEG는 embedding space 복원이라 raw 시각화 불가 → 스킵
            if is_v2 and sig_type == 2:
                continue
            sig_name = SIGNAL_TYPE_NAMES.get(sig_type, "?")

            candidates.append(RowCandidate(
                orig_patches=seg_orig,
                pred_patches=pred,
                signal_type=sig_type,
                signal_name=sig_name,
                n_valid=n_seg,
                patch_size=P,
            ))

    return candidates


@torch.no_grad()
def save_next_pred_figure(
    model: BiosignalFoundationModel,
    batch: PackedBatch | list[PackedBatch],
    epoch: int,
    output_dir: str | Path,
    horizon: int = 1,
    max_rows: int = 4,
    max_duration_s: float = 60.0,
    sampling_rate: float = 100.0,
    device: torch.device | None = None,
) -> Path:
    """Next-patch prediction의 원본 vs 예측 비교 figure를 저장한다."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = batch if isinstance(batch, list) else [batch]

    all_candidates: list[RowCandidate] = []
    for b in batches:
        all_candidates.extend(_process_batch(model, b, horizon, device))

    selected = select_diverse(all_candidates, max_rows)

    if not selected:
        model.train()
        return output_dir / f"next_pred_epoch{epoch:03d}.png"

    path = plot_figure(
        selected, epoch, output_dir,
        max_duration_s, sampling_rate, mode="next_pred", horizon=horizon,
    )
    model.train()
    return path