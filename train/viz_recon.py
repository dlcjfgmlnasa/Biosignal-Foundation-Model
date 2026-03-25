# -*- coding:utf-8 -*-
"""Masked Reconstruction 시각화.

에폭 종료 후 마스킹된 패치의 원본 vs 복원 파형을 비교한다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.collate import PackedBatch
from loss.masked_mse_loss import create_patch_mask
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
    mask_ratio: float,
    device: torch.device | None,
) -> list[RowCandidate]:
    """Masked reconstruction 후보를 (sample_id, variate_id) 단위로 추출한다."""
    if device is not None:
        batch_to_device(batch, device)

    out = model(batch, task="masked")
    reconstructed = out["reconstructed"]  # (B, N, P)
    patch_mask = out["patch_mask"]        # (B, N)
    p_sid = out["patch_sample_id"]        # (B, N)
    p_vid = out["patch_variate_id"]       # (B, N)

    pred_mask = create_patch_mask(patch_mask, mask_ratio=mask_ratio)

    P = model.patch_size
    normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
    B, L = normalized.shape
    N = L // P
    original_patches = normalized[:, :N * P].reshape(B, N, P)

    sig_map = build_signal_map(batch, p_sid, p_vid, patch_mask, B)

    candidates: list[RowCandidate] = []
    for b in range(B):
        valid = patch_mask[b]
        if not valid.any():
            continue

        masked = pred_mask[b] & valid

        combo = p_sid[b] * 10000 + p_vid[b]
        unique_combos = combo[valid].unique().tolist()
        for c in unique_combos:
            sid = int(c) // 10000
            vid = int(c) % 10000
            seg_mask = valid & (combo == c)
            n_seg = seg_mask.sum().item()
            if n_seg == 0:
                continue

            seg_indices = seg_mask.nonzero(as_tuple=True)[0]
            seg_orig = original_patches[b, seg_indices].cpu().numpy()
            seg_masked = masked[seg_indices].cpu().numpy()
            seg_recon = reconstructed[b, seg_indices].cpu().numpy()

            pred = np.full_like(seg_orig, np.nan)
            pred[seg_masked] = seg_recon[seg_masked]

            sig_type = sig_map.get((b, sid, vid), -1)
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
def save_reconstruction_figure(
    model: BiosignalFoundationModel,
    batch: PackedBatch | list[PackedBatch],
    epoch: int,
    output_dir: str | Path,
    mask_ratio: float = 0.15,
    max_rows: int = 4,
    max_duration_s: float = 60.0,
    sampling_rate: float = 100.0,
    device: torch.device | None = None,
) -> Path:
    """마스킹된 패치의 원본 vs 복원 비교 figure를 저장한다."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = batch if isinstance(batch, list) else [batch]

    all_candidates: list[RowCandidate] = []
    for b in batches:
        all_candidates.extend(_process_batch(model, b, mask_ratio, device))

    selected = select_diverse(all_candidates, max_rows)

    if not selected:
        model.train()
        return output_dir / f"recon_epoch{epoch:03d}.png"

    path = plot_figure(
        selected, epoch, output_dir,
        max_duration_s, sampling_rate, mode="masked",
    )
    model.train()
    return path