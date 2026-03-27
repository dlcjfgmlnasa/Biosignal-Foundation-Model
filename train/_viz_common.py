# -*- coding:utf-8 -*-
"""시각화 공통 유틸리티.

``viz_recon.py``와 ``viz_next_pred.py``에서 공유하는 데이터 구조,
signal type 매핑, 후보 선택, 플로팅 함수를 정의한다.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from data.collate import PackedBatch


SIGNAL_TYPE_NAMES = {0: "ECG", 1: "ABP", 2: "EEG", 3: "PPG", 4: "CVP", 5: "CO2", 6: "AWP"}


# ── 내부 데이터 구조 ──────────────────────────────────────────────


@dataclass
class RowCandidate:
    """시각화 후보 row 데이터."""
    orig_patches: np.ndarray     # (N_valid, P)
    pred_patches: np.ndarray     # (N_valid, P)
    signal_type: int
    signal_name: str
    n_valid: int
    patch_size: int


# ── 공통 유틸 ─────────────────────────────────────────────────────


def batch_to_device(batch: PackedBatch, device: torch.device) -> None:
    """배치의 주요 텐서를 device로 이동한다 (in-place)."""
    batch.values = batch.values.to(device)
    batch.sample_id = batch.sample_id.to(device)
    batch.variate_id = batch.variate_id.to(device)


def build_signal_map(
    batch: PackedBatch,
    p_sid: torch.Tensor,  # (B, N) patch_sample_id
    p_vid: torch.Tensor,  # (B, N) patch_variate_id
    patch_mask: torch.Tensor,  # (B, N)
    B: int,
) -> dict[tuple[int, int, int], int]:
    """(row, sample_id, variate_id) → signal_type 매핑을 반환한다.

    ``batch.signal_types``는 ``(total_variates,)``로, 배치 전체의 variate를
    row 순서 → unit(sample) 순서 → variate 순서로 나열한 것이다.
    CI 모드에서는 각 unit이 1개 variate를 가지므로 sample_id로 구분해야 한다.
    """
    has_signal_types = hasattr(batch, "signal_types") and batch.signal_types is not None
    mapping: dict[tuple[int, int, int], int] = {}
    if not has_signal_types:
        return mapping

    gvi = 0
    for b in range(B):
        valid = patch_mask[b]
        if not valid.any():
            continue

        sid_valid = p_sid[b, valid]
        vid_valid = p_vid[b, valid]
        seen: set[tuple[int, int]] = set()
        ordered_pairs: list[tuple[int, int]] = []
        for i in range(len(sid_valid)):
            pair = (int(sid_valid[i].item()), int(vid_valid[i].item()))
            if pair not in seen:
                seen.add(pair)
                ordered_pairs.append(pair)

        for sid, vid in ordered_pairs:
            if gvi < len(batch.signal_types):
                mapping[(b, sid, vid)] = batch.signal_types[gvi].item()
            gvi += 1

    return mapping


def select_diverse(candidates: list[RowCandidate], max_rows: int) -> list[RowCandidate]:
    """신호 타입이 고루 분포하도록 후보를 선택한다 (하위 호환).

    각 신호 타입에서 가장 긴(n_valid가 큰) 후보 1개를 선택한다.
    """
    grid = select_diverse_grid(candidates, samples_per_type=1)
    selected: list[RowCandidate] = []
    for t in sorted(grid.keys()):
        if grid[t]:
            selected.append(grid[t][0])
        if len(selected) >= max_rows:
            break
    return selected


def select_diverse_grid(
    candidates: list[RowCandidate],
    samples_per_type: int = 3,
) -> dict[int, list[RowCandidate]]:
    """신호 타입별 여러 샘플을 선택한다.

    Returns
    -------
    {signal_type: [RowCandidate, ...]} — 타입별 최대 samples_per_type개.
    """
    by_type: dict[int, list[RowCandidate]] = {}
    for c in candidates:
        by_type.setdefault(c.signal_type, []).append(c)

    # 타입별로 n_valid 내림차순 정렬 → 긴 후보 우선
    for t in by_type:
        by_type[t].sort(key=lambda c: c.n_valid, reverse=True)

    grid: dict[int, list[RowCandidate]] = {}
    for t in sorted(by_type.keys()):
        items = by_type[t]
        n = min(samples_per_type, len(items))
        if n <= 0:
            continue
        # 균등 간격으로 선택 (다양한 샘플)
        indices = [int(i / max(n - 1, 1) * (len(items) - 1)) for i in range(n)]
        grid[t] = [items[idx] for idx in indices]

    return grid


def plot_figure(
    selected: list[RowCandidate],
    epoch: int,
    output_dir: Path,
    max_duration_s: float,
    sampling_rate: float,
    mode: str,  # "masked" or "next_pred"
    horizon: int = 1,
    next_pred_ratio: float = 0.3,
) -> Path:
    """선택된 후보들로 figure를 그려 저장한다 (하위 호환 — 1열)."""
    grid = {}
    for c in selected:
        grid.setdefault(c.signal_type, []).append(c)
    return plot_figure_grid(
        grid, epoch, output_dir, max_duration_s, sampling_rate,
        mode, horizon, next_pred_ratio,
    )


def plot_figure_grid(
    grid: dict[int, list[RowCandidate]],
    epoch: int,
    output_dir: Path,
    max_duration_s: float,
    sampling_rate: float,
    mode: str,  # "masked" or "next_pred"
    horizon: int = 1,
    next_pred_ratio: float = 0.3,
) -> Path:
    """행=신호 타입, 열=샘플로 grid figure를 그려 저장한다.

    Parameters
    ----------
    grid:
        {signal_type: [RowCandidate, ...]} — select_diverse_grid()의 출력.
    """
    types = sorted(grid.keys())
    n_rows = len(types)
    n_cols = max(len(v) for v in grid.values()) if grid else 1
    if n_rows == 0:
        return output_dir / f"empty_epoch{epoch:03d}.png"

    P = next(iter(grid.values()))[0].patch_size
    max_patches = max(1, int(max_duration_s * sampling_rate / P))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), squeeze=False)

    if mode == "masked":
        pred_color = "orangered"
        highlight_color = "red"
        pred_label = "Reconstructed"
        highlight_label = "Masked region"
    else:
        pred_color = "seagreen"
        highlight_color = "green"
        pred_label = f"Predicted (H={horizon})"
        highlight_label = "Prediction region"

    for row, sig_type in enumerate(types):
        cands = grid[sig_type]
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(cands):
                ax.axis("off")
                continue

            cand = cands[col]
            n_show = min(cand.n_valid, max_patches)
            orig_wave = cand.orig_patches[:n_show].reshape(-1)
            pred_patches_cropped = cand.pred_patches[:n_show].copy()

            # next_pred: crop 후 범위에서 균등 샘플링
            if mode == "next_pred":
                valid_pred_indices = [
                    p_idx for p_idx in range(n_show)
                    if not np.isnan(pred_patches_cropped[p_idx, 0])
                ]
                n_to_show = max(1, int(len(valid_pred_indices) * next_pred_ratio))
                if len(valid_pred_indices) > n_to_show:
                    sampled = np.linspace(0, len(valid_pred_indices) - 1, n_to_show, dtype=int)
                    keep = set(valid_pred_indices[s] for s in sampled)
                else:
                    keep = set(valid_pred_indices)
                for p_idx in valid_pred_indices:
                    if p_idx not in keep:
                        pred_patches_cropped[p_idx] = np.nan

            pred_wave = pred_patches_cropped.reshape(-1)
            t = np.arange(len(orig_wave)) / sampling_rate

            ax.plot(t, orig_wave, color="steelblue", linewidth=0.8, alpha=0.9)

            n_pred = 0
            for patch_idx in range(n_show):
                start = patch_idx * P
                end = start + P
                patch_pred = pred_wave[start:end]
                if not np.isnan(patch_pred[0]):
                    ax.axvspan(start / sampling_rate, end / sampling_rate,
                               alpha=0.12, color=highlight_color)
                    ax.plot(t[start:end], patch_pred,
                            color=pred_color, linewidth=1.2)
                    n_pred += 1

            if col == 0:
                ax.set_ylabel(cand.signal_name, fontsize=10)

            duration_shown = n_show * P / sampling_rate
            ax.set_title(
                f"{n_pred}/{n_show} patches | {duration_shown:.0f}s",
                fontsize=8, loc="right",
            )
            ax.tick_params(labelsize=7)
            if row == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    # 범례
    legend_elements = [
        Line2D([0], [0], color="steelblue", linewidth=1, label="Original"),
        Line2D([0], [0], color=pred_color, linewidth=1.2, label=pred_label),
        Patch(facecolor=highlight_color, alpha=0.12, label=highlight_label),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="upper right", fontsize=7)

    if mode == "masked":
        title = f"Masked Reconstruction — Epoch {epoch}"
        fname = f"recon_epoch{epoch:03d}.png"
    else:
        title = f"Next-Patch Prediction (H={horizon}) — Epoch {epoch}"
        fname = f"next_pred_epoch{epoch:03d}.png"

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    path = output_dir / fname
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path