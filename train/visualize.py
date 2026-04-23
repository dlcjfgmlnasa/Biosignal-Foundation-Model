# -*- coding:utf-8 -*-
"""학습 중 시각화 — Masked Reconstruction & Next-Patch Prediction.

에폭 종료 후 원본 vs 복원/예측 파형을 비교하는 figure를 생성한다.
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
from model import BiosignalFoundationModel


# ── 상수 & 데이터 구조 ────────────────────────────────────────

SIGNAL_TYPE_NAMES = {
    0: "ECG",
    1: "ABP",
    2: "PPG",
    3: "CVP",
    4: "CO2",
    5: "AWP",
    6: "PAP",
    7: "ICP",
}


@dataclass
class RowCandidate:
    """시각화 후보 row 데이터."""

    orig_patches: np.ndarray  # (N_valid, P)
    pred_patches: np.ndarray  # (N_valid, P)
    signal_type: int
    signal_name: str
    n_valid: int
    patch_size: int
    # 디버그용 — 어느 batch row / sample / variate에서 왔는지 추적
    batch_idx: int = -1
    sample_id: int = -1
    variate_id: int = -1


# ── 공통 유틸 ─────────────────────────────────────────────────


def _batch_to_device(batch: PackedBatch, device: torch.device) -> None:
    """배치의 주요 텐서를 device로 이동한다 (in-place)."""
    batch.values = batch.values.to(device)
    batch.sample_id = batch.sample_id.to(device)
    batch.variate_id = batch.variate_id.to(device)


def _build_signal_map(
    batch: PackedBatch,
    p_sid: torch.Tensor,  # (B, N) patch_sample_id
    p_vid: torch.Tensor,  # (B, N) patch_variate_id
    patch_mask: torch.Tensor,  # (B, N)
    b: int,
) -> dict[tuple[int, int, int], int]:
    """(row, sample_id, variate_id) → signal_type 매핑을 반환한다."""
    has_signal_types = hasattr(batch, "signal_types") and batch.signal_types is not None
    mapping: dict[tuple[int, int, int], int] = {}
    if not has_signal_types:
        return mapping

    gvi = 0
    for bi in range(b):
        valid = patch_mask[bi]
        if not valid.any():
            continue

        sid_valid = p_sid[bi, valid]
        vid_valid = p_vid[bi, valid]
        seen: set[tuple[int, int]] = set()
        ordered_pairs: list[tuple[int, int]] = []
        for i in range(len(sid_valid)):
            pair = (int(sid_valid[i].item()), int(vid_valid[i].item()))
            if pair not in seen:
                seen.add(pair)
                ordered_pairs.append(pair)

        for sid, vid in ordered_pairs:
            if gvi < len(batch.signal_types):
                mapping[(bi, sid, vid)] = batch.signal_types[gvi].item()
            gvi += 1

    return mapping


def _select_diverse_grid(
    candidates: list[RowCandidate],
    samples_per_type: int = 3,
    min_n_valid: int = 5,
) -> dict[int, list[RowCandidate]]:
    """신호 타입별 여러 샘플을 선택한다.

    매우 짧은 candidate (예: n_valid=1)는 mask 시 모든 patch가 가려져
    실제 신호가 거의 안 보이고, 모델의 random 예측이 panel을 차지해
    "다른 신호"처럼 오인됨. min_n_valid로 필터링.

    Parameters
    ----------
    samples_per_type:
        signal type별 표시할 panel 수.
    min_n_valid:
        candidate 최소 patch 수. 이보다 짧은 segment는 시각화에서 제외.
        기본 5 = 10초 (patch_size=200, 100Hz 기준).

    Returns
    -------
    {signal_type: [RowCandidate, ...]} — 타입별 최대 samples_per_type개.
    """
    by_type: dict[int, list[RowCandidate]] = {}
    for c in candidates:
        # 너무 짧은 segment 필터링
        if c.n_valid < min_n_valid:
            continue
        by_type.setdefault(c.signal_type, []).append(c)

    for t in by_type:
        by_type[t].sort(key=lambda c: c.n_valid, reverse=True)

    grid: dict[int, list[RowCandidate]] = {}
    for t in sorted(by_type.keys()):
        items = by_type[t]
        n = min(samples_per_type, len(items))
        if n <= 0:
            continue
        indices = [int(i / max(n - 1, 1) * (len(items) - 1)) for i in range(n)]
        grid[t] = [items[idx] for idx in indices]

    return grid


def _plot_figure_grid(
    grid: dict[int, list[RowCandidate]],
    epoch: int,
    output_dir: Path,
    max_duration_s: float,
    sampling_rate: float,
    mode: str,  # "masked" or "next_pred"
    next_block_size: int = 1,
    next_pred_ratio: float = 0.3,
    context_patches: int = 10,
) -> Path:
    """행=신호 타입, 열=샘플로 grid figure를 그려 저장한다."""
    types = sorted(grid.keys())
    n_rows = len(types)
    n_cols = max(len(v) for v in grid.values()) if grid else 1

    if n_rows == 0:
        return output_dir / f"empty_epoch{epoch:03d}.png"

    p = next(iter(grid.values()))[0].patch_size
    max_patches = max(1, int(max_duration_s * sampling_rate / p))

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(8 * n_cols, 3 * n_rows), squeeze=False
    )

    if mode == "masked":
        pred_color = "orangered"
        highlight_color = "red"
        pred_label = "Reconstructed"
        highlight_label = "Masked region"
    else:
        pred_color = "crimson"
        highlight_color = "red"
        pred_label = f"Predicted block (K={next_block_size})"
        highlight_label = "Prediction block"

    for row, sig_type in enumerate(types):
        cands = grid[sig_type]
        for col in range(n_cols):
            ax = axes[row, col]
            if col >= len(cands):
                ax.axis("off")
                continue

            cand = cands[col]

            if mode == "next_pred":
                # Block Next Prediction 시각화:
                # 1) "예측 패치" 위치(=NaN이 아닌 패치)가 있는 연속 블록을 찾는다.
                # 2) 그 블록 바로 앞 ~context_patches만큼 context를 함께 그린다.
                pred_valid = ~np.isnan(cand.pred_patches[:, 0])
                block_indices = np.nonzero(pred_valid)[0]
                if len(block_indices) == 0:
                    ax.axis("off")
                    continue

                block_start = int(block_indices.min())
                block_end = int(block_indices.max()) + 1  # exclusive
                ctx_start = max(0, block_start - context_patches)
                view_end = min(cand.n_valid, block_end)

                orig_view = cand.orig_patches[ctx_start:view_end].reshape(-1)
                pred_view_patches = cand.pred_patches[ctx_start:view_end]
                pred_view = pred_view_patches.reshape(-1)
                t_offset = ctx_start * p / sampling_rate
                t = np.arange(len(orig_view)) / sampling_rate + t_offset

                # Context + ground-truth(예측 구간) 전체를 steelblue로, 예측 구간은
                # 점선으로 덧씌워 GT임을 강조.
                ax.plot(t, orig_view, color="steelblue", linewidth=0.9, alpha=0.8)

                # Predicted block: K patches를 빨간색 실선으로
                ctx_len_patches = block_start - ctx_start
                for j in range(block_end - block_start):
                    patch_idx_local = ctx_len_patches + j
                    start = patch_idx_local * p
                    end = start + p
                    ax.axvspan(
                        t[start],
                        t[end - 1] + 1.0 / sampling_rate,
                        alpha=0.12,
                        color=highlight_color,
                    )
                    ax.plot(
                        t[start:end],
                        pred_view[start:end],
                        color=pred_color,
                        linewidth=1.4,
                    )
                    # GT 점선 overlay (같은 구간)
                    ax.plot(
                        t[start:end],
                        orig_view[start:end],
                        color="gray",
                        linewidth=1.0,
                        linestyle=":",
                        alpha=0.9,
                    )

                n_pred = block_end - block_start
                n_show = view_end - ctx_start
                duration_shown = n_show * p / sampling_rate
            else:
                # Masked reconstruction: 기존 로직 유지
                n_show = min(cand.n_valid, max_patches)
                orig_wave = cand.orig_patches[:n_show].reshape(-1)
                pred_patches_cropped = cand.pred_patches[:n_show].copy()

                pred_wave = pred_patches_cropped.reshape(-1)
                t = np.arange(len(orig_wave)) / sampling_rate

                ax.plot(t, orig_wave, color="steelblue", linewidth=0.8, alpha=0.9)

                n_pred = 0
                for patch_idx in range(n_show):
                    start = patch_idx * p
                    end = start + p
                    patch_pred = pred_wave[start:end]
                    if not np.isnan(patch_pred[0]):
                        ax.axvspan(
                            start / sampling_rate,
                            end / sampling_rate,
                            alpha=0.12,
                            color=highlight_color,
                        )
                        ax.plot(
                            t[start:end], patch_pred, color=pred_color, linewidth=1.2
                        )
                        n_pred += 1
                duration_shown = n_show * p / sampling_rate

            if col == 0:
                ax.set_ylabel(cand.signal_name, fontsize=10)

            # 디버그 라벨 — 실제로 어느 batch row / sample / variate에서 왔는지
            # 명시. 같은 row(=signal_type) 안 패널이 다른 신호처럼 보이면 여기서
            # bi/sid가 다른지 확인 → 데이터 매핑 추적 가능.
            ax.set_title(
                f"{cand.signal_name} | bi={cand.batch_idx} sid={cand.sample_id} "
                f"vid={cand.variate_id} | {n_pred}/{n_show} patches | "
                f"{duration_shown:.0f}s",
                fontsize=7,
                loc="right",
            )
            ax.tick_params(labelsize=7)
            if row == n_rows - 1:
                ax.set_xlabel("Time (s)", fontsize=8)

    if mode == "next_pred":
        legend_elements = [
            Line2D(
                [0], [0], color="steelblue", linewidth=1, label="Context / Ground truth"
            ),
            Line2D([0], [0], color=pred_color, linewidth=1.4, label=pred_label),
            Line2D(
                [0], [0], color="gray", linewidth=1.0, linestyle=":", label="GT overlay"
            ),
            Patch(facecolor=highlight_color, alpha=0.12, label=highlight_label),
        ]
    else:
        legend_elements = [
            Line2D([0], [0], color="steelblue", linewidth=1, label="Original"),
            Line2D([0], [0], color=pred_color, linewidth=1.2, label=pred_label),
            Patch(facecolor=highlight_color, alpha=0.12, label=highlight_label),
        ]
    axes[0, 0].legend(handles=legend_elements, loc="upper right", fontsize=7)

    if mode == "masked":
        title = f"Masked Reconstruction - Epoch {epoch}"
        fname = f"recon_epoch{epoch:03d}.png"
    else:
        title = f"Block Next Prediction (K={next_block_size}) - Epoch {epoch}"
        fname = f"next_pred_epoch{epoch:03d}.png"

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()

    path = output_dir / fname
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ── 공통 패치 추출 헬퍼 ───────────────────────────────────────


def _extract_patches_and_scales(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    out: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """모델 출력에서 original_patches, patch_loc, patch_scale, N을 추출한다.

    Returns
    -------
    (original_patches, patch_loc, patch_scale, N)
        original_patches: (B, N, P) 원본 스케일
        patch_loc: (B, N)
        patch_scale: (B, N)
        N: 패치 수
    """
    p = model.patch_size
    loc = out["loc"]  # (B, L, 1)
    scale = out["scale"]  # (B, L, 1)
    b, l = batch.values.shape[0], batch.values.shape[1]
    n = l // p
    original_patches = batch.values[:, : n * p].reshape(b, n, p)

    stride = model.patch_embed.stride
    patch_starts = torch.arange(n, device=loc.device) * stride
    patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
    patch_loc = loc[:, patch_starts, 0]  # (B, N)
    patch_scale = scale[:, patch_starts, 0]  # (B, N)

    return original_patches, patch_loc, patch_scale, n


def _extract_candidates(
    batch: PackedBatch,
    out: dict,
    original_patches: torch.Tensor,  # (B, N, P)
    patch_loc: torch.Tensor,  # (B, N)
    patch_scale: torch.Tensor,  # (B, N)
    pred_tensor: torch.Tensor,  # (B, N, P) — reconstructed or next_pred
    build_pred_fn,  # (seg_orig, seg_pred_denorm, seg_indices, ...) -> pred array
    patch_size: int,
) -> list[RowCandidate]:
    """(sample_id, variate_id) 단위로 RowCandidate를 추출한다."""
    patch_mask = out["patch_mask"]  # (B, N)
    p_sid = out["patch_sample_id"]  # (B, N)
    p_vid = out["patch_variate_id"]  # (B, N)
    batch_size = original_patches.shape[0]

    sig_map = _build_signal_map(batch, p_sid, p_vid, patch_mask, batch_size)
    candidates: list[RowCandidate] = []
    for bi in range(batch_size):
        valid = patch_mask[bi]
        if not valid.any():
            continue

        combo = p_sid[bi] * 10000 + p_vid[bi]
        unique_combos = combo[valid].unique().tolist()
        for c in unique_combos:
            sid = int(c) // 10000
            vid = int(c) % 10000
            seg_mask = valid & (combo == c)
            seg_indices = seg_mask.nonzero(as_tuple=True)[0]
            n_seg = len(seg_indices)
            if n_seg == 0:
                continue

            seg_orig = original_patches[bi, seg_indices].cpu().numpy()

            # denormalize
            seg_pred_norm = pred_tensor[bi, seg_indices]
            seg_loc = patch_loc[bi, seg_indices].unsqueeze(-1)
            seg_scale = patch_scale[bi, seg_indices].unsqueeze(-1)
            seg_pred_denorm = (seg_pred_norm * seg_scale + seg_loc).cpu().numpy()

            pred = build_pred_fn(seg_orig, seg_pred_denorm, seg_indices, n_seg)
            if pred is None:
                continue

            sig_type = sig_map.get((bi, sid, vid), -1)
            sig_name = SIGNAL_TYPE_NAMES.get(sig_type, "?")

            candidates.append(
                RowCandidate(
                    orig_patches=seg_orig,
                    pred_patches=pred,
                    signal_type=sig_type,
                    signal_name=sig_name,
                    n_valid=n_seg,
                    patch_size=patch_size,
                    batch_idx=bi,
                    sample_id=sid,
                    variate_id=vid,
                )
            )

    return candidates


# ── Masked Reconstruction ────────────────────────────────────


def _process_recon_batch(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    mask_ratio: float,
    device: torch.device | None,
    block_mask: bool = False,
    block_size_min: int = 3,
    block_size_max: int = 8,
) -> list[RowCandidate]:
    """Masked reconstruction 후보를 추출한다."""
    if device is not None:
        _batch_to_device(batch, device)

    out = model(
        batch,
        task="masked",
        mask_ratio=mask_ratio,
        block_mask=block_mask,
        block_size_min=block_size_min,
        block_size_max=block_size_max,
    )
    original_patches, patch_loc, patch_scale, n = _extract_patches_and_scales(
        model, batch, out
    )
    p = model.patch_size

    pred_mask = out["pred_mask"]

    def build_pred(seg_orig, seg_pred_denorm, seg_indices, n_seg):
        seg_masked = pred_mask[0][seg_indices].cpu().numpy()  # noqa — closure captures b
        pred = np.full_like(seg_orig, np.nan)
        pred[seg_masked] = seg_pred_denorm[seg_masked]
        return pred

    # pred_mask는 batch 차원이 있으므로 per-batch closure 필요
    patch_mask = out["patch_mask"]
    p_sid = out["patch_sample_id"]
    p_vid = out["patch_variate_id"]
    batch_size = original_patches.shape[0]

    sig_map = _build_signal_map(batch, p_sid, p_vid, patch_mask, batch_size)

    reconstructed = out["reconstructed"]
    pred_mask_full = out["pred_mask"]

    candidates: list[RowCandidate] = []
    for bi in range(batch_size):
        valid = patch_mask[bi]
        if not valid.any():
            continue
        masked = pred_mask_full[bi] & valid

        combo = p_sid[bi] * 10000 + p_vid[bi]
        unique_combos = combo[valid].unique().tolist()
        for c in unique_combos:
            sid = int(c) // 10000
            vid = int(c) % 10000
            seg_mask = valid & (combo == c)
            n_seg = seg_mask.sum().item()
            if n_seg == 0:
                continue

            seg_indices = seg_mask.nonzero(as_tuple=True)[0]
            seg_orig = original_patches[bi, seg_indices].cpu().numpy()
            seg_masked = masked[seg_indices].cpu().numpy()

            seg_recon_norm = reconstructed[bi, seg_indices]
            seg_loc = patch_loc[bi, seg_indices].unsqueeze(-1)
            seg_scale = patch_scale[bi, seg_indices].unsqueeze(-1)
            seg_recon = (seg_recon_norm * seg_scale + seg_loc).cpu().numpy()

            pred = np.full_like(seg_orig, np.nan)
            pred[seg_masked] = seg_recon[seg_masked]

            sig_type = sig_map.get((bi, sid, vid), -1)
            sig_name = SIGNAL_TYPE_NAMES.get(sig_type, "?")

            candidates.append(
                RowCandidate(
                    orig_patches=seg_orig,
                    pred_patches=pred,
                    signal_type=sig_type,
                    signal_name=sig_name,
                    n_valid=n_seg,
                    patch_size=p,
                    batch_idx=bi,
                    sample_id=sid,
                    variate_id=vid,
                )
            )

    return candidates


# ── Next-Patch Prediction ────────────────────────────────────


def _process_next_pred_batch(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    device: torch.device | None,
) -> list[RowCandidate]:
    """Block Next Prediction 후보를 추출한다.

    각 (sample_id, variate_id) 세그먼트에서 context 끝 위치를 ``n_seg - K``로 두고,
    그 위치에서 예측된 K개 future patches를 그 이후 K개 자리에 배치한다. 해당 위치에는
    ground truth가 함께 존재하므로 시각적으로 비교 가능하다.
    """
    if device is not None:
        _batch_to_device(batch, device)

    out = model(batch, task="next_pred")
    original_patches, patch_loc, patch_scale, n = _extract_patches_and_scales(
        model, batch, out
    )
    p = model.patch_size
    k = model.next_block_size

    next_pred = out["next_pred"]  # (B, N, K, P)
    patch_mask = out["patch_mask"]
    p_sid = out["patch_sample_id"]
    p_vid = out["patch_variate_id"]
    batch_size = original_patches.shape[0]

    sig_map = _build_signal_map(batch, p_sid, p_vid, patch_mask, batch_size)
    candidates: list[RowCandidate] = []
    for bi in range(batch_size):
        valid = patch_mask[bi]
        if not valid.any():
            continue

        combo = p_sid[bi] * 10000 + p_vid[bi]
        unique_combos = combo[valid].unique().tolist()
        for c in unique_combos:
            sid = int(c) // 10000
            vid = int(c) % 10000
            seg_mask = valid & (combo == c)
            seg_indices = seg_mask.nonzero(as_tuple=True)[0]
            n_seg = int(len(seg_indices))
            # 최소 context 1 + 예측 K 필요.
            if n_seg <= k:
                continue

            seg_orig = original_patches[bi, seg_indices].cpu().numpy()  # (n_seg, P)

            # context 끝 패치 = n_seg - K - 1 (그 다음 K개 자리에 GT가 존재)
            ctx_end_local = n_seg - k - 1
            ctx_end_global = int(seg_indices[ctx_end_local].item())

            seg_loc = patch_loc[bi, seg_indices]  # (n_seg,)
            seg_scale = patch_scale[bi, seg_indices]  # (n_seg,)

            # ctx_end 위치의 K-block 예측을 ctx_end의 loc/scale로 denormalize
            block_norm = next_pred[bi, ctx_end_global]  # (K, P)
            ctx_loc = seg_loc[ctx_end_local]  # scalar
            ctx_scale = seg_scale[ctx_end_local]  # scalar
            block_denorm = (block_norm * ctx_scale + ctx_loc).cpu().numpy()  # (K, P)

            pred = np.full((n_seg, p), np.nan)
            # 예측 블록: ctx_end_local + 1 .. ctx_end_local + K
            pred_start = ctx_end_local + 1
            pred_end = pred_start + k
            pred[pred_start:pred_end] = block_denorm

            sig_type = sig_map.get((bi, sid, vid), -1)
            sig_name = SIGNAL_TYPE_NAMES.get(sig_type, "?")

            candidates.append(
                RowCandidate(
                    orig_patches=seg_orig,
                    pred_patches=pred,
                    signal_type=sig_type,
                    signal_name=sig_name,
                    n_valid=n_seg,
                    patch_size=p,
                    batch_idx=bi,
                    sample_id=sid,
                    variate_id=vid,
                )
            )

    return candidates


# ── Public API ────────────────────────────────────────────────


@torch.no_grad()
def save_reconstruction_figure(
    model: BiosignalFoundationModel,
    batch: PackedBatch | list[PackedBatch],
    epoch: int,
    output_dir: str | Path,
    mask_ratio: float = 0.15,
    max_rows: int = 4,
    samples_per_type: int = 3,
    max_duration_s: float = 60.0,
    sampling_rate: float = 100.0,
    device: torch.device | None = None,
    block_mask: bool = False,
    block_size_min: int = 3,
    block_size_max: int = 8,
) -> Path:
    """마스킹된 패치의 원본 vs 복원 비교 figure를 저장한다.

    Parameters
    ----------
    samples_per_type:
        신호 타입당 표시할 샘플 수. 행=신호 타입, 열=샘플.
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = batch if isinstance(batch, list) else [batch]

    all_candidates: list[RowCandidate] = []
    for b in batches:
        all_candidates.extend(
            _process_recon_batch(
                model,
                b,
                mask_ratio,
                device,
                block_mask=block_mask,
                block_size_min=block_size_min,
                block_size_max=block_size_max,
            )
        )

    grid = _select_diverse_grid(all_candidates, samples_per_type=samples_per_type)

    if not grid:
        model.train()
        return output_dir / f"recon_epoch{epoch:03d}.png"

    path = _plot_figure_grid(
        grid,
        epoch,
        output_dir,
        max_duration_s,
        sampling_rate,
        mode="masked",
    )

    model.train()
    return path


@torch.no_grad()
def save_next_pred_figure(
    model: BiosignalFoundationModel,
    batch: PackedBatch | list[PackedBatch],
    epoch: int,
    output_dir: str | Path,
    max_rows: int = 4,
    samples_per_type: int = 3,
    max_duration_s: float = 60.0,
    sampling_rate: float = 100.0,
    device: torch.device | None = None,
) -> Path:
    """Block Next Prediction 시각화 figure를 저장한다.

    각 세그먼트에서 context 마지막 위치의 K-block 예측을 그 이후 K개 자리에 그려,
    context(파란색) + 예측 블록(빨간색) + 점선 GT overlay로 비교한다.

    Parameters
    ----------
    samples_per_type:
        신호 타입당 표시할 샘플 수. 행=신호 타입, 열=샘플.
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = batch if isinstance(batch, list) else [batch]

    all_candidates: list[RowCandidate] = []
    for b in batches:
        all_candidates.extend(_process_next_pred_batch(model, b, device))

    grid = _select_diverse_grid(all_candidates, samples_per_type=samples_per_type)

    if not grid:
        model.train()
        return output_dir / f"next_pred_epoch{epoch:03d}.png"

    path = _plot_figure_grid(
        grid,
        epoch,
        output_dir,
        max_duration_s,
        sampling_rate,
        mode="next_pred",
        next_block_size=model.next_block_size,
    )
    model.train()
    return path
