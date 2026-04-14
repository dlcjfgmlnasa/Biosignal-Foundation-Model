# -*- coding:utf-8 -*-
"""Phase 2 전용 시각화 — Cross-Modal Prediction & Contrastive Embedding.

Phase 2 (Any-Variate) 학습에서 cross-modal 관계 학습 정도를 모니터링하는
시각화 함수들을 제공한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.collate import PackedBatch
from data.spatial_map import CROSS_PRED_ALLOWED_PAIRS
from model import BiosignalFoundationModel


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

SIGNAL_TYPE_COLORS = {
    0: "#e41a1c",  # ECG - red
    1: "#377eb8",  # ABP - blue
    2: "#4daf4a",  # PPG - green
    3: "#984ea3",  # CVP - purple
    4: "#ff7f00",  # CO2 - orange
    5: "#a65628",  # AWP - brown
    6: "#f781bf",  # PAP - pink
    7: "#999999",  # ICP - gray
}

MECHANISM_GROUP_COLORS = {
    0: "#1f77b4",  # Cardiovascular - blue
    1: "#ff7f0e",  # Respiratory - orange
}

MECHANISM_GROUP = {
    0: 0,
    1: 0,
    2: 0,
    3: 0,  # Cardiovascular
    4: 1,
    5: 1,  # Respiratory
    6: 0,
    7: 0,  # Cardiovascular
}


# ── 유틸 ─────────────────────────────────────────────────────────


def _batch_to_device(batch: PackedBatch, device: torch.device) -> None:
    batch.values = batch.values.to(device)
    batch.sample_id = batch.sample_id.to(device)
    batch.variate_id = batch.variate_id.to(device)


def _build_signal_map(
    batch: PackedBatch,
    p_sid: torch.Tensor,
    p_vid: torch.Tensor,
    patch_mask: torch.Tensor,
    b: int,
) -> dict[tuple[int, int, int], int]:
    """(row, sample_id, variate_id) → signal_type 매핑."""
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


# ── 1. Cross-Modal Prediction Visualization ─────────────────────


@torch.no_grad()
def save_cross_modal_figure(
    model: BiosignalFoundationModel,
    batch: PackedBatch | list[PackedBatch],
    epoch: int,
    output_dir: str | Path,
    mask_ratio: float = 0.15,
    max_pairs: int = 4,
    max_duration_s: float = 30.0,
    sampling_rate: float = 100.0,
    device: torch.device | None = None,
    block_mask: bool = False,
    block_size_min: int = 3,
    block_size_max: int = 8,
    variate_mask_prob: float = 0.3,
) -> Path:
    """Cross-modal prediction 결과를 시각화한다.

    같은 시간대(time_id)에서 서로 다른 variate 간 예측을 보여준다.
    상단: 원본 variate A + cross_pred(A→B),
    하단: 원본 variate B + cross_pred(B→A).

    Parameters
    ----------
    max_pairs:
        표시할 최대 cross-modal 쌍 수.
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = batch if isinstance(batch, list) else [batch]

    all_pairs: list[dict] = []
    seen_type_pairs: set[tuple[int, int]] = set()
    for b in batches:
        pairs = _extract_cross_modal_pairs(
            model,
            b,
            mask_ratio,
            device,
            block_mask=block_mask,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
            variate_mask_prob=variate_mask_prob,
        )
        for p in pairs:
            tp = (
                min(p["sig_type_a"], p["sig_type_b"]),
                max(p["sig_type_a"], p["sig_type_b"]),
            )
            if tp in seen_type_pairs:
                continue
            seen_type_pairs.add(tp)
            all_pairs.append(p)
        if len(all_pairs) >= max_pairs * 2:
            break

    if not all_pairs:
        model.train()
        empty = output_dir / f"cross_modal_epoch{epoch:03d}.png"
        return empty

    # whitelist 쌍만 필터 + 그룹별 분리
    allowed = {(min(a, b), max(a, b)) for a, b in CROSS_PRED_ALLOWED_PAIRS}
    cardio_pairs = []
    resp_pairs = []
    for p in all_pairs:
        tp = (min(p["sig_type_a"], p["sig_type_b"]), max(p["sig_type_a"], p["sig_type_b"]))
        if tp not in allowed:
            continue
        group_a = MECHANISM_GROUP.get(p["sig_type_a"], -1)
        group_b = MECHANISM_GROUP.get(p["sig_type_b"], -1)
        if group_a == 0 and group_b == 0:
            if len(cardio_pairs) < max_pairs:
                cardio_pairs.append(p)
        elif group_a == 1 and group_b == 1:
            if len(resp_pairs) < max_pairs:
                resp_pairs.append(p)

    path = _plot_cross_modal_grouped(
        cardio_pairs,
        resp_pairs,
        epoch,
        output_dir,
        max_duration_s,
        sampling_rate,
    )

    model.train()
    return path


def _extract_cross_modal_pairs(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    mask_ratio: float,
    device: torch.device | None,
    block_mask: bool = False,
    block_size_min: int = 3,
    block_size_max: int = 8,
    variate_mask_prob: float = 0.3,
) -> list[dict]:
    """Cross-modal prediction 쌍을 추출한다.

    Returns
    -------
    list of dict, each with keys:
        orig_a, orig_b: (N, P) 원본 패치 (denormalized)
        cross_pred_a, cross_pred_b: (N, P) cross-modal 예측 (denormalized)
        sig_type_a, sig_type_b: signal type IDs
        sig_name_a, sig_name_b: signal type names
        n_patches: 공유 패치 수
    """
    if device is not None:
        _batch_to_device(batch, device)

    out = model(
        batch,
        task="masked",
        mask_ratio=mask_ratio,
        block_mask=block_mask,
        block_size_min=block_size_min,
        block_size_max=block_size_max,
        variate_mask_prob=variate_mask_prob,
    )

    p = model.patch_size
    loc = out["loc"]
    scale = out["scale"]
    b_size, l = batch.values.shape[0], batch.values.shape[1]
    n = l // p

    patch_mask = out["patch_mask"]
    p_sid = out["patch_sample_id"]
    p_vid = out["patch_variate_id"]
    time_id = out["time_id"]
    cross_pred_per_type = out["cross_pred_per_type"]  # (B, N, T, P)

    # denorm helpers
    stride = model.patch_embed.stride
    patch_starts = torch.arange(n, device=loc.device) * stride
    patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
    patch_loc = loc[:, patch_starts, 0]
    patch_scale = scale[:, patch_starts, 0]

    original_patches = batch.values[:, : n * p].reshape(b_size, n, p)

    sig_map = _build_signal_map(batch, p_sid, p_vid, patch_mask, b_size)

    pairs: list[dict] = []
    for bi in range(b_size):
        valid = patch_mask[bi]
        if not valid.any():
            continue

        # 같은 sample_id 내에서 서로 다른 variate 쌍 찾기
        combo = p_sid[bi] * 10000 + p_vid[bi]
        unique_combos = combo[valid].unique().tolist()

        # sample_id별로 그룹핑
        sid_to_variates: dict[int, list[int]] = {}
        for c in unique_combos:
            sid = int(c) // 10000
            vid = int(c) % 10000
            sid_to_variates.setdefault(sid, []).append(vid)

        for sid, vids in sid_to_variates.items():
            if len(vids) < 2:
                continue

            # 서로 다른 signal_type을 가진 모든 쌍 찾기 (중복 signal_type 쌍 제거)
            seen_type_pairs: set[tuple[int, int]] = set()
            for i in range(len(vids)):
                st_i = sig_map.get((bi, sid, vids[i]), -1)
                for j in range(i + 1, len(vids)):
                    st_j = sig_map.get((bi, sid, vids[j]), -1)
                    if st_i == st_j:
                        continue
                    type_pair = (min(st_i, st_j), max(st_i, st_j))
                    if type_pair in seen_type_pairs:
                        continue
                    seen_type_pairs.add(type_pair)

                    vid_a, vid_b = vids[i], vids[j]
                    combo_a = sid * 10000 + vid_a
                    combo_b = sid * 10000 + vid_b

                    mask_a = valid & (combo == combo_a)
                    mask_b = valid & (combo == combo_b)

                    idx_a = mask_a.nonzero(as_tuple=True)[0]
                    idx_b = mask_b.nonzero(as_tuple=True)[0]

                    if len(idx_a) == 0 or len(idx_b) == 0:
                        continue

                    tid_a = time_id[bi, idx_a]
                    tid_b = time_id[bi, idx_b]

                    sig_type_a = sig_map.get((bi, sid, vid_a), -1)
                    sig_type_b = sig_map.get((bi, sid, vid_b), -1)

                    # target-conditioned cross_pred 선택 + target의 loc/scale로 denorm
                    # cp_a: A 위치에서 B(target)를 예측 → B의 loc/scale로 denorm
                    cp_a = cross_pred_per_type[bi, idx_a, sig_type_b] if sig_type_b >= 0 else cross_pred_per_type[bi, idx_a, 0]
                    loc_b_denorm = patch_loc[bi, idx_b].unsqueeze(-1)
                    scl_b_denorm = patch_scale[bi, idx_b].unsqueeze(-1)
                    # idx_a와 idx_b 길이가 다를 수 있으므로, 짧은 쪽에 맞춤
                    n_common = min(len(idx_a), len(idx_b))
                    cpred_a = (cp_a[:n_common] * scl_b_denorm[:n_common] + loc_b_denorm[:n_common]).detach().cpu().numpy()

                    # cp_b: B 위치에서 A(target)를 예측 → A의 loc/scale로 denorm
                    cp_b = cross_pred_per_type[bi, idx_b, sig_type_a] if sig_type_a >= 0 else cross_pred_per_type[bi, idx_b, 0]
                    loc_a_denorm = patch_loc[bi, idx_a].unsqueeze(-1)
                    scl_a_denorm = patch_scale[bi, idx_a].unsqueeze(-1)
                    cpred_b = (cp_b[:n_common] * scl_a_denorm[:n_common] + loc_a_denorm[:n_common]).detach().cpu().numpy()

                    # 원본도 n_common에 맞춤
                    orig_a = original_patches[bi, idx_a[:n_common]].cpu().numpy()
                    orig_b = original_patches[bi, idx_b[:n_common]].cpu().numpy()

                    pairs.append(
                        {
                            "orig_a": orig_a,
                            "orig_b": orig_b,
                            "cross_pred_a": cpred_a,
                            "cross_pred_b": cpred_b,
                            "sig_type_a": sig_type_a,
                            "sig_type_b": sig_type_b,
                            "sig_name_a": SIGNAL_TYPE_NAMES.get(sig_type_a, "?"),
                            "sig_name_b": SIGNAL_TYPE_NAMES.get(sig_type_b, "?"),
                            "n_patches_a": n_common,
                            "n_patches_b": n_common,
                            "time_ids_a": tid_a.cpu().numpy(),
                            "time_ids_b": tid_b.cpu().numpy(),
                        }
                    )

    return pairs


def _plot_pair_rows(
    axes,
    pairs: list[dict],
    row_offset: int,
    max_duration_s: float,
    sampling_rate: float,
) -> None:
    """Cross-modal 쌍을 axes에 그린다. 각 쌍마다 2행."""
    for pi, pair in enumerate(pairs):
        p = pair["orig_a"].shape[1]  # patch_size
        max_patches = max(1, int(max_duration_s * sampling_rate / p))

        for sub_row, (
            key_orig,
            key_cross,
            sig_name,
            sig_type,
            n_p,
            other_name,
        ) in enumerate(
            [
                (
                    "orig_a",
                    "cross_pred_a",
                    pair["sig_name_a"],
                    pair["sig_type_a"],
                    pair["n_patches_a"],
                    pair["sig_name_b"],
                ),
                (
                    "orig_b",
                    "cross_pred_b",
                    pair["sig_name_b"],
                    pair["sig_type_b"],
                    pair["n_patches_b"],
                    pair["sig_name_a"],
                ),
            ]
        ):
            row_idx = row_offset + pi * 2 + sub_row
            ax = axes[row_idx, 0]
            orig = pair[key_orig]
            cross = pair[key_cross]

            n_show = min(n_p, max_patches)
            orig_wave = orig[:n_show].reshape(-1)
            cross_wave = cross[:n_show].reshape(-1)
            t = np.arange(len(orig_wave)) / sampling_rate

            orig_color = SIGNAL_TYPE_COLORS.get(sig_type, "steelblue")
            ax.plot(
                t,
                orig_wave,
                color=orig_color,
                linewidth=0.8,
                alpha=0.7,
                label=f"Original {sig_name}",
            )
            ax.plot(
                t,
                cross_wave,
                color="darkorange",
                linewidth=0.9,
                alpha=0.8,
                label=f"Cross-Pred (from {other_name})",
            )

            mse = np.mean((orig_wave - cross_wave) ** 2)
            corr = (
                np.corrcoef(orig_wave, cross_wave)[0, 1] if len(orig_wave) > 1 else 0.0
            )

            duration = n_show * p / sampling_rate
            ax.set_ylabel(sig_name, fontsize=10, color=orig_color, fontweight="bold")
            ax.tick_params(labelsize=7)
            ax.legend(loc="upper right", fontsize=7)

            if sub_row == 0:
                ax.set_title(
                    f"{pair['sig_name_a']} \u2194 {pair['sig_name_b']}  |  "
                    f"MSE={mse:.4f}  r={corr:.3f}  |  {duration:.0f}s",
                    fontsize=10,
                    loc="left",
                    fontweight="bold",
                )
            else:
                ax.set_title(
                    f"MSE={mse:.4f}  r={corr:.3f}  |  {duration:.0f}s",
                    fontsize=9,
                    loc="left",
                )
                ax.set_xlabel("Time (s)", fontsize=9)


def _plot_cross_modal_grouped(
    cardio_pairs: list[dict],
    resp_pairs: list[dict],
    epoch: int,
    output_dir: Path,
    max_duration_s: float,
    sampling_rate: float,
) -> Path:
    """Cardiovascular / Respiratory 그룹별로 분리하여 시각화한다.

    상단: Cardiovascular 그룹 (ECG, ABP, PPG, CVP, PAP, ICP)
    하단: Respiratory 그룹 (CO2, AWP)
    각 쌍마다 2행 (variate A, variate B).
    """
    n_cardio_rows = len(cardio_pairs) * 2
    n_resp_rows = len(resp_pairs) * 2
    # 그룹 헤더 행 추가 (있는 그룹만)
    n_sections = (1 if cardio_pairs else 0) + (1 if resp_pairs else 0)
    n_rows = n_cardio_rows + n_resp_rows + n_sections

    if n_rows == 0:
        empty = output_dir / f"cross_modal_epoch{epoch:03d}.png"
        return empty

    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(14, 3 * n_rows),
        squeeze=False,
        gridspec_kw={"height_ratios": _build_height_ratios(
            n_cardio_rows, n_resp_rows, cardio_pairs, resp_pairs
        )},
    )

    row = 0

    # ── Cardiovascular section ──
    if cardio_pairs:
        ax_header = axes[row, 0]
        ax_header.set_facecolor(MECHANISM_GROUP_COLORS[0] + "15")
        ax_header.text(
            0.5, 0.5,
            "[Cardiovascular] ECG, ABP, PPG, CVP, PAP, ICP",
            transform=ax_header.transAxes,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=MECHANISM_GROUP_COLORS[0],
        )
        ax_header.set_xlim(0, 1)
        ax_header.set_ylim(0, 1)
        ax_header.axis("off")
        row += 1

        _plot_pair_rows(axes, cardio_pairs, row, max_duration_s, sampling_rate)
        row += n_cardio_rows

    # ── Respiratory section ──
    if resp_pairs:
        ax_header = axes[row, 0]
        ax_header.set_facecolor(MECHANISM_GROUP_COLORS[1] + "15")
        ax_header.text(
            0.5, 0.5,
            "[Respiratory] CO2, AWP",
            transform=ax_header.transAxes,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color=MECHANISM_GROUP_COLORS[1],
        )
        ax_header.set_xlim(0, 1)
        ax_header.set_ylim(0, 1)
        ax_header.axis("off")
        row += 1

        _plot_pair_rows(axes, resp_pairs, row, max_duration_s, sampling_rate)

    fig.suptitle(
        f"Cross-Modal Prediction — Epoch {epoch}  "
        f"[Cardio: {len(cardio_pairs)} pairs, Resp: {len(resp_pairs)} pairs]",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()

    path = output_dir / f"cross_modal_epoch{epoch:03d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_height_ratios(
    n_cardio_rows: int,
    n_resp_rows: int,
    cardio_pairs: list,
    resp_pairs: list,
) -> list[float]:
    """그룹 헤더(작은 높이)와 데이터 행(표준 높이)의 height_ratios."""
    ratios: list[float] = []
    if cardio_pairs:
        ratios.append(0.3)  # 헤더
        ratios.extend([1.0] * n_cardio_rows)
    if resp_pairs:
        ratios.append(0.3)  # 헤더
        ratios.extend([1.0] * n_resp_rows)
    return ratios
