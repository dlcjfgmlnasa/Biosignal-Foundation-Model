# -*- coding:utf-8 -*-
"""학습 중 Masked Reconstruction & Next-Patch Prediction 시각화.

에폭 종료 후 모델의 복원/예측 능력을 원본 vs 예측 파형으로 비교한다.

- 최대 ``max_duration_s`` 초(기본 60 s) 단위로 잘라서 표시
- 여러 배치에서 신호 타입별로 다양하게 선택
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from data.collate import PackedBatch
from loss.masked_mse_loss import create_patch_mask
from model import BiosignalFoundationModel


SIGNAL_TYPE_NAMES = {0: "ECG", 1: "ABP", 2: "EEG", 3: "PPG", 4: "CVP", 5: "CO2", 6: "AWP"}


# ── 내부 데이터 구조 ──────────────────────────────────────────────


@dataclass
class _RowCandidate:
    """시각화 후보 row 데이터."""
    orig_patches: np.ndarray     # (N_valid, P)
    pred_patches: np.ndarray     # (N_valid, P)
    signal_type: int
    signal_name: str
    n_valid: int
    patch_size: int


# ── 공통 유틸 ─────────────────────────────────────────────────────


def _batch_to_device(batch: PackedBatch, device: torch.device) -> None:
    """배치의 주요 텐서를 device로 이동한다 (in-place)."""
    batch.values = batch.values.to(device)
    batch.sample_id = batch.sample_id.to(device)
    batch.variate_id = batch.variate_id.to(device)


def _build_variate_signal_map(
    batch: PackedBatch,
    p_vid: torch.Tensor,  # (B, N)
    patch_mask: torch.Tensor,  # (B, N)
    B: int,
) -> dict[tuple[int, int], int]:
    """(row, variate_id) → signal_type 매핑을 반환한다.

    ``batch.signal_types``는 ``(total_variates,)``로, 배치 전체의 variate를
    순서대로 나열한 것이다. 각 row의 variate_id를 global index로 변환하여
    올바른 signal type을 찾는다.
    """
    has_signal_types = hasattr(batch, "signal_types") and batch.signal_types is not None
    mapping: dict[tuple[int, int], int] = {}
    if not has_signal_types:
        return mapping

    # row별 max variate_id → global offset 계산
    per_row_max_var = patch_mask.long() * p_vid  # padding 제외
    per_row_n_var = per_row_max_var.max(dim=-1).values  # (B,)
    var_offsets = torch.zeros(B, dtype=torch.long, device=p_vid.device)
    if B > 1:
        var_offsets[1:] = per_row_n_var[:-1].cumsum(dim=0)

    for b in range(B):
        valid = patch_mask[b]
        if not valid.any():
            continue
        # row 내 고유 variate_id 추출
        unique_vids = p_vid[b, valid].unique().tolist()
        for vid in unique_vids:
            gvi = int(var_offsets[b].item()) + int(vid) - 1
            if 0 <= gvi < len(batch.signal_types):
                mapping[(b, int(vid))] = batch.signal_types[gvi].item()

    return mapping


def _select_diverse(candidates: list[_RowCandidate], max_rows: int) -> list[_RowCandidate]:
    """신호 타입이 고루 분포하도록 후보를 선택한다.

    각 신호 타입에서 1개씩 선택한다. 타입 수가 max_rows보다 적으면
    타입 수만큼만 선택하여 중복을 방지한다.
    """
    by_type: dict[int, list[_RowCandidate]] = {}
    for c in candidates:
        by_type.setdefault(c.signal_type, []).append(c)

    n_types = len(by_type)
    effective_rows = min(max_rows, n_types)

    selected: list[_RowCandidate] = []
    types = sorted(by_type.keys())

    for t in types:
        if len(selected) >= effective_rows:
            break
        if by_type[t]:
            selected.append(by_type[t].pop(0))

    return selected


# ── Masked Reconstruction 배치 처리 ──────────────────────────────


def _process_batch_masked(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    mask_ratio: float,
    device: torch.device | None,
) -> list[_RowCandidate]:
    """Masked reconstruction 후보를 variate 단위로 추출한다.

    한 row에 여러 signal type의 variate가 packed될 수 있으므로,
    variate_id별로 분리하여 각각 별도의 후보로 만든다.
    """
    if device is not None:
        _batch_to_device(batch, device)

    out = model(batch, task="masked")
    reconstructed = out["reconstructed"]  # (B, N, P)
    patch_mask = out["patch_mask"]        # (B, N)
    p_vid = out["patch_variate_id"]       # (B, N)

    pred_mask = create_patch_mask(patch_mask, mask_ratio=mask_ratio)

    P = model.patch_size
    normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
    B, L = normalized.shape
    N = L // P
    original_patches = normalized[:, :N * P].reshape(B, N, P)

    sig_map = _build_variate_signal_map(batch, p_vid, patch_mask, B)

    candidates: list[_RowCandidate] = []
    for b in range(B):
        valid = patch_mask[b]
        if not valid.any():
            continue

        masked = pred_mask[b] & valid

        # row 내 고유 variate_id별로 분리
        unique_vids = p_vid[b, valid].unique().tolist()
        for vid in unique_vids:
            vid = int(vid)
            var_mask = valid & (p_vid[b] == vid)  # 이 variate에 속하는 valid 패치
            n_var = var_mask.sum().item()
            if n_var == 0:
                continue

            var_indices = var_mask.nonzero(as_tuple=True)[0]
            var_orig = original_patches[b, var_indices].cpu().numpy()  # (n_var, P)
            var_masked = masked[var_indices].cpu().numpy()
            var_recon = reconstructed[b, var_indices].cpu().numpy()

            pred = np.full_like(var_orig, np.nan)
            pred[var_masked] = var_recon[var_masked]

            sig_type = sig_map.get((b, vid), -1)
            sig_name = SIGNAL_TYPE_NAMES.get(sig_type, "?")

            candidates.append(_RowCandidate(
                orig_patches=var_orig,
                pred_patches=pred,
                signal_type=sig_type,
                signal_name=sig_name,
                n_valid=n_var,
                patch_size=P,
            ))

    return candidates


# ── Next-Patch Prediction 배치 처리 ──────────────────────────────


def _process_batch_next_pred(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    horizon: int,
    device: torch.device | None,
) -> list[_RowCandidate]:
    """Next-patch prediction 후보를 variate 단위로 추출한다.

    next_pred[i]가 original[i+H]를 예측하므로,
    같은 variate 내에서 H만큼 shift하여 대응시킨다.
    """
    if device is not None:
        _batch_to_device(batch, device)

    out = model(batch, task="next_pred", horizon=horizon)
    next_pred = out["next_pred"]      # (B, N, P)
    patch_mask = out["patch_mask"]    # (B, N)
    p_vid = out["patch_variate_id"]   # (B, N)

    P = model.patch_size
    normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
    B, L = normalized.shape
    N = L // P
    original_patches = normalized[:, :N * P].reshape(B, N, P)

    sig_map = _build_variate_signal_map(batch, p_vid, patch_mask, B)

    candidates: list[_RowCandidate] = []
    for b in range(B):
        valid = patch_mask[b]
        if not valid.any():
            continue

        unique_vids = p_vid[b, valid].unique().tolist()
        for vid in unique_vids:
            vid = int(vid)
            var_mask = valid & (p_vid[b] == vid)
            var_indices = var_mask.nonzero(as_tuple=True)[0]
            n_var = len(var_indices)
            if n_var <= horizon:
                continue

            var_orig = original_patches[b, var_indices].cpu().numpy()  # (n_var, P)
            # next_pred[i] → original[i+H]: variate 내 인덱스 기준으로 shift
            var_next = next_pred[b, var_indices].cpu().numpy()
            pred = np.full((n_var, P), np.nan)
            pred[horizon:n_var] = var_next[:n_var - horizon]

            sig_type = sig_map.get((b, vid), -1)
            sig_name = SIGNAL_TYPE_NAMES.get(sig_type, "?")

            candidates.append(_RowCandidate(
                orig_patches=var_orig,
                pred_patches=pred,
                signal_type=sig_type,
                signal_name=sig_name,
                n_valid=n_var,
                patch_size=P,
            ))

    return candidates


# ── 공통 플로팅 ───────────────────────────────────────────────────


def _plot_figure(
    selected: list[_RowCandidate],
    epoch: int,
    output_dir: Path,
    max_duration_s: float,
    sampling_rate: float,
    mode: str,  # "masked" or "next_pred"
    horizon: int = 1,
    next_pred_ratio: float = 0.3,
) -> Path:
    """선택된 후보들로 figure를 그려 저장한다.

    Parameters
    ----------
    next_pred_ratio:
        next_pred 모드에서 crop된 패치 중 표시할 비율 (기본 30%).
    """
    P = selected[0].patch_size
    max_patches = max(1, int(max_duration_s * sampling_rate / P))
    n_rows = len(selected)

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3 * n_rows), squeeze=False)

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

    for i, cand in enumerate(selected):
        ax = axes[i, 0]

        n_show = min(cand.n_valid, max_patches)
        orig_wave = cand.orig_patches[:n_show].reshape(-1)
        pred_patches_cropped = cand.pred_patches[:n_show]  # (n_show, P)

        # next_pred: crop 후 범위에서 균등 샘플링
        if mode == "next_pred":
            # crop 범위 내 예측 가능한 위치 (NaN이 아닌 위치)
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
            # 선택되지 않은 위치는 NaN으로
            for p_idx in valid_pred_indices:
                if p_idx not in keep:
                    pred_patches_cropped[p_idx] = np.nan

        pred_wave = pred_patches_cropped.reshape(-1)
        t = np.arange(len(orig_wave)) / sampling_rate

        ax.plot(t, orig_wave, color="steelblue", linewidth=0.8,
                label="Original", alpha=0.9)

        # 예측이 있는 패치만 하이라이트 + 오버레이
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

        ax.set_ylabel(cand.signal_name, fontsize=10)

        duration_shown = n_show * P / sampling_rate
        ax.set_title(
            f"Predicted: {n_pred}/{n_show} patches  |  "
            f"{duration_shown:.0f}s shown",
            fontsize=9, loc="right",
        )
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=8)

    # 범례
    legend_elements = [
        Line2D([0], [0], color="steelblue", linewidth=1, label="Original"),
        Line2D([0], [0], color=pred_color, linewidth=1.2, label=pred_label),
        Patch(facecolor=highlight_color, alpha=0.12, label=highlight_label),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="upper right", fontsize=8)

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


# ── 메인 시각화 함수 ──────────────────────────────────────────────


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
    """마스킹된 패치의 원본 vs 복원 비교 figure를 저장한다.

    Parameters
    ----------
    model : BiosignalFoundationModel
    batch : PackedBatch 또는 list[PackedBatch]
        시각화용 배치. 여러 배치를 넘기면 신호 타입 다양성이 높아진다.
    epoch : int
    output_dir : 저장 디렉토리
    mask_ratio : 마스킹 비율
    max_rows : figure에 그릴 최대 variate 수
    max_duration_s : subplot당 최대 표시 시간 (초). 기본 60초.
    sampling_rate : 시각화 시간축 계산용 샘플링 레이트 (Hz). 기본 100.
    device : 모델이 올라가 있는 디바이스

    Returns
    -------
    Path — 저장된 figure 경로
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = batch if isinstance(batch, list) else [batch]

    all_candidates: list[_RowCandidate] = []
    for b in batches:
        all_candidates.extend(_process_batch_masked(model, b, mask_ratio, device))

    selected = _select_diverse(all_candidates, max_rows)

    if not selected:
        model.train()
        return output_dir / f"recon_epoch{epoch:03d}.png"

    path = _plot_figure(
        selected, epoch, output_dir,
        max_duration_s, sampling_rate, mode="masked",
    )
    model.train()
    return path


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
    """Next-patch prediction의 원본 vs 예측 비교 figure를 저장한다.

    Parameters
    ----------
    model : BiosignalFoundationModel
    batch : PackedBatch 또는 list[PackedBatch]
    epoch : int
    output_dir : 저장 디렉토리
    horizon : 예측 거리 (패치 단위). 기본 1.
    max_rows : figure에 그릴 최대 variate 수
    max_duration_s : subplot당 최대 표시 시간 (초). 기본 60초.
    sampling_rate : 시각화 시간축 계산용 샘플링 레이트 (Hz). 기본 100.
    device : 모델이 올라가 있는 디바이스

    Returns
    -------
    Path — 저장된 figure 경로
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batches = batch if isinstance(batch, list) else [batch]

    all_candidates: list[_RowCandidate] = []
    for b in batches:
        all_candidates.extend(_process_batch_next_pred(model, b, horizon, device))

    selected = _select_diverse(all_candidates, max_rows)

    if not selected:
        model.train()
        return output_dir / f"next_pred_epoch{epoch:03d}.png"

    path = _plot_figure(
        selected, epoch, output_dir,
        max_duration_s, sampling_rate, mode="next_pred", horizon=horizon,
    )
    model.train()
    return path