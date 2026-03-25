# -*- coding:utf-8 -*-
"""학습 중 Masked Reconstruction 시각화.

에폭 종료 후 모델이 마스킹된 패치를 얼마나 잘 복원하는지
원본 vs 복원 파형을 비교하는 figure를 저장한다.

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

from data.collate import PackedBatch
from loss.masked_mse_loss import create_patch_mask
from model import BiosignalFoundationModel


SIGNAL_TYPE_NAMES = {0: "ECG", 1: "ABP", 2: "EEG", 3: "PPG", 4: "CVP", 5: "CO2", 6: "AWP"}


# ── 내부 데이터 구조 ──────────────────────────────────────────────


@dataclass
class _RowCandidate:
    """시각화 후보 row 데이터."""
    orig_patches: np.ndarray     # (N_valid, P)
    recon_patches: np.ndarray    # (N_valid, P)
    patch_masked: np.ndarray     # (N_valid,) bool
    signal_type: int
    signal_name: str
    n_valid: int
    patch_size: int


# ── 배치 처리 ─────────────────────────────────────────────────────


def _process_batch(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    mask_ratio: float,
    device: torch.device | None,
) -> list[_RowCandidate]:
    """단일 배치를 forward하여 row별 시각화 데이터를 추출한다."""
    if device is not None:
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

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

    # ── row별 signal type 매핑 ──
    has_signal_types = hasattr(batch, "signal_types") and batch.signal_types is not None
    row_signal_types: dict[int, int] = {}
    if has_signal_types:
        per_row_max_var = p_vid.max(dim=-1).values  # (B,)
        var_offsets = torch.zeros(B, dtype=torch.long, device=p_vid.device)
        if B > 1:
            var_offsets[1:] = per_row_max_var[:-1].cumsum(dim=0)
        for b in range(B):
            gvi = (var_offsets[b] + (p_vid[b, 0] - 1)).clamp(min=0).item()
            if gvi < len(batch.signal_types):
                row_signal_types[b] = batch.signal_types[gvi].item()

    candidates: list[_RowCandidate] = []
    for b in range(B):
        valid = patch_mask[b]
        if not valid.any():
            continue

        n_valid = valid.sum().item()
        masked = pred_mask[b] & valid

        sig_type = row_signal_types.get(b, -1)
        sig_name = SIGNAL_TYPE_NAMES.get(sig_type, "?")

        candidates.append(_RowCandidate(
            orig_patches=original_patches[b, :n_valid].cpu().numpy(),
            recon_patches=reconstructed[b, :n_valid].cpu().numpy(),
            patch_masked=masked[:n_valid].cpu().numpy(),
            signal_type=sig_type,
            signal_name=sig_name,
            n_valid=n_valid,
            patch_size=P,
        ))

    return candidates


# ── 다양성 선택 ───────────────────────────────────────────────────


def _select_diverse(candidates: list[_RowCandidate], max_rows: int) -> list[_RowCandidate]:
    """신호 타입이 고루 분포하도록 후보를 선택한다.

    각 신호 타입에서 라운드-로빈으로 1개씩 선택한다.
    """
    by_type: dict[int, list[_RowCandidate]] = {}
    for c in candidates:
        by_type.setdefault(c.signal_type, []).append(c)

    selected: list[_RowCandidate] = []
    types = sorted(by_type.keys())

    while len(selected) < max_rows:
        added = False
        for t in types:
            if len(selected) >= max_rows:
                break
            if by_type[t]:
                selected.append(by_type[t].pop(0))
                added = True
        if not added:
            break

    return selected


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

    # 입력 정규화: 단일 배치 → 리스트
    batches = batch if isinstance(batch, list) else [batch]

    # 모든 배치에서 후보 수집
    all_candidates: list[_RowCandidate] = []
    for b in batches:
        all_candidates.extend(_process_batch(model, b, mask_ratio, device))

    # 신호 타입 다양성 기반 선택
    selected = _select_diverse(all_candidates, max_rows)

    n_rows = len(selected)
    if n_rows == 0:
        model.train()
        return output_dir / f"recon_epoch{epoch:03d}.png"

    # 최대 패치 수 (1분 = max_duration_s 초)
    P = model.patch_size
    max_patches = max(1, int(max_duration_s * sampling_rate / P))

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3 * n_rows), squeeze=False)

    for i, cand in enumerate(selected):
        ax = axes[i, 0]

        # 최대 duration으로 crop
        n_show = min(cand.n_valid, max_patches)

        orig_wave = cand.orig_patches[:n_show].reshape(-1)
        recon_wave = cand.recon_patches[:n_show].reshape(-1)
        masked = cand.patch_masked[:n_show]

        # 시간축 (초)
        t = np.arange(len(orig_wave)) / sampling_rate

        ax.plot(t, orig_wave, color="steelblue", linewidth=0.8,
                label="Original", alpha=0.9)

        # 마스킹된 구간: 복원 파형 + 배경 하이라이트
        for patch_idx in range(n_show):
            if masked[patch_idx]:
                start = patch_idx * P
                end = start + P
                ax.axvspan(start / sampling_rate, end / sampling_rate,
                           alpha=0.15, color="red")
                ax.plot(t[start:end], recon_wave[start:end],
                        color="orangered", linewidth=1.2)

        ax.set_ylabel(cand.signal_name, fontsize=10)

        n_masked = int(masked.sum())
        duration_shown = n_show * P / sampling_rate
        ax.set_title(
            f"Masked: {n_masked}/{n_show} patches "
            f"(ratio={n_masked / max(n_show, 1):.0%})  |  "
            f"{duration_shown:.0f}s shown",
            fontsize=9, loc="right",
        )
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=8)

    # 범례 (첫 번째 subplot에만)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="steelblue", linewidth=1, label="Original"),
        Line2D([0], [0], color="orangered", linewidth=1.2, label="Reconstructed"),
        Patch(facecolor="red", alpha=0.15, label="Masked region"),
    ]
    axes[0, 0].legend(handles=legend_elements, loc="upper right", fontsize=8)

    fig.suptitle(f"Masked Reconstruction — Epoch {epoch}", fontsize=13, y=1.01)
    fig.tight_layout()

    path = output_dir / f"recon_epoch{epoch:03d}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    model.train()
    return path