# -*- coding:utf-8 -*-
"""학습 중 Masked Reconstruction 시각화.

에폭 종료 후 모델이 마스킹된 패치를 얼마나 잘 복원하는지
원본 vs 복원 파형을 비교하는 figure를 저장한다.
"""
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.collate import PackedBatch
from loss.masked_mse_loss import create_patch_mask
from model import BiosignalFoundationModel


SIGNAL_TYPE_NAMES = {0: "ECG", 1: "ABP", 2: "EEG", 3: "PPG", 4: "CVP", 5: "CO2", 6: "AWP"}


@torch.no_grad()
def save_reconstruction_figure(
    model: BiosignalFoundationModel,
    batch: PackedBatch,
    epoch: int,
    output_dir: str | Path,
    mask_ratio: float = 0.15,
    max_rows: int = 4,
    device: torch.device | None = None,
) -> Path:
    """마스킹된 패치의 원본 vs 복원 비교 figure를 저장한다.

    Parameters
    ----------
    model : BiosignalFoundationModel
    batch : PackedBatch — 시각화용 배치 (1개)
    epoch : int
    output_dir : 저장 디렉토리
    mask_ratio : 마스킹 비율
    max_rows : figure에 그릴 최대 variate 수
    device : 모델이 올라가 있는 디바이스

    Returns
    -------
    Path — 저장된 figure 경로
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is not None:
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

    # Forward
    out = model(batch, task="masked")
    reconstructed = out["reconstructed"]  # (B, N, P)
    patch_mask = out["patch_mask"]        # (B, N)
    p_vid = out["patch_variate_id"]       # (B, N)

    # 마스크 생성
    pred_mask = create_patch_mask(patch_mask, mask_ratio=mask_ratio)  # (B, N) bool

    # 원본 패치 (정규화된)
    P = model.patch_size
    normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
    B, L = normalized.shape
    N = L // P
    original_patches = normalized[:, :N * P].reshape(B, N, P)

    # signal type 정보
    has_signal_types = hasattr(batch, "signal_types") and batch.signal_types is not None

    # 시각화할 row 선택: 유효한 variate (p_vid > 0)인 row만
    valid_rows = []
    for b in range(B):
        if patch_mask[b].any():
            valid_rows.append(b)
        if len(valid_rows) >= max_rows:
            break

    n_rows = len(valid_rows)
    if n_rows == 0:
        model.train()
        return output_dir / f"recon_epoch{epoch:03d}.png"

    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 3 * n_rows), squeeze=False)

    for i, b in enumerate(valid_rows):
        ax = axes[i, 0]

        # 유효 패치만 추출
        valid = patch_mask[b]  # (N,)
        masked = pred_mask[b] & valid  # (N,) — 마스킹된 패치
        n_valid = valid.sum().item()

        # 시간축 재구성: 패치를 이어붙여 연속 파형으로
        orig_wave = original_patches[b, :n_valid].reshape(-1).cpu().numpy()
        recon_wave = reconstructed[b, :n_valid].reshape(-1).cpu().numpy()

        t = range(len(orig_wave))
        ax.plot(t, orig_wave, color="steelblue", linewidth=0.8, label="Original", alpha=0.9)

        # 마스킹된 구간만 복원 파형 표시
        masked_indices = masked[:n_valid].cpu().numpy()
        for patch_idx in range(n_valid):
            if masked_indices[patch_idx]:
                start = patch_idx * P
                end = start + P
                ax.axvspan(start, end, alpha=0.15, color="red")
                ax.plot(
                    range(start, end),
                    recon_wave[start:end],
                    color="orangered", linewidth=1.2,
                )

        # 라벨
        vid = p_vid[b, 0].item()
        if has_signal_types:
            # global_var_idx → signal_type
            per_row_max_var = p_vid[b].max().item()
            sig_type_idx = min(vid - 1, len(batch.signal_types) - 1) if vid > 0 else 0
            sig_type_idx = max(sig_type_idx, 0)
            sig_name = SIGNAL_TYPE_NAMES.get(batch.signal_types[sig_type_idx].item(), "?")
            ax.set_ylabel(f"Row {b}\n{sig_name}", fontsize=10)
        else:
            ax.set_ylabel(f"Row {b}", fontsize=10)

        n_masked = masked_indices.sum()
        ax.set_title(
            f"Masked: {n_masked}/{n_valid} patches "
            f"(ratio={n_masked/max(n_valid,1):.0%})",
            fontsize=9, loc="right",
        )
        ax.tick_params(labelsize=8)

    # 범례
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
