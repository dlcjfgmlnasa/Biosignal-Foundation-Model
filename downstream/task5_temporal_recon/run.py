# -*- coding:utf-8 -*-
"""Task 5: 시간 구간 복원 (Temporal Reconstruction).

같은 채널 내 연속 패치를 마스킹하고 앞뒤 문맥으로 복원하여
foundation model의 temporal representation 능력을 평가한다.

사용법:
    # 더미 테스트 (checkpoint 없이 파이프라인 검증)
    python -m downstream.task5_temporal_recon.run --dummy --n-cases 2

    # 실제 평가
    python -m downstream.task5_temporal_recon.run \
        --checkpoint checkpoints/best.pt --n-cases 20 --mask-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import get_global_spatial_id
from data.parser.vitaldb import SIGNAL_TYPES

from downstream.common.data_utils import (
    CaseData,
    Window,
    extract_windows,
    apply_pipeline,
    load_pilot_cases,
    split_by_subject,
)
from downstream.common.eval_utils import compute_mse, compute_mae, compute_pearson_r


# ── 설정 ──────────────────────────────────────────────────────

SIGNAL_TYPE_NAMES = {0: "ECG", 1: "ABP", 2: "EEG", 3: "PPG", 4: "CVP", 5: "CO2", 6: "AWP"}
DEFAULT_PATCH_SIZE = 100
DEFAULT_MAX_LENGTH = 6000  # 60s at 100Hz
DEFAULT_SR = 100.0


# ── 배치 생성 ─────────────────────────────────────────────────


def _windows_to_samples(
    windows: list[Window],
) -> list[BiosignalSample]:
    """Window 리스트를 BiosignalSample 리스트로 변환."""
    samples = []
    for i, w in enumerate(windows):
        stype_int = SIGNAL_TYPES.get(w.signal_type, 0)
        spatial_id = get_global_spatial_id(w.signal_type, 0)
        samples.append(BiosignalSample(
            values=torch.from_numpy(w.signal).float(),
            length=len(w.signal),
            channel_idx=0,
            recording_idx=i,
            sampling_rate=DEFAULT_SR,
            n_channels=1,
            win_start=w.win_start,
            signal_type=stype_int,
            session_id=f"case_{w.case_id}",
            spatial_id=spatial_id,
        ))
    return samples


def _create_batch(
    samples: list[BiosignalSample],
    patch_size: int = DEFAULT_PATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> PackedBatch:
    """BiosignalSample 리스트를 PackedBatch로 변환."""
    collate = PackCollate(
        max_length=max_length,
        collate_mode="ci",
        patch_size=patch_size,
    )
    return collate(samples)


# ── 마스킹 ────────────────────────────────────────────────────


def create_contiguous_mask(
    n_patches: int,
    mask_ratio: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """연속 패치 마스킹 (중앙 부근).

    Parameters
    ----------
    n_patches: 전체 패치 수.
    mask_ratio: 마스킹할 비율 (0~1).
    rng: 랜덤 생성기.

    Returns
    -------
    (n_patches,) bool 배열 -- True = 마스킹된 패치.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_mask = max(1, int(n_patches * mask_ratio))
    # 시작점을 랜덤으로 선택 (양끝 패딩 최소 2패치 확보)
    margin = min(2, n_patches // 4)
    max_start = n_patches - n_mask - margin
    min_start = margin
    if max_start <= min_start:
        start = max(0, (n_patches - n_mask) // 2)
    else:
        start = rng.integers(min_start, max_start + 1)

    mask = np.zeros(n_patches, dtype=bool)
    mask[start:start + n_mask] = True
    return mask


def apply_mask_to_batch(
    batch: PackedBatch,
    mask_ratio: float,
    patch_size: int = DEFAULT_PATCH_SIZE,
    seed: int = 42,
) -> tuple[PackedBatch, torch.Tensor]:
    """PackedBatch의 각 row에 연속 마스킹을 적용한다.

    마스킹된 패치의 값은 0으로 교체된다 (모델의 mask token이 처리).

    Returns
    -------
    (masked_batch, patch_mask_bool)
    - masked_batch: 마스킹 적용된 배치.
    - patch_mask_bool: (B, N) -- True = 마스킹된 패치 (복원 대상).
    """
    rng = np.random.default_rng(seed)
    B, L = batch.values.shape
    N = L // patch_size

    all_masks = torch.zeros(B, N, dtype=torch.bool)

    # 원본 values 복사 (마스킹 전)
    masked_values = batch.values.clone()

    for b in range(B):
        # 유효 길이 계산 (padded_lengths 사용)
        if hasattr(batch, "padded_lengths") and batch.padded_lengths is not None:
            valid_patches = int(batch.padded_lengths[b].item()) // patch_size
        else:
            valid_patches = N

        if valid_patches < 3:
            continue

        mask = create_contiguous_mask(valid_patches, mask_ratio, rng)
        all_masks[b, :valid_patches] = torch.from_numpy(mask)

        # 마스킹된 패치의 값을 0으로 교체
        for p in range(valid_patches):
            if mask[p]:
                start = p * patch_size
                end = start + patch_size
                masked_values[b, start:end] = 0.0

    batch.values = masked_values
    return batch, all_masks


# ── 더미 모델 ─────────────────────────────────────────────────


class DummyModel:
    """Checkpoint 없이 파이프라인을 테스트하기 위한 더미 모델.

    forward_masked()는 0 텐서를 반환한다.
    """

    def __init__(self, patch_size: int = DEFAULT_PATCH_SIZE, d_model: int = 128):
        self.patch_size = patch_size
        self.d_model = d_model
        self.device = torch.device("cpu")

    def forward_masked(self, batch: PackedBatch) -> dict[str, torch.Tensor]:
        B, L = batch.values.shape
        N = L // self.patch_size
        return {
            "reconstructed": torch.randn(B, N, self.patch_size),
            "encoded": torch.randn(B, N, self.d_model),
            "patch_mask": torch.ones(B, N, dtype=torch.bool),
            "loc": torch.zeros(B, 1, 1),
            "scale": torch.ones(B, 1, 1),
        }


# ── 평가 ──────────────────────────────────────────────────────


@dataclass
class ReconResult:
    """단일 샘플의 복원 결과."""
    signal_type: str
    case_id: int
    n_masked_patches: int
    mse: float
    mae: float
    pearson_r: float
    original: np.ndarray     # 마스킹 구간 원본
    reconstructed: np.ndarray  # 마스킹 구간 복원


def evaluate_temporal_recon(
    model,
    windows: list[Window],
    mask_ratio: float = 0.3,
    patch_size: int = DEFAULT_PATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 16,
    seed: int = 42,
) -> list[ReconResult]:
    """Temporal reconstruction 평가를 실행한다.

    Parameters
    ----------
    model:
        DownstreamModelWrapper 또는 DummyModel.
    windows:
        품질 검사 통과한 Window 리스트.
    mask_ratio:
        마스킹 비율 (0~1).
    patch_size:
        패치 크기.
    max_length:
        배치 최대 길이.
    batch_size:
        배치 크기.
    seed:
        랜덤 시드.

    Returns
    -------
    list[ReconResult]
    """
    results: list[ReconResult] = []

    for i in range(0, len(windows), batch_size):
        batch_windows = windows[i:i + batch_size]

        # 원본 보존 (마스킹 전)
        original_signals = [w.signal.copy() for w in batch_windows]
        signal_types = [w.signal_type for w in batch_windows]
        case_ids = [w.case_id for w in batch_windows]

        # 배치 생성
        samples = _windows_to_samples(batch_windows)
        batch = _create_batch(samples, patch_size=patch_size, max_length=max_length)

        # 원본 패치 추출 (마스킹 전)
        B, L = batch.values.shape
        N = L // patch_size
        orig_patches = batch.values.clone().reshape(B, N, patch_size)

        # 마스킹 적용
        batch, mask_bool = apply_mask_to_batch(
            batch, mask_ratio=mask_ratio, patch_size=patch_size, seed=seed + i,
        )

        # Forward
        with torch.no_grad():
            out = model.forward_masked(batch)

        reconstructed = out["reconstructed"]  # (B, N, patch_size)

        # 각 row에서 마스킹된 패치의 복원 결과 추출
        for b in range(min(B, len(batch_windows))):
            mask_b = mask_bool[b]  # (N,)
            if not mask_b.any():
                continue

            orig_masked = orig_patches[b][mask_b].numpy().flatten()
            recon_masked = reconstructed[b][mask_b].detach().numpy().flatten()

            if len(orig_masked) == 0:
                continue

            mse = compute_mse(orig_masked, recon_masked)
            mae = compute_mae(orig_masked, recon_masked)
            pr = compute_pearson_r(orig_masked, recon_masked)

            results.append(ReconResult(
                signal_type=signal_types[b] if b < len(signal_types) else "unknown",
                case_id=case_ids[b] if b < len(case_ids) else -1,
                n_masked_patches=int(mask_b.sum()),
                mse=mse,
                mae=mae,
                pearson_r=pr,
                original=orig_masked,
                reconstructed=recon_masked,
            ))

    return results


# ── 시각화 ────────────────────────────────────────────────────


def plot_reconstruction(
    results: list[ReconResult],
    out_path: Path,
    n_examples: int = 6,
) -> None:
    """복원 결과를 시각화한다.

    Parameters
    ----------
    results:
        ReconResult 리스트.
    out_path:
        출력 PNG 경로.
    n_examples:
        시각화할 예시 수.
    """
    examples = results[:n_examples]
    if not examples:
        return

    n_cols = 2
    n_rows = (len(examples) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 3.5 * n_rows), squeeze=False)

    for i, res in enumerate(examples):
        row, col = divmod(i, n_cols)
        ax = axes[row][col]

        t = np.arange(len(res.original))
        ax.plot(t, res.original, color="steelblue", linewidth=0.8, label="Original")
        ax.plot(t, res.reconstructed, color="orangered", linewidth=0.8, alpha=0.8, label="Reconstructed")

        stype_name = SIGNAL_TYPE_NAMES.get(SIGNAL_TYPES.get(res.signal_type, -1), res.signal_type.upper())
        ax.set_title(
            f"{stype_name} case#{res.case_id}  |  "
            f"MSE={res.mse:.4f}  MAE={res.mae:.4f}  r={res.pearson_r:.3f}",
            fontsize=9,
        )
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Sample", fontsize=8)

    for i in range(len(examples), n_rows * n_cols):
        row, col = divmod(i, n_cols)
        axes[row][col].axis("off")

    fig.suptitle("Task 5: Temporal Reconstruction", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── メイン ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 5: Temporal Reconstruction")
    parser.add_argument("--checkpoint", type=str, default=None, help="사전학습 checkpoint 경로")
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--n-cases", type=int, default=10, help="로드할 pilot 케이스 수")
    parser.add_argument("--mask-ratio", type=float, default=0.3, help="마스킹 비율 (0~1)")
    parser.add_argument("--signal-types", type=str, nargs="+", default=["ecg", "abp", "ppg"],
                        help="평가할 signal type")
    parser.add_argument("--window-sec", type=float, default=60.0, help="윈도우 길이 (초)")
    parser.add_argument("--stride-sec", type=float, default=30.0, help="슬라이드 보폭 (초)")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-dir", type=str, default=".", help="결과 저장 디렉토리")
    parser.add_argument("--dummy", action="store_true", help="더미 모델로 파이프라인 검증")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 모델 로드 ──
    if args.dummy:
        print("Using dummy model (pipeline verification mode)")
        model = DummyModel(patch_size=args.patch_size)
    elif args.checkpoint:
        from downstream.common.model_wrapper import DownstreamModelWrapper
        print(f"Loading checkpoint: {args.checkpoint}")
        model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    # ── 데이터 로드 ──
    print(f"\nLoading {args.n_cases} pilot cases...")
    cases = load_pilot_cases(
        n_cases=args.n_cases,
        offset_from_end=100,
        signal_types=args.signal_types,
    )

    if not cases:
        print("No cases loaded.", file=sys.stderr)
        sys.exit(1)

    # ── 윈도우 추출 + 품질 필터 ──
    all_windows: list[Window] = []
    for stype in args.signal_types:
        for case in cases:
            wins = extract_windows(
                case, stype,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
            )
            good = apply_pipeline(wins)
            all_windows.extend(good)

    print(f"Total quality-passed windows: {len(all_windows)}")
    if not all_windows:
        print("No quality-passed windows.", file=sys.stderr)
        sys.exit(1)

    # ── 마스킹 비율별 평가 ──
    mask_ratios = [args.mask_ratio]
    if not args.dummy:
        mask_ratios = [0.1, 0.2, 0.3, 0.5]

    all_summaries: list[dict] = []

    for mr in mask_ratios:
        print(f"\n--- Mask ratio: {mr:.0%} ---")
        results = evaluate_temporal_recon(
            model, all_windows,
            mask_ratio=mr,
            patch_size=args.patch_size,
            max_length=int(args.window_sec * DEFAULT_SR),
            batch_size=args.batch_size,
        )

        if not results:
            print("  No results.")
            continue

        # Signal type별 집계
        by_stype: dict[str, list[ReconResult]] = {}
        for r in results:
            by_stype.setdefault(r.signal_type, []).append(r)

        summary = {"mask_ratio": mr, "total": len(results), "by_signal_type": {}}
        for stype, stype_results in sorted(by_stype.items()):
            mses = [r.mse for r in stype_results]
            maes = [r.mae for r in stype_results]
            prs = [r.pearson_r for r in stype_results]

            stats = {
                "n": len(stype_results),
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses)),
                "mae_mean": float(np.mean(maes)),
                "pearson_r_mean": float(np.mean(prs)),
                "pearson_r_std": float(np.std(prs)),
            }
            summary["by_signal_type"][stype] = stats

            stype_name = stype.upper()
            print(
                f"  {stype_name}: n={stats['n']}  "
                f"MSE={stats['mse_mean']:.4f}+/-{stats['mse_std']:.4f}  "
                f"MAE={stats['mae_mean']:.4f}  "
                f"r={stats['pearson_r_mean']:.3f}+/-{stats['pearson_r_std']:.3f}"
            )

        all_summaries.append(summary)

        # 시각화 (첫 번째 mask_ratio만)
        if mr == mask_ratios[0]:
            plot_path = out_dir / "task5_temporal_recon.png"
            plot_reconstruction(results, plot_path, n_examples=6)
            print(f"\n  Visualization saved: {plot_path}")

    # ── 결과 저장 ──
    results_path = out_dir / "task5_results.json"
    with open(results_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
