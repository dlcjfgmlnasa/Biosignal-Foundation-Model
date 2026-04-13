# -*- coding:utf-8 -*-
"""Temporal Imputation - 평가 스크립트.

같은 채널의 시간 구간을 마스킹하고, 앞뒤 context로 복원하는 능력 평가.
Zero-shot (forward(masked) → reconstructed) — 추가 학습 없음.

사용법:
    python -m downstream.generation.imputation.run --dummy
    python -m downstream.generation.imputation.run --checkpoint path/to/best.pt \
        --data-path outputs/downstream/imputation/imputation_vitaldb_ecg.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from data.parser.vitaldb import SIGNAL_TYPES

PATCH_SIZE = 100
TARGET_SR = 100.0


# ---- Dummy model ----


class DummyModel:
    def __init__(self, d_model=128, patch_size=100):
        self.d_model = d_model
        self.patch_size = patch_size

    def eval(self):
        return self

    def __call__(self, batch, task="masked", **kwargs):
        b, l = batch.values.shape
        p = self.patch_size
        n = l // p
        return {
            "reconstructed": torch.randn(b, n, p),
            "loc": torch.zeros(b, l, 1),
            "scale": torch.ones(b, l, 1),
            "patch_mask": torch.ones(b, n, dtype=torch.bool),
        }


# ---- Batch construction ----


def _signals_to_batch(
    signals: torch.Tensor, signal_type: str, patch_size: int
) -> PackedBatch:
    """(N, win_samples) tensor -> PackedBatch list."""
    stype_int = SIGNAL_TYPES.get(signal_type, 0)
    samples = []
    for i in range(signals.shape[0]):
        sig = signals[i]
        n_usable = (len(sig) // patch_size) * patch_size
        samples.append(
            BiosignalSample(
                values=sig[:n_usable].float(),
                length=n_usable,
                channel_idx=0,
                recording_idx=i,
                sampling_rate=TARGET_SR,
                n_channels=1,
                win_start=0,
                signal_type=stype_int,
                session_id=f"imp_{i}",
                spatial_id=0,
            )
        )
    collate = PackCollate(
        max_length=int(signals.shape[1]),
        collate_mode="ci",
        patch_size=patch_size,
    )
    return collate(samples)


# ---- Evaluation ----


@torch.no_grad()
def evaluate_imputation(
    model,
    signals: torch.Tensor,
    signal_type: str,
    patch_size: int,
    mask_ratio: float = 0.3,
    device: str = "cpu",
) -> dict:
    """Temporal imputation 평가: 마스킹 구간의 MSE/MAE/Pearson r."""
    from loss.masked_mse_loss import create_patch_mask

    n_samples = signals.shape[0]
    all_mse, all_mae, all_r = [], [], []

    batch_size = 32
    for start in range(0, n_samples, batch_size):
        chunk = signals[start : start + batch_size]
        batch = _signals_to_batch(chunk, signal_type, patch_size)
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

        out = model(batch, task="masked")
        reconstructed = out["reconstructed"]  # (b, n, p)
        patch_mask = out["patch_mask"]  # (b, n)

        # Original patches (normalized)
        loc = out["loc"]
        scale = out["scale"]
        b, l = batch.values.shape
        n = l // patch_size
        normalized = (
            (batch.values.unsqueeze(-1) - loc) / scale.clamp(min=1e-8)
        ).squeeze(-1)
        original_patches = normalized[:, : n * patch_size].reshape(b, n, patch_size)

        # Create mask
        pred_mask = create_patch_mask(patch_mask, mask_ratio=mask_ratio)

        for bi in range(b):
            masked = pred_mask[bi] & patch_mask[bi]
            if not masked.any():
                continue

            orig = original_patches[bi, masked].cpu().numpy().reshape(-1)
            pred = reconstructed[bi, masked].cpu().numpy().reshape(-1)

            mse = float(np.mean((orig - pred) ** 2))
            mae = float(np.mean(np.abs(orig - pred)))

            # Pearson r
            if np.std(orig) > 1e-8 and np.std(pred) > 1e-8:
                r = float(np.corrcoef(orig, pred)[0, 1])
            else:
                r = 0.0

            all_mse.append(mse)
            all_mae.append(mae)
            all_r.append(r)

    return {
        "mse": float(np.mean(all_mse)) if all_mse else 0.0,
        "mae": float(np.mean(all_mae)) if all_mae else 0.0,
        "pearson_r": float(np.mean(all_r)) if all_r else 0.0,
        "n_windows": len(all_mse),
    }


# ---- Main ----


def main() -> None:
    parser = argparse.ArgumentParser(description="Temporal Imputation Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--signal-type", type=str, default="ecg")
    parser.add_argument("--mask-ratio", type=float, default=0.3)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/imputation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    if args.dummy:
        print("Using dummy model (pipeline verification)")
        model = DummyModel()
        device = "cpu"
    elif args.checkpoint:
        from downstream.model_wrapper import DownstreamModelWrapper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapper = DownstreamModelWrapper(args.checkpoint, args.model_version, device)
        model = wrapper.model
        device = device
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    # Data
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading prepared data: {args.data_path}")
        data = torch.load(args.data_path, weights_only=False)
        signal_type = data["metadata"]["signal_type"]
        test_signals = data["test"]["signals"]
        print(f"  Signal: {signal_type}, Test: {test_signals.shape[0]} samples")
    elif args.dummy:
        signal_type = args.signal_type
        test_signals = torch.randn(50, 3000)  # 50 x 30s
        print(f"  Dummy: {signal_type}, {test_signals.shape[0]} synthetic samples")
    else:
        print("ERROR: --data-path required.", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    print(f"\nEvaluating temporal imputation (mask_ratio={args.mask_ratio})...")
    metrics = evaluate_imputation(
        model,
        test_signals,
        signal_type,
        patch_size=PATCH_SIZE,
        mask_ratio=args.mask_ratio,
        device=device,
    )

    print(f"\n{'=' * 50}")
    print(f"  Temporal Imputation - {signal_type.upper()}")
    print(f"{'=' * 50}")
    print(f"  MSE:       {metrics['mse']:.6f}")
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Windows:   {metrics['n_windows']}")
    print(f"{'=' * 50}")

    results_path = out_dir / f"imputation_{signal_type}_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
