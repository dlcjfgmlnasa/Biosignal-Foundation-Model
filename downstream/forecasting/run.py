# -*- coding:utf-8 -*-
"""Vital Sign Forecasting - 평가 스크립트.

과거 waveform으로 미래 waveform을 autoregressive 생성하여 평가.
model.generate() API 사용. Zero-shot — 추가 학습 없음.

사용법:
    python -m downstream.forecasting.run --dummy
    python -m downstream.forecasting.run --checkpoint path/to/best.pt \
        --data-path outputs/downstream/forecasting/forecasting_vitaldb_ecg_ctx30s_tgt10s.pt
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
    def __init__(self, patch_size=100):
        self.patch_size = patch_size

    def eval(self):
        return self

    def generate(self, batch, n_steps, denormalize=True):
        b = batch.values.shape[0]
        return torch.randn(n_steps, b, self.patch_size)


# ---- Batch construction ----


def _context_to_batch(
    context: torch.Tensor, signal_type: str, patch_size: int
) -> PackedBatch:
    """Single context (win_samples,) -> PackedBatch."""
    stype_int = SIGNAL_TYPES.get(signal_type, 0)
    n_usable = (len(context) // patch_size) * patch_size
    sample = BiosignalSample(
        values=context[:n_usable].float(),
        length=n_usable,
        channel_idx=0,
        recording_idx=0,
        sampling_rate=TARGET_SR,
        n_channels=1,
        win_start=0,
        signal_type=stype_int,
        session_id="forecast_0",
        spatial_id=0,
    )
    collate = PackCollate(max_length=n_usable, collate_mode="ci", patch_size=patch_size)
    return collate([sample])


# ---- Evaluation ----


@torch.no_grad()
def evaluate_forecasting(
    model,
    contexts: torch.Tensor,  # (N, context_samples)
    targets: torch.Tensor,  # (N, target_samples)
    signal_type: str,
    patch_size: int,
    device: str = "cpu",
) -> dict:
    """Forecasting 평가: generated vs target waveform의 MSE/MAE/Pearson r."""
    n_target_patches = targets.shape[1] // patch_size
    if n_target_patches < 1:
        return {"mse": 0, "mae": 0, "pearson_r": 0, "n_samples": 0}

    all_mse, all_mae, all_r = [], [], []

    for i in range(contexts.shape[0]):
        batch = _context_to_batch(contexts[i], signal_type, patch_size)
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

        generated = model.generate(batch, n_steps=n_target_patches, denormalize=True)
        # generated: (n_steps, 1, patch_size)
        gen_wave = generated[:, 0, :].cpu().numpy().reshape(-1)

        target_wave = targets[i].numpy()[: len(gen_wave)]

        if len(gen_wave) == 0 or len(target_wave) == 0:
            continue

        min_len = min(len(gen_wave), len(target_wave))
        gen_wave = gen_wave[:min_len]
        target_wave = target_wave[:min_len]

        mse = float(np.mean((gen_wave - target_wave) ** 2))
        mae = float(np.mean(np.abs(gen_wave - target_wave)))

        if np.std(gen_wave) > 1e-8 and np.std(target_wave) > 1e-8:
            r = float(np.corrcoef(gen_wave, target_wave)[0, 1])
        else:
            r = 0.0

        all_mse.append(mse)
        all_mae.append(mae)
        all_r.append(r)

    return {
        "mse": float(np.mean(all_mse)) if all_mse else 0.0,
        "mae": float(np.mean(all_mae)) if all_mae else 0.0,
        "pearson_r": float(np.mean(all_r)) if all_r else 0.0,
        "n_samples": len(all_mse),
    }


# ---- Main ----


def main() -> None:
    parser = argparse.ArgumentParser(description="Vital Sign Forecasting Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--signal-type", type=str, default="ecg")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/forecasting")
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
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    # Data
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading prepared data: {args.data_path}")
        data = torch.load(args.data_path, weights_only=False)
        signal_type = data["metadata"]["signal_type"]
        contexts = data["test"]["context"]  # (N, context_samples)
        targets = data["test"]["target"]  # (N, target_samples)
        ctx_sec = data["metadata"]["context_sec"]
        tgt_sec = data["metadata"]["target_sec"]
        print(f"  Signal: {signal_type}, Test: {contexts.shape[0]} samples")
        print(f"  Context: {ctx_sec}s, Target: {tgt_sec}s")
    elif args.dummy:
        signal_type = args.signal_type
        contexts = torch.randn(20, 3000)  # 20 x 30s
        targets = torch.randn(20, 1000)  # 20 x 10s
        print(f"  Dummy: {signal_type}, 20 synthetic samples (30s->10s)")
    else:
        print("ERROR: --data-path required.", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    n_target_patches = targets.shape[1] // PATCH_SIZE
    print(
        f"\nForecasting: {signal_type.upper()} ({n_target_patches} patches to generate)..."
    )

    metrics = evaluate_forecasting(
        model,
        contexts,
        targets,
        signal_type,
        patch_size=PATCH_SIZE,
        device=device,
    )

    print(f"\n{'=' * 50}")
    print(f"  Vital Sign Forecasting - {signal_type.upper()}")
    print(f"{'=' * 50}")
    print(f"  MSE:       {metrics['mse']:.6f}")
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Samples:   {metrics['n_samples']}")
    print(f"{'=' * 50}")

    results_path = out_dir / f"forecasting_{signal_type}_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
