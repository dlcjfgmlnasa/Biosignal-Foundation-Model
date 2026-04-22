# -*- coding:utf-8 -*-
"""Vital Sign Forecasting - 평가 스크립트.

과거 waveform으로 미래 waveform을 autoregressive 생성하여 평가.
model.generate() API 사용. Zero-shot — 추가 학습 없음.

Long-horizon sweep 지원: 하나의 prepared data (context)를 기반으로 여러
target length에서 동시 평가 → "horizon length vs MAE" curve 생성.
Target이 준비된 것보다 길면 자동으로 truncate.

사용법:
    # 단일 horizon (single-input)
    python -m downstream.generation.intra_modal_forecast.run --dummy

    # Long-horizon sweep (30s context → 10/30/60/300s)
    python -m downstream.generation.intra_modal_forecast.run \\
        --checkpoint path/to/best.pt \\
        --data-path outputs/downstream/forecasting/forecasting_mimic3_abp_ctx30s_tgt300s.pt \\
        --target-sec 10 30 60 300

    # Multi-input (ABP+ICP → ICP, prepared data가 multi_input_forecasting task)
    python -m downstream.generation.intra_modal_forecast.run \\
        --checkpoint path/to/best.pt \\
        --data-path outputs/downstream/forecasting/forecasting_multi_mimic3_abp+icp_to_icp_ctx30s_tgt10s.pt
    # → metadata의 task="multi_input_forecasting" 자동 감지 → model.forecast() 사용
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
    def __init__(self, patch_size=100, next_block_size=4):
        self.patch_size = patch_size
        self.next_block_size = next_block_size

    def eval(self):
        return self

    def generate(self, batch, n_steps, denormalize=True):
        b = batch.values.shape[0]
        return torch.randn(n_steps, b, self.patch_size)

    def forecast(self, batch, denormalize=True):
        # (B, N, K, patch_size) — dummy output
        b = batch.values.shape[0]
        n = max(1, batch.values.shape[1] // self.patch_size)
        return torch.randn(b, n, self.next_block_size, self.patch_size)


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


def _multi_context_to_batch(
    context: torch.Tensor,  # (n_context_signals, win_samples)
    context_signal_types: list[str],
    patch_size: int,
) -> PackedBatch:
    """Multi-channel context → any_variate PackedBatch (모든 variate 같은 session).

    각 채널을 별도 BiosignalSample로 만들되 session_id를 동일하게 두어
    PackCollate가 cross-modal pair로 같은 row에 packing.
    """
    n_usable = (context.shape[1] // patch_size) * patch_size
    samples: list[BiosignalSample] = []
    for ch_idx, stype in enumerate(context_signal_types):
        stype_int = SIGNAL_TYPES.get(stype, 0)
        samples.append(BiosignalSample(
            values=context[ch_idx, :n_usable].float(),
            length=n_usable,
            channel_idx=ch_idx,
            recording_idx=0,
            sampling_rate=TARGET_SR,
            n_channels=1,
            win_start=0,
            signal_type=stype_int,
            session_id="forecast_multi_0",
            spatial_id=0,
        ))
    collate = PackCollate(
        max_length=n_usable * len(samples),
        collate_mode="any_variate",
        patch_size=patch_size,
    )
    return collate(samples)


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


@torch.no_grad()
def evaluate_multi_input_forecasting(
    model,
    contexts: torch.Tensor,  # (N, n_context_signals, context_samples)
    targets: torch.Tensor,  # (N, target_samples)
    context_signal_types: list[str],
    target_signal_type: str,
    patch_size: int,
    device: str = "cpu",
) -> dict:
    """Multi-input forecasting 평가.

    `model.forecast()`로 K-patch block 예측 수행 (non-autoregressive).
    Target variate의 마지막 valid 패치 위치에서 K개 미래 패치를 추출.
    target_samples가 K*patch_size보다 길면 truncate (단일 shot 한계).
    """
    k = int(getattr(model, "next_block_size", 4))
    max_gen_samples = k * patch_size
    tgt_samples = min(targets.shape[1], max_gen_samples)
    if tgt_samples < patch_size:
        return {"mse": 0, "mae": 0, "pearson_r": 0, "n_samples": 0}

    target_stype_int = SIGNAL_TYPES.get(target_signal_type, 0)

    all_mse, all_mae, all_r = [], [], []

    for i in range(contexts.shape[0]):
        batch = _multi_context_to_batch(
            contexts[i], context_signal_types, patch_size,
        )
        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)
        if hasattr(batch, "signal_type") and batch.signal_type is not None:
            batch.signal_type = batch.signal_type.to(device)

        # Target variate 위치 찾기: batch.signal_type과 일치하는 패치 중 마지막
        sig_type_tensor = getattr(batch, "signal_type", None)
        if sig_type_tensor is None:
            # Fallback — 첫 variate를 target으로 간주
            target_rows = None
        else:
            target_rows = (sig_type_tensor == target_stype_int)

        out = model.forecast(batch, denormalize=True)  # (B, N, K, patch_size)
        if out is None:
            continue

        b = out.shape[0]
        assert b == 1, "multi-input eval assumes single-sample batch"
        # Target variate의 마지막 valid 패치 인덱스
        if target_rows is not None and target_rows.any():
            last_idx = target_rows.nonzero(as_tuple=True)[0][-1].item()
        else:
            # Fallback: 전체 마지막 패치
            last_idx = out.shape[1] - 1

        block = out[0, last_idx]  # (K, patch_size)
        n_patches_needed = tgt_samples // patch_size
        gen_wave = block[:n_patches_needed].cpu().numpy().reshape(-1)

        target_wave = targets[i, :len(gen_wave)].numpy()

        min_len = min(len(gen_wave), len(target_wave))
        if min_len == 0:
            continue
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
        "max_generated_sec": float(max_gen_samples / TARGET_SR),
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
    parser.add_argument(
        "--target-sec", type=float, nargs="+", default=None,
        help="평가할 target horizon 초 단위 리스트 (예: --target-sec 10 30 60 300). "
             "지정하지 않으면 --data-path의 tgt_sec를 단일 평가.",
    )
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
    is_multi_input = False
    context_signal_types: list[str] | None = None
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading prepared data: {args.data_path}")
        data = torch.load(args.data_path, weights_only=False)
        meta = data["metadata"]
        is_multi_input = meta.get("task") == "multi_input_forecasting"
        if is_multi_input:
            context_signal_types = meta["context_signal_types"]
            signal_type = meta["target_signal_type"]
            contexts = data["test"]["context"]  # (N, n_context, context_samples)
            targets = data["test"]["target"]
            ctx_sec = meta["context_sec"]
            tgt_sec = meta["target_sec"]
            print(f"  Multi-input: {'+'.join(context_signal_types)} → {signal_type}")
            print(f"  Test: {contexts.shape[0]} samples, "
                  f"Context shape: {tuple(contexts.shape)}")
            print(f"  Context: {ctx_sec}s, Target: {tgt_sec}s")
        else:
            signal_type = meta["signal_type"]
            contexts = data["test"]["context"]  # (N, context_samples)
            targets = data["test"]["target"]
            ctx_sec = meta["context_sec"]
            tgt_sec = meta["target_sec"]
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

    # Horizon 리스트 결정: --target-sec 있으면 sweep, 없으면 prepared data 그대로
    max_target_samples = targets.shape[1]
    max_target_sec = max_target_samples / TARGET_SR

    if args.target_sec:
        horizons_sec = sorted(set(args.target_sec))
        oversize = [h for h in horizons_sec if h > max_target_sec + 1e-6]
        if oversize:
            print(
                f"WARN: requested horizons {oversize}s > available {max_target_sec:.1f}s "
                f"— truncated to available length.",
                file=sys.stderr,
            )
            horizons_sec = [h for h in horizons_sec if h <= max_target_sec + 1e-6]
    else:
        horizons_sec = [max_target_sec]

    print(f"\nForecasting: {signal_type.upper()} at horizons {horizons_sec}s...")

    sweep: list[dict] = []
    for tgt_sec in horizons_sec:
        n_samples = int(round(tgt_sec * TARGET_SR))
        n_samples = min(n_samples, max_target_samples)
        targets_trunc = targets[:, :n_samples]
        n_target_patches = n_samples // PATCH_SIZE

        if is_multi_input:
            metrics = evaluate_multi_input_forecasting(
                model,
                contexts,
                targets_trunc,
                context_signal_types=context_signal_types,
                target_signal_type=signal_type,
                patch_size=PATCH_SIZE,
                device=device,
            )
            tag = f"{'+'.join(context_signal_types)}→{signal_type.upper()}"
        else:
            metrics = evaluate_forecasting(
                model,
                contexts,
                targets_trunc,
                signal_type,
                patch_size=PATCH_SIZE,
                device=device,
            )
            tag = signal_type.upper()

        metrics["target_sec"] = float(tgt_sec)
        metrics["n_patches"] = int(n_target_patches)
        sweep.append(metrics)

        print(f"\n{'=' * 50}")
        print(f"  {tag} @ target={tgt_sec}s ({n_target_patches} patches)")
        print(f"{'=' * 50}")
        print(f"  MSE:       {metrics['mse']:.6f}")
        print(f"  MAE:       {metrics['mae']:.6f}")
        print(f"  Pearson r: {metrics['pearson_r']:.4f}")
        print(f"  Samples:   {metrics['n_samples']}")

    # Horizon sweep 요약
    if len(sweep) > 1:
        print(f"\n{'=' * 50}")
        print(f"  Horizon sweep — MAE(horizon)")
        print(f"{'=' * 50}")
        print(f"  {'target_sec':>12}  {'MAE':>10}  {'Pearson r':>10}")
        for m in sweep:
            print(f"  {m['target_sec']:>12.1f}  {m['mae']:>10.6f}  {m['pearson_r']:>10.4f}")

    results_path = out_dir / f"forecasting_{signal_type}_results.json"
    with open(results_path, "w") as f:
        json.dump({"signal_type": signal_type, "sweep": sweep}, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
