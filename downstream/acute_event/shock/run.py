# -*- coding:utf-8 -*-
"""Shock Prediction (Circulatory Shock, MAP<65 AND HR>100)  - Paper 4.2.2.

미래 horizon 내 순환성 쇼크 (MAP<65 AND HR>100, ≥1분 sustained) 예측.
Hypotension task와 유사하지만 HR tachycardia 조건이 추가되어 더 엄격.

2가지 모드:
  - linear_probe: Frozen encoder + LinearProbe
  - lora:         Frozen encoder + LoRA adapters + LinearProbe

사용법:
    # Dummy
    python -m downstream.acute_event.shock.run --dummy

    # Linear probe
    python -m downstream.acute_event.shock.run \
        --checkpoint best.pt --mode linear_probe \
        --data-path outputs/downstream/shock/shock_vitaldb_abp_ecg_h5min.pt

    # LoRA
    python -m downstream.acute_event.shock.run \
        --checkpoint best.pt --mode lora --lr 1e-4 --epochs 30 \
        --data-path outputs/downstream/shock/shock_vitaldb_abp_ecg_h5min.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from downstream.model_wrapper import LinearProbe
from downstream.viz import plot_roc_curve
from downstream.window_task import (
    DEFAULT_PATCH_SIZE,
    DummyFeatureExtractor,
    MultiSignalWindow,
    evaluate_linear_probe,
    evaluate_lora,
    make_batches,
    make_dummy_windows,
    train_linear_probe,
    train_lora,
)


def _load_data(
    data_path: str, input_signals: list[str]
) -> tuple[list[MultiSignalWindow], list[MultiSignalWindow]]:
    print(f"\nLoading: {data_path}")
    data = torch.load(data_path, weights_only=False)
    meta = data.get("metadata", {})
    print(f"  Task={meta.get('task', '?')} Horizon={meta.get('horizon_sec', '?')}s")

    def _split(split):
        labels_t = split["labels"]
        label_values_t = split["label_values"]
        n = len(labels_t)
        out = []
        for i in range(n):
            sigs = {
                k: split["signals"][k][i].numpy()
                for k in input_signals
                if k in split["signals"]
            }
            out.append(
                MultiSignalWindow(
                    signals=sigs,
                    label=int(labels_t[i].item()),
                    label_value=float(label_values_t[i].item()),
                    case_id=split["case_ids"][i] if "case_ids" in split else i,
                )
            )
        return out

    return _split(data["train"]), _split(data["test"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Shock Prediction (MAP<65 AND HR>100)"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--mode", type=str, default="linear_probe",
        choices=["linear_probe", "lora"],
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--window-sec", type=float, default=600.0)
    parser.add_argument(
        "--input-signals", nargs="+", default=["abp", "ecg"],
        choices=["abp", "ecg", "ppg"],
    )
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    if args.dummy:
        print("Using dummy feature extractor")
        model = DummyFeatureExtractor(d_model=128)
        d_model = 128
        train_windows = make_dummy_windows(
            64, args.input_signals, win_samples=int(args.window_sec * 100)
        )
        test_windows = make_dummy_windows(
            32, args.input_signals, win_samples=int(args.window_sec * 100), seed=7
        )
    elif args.checkpoint:
        from downstream.model_wrapper import DownstreamModelWrapper

        print(f"Loading checkpoint: {args.checkpoint}")
        model = DownstreamModelWrapper(
            args.checkpoint, args.model_version, args.device
        )
        d_model = model.d_model
        if args.mode == "lora":
            model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)

        if not args.data_path:
            print("ERROR: --data-path required (or use --dummy).", file=sys.stderr)
            sys.exit(1)
        train_windows, test_windows = _load_data(args.data_path, args.input_signals)
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    sig_str = " + ".join(s.upper() for s in args.input_signals)
    print(f"Mode: {args.mode} | Input: {sig_str} | Window: {args.window_sec}s")
    n_pos_tr = sum(1 for w in train_windows if w.label == 1)
    n_pos_te = sum(1 for w in test_windows if w.label == 1)
    print(f"  Train: {len(train_windows)} ({n_pos_tr} shock)")
    print(f"  Test:  {len(test_windows)} ({n_pos_te} shock)")

    first_sig = next(iter(train_windows[0].signals.values()))
    max_length = len(first_sig)
    train_batches = make_batches(
        train_windows, args.batch_size, args.patch_size, max_length
    )
    test_batches = make_batches(
        test_windows, args.batch_size, args.patch_size, max_length
    )

    probe = LinearProbe(d_model, n_classes=1)

    if args.mode == "linear_probe":
        print(f"\nTraining LinearProbe (d_model={d_model})...")
        losses = train_linear_probe(
            model, probe, train_batches, args.epochs, args.lr, device
        )
        metrics = evaluate_linear_probe(model, probe, test_batches, device)
    else:
        print(f"\nTraining LoRA (rank={args.lora_rank})...")
        losses = train_lora(
            model, probe, train_batches, args.epochs, args.lr, device
        )
        metrics = evaluate_lora(model, probe, test_batches, device)

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 50}")
    print(f"  Shock  - {args.mode}")
    print(f"  AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}")
    print(f"  Sens: {metrics['sensitivity']:.4f}  Spec: {metrics['specificity']:.4f}")
    print(f"  Prevalence: {metrics['prevalence']:.3f}")
    print(f"{'=' * 50}")

    roc_path = out_dir / f"shock_roc_{args.mode}.png"
    plot_roc_curve(y_true, y_score, roc_path, title=f"Shock  - {args.mode}")
    print(f"\nROC: {roc_path}")

    results = {
        **metrics,
        "train_losses": losses,
        "config": {
            "task": "shock_prediction",
            "mode": args.mode,
            "input_signals": args.input_signals,
            "window_sec": args.window_sec,
            "epochs": args.epochs,
            "lr": args.lr,
        },
    }
    results_path = out_dir / f"shock_results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
