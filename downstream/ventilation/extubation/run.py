# -*- coding:utf-8 -*-
"""Extubation Success Prediction  - Paper 4.2.4.

발관 직전 30분 waveform (CO2 + AWP + ECG + ABP)으로 발관 성공 여부 예측.
Success = (48시간 내 reintubation 없음) AND (48시간 내 사망 없음).

환자 단위 예측: Mortality와 동일 구조  - 여러 윈도우 × TransformerAggregator.

구조:
    Pre-extubation 30분
    → 5분 윈도우 × 6개 슬라이딩
    → Foundation Model Encoder (frozen/LoRA) → h_1..h_6
    → [CLS] + h_1..h_6 → Transformer Aggregator → CLS → LinearProbe → success/failure

사용법:
    # Dummy
    python -m downstream.ventilation.extubation.run --dummy --max-windows 6

    # Linear probe
    python -m downstream.ventilation.extubation.run \
        --checkpoint best.pt --mode linear_probe \
        --data-path datasets/processed/extubation/extubation_vitaldb_w300s.pt \
        --epochs 30 --max-windows 6

    # LoRA
    python -m downstream.ventilation.extubation.run \
        --checkpoint best.pt --mode lora --lora-rank 8 --lr 1e-4 \
        --data-path datasets/processed/extubation/extubation_vitaldb_w300s.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from downstream.shared.aggregator import (
    SIGNAL_TYPE_INT,
    TransformerAggregator,
    collate_patients,
    encode_patient_windows,
)
from downstream.shared.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.shared.model_wrapper import LinearProbe
from downstream.shared.viz import plot_roc_curve
from downstream.shared.window_task import compute_binary_metrics


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0


def train_model(
    model,
    aggregator: TransformerAggregator,
    probe: LinearProbe,
    train_patients: list[dict],
    epochs: int,
    lr: float,
    device: torch.device,
    patch_size: int,
    max_windows: int,
    batch_size: int = 8,
    use_lora: bool = False,
    gradient_clip: float = 1.0,
) -> list[float]:
    aggregator = aggregator.to(device).train()
    probe = probe.to(device).train()

    params = list(aggregator.parameters()) + list(probe.parameters())
    if use_lora:
        model.model.train()
        params += model.lora_parameters()

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(train_patients))
        epoch_loss, n_batches = 0.0, 0

        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start : batch_start + batch_size]

            patient_reprs = []
            batch_labels = []
            for idx in batch_indices:
                p = train_patients[idx]
                reprs = encode_patient_windows(
                    model, p, patch_size, max_windows,
                    use_lora=use_lora, session_prefix="ext",
                )
                patient_reprs.append(reprs)
                batch_labels.append(p["label"])

            padded, mask, labels = collate_patients(
                patient_reprs, batch_labels, device
            )
            patient_repr = aggregator(padded, mask)
            logits = probe(patient_repr)
            loss = criterion(logits.squeeze(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    return losses


@torch.no_grad()
def evaluate_model(
    model,
    aggregator: TransformerAggregator,
    probe: LinearProbe,
    test_patients: list[dict],
    device: torch.device,
    patch_size: int,
    max_windows: int,
) -> dict:
    aggregator.to(device).eval()
    probe.to(device).eval()
    if hasattr(model, "model"):
        model.model.eval()

    all_labels, all_scores = [], []

    for p in test_patients:
        reprs = encode_patient_windows(
            model, p, patch_size, max_windows, session_prefix="ext"
        )
        padded = reprs.unsqueeze(0).to(device)
        mask = torch.ones(1, reprs.shape[0], dtype=torch.bool, device=device)

        patient_repr = aggregator(padded, mask)
        logit = probe(patient_repr)
        prob = torch.sigmoid(logit).squeeze().cpu().item()

        all_labels.append(p["label"])
        all_scores.append(prob)

    return compute_binary_metrics(np.array(all_labels), np.array(all_scores))


def _load_data(data_path: str) -> tuple[list[dict], list[dict], dict]:
    print(f"\nLoading: {data_path}")
    data = torch.load(data_path, weights_only=False)
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    return data["train"], data["test"], meta


def _make_dummy_patients(
    n: int,
    win_samples: int,
    n_windows: int,
    sig_types: list[str] = ("co2", "awp", "ecg", "abp"),
    seed: int = 42,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        signals = {
            st: torch.from_numpy(
                rng.standard_normal((n_windows, win_samples)).astype(np.float32)
            )
            for st in sig_types
        }
        out.append(
            {
                "signals": signals,
                "n_windows": n_windows,
                "label": int(rng.integers(0, 2)),
                "case_id": f"dummy_{i}",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extubation Success Prediction (Patient-Level Aggregation)"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--mode", type=str, default="linear_probe",
        choices=["linear_probe", "lora"],
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--max-windows", type=int, default=6)
    parser.add_argument("--agg-layers", type=int, default=2)
    parser.add_argument("--agg-heads", type=int, default=4)
    parser.add_argument("--window-sec", type=float, default=300.0)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── 모델 ──
    if args.dummy:
        print("Using dummy feature extractor")
        from downstream.shared.window_task import DummyFeatureExtractor

        model = DummyFeatureExtractor(d_model=128)
        d_model = 128
        win_samples = int(args.window_sec * 100)
        train_patients = _make_dummy_patients(16, win_samples, args.max_windows)
        test_patients = _make_dummy_patients(
            8, win_samples, args.max_windows, seed=7
        )
    elif args.checkpoint:
        from downstream.shared.model_wrapper import DownstreamModelWrapper

        print(f"Loading: {args.checkpoint}")
        model = DownstreamModelWrapper(
            args.checkpoint, args.model_version, args.device
        )
        d_model = model.d_model
        if args.mode == "lora":
            model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)

        if not args.data_path:
            print("ERROR: --data-path required.", file=sys.stderr)
            sys.exit(1)
        train_patients, test_patients, _ = _load_data(args.data_path)
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    # ── Aggregator + Probe ──
    aggregator = TransformerAggregator(
        d_model=d_model, n_heads=args.agg_heads,
        n_layers=args.agg_layers, max_windows=args.max_windows,
    )
    probe = LinearProbe(d_model, n_classes=1)

    n_pos_tr = sum(1 for p in train_patients if p["label"] == 1)
    n_pos_te = sum(1 for p in test_patients if p["label"] == 1)
    print(f"  Train: {len(train_patients)} patients ({n_pos_tr} success)")
    print(f"  Test:  {len(test_patients)} patients ({n_pos_te} success)")
    print(f"  Max windows/patient: {args.max_windows}")

    # ── 학습 ──
    print(f"\nTraining ({args.mode})...")
    use_lora = args.mode == "lora"
    if args.dummy and use_lora:
        print(
            "  (Note: --dummy + lora  - DummyFeatureExtractor는 LoRA 미지원, "
            "linear_probe 루프 사용)"
        )
        use_lora = False
    patch_size = args.patch_size if args.dummy else model.patch_size

    train_losses = train_model(
        model, aggregator, probe, train_patients,
        epochs=args.epochs, lr=args.lr, device=device,
        patch_size=patch_size, max_windows=args.max_windows,
        batch_size=args.batch_size, use_lora=use_lora,
    )

    print("\nEvaluating...")
    metrics = evaluate_model(
        model, aggregator, probe, test_patients,
        device=device, patch_size=patch_size, max_windows=args.max_windows,
    )

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 60}")
    print(f"  Extubation Success  - {args.mode} (Transformer Aggregator)")
    print(f"  AUROC: {metrics['auroc']:.4f}  AUPRC: {metrics['auprc']:.4f}")
    print(f"  Sens: {metrics['sensitivity']:.4f}  Spec: {metrics['specificity']:.4f}")
    print(f"  Prevalence: {metrics['prevalence']:.3f}")
    print(f"{'=' * 60}")

    roc_path = out_dir / f"extubation_roc_{args.mode}.png"
    plot_roc_curve(
        y_true, y_score, roc_path, title=f"Extubation  - {args.mode}"
    )
    print(f"\nROC: {roc_path}")

    results = {
        **metrics,
        "train_losses": train_losses,
        "config": {
            "task": "extubation_success",
            "mode": args.mode,
            "max_windows": args.max_windows,
            "agg_layers": args.agg_layers,
            "agg_heads": args.agg_heads,
            "window_sec": args.window_sec,
            "epochs": args.epochs,
            "lr": args.lr,
        },
    }
    results_path = out_dir / f"extubation_results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
