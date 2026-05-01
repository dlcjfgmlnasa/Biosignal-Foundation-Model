# -*- coding:utf-8 -*-
"""ICU Extubation Failure Prediction (extub_fail_48h).

환자 단위 예측: 여러 10분 윈도우를 Foundation Model로 인코딩한 뒤,
Transformer Aggregator로 시간 순서를 반영하여 환자 수준 표현을 생성한다.

구조:
    ICU Stay (수시간)
    → 10분 윈도우 × K개 슬라이딩
    → Foundation Model Encoder (frozen) → h_1, h_2, ..., h_K  (d_model)
    → [CLS] + h_1..h_K → Transformer Aggregator (학습 가능)
    → CLS output → LinearProbe → extubation 예측

2가지 모드:
  - linear_probe: Frozen encoder + Transformer Aggregator + LinearProbe
  - lora:         Frozen encoder + LoRA + Transformer Aggregator + LinearProbe

사용법:
    python -m downstream.outcome.extubation.run \
        --checkpoint best.pt \
        --data-path datasets/processed/extubation/extubation_w600s.pt \
        --mode linear_probe --epochs 30 --max-windows 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe
from downstream.aggregator import (
    SIGNAL_TYPE_INT,
    TransformerAggregator,
    collate_patients,
    encode_patient_windows,
)


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0


# ── 학습 ─────────────────────────────────────────────────────


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
    aggregator = aggregator.to(device)
    probe = probe.to(device)
    aggregator.train()
    probe.train()

    params = list(aggregator.parameters()) + list(probe.parameters())
    if use_lora:
        model.model.train()
        params += model.lora_parameters()

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        # 환자 셔플
        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(train_patients))

        epoch_loss, n_batches = 0.0, 0

        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start: batch_start + batch_size]

            # 1. 각 환자의 윈도우를 인코딩
            patient_reprs = []
            batch_labels = []
            for idx in batch_indices:
                p = train_patients[idx]
                reprs = encode_patient_windows(
                    model, p, patch_size, max_windows,
                    use_lora=use_lora, session_prefix="mort",
                )
                patient_reprs.append(reprs)
                batch_labels.append(p["extubation"])

            # 2. 패딩 + Aggregator
            padded, mask, labels, _ = collate_patients(
                patient_reprs, batch_labels, device
            )
            patient_repr = aggregator(padded, mask)  # (B, d_model)

            # 3. Probe + Loss
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


# ── 평가 ─────────────────────────────────────────────────────


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
        reprs = encode_patient_windows(model, p, patch_size, max_windows)
        padded = reprs.unsqueeze(0).to(device)  # (1, K, d_model)
        mask = torch.ones(1, reprs.shape[0], dtype=torch.bool, device=device)

        patient_repr = aggregator(padded, mask)
        logit = probe(patient_repr)
        prob = torch.sigmoid(logit).squeeze().cpu().item()

        all_labels.append(p["extubation"])
        all_scores.append(prob)

    return _compute_metrics(np.array(all_labels), np.array(all_scores))


# ── 메트릭 ───────────────────────────────────────────────────


def _compute_metrics(y_true, y_score):
    auroc = compute_auroc(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)

    best_thresh, best_j = 0.5, -1.0
    for thresh in np.linspace(0.01, 0.99, 99):
        y_pred = (y_score >= thresh).astype(int)
        ss = compute_sensitivity_specificity(y_true, y_pred)
        j = ss["sensitivity"] + ss["specificity"] - 1.0
        if j > best_j:
            best_j = j
            best_thresh = thresh

    y_pred_opt = (y_score >= best_thresh).astype(int)
    ss_opt = compute_sensitivity_specificity(y_true, y_pred_opt)
    f1 = compute_f1(y_true, y_pred_opt, average="macro")

    return {
        "auroc": auroc, "auprc": auprc, "f1_macro": f1,
        "optimal_threshold": float(best_thresh),
        "sensitivity": ss_opt["sensitivity"],
        "specificity": ss_opt["specificity"],
        "n_total": len(y_true),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) > 0 else 0.0,
        "y_true": y_true, "y_score": y_score,
    }


# ── 데이터 로딩 ──────────────────────────────────────────────


def _load_data(data_path: str) -> tuple[list[dict], list[dict], dict]:
    """환자 단위로 그룹핑된 .pt 파일을 로드한다."""
    print(f"\nLoading data: {data_path}")
    data = torch.load(data_path, weights_only=False)
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    print(f"  Signals: {meta.get('input_signals', '?')}")
    print(f"  Window: {meta.get('window_sec', '?')}s")
    print(f"  Aggregation: {meta.get('aggregation', 'window_level')}")

    return data["train"], data["test"], meta


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICU Extubation Failure Prediction (Patient-Level Transformer Aggregation)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="linear_probe",
                        choices=["linear_probe", "lora"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="환자 수 per batch (윈도우 수 아님)")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--max-windows", type=int, default=24,
                        help="환자당 최대 윈도우 수 (초과 시 균등 샘플링)")
    parser.add_argument("--agg-layers", type=int, default=2,
                        help="Transformer Aggregator 레이어 수")
    parser.add_argument("--agg-heads", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── 모델 로드 ──
    from downstream.model_wrapper import DownstreamModelWrapper

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
    d_model = model.d_model
    patch_size = model.patch_size
    print(f"  d_model={d_model}, patch_size={patch_size}")

    use_lora = args.mode == "lora"
    if use_lora:
        model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    # ── 데이터 로드 ──
    train_patients, test_patients, meta = _load_data(args.data_path)

    n_dead_train = sum(1 for p in train_patients if p["extubation"] == 1)
    n_dead_test = sum(1 for p in test_patients if p["extubation"] == 1)
    avg_win_train = np.mean([p["n_windows"] for p in train_patients])
    avg_win_test = np.mean([p["n_windows"] for p in test_patients])
    print(f"  Train: {len(train_patients)} patients "
          f"({n_dead_train} dead, avg {avg_win_train:.1f} windows)")
    print(f"  Test:  {len(test_patients)} patients "
          f"({n_dead_test} dead, avg {avg_win_test:.1f} windows)")
    print(f"  Max windows per patient: {args.max_windows}")

    # ── Aggregator + Probe ──
    aggregator = TransformerAggregator(
        d_model=d_model,
        n_heads=args.agg_heads,
        n_layers=args.agg_layers,
        max_windows=args.max_windows,
    )
    probe = LinearProbe(d_model, n_classes=1)

    n_agg = sum(p.numel() for p in aggregator.parameters())
    n_probe = sum(p.numel() for p in probe.parameters())
    print(f"\n  Aggregator: {n_agg:,} params ({args.agg_layers} layers, "
          f"{args.agg_heads} heads)")
    print(f"  Probe: {n_probe:,} params")
    if use_lora:
        n_lora = sum(p.numel() for p in model.lora_parameters())
        print(f"  LoRA: {n_lora:,} params (rank={args.lora_rank})")

    # ── 학습 ──
    print(f"\nTraining ({args.mode})...")
    train_losses = train_model(
        model, aggregator, probe, train_patients,
        epochs=args.epochs, lr=args.lr, device=device,
        patch_size=patch_size, max_windows=args.max_windows,
        batch_size=args.batch_size, use_lora=use_lora,
    )

    # ── 평가 ──
    print("\nEvaluating...")
    metrics = evaluate_model(
        model, aggregator, probe, test_patients,
        device=device, patch_size=patch_size,
        max_windows=args.max_windows,
    )

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 60}")
    print(f"  ICU Extubation Failure — {args.mode} (Transformer Aggregator)")
    print(f"{'=' * 60}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Prevalence:  {metrics['prevalence']:.3f} "
          f"({metrics['n_positive']}/{metrics['n_total']})")
    print(f"{'=' * 60}")

    roc_path = out_dir / f"extubation_roc_{args.mode}.png"
    plot_roc_curve(y_true, y_score, roc_path,
                   title=f"ICU Extubation Failure — {args.mode} ROC")
    print(f"\nROC curve: {roc_path}")

    results = {
        **metrics, "train_losses": train_losses,
        "config": {
            "task": "icu_extubation_prediction",
            "mode": args.mode,
            "aggregation": "transformer",
            "agg_layers": args.agg_layers,
            "agg_heads": args.agg_heads,
            "max_windows": args.max_windows,
            "data_path": args.data_path,
            "epochs": args.epochs, "lr": args.lr,
        },
    }
    results_path = out_dir / f"extubation_results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
