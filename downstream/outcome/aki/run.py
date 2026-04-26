# -*- coding:utf-8 -*-
"""Postoperative AKI Prediction (KDIGO Cr 기준).

환자 단위 예측: 수술 중 다채널 신호 (ABP/ECG/PPG/CVP) → postop AKI.

구조 (mortality와 동일 패턴):
    Intraop session (수십 분~수시간)
    → 10분 윈도우 × K개 슬라이딩
    → Foundation Model Encoder (frozen or LoRA) → h_1..h_K
    → [CLS] + h_1..h_K → Transformer Aggregator
    → CLS → LinearProbe → AKI 예측

라벨 모드 (prepare_data.py의 `--label-mode`와 일치):
    binary : Stage ≥1 vs no AKI (BCEWithLogitsLoss)
    stage  : KDIGO 0/1/2/3 (CrossEntropyLoss + macro AUROC OvR)

사용법:
    python -m downstream.outcome.aki.run \
        --checkpoint best.pt \
        --data-path datasets/processed/aki/aki_binary_abp_cvp_ecg_ppg_w600s.pt \
        --mode linear_probe --epochs 30 --max-windows 24
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from downstream.metrics import (
    compute_auprc,
    compute_auroc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe
from downstream.aggregator import (
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
    label_mode: str,
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

    if label_mode == "binary":
        # 클래스 불균형 보정: pos_weight = n_neg / n_pos
        n_pos = sum(1 for p in train_patients if p["label"] == 1)
        n_neg = len(train_patients) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  BCE pos_weight={pos_weight.item():.2f} (n_pos={n_pos}, n_neg={n_neg})")
    else:
        # stage: 4-class CE
        class_counts = np.bincount(
            [p["label"] for p in train_patients], minlength=4
        )
        weights = 1.0 / np.clip(class_counts, 1, None)
        weights = weights / weights.sum() * 4
        cls_w = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cls_w)
        print(f"  CE class_weights={weights.round(3).tolist()} "
              f"(counts={class_counts.tolist()})")

    losses: list[float] = []

    for epoch in range(epochs):
        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(train_patients))

        epoch_loss, n_batches = 0.0, 0

        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start: batch_start + batch_size]

            patient_reprs = []
            batch_labels = []
            for idx in batch_indices:
                p = train_patients[idx]
                reprs = encode_patient_windows(
                    model, p, patch_size, max_windows,
                    use_lora=use_lora, session_prefix="aki",
                )
                patient_reprs.append(reprs)
                batch_labels.append(p["label"])

            padded, mask, labels = collate_patients(
                patient_reprs, batch_labels, device
            )
            patient_repr = aggregator(padded, mask)  # (B, d_model)

            logits = probe(patient_repr)
            if label_mode == "binary":
                loss = criterion(logits.squeeze(-1), labels.float())
            else:
                loss = criterion(logits, labels.long())

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
    label_mode: str,
    device: torch.device,
    patch_size: int,
    max_windows: int,
) -> dict:
    aggregator.to(device).eval()
    probe.to(device).eval()
    if hasattr(model, "model"):
        model.model.eval()

    all_labels: list[int] = []
    all_scores: list[np.ndarray] = []  # binary: scalar, stage: (4,) softmax

    for p in test_patients:
        reprs = encode_patient_windows(model, p, patch_size, max_windows)
        padded = reprs.unsqueeze(0).to(device)  # (1, K, d_model)
        mask = torch.ones(1, reprs.shape[0], dtype=torch.bool, device=device)

        patient_repr = aggregator(padded, mask)
        logit = probe(patient_repr)

        if label_mode == "binary":
            prob = torch.sigmoid(logit).squeeze().cpu().item()
            all_scores.append(np.array([prob]))
        else:
            probs = torch.softmax(logit, dim=-1).squeeze(0).cpu().numpy()
            all_scores.append(probs)

        all_labels.append(p["label"])

    y_true = np.array(all_labels)
    y_score = np.stack(all_scores)  # (N,1) binary or (N,4) stage

    if label_mode == "binary":
        return _compute_metrics_binary(y_true, y_score[:, 0])
    return _compute_metrics_stage(y_true, y_score)


# ── 메트릭 ───────────────────────────────────────────────────


def _compute_metrics_binary(y_true: np.ndarray, y_score: np.ndarray) -> dict:
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


def _compute_metrics_stage(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """4-class KDIGO stage 평가: macro AUROC (OvR), accuracy, per-class AUROC."""
    n_classes = 4
    per_class_auroc: list[float] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class_auroc.append(float("nan"))
            continue
        per_class_auroc.append(compute_auroc(y_bin, y_score[:, c]))

    valid = [a for a in per_class_auroc if not np.isnan(a)]
    macro_auroc = float(np.mean(valid)) if valid else float("nan")

    y_pred = y_score.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    f1 = compute_f1(y_true, y_pred, average="macro")
    counts = np.bincount(y_true, minlength=n_classes).tolist()

    # AKI vs no-AKI binary view (stage>=1 == positive)
    y_true_bin = (y_true >= 1).astype(int)
    y_score_bin = 1.0 - y_score[:, 0]  # P(stage >= 1) = 1 - P(stage=0)
    auroc_bin = (
        compute_auroc(y_true_bin, y_score_bin)
        if 0 < y_true_bin.sum() < len(y_true_bin)
        else float("nan")
    )

    return {
        "macro_auroc": macro_auroc,
        "per_class_auroc": per_class_auroc,
        "accuracy": acc,
        "f1_macro": f1,
        "binary_auroc": auroc_bin,
        "class_counts": counts,
        "n_total": len(y_true),
        "y_true": y_true,
        "y_score": y_score,
    }


# ── 데이터 로딩 ──────────────────────────────────────────────


def _load_data(data_path: str) -> tuple[list[dict], list[dict], dict]:
    print(f"\nLoading data: {data_path}")
    data = torch.load(data_path, weights_only=False)
    meta = data.get("metadata", {})
    print(f"  Task:        {meta.get('task', '?')}")
    print(f"  Label mode:  {meta.get('label_mode', '?')}")
    print(f"  Signals:     {meta.get('input_signals', '?')}")
    print(f"  Window:      {meta.get('window_sec', '?')}s")
    print(f"  Postop win:  {meta.get('max_postop_days', '?')} days")
    return data["train"], data["test"], meta


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postop AKI Prediction (Patient-Level Transformer Aggregation)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-version", type=str, default="v1", choices=["v1", "v2"]
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, default="linear_probe",
        choices=["linear_probe", "lora"],
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="환자 수 per batch (윈도우 수 아님)",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument(
        "--max-windows", type=int, default=24,
        help="환자당 최대 윈도우 수 (초과 시 균등 샘플링)",
    )
    parser.add_argument("--agg-layers", type=int, default=2)
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
    label_mode = meta.get("label_mode", "binary")
    if label_mode not in {"binary", "stage"}:
        raise ValueError(f"Unknown label_mode in data: {label_mode}")

    if label_mode == "binary":
        n_pos_tr = sum(1 for p in train_patients if p["label"] == 1)
        n_pos_te = sum(1 for p in test_patients if p["label"] == 1)
        print(f"  Train: {len(train_patients)} patients (AKI={n_pos_tr})")
        print(f"  Test:  {len(test_patients)} patients (AKI={n_pos_te})")
    else:
        tr_counts = np.bincount(
            [p["label"] for p in train_patients], minlength=4
        ).tolist()
        te_counts = np.bincount(
            [p["label"] for p in test_patients], minlength=4
        ).tolist()
        print(f"  Train stages: {tr_counts}")
        print(f"  Test stages:  {te_counts}")
    avg_win = float(np.mean([p["n_windows"] for p in train_patients]))
    print(f"  Avg windows/patient (train): {avg_win:.1f}")
    print(f"  Max windows per patient: {args.max_windows}")

    # ── Aggregator + Probe ──
    n_classes = 1 if label_mode == "binary" else 4
    aggregator = TransformerAggregator(
        d_model=d_model,
        n_heads=args.agg_heads,
        n_layers=args.agg_layers,
        max_windows=args.max_windows,
    )
    probe = LinearProbe(d_model, n_classes=n_classes)

    n_agg = sum(p.numel() for p in aggregator.parameters())
    n_probe = sum(p.numel() for p in probe.parameters())
    print(f"\n  Aggregator: {n_agg:,} params ({args.agg_layers} layers, "
          f"{args.agg_heads} heads)")
    print(f"  Probe: {n_probe:,} params (n_classes={n_classes})")
    if use_lora:
        n_lora = sum(p.numel() for p in model.lora_parameters())
        print(f"  LoRA: {n_lora:,} params (rank={args.lora_rank})")

    # ── 학습 ──
    print(f"\nTraining ({args.mode}, label_mode={label_mode})...")
    train_losses = train_model(
        model, aggregator, probe, train_patients, label_mode,
        epochs=args.epochs, lr=args.lr, device=device,
        patch_size=patch_size, max_windows=args.max_windows,
        batch_size=args.batch_size, use_lora=use_lora,
    )

    # ── 평가 ──
    print("\nEvaluating...")
    metrics = evaluate_model(
        model, aggregator, probe, test_patients, label_mode,
        device=device, patch_size=patch_size, max_windows=args.max_windows,
    )

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 60}")
    print(f"  Postop AKI ({label_mode}) — {args.mode}")
    print(f"{'=' * 60}")
    if label_mode == "binary":
        print(f"  AUROC:       {metrics['auroc']:.4f}")
        print(f"  AUPRC:       {metrics['auprc']:.4f}")
        print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Prevalence:  {metrics['prevalence']:.3f} "
              f"({metrics['n_positive']}/{metrics['n_total']})")
        roc_path = out_dir / f"aki_roc_{args.mode}.png"
        plot_roc_curve(
            y_true, y_score[:, 0], roc_path,
            title=f"Postop AKI — {args.mode} ROC",
        )
        print(f"\nROC curve: {roc_path}")
    else:
        print(f"  Macro AUROC:    {metrics['macro_auroc']:.4f}")
        print(f"  Per-class AUROC: "
              f"{[f'{a:.3f}' for a in metrics['per_class_auroc']]}")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
        print(f"  Binary AUROC:   {metrics['binary_auroc']:.4f}  (stage≥1)")
        print(f"  Class counts:   {metrics['class_counts']}")
        # 이항 환원 ROC (stage>=1 vs no AKI)
        y_true_bin = (y_true >= 1).astype(int)
        y_score_bin = 1.0 - y_score[:, 0]
        roc_path = out_dir / f"aki_stage_binary_roc_{args.mode}.png"
        plot_roc_curve(
            y_true_bin, y_score_bin, roc_path,
            title=f"Postop AKI (stage≥1) — {args.mode} ROC",
        )
        print(f"\nBinary ROC curve: {roc_path}")
    print(f"{'=' * 60}")

    results = {
        **metrics,
        "train_losses": train_losses,
        "config": {
            "task": "postop_aki_prediction",
            "label_mode": label_mode,
            "mode": args.mode,
            "aggregation": "transformer",
            "agg_layers": args.agg_layers,
            "agg_heads": args.agg_heads,
            "max_windows": args.max_windows,
            "data_path": args.data_path,
            "epochs": args.epochs,
            "lr": args.lr,
        },
    }
    results_path = out_dir / f"aki_results_{args.mode}_{label_mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
