# -*- coding:utf-8 -*-
"""Intracranial Hypertension Detection (ICP > 20mmHg).

MIMIC-III ICP 기반 두개내 고혈압 탐지 — Foundation model representation 평가.

2가지 모드:
  - linear_probe: Frozen encoder + LinearProbe
  - lora:         Frozen encoder + LoRA adapters + LinearProbe

입력: ICP + ECG/ABP/PPG 윈도우 → encoder → mean pool → LinearProbe
라벨: 미래 구간 ICP > 20mmHg ≥1분 지속 여부

사용법:
    python -m downstream.classification.intracranial_hypertension.run \
        --checkpoint best.pt \
        --data-path datasets/processed/ich/ich_icp_w60s_h5min.pt \
        --mode linear_probe --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import get_global_spatial_id

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0

SIGNAL_TYPE_INT: dict[str, int] = {
    "ecg": 0,
    "abp": 1,
    "ppg": 2,
    "cvp": 3,
    "pap": 6,
    "icp": 7,
}


# ── 배치 생성 ─────────────────────────────────────────────────


def _make_samples(
    signals: dict[str, np.ndarray],
    idx: int,
) -> list[BiosignalSample]:
    samples = []
    for ch, (sig_type, signal) in enumerate(signals.items()):
        stype_int = SIGNAL_TYPE_INT.get(sig_type, 0)
        spatial_id = get_global_spatial_id(stype_int, 0)
        samples.append(
            BiosignalSample(
                values=torch.from_numpy(signal).float(),
                length=len(signal),
                channel_idx=ch,
                recording_idx=idx,
                sampling_rate=DEFAULT_SR,
                n_channels=len(signals),
                win_start=0,
                signal_type=stype_int,
                session_id=f"ich_{idx}",
                spatial_id=spatial_id,
            )
        )
    return samples


def _make_batches(
    windows: list[dict],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    multi = any(len(w["signals"]) > 1 for w in windows)
    collate_mode = "any_variate" if multi else "ci"
    collate = PackCollate(
        max_length=max_length, collate_mode=collate_mode, patch_size=patch_size
    )

    batches = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i: i + batch_size]
        all_samples = []
        for j, w in enumerate(chunk):
            all_samples.extend(_make_samples(w["signals"], idx=i + j))
        labels = torch.tensor([w["label"] for w in chunk], dtype=torch.float32)
        batch = collate(all_samples)
        batches.append((batch, labels))
    return batches


# ── Mean pooling ─────────────────────────────────────────────


def _mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# ── Linear Probe ─────────────────────────────────────────────


def train_linear_probe(model, probe, train_batches, epochs, lr, device):
    probe = probe.to(device)
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for batch, labels in train_batches:
            with torch.no_grad():
                features = model.extract_features(batch, pool="mean").to(device)
            logits = probe(features)
            loss = criterion(logits, labels.to(device).unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


@torch.no_grad()
def evaluate_linear_probe(model, probe, test_batches, device):
    probe.to(device).eval()
    all_labels, all_scores = [], []
    for batch, labels in test_batches:
        features = model.extract_features(batch, pool="mean").to(device)
        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
    return _compute_metrics(np.concatenate(all_labels), np.concatenate(all_scores))


# ── LoRA ─────────────────────────────────────────────────────


def train_lora(model, probe, train_batches, epochs, lr, device, gradient_clip=1.0):
    model.model.train()
    probe = probe.to(device)
    probe.train()
    lora_params = model.lora_parameters()
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr},
        {"params": probe.parameters(), "lr": lr},
    ], weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for batch, labels in train_batches:
            batch = model.batch_to_device(batch)
            out = model.model(batch, task="masked")
            features = _mean_pool(out["encoded"], out["patch_mask"])
            logits = probe(features)
            loss = criterion(logits, labels.to(device).unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                lora_params + list(probe.parameters()), gradient_clip,
            )
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


@torch.no_grad()
def evaluate_lora(model, probe, test_batches, device):
    model.model.eval()
    probe.to(device).eval()
    all_labels, all_scores = [], []
    for batch, labels in test_batches:
        batch = model.batch_to_device(batch)
        out = model.model(batch, task="masked")
        features = _mean_pool(out["encoded"], out["patch_mask"])
        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
    return _compute_metrics(np.concatenate(all_labels), np.concatenate(all_scores))


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


def _load_data(args):
    if not args.data_path or not Path(args.data_path).exists():
        print("ERROR: --data-path required", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading data: {args.data_path}")
    data = torch.load(args.data_path, weights_only=False)
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    print(f"  Signals: {meta.get('input_signals', '?')}")
    print(f"  Window: {meta.get('window_sec', '?')}s, "
          f"Horizon: {meta.get('horizon_sec', 0) / 60:.0f}min")

    def _to_windows(split_data):
        windows = []
        labels = split_data["labels"]
        sig_types = list(split_data["signals"].keys())
        for i in range(len(labels)):
            signals = {st: split_data["signals"][st][i].numpy() for st in sig_types}
            windows.append({
                "signals": signals,
                "label": int(labels[i].item()),
                "case_id": split_data["case_ids"][i] if "case_ids" in split_data else 0,
            })
        return windows

    return _to_windows(data["train"]), _to_windows(data["test"])


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Intracranial Hypertension Detection (ICP > 20mmHg)"
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    from downstream.model_wrapper import DownstreamModelWrapper

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
    d_model = model.d_model

    if args.mode == "lora":
        model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    train_windows, test_windows = _load_data(args)

    n_pos_train = sum(1 for w in train_windows if w["label"] == 1)
    n_pos_test = sum(1 for w in test_windows if w["label"] == 1)
    print(f"  Train: {len(train_windows)} ({n_pos_train} ICH, "
          f"{n_pos_train / max(len(train_windows), 1) * 100:.1f}%)")
    print(f"  Test:  {len(test_windows)} ({n_pos_test} ICH, "
          f"{n_pos_test / max(len(test_windows), 1) * 100:.1f}%)")

    first_sig = next(iter(train_windows[0]["signals"].values()))
    max_length = len(first_sig)
    train_batches = _make_batches(train_windows, args.batch_size, args.patch_size, max_length)
    test_batches = _make_batches(test_windows, args.batch_size, args.patch_size, max_length)

    probe = LinearProbe(d_model, n_classes=1)

    if args.mode == "linear_probe":
        print(f"\nTraining LinearProbe (d_model={d_model})...")
        train_losses = train_linear_probe(model, probe, train_batches, args.epochs, args.lr, device)
        metrics = evaluate_linear_probe(model, probe, test_batches, device)
    else:
        n_lora = sum(p.numel() for p in model.lora_parameters())
        print(f"\nTraining LoRA + Probe (rank={args.lora_rank}, LoRA={n_lora:,})...")
        train_losses = train_lora(model, probe, train_batches, args.epochs, args.lr, device)
        metrics = evaluate_lora(model, probe, test_batches, device)

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 60}")
    print(f"  Intracranial Hypertension Detection - {args.mode}")
    print(f"{'=' * 60}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Prevalence:  {metrics['prevalence']:.3f} "
          f"({metrics['n_positive']}/{metrics['n_total']})")
    print(f"{'=' * 60}")

    roc_path = out_dir / f"ich_roc_{args.mode}.png"
    plot_roc_curve(y_true, y_score, roc_path,
                   title=f"Intracranial Hypertension - {args.mode} ROC")
    print(f"\nROC curve: {roc_path}")

    results = {
        **metrics, "train_losses": train_losses,
        "config": {
            "task": "intracranial_hypertension_detection",
            "mode": args.mode,
            "data_path": args.data_path,
            "epochs": args.epochs, "lr": args.lr,
        },
    }
    results_path = out_dir / f"ich_results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
