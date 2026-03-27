# -*- coding:utf-8 -*-
"""Task 2: Bradycardia/Tachycardia Detection.

Frozen encoder + LinearProbe로 ECG 윈도우의 심박 리듬을 3-class 분류한다.
  - 0: Bradycardia (HR < 60 bpm)
  - 1: Normal (60 <= HR <= 100 bpm)
  - 2: Tachycardia (HR > 100 bpm)

사용법:
    python -m downstream.task2_bradytachy.run --dummy --n-cases 5
    python -m downstream.task2_bradytachy.run --checkpoint best.pt --n-cases 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import get_global_spatial_id
from data.parser.vitaldb import SIGNAL_TYPES

from downstream.common.data_utils import (
    CaseData,
    LabeledWindow,
    Window,
    extract_windows,
    apply_pipeline,
    create_labeled_dataset_bradytachy,
    load_pilot_cases,
    split_by_subject,
)
from downstream.common.eval_utils import compute_auroc, compute_f1
from downstream.common.model_wrapper import LinearProbe


# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0
CLASS_NAMES = {0: "Bradycardia", 1: "Normal", 2: "Tachycardia"}


# ── 배치 생성 ─────────────────────────────────────────────────


def _labeled_windows_to_samples(labeled: list[LabeledWindow]) -> list[BiosignalSample]:
    samples = []
    for i, lw in enumerate(labeled):
        stype_int = SIGNAL_TYPES.get(lw.signal_type, 0)
        spatial_id = get_global_spatial_id(lw.signal_type, 0)
        samples.append(BiosignalSample(
            values=torch.from_numpy(lw.signal).float(),
            length=len(lw.signal),
            channel_idx=0,
            recording_idx=i,
            sampling_rate=DEFAULT_SR,
            n_channels=1,
            win_start=0,
            signal_type=stype_int,
            session_id=f"case_{lw.case_id}",
            spatial_id=spatial_id,
        ))
    return samples


def _make_batches(
    labeled: list[LabeledWindow],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    batches = []
    collate = PackCollate(max_length=max_length, collate_mode="ci", patch_size=patch_size)
    for i in range(0, len(labeled), batch_size):
        chunk = labeled[i:i + batch_size]
        samples = _labeled_windows_to_samples(chunk)
        labels = torch.tensor([lw.label for lw in chunk], dtype=torch.long)
        batch = collate(samples)
        batches.append((batch, labels))
    return batches


# ── 더미 feature 추출기 ──────────────────────────────────────


class DummyFeatureExtractor:
    def __init__(self, d_model: int = 128):
        self.d_model = d_model
        self.device = torch.device("cpu")

    def extract_features(self, batch: PackedBatch, pool: str = "mean") -> torch.Tensor:
        B = batch.values.shape[0]
        return torch.randn(B, self.d_model)


# ── 학습/평가 ─────────────────────────────────────────────────


def train_probe(
    model,
    probe: LinearProbe,
    train_batches: list[tuple[PackedBatch, torch.Tensor]],
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    probe = probe.to(device)
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch, labels in train_batches:
            with torch.no_grad():
                features = model.extract_features(batch, pool="mean").to(device)
            labels_dev = labels.to(device)

            logits = probe(features)  # (B, 3)
            loss = criterion(logits, labels_dev)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    return losses


@torch.no_grad()
def evaluate_probe(
    model,
    probe: LinearProbe,
    test_batches: list[tuple[PackedBatch, torch.Tensor]],
    device: torch.device = torch.device("cpu"),
) -> dict:
    probe = probe.to(device)
    probe.eval()

    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for batch, labels in test_batches:
        features = model.extract_features(batch, pool="mean").to(device)
        logits = probe(features)  # (B, 3)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)  # (N, 3)
    y_pred = y_prob.argmax(axis=1)

    # AUROC (one-vs-rest per class)
    auroc_per_class = compute_auroc(y_true, y_prob)
    auroc_macro = float(np.mean(list(auroc_per_class.values()))) if isinstance(auroc_per_class, dict) else auroc_per_class

    # F1
    f1_macro = compute_f1(y_true, y_pred, average="macro")
    f1_weighted = compute_f1(y_true, y_pred, average="weighted")

    # Per-class accuracy
    per_class_acc = {}
    for c in range(3):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[c]] = float((y_pred[mask] == c).mean())
        else:
            per_class_acc[CLASS_NAMES[c]] = 0.0

    # Confusion matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Class distribution
    class_dist = {CLASS_NAMES[c]: int((y_true == c).sum()) for c in range(3)}

    return {
        "auroc_per_class": {CLASS_NAMES[c]: v for c, v in auroc_per_class.items()} if isinstance(auroc_per_class, dict) else {},
        "auroc_macro": auroc_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "class_distribution": class_dist,
        "n_total": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ── Confusion matrix 시각화 ──────────────────────────────────


def plot_confusion_matrix(
    cm: list[list[int]],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    classes = [CLASS_NAMES[i] for i in range(3)]
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(classes, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(classes, fontsize=9)

    # 셀에 숫자 표시
    thresh = cm_arr.max() / 2.0
    for i in range(3):
        for j in range(3):
            color = "white" if cm_arr[i, j] > thresh else "black"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── メイン ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 2: Bradycardia/Tachycardia Detection")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=10.0)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    max_length = int(args.window_sec * DEFAULT_SR)

    # ── 모델 로드 ──
    if args.dummy:
        print("Using dummy feature extractor (pipeline verification)")
        model = DummyFeatureExtractor(d_model=128)
        d_model = 128
    elif args.checkpoint:
        from downstream.common.model_wrapper import DownstreamModelWrapper
        print(f"Loading checkpoint: {args.checkpoint}")
        model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
        d_model = model.d_model
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    # ── 데이터 로드 ──
    print(f"\nLoading {args.n_cases} pilot cases (ECG)...")
    cases = load_pilot_cases(
        n_cases=args.n_cases,
        offset_from_end=200,
        signal_types=["ecg"],
    )

    if not cases:
        print("No cases loaded.", file=sys.stderr)
        sys.exit(1)

    # ── Subject-level split ──
    train_cases, test_cases = split_by_subject(cases, train_ratio=args.train_ratio)
    print(f"Split: {len(train_cases)} train, {len(test_cases)} test cases")

    # ── 윈도우 추출 + 라벨링 ──
    def _extract_labeled(case_list: list[CaseData]) -> list[LabeledWindow]:
        windows: list[Window] = []
        for case in case_list:
            wins = extract_windows(case, "ecg", args.window_sec, args.stride_sec)
            windows.extend(apply_pipeline(wins))
        return create_labeled_dataset_bradytachy(windows)

    train_labeled = _extract_labeled(train_cases)
    test_labeled = _extract_labeled(test_cases)

    # Class distribution
    for split_name, labeled in [("Train", train_labeled), ("Test", test_labeled)]:
        dist = {CLASS_NAMES[c]: sum(1 for lw in labeled if lw.label == c) for c in range(3)}
        print(f"{split_name}: {len(labeled)} windows  {dist}")

    if len(train_labeled) == 0 or len(test_labeled) == 0:
        print("Insufficient data for train/test.", file=sys.stderr)
        sys.exit(1)

    # ── 배치 생성 ──
    train_batches = _make_batches(train_labeled, args.batch_size, args.patch_size, max_length)
    test_batches = _make_batches(test_labeled, args.batch_size, args.patch_size, max_length)

    # ── Probe 학습 ──
    print(f"\nTraining LinearProbe (d_model={d_model}, 3 classes, epochs={args.epochs})...")
    probe = LinearProbe(d_model, n_classes=3)
    train_losses = train_probe(model, probe, train_batches, epochs=args.epochs, lr=args.lr, device=device)

    # ── 평가 ──
    print("\nEvaluating...")
    metrics = evaluate_probe(model, probe, test_batches, device=device)

    y_true = metrics.pop("y_true")
    y_pred = metrics.pop("y_pred")

    print(f"\n{'='*50}")
    print(f"  AUROC (macro):  {metrics['auroc_macro']:.4f}")
    print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):  {metrics['f1_weighted']:.4f}")
    for cls_name, acc in metrics["per_class_accuracy"].items():
        print(f"  {cls_name:15s}: acc={acc:.4f}")
    print(f"  Distribution:   {metrics['class_distribution']}")
    print(f"{'='*50}")

    # ── Confusion matrix 시각화 ──
    cm_path = out_dir / "task2_confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"], cm_path,
        title="Task 2: Bradycardia/Tachycardia — Confusion Matrix",
    )
    print(f"\nConfusion matrix saved: {cm_path}")

    # ── 결과 저장 ──
    results = {
        **metrics,
        "train_losses": train_losses,
        "config": {
            "n_cases": args.n_cases,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
            "n_train": len(train_labeled),
            "n_test": len(test_labeled),
        },
    }
    results_path = out_dir / "task2_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
