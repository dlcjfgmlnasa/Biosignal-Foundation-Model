# -*- coding:utf-8 -*-
"""Arrhythmia Detection (PTB-XL 5-class superclass).

Frozen encoder + LinearProbe로 ECG 10s 윈도우를 5-class 분류한다.
  - NORM(0): Normal ECG
  - MI(1):   Myocardial Infarction
  - STTC(2): ST/T Change
  - CD(3):   Conduction Disturbance
  - HYP(4):  Hypertrophy

데이터는 prepare_data.py로 생성한 .pt 파일을 사용한다.

사용법:
    # 더미 모드 (파이프라인 검증)
    python -m downstream.arrhythmia.run --dummy \
        --data-path outputs/downstream/arrhythmia/arrhythmia_ptbxl_II.pt

    # 실제 checkpoint
    python -m downstream.arrhythmia.run \
        --checkpoint checkpoints/best.pt \
        --data-path outputs/downstream/arrhythmia/arrhythmia_ptbxl_II.pt
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

from downstream.metrics import compute_auroc, compute_auprc, compute_f1
from downstream.model_wrapper import LinearProbe


# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0
N_CLASSES = 5
CLASS_NAMES = {0: "NORM", 1: "MI", 2: "STTC", 3: "CD", 4: "HYP"}


# ── 데이터 로딩 ──────────────────────────────────────────────────


def load_prepared_data(
    data_path: str | Path,
    max_samples: int = 0,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """prepare_data.py가 생성한 .pt 파일을 로드한다.

    Parameters
    ----------
    data_path : .pt 파일 경로.
    max_samples : 0이면 전체, >0이면 각 split에서 최대 N개.

    Returns
    -------
    {"train": (signals, labels), "val": (signals, labels), "test": (signals, labels)}
    """
    data = torch.load(data_path, map_location="cpu", weights_only=False)

    metadata = data["metadata"]
    print(f"  Task: {metadata['task']}")
    print(f"  Source: {metadata['source']}, Lead: {metadata['lead']}")
    print(f"  SR: {metadata['sampling_rate']}Hz, Length: {metadata['signal_length']}")
    print(f"  Classes: {metadata['class_names']}")

    splits = {}
    for split_name in ["train", "val", "test"]:
        signals = data[split_name]["signals"]  # (N, 1000)
        labels = data[split_name]["labels"]    # (N,)

        if max_samples > 0 and len(signals) > max_samples:
            # 클래스 비율 유지하면서 서브샘플링
            rng = np.random.default_rng(42)
            idx = rng.choice(len(signals), size=max_samples, replace=False)
            idx.sort()
            signals = signals[idx]
            labels = labels[idx]

        splits[split_name] = (signals, labels)

    return splits


# ── 배치 생성 ─────────────────────────────────────────────────


def _signals_to_batches(
    signals: torch.Tensor,  # (N, signal_len)
    labels: torch.Tensor,   # (N,)
    batch_size: int,
    patch_size: int,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    """ECG 텐서를 PackedBatch + labels 리스트로 변환한다."""
    max_length = signals.shape[1]
    collate = PackCollate(max_length=max_length, collate_mode="ci", patch_size=patch_size)

    # ECG signal_type=0, spatial_id는 ecg lead II
    spatial_id = get_global_spatial_id("ecg", 0)

    batches = []
    for i in range(0, len(signals), batch_size):
        batch_signals = signals[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        samples = []
        for j, sig in enumerate(batch_signals):
            samples.append(BiosignalSample(
                values=sig.float(),
                length=len(sig),
                channel_idx=0,
                recording_idx=i + j,
                sampling_rate=DEFAULT_SR,
                n_channels=1,
                win_start=0,
                signal_type=0,  # ecg
                session_id=f"ptbxl_{i + j}",
                spatial_id=spatial_id,
            ))

        batch = collate(samples)
        batches.append((batch, batch_labels))

    return batches


# ── 더미 feature 추출기 ──────────────────────────────────────


class DummyFeatureExtractor:
    def __init__(self, d_model: int = 128):
        self.d_model = d_model
        self.device = torch.device("cpu")

    def extract_features(self, batch: PackedBatch, pool: str = "mean") -> torch.Tensor:
        B = batch.values.shape[0]
        return torch.randn(B, self.d_model)


# ── 학습 ─────────────────────────────────────────────────────


def train_probe(
    model,
    probe: LinearProbe,
    train_batches: list[tuple[PackedBatch, torch.Tensor]],
    val_batches: list[tuple[PackedBatch, torch.Tensor]] | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict[str, list[float]]:
    """LinearProbe를 학습한다.

    Returns
    -------
    {"train_losses": [...], "val_losses": [...]}
    """
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history: dict[str, list[float]] = {"train_losses": [], "val_losses": []}

    for epoch in range(epochs):
        # Train
        probe.train()
        epoch_loss, n = 0.0, 0
        for batch, labels in train_batches:
            with torch.no_grad():
                features = model.extract_features(batch, pool="mean").to(device)
            logits = probe(features)  # (B, 5)
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n += 1

        train_avg = epoch_loss / max(n, 1)
        history["train_losses"].append(train_avg)

        # Val
        val_avg = 0.0
        if val_batches:
            probe.eval()
            val_loss, vn = 0.0, 0
            with torch.no_grad():
                for batch, labels in val_batches:
                    features = model.extract_features(batch, pool="mean").to(device)
                    logits = probe(features)
                    loss = criterion(logits, labels.to(device))
                    val_loss += loss.item()
                    vn += 1
            val_avg = val_loss / max(vn, 1)
        history["val_losses"].append(val_avg)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            msg = f"  Epoch {epoch + 1}/{epochs}  train_loss={train_avg:.4f}"
            if val_batches:
                msg += f"  val_loss={val_avg:.4f}"
            print(msg)

    return history


# ── 평가 ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_probe(
    model,
    probe: LinearProbe,
    test_batches: list[tuple[PackedBatch, torch.Tensor]],
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Test set 평가. AUROC, AUPRC, F1, confusion matrix 등."""
    probe = probe.to(device)
    probe.eval()

    all_labels: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for batch, labels in test_batches:
        features = model.extract_features(batch, pool="mean").to(device)
        logits = probe(features)  # (B, 5)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)  # (N, 5)
    y_pred = y_prob.argmax(axis=1)

    # AUROC (one-vs-rest per class)
    auroc_per_class = compute_auroc(y_true, y_prob)
    if isinstance(auroc_per_class, dict):
        auroc_macro = float(np.mean(list(auroc_per_class.values())))
        auroc_display = {CLASS_NAMES[c]: v for c, v in auroc_per_class.items()}
    else:
        auroc_macro = float(auroc_per_class)
        auroc_display = {}

    # AUPRC (one-vs-rest per class)
    auprc_per_class = compute_auprc(y_true, y_prob)
    if isinstance(auprc_per_class, dict):
        auprc_macro = float(np.mean(list(auprc_per_class.values())))
        auprc_display = {CLASS_NAMES[c]: v for c, v in auprc_per_class.items()}
    else:
        auprc_macro = float(auprc_per_class)
        auprc_display = {}

    # F1
    f1_macro = compute_f1(y_true, y_pred, average="macro")
    f1_weighted = compute_f1(y_true, y_pred, average="weighted")

    # Overall accuracy
    accuracy = float((y_pred == y_true).mean()) if len(y_true) > 0 else 0.0

    # Per-class accuracy
    per_class_acc = {}
    for c in range(N_CLASSES):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_acc[CLASS_NAMES[c]] = float((y_pred[mask] == c).mean())
        else:
            per_class_acc[CLASS_NAMES[c]] = 0.0

    # Confusion matrix (5x5)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # Class distribution
    class_dist = {CLASS_NAMES[c]: int((y_true == c).sum()) for c in range(N_CLASSES)}

    return {
        "accuracy": accuracy,
        "auroc_per_class": auroc_display,
        "auroc_macro": auroc_macro,
        "auprc_per_class": auprc_display,
        "auprc_macro": auprc_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_accuracy": per_class_acc,
        "confusion_matrix": cm.tolist(),
        "class_distribution": class_dist,
        "n_total": len(y_true),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ── 시각화 ──────────────────────────────────────────────────


def plot_confusion_matrix(
    cm: list[list[int]],
    save_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """5x5 confusion matrix를 시각화한다."""
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    im = ax.imshow(cm_arr, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)

    classes = [CLASS_NAMES[i] for i in range(N_CLASSES)]
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(classes, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(classes, fontsize=9)

    thresh = cm_arr.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = "white" if cm_arr[i, j] > thresh else "black"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=color)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_auroc(
    auroc_per_class: dict[str, float],
    save_path: Path,
    title: str = "Per-class AUROC",
) -> None:
    """클래스별 AUROC 바 차트."""
    if not auroc_per_class:
        return

    classes = list(auroc_per_class.keys())
    values = list(auroc_per_class.values())
    macro = float(np.mean(values))

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6", "#3498db"]
    bars = ax.bar(classes, values, color=colors[:len(classes)], edgecolor="black", linewidth=0.5)

    # Macro 라인
    ax.axhline(y=macro, color="black", linestyle="--", linewidth=1.0, label=f"Macro={macro:.3f}")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=9)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUROC", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arrhythmia Detection: PTB-XL 5-class ECG Classification",
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="사전학습 checkpoint 경로")
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--data-path", type=str,
                        default="outputs/downstream/arrhythmia/arrhythmia_ptbxl_II.pt",
                        help="prepare_data.py가 생성한 .pt 파일 경로")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="각 split 최대 샘플 수 (0=전체)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--dummy", action="store_true",
                        help="더미 feature extractor로 파이프라인 검증")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── 모델 로드 ──
    if args.dummy:
        print("Using dummy feature extractor (pipeline verification)")
        model = DummyFeatureExtractor(d_model=128)
        d_model = 128
    elif args.checkpoint:
        from downstream.model_wrapper import DownstreamModelWrapper
        print(f"Loading checkpoint: {args.checkpoint}")
        model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
        d_model = model.d_model
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    # ── 데이터 로드 ──
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}", file=sys.stderr)
        print("  Run prepare_data.py first:")
        print("    python -m downstream.arrhythmia.prepare_data --download")
        sys.exit(1)

    print(f"\nLoading prepared data: {data_path}")
    splits = load_prepared_data(data_path, max_samples=args.max_samples)

    for split_name, (signals, labels) in splits.items():
        n = len(labels)
        if n == 0:
            print(f"  {split_name}: 0 samples")
            continue
        dist = {CLASS_NAMES[c]: int((labels == c).sum()) for c in range(N_CLASSES)}
        print(f"  {split_name}: {n} samples  {dist}")

    # ── 배치 생성 ──
    print(f"\nCreating batches (batch_size={args.batch_size}, patch_size={args.patch_size})...")
    train_signals, train_labels = splits["train"]
    val_signals, val_labels = splits["val"]
    test_signals, test_labels = splits["test"]

    if len(train_signals) == 0 or len(test_signals) == 0:
        print("ERROR: Insufficient data.", file=sys.stderr)
        sys.exit(1)

    train_batches = _signals_to_batches(train_signals, train_labels, args.batch_size, args.patch_size)
    val_batches = _signals_to_batches(val_signals, val_labels, args.batch_size, args.patch_size) if len(val_signals) > 0 else None
    test_batches = _signals_to_batches(test_signals, test_labels, args.batch_size, args.patch_size)

    print(f"  Train: {len(train_batches)} batches, Val: {len(val_batches) if val_batches else 0} batches, Test: {len(test_batches)} batches")

    # ── Probe 학습 ──
    probe = LinearProbe(d_model, n_classes=N_CLASSES)
    print(f"\nTraining LinearProbe (d_model={d_model}, {N_CLASSES} classes, epochs={args.epochs})...")
    history = train_probe(
        model, probe, train_batches,
        val_batches=val_batches,
        epochs=args.epochs, lr=args.lr, device=device,
    )

    # ── 평가 ──
    print("\nEvaluating on test set...")
    metrics = evaluate_probe(model, probe, test_batches, device=device)

    y_true = metrics.pop("y_true")
    y_pred = metrics.pop("y_pred")
    y_prob = metrics.pop("y_prob")

    # ── 결과 출력 ──
    print(f"\n{'='*55}")
    print(f"  Arrhythmia Detection (PTB-XL 5-class)")
    print(f"{'='*55}")
    print(f"  Accuracy:       {metrics['accuracy']:.4f}")
    print(f"  AUROC (macro):  {metrics['auroc_macro']:.4f}")
    print(f"  AUPRC (macro):  {metrics['auprc_macro']:.4f}")
    print(f"  F1 (macro):     {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):  {metrics['f1_weighted']:.4f}")
    print(f"  ---")
    for cls_name in CLASS_NAMES.values():
        auroc = metrics["auroc_per_class"].get(cls_name, 0.0)
        acc = metrics["per_class_accuracy"].get(cls_name, 0.0)
        n = metrics["class_distribution"].get(cls_name, 0)
        print(f"  {cls_name:5s}: AUROC={auroc:.4f}  acc={acc:.4f}  n={n}")
    print(f"{'='*55}")

    # ── 시각화 ──
    cm_path = out_dir / "arrhythmia_confusion_matrix.png"
    plot_confusion_matrix(
        metrics["confusion_matrix"], cm_path,
        title="Arrhythmia Detection (PTB-XL) — Confusion Matrix",
    )
    print(f"\nConfusion matrix saved: {cm_path}")

    auroc_path = out_dir / "arrhythmia_auroc_per_class.png"
    plot_per_class_auroc(
        metrics["auroc_per_class"], auroc_path,
        title="Arrhythmia Detection — Per-class AUROC",
    )
    print(f"Per-class AUROC saved: {auroc_path}")

    # ── 결과 저장 ──
    results = {
        **metrics,
        **history,
        "config": {
            "data_path": str(args.data_path),
            "max_samples": args.max_samples,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "n_train": len(train_labels),
            "n_val": len(val_labels),
            "n_test": len(test_labels),
        },
    }
    results_path = out_dir / "arrhythmia_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()