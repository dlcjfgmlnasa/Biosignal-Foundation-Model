# -*- coding:utf-8 -*-
"""Task 1: Hypotension Prediction (MAP < 65 mmHg).

Frozen encoder + LinearProbe로 ABP 윈도우의 저혈압 여부를 분류한다.

사용법:
    # 더미 테스트 (random features, 파이프라인 검증)
    python -m downstream.task1_hypotension.run --dummy --n-cases 3

    # 실제 평가
    python -m downstream.task1_hypotension.run \
        --checkpoint checkpoints/best.pt --n-cases 30 --epochs 20
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
from data.parser.vitaldb import SIGNAL_TYPES

from downstream.common.data_utils import (
    CaseData,
    LabeledWindow,
    Window,
    extract_windows,
    apply_pipeline,
    create_labeled_dataset_hypotension,
    load_pilot_cases,
    split_by_subject,
)
from downstream.common.eval_utils import (
    compute_auroc,
    compute_auprc,
    compute_sensitivity_specificity,
    plot_roc_curve,
)
from downstream.common.model_wrapper import LinearProbe


# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0


# ── 배치 생성 ─────────────────────────────────────────────────


def _labeled_windows_to_samples(
    labeled: list[LabeledWindow],
) -> list[BiosignalSample]:
    """LabeledWindow -> BiosignalSample 변환."""
    samples = []
    for i, lw in enumerate(labeled):
        stype_int = SIGNAL_TYPES.get(lw.signal_type, 1)
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
    """LabeledWindow 리스트를 (batch, labels) 튜플 리스트로 변환."""
    batches = []
    collate = PackCollate(max_length=max_length, collate_mode="ci", patch_size=patch_size)

    for i in range(0, len(labeled), batch_size):
        chunk = labeled[i:i + batch_size]
        samples = _labeled_windows_to_samples(chunk)
        labels = torch.tensor([lw.label for lw in chunk], dtype=torch.float32)
        batch = collate(samples)
        batches.append((batch, labels))

    return batches


# ── 더미 feature 추출기 ──────────────────────────────────────


class DummyFeatureExtractor:
    """Random features for pipeline verification."""

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
    """LinearProbe를 학습한다 (encoder frozen).

    Returns
    -------
    list[float] -- epoch별 train loss.
    """
    probe = probe.to(device)
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    losses: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch, labels in train_batches:
            # Extract frozen features
            with torch.no_grad():
                features = model.extract_features(batch, pool="mean")  # (B, d_model)
                features = features.to(device)

            labels_dev = labels.to(device).unsqueeze(-1)  # (B, 1)

            logits = probe(features)  # (B, 1)
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
    """Probe를 평가한다.

    Returns
    -------
    dict with: auroc, auprc, sensitivity, specificity, y_true, y_score.
    """
    probe = probe.to(device)
    probe.eval()

    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for batch, labels in test_batches:
        features = model.extract_features(batch, pool="mean").to(device)
        logits = probe(features)  # (B, 1)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()

        all_labels.append(labels.numpy())
        all_scores.append(probs)

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)

    auroc = compute_auroc(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)

    # Optimal threshold (Youden's J)
    best_thresh = 0.5
    best_j = -1.0
    for thresh in np.linspace(0.01, 0.99, 99):
        y_pred = (y_score >= thresh).astype(int)
        ss = compute_sensitivity_specificity(y_true, y_pred)
        j = ss["sensitivity"] + ss["specificity"] - 1.0
        if j > best_j:
            best_j = j
            best_thresh = thresh

    y_pred_opt = (y_score >= best_thresh).astype(int)
    ss_opt = compute_sensitivity_specificity(y_true, y_pred_opt)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "optimal_threshold": float(best_thresh),
        "sensitivity": ss_opt["sensitivity"],
        "specificity": ss_opt["specificity"],
        "n_total": len(y_true),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "y_true": y_true,
        "y_score": y_score,
    }


# ── メイン ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: Hypotension Prediction")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-sec", type=float, default=10.0, help="ABP 윈도우 길이 (초)")
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
    print(f"\nLoading {args.n_cases} pilot cases (ABP only)...")
    cases = load_pilot_cases(
        n_cases=args.n_cases,
        offset_from_end=200,
        signal_types=["abp"],
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
            wins = extract_windows(case, "abp", args.window_sec, args.stride_sec)
            windows.extend(apply_pipeline(wins))
        return create_labeled_dataset_hypotension(windows)

    train_labeled = _extract_labeled(train_cases)
    test_labeled = _extract_labeled(test_cases)

    n_pos_train = sum(1 for lw in train_labeled if lw.label == 1)
    n_pos_test = sum(1 for lw in test_labeled if lw.label == 1)
    print(f"Train: {len(train_labeled)} windows ({n_pos_train} hypotensive, {n_pos_train / max(len(train_labeled), 1) * 100:.1f}%)")
    print(f"Test:  {len(test_labeled)} windows ({n_pos_test} hypotensive, {n_pos_test / max(len(test_labeled), 1) * 100:.1f}%)")

    if len(train_labeled) == 0 or len(test_labeled) == 0:
        print("Insufficient data for train/test.", file=sys.stderr)
        sys.exit(1)

    # ── 배치 생성 ──
    train_batches = _make_batches(train_labeled, args.batch_size, args.patch_size, max_length)
    test_batches = _make_batches(test_labeled, args.batch_size, args.patch_size, max_length)

    # ── Probe 학습 ──
    print(f"\nTraining LinearProbe (d_model={d_model}, epochs={args.epochs})...")
    probe = LinearProbe(d_model, n_classes=1)
    train_losses = train_probe(model, probe, train_batches, epochs=args.epochs, lr=args.lr, device=device)

    # ── 평가 ──
    print("\nEvaluating...")
    metrics = evaluate_probe(model, probe, test_batches, device=device)

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'='*50}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  Threshold:   {metrics['optimal_threshold']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Prevalence:  {metrics['prevalence']:.3f} ({metrics['n_positive']}/{metrics['n_total']})")
    print(f"{'='*50}")

    # ── ROC curve 시각화 ──
    roc_path = out_dir / "task1_roc_curve.png"
    plot_roc_curve(y_true, y_score, roc_path, title="Task 1: Hypotension Prediction — ROC Curve")
    print(f"\nROC curve saved: {roc_path}")

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
    results_path = out_dir / "task1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
