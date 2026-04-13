# -*- coding:utf-8 -*-
"""Atrial Fibrillation Detection — Frozen encoder + probe 평가.

ECG 윈도우의 AF 여부를 분류한다.

사용법:
    # Prepared .pt 파일 사용
    python -m downstream.classification.atrial_fibrillation.run \
        --data-path outputs/downstream/atrial_fibrillation/af_detection_ecg.pt \
        --dummy

    # 로컬 .pt 디렉토리에서 직접 로딩
    python -m downstream.classification.atrial_fibrillation.run \
        --data-dir vitaldb_pt_test --dummy
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

from downstream.data_utils import LabeledWindow
from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0


# ── 배치 생성 ─────────────────────────────────────────────────


def _labeled_windows_to_samples(labeled: list[LabeledWindow]) -> list[BiosignalSample]:
    samples = []
    for i, lw in enumerate(labeled):
        stype_int = SIGNAL_TYPES.get(lw.signal_type, 0)
        spatial_id = get_global_spatial_id(lw.signal_type, 0)
        samples.append(
            BiosignalSample(
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
            )
        )
    return samples


def _make_batches(
    labeled: list[LabeledWindow],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    batches = []
    collate = PackCollate(
        max_length=max_length, collate_mode="ci", patch_size=patch_size
    )
    for i in range(0, len(labeled), batch_size):
        chunk = labeled[i : i + batch_size]
        samples = _labeled_windows_to_samples(chunk)
        labels = torch.tensor([lw.label for lw in chunk], dtype=torch.float32)
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


# ── 학습/평가 ────────────────────────────────────────────────


def train_probe(
    model,
    probe: nn.Module,
    train_batches: list,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
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
def evaluate_probe(
    model,
    probe: nn.Module,
    test_batches: list,
    device: torch.device,
) -> dict:
    probe.to(device).eval()
    all_labels, all_scores = [], []
    for batch, labels in test_batches:
        features = model.extract_features(batch, pool="mean").to(device)
        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)

    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores)

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

    return {
        "auroc": auroc,
        "auprc": auprc,
        "optimal_threshold": float(best_thresh),
        "sensitivity": ss_opt["sensitivity"],
        "specificity": ss_opt["specificity"],
        "n_total": len(y_true),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if len(y_true) > 0 else 0.0,
        "y_true": y_true,
        "y_score": y_score,
    }


# ── 메인 ─────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atrial Fibrillation Detection"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--signal-type",
        type=str,
        default="ecg",
        choices=["ecg", "ppg", "abp"],
        help="Input signal type for peak-based AF detection",
    )
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
        from downstream.model_wrapper import DownstreamModelWrapper

        print(f"Loading checkpoint: {args.checkpoint}")
        model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
        d_model = model.d_model
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    # ── 데이터 로드 ──
    if args.data_path and Path(args.data_path).exists():
        print(f"\nLoading prepared data: {args.data_path}")
        data = torch.load(args.data_path, weights_only=False)
        meta = data.get("metadata", {})
        sig_type = meta.get("input_signal", args.signal_type)
        print(f"  Task: {meta.get('task', '?')}")
        print(f"  Signal: {sig_type.upper()}")
        print(f"  Window: {meta.get('window_sec', '?')}s")

        def _pt_to_labeled(split_data):
            labeled = []
            signals_t = split_data["signals"]  # (N, win_samples)
            labels_t = split_data["labels"]  # (N,)
            for i in range(len(labels_t)):
                labeled.append(
                    LabeledWindow(
                        signal=signals_t[i].numpy(),
                        signal_type=sig_type,
                        case_id=split_data["case_ids"][i]
                        if "case_ids" in split_data
                        else 0,
                        label=int(labels_t[i].item()),
                        label_value=0.0,
                    )
                )
            return labeled

        train_labeled = _pt_to_labeled(data["train"])
        test_labeled = _pt_to_labeled(data["test"])

    elif args.data_dir and Path(args.data_dir).is_dir():
        from downstream.classification.atrial_fibrillation.prepare_data import (
            _load_signal_segments,
            extract_af_samples,
        )

        sig_type = args.signal_type
        print(f"\nLoading {sig_type.upper()} from local .pt directory: {args.data_dir}")
        min_dur = args.window_sec + args.stride_sec
        segments = _load_signal_segments(
            args.data_dir, sig_type, min_dur, max_subjects=args.max_subjects
        )
        if not segments:
            print(f"No valid {sig_type.upper()} segments loaded.", file=sys.stderr)
            sys.exit(1)

        # Patient-level split
        rng = np.random.default_rng(42)
        patient_ids = list({s["patient_id"] for s in segments})
        rng.shuffle(patient_ids)
        n_train_p = max(1, int(len(patient_ids) * args.train_ratio))
        train_pats = set(patient_ids[:n_train_p])

        train_segs = [s for s in segments if s["patient_id"] in train_pats]
        test_segs = [s for s in segments if s["patient_id"] not in train_pats]

        train_af = extract_af_samples(train_segs, args.window_sec, args.stride_sec)
        test_af = extract_af_samples(test_segs, args.window_sec, args.stride_sec)

        train_labeled = [
            LabeledWindow(
                signal=s.signal,
                signal_type=sig_type,
                case_id=s.case_id,
                label=s.label,
                label_value=s.beat_cv,
            )
            for s in train_af
        ]
        test_labeled = [
            LabeledWindow(
                signal=s.signal,
                signal_type=sig_type,
                case_id=s.case_id,
                label=s.label,
                label_value=s.beat_cv,
            )
            for s in test_af
        ]
    else:
        print("ERROR: --data-path or --data-dir required.", file=sys.stderr)
        sys.exit(1)

    n_pos_train = sum(1 for lw in train_labeled if lw.label == 1)
    n_pos_test = sum(1 for lw in test_labeled if lw.label == 1)
    print(
        f"  Train: {len(train_labeled)} samples "
        f"({n_pos_train} AF, {n_pos_train / max(len(train_labeled), 1) * 100:.1f}%)"
    )
    print(
        f"  Test:  {len(test_labeled)} samples "
        f"({n_pos_test} AF, {n_pos_test / max(len(test_labeled), 1) * 100:.1f}%)"
    )

    if not train_labeled or not test_labeled:
        print("Insufficient data.", file=sys.stderr)
        sys.exit(1)

    max_length = len(train_labeled[0].signal)
    train_batches = _make_batches(
        train_labeled, args.batch_size, args.patch_size, max_length
    )
    test_batches = _make_batches(
        test_labeled, args.batch_size, args.patch_size, max_length
    )

    # ── 학습 ──
    probe = LinearProbe(d_model, n_classes=1)
    print(f"\nTraining LinearProbe (d_model={d_model})...")
    train_losses = train_probe(
        model, probe, train_batches, args.epochs, args.lr, device
    )

    # ── 평가 ──
    print("\nEvaluating...")
    metrics = evaluate_probe(model, probe, test_batches, device)

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 50}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  Threshold:   {metrics['optimal_threshold']:.3f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(
        f"  Prevalence:  {metrics['prevalence']:.3f} "
        f"({metrics['n_positive']}/{metrics['n_total']})"
    )
    print(f"{'=' * 50}")

    roc_path = out_dir / "af_roc_curve.png"
    plot_roc_curve(
        y_true, y_score, roc_path,
        title="Atrial Fibrillation Detection — ROC Curve",
    )
    print(f"\nROC curve saved: {roc_path}")

    results = {
        **metrics,
        "train_losses": train_losses,
        "config": {
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
        },
    }
    results_path = out_dir / "af_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
