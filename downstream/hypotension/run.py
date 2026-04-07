# -*- coding:utf-8 -*-
"""Task 1: Hypotension Prediction (MAP < 65 mmHg).

Frozen encoder + probe로 ABP 윈도우의 저혈압 여부를 분류한다.

모드:
  - 단일 윈도우: 현재 윈도우 feature -> probe -> 현재 MAP<65 분류
  - 다중 윈도우: N개 과거 윈도우 features -> aggregator -> 미래 MAP<65 예측

사용법:
    # 단일 윈도우 (기존 동작)
    python -m downstream.hypotension.run --dummy --n-cases 10

    # 다중 윈도우 + prediction horizon
    python -m downstream.hypotension.run --dummy --n-cases 10 \
        --n-context-windows 5 --prediction-horizon-sec 300 --aggregator lstm
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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
    TARGET_SR,
)
from eval._metrics import (
    compute_auroc,
    compute_auprc,
    compute_sensitivity_specificity,
    plot_roc_curve,
)
from downstream.common.model_wrapper import LinearProbe


# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0


# ── Aggregator probes ────────────────────────────────────────


class ConcatProbe(nn.Module):
    """N개 윈도우 feature를 concatenate -> Linear."""

    def __init__(self, d_model: int, n_windows: int, n_classes: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * n_windows),
            nn.Dropout(dropout_p),
            nn.Linear(d_model * n_windows, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # (B, n_windows, d_model)
        B = features.shape[0]
        flat = features.reshape(B, -1)  # (B, n_windows * d_model)
        return self.head(flat)  # (B, n_classes)


class LSTMProbe(nn.Module):
    """N개 윈도우 feature를 LSTM -> last hidden -> Linear."""

    def __init__(self, d_model: int, hidden_dim: int = 64, n_classes: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(d_model, hidden_dim, batch_first=True, num_layers=1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # (B, n_windows, d_model)
        _, (h_n, _) = self.lstm(features)  # h_n: (1, B, hidden_dim)
        return self.head(h_n.squeeze(0))  # (B, n_classes)


class MeanProbe(nn.Module):
    """N개 윈도우 feature를 mean pool -> Linear."""

    def __init__(self, d_model: int, n_classes: int = 1, dropout_p: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # (B, n_windows, d_model)
        pooled = features.mean(dim=1)  # (B, d_model)
        return self.head(pooled)  # (B, n_classes)


# ── Multi-window 데이터 생성 ─────────────────────────────────


@dataclass
class MultiWindowSample:
    """N개 연속 윈도우 + 라벨."""
    windows: list[LabeledWindow]   # context windows (시간순)
    label: int                     # target label (0=normal, 1=hypotension)
    label_value: float             # target MAP
    case_id: int


def create_multiwindow_samples(
    case_data: CaseData,
    n_context: int,
    window_sec: float,
    stride_sec: float,
    horizon_sec: float = 0.0,
    map_threshold: float = 65.0,
) -> list[MultiWindowSample]:
    """연속 N개 윈도우 + horizon 뒤의 라벨을 생성한다.

    Parameters
    ----------
    case_data: 로드된 케이스.
    n_context: 과거 context 윈도우 수.
    window_sec: 각 윈도우 길이 (초).
    stride_sec: 윈도우 간 보폭 (초).
    horizon_sec: prediction horizon (초). 0이면 마지막 context 윈도우로 라벨링.
    map_threshold: MAP 미만이면 hypotension.

    Returns
    -------
    list[MultiWindowSample]
    """
    if "abp" not in case_data.tracks:
        return []

    signal = case_data.tracks["abp"]
    sr = DEFAULT_SR
    win_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    horizon_samples = int(horizon_sec * sr)

    # 모든 유효 윈도우를 시간순으로 추출
    all_windows: list[LabeledWindow] = []
    wins = extract_windows(case_data, "abp", window_sec, stride_sec)
    good_wins = apply_pipeline(wins)
    labeled = create_labeled_dataset_hypotension(good_wins, map_threshold)

    if len(labeled) < n_context + (1 if horizon_sec > 0 else 0):
        return []

    # 시간순 정렬 (win_start 기준, 같은 case 내이므로 이미 정렬됨)
    samples: list[MultiWindowSample] = []

    if horizon_sec <= 0:
        # No horizon: 연속 N개 윈도우, 마지막 윈도우가 label
        for i in range(n_context - 1, len(labeled)):
            context = labeled[i - n_context + 1:i + 1]

            # 연속성 체크: stride 간격으로 정렬되어 있는지 확인
            # (extract_windows가 stride로 추출하므로 인덱스 연속이면 시간 연속)
            samples.append(MultiWindowSample(
                windows=context,
                label=context[-1].label,
                label_value=context[-1].label_value,
                case_id=case_data.case_id,
            ))
    else:
        # With horizon: context N개 + horizon 뒤의 윈도우에서 라벨
        horizon_windows = int(horizon_sec / stride_sec)  # horizon에 해당하는 윈도우 수
        total_needed = n_context + horizon_windows

        for i in range(n_context - 1, len(labeled) - horizon_windows):
            context = labeled[i - n_context + 1:i + 1]
            target_idx = i + horizon_windows

            if target_idx >= len(labeled):
                break

            target = labeled[target_idx]
            samples.append(MultiWindowSample(
                windows=context,
                label=target.label,
                label_value=target.label_value,
                case_id=case_data.case_id,
            ))

    return samples


# ── 배치 생성 ─────────────────────────────────────────────────


def _labeled_windows_to_samples(labeled: list[LabeledWindow]) -> list[BiosignalSample]:
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


def _make_single_batches(
    labeled: list[LabeledWindow],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    """단일 윈도우 모드: (batch, labels)."""
    batches = []
    collate = PackCollate(max_length=max_length, collate_mode="ci", patch_size=patch_size)
    for i in range(0, len(labeled), batch_size):
        chunk = labeled[i:i + batch_size]
        samples = _labeled_windows_to_samples(chunk)
        labels = torch.tensor([lw.label for lw in chunk], dtype=torch.float32)
        batch = collate(samples)
        batches.append((batch, labels))
    return batches


def _make_multi_batches(
    mw_samples: list[MultiWindowSample],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[list[PackedBatch], torch.Tensor]]:
    """다중 윈도우 모드: (list[batch_per_window], labels).

    각 시간 스텝의 윈도우를 별도 PackedBatch로 생성한다.
    """
    batches = []
    collate = PackCollate(max_length=max_length, collate_mode="ci", patch_size=patch_size)

    for i in range(0, len(mw_samples), batch_size):
        chunk = mw_samples[i:i + batch_size]
        n_ctx = len(chunk[0].windows)
        labels = torch.tensor([s.label for s in chunk], dtype=torch.float32)

        # 각 시간 스텝별로 배치 생성
        step_batches: list[PackedBatch] = []
        for t in range(n_ctx):
            step_windows = [s.windows[t] for s in chunk]
            step_samples = _labeled_windows_to_samples(step_windows)
            step_batch = collate(step_samples)
            step_batches.append(step_batch)

        batches.append((step_batches, labels))

    return batches


# ── 더미 feature 추출기 ──────────────────────────────────────


class DummyFeatureExtractor:
    def __init__(self, d_model: int = 128):
        self.d_model = d_model
        self.device = torch.device("cpu")

    def extract_features(self, batch: PackedBatch, pool: str = "mean") -> torch.Tensor:
        B = batch.values.shape[0]
        return torch.randn(B, self.d_model)


# ── 학습/평가 (단일 윈도우) ──────────────────────────────────


def train_probe_single(
    model, probe, train_batches, epochs, lr, device,
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
def evaluate_probe_single(model, probe, test_batches, device):
    probe.to(device).eval()
    all_labels, all_scores = [], []
    for batch, labels in test_batches:
        features = model.extract_features(batch, pool="mean").to(device)
        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
    return _compute_metrics(np.concatenate(all_labels), np.concatenate(all_scores))


# ── 학습/평가 (다중 윈도우) ──────────────────────────────────


def train_probe_multi(
    model, probe, train_batches, epochs, lr, device,
) -> list[float]:
    probe = probe.to(device)
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for step_batches, labels in train_batches:
            # Extract features for each time step
            step_features = []
            for step_batch in step_batches:
                with torch.no_grad():
                    feat = model.extract_features(step_batch, pool="mean").to(device)
                step_features.append(feat)

            features = torch.stack(step_features, dim=1)  # (B, n_windows, d_model)
            logits = probe(features)  # (B, 1)
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
def evaluate_probe_multi(model, probe, test_batches, device):
    probe.to(device).eval()
    all_labels, all_scores = [], []
    for step_batches, labels in test_batches:
        step_features = []
        for step_batch in step_batches:
            feat = model.extract_features(step_batch, pool="mean").to(device)
            step_features.append(feat)
        features = torch.stack(step_features, dim=1)
        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
    return _compute_metrics(np.concatenate(all_labels), np.concatenate(all_scores))


# ── 공통 메트릭 계산 ─────────────────────────────────────────


def _compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
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


# ── メイン ────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: Hypotension Prediction")
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
    # Multi-window args
    parser.add_argument("--n-context-windows", type=int, default=1,
                        help="과거 context 윈도우 수 (1=단일 윈도우, >1=다중)")
    parser.add_argument("--prediction-horizon-sec", type=float, default=0.0,
                        help="예측 horizon (초). 0=현재 윈도우 분류")
    parser.add_argument("--aggregator", type=str, default="concat",
                        choices=["concat", "lstm", "mean"],
                        help="다중 윈도우 집계 방식")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    max_length = int(args.window_sec * DEFAULT_SR)
    multi_mode = args.n_context_windows > 1

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

    mode_str = (
        f"Multi-window (n_ctx={args.n_context_windows}, "
        f"horizon={args.prediction_horizon_sec}s, agg={args.aggregator})"
        if multi_mode else "Single-window"
    )
    print(f"Mode: {mode_str}")

    # ── 데이터 로드 ──
    print(f"\nLoading {args.n_cases} pilot cases (ABP)...")
    cases = load_pilot_cases(n_cases=args.n_cases, offset_from_end=200, signal_types=["abp"])
    if not cases:
        print("No cases loaded.", file=sys.stderr)
        sys.exit(1)

    train_cases, test_cases = split_by_subject(cases, train_ratio=args.train_ratio)
    print(f"Split: {len(train_cases)} train, {len(test_cases)} test cases")

    if multi_mode:
        # ── 다중 윈도우 모드 ──
        def _extract_multi(case_list):
            all_samples = []
            for case in case_list:
                mw = create_multiwindow_samples(
                    case, n_context=args.n_context_windows,
                    window_sec=args.window_sec, stride_sec=args.stride_sec,
                    horizon_sec=args.prediction_horizon_sec,
                )
                all_samples.extend(mw)
            return all_samples

        train_mw = _extract_multi(train_cases)
        test_mw = _extract_multi(test_cases)

        n_pos_train = sum(1 for s in train_mw if s.label == 1)
        n_pos_test = sum(1 for s in test_mw if s.label == 1)
        print(f"Train: {len(train_mw)} samples ({n_pos_train} hypo, {n_pos_train / max(len(train_mw), 1) * 100:.1f}%)")
        print(f"Test:  {len(test_mw)} samples ({n_pos_test} hypo, {n_pos_test / max(len(test_mw), 1) * 100:.1f}%)")

        if not train_mw or not test_mw:
            print("Insufficient multi-window data.", file=sys.stderr)
            sys.exit(1)

        train_batches = _make_multi_batches(train_mw, args.batch_size, args.patch_size, max_length)
        test_batches = _make_multi_batches(test_mw, args.batch_size, args.patch_size, max_length)

        # Probe 선택
        if args.aggregator == "concat":
            probe = ConcatProbe(d_model, args.n_context_windows, n_classes=1)
        elif args.aggregator == "lstm":
            probe = LSTMProbe(d_model, hidden_dim=64, n_classes=1)
        elif args.aggregator == "mean":
            probe = MeanProbe(d_model, n_classes=1)

        print(f"\nTraining {args.aggregator} probe (d_model={d_model}, n_ctx={args.n_context_windows})...")
        train_losses = train_probe_multi(model, probe, train_batches, args.epochs, args.lr, device)

        print("\nEvaluating...")
        metrics = evaluate_probe_multi(model, probe, test_batches, device)

    else:
        # ── 단일 윈도우 모드 (기존) ──
        def _extract_labeled(case_list):
            windows = []
            for case in case_list:
                wins = extract_windows(case, "abp", args.window_sec, args.stride_sec)
                windows.extend(apply_pipeline(wins))
            return create_labeled_dataset_hypotension(windows)

        train_labeled = _extract_labeled(train_cases)
        test_labeled = _extract_labeled(test_cases)

        n_pos_train = sum(1 for lw in train_labeled if lw.label == 1)
        n_pos_test = sum(1 for lw in test_labeled if lw.label == 1)
        print(f"Train: {len(train_labeled)} windows ({n_pos_train} hypo, {n_pos_train / max(len(train_labeled), 1) * 100:.1f}%)")
        print(f"Test:  {len(test_labeled)} windows ({n_pos_test} hypo, {n_pos_test / max(len(test_labeled), 1) * 100:.1f}%)")

        if not train_labeled or not test_labeled:
            print("Insufficient data.", file=sys.stderr)
            sys.exit(1)

        train_batches = _make_single_batches(train_labeled, args.batch_size, args.patch_size, max_length)
        test_batches = _make_single_batches(test_labeled, args.batch_size, args.patch_size, max_length)

        probe = LinearProbe(d_model, n_classes=1)
        print(f"\nTraining LinearProbe (d_model={d_model})...")
        train_losses = train_probe_single(model, probe, train_batches, args.epochs, args.lr, device)

        print("\nEvaluating...")
        metrics = evaluate_probe_single(model, probe, test_batches, device)

    # ── 결과 출력 ──
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

    roc_path = out_dir / "task1_roc_curve.png"
    plot_roc_curve(y_true, y_score, roc_path, title="Task 1: Hypotension Prediction — ROC Curve")
    print(f"\nROC curve saved: {roc_path}")

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
            "n_context_windows": args.n_context_windows,
            "prediction_horizon_sec": args.prediction_horizon_sec,
            "aggregator": args.aggregator if multi_mode else "linear",
        },
    }
    results_path = out_dir / "task1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
