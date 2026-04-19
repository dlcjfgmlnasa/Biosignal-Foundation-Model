# -*- coding:utf-8 -*-
"""Task 1: Hypotension Prediction (MAP < 65 mmHg, >=1min sustained).

2가지 모드:
  - linear_probe: Frozen encoder + LinearProbe (representation 품질 평가)
  - lora:         Frozen encoder + LoRA adapters + LinearProbe (효율적 fine-tuning)

입력: 최대 10분(600초) 윈도우 → Foundation Model encoder → mean pool → LinearProbe

사용법:
    # Linear probe (기본)
    python -m downstream.acute_event.hypotension.run \
        --checkpoint best.pt --mode linear_probe

    # LoRA fine-tuning
    python -m downstream.acute_event.hypotension.run \
        --checkpoint best.pt --mode lora --lr 1e-4 --epochs 30 --lora-rank 8
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

from downstream.shared.metrics import (
    compute_auroc,
    compute_auprc,
    compute_sensitivity_specificity,
)
from downstream.shared.viz import plot_roc_curve
from downstream.shared.model_wrapper import LinearProbe


# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0


# ── 배치 생성 ─────────────────────────────────────────────────


@dataclass
class MultiSignalWindow:
    """다중 신호 윈도우 + 라벨."""

    signals: dict[str, np.ndarray]  # {"ecg": (win_samples,), "ppg": ...}
    label: int
    label_value: float
    case_id: str | int


def _multi_window_to_samples(mw: MultiSignalWindow, idx: int) -> list[BiosignalSample]:
    """MultiSignalWindow → 신호별 BiosignalSample 리스트."""
    samples = []
    for ch, (sig_type, signal) in enumerate(mw.signals.items()):
        stype_int = SIGNAL_TYPES.get(sig_type, 1)
        spatial_id = get_global_spatial_id(sig_type, 0)
        samples.append(
            BiosignalSample(
                values=torch.from_numpy(signal).float(),
                length=len(signal),
                channel_idx=ch,
                recording_idx=idx,
                sampling_rate=DEFAULT_SR,
                n_channels=len(mw.signals),
                win_start=0,
                signal_type=stype_int,
                session_id=f"case_{mw.case_id}",
                spatial_id=spatial_id,
            )
        )
    return samples


def _make_batches(
    windows: list[MultiSignalWindow],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    """(batch, labels) 리스트 생성. 다중 신호는 any_variate collate."""
    multi = any(len(w.signals) > 1 for w in windows)
    collate_mode = "any_variate" if multi else "ci"
    collate = PackCollate(
        max_length=max_length, collate_mode=collate_mode, patch_size=patch_size
    )

    batches = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        all_samples = []
        for j, mw in enumerate(chunk):
            all_samples.extend(_multi_window_to_samples(mw, idx=i + j))
        labels = torch.tensor([mw.label for mw in chunk], dtype=torch.float32)
        batch = collate(all_samples)
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


# ── Mean pooling helper ──────────────────────────────────────


def _mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    mask_f = patch_mask.unsqueeze(-1).float()  # (B, N, 1)
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# ── Linear Probe (encoder frozen) ───────────────────────────


def train_linear_probe(
    model,
    probe: LinearProbe,
    train_batches: list[tuple[PackedBatch, torch.Tensor]],
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
def evaluate_linear_probe(
    model,
    probe: LinearProbe,
    test_batches: list[tuple[PackedBatch, torch.Tensor]],
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
    return _compute_metrics(np.concatenate(all_labels), np.concatenate(all_scores))


# ── LoRA fine-tuning (encoder frozen + LoRA adapters) ────────


def train_lora(
    model,  # DownstreamModelWrapper (with LoRA injected)
    probe: LinearProbe,
    train_batches: list[tuple[PackedBatch, torch.Tensor]],
    epochs: int,
    lr: float,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> list[float]:
    model.model.train()
    probe = probe.to(device)
    probe.train()

    # LoRA params + probe params만 학습
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
                lora_params + list(probe.parameters()),
                gradient_clip,
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
def evaluate_lora(
    model,
    probe: LinearProbe,
    test_batches: list[tuple[PackedBatch, torch.Tensor]],
    device: torch.device,
) -> dict:
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


# ── 데이터 로딩 ──────────────────────────────────────────────


def _load_data(args) -> tuple[list[MultiSignalWindow], list[MultiSignalWindow]]:
    """args에서 train/test MultiSignalWindow 리스트를 반환한다."""
    if args.data_path and Path(args.data_path).exists():
        print(f"\nLoading prepared data: {args.data_path}")
        data = torch.load(args.data_path, weights_only=False)
        meta = data.get("metadata", {})
        print(f"  Task: {meta.get('task', '?')}, Source: {meta.get('source', '?')}")
        print(f"  Input signals: {meta.get('input_signals', '?')}")
        print(
            f"  Horizon: {meta.get('horizon_sec', 0) / 60:.0f}min, "
            f"Window: {meta.get('window_sec', '?')}s"
        )

        train_data = data["train"]
        test_data = data["test"]

        input_keys = list(train_data["signals"].keys())
        if not input_keys:
            print("ERROR: No signals in prepared data.", file=sys.stderr)
            sys.exit(1)

        def _pt_to_windows(split_data):
            windows = []
            labels_t = split_data["labels"]
            label_values_t = split_data["label_values"]
            n = len(labels_t)
            for i in range(n):
                signals = {
                    k: split_data["signals"][k][i].numpy()
                    for k in input_keys
                    if k in split_data["signals"]
                }
                windows.append(
                    MultiSignalWindow(
                        signals=signals,
                        label=int(labels_t[i].item()),
                        label_value=float(label_values_t[i].item()),
                        case_id=split_data["case_ids"][i]
                        if "case_ids" in split_data
                        else 0,
                    )
                )
            return windows

        return _pt_to_windows(train_data), _pt_to_windows(test_data)

    elif args.data_dir and Path(args.data_dir).is_dir():
        from downstream.acute_event.hypotension.prepare_data import (
            _load_local_pt_aligned_signals,
            extract_forecast_samples,
        )

        print(f"\nLoading from local .pt directory: {args.data_dir}")
        input_sigs = args.input_signals
        horizon_sec = 300.0
        min_dur = args.window_sec + horizon_sec + args.stride_sec

        cases = _load_local_pt_aligned_signals(
            args.data_dir, input_sigs, min_dur, max_subjects=args.n_cases,
        )
        if not cases:
            print("No valid cases loaded.", file=sys.stderr)
            sys.exit(1)

        rng = np.random.default_rng(42)
        patient_ids = list({c["patient_id"] for c in cases})
        rng.shuffle(patient_ids)
        n_train_p = max(1, int(len(patient_ids) * args.train_ratio))
        train_pats = set(patient_ids[:n_train_p])

        train_cases = [c for c in cases if c["patient_id"] in train_pats]
        test_cases = [c for c in cases if c["patient_id"] not in train_pats]
        print(f"Split: {len(train_cases)} train, {len(test_cases)} test cases")

        train_samples = extract_forecast_samples(
            train_cases, input_sigs, args.window_sec, args.stride_sec, horizon_sec,
        )
        test_samples = extract_forecast_samples(
            test_cases, input_sigs, args.window_sec, args.stride_sec, horizon_sec,
        )

        def _forecast_to_windows(samples):
            windows = []
            for s in samples:
                windows.append(
                    MultiSignalWindow(
                        signals=s.input_signals,
                        label=s.label,
                        label_value=s.label_value,
                        case_id=s.case_id,
                    )
                )
            return windows

        return _forecast_to_windows(train_samples), _forecast_to_windows(test_samples)

    else:
        print("ERROR: --data-path or --data-dir required.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: Hypotension Prediction")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--mode",
        type=str,
        default="linear_probe",
        choices=["linear_probe", "lora"],
        help="linear_probe: frozen encoder, lora: LoRA adapters",
    )
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--window-sec", type=float, default=600.0)
    parser.add_argument("--stride-sec", type=float, default=60.0)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--input-signals",
        nargs="+",
        default=["abp"],
        choices=["abp", "ecg", "ppg"],
        help="Input signal types (label always from ABP)",
    )
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
        from downstream.shared.model_wrapper import DownstreamModelWrapper

        print(f"Loading checkpoint: {args.checkpoint}")
        model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
        d_model = model.d_model

        if args.mode == "lora":
            model.inject_lora(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
            )
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    sig_str = " + ".join(s.upper() for s in args.input_signals)
    print(f"Mode: {args.mode} | Input: {sig_str} | Window: {args.window_sec}s")

    # ── 데이터 로드 ──
    train_labeled, test_labeled = _load_data(args)

    n_pos_train = sum(1 for lw in train_labeled if lw.label == 1)
    n_pos_test = sum(1 for lw in test_labeled if lw.label == 1)
    print(
        f"  Train: {len(train_labeled)} samples "
        f"({n_pos_train} hypo, {n_pos_train / max(len(train_labeled), 1) * 100:.1f}%)"
    )
    print(
        f"  Test:  {len(test_labeled)} samples "
        f"({n_pos_test} hypo, {n_pos_test / max(len(test_labeled), 1) * 100:.1f}%)"
    )

    if not train_labeled or not test_labeled:
        print("Insufficient data.", file=sys.stderr)
        sys.exit(1)

    first_sig = next(iter(train_labeled[0].signals.values()))
    max_length = len(first_sig)
    train_batches = _make_batches(
        train_labeled, args.batch_size, args.patch_size, max_length
    )
    test_batches = _make_batches(
        test_labeled, args.batch_size, args.patch_size, max_length
    )

    # ── 학습 ──
    probe = LinearProbe(d_model, n_classes=1)

    if args.mode == "linear_probe":
        print(f"\nTraining LinearProbe (frozen encoder, d_model={d_model})...")
        train_losses = train_linear_probe(
            model, probe, train_batches, args.epochs, args.lr, device
        )
        print("\nEvaluating...")
        metrics = evaluate_linear_probe(model, probe, test_batches, device)

    elif args.mode == "lora":
        n_lora = sum(p.numel() for p in model.lora_parameters())
        n_probe = sum(p.numel() for p in probe.parameters())
        print(f"\nTraining LoRA + Probe (rank={args.lora_rank}, "
              f"LoRA={n_lora:,} + Probe={n_probe:,} params)...")
        train_losses = train_lora(
            model, probe, train_batches, args.epochs, args.lr, device
        )
        print("\nEvaluating...")
        metrics = evaluate_lora(model, probe, test_batches, device)

    # ── 결과 출력 ──
    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 50}")
    print(f"  Mode:        {args.mode}")
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

    roc_path = out_dir / f"task1_roc_{args.mode}.png"
    plot_roc_curve(
        y_true, y_score, roc_path,
        title=f"Task 1: Hypotension — {args.mode} ROC",
    )
    print(f"\nROC curve saved: {roc_path}")

    results = {
        **metrics,
        "train_losses": train_losses,
        "config": {
            "mode": args.mode,
            "input_signals": args.input_signals,
            "lora_rank": args.lora_rank if args.mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
            "n_cases": args.n_cases,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
        },
    }
    results_path = out_dir / f"task1_results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
