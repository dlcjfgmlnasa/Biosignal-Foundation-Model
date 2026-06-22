# -*- coding:utf-8 -*-
"""Arrhythmia Classification (5-class, MIMIC-III-Ext-PPG).

2가지 모드:
  - linear_probe: Frozen encoder + LinearProbe
  - lora:         Frozen encoder + LoRA adapters + LinearProbe

5-class: SR / AF / STACH / SBRAD / AFLT
입력: PPG / ECG / PPG+ECG (30초, 100Hz)

사용법:
    # Linear probe
    python -m downstream.acute_event.arrhythmia.run \
        --checkpoint best.pt --mode linear_probe \
        --data-path outputs/downstream/arrhythmia/arrhythmia_ppg.pt

    # LoRA
    python -m downstream.acute_event.arrhythmia.run \
        --checkpoint best.pt --mode lora --lr 1e-4 --epochs 30 \
        --data-path outputs/downstream/arrhythmia/arrhythmia_ppg.pt
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

from data.collate import PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import get_global_spatial_id
from data.parser.vitaldb import SIGNAL_TYPES

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_sensitivity_specificity,
)
from downstream.model_wrapper import LinearProbe
from downstream.window_task import make_window_batches, iter_window_batches
from downstream._eval_utils import dump_fold_predictions

# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0

CLASS_NAMES = ["SR", "AF", "STACH", "SBRAD", "AFLT"]
N_CLASSES = len(CLASS_NAMES)


# ── 배치 생성 ─────────────────────────────────────────────────


@dataclass
class MultiSignalWindow:
    """다중 신호 윈도우 + 라벨."""

    signals: dict[str, np.ndarray]
    label: int  # 0~4
    patient: str


def _multi_window_to_samples(mw: MultiSignalWindow, idx: int) -> list[BiosignalSample]:
    samples = []
    for ch, (sig_type, signal) in enumerate(mw.signals.items()):
        stype_int = SIGNAL_TYPES.get(sig_type, 0)
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
                session_id=f"patient_{mw.patient}",
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
    # 버그 수정(2026-06-17): 윈도우당 1 행 보장(공유 헬퍼 위임). 예전엔 batch 의
    # 여러 윈도우가 1 pack 행으로 합쳐져 extract_features 가 B≠n_windows 를 냈다.
    # arrhythmia 는 multi-class → labels 는 long 유지(label_dtype).
    return make_window_batches(
        windows,
        batch_size,
        patch_size,
        to_samples=_multi_window_to_samples,
        get_label=lambda w: w.label,
        label_dtype=torch.long,
    )


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
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# ── Linear Probe (encoder frozen) ───────────────────────────


@torch.no_grad()
def _stream_extract_features(
    model, windows, batch_size, patch_size, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter_window_batches 로 batch 를 하나씩 빌드·forward·폐기 (peak = batch 1개).

    make_window_batches 는 모든 batch 를 리스트로 미리 빌드 → 큰 윈도우(30s×다신호)
    × 수만 윈도우에서 host-RAM OOM 이 났다. 스트리밍하면 한 번에 batch 1개만 메모리에
    두고 d_model feature(작음)만 누적한다. frozen encoder 라 feature 는 1회 추출이면
    충분(매 epoch 재추출 제거). 멀티클래스 → labels 는 long 유지.

    returns (features (N, d_model) cpu, labels (N,) cpu long).
    """
    feats: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    for batch, labels in iter_window_batches(
        windows, batch_size, patch_size,
        to_samples=_multi_window_to_samples,
        get_label=lambda w: w.label,
        label_dtype=torch.long,
    ):
        f = model.extract_features(batch, pool="mean").detach().cpu()
        feats.append(f)
        labs.append(labels)
    if not feats:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(feats, dim=0), torch.cat(labs, dim=0)


def train_linear_probe(
    model,
    probe: LinearProbe,
    train_windows: list[MultiSignalWindow],
    batch_size: int,
    patch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    # frozen encoder feature 를 스트리밍으로 1회만 추출(전체 batch 리스트 미빌드 →
    # OOM 회피) 후, probe 만 cached feature 로 학습한다(encoder forward 1패스).
    # contiguous chunk 순회 → make_window_batches 와 동일한 minibatch 경계·순서
    # 보존(수치적 동일).
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print("  Caching frozen encoder features (streaming, 1 pass)...")
    features, labels = _stream_extract_features(
        model, train_windows, batch_size, patch_size, device,
    )
    features = features.to(device)
    labels = labels.to(device)
    n = features.size(0)

    losses = []
    probe.train()
    for epoch in range(epochs):
        epoch_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            feats = features[i: i + batch_size]
            logits = probe(feats)  # (B, N_CLASSES)
            loss = criterion(logits, labels[i: i + batch_size])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        avg = epoch_loss / max(nb, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


@torch.no_grad()
def evaluate_linear_probe(
    model,
    probe: LinearProbe,
    test_windows: list[MultiSignalWindow],
    batch_size: int,
    patch_size: int,
    device: torch.device,
) -> dict:
    probe.to(device).eval()
    features, labels = _stream_extract_features(
        model, test_windows, batch_size, patch_size, device,
    )
    logits = probe(features.to(device))  # (N, N_CLASSES)
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    return _compute_multiclass_metrics(labels.numpy(), probs)


# ── LoRA fine-tuning ─────────────────────────────────────────


def train_lora(
    model,
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

    lora_params = model.lora_parameters()
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr},
        {"params": probe.parameters(), "lr": lr},
    ], weight_decay=0.01)

    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for batch, labels in train_batches:
            batch = model.batch_to_device(batch)
            out = model.model(batch, task="masked")
            features = _mean_pool(out["encoded"], out["patch_mask"])

            logits = probe(features)
            loss = criterion(logits, labels.to(device))

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
    all_labels, all_probs = [], []

    for batch, labels in test_batches:
        batch = model.batch_to_device(batch)
        out = model.model(batch, task="masked")
        features = _mean_pool(out["encoded"], out["patch_mask"])

        logits = probe(features)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_probs.append(probs)

    return _compute_multiclass_metrics(
        np.concatenate(all_labels), np.concatenate(all_probs)
    )


# ── Multi-class 메트릭 ───────────────────────────────────────


def _compute_multiclass_metrics(
    y_true: np.ndarray,  # (N,) int
    y_probs: np.ndarray,  # (N, n_classes)
) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        classification_report,
        confusion_matrix,
    )

    y_pred = y_probs.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class AUROC (one-vs-rest)
    per_class_auroc = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == cls_id).astype(int)
        if binary_true.sum() == 0 or binary_true.sum() == len(binary_true):
            per_class_auroc[cls_name] = float("nan")
            continue
        per_class_auroc[cls_name] = float(compute_auroc(binary_true, y_probs[:, cls_id]))

    # Per-class sensitivity/specificity
    per_class_stats = {}
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        binary_true = (y_true == cls_id).astype(int)
        binary_pred = (y_pred == cls_id).astype(int)
        ss = compute_sensitivity_specificity(binary_true, binary_pred)
        per_class_stats[cls_name] = ss

    cm = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_auroc": per_class_auroc,
        "per_class_stats": per_class_stats,
        "confusion_matrix": cm.tolist(),
        "n_total": len(y_true),
        "class_distribution": {
            CLASS_NAMES[i]: int((y_true == i).sum()) for i in range(N_CLASSES)
        },
        "y_true": y_true,
        "y_probs": y_probs,
    }


# ── 데이터 로딩 ──────────────────────────────────────────────


def _load_data(args) -> tuple[list[MultiSignalWindow], list[MultiSignalWindow]]:
    # data_path 가 그대로 존재하면 단일 파일(back-compat). 없으면 fold-prefix 로
    # 해석해 ``{data_path}_fold{fold}.pt`` 를 로드한다 (ablation runner 는 prefix +
    # --fold 를 넘기고, arrhythmia 데이터는 fold 당 완성 파일이라 fold 별로 선택).
    path = Path(args.data_path) if args.data_path else None
    if path is not None and not path.exists():
        cand = Path(f"{args.data_path}_fold{int(args.fold)}.pt")
        if cand.exists():
            path = cand
    if path is None or not path.exists():
        print("ERROR: --data-path required (단일 .pt 또는 '{prefix}_fold{F}.pt').",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading prepared data: {path}")
    data = torch.load(path, weights_only=False)
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    print(f"  Input signals: {meta.get('input_signals', '?')}")
    print(f"  Classes: {meta.get('class_names', '?')}")
    print(f"  N classes: {meta.get('n_classes', '?')}")

    train_data = data["train"]
    test_data = data["test"]

    input_keys = list(train_data["signals"].keys())
    if not input_keys:
        print("ERROR: No signals in prepared data.", file=sys.stderr)
        sys.exit(1)

    def _pt_to_windows(split_data):
        windows = []
        labels_t = split_data["labels"]
        patients = split_data.get("patients", [""] * len(labels_t))
        n = len(labels_t)
        for i in range(n):
            signals = {
                k: split_data["signals"][k][i].numpy()
                for k in input_keys
            }
            windows.append(
                MultiSignalWindow(
                    signals=signals,
                    label=int(labels_t[i].item()),
                    patient=patients[i] if i < len(patients) else "",
                )
            )
        return windows

    return _pt_to_windows(train_data), _pt_to_windows(test_data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Arrhythmia Classification (5-class, MIMIC-III-Ext-PPG)"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--mode",
        type=str,
        default="linear_probe",
        choices=["linear_probe", "lora"],
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--fold", type=int, default=0,
                        help="현재 fold 인덱스 (run_eval OOF 집계용 .npz 라벨)")
    parser.add_argument("--n-folds", type=int, default=1, help="전체 fold 수")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-path", type=str, required=True)
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

        if args.mode == "lora":
            model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    print(f"Mode: {args.mode} | Classes: {CLASS_NAMES}")

    # ── 데이터 로드 ──
    train_windows, test_windows = _load_data(args)

    print(f"  Train: {len(train_windows)} samples")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cnt = sum(1 for w in train_windows if w.label == cls_id)
        print(f"    {cls_name}: {cnt}")
    print(f"  Test: {len(test_windows)} samples")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        cnt = sum(1 for w in test_windows if w.label == cls_id)
        print(f"    {cls_name}: {cnt}")

    if not train_windows or not test_windows:
        print("Insufficient data.", file=sys.stderr)
        sys.exit(1)

    # ── 학습 ──
    probe = LinearProbe(d_model, n_classes=N_CLASSES)

    if args.mode == "linear_probe":
        # 스트리밍 추출 → 전체 batch 리스트 미빌드(OOM 회피). windows 를 그대로 전달.
        print(f"\nTraining LinearProbe (frozen encoder, d_model={d_model})...")
        train_losses = train_linear_probe(
            model, probe, train_windows, args.batch_size, args.patch_size,
            args.epochs, args.lr, device,
        )
        print("\nEvaluating...")
        metrics = evaluate_linear_probe(
            model, probe, test_windows, args.batch_size, args.patch_size, device,
        )

    elif args.mode == "lora":
        # lora: encoder fine-tune → feature 캐싱 불가, pre-built batches 필요.
        # 주의: 큰 윈도우면 RAM 위험 — ablation 은 linear_probe 만 사용.
        first_sig = next(iter(train_windows[0].signals.values()))
        max_length = len(first_sig)
        train_batches = _make_batches(
            train_windows, args.batch_size, args.patch_size, max_length
        )
        test_batches = _make_batches(
            test_windows, args.batch_size, args.patch_size, max_length
        )
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
    print(f"\n{'=' * 50}")
    print(f"  Mode:       {args.mode}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  Macro F1:   {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:{metrics['weighted_f1']:.4f}")
    print(f"\n  Per-class AUROC:")
    for cls_name, auroc in metrics["per_class_auroc"].items():
        print(f"    {cls_name}: {auroc:.4f}")
    print(f"\n  Per-class Sensitivity / Specificity:")
    for cls_name, ss in metrics["per_class_stats"].items():
        print(f"    {cls_name}: sens={ss['sensitivity']:.4f} spec={ss['specificity']:.4f}")
    print(f"\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    header = "        " + "  ".join(f"{c:>5}" for c in CLASS_NAMES)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>5}" for v in row)
        print(f"  {CLASS_NAMES[i]:>5} {row_str}")
    print(f"{'=' * 50}")

    results = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "per_class_auroc": metrics["per_class_auroc"],
        "per_class_stats": metrics["per_class_stats"],
        "confusion_matrix": metrics["confusion_matrix"],
        "class_distribution": metrics["class_distribution"],
        "y_true": metrics["y_true"].tolist(),
        "y_probs": metrics["y_probs"].tolist(),
        # patient-level grouping id (test_windows 순서 = y_true 순서).
        # 멀티클래스 — run_eval 가 auroc_macro/auprc_macro (OvR) 로 집계.
        "patient_ids": [str(w.patient) for w in test_windows],
        "train_losses": train_losses,
        "config": {
            "mode": args.mode,
            "lora_rank": args.lora_rank if args.mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
            "epochs": args.epochs,
            "lr": args.lr,
        },
    }
    results_path = out_dir / f"arrhythmia_results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {results_path}")

    # fold prediction(.npz) — multi-class (y_probs (N,K) + classes).
    npz_path = dump_fold_predictions(
        out_dir, task="arrhythmia", fold_idx=args.fold, n_folds=args.n_folds,
        y_true=metrics["y_true"], y_score=metrics["y_probs"],
        patient_ids=[str(w.patient) for w in test_windows], classes=CLASS_NAMES,
    )
    print(f"Fold predictions: {npz_path}")


if __name__ == "__main__":
    main()
