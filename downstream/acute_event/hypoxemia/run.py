# -*- coding:utf-8 -*-
"""Task 1: Hypoxemia Prediction (SpO2 < 90%, >=1min sustained).

2가지 모드:
  - linear_probe: Frozen encoder + LinearProbe (representation 품질 평가)
  - lora:         Frozen encoder + LoRA adapters + LinearProbe (효율적 fine-tuning)

입력: 최대 10분(600초) 윈도우 → Foundation Model encoder → mean pool → LinearProbe

사용법:
    # Linear probe (기본)
    python -m downstream.acute_event.hypoxemia.run \
        --checkpoint best.pt --mode linear_probe

    # LoRA fine-tuning
    python -m downstream.acute_event.hypoxemia.run \
        --checkpoint best.pt --mode lora --lr 1e-4 --epochs 30 --lora-rank 8
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
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
    bootstrap_ci,
    compute_auprc,
    compute_auroc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe
from downstream.window_task import make_window_batches, iter_window_batches
from downstream._eval_utils import dump_fold_predictions
from downstream._save_utils import load_prepared_split_chunked


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
        # sig_type 은 문자열. SIGNAL_TYPES 로 int 인덱스 변환 후
        # get_global_spatial_id 도 int 인덱스로 호출해야 한다 (Patch C).
        stype_int = SIGNAL_TYPES.get(sig_type, 1)
        spatial_id = get_global_spatial_id(stype_int, 0)
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
    """(batch, labels) 리스트 생성 — 윈도우당 1 행(공유 헬퍼 위임).

    버그 수정(2026-06-17): 예전엔 batch 의 여러 윈도우가 1 pack 행으로 합쳐져
    extract_features 의 행 단위 mean-pool 이 B≠n_windows 를 내 logits/labels 가
    어긋났다. 이제 윈도우별 collate 후 stack 한다(``max_length`` 는 하위 호환용).
    """
    return make_window_batches(
        windows,
        batch_size,
        patch_size,
        to_samples=_multi_window_to_samples,
        get_label=lambda w: w.label,
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
    mask_f = patch_mask.unsqueeze(-1).float()  # (B, N, 1)
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# ── Linear Probe (encoder frozen) ───────────────────────────


@torch.no_grad()
def _stream_extract_features(
    model,
    windows: list[MultiSignalWindow],
    batch_size: int,
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter_window_batches 로 batch 를 하나씩 빌드·forward·폐기 (peak = batch 1개).

    make_window_batches 는 모든 batch 를 리스트로 미리 빌드 → 큰 윈도우(10min×다신호)
    × 수만 윈도우에서 host-RAM OOM 이 났다. 스트리밍하면 한 번에 batch 1개만 메모리에
    두고 d_model feature(작음)만 누적한다. frozen encoder 라 feature 는 1회 추출이면
    충분(매 epoch 재추출 제거).

    returns (features (N, d_model) cpu, labels (N,) cpu float).
    """
    feats: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    for batch, labels in iter_window_batches(
        windows, batch_size, patch_size,
        to_samples=_multi_window_to_samples,
        get_label=lambda w: w.label,
    ):
        f = model.extract_features(batch, pool="mean").detach().cpu()
        feats.append(f)
        labs.append(labels)
    if not feats:
        return torch.empty(0), torch.empty(0)
    return torch.cat(feats, dim=0), torch.cat(labs, dim=0).float()


def train_linear_probe(
    model,
    probe: LinearProbe,
    train_windows: list[MultiSignalWindow],
    val_windows: list[MultiSignalWindow],
    batch_size: int,
    patch_size: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[list[float], list[float], float, int]:
    """Frozen encoder linear probe + val-AUROC best-ckpt 선택.

    frozen encoder feature 를 train/val 각각 스트리밍으로 1회만 추출(전체 batch 리스트
    미빌드 → host-RAM OOM 회피, 매 epoch 재추출 제거)해 캐시한 뒤, probe 만 cached
    feature 로 학습한다. contiguous chunk 순회로 make_window_batches 와 동일한 minibatch
    경계·순서를 보존(수치적 동일). best probe state 는 in-place 로 복원한다.

    returns (train_losses, val_aurocs, best_val_auroc, best_epoch).
    """
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    print("  Caching frozen encoder features (streaming, 1 pass)...")
    train_features, train_labels = _stream_extract_features(
        model, train_windows, batch_size, patch_size, device,
    )
    val_features, val_labels = _stream_extract_features(
        model, val_windows, batch_size, patch_size, device,
    )
    train_features = train_features.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_labels_np = val_labels.numpy()
    n = train_features.size(0)

    train_losses: list[float] = []
    val_aurocs: list[float] = []
    best_val_auroc = -1.0
    best_epoch = -1
    best_probe_state: dict | None = None

    for epoch in range(epochs):
        probe.train()
        epoch_loss, n_steps = 0.0, 0
        for i in range(0, n, batch_size):
            feats = train_features[i: i + batch_size]
            logits = probe(feats)
            loss = criterion(logits, train_labels[i: i + batch_size].unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1
        avg_loss = epoch_loss / max(n_steps, 1)
        train_losses.append(avg_loss)

        # ── val eval (cached features, no encoder forward) ──
        probe.eval()
        with torch.no_grad():
            val_logits = probe(val_features)
            val_scores = torch.sigmoid(val_logits).squeeze(-1).cpu().numpy()
        val_metrics = _compute_metrics(val_labels_np, val_scores)
        val_auroc = float(val_metrics["auroc"])
        val_aurocs.append(val_auroc)

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_epoch = epoch
            # deepcopy to detach from current parameters (probe is small — cheap).
            best_probe_state = copy.deepcopy(probe.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"  Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  "
                f"val_auroc={val_auroc:.4f}  "
                f"(best={best_val_auroc:.4f}@ep{best_epoch + 1})"
            )

    if best_probe_state is None:
        print("WARNING: no best ckpt captured; using last epoch.", file=sys.stderr)
    else:
        probe.load_state_dict(best_probe_state)
    return train_losses, val_aurocs, best_val_auroc, best_epoch


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
    logits = probe(features.to(device))
    scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return _compute_metrics(labels.numpy(), scores)


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
    # Patch B: F1 at best-Youden threshold (binary, 'macro' avg for parity with
    # other downstream tasks like sepsis/aki/cardiac_arrest).
    f1 = compute_f1(y_true, y_pred_opt, average="macro")

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": float(f1),
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


def _load_data(
    args,
) -> tuple[
    list[MultiSignalWindow],
    list[MultiSignalWindow],
    list[MultiSignalWindow],
]:
    """args에서 train/val/test MultiSignalWindow 리스트를 반환한다.

    prepare_data.py 가 ``data["val"]`` 을 포함해 저장하면 그대로 사용한다.
    구(legacy) 산출물에 ``"val"`` 키가 없으면 backward-compat 으로 train 의
    20%(patient/case 단위 X — sample 단위)를 dynamic split 하고 ``warnings.warn``
    으로 알린다.
    """
    if args.data_path:
        # 단일 .pt(back-compat) 또는 fold-chunked prefix(ablation runner) 모두 처리.
        load_fold = int(args.fold) if int(args.n_folds) > 1 else None
        print(f"\nLoading prepared data: {args.data_path} (fold={load_fold})")
        data = load_prepared_split_chunked(args.data_path, fold=load_fold)
        meta = data.get("metadata", {})
        print(f"  Task: {meta.get('task', '?')}, Source: {meta.get('source', '?')}")
        print(f"  Input signals: {meta.get('input_signals', '?')}")
        print(
            f"  Horizon: {meta.get('horizon_sec', 0) / 60:.0f}min, "
            f"Window: {meta.get('window_sec', '?')}s"
        )

        train_data = data["train"]
        test_data = data["test"]
        val_data = data.get("val")  # Patch A: prepare_data 가 추가하는 새 키

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

        train_windows = _pt_to_windows(train_data)
        test_windows = _pt_to_windows(test_data)

        if val_data is not None:
            val_windows = _pt_to_windows(val_data)
            print(f"  val split (from prepare_data): {len(val_windows)} samples")
        else:
            # Backward-compat: legacy 산출물 — train 에서 20% 동적 split.
            warnings.warn(
                "data['val'] not found; falling back to a 20% dynamic split of "
                "train. Re-run prepare_data.py to get a deterministic val split.",
                stacklevel=2,
            )
            rng = np.random.default_rng(args.val_split_seed)
            n_train = len(train_windows)
            idx = np.arange(n_train)
            rng.shuffle(idx)
            n_val = max(1, int(n_train * 0.2))
            val_idx = set(idx[:n_val].tolist())
            new_train = [w for i, w in enumerate(train_windows) if i not in val_idx]
            val_windows = [w for i, w in enumerate(train_windows) if i in val_idx]
            train_windows = new_train
            print(
                f"  val split (dynamic, seed={args.val_split_seed}): "
                f"{len(val_windows)} samples"
            )

        return train_windows, val_windows, test_windows

    elif args.data_dir and Path(args.data_dir).is_dir():
        from downstream.acute_event.hypoxemia.prepare_data import (
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
        # Patch A: train/val/test 3-way patient-level split.
        # 남은(test+val) 환자를 절반씩 val/test 로 분할 — val 0 명 방지.
        remaining = patient_ids[n_train_p:]
        n_val_p = max(1, len(remaining) // 2) if len(remaining) >= 2 else 0
        train_pats = set(patient_ids[:n_train_p])
        val_pats = set(remaining[:n_val_p])
        test_pats = set(remaining[n_val_p:])

        train_cases = [c for c in cases if c["patient_id"] in train_pats]
        val_cases = [c for c in cases if c["patient_id"] in val_pats]
        test_cases = [c for c in cases if c["patient_id"] in test_pats]
        print(
            f"Split: {len(train_cases)} train, {len(val_cases)} val, "
            f"{len(test_cases)} test cases"
        )

        train_samples = extract_forecast_samples(
            train_cases, input_sigs, args.window_sec, args.stride_sec, horizon_sec,
        )
        val_samples = extract_forecast_samples(
            val_cases, input_sigs, args.window_sec, args.stride_sec, horizon_sec,
        ) if val_cases else []
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

        train_windows = _forecast_to_windows(train_samples)
        val_windows = _forecast_to_windows(val_samples)
        test_windows = _forecast_to_windows(test_samples)

        # data_dir 경로에서도 val 이 비면 train 에서 20% 동적 split (backward-compat).
        if not val_windows:
            warnings.warn(
                "No val cases produced from --data-dir split; falling back to a "
                "20% dynamic split of train.",
                stacklevel=2,
            )
            rng2 = np.random.default_rng(args.val_split_seed)
            idx = np.arange(len(train_windows))
            rng2.shuffle(idx)
            n_val = max(1, int(len(train_windows) * 0.2))
            val_idx = set(idx[:n_val].tolist())
            new_train = [w for i, w in enumerate(train_windows) if i not in val_idx]
            val_windows = [w for i, w in enumerate(train_windows) if i in val_idx]
            train_windows = new_train

        return train_windows, val_windows, test_windows

    else:
        print("ERROR: --data-path or --data-dir required.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 1: Hypoxemia Prediction")
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
    parser.add_argument("--fold", type=int, default=0,
                        help="현재 fold 인덱스 (run_eval OOF 집계용 .npz 라벨)")
    parser.add_argument("--n-folds", type=int, default=1, help="전체 fold 수")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--input-signals",
        nargs="+",
        default=["ppg", "ecg", "abp", "co2"],
        choices=["abp", "ecg", "ppg", "co2"],
        help="Input signal types (Task #6: PPG, ECG, ABP, CO2; label from SpO2 numerics)",
    )
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=42,
        help="Seed for backward-compat dynamic val split when data['val'] is missing.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Bootstrap iterations for 95%% CI on test metrics (Patch D).",
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
        from downstream.model_wrapper import DownstreamModelWrapper

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

    # ── 데이터 로드 (Patch A: train/val/test 3-way) ──
    train_labeled, val_labeled, test_labeled = _load_data(args)

    # 실제 사용 신호/윈도우 = 로드된 데이터 기준. prepared-data 는 args 가 아니라
    # 파일의 signals 를 쓰므로, print/저장 metadata 가 args 기본값을 잘못 보이지
    # 않도록 데이터에서 직접 추출한다.
    if train_labeled:
        actual_signals = list(train_labeled[0].signals.keys())
        actual_window_sec = (
            len(next(iter(train_labeled[0].signals.values()))) / DEFAULT_SR
        )
    else:
        actual_signals = list(args.input_signals)
        actual_window_sec = float(args.window_sec)
    sig_str = " + ".join(s.upper() for s in actual_signals)
    print(f"Mode: {args.mode} | Input: {sig_str} | Window: {actual_window_sec:.0f}s")

    def _pos_stats(name: str, ws: list[MultiSignalWindow]) -> None:
        n_pos = sum(1 for lw in ws if lw.label == 1)
        print(
            f"  {name}: {len(ws)} samples "
            f"({n_pos} hypo, {n_pos / max(len(ws), 1) * 100:.1f}%)"
        )

    _pos_stats("Train", train_labeled)
    _pos_stats("Val  ", val_labeled)
    _pos_stats("Test ", test_labeled)

    if not train_labeled or not test_labeled:
        print("Insufficient data.", file=sys.stderr)
        sys.exit(1)
    if not val_labeled:
        print(
            "ERROR: val split is empty even after fallback — cannot do best-ckpt "
            "selection.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── 학습 + Best-ckpt selection on val (Patch A) ──
    probe = LinearProbe(d_model, n_classes=1)

    is_lora = args.mode == "lora"
    if is_lora:
        # lora: encoder fine-tune → feature 캐싱 불가, pre-built batches 필요.
        # 주의: 큰 윈도우면 RAM 위험 — ablation 은 linear_probe 만 사용.
        first_sig = next(iter(train_labeled[0].signals.values()))
        max_length = len(first_sig)
        train_batches = _make_batches(
            train_labeled, args.batch_size, args.patch_size, max_length
        )
        val_batches = _make_batches(
            val_labeled, args.batch_size, args.patch_size, max_length
        )
        test_batches = _make_batches(
            test_labeled, args.batch_size, args.patch_size, max_length
        )

        n_lora = sum(p.numel() for p in model.lora_parameters())
        n_probe = sum(p.numel() for p in probe.parameters())
        print(
            f"\nTraining LoRA + Probe (rank={args.lora_rank}, "
            f"LoRA={n_lora:,} + Probe={n_probe:,} params)..."
        )
        probe = probe.to(device)
        probe.train()
        model.model.train()
        lora_params = model.lora_parameters()
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": args.lr},
                {"params": probe.parameters(), "lr": args.lr},
            ],
            weight_decay=0.01,
        )
        criterion = nn.BCEWithLogitsLoss()
        train_losses: list[float] = []
        val_aurocs: list[float] = []
        best_val_auroc = -1.0
        best_epoch = -1
        best_probe_state: dict | None = None
        best_model_state: dict | None = None  # LoRA mode 에서만 저장

        for epoch in range(args.epochs):
            # ── train one epoch ──
            probe.train()
            model.model.train()
            epoch_loss, n_steps = 0.0, 0

            for batch, labels in train_batches:
                batch = model.batch_to_device(batch)
                out = model.model(batch, task="masked")
                features = _mean_pool(out["encoded"], out["patch_mask"])

                logits = probe(features)
                loss = criterion(logits, labels.to(device).unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    lora_params + list(probe.parameters()), 1.0
                )
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1

            avg_loss = epoch_loss / max(n_steps, 1)
            train_losses.append(avg_loss)

            # ── val evaluation ──
            val_metrics = evaluate_lora(model, probe, val_batches, device)
            val_auroc = float(val_metrics["auroc"])
            val_aurocs.append(val_auroc)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch
                # deepcopy to detach from current parameters (probe is small — cheap).
                best_probe_state = copy.deepcopy(probe.state_dict())
                # LoRA params live inside model.model — copy entire encoder state.
                best_model_state = copy.deepcopy(model.model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
                print(
                    f"  Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}  "
                    f"val_auroc={val_auroc:.4f}  "
                    f"(best={best_val_auroc:.4f}@ep{best_epoch + 1})"
                )

        # ── Restore best ckpt and run test ONCE (Patch A) ──
        if best_probe_state is None:
            print("WARNING: no best ckpt captured; using last epoch.", file=sys.stderr)
        else:
            probe.load_state_dict(best_probe_state)
            if best_model_state is not None:
                model.model.load_state_dict(best_model_state)

        print("\nEvaluating on test set with best-val ckpt...")
        metrics = evaluate_lora(model, probe, test_batches, device)
    else:
        # linear_probe: frozen encoder feature 를 스트리밍으로 1회만 추출(전체 batch
        # 리스트 미빌드 → OOM 회피, 매 epoch 재추출 제거). probe 만 cached feature 로
        # 학습하며 val-AUROC 기준 best-ckpt 선택(기존 동작 보존).
        print(f"\nTraining LinearProbe (frozen encoder, d_model={d_model})...")
        train_losses, val_aurocs, best_val_auroc, best_epoch = train_linear_probe(
            model, probe, train_labeled, val_labeled,
            args.batch_size, args.patch_size, args.epochs, args.lr, device,
        )

        print("\nEvaluating on test set with best-val ckpt...")
        metrics = evaluate_linear_probe(
            model, probe, test_labeled, args.batch_size, args.patch_size, device,
        )

    # ── 결과 출력 ──
    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")
    # patient(=case)-level grouping id, y_true/y_score 와 동일 순서 (test_labeled 순서).
    # _eval_utils.bootstrap_ci 에서 환자 단위 resampling 에 사용.
    patient_ids = [str(w.case_id) for w in test_labeled]

    # Patch D: bootstrap 95% CI on test AUROC/AUPRC/F1 (1000 iter).
    thr = metrics["optimal_threshold"]
    y_pred_opt = (y_score >= thr).astype(int)

    auroc_ci = bootstrap_ci(
        compute_auroc, y_true, y_score, n_iter=args.bootstrap_iters,
    )
    auprc_ci = bootstrap_ci(
        compute_auprc, y_true, y_score, n_iter=args.bootstrap_iters,
    )
    # F1 은 binarized prediction 기반 — best-Youden threshold 고정 사용.
    f1_ci = bootstrap_ci(
        lambda yt, yp: compute_f1(yt, yp, average="macro"),
        y_true,
        y_pred_opt,
        n_iter=args.bootstrap_iters,
    )

    print(f"\n{'=' * 50}")
    print(f"  Mode:         {args.mode}")
    print(f"  Best epoch:   {best_epoch + 1}/{args.epochs}")
    print(f"  Val AUROC:    {best_val_auroc:.4f}")
    print(
        f"  AUROC:        {metrics['auroc']:.4f} "
        f"[{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]"
    )
    print(
        f"  AUPRC:        {metrics['auprc']:.4f} "
        f"[{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]"
    )
    print(
        f"  F1:           {metrics['f1']:.4f} "
        f"[{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]"
    )
    print(f"  Threshold:    {metrics['optimal_threshold']:.3f}")
    print(f"  Sensitivity:  {metrics['sensitivity']:.4f}")
    print(f"  Specificity:  {metrics['specificity']:.4f}")
    print(
        f"  Prevalence:   {metrics['prevalence']:.3f} "
        f"({metrics['n_positive']}/{metrics['n_total']})"
    )
    print(f"{'=' * 50}")

    roc_path = out_dir / f"task1_roc_{args.mode}.png"
    plot_roc_curve(
        y_true, y_score, roc_path,
        title=f"Task 1: Hypoxemia — {args.mode} ROC",
    )
    print(f"\nROC curve saved: {roc_path}")

    results = {
        **metrics,
        "y_true": y_true.tolist(),
        "y_score": y_score.tolist(),
        "patient_ids": patient_ids,
        "val_auroc_best": float(best_val_auroc),
        "best_epoch": int(best_epoch),
        "auroc_ci": [float(auroc_ci[0]), float(auroc_ci[1])],
        "auprc_ci": [float(auprc_ci[0]), float(auprc_ci[1])],
        "f1_ci": [float(f1_ci[0]), float(f1_ci[1])],
        "train_losses": train_losses,
        "val_aurocs": val_aurocs,
        "config": {
            "mode": args.mode,
            "input_signals": actual_signals,
            "lora_rank": args.lora_rank if args.mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
            "n_cases": args.n_cases,
            "window_sec": actual_window_sec,
            "stride_sec": args.stride_sec,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
            "bootstrap_iters": args.bootstrap_iters,
            "val_split_seed": args.val_split_seed,
        },
    }
    results_path = out_dir / f"task1_results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    # fold prediction(.npz) — downstream.run_eval 가 OOF 집계에 사용.
    npz_path = dump_fold_predictions(
        out_dir, task="hypoxemia", fold_idx=args.fold, n_folds=args.n_folds,
        y_true=y_true, y_score=y_score, patient_ids=patient_ids,
    )
    print(f"Fold predictions: {npz_path}")


if __name__ == "__main__":
    main()
