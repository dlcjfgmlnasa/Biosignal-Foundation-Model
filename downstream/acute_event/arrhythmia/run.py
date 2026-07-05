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
    compute_auroc,
    compute_auprc,
    compute_sensitivity_specificity,
)
from downstream.model_wrapper import LinearProbe
from downstream.window_task import make_window_batches, iter_window_batches
from downstream._eval_utils import dump_fold_predictions
from downstream._ddp_utils import (
    broadcast_should_stop,
    ddp_enabled,
    ddp_world_size,
    equalize_shard,
    gather_concat,
    is_main,
    maybe_init_ddp,
    shard_for_rank,
    wrap_lora_ddp,
)

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


# ── linear_probe feature 추출 torchrun 병렬화 (multiclass) ────


def _extract_features_maybe_sharded(
    model, windows, batch_size, patch_size, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """linear_probe feature 추출 (torchrun 병렬화, 멀티클래스 long labels).

    DDP 면 각 rank 가 windows 의 **shard 만** 스트리밍 추출(peak=batch 1개 유지)한 뒤
    ``gather_concat`` 으로 모으고 **글로벌 idx 로 argsort 해 원본 순서를 복원**한다 →
    rank 수와 무관하게 동일한 (feature,label) 순서(deterministic frozen encoder).
    비-DDP 면 ``_stream_extract_features`` 그대로(기존 경로 byte-identical).

    co-index: (글로벌 idx, feats, labels) 를 한 rank=한 번들 로 묶어 한 번에 gather →
    rank 경계에서 어긋나지 않는다. gather 직전 cpu, rank0 가 사용 직전 device 로 올림.
    non-main 은 gather 참여 후 빈 텐서 반환(직후 호출측 종료).
    """
    if not ddp_enabled():
        return _stream_extract_features(model, windows, batch_size, patch_size, device)
    my_windows = shard_for_rank(windows)
    my_gidx = shard_for_rank(list(range(len(windows))))
    feats, labels = _stream_extract_features(
        model, my_windows, batch_size, patch_size, device,
    )
    local = [(torch.tensor(my_gidx, dtype=torch.long), feats, labels)]
    gathered = gather_concat(local)
    if not is_main():
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    parts = [g for g in gathered if g[0].numel() > 0]
    if not parts:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    idx_all = torch.cat([g[0] for g in parts])
    feats_all = torch.cat([g[1] for g in parts], dim=0)
    labels_all = torch.cat([g[2] for g in parts], dim=0)
    order = torch.argsort(idx_all)
    return feats_all[order], labels_all[order]


def _fit_probe_cached(
    probe, features, labels, batch_size, epochs, lr, device,
    val_features=None, val_labels=None, patience: int = 0,
):
    """미리 추출된 cached feature 로 probe 학습 (멀티클래스 CrossEntropy, 순차 minibatch).

    ``val_features`` 가 주어지면 매 epoch 후 val macro-AUROC 를 재고, best-val probe
    state 를 보관한다(deepcopy). ``patience>0`` 이면 val 무개선 patience epoch 시
    early stop. 학습 종료 시 best-val state 를 probe 에 복원하고 반환한다.
    val 이 없으면(None) 기존 고정-epoch·마지막 상태 동작(byte-identical).

    returns (losses, best_epoch, val_aurocs).
    """
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    features = features.to(device)
    labels = labels.to(device)
    n = features.size(0)

    use_val = val_features is not None and val_features.numel() > 0
    if use_val:
        val_features = val_features.to(device)
        val_labels_np = val_labels.cpu().numpy()

    losses, val_aurocs = [], []
    best_val = -float("inf")
    best_epoch = -1
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        probe.train()
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

        if use_val:
            probe.eval()
            with torch.no_grad():
                v_logits = probe(val_features)
                v_probs = torch.softmax(v_logits, dim=-1).cpu().numpy()
            val_auroc = _macro_auroc_from_probs(val_labels_np, v_probs)
            val_aurocs.append(val_auroc)
            if val_auroc > best_val:
                best_val = val_auroc
                best_epoch = epoch
                best_state = copy.deepcopy(probe.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}  "
                    f"val_macroAUROC={val_auroc:.4f}  "
                    f"(best={best_val:.4f}@ep{best_epoch + 1})"
                )
            if patience > 0 and no_improve >= patience:
                print(f"  [early-stop] epoch {epoch + 1}: val 무개선 {patience} epoch → 중단")
                break
        else:
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    if use_val and best_state is not None:
        probe.load_state_dict(best_state)
        print(f"  Restored best-val probe (macroAUROC={best_val:.4f} @ep{best_epoch + 1})")
    return losses, best_epoch, val_aurocs


@torch.no_grad()
def _eval_probe_cached(probe, features, labels, device):
    """미리 추출된 cached feature 로 probe 평가 (evaluate_linear_probe 와 동일 산출)."""
    probe.to(device).eval()
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
    ddp_module=None,
    val_batches: list[tuple[PackedBatch, torch.Tensor]] | None = None,
    patience: int = 0,
    use_ddp: bool = False,
    use_val: bool = False,
) -> list[float]:
    """LoRA fine-tune (multi-class, CrossEntropy).

    ``ddp_module`` 가 주어지면(torchrun) encode→pool→probe 를 DDP 래퍼 forward 로
    호출해 grad all-reduce 가 등록되게 한다. wrap_lora_ddp 는 logits 만 반환하므로
    CrossEntropy loss 는 모듈 밖에서 그대로 적용한다(BCE/CE 무관 동일 패턴).
    ddp_module=None 이면 기존 단일 GPU 경로 그대로(수치·결과 불변).

    ``use_val=True`` 이면 매 epoch 후 (rank0 에서) val macro-AUROC 를 재고 best-val
    LoRA+probe state 를 보관한다. ``patience>0`` 이면 early stop 하되, DDP(lora)
    에서는 rank0 의 stop 결정을 ``broadcast_should_stop`` 으로 전 rank 에 전파해 함께
    break(desync 방지). 학습 종료 시 best-val state 를 복원한다(rank0).
    ⚠ ``use_val`` 은 전 rank 에서 동일해야 한다 — early-stop broadcast 는 collective
    라 일부 rank 만 호출하면 hang. ``val_batches`` 는 rank0 만 있으면 되고(non-rank0
    는 None 가능), 실제 평가는 ``is_main()`` 에서만 수행한다.
    val 이 없으면(use_val=False) 기존 고정-epoch·마지막 상태 동작(불변).
    """
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

    best_val = -float("inf")
    best_epoch = -1
    best_probe_state = None
    best_model_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.model.train()
        probe.train()
        if ddp_module is not None:
            ddp_module.train()
        epoch_loss, n = 0.0, 0
        for batch, labels in train_batches:
            batch = model.batch_to_device(batch)
            if ddp_module is not None:
                # DDP: forward 가 LoRATrainModule(encode→pool→probe)를 타야
                # grad all-reduce 가 등록된다(probe 가 모듈 안에 포함됨).
                logits = ddp_module(batch)
            else:
                out = model.model(batch, task="masked")
                features = _mean_pool(out["encoded"], out["patch_mask"])
                logits = probe(features)

            # multi-class → CrossEntropy (labels long). 모듈 밖에서 적용.
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

        # ── val evaluation + best-ckpt 선택 (DDP: rank0 만 수행) ──
        # all-reduce + optimizer.step 후 전 rank 파라미터가 동일하므로 rank0 평가가
        # 전체를 대표한다. best-ckpt deepcopy 도 rank0 에서만(저장 주체).
        if use_val and is_main() and val_batches is not None:
            v = evaluate_lora(model, probe, val_batches, device)
            val_auroc = _macro_auroc_from_probs(v["y_true"], v["y_probs"])
            if val_auroc > best_val:
                best_val = val_auroc
                best_epoch = epoch
                best_probe_state = copy.deepcopy(probe.state_dict())
                best_model_state = copy.deepcopy(model.model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(
                    f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}  "
                    f"val_macroAUROC={val_auroc:.4f}  "
                    f"(best={best_val:.4f}@ep{best_epoch + 1})"
                )
        elif (epoch + 1) % 5 == 0 or epoch == 0:
            if is_main():
                print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

        # ── early stopping (DDP: rank0 결정을 broadcast 해 함께 break) ──
        if use_val and patience > 0:
            stop_local = is_main() and no_improve >= patience
            stop = (
                broadcast_should_stop(stop_local, device)
                if (use_ddp and ddp_module is not None) else stop_local
            )
            if stop:
                if is_main():
                    print(
                        f"  [early-stop] epoch {epoch + 1}: val 무개선 "
                        f"{patience} epoch → 중단"
                    )
                break

    # ── best-val state 복원 (rank0; non-rank0 는 호출측에서 학습 후 종료) ──
    if use_val and is_main():
        if best_probe_state is not None:
            probe.load_state_dict(best_probe_state)
            model.model.load_state_dict(best_model_state)
            print(
                f"  Restored best-val LoRA+probe "
                f"(macroAUROC={best_val:.4f} @ep{best_epoch + 1})"
            )
        else:
            print("  WARNING: no best-val ckpt captured; using last epoch.",
                  file=sys.stderr)

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


# ── Val 선택 지표 (macro OvR AUROC) ──────────────────────────


def _macro_auroc_from_probs(
    y_true: np.ndarray,   # (N,) int
    y_probs: np.ndarray,  # (N, n_classes)
) -> float:
    """멀티클래스 macro AUROC (one-vs-rest). run_eval 의 auroc_macro 집계·본문 보고
    지표와 동일 정의 → val 모델선택(early stop)이 최종 보고 지표와 정합.

    한 클래스가 val 에 아예 없거나 전부이면(0/all-positive) 그 클래스는 제외하고
    나머지로 평균한다. 유효 클래스가 없으면 nan.
    """
    aurocs = []
    for cls_id in range(y_probs.shape[1]):
        binary = (y_true == cls_id).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        aurocs.append(float(compute_auroc(binary, y_probs[:, cls_id])))
    return float(np.mean(aurocs)) if aurocs else float("nan")


# ── 데이터 로딩 ──────────────────────────────────────────────


def _load_data(
    args,
) -> tuple[
    list[MultiSignalWindow], list[MultiSignalWindow], list[MultiSignalWindow]
]:
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

    # CLASS_NAMES/N_CLASSES 를 prepared data 의 metadata 에서 받아 source-agnostic 하게
    # 만든다 (VitalDB collapse = NSR/AF/SV/VA/BradyCond, MIMIC = SR/AF/STACH/SBRAD/AFLT).
    # main() 에서 _load_data 호출 이후 LinearProbe(N_CLASSES)·메트릭이 모두 이 값을 읽음.
    global CLASS_NAMES, N_CLASSES
    _cn = meta.get("class_names") or meta.get("classes")
    if _cn:
        CLASS_NAMES = list(_cn)
        N_CLASSES = len(CLASS_NAMES)

    train_data = data["train"]
    test_data = data["test"]
    val_data = data.get("val")  # prepare_data_vitaldb 가 저장하는 3-way val split

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

    train_windows = _pt_to_windows(train_data)
    test_windows = _pt_to_windows(test_data)

    if val_data is not None and len(val_data.get("labels", [])) > 0:
        val_windows = _pt_to_windows(val_data)
        print(f"  val split (from prepare_data): {len(val_windows)} samples")
    else:
        # Backward-compat: 구 산출물(val 키 없음/빈 fold) — train 에서 20% 동적 split.
        # patient 단위가 아닌 sample 단위(구 hypotension fallback 과 동일 정책) — 정식
        # 산출물은 prepare_data 의 patient-level val 을 쓰므로 이 경로는 legacy 안전망.
        warnings.warn(
            "data['val'] not found (or empty); falling back to a 20% dynamic "
            "split of train. Re-run prepare_data_vitaldb.py for a deterministic "
            "patient-level val split.",
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
    parser.add_argument("--patience", type=int, default=0,
                        help="val macro-AUROC 무개선 시 early stop epoch 수 "
                             "(0=early stop 없이 전 epoch 수행하되 best-val ckpt 는 "
                             "여전히 선택·복원). hypotension 관례: LP=0, LoRA=20")
    parser.add_argument("--val-split-seed", type=int, default=42,
                        help="data['val'] 부재 시(legacy) train 20%% 동적 split seed")
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

    # ── DDP (B안): torchrun 으로 실행되면 rank 별 GPU 핀 + device 강제.
    # torchrun 이 아니면 maybe_init_ddp()=None → 단일 GPU 경로(기존) 그대로.
    ddp_device = maybe_init_ddp()
    use_ddp = ddp_device is not None
    if use_ddp:
        device = ddp_device
        args.device = str(ddp_device)
        # 두 모드 모두 torchrun 지원: lora=grad all-reduce 데이터 병렬,
        # linear_probe=feature 추출 shard→gather(grad sync 없음). 저장은 rank0
        # 전담이라 결과파일 race 없음.
        if is_main():
            print(f"[DDP] world_size={ddp_world_size()}  device={device}")
    else:
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

    # ── 데이터 로드 (metadata 에서 CLASS_NAMES/N_CLASSES override) ──
    train_windows, val_windows, test_windows = _load_data(args)

    # CLASS_NAMES 는 _load_data 안에서 prepared metadata 로 갱신되므로 로드 후 출력.
    print(f"Mode: {args.mode} | Classes: {CLASS_NAMES}")

    def _print_dist(name, windows):
        print(f"  {name}: {len(windows)} samples")
        for cls_id, cls_name in enumerate(CLASS_NAMES):
            cnt = sum(1 for w in windows if w.label == cls_id)
            print(f"    {cls_name}: {cnt}")

    _print_dist("Train", train_windows)
    _print_dist("Val", val_windows)
    _print_dist("Test", test_windows)

    if not train_windows or not val_windows or not test_windows:
        print("Insufficient data.", file=sys.stderr)
        sys.exit(1)

    # ── 학습 ──
    probe = LinearProbe(d_model, n_classes=N_CLASSES)

    if args.mode == "linear_probe":
        # linear_probe: frozen encoder feature 1회 추출 후 probe 만 학습. encoder
        # backprop 없음 → DDP grad sync 불필요 → torchrun 에선 feature 추출만 rank 별
        # shard 병렬화하고 gather(원본 순서 복원). train/test 각각 gather 1회 → 모든
        # rank 가 2 추출 참여 후 non-rank0 종료, rank0 만 probe 학습·평가·저장.
        # 단일 GPU(use_ddp=False)면 _extract_features_maybe_sharded→_stream_extract_features
        # + 동일 학습 루프 → 기존 train/evaluate_linear_probe 와 byte-identical.
        # train/val/test 모두 rank 별 shard→gather (frozen encoder, 순서 복원).
        # 전 rank 가 3 추출에 함께 참여해야 gather collective 가 성립 → non-rank0 는
        # 3 추출 이후 종료. best-val 모델선택은 rank0 가 cached feature 로 수행.
        train_features, train_labels = _extract_features_maybe_sharded(
            model, train_windows, args.batch_size, args.patch_size, device,
        )
        val_features, val_labels = _extract_features_maybe_sharded(
            model, val_windows, args.batch_size, args.patch_size, device,
        )
        test_features, test_labels = _extract_features_maybe_sharded(
            model, test_windows, args.batch_size, args.patch_size, device,
        )
        if use_ddp and not is_main():
            import torch.distributed as dist
            dist.destroy_process_group()
            return
        print(f"\nTraining LinearProbe (frozen encoder, d_model={d_model})...")
        train_losses, _best_ep, _val_aurocs = _fit_probe_cached(
            probe, train_features, train_labels,
            args.batch_size, args.epochs, args.lr, device,
            val_features=val_features, val_labels=val_labels,
            patience=args.patience,
        )
        print("\nEvaluating on test set with best-val probe...")
        metrics = _eval_probe_cached(probe, test_features, test_labels, device)

    elif args.mode == "lora":
        # lora: encoder fine-tune → feature 캐싱 불가, pre-built batches 필요.
        # 주의: 큰 윈도우면 RAM 위험 — ablation 은 linear_probe 만 사용.
        # ── DDP(B안): train window 를 rank 별로 분할 후 전 rank 최소 길이로 정렬
        #   (step 동기화). test 는 분할하지 않는다(평가는 rank0 이 full set 으로 수행).
        if use_ddp:
            n_full = len(train_windows)
            train_windows = equalize_shard(shard_for_rank(train_windows))
            if is_main():
                print(
                    f"  [DDP] train shard: {n_full} → {len(train_windows)}"
                    f"/rank × {ddp_world_size()} ranks"
                )
            if len(train_windows) == 0:
                # world_size 가 train window 수보다 크면 빈 shard → random-init LoRA
                # 가 "결과"로 저장될 위험. 명시적으로 차단.
                if is_main():
                    print(
                        "ERROR: DDP train shard 가 비었습니다 (nproc_per_node 가 "
                        "train window 수보다 큼). nproc 를 줄이거나 단일 GPU 로 "
                        "실행하세요.",
                        file=sys.stderr,
                    )
                import torch.distributed as dist
                dist.destroy_process_group()
                sys.exit(2)
        first_sig = next(iter(train_windows[0].signals.values()))
        max_length = len(first_sig)
        train_batches = _make_batches(
            train_windows, args.batch_size, args.patch_size, max_length
        )
        n_lora = sum(p.numel() for p in model.lora_parameters())
        n_probe = sum(p.numel() for p in probe.parameters())
        print(f"\nTraining LoRA + Probe (rank={args.lora_rank}, "
              f"LoRA={n_lora:,} + Probe={n_probe:,} params)...")
        # DDP: encode→pool→probe 를 한 forward 로 묶어 grad all-reduce 등록. 단일 GPU
        # 면 use_ddp=False → ddp_module=None → 기존 forward 경로 그대로(불변).
        ddp_module = wrap_lora_ddp(model.model, probe) if use_ddp else None
        # val batch 는 매 epoch rank0 만 평가(best-ckpt 선택)에 쓴다. non-rank0 는
        # 만들 필요 없음(메모리 절약) — train_lora 에서 is_main() 만 val 을 본다.
        val_batches = (
            _make_batches(val_windows, args.batch_size, args.patch_size, max_length)
            if is_main() else None
        )
        train_losses = train_lora(
            model, probe, train_batches, args.epochs, args.lr, device,
            ddp_module=ddp_module,
            val_batches=val_batches, patience=args.patience, use_ddp=use_ddp,
            use_val=True,
        )
        # DDP: 학습 종료 동기화 후 test/저장은 rank0 전담 → non-rank0 는 정리·종료.
        if use_ddp:
            import torch.distributed as dist
            dist.barrier()
            if not is_main():
                dist.destroy_process_group()
                return
        test_batches = _make_batches(
            test_windows, args.batch_size, args.patch_size, max_length
        )
        print("\nEvaluating on test set with best-val ckpt...")
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
    # fold suffix: n_folds>1 동시 실행 시 JSON 충돌(torn-file) 방지.
    fold_suffix = f"_fold{args.fold}" if int(args.n_folds) > 1 else ""
    results_path = out_dir / f"arrhythmia_results_{args.mode}{fold_suffix}.json"
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

    # DDP: rank0 의 프로세스 그룹 정리(non-rank0 는 학습 직후 이미 정리·종료).
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
