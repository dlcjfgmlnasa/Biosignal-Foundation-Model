# -*- coding:utf-8 -*-
"""Intracranial Hypertension Detection (ICP > 20mmHg).

MIMIC-III ICP 기반 두개내 고혈압 탐지 — Foundation model representation 평가.

2가지 모드:
  - linear_probe: Frozen encoder + LinearProbe
  - lora:         Frozen encoder + LoRA adapters + LinearProbe

입력: ICP + ECG/ABP/PPG 윈도우 → encoder → mean pool → LinearProbe
라벨: 미래 구간 ICP > 20mmHg ≥1분 지속 여부

사용법:
    python -m downstream.acute_event.intracranial_hypertension.run \
        --checkpoint best.pt \
        --data-path datasets/processed/ich/intracranial_hypertension_abp_icp_ecg_w1200s_h30min \
        --mode linear_probe --epochs 30
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.collate import PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import SIGNAL_KEY_TO_TYPE, get_global_spatial_id

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe
from downstream.window_task import make_window_batches, iter_window_batches
from downstream._eval_utils import dump_fold_predictions
from downstream._save_utils import (
    iter_prepared_split_chunks,
    load_prepared_split_chunked,
)
from downstream._ddp_utils import (
    ddp_enabled,
    ddp_rank,
    ddp_world_size,
    equalize_shard,
    gather_concat,
    is_main,
    maybe_init_ddp,
    shard_for_rank,
    wrap_lora_ddp,
)


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0

# v2: data.spatial_map 의 SSOT(SIGNAL_KEY_TO_TYPE) 사용. ICH 는 ECG/ABP/PPG/
# CVP/ICP 등 일부만 입력하지만, 번호 정의는 단일 소스로 통일한다(로컬 dict drift 방지).
SIGNAL_TYPE_INT: dict[str, int] = SIGNAL_KEY_TO_TYPE


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
    # 버그 수정(2026-06-17): 윈도우당 1 행 보장(공유 헬퍼 위임). 예전엔 batch 의
    # 여러 윈도우가 1 pack 행으로 합쳐져 extract_features 가 B≠n_windows 를 냈다.
    return make_window_batches(
        windows,
        batch_size,
        patch_size,
        to_samples=lambda w, idx: _make_samples(w["signals"], idx),
        get_label=lambda w: w["label"],
    )


# ── Mean pooling ─────────────────────────────────────────────


def _mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# ── Linear Probe ─────────────────────────────────────────────


@torch.no_grad()
def _stream_extract_features(
    model, windows, batch_size, patch_size, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter_window_batches 로 batch 를 하나씩 빌드·forward·폐기 (peak = batch 1개).

    make_window_batches 는 모든 batch 를 리스트로 미리 빌드 → 큰 윈도우
    (15min×3신호, row당 ~270k 샘플 → 배치당 ~수백 MB)에서 수백 GB RAM 으로 서버
    OOM 이 났다. 스트리밍하면 한 번에 batch 1개만 메모리에 두고 d_model feature
    (작음)만 누적한다. frozen encoder 라 feature 는 1회 추출이면 충분.

    returns (features (N, d_model) cpu, labels (N,) cpu float).
    """
    feats: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    for batch, labels in iter_window_batches(
        windows, batch_size, patch_size,
        to_samples=lambda w, idx: _make_samples(w["signals"], idx),
        get_label=lambda w: w["label"],
    ):
        f = model.extract_features(batch, pool="mean").detach().cpu()
        feats.append(f)
        labs.append(labels)
    if not feats:
        return torch.empty(0), torch.empty(0)
    return torch.cat(feats, dim=0), torch.cat(labs, dim=0).float()


def train_linear_probe(
    model, probe, train_windows, batch_size, patch_size, epochs, lr, device,
):
    # frozen encoder feature 를 스트리밍으로 1회만 추출(전체 batch 리스트 미빌드 →
    # OOM 회피) 후, probe 만 cached feature 로 학습한다(encoder forward 1패스).
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

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
        perm = torch.randperm(n, device=device)
        epoch_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            logits = probe(features[idx])
            loss = criterion(logits, labels[idx].unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        avg = epoch_loss / max(nb, 1)
        losses.append(avg)
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


@torch.no_grad()
def evaluate_linear_probe(model, probe, test_windows, batch_size, patch_size, device):
    probe.to(device).eval()
    features, labels = _stream_extract_features(
        model, test_windows, batch_size, patch_size, device,
    )
    logits = probe(features.to(device))
    scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return _compute_metrics(labels.numpy().astype(int), scores)


# ── linear_probe feature 추출 torchrun 병렬화 ────────────────


def _extract_features_maybe_sharded(
    model, windows, batch_size, patch_size, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """linear_probe feature 추출 (torchrun 병렬화).

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
        return torch.empty(0), torch.empty(0)
    parts = [g for g in gathered if g[0].numel() > 0]
    if not parts:
        return torch.empty(0), torch.empty(0)
    idx_all = torch.cat([g[0] for g in parts])
    feats_all = torch.cat([g[1] for g in parts], dim=0)
    labels_all = torch.cat([g[2] for g in parts], dim=0)
    order = torch.argsort(idx_all)
    return feats_all[order], labels_all[order]


def _fit_probe_cached(probe, features, labels, batch_size, epochs, lr, device):
    """미리 추출된 cached feature 로 probe 학습 (train_linear_probe 의 학습부와 동일,
    randperm 셔플 포함). 단일 GPU 에서 (추출 → 이 함수) 는 기존 train_linear_probe 와
    byte-identical (추출이 RNG 미소비 → randperm 시퀀스 동일)."""
    probe = probe.to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    features = features.to(device)
    labels = labels.to(device)
    n = features.size(0)
    losses = []
    probe.train()
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        epoch_loss, nb = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            logits = probe(features[idx])
            loss = criterion(logits, labels[idx].unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            nb += 1
        avg = epoch_loss / max(nb, 1)
        losses.append(avg)
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
    return losses


@torch.no_grad()
def _eval_probe_cached(probe, features, labels, device):
    """미리 추출된 cached feature 로 probe 평가 (evaluate_linear_probe 와 동일 산출)."""
    probe.to(device).eval()
    logits = probe(features.to(device))
    scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return _compute_metrics(labels.numpy().astype(int), scores)


# ── LoRA ─────────────────────────────────────────────────────


def train_lora(
    model, probe, train_batches, epochs, lr, device, gradient_clip=1.0,
    ddp_module=None,
):
    """LoRA fine-tune. ``ddp_module`` 가 주어지면(torchrun) encode→pool→probe 를
    DDP 래퍼의 forward 로 호출해 grad all-reduce 가 등록되게 한다. ddp_module=None
    이면 기존 단일 GPU 경로(``model.model(batch, task="masked")``)를 그대로 탄다
    (수치·결과 불변).
    """
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
        if (epoch + 1) % 1 == 0 or epoch == 0:
            if is_main():  # DDP: rank0 만 출력(비-DDP 면 항상 True → 불변)
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
    if not args.data_path:
        print("ERROR: --data-path required", file=sys.stderr)
        sys.exit(1)

    # data_path 는 단일 통합 .pt (back-compat) 또는 per-(fold,split)[_chunk]
    # prefix 묶음 (ablation runner). n_folds>1 이면 해당 fold 의 chunk 만 로드.
    load_fold = int(args.fold) if int(args.n_folds) > 1 else None
    print(f"\nLoading data: {args.data_path} (fold={load_fold})")
    data = load_prepared_split_chunked(args.data_path, fold=load_fold)
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    print(f"  Signals: {meta.get('input_signals', '?')}")
    print(f"  Window: {meta.get('window_sec', '?')}s, "
          f"Horizon: {meta.get('horizon_sec', 0) / 60:.0f}min")

    def _to_windows(split_data):
        windows = []
        labels = split_data["labels"]
        sig_types = list(split_data["signals"].keys())
        subject_ids = split_data.get("subject_ids", None)
        case_ids = split_data.get("case_ids", None)
        for i in range(len(labels)):
            signals = {st: split_data["signals"][st][i].numpy() for st in sig_types}
            windows.append({
                "signals": signals,
                "label": int(labels[i].item()),
                "case_id": case_ids[i] if case_ids is not None else 0,
                "subject_id": subject_ids[i] if subject_ids is not None else "unknown",
            })
        return windows

    train_w = _to_windows(data["train"])
    test_w = _to_windows(data["test"])
    val_data = data.get("val")  # prepare_ich_sweep 가 저장하는 val split
    if val_data is not None:
        val_w = _to_windows(val_data)
        print(f"  val split (from prepare_data): {len(val_w)} windows")
    else:
        # Backward-compat: legacy(no-val) 산출물 — train 에서 20% 동적 split.
        warnings.warn(
            "data['val'] not found; falling back to a 20% dynamic split of train. "
            "Re-run prepare_data.py to get a deterministic val split.",
            stacklevel=2,
        )
        seed = getattr(args, "val_split_seed", 42)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(train_w))
        rng.shuffle(idx)
        n_val = max(1, int(len(train_w) * 0.2))
        vset = set(idx[:n_val].tolist())
        val_w = [w for i, w in enumerate(train_w) if i in vset]
        train_w = [w for i, w in enumerate(train_w) if i not in vset]
        print(f"  val split (dynamic, seed={seed}): {len(val_w)} windows")
    return train_w, val_w, test_w


# ── OOM 하드닝: 스트리밍 + DDP shard 로드 ──────────────────────
# 기존 _load_data 는 load_prepared_split_chunked(전체 chunk concat) + _to_windows
# (윈도우별 numpy 복제)로 **rank 마다 전체 split 2벌**을 materialize → 4-GPU × 큰
# window(1200s) 에서 host-RAM OOM(SIGKILL). 아래는 chunk 를 하나씩 스트리밍하며
# shard=True(&DDP) 면 이 rank 의 stride(전역 idx % world == rank, shard_for_rank 와
# 동일 선택)만 materialize → peak = chunk 1개 + 이 rank 몫(1/world).


def _stream_load_windows(
    data_path: str,
    fold: int | None,
    split: str,
    shard: bool,
) -> tuple[list[dict], list[int]]:
    """chunk 스트리밍으로 window dict 리스트 생성 (+ 전역 인덱스 gidx).

    shard=True & DDP 면 이 rank 의 stride 만 유지(1/world RAM). 비-DDP/shard=False
    면 전체를 순서대로 materialize(gidx=range) → 기존 _to_windows 와 동일 순서.
    val 없는 legacy 산출물이면 빈 리스트 반환(호출측 fallback).
    """
    do_shard = shard and ddp_enabled()
    world = ddp_world_size() if do_shard else 1
    rank = ddp_rank() if do_shard else 0
    windows: list[dict] = []
    gidx: list[int] = []
    g = 0
    for payload in iter_prepared_split_chunks(data_path, fold, split):
        sig = payload.get("signals") if isinstance(payload, dict) else None
        if not sig:
            continue
        labels = payload["labels"]
        sig_types = list(sig.keys())
        subj = payload.get("subject_ids")
        case = payload.get("case_ids")
        n = len(labels)
        for i in range(n):
            if g % world == rank:
                windows.append({
                    "signals": {st: sig[st][i].numpy() for st in sig_types},
                    "label": int(labels[i].item()),
                    "case_id": case[i] if case is not None else 0,
                    "subject_id": subj[i] if subj is not None else "unknown",
                })
                gidx.append(g)
            g += 1
        del sig, payload
    return windows, gidx


@torch.no_grad()
def _extract_features_presharded(
    model, windows, gidx, batch_size, patch_size, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """windows 가 **이미 이 rank 의 shard**(gidx=전역 인덱스)일 때의 feature 추출.

    _extract_features_maybe_sharded 와 동일하되 내부 재-shard 를 하지 않는다
    (windows 가 로드 시점에 stride 로 들어옴). 비-DDP 면 순서 추출 그대로,
    DDP 면 각 rank shard 추출 → gather_concat → rank0 가 gidx argsort 로 복원.
    non-main 은 gather 참여 후 빈 텐서 반환.
    """
    feats, labels = _stream_extract_features(
        model, windows, batch_size, patch_size, device,
    )
    if not ddp_enabled():
        return feats, labels
    local = [(torch.tensor(gidx, dtype=torch.long), feats, labels)]
    gathered = gather_concat(local)
    if not is_main():
        return torch.empty(0), torch.empty(0)
    parts = [g for g in gathered if g[0].numel() > 0]
    if not parts:
        return torch.empty(0), torch.empty(0)
    idx_all = torch.cat([g[0] for g in parts])
    feats_all = torch.cat([g[1] for g in parts], dim=0)
    labels_all = torch.cat([g[2] for g in parts], dim=0)
    order = torch.argsort(idx_all)
    return feats_all[order], labels_all[order]


# ── LOSO (Leave-One-Subject-Out) ─────────────────────────────


@torch.no_grad()
def _extract_all_embeddings(
    model,
    windows: list[dict],
    batch_size: int,
    patch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:  # (N_windows, d_model)
    """전체 window에 대해 encoder mean-pool feature를 한 번에 추출한다."""
    batches = _make_batches(windows, batch_size, patch_size, max_length)
    feats = []
    for batch, _ in batches:
        f = model.extract_features(batch, pool="mean").to(device)
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)  # (N, d_model)


def _train_probe_on_features(
    features: torch.Tensor,  # (N, d_model)
    labels: torch.Tensor,  # (N,) float
    d_model: int,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> LinearProbe:
    probe = LinearProbe(d_model, n_classes=1).to(device)
    probe.train()
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    n = features.size(0)
    features = features.to(device)
    labels = labels.to(device)
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i: i + batch_size]
            logits = probe(features[idx])
            loss = criterion(logits, labels[idx].unsqueeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return probe


@torch.no_grad()
def _predict_on_features(
    probe: LinearProbe,
    features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    probe.eval()
    logits = probe(features.to(device))
    return torch.sigmoid(logits).squeeze(-1).cpu().numpy()


def _bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iter: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        ys = y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        scores.append(compute_auroc(yt, ys))
    if not scores:
        return (float("nan"), float("nan"))
    scores = np.array(scores)
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def run_loso(
    model,
    d_model: int,
    windows: list[dict],
    args,
    out_dir: Path,
    device: torch.device,
) -> None:
    """LOSO 평가: encoder embedding 캐싱 → per-patient fold 학습/평가 → aggregate."""
    first_sig = next(iter(windows[0]["signals"].values()))
    max_length = len(first_sig)

    # ── Embedding 캐시 ──
    cache_path = out_dir / "embeddings.pt"
    if cache_path.exists():
        print(f"\nLoading cached embeddings: {cache_path}")
        cache = torch.load(cache_path, weights_only=False)
        features = cache["features"]  # (N, d_model)
        subject_ids = cache["subject_ids"]
        labels = cache["labels"]
    else:
        print(f"\nExtracting embeddings for {len(windows)} windows "
              f"(one-time cache)...")
        features = _extract_all_embeddings(
            model, windows, args.batch_size, args.patch_size, max_length, device,
        )
        subject_ids = [w["subject_id"] for w in windows]
        labels = torch.tensor([w["label"] for w in windows], dtype=torch.float32)
        torch.save(
            {"features": features, "subject_ids": subject_ids, "labels": labels},
            cache_path,
        )
        print(f"  Saved embeddings cache: {cache_path} "
              f"(shape={tuple(features.shape)})")

    unique_subjects = sorted(set(subject_ids))
    print(f"\nLOSO: {len(unique_subjects)} patients, {len(windows)} windows")

    sid_arr = np.array(subject_ids)
    y_arr = labels.numpy().astype(int)

    per_fold = []
    all_y_true = []
    all_y_score = []
    all_sid = []
    all_widx = []

    for fi, test_sid in enumerate(unique_subjects):
        test_mask = sid_arr == test_sid
        train_mask = ~test_mask
        n_test = int(test_mask.sum())
        n_train_pos = int(y_arr[train_mask].sum())
        n_test_pos = int(y_arr[test_mask].sum())

        # Train set에 양/음 모두 필요
        if y_arr[train_mask].sum() == 0 or y_arr[train_mask].sum() == train_mask.sum():
            print(f"  [fold {fi + 1}/{len(unique_subjects)}] {test_sid} "
                  f"skip (train imbalance)")
            continue

        train_feats = features[torch.from_numpy(train_mask)]
        train_labels = labels[torch.from_numpy(train_mask)]
        test_feats = features[torch.from_numpy(test_mask)]
        test_labels = labels[torch.from_numpy(test_mask)]

        probe = _train_probe_on_features(
            train_feats, train_labels, d_model,
            epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size, device=device,
        )
        y_score_fold = _predict_on_features(probe, test_feats, device)
        y_true_fold = test_labels.numpy().astype(int)

        fold_auroc = float("nan")
        if n_test_pos > 0 and n_test_pos < n_test:
            fold_auroc = compute_auroc(y_true_fold, y_score_fold)

        per_fold.append({
            "test_patient": test_sid,
            "auroc": fold_auroc,
            "n_windows": n_test,
            "n_positive": n_test_pos,
            "n_train": int(train_mask.sum()),
            "n_train_positive": n_train_pos,
        })

        all_y_true.append(y_true_fold)
        all_y_score.append(y_score_fold)
        all_sid.extend([test_sid] * n_test)
        all_widx.extend(list(np.where(test_mask)[0]))

        if (fi + 1) % 10 == 0 or fi == 0:
            print(f"  [fold {fi + 1}/{len(unique_subjects)}] {test_sid} "
                  f"n={n_test} pos={n_test_pos} auroc={fold_auroc:.4f}")

    y_true = np.concatenate(all_y_true)
    y_score = np.concatenate(all_y_score)

    agg = _compute_metrics(y_true, y_score)
    auroc_lo, auroc_hi = _bootstrap_auroc_ci(y_true, y_score, n_iter=1000)

    aggregate = {
        "auroc": agg["auroc"],
        "auprc": agg["auprc"],
        "f1_macro": agg["f1_macro"],
        "sensitivity": agg["sensitivity"],
        "specificity": agg["specificity"],
        "optimal_threshold": agg["optimal_threshold"],
        "n_total": agg["n_total"],
        "n_positive": agg["n_positive"],
        "prevalence": agg["prevalence"],
        "auroc_95ci": [auroc_lo, auroc_hi],
    }

    print(f"\n{'=' * 60}")
    print(f"  LOSO Aggregate ({len(per_fold)} folds)")
    print(f"{'=' * 60}")
    print(f"  AUROC:       {aggregate['auroc']:.4f} "
          f"[{auroc_lo:.4f}, {auroc_hi:.4f}]")
    print(f"  AUPRC:       {aggregate['auprc']:.4f}")
    print(f"  F1 (macro):  {aggregate['f1_macro']:.4f}")
    print(f"  Sensitivity: {aggregate['sensitivity']:.4f}")
    print(f"  Specificity: {aggregate['specificity']:.4f}")
    print(f"{'=' * 60}")

    # ── 저장 ──
    results = {
        "per_fold": per_fold,
        "aggregate": aggregate,
        "y_true": y_true.tolist(),
        "y_score": y_score.tolist(),
        "patient_ids": [str(s) for s in all_sid],  # subject-level grouping (all_sid 순서 = y_true)
        "config": {
            "task": "intracranial_hypertension_detection",
            "mode": args.mode,
            "eval_mode": "loso",
            "data_path": args.data_path,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "n_patients": len(unique_subjects),
            "n_folds": len(per_fold),
        },
    }
    results_path = out_dir / "loso_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")

    # LOSO 는 내부에서 이미 OOF 집계 — 단일 OOF .npz 로 저장 (fold_idx=-1).
    npz_path = dump_fold_predictions(
        out_dir, task="intracranial_hypertension", fold_idx=-1,
        n_folds=len(per_fold), y_true=y_true, y_score=y_score,
        patient_ids=[str(s) for s in all_sid],
    )
    print(f"Fold predictions: {npz_path}")

    pred_path = out_dir / "loso_predictions.csv"
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "window_id", "label", "score"])
        for sid, widx, lab, sc in zip(all_sid, all_widx, y_true, y_score):
            w.writerow([sid, int(widx), int(lab), float(sc)])
    print(f"Predictions: {pred_path}")

    roc_path = out_dir / "loso_roc.png"
    plot_roc_curve(y_true, y_score, roc_path, title="ICH LOSO ROC")
    print(f"ROC curve: {roc_path}")


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
    parser.add_argument("--fold", type=int, default=0,
                        help="standard 모드 fold 인덱스 (run_eval OOF 집계용)")
    parser.add_argument("--n-folds", type=int, default=1, help="전체 fold 수")
    parser.add_argument("--val-split-seed", type=int, default=42,
                        help="legacy(no-val) 산출물에서 동적 val split 시 seed.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--eval-mode", type=str, default="standard",
                        choices=["standard", "loso"],
                        help="standard: train/test split; loso: leave-one-subject-out CV")
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
        # 전담이라 결과파일 race 없음. (loso 는 아래에서 별도 차단.)
        if is_main():
            print(f"[DDP] world_size={ddp_world_size()}  device={device}")
    else:
        device = torch.device(args.device)

    from downstream.model_wrapper import DownstreamModelWrapper

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
    d_model = model.d_model

    if args.mode == "lora":
        model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    # ── 데이터 로드 (스트리밍 + DDP stride shard; OOM 하드닝) ──
    # 각 rank 가 전체 split 을 materialize 하면 4-GPU × 큰 window(1200s)에서 host-RAM
    # OOM(SIGKILL). linear_probe(DDP)=각 rank stride 로드→추출→gather 로 rank0 복원,
    # lora(DDP)=train 만 stride(학습 병렬)·val/test 는 rank0 full, 단일 GPU/LOSO=full.
    load_fold = int(args.fold) if int(args.n_folds) > 1 else None
    is_lora = args.mode == "lora"
    is_loso = args.eval_mode == "loso"
    tr_gidx = va_gidx = te_gidx = None
    print(f"\nLoading data: {args.data_path} (fold={load_fold})"
          f"{'  [DDP stride shard]' if (use_ddp and not is_loso) else '  [stream]'}")

    if use_ddp and not is_loso and not is_lora:
        # linear_probe DDP: 3 split 모두 stride 샤딩 → 추출 후 gather 로 rank0 복원.
        train_windows, tr_gidx = _stream_load_windows(args.data_path, load_fold, "train", shard=True)
        val_windows, va_gidx = _stream_load_windows(args.data_path, load_fold, "val", shard=True)
        test_windows, te_gidx = _stream_load_windows(args.data_path, load_fold, "test", shard=True)
        if not val_windows:
            if is_main():
                print("ERROR: DDP 는 val 없는 legacy 산출물 미지원 — prepare_data 로 val "
                      "split 저장하거나 단일 GPU(NPROC=1)로 실행.", file=sys.stderr)
            import torch.distributed as dist
            dist.destroy_process_group(); sys.exit(2)
    elif use_ddp and is_lora:
        # lora DDP: train 만 stride(데이터 병렬)·val/test 는 rank0 full(평가 전담).
        train_windows, tr_gidx = _stream_load_windows(args.data_path, load_fold, "train", shard=True)
        if is_main():
            val_windows, _ = _stream_load_windows(args.data_path, load_fold, "val", shard=False)
            test_windows, _ = _stream_load_windows(args.data_path, load_fold, "test", shard=False)
        else:
            val_windows, test_windows = [], []
    else:
        # 단일 GPU / LOSO: full 스트리밍 로드 (기존 _to_windows 와 동일 순서).
        train_windows, _ = _stream_load_windows(args.data_path, load_fold, "train", shard=False)
        val_windows, _ = _stream_load_windows(args.data_path, load_fold, "val", shard=False)
        test_windows, _ = _stream_load_windows(args.data_path, load_fold, "test", shard=False)
        if not val_windows:
            # legacy val-missing fallback (단일 GPU 만) — train 20% 동적 split.
            seed = getattr(args, "val_split_seed", 42)
            rng = np.random.default_rng(seed)
            idx = np.arange(len(train_windows)); rng.shuffle(idx)
            n_val = max(1, int(len(train_windows) * 0.2))
            vset = set(idx[:n_val].tolist())
            val_windows = [w for i, w in enumerate(train_windows) if i in vset]
            train_windows = [w for i, w in enumerate(train_windows) if i not in vset]
            print(f"  val split (dynamic, seed={seed}): {len(val_windows)} windows")

    if train_windows:
        print(f"  Signals: {list(train_windows[0]['signals'].keys())}")
    _sh = "  [rank shard]" if (use_ddp and not is_loso) else ""
    n_pos_train = sum(1 for w in train_windows if w["label"] == 1)
    n_pos_val = sum(1 for w in val_windows if w["label"] == 1)
    n_pos_test = sum(1 for w in test_windows if w["label"] == 1)
    print(f"  Train: {len(train_windows)} ({n_pos_train} ICH){_sh}")
    print(f"  Val:   {len(val_windows)} ({n_pos_val} ICH){_sh}")
    print(f"  Test:  {len(test_windows)} ({n_pos_test} ICH){_sh}")

    # ── LOSO 모드 분기 ──
    if args.eval_mode == "loso":
        if use_ddp:
            # LOSO 는 per-subject fold 구조라 단일 feature shard 병렬화와 맞지 않는다.
            # 단일 GPU 로 실행하라(torchrun 미지원).
            if is_main():
                print(
                    "ERROR: LOSO 모드는 torchrun/DDP 미지원 — 단일 GPU(`python -m ...`)"
                    "로 실행하세요.",
                    file=sys.stderr,
                )
            import torch.distributed as dist
            dist.destroy_process_group()
            sys.exit(2)
        if args.mode != "linear_probe":
            print("ERROR: LOSO mode only supports --mode linear_probe", file=sys.stderr)
            sys.exit(1)
        all_windows = train_windows + val_windows + test_windows
        run_loso(model, d_model, all_windows, args, out_dir, device)
        return

    probe = LinearProbe(d_model, n_classes=1)
    is_lora = args.mode == "lora"

    # ── 준비: linear_probe=frozen feature 캐싱 / lora=batch 빌드 + DDP shard ──
    # hypotension 과 동일하게 매 epoch val AUROC 로 best-ckpt 를 잡고, 마지막에
    # best-val 시점을 복원해 test 1회. 100 epochs 같은 긴 학습에서도 overfitting 이
    # test 지표를 해치지 않게 한다(두 task protocol 일치 → 비교성).
    max_length: int | None = None
    train_features = train_labels = None
    val_features = val_labels = None
    test_features = test_labels = None
    train_batches = None
    val_batches = None
    lora_params = None
    ddp_module = None

    if not is_lora:
        # frozen encoder feature 1회 추출(train/val/test). DDP 면 rank 별 shard 추출
        # → gather(원본 순서 복원) → rank0 만 probe 학습·평가. 단일 GPU 면 동일 경로.
        # windows 는 이미 이 rank 의 stride(DDP)/full(단일). gidx 로 gather 복원.
        train_features, train_labels = _extract_features_presharded(
            model, train_windows, tr_gidx, args.batch_size, args.patch_size, device,
        )
        val_features, val_labels = _extract_features_presharded(
            model, val_windows, va_gidx, args.batch_size, args.patch_size, device,
        )
        test_features, test_labels = _extract_features_presharded(
            model, test_windows, te_gidx, args.batch_size, args.patch_size, device,
        )
        if use_ddp and not is_main():
            import torch.distributed as dist
            dist.destroy_process_group()
            return
        probe = probe.to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        val_features = val_features.to(device)
        print(f"\nTraining LinearProbe (d_model={d_model}) with val best-ckpt...")
    else:
        # lora: encoder fine-tune → feature 캐싱 불가, pre-built batches 필요.
        # ── DDP(B안): train window 를 rank 별 분할 + 전 rank 최소 길이 정렬(step 동기화).
        #   val/test 는 분할하지 않는다(평가는 rank0 가 full set 으로 수행).
        if use_ddp:
            # train_windows 는 이미 이 rank 의 stride(로드 시 shard=True) → 재-shard 금지.
            # 전 rank 최소 길이로만 정렬(step 동기화). (val/test 는 rank0 full.)
            n_shard = len(train_windows)
            train_windows = equalize_shard(train_windows)
            if is_main():
                print(
                    f"  [DDP] train shard: {n_shard} → {len(train_windows)}"
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
        first_sig = next(iter(train_windows[0]["signals"].values()))
        max_length = len(first_sig)
        train_batches = _make_batches(
            train_windows, args.batch_size, args.patch_size, max_length,
        )
        # val batch 는 rank0 만 평가에 쓴다(분할 X — full val set).
        if is_main():
            val_batches = _make_batches(
                val_windows, args.batch_size, args.patch_size, max_length,
            )
        probe = probe.to(device)
        lora_params = model.lora_parameters()
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": args.lr},
                {"params": probe.parameters(), "lr": args.lr},
            ],
            weight_decay=0.01,
        )
        # DDP: encode→pool→probe 를 한 forward 로 묶어 grad all-reduce 등록. 단일 GPU
        # 면 use_ddp=False → ddp_module=None → 기존 _make/forward 경로 그대로(불변).
        ddp_module = wrap_lora_ddp(model.model, probe) if use_ddp else None
        n_lora = sum(p.numel() for p in model.lora_parameters())
        print(f"\nTraining LoRA + Probe (rank={args.lora_rank}, LoRA={n_lora:,}) "
              f"with val best-ckpt...")

    criterion = nn.BCEWithLogitsLoss()
    train_losses: list[float] = []
    best_val_auroc = -1.0
    best_epoch = -1
    best_probe_state: dict | None = None
    best_model_state: dict | None = None  # lora 에서만

    for epoch in range(args.epochs):
        probe.train()
        if is_lora:
            model.model.train()
            if ddp_module is not None:
                ddp_module.train()
        epoch_loss, n_steps = 0.0, 0

        if not is_lora:
            n = train_features.size(0)
            perm = torch.randperm(n, device=device)
            for i in range(0, n, args.batch_size):
                idx = perm[i: i + args.batch_size]
                logits = probe(train_features[idx])
                loss = criterion(logits, train_labels[idx].unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1
        else:
            for batch, labels in train_batches:
                batch = model.batch_to_device(batch)
                if ddp_module is not None:
                    logits = ddp_module(batch)
                else:
                    out = model.model(batch, task="masked")
                    features = _mean_pool(out["encoded"], out["patch_mask"])
                    logits = probe(features)
                loss = criterion(logits, labels.to(device).unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    lora_params + list(probe.parameters()), 1.0,
                )
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1

        avg = epoch_loss / max(n_steps, 1)
        train_losses.append(avg)

        # ── val best-ckpt (DDP: rank0 만; all-reduce 후 전 rank 파라미터 동일) ──
        if is_main():
            probe.eval()
            if not is_lora:
                vm = _eval_probe_cached(probe, val_features, val_labels, device)
            else:
                model.model.eval()
                vm = evaluate_lora(model, probe, val_batches, device)
            val_auroc = float(vm["auroc"])
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch
                best_probe_state = copy.deepcopy(probe.state_dict())
                if is_lora:
                    best_model_state = copy.deepcopy(model.model.state_dict())
            if (epoch + 1) % 1 == 0 or epoch == 0 or epoch == args.epochs - 1:
                print(
                    f"  Epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}  "
                    f"val_auroc={val_auroc:.4f}  "
                    f"(best={best_val_auroc:.4f}@ep{best_epoch + 1})"
                )

    # DDP(lora): 학습 종료 동기화 후 test/저장은 rank0 전담 → non-rank0 정리·종료.
    # (linear_probe non-main 은 위 feature 추출 직후 이미 종료했다.)
    if use_ddp and is_lora:
        import torch.distributed as dist
        dist.barrier()
        if not is_main():
            dist.destroy_process_group()
            return

    # ── best-val ckpt 복원 후 test 1회 ──
    if best_probe_state is None:
        print("WARNING: no best ckpt captured; using last epoch.", file=sys.stderr)
    else:
        probe.load_state_dict(best_probe_state)
        if is_lora and best_model_state is not None:
            model.model.load_state_dict(best_model_state)

    print("\nEvaluating on test set with best-val ckpt...")
    if not is_lora:
        metrics = _eval_probe_cached(probe, test_features, test_labels, device)
    else:
        test_batches = _make_batches(
            test_windows, args.batch_size, args.patch_size, max_length,
        )
        metrics = evaluate_lora(model, probe, test_batches, device)

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")
    # subject-level grouping id (test_windows 순서 = 예측 순서).
    patient_ids = [str(w.get("subject_id", "unknown")) for w in test_windows]

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

    # fold suffix: n_folds>1 동시 실행 시 ROC PNG/JSON 충돌(torn-file) 방지.
    fold_suffix = f"_fold{args.fold}" if int(args.n_folds) > 1 else ""
    roc_path = out_dir / f"ich_roc_{args.mode}{fold_suffix}.png"
    plot_roc_curve(y_true, y_score, roc_path,
                   title=f"Intracranial Hypertension - {args.mode} ROC")
    print(f"\nROC curve: {roc_path}")

    results = {
        **metrics,
        "y_true": y_true.tolist(),
        "y_score": y_score.tolist(),
        "patient_ids": patient_ids,
        "train_losses": train_losses,
        "config": {
            "task": "intracranial_hypertension_detection",
            "mode": args.mode,
            "data_path": args.data_path,
            "epochs": args.epochs, "lr": args.lr,
        },
    }
    results_path = out_dir / f"ich_results_{args.mode}{fold_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")

    npz_path = dump_fold_predictions(
        out_dir, task="intracranial_hypertension", fold_idx=args.fold,
        n_folds=args.n_folds, y_true=y_true, y_score=y_score,
        patient_ids=patient_ids,
    )
    print(f"Fold predictions: {npz_path}")

    # DDP: rank0 의 프로세스 그룹 정리(non-rank0 는 학습 직후 이미 정리·종료).
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
