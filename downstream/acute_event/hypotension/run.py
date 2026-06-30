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
import contextlib
import copy
import gc
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from tqdm import tqdm
except ImportError:  # tqdm 미설치 시 no-op 폴백
    def tqdm(it, **_kwargs):  # type: ignore[no-redef]
        return it

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
from downstream.window_task import (
    infer_collate_mode,
    iter_window_batches,
    make_window_batches,
)
from downstream._eval_utils import dump_fold_predictions
from downstream._save_utils import (
    iter_prepared_split_chunks,
    load_prepared_split_chunked,
)
from downstream._ddp_utils import (
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
        # fp16 보존: prepare_data 가 fp16 로 저장한 window 를 그대로 들고 간다.
        # (PackCollate 의 padded_values 가 fp32 라 batch 단계에서 1회만 fp32 로
        #  올라가고, 모델 forward 직전 batch_to_device 에서 모델 dtype 캐스팅.)
        # 여기서 .float() 로 미리 업캐스트하면 윈도우마다 fp32 복사본이 생겨
        # 스트리밍 caching 의 peak 가 2배가 된다 → 제거.
        samples.append(
            BiosignalSample(
                values=torch.from_numpy(np.ascontiguousarray(signal)),
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
    """(batch, labels) 리스트 생성 — 윈도우당 정확히 1 행.

    버그 수정(2026-06-17): 예전 구현은 batch_size 개 윈도우 × N 신호를 한 번에
    FFD bin-packing collate 해 여러 윈도우가 1 행으로 합쳐졌고, 그 결과
    ``extract_features`` 의 행 단위 mean-pool 이 B≠n_windows 를 내놓아
    logits/labels 크기가 어긋났다. 이제 윈도우별 collate 후 stack 하는 공유
    헬퍼로 위임한다(``max_length`` 는 하위 호환용, 내부 자동 산정).
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


# ── bf16 autocast helper (lora 전용) ─────────────────────────
#
# lora train/eval 의 encoder forward + _mean_pool + probe + BCE 를 bf16 autocast
# 로 감싸 가속(사전학습 AMP 와 정합). bf16 은 loss scaling 불필요(GradScaler X).
# linear_probe(use_autocast=False)는 nullcontext → frozen feature 경로가 fp32
# 그대로 유지되어 결과 불변(기존 cell 재실행 회피). cpu 등 cuda 가 아니면
# enabled=False 로 autocast 를 안전 무력화한다.


def _maybe_bf16_autocast(device: torch.device, use_autocast: bool):
    if not use_autocast:
        return contextlib.nullcontext()
    return torch.amp.autocast(
        device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")
    )


@torch.no_grad()
def _eval_probe_cached(
    probe: LinearProbe,
    cached: list[tuple[torch.Tensor, torch.Tensor]],  # [(features(B,d), labels(B,))]
    device: torch.device,
) -> dict:
    """미리 추출된 frozen-encoder feature 로 probe 평가 (linear_probe 전용).

    ``evaluate_linear_probe`` 와 동일 산출(y_true/y_score 포함 metrics)이되,
    encoder forward 없이 캐시된 feature 만 사용한다.
    """
    probe.to(device).eval()
    all_labels, all_scores = [], []
    for features, labels in cached:
        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.detach().cpu().numpy())
        all_scores.append(probs)
    return _compute_metrics(np.concatenate(all_labels), np.concatenate(all_scores))


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
            out = model.model._encode(batch, task="masked")
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
        out = model.model._encode(batch, task="masked")
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


def _payload_to_windows(
    split_data: dict,
    input_keys: list[str],
) -> list[MultiSignalWindow]:
    """prepare_data split payload → MultiSignalWindow 리스트.

    ``signals[k][i].numpy()`` 는 (N,T) fp16 텐서의 row view 라 복사가 없다
    (fp16 보존). 업캐스트는 batch collate 단계에서 1회만 일어난다.
    """
    windows: list[MultiSignalWindow] = []
    labels_t = split_data["labels"]
    label_values_t = split_data["label_values"]
    case_ids = split_data.get("case_ids")
    sig = split_data["signals"]
    n = len(labels_t)
    for i in range(n):
        signals = {k: sig[k][i].numpy() for k in input_keys if k in sig}
        windows.append(
            MultiSignalWindow(
                signals=signals,
                label=int(labels_t[i].item()),
                label_value=float(label_values_t[i].item()),
                case_id=case_ids[i] if case_ids is not None else 0,
            )
        )
    return windows


@torch.no_grad()
def _cache_features_from_windows(
    windows: list[MultiSignalWindow],
    model,
    device: torch.device,
    batch_size: int,
    patch_size: int,
    progress: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """frozen-encoder feature 를 **batch_size 윈도우씩** 즉시 추출하고 batch 폐기.

    예전 구현은 split 전체 batch(fp32 packed)를 미리 빌드한 뒤 caching 해
    프로세스당 ~80 GB peak 를 만들었다. 여기서는 한 번에 batch_size 윈도우만
    fp32 packed 로 materialize → extract_features → (feats,labels) 누적 → batch
    즉시 free. feature 는 (B,d) 라 누적해도 작다(수십~수백 MB).

    행 i == window i == label i 정렬은 make_window_batches 가 보장하므로,
    cached 순서 == windows 순서 == 라벨/patient_id 순서 가 유지된다.
    """
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    rng = range(0, len(windows), batch_size)
    if progress:
        rng = tqdm(rng, desc="cache features", unit="batch", mininterval=0.5)
    for i in rng:
        chunk = windows[i : i + batch_size]
        batches = make_window_batches(
            chunk,
            batch_size,
            patch_size,
            to_samples=_multi_window_to_samples,
            get_label=lambda w: w.label,
        )
        for batch, labels in batches:
            feats = model.extract_features(batch, pool="mean").to(device)
            cached.append((feats.detach(), labels.to(device)))
        del batches
    return cached


@torch.no_grad()
def _extract_rows(
    windows: list[MultiSignalWindow],
    model,
    device: torch.device,
    batch_size: int,
    patch_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """windows → **per-row** (feat_cpu (d,), label_cpu scalar) 리스트 (스트리밍).

    ``_cache_features_from_windows`` 와 동일한 make_window_batches 추출 경로를 쓰되
    배치를 행으로 분해해 반환한다. transformer attention 은 row(window) 내부에서만
    일어나므로 per-window feature 는 배치 구성과 무관(독립) → rank 별 shard 추출 후
    rank0 가 재배치해도 값이 byte-identical. peak = batch 1개(스트리밍).
    """
    rows_feat: list[torch.Tensor] = []
    rows_lab: list[torch.Tensor] = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        batches = make_window_batches(
            chunk, batch_size, patch_size,
            to_samples=_multi_window_to_samples, get_label=lambda w: w.label,
        )
        for batch, labels in batches:
            feats = model.extract_features(batch, pool="mean").detach().cpu()  # (B,d)
            for r in range(feats.shape[0]):
                rows_feat.append(feats[r])
                rows_lab.append(labels[r])
        del batches
    return rows_feat, rows_lab


def _rebatch_rows(
    rows_feat: list[torch.Tensor],
    rows_lab: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """정렬된 per-row feature/label 을 batch_size 로 재배치 (cached 형식, device 로 올림).

    원본 순서 행을 batch_size 로 그룹핑 → single-GPU ``_cache_features_from_windows`` 의
    (feats(B,d), labels(B,)) 배치 시퀀스와 동일(per-window feature 독립이라 값도 동일).
    """
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    n = len(rows_feat)
    for i in range(0, n, batch_size):
        fb = torch.stack(rows_feat[i : i + batch_size]).to(device)
        lb = torch.stack(rows_lab[i : i + batch_size]).to(device)
        cached.append((fb, lb))
    return cached


def _cache_features_sharded(
    windows: list[MultiSignalWindow],
    model,
    device: torch.device,
    batch_size: int,
    patch_size: int,
    progress: bool = False,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """linear_probe feature 캐싱 (torchrun 병렬, non-stream 경로용).

    DDP 면 windows 를 rank 별 shard 로 나눠 **이 rank 몫만** per-row 추출(peak=batch
    1개) 후 gather_concat → 글로벌 idx argsort 로 원본 순서 복원 → batch_size 재배치.
    원본 windows 전체에 대한 single-GPU ``_cache_features_from_windows`` 와 동일한 배치
    시퀀스를 만든다(균일 bs 그룹핑). 비-DDP 면 그대로 위임(byte-identical).

    non-main rank 는 gather(collective)에만 참여하고 빈 리스트를 반환(직후 호출측 종료).
    """
    if not ddp_enabled():
        return _cache_features_from_windows(
            windows, model, device, batch_size, patch_size, progress=progress,
        )
    my_windows = shard_for_rank(windows)
    my_gidx = shard_for_rank(list(range(len(windows))))
    rf, rl = _extract_rows(my_windows, model, device, batch_size, patch_size)
    local = [(
        torch.tensor(my_gidx, dtype=torch.long),
        torch.stack(rf) if rf else torch.empty(0),
        torch.stack(rl) if rl else torch.empty(0),
    )]
    gathered = gather_concat(local)
    if not is_main():
        return []
    parts = [g for g in gathered if g[0].numel() > 0]
    if not parts:
        return []
    idx_all = torch.cat([g[0] for g in parts])
    feat_all = torch.cat([g[1] for g in parts], dim=0)
    lab_all = torch.cat([g[2] for g in parts], dim=0)
    order = torch.argsort(idx_all).tolist()
    rows_feat = [feat_all[o] for o in order]
    rows_lab = [lab_all[o] for o in order]
    return _rebatch_rows(rows_feat, rows_lab, batch_size, device)


def _stream_split_features(
    args,
    split: str,
    load_fold: int | None,
    model,
    device: torch.device,
) -> dict | None:
    """data_path 의 한 split 을 **chunk 단위 스트리밍**으로 feature 만 적재.

    chunk → windows(fp16 view) → per-batch collate(fp32 1개씩) → extract_features
    → (feats,labels) 누적 → chunk/배치 즉시 폐기. split 전체를 fp32 batch 나
    windows 로 동시에 들고 있지 않으므로 peak ≈ chunk 1개(≤ 수 GB) + batch 1개.

    DDP(torchrun): **청크마다** windows 를 rank 별 shard 로 나눠 이 rank 몫만 per-row
    추출 → gather_concat → 청크 내 글로벌 idx argsort 로 원본 순서 복원 → 청크를
    batch_size 로 재배치(single-GPU 의 청크별 _cache_features_from_windows 와 동일 배치
    시퀀스 → rank 수 무관·단일 GPU 와 byte-identical). gather 는 청크당 1회(collective)
    이고 청크 수는 전 rank 동일. non-main 은 gather 참여만 하고 빈 dict 반환(직후 종료).
    case_ids 도 같은 정렬로 복원해 cached 와 co-index 유지.

    Returns
    -------
    ``{"cached", "n_pos", "n_total", "case_ids"}`` 또는 chunk 부재 시 ``None``.
    case_ids/cached 는 windows 순서(=chunk·라벨 순서) 그대로라 정렬이 보존된다.
    """
    cached: list[tuple[torch.Tensor, torch.Tensor]] = []
    case_ids: list[str] = []
    n_pos = n_total = 0
    found = False
    input_keys: list[str] | None = None

    for payload in iter_prepared_split_chunks(args.data_path, load_fold, split):
        found = True
        if input_keys is None:
            input_keys = list(payload["signals"].keys())
            if not input_keys:
                print("ERROR: No signals in prepared data.", file=sys.stderr)
                sys.exit(1)
        windows = _payload_to_windows(payload, input_keys)
        del payload

        if not ddp_enabled():
            # 단일 GPU: 기존 경로 그대로 (byte-identical).
            for w in windows:
                n_total += 1
                n_pos += int(w.label == 1)
                case_ids.append(str(w.case_id))
            cached.extend(
                _cache_features_from_windows(
                    windows, model, device, args.batch_size, args.patch_size,
                )
            )
            del windows
            gc.collect()
            continue

        # ── DDP: 청크 내 shard 추출 → gather → 정렬 → 재배치 ──
        my_windows = shard_for_rank(windows)
        my_gidx = shard_for_rank(list(range(len(windows))))
        my_cid = [str(w.case_id) for w in my_windows]
        rf, rl = _extract_rows(
            my_windows, model, device, args.batch_size, args.patch_size,
        )
        local = [(
            torch.tensor(my_gidx, dtype=torch.long),
            torch.stack(rf) if rf else torch.empty(0),
            torch.stack(rl) if rl else torch.empty(0),
            my_cid,
        )]
        gathered = gather_concat(local)  # 청크당 1회 collective(전 rank)
        del windows
        gc.collect()
        if not is_main():
            continue
        parts = [g for g in gathered if g[0].numel() > 0]
        if not parts:
            continue
        c_idx = torch.cat([g[0] for g in parts])
        c_feat = torch.cat([g[1] for g in parts], dim=0)
        c_lab = torch.cat([g[2] for g in parts], dim=0)
        c_cid = [cid for g in parts for cid in g[3]]
        order = torch.argsort(c_idx).tolist()  # 청크 원본 순서 복원
        rows_feat = [c_feat[o] for o in order]
        rows_lab = [c_lab[o] for o in order]
        cached.extend(_rebatch_rows(rows_feat, rows_lab, args.batch_size, device))
        for o in order:
            n_total += 1
            n_pos += int(c_lab[o].item() == 1)
            case_ids.append(c_cid[o])

    if not found:
        return None
    if not is_main():
        # non-main: gather 에만 참여, 결과는 rank0 가 전담 → 빈 통계 반환.
        return {"cached": [], "n_pos": 0, "n_total": 0, "case_ids": []}
    return {
        "cached": cached,
        "n_pos": n_pos,
        "n_total": n_total,
        "case_ids": case_ids,
    }


# ── lora on-the-fly 배칭 (host-RAM OOM 회피) ─────────────────
#
# linear_probe 는 frozen-encoder feature 를 1회 추출·캐싱하면 epoch 간 불변이라
# 위 스트리밍 caching 으로 충분하다. 그러나 lora 는 encoder(LoRA adapter)가 매
# epoch 갱신되므로 feature 캐싱이 불가능 — 매 epoch 실제 window 로 forward 해야
# 한다. 예전 lora 경로는 fold 전체 batch 를 fp32 padded 로 **미리** 빌드해
# (~150 GB) OOM 을 냈다. 아래 헬퍼는 fp16 window 만 보유하고 batch 를 epoch 마다
# on-the-fly 로 빌드(iter_window_batches)해 peak 를 batch 1개로 누른다.


def _load_split_windows_streaming(
    args, split: str, load_fold: int | None,
) -> list[MultiSignalWindow]:
    """data_path 의 한 split 을 chunk 스트리밍으로 **fp16 window 리스트**로 적재.

    lora 전용: batch 는 만들지 않는다(OOM 회피). chunk 를 하나씩 ``torch.load`` →
    ``_payload_to_windows`` 로 window 만 누적한다. window 의 signals numpy 는 chunk
    fp16 텐서의 row view 라 추가 복사가 없다(업캐스트는 batch 빌드 시 1회). 단일
    통합 .pt(back-compat)도 ``iter_prepared_split_chunks`` 가 1회 yield 하므로 동작.
    """
    windows: list[MultiSignalWindow] = []
    input_keys: list[str] | None = None
    for payload in iter_prepared_split_chunks(args.data_path, load_fold, split):
        if input_keys is None:
            input_keys = list(payload["signals"].keys())
            if not input_keys:
                print("ERROR: No signals in prepared data.", file=sys.stderr)
                sys.exit(1)
        windows.extend(_payload_to_windows(payload, input_keys))
        del payload
    return windows


@torch.no_grad()
def _evaluate_lora_windows(
    model,
    probe: LinearProbe,
    windows: list[MultiSignalWindow],
    args,
    device: torch.device,
    collate_mode: str,
) -> dict:
    """lora val/test 평가 — windows 를 on-the-fly batch 로 빌드(미리 안 만듦).

    ``iter_window_batches`` 가 batch 를 하나씩 yield → forward → 즉시 폐기.
    feature 캐싱 불가(LoRA encoder 가 epoch 마다 갱신)하므로 매 호출 forward 한다.
    행/라벨 순서는 iter_window_batches 가 보존하므로 y_true/y_score 정렬 유지.
    """
    model.model.eval()
    probe.to(device).eval()
    all_labels, all_scores = [], []
    for batch, labels in iter_window_batches(
        windows, args.batch_size, args.patch_size,
        to_samples=_multi_window_to_samples, get_label=lambda w: w.label,
        collate_mode=collate_mode,
    ):
        batch = model.batch_to_device(batch)
        # 추론도 lora train 과 동일 bf16 autocast(backward 없음).
        with _maybe_bf16_autocast(device, True):
            out = model.model._encode(batch, task="masked")
            features = _mean_pool(out["encoded"], out["patch_mask"])
            logits = probe(features)
        # bf16 autocast 출력 logits → numpy 가 bfloat16 미지원이라 .float() 필수.
        probs = torch.sigmoid(logits).float().squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
    return _compute_metrics(np.concatenate(all_labels), np.concatenate(all_scores))


@torch.no_grad()
def _evaluate_lora_test_streaming(
    args,
    load_fold: int | None,
    model,
    probe: LinearProbe,
    device: torch.device,
    collate_mode: str,
) -> tuple[np.ndarray, np.ndarray, list[str]] | tuple[None, None, None]:
    """lora test 평가 — test chunk 를 **스트리밍**하며 on-the-fly batch eval.

    test split 전체를 RAM 에 올리지 않는다(peak ≈ chunk 1개 + batch). chunk →
    windows(fp16 view) → iter_window_batches → forward → 누적 → chunk 폐기.
    y_true/y_score/case_ids 를 chunk·window 순서대로 누적해 정렬을 보존한다
    (``dump_fold_predictions`` 의 patient 단위 집계·정렬용).
    """
    model.model.eval()
    probe.to(device).eval()
    all_labels, all_scores, case_ids = [], [], []
    input_keys: list[str] | None = None
    found = False
    for payload in iter_prepared_split_chunks(args.data_path, load_fold, "test"):
        found = True
        if input_keys is None:
            input_keys = list(payload["signals"].keys())
        windows = _payload_to_windows(payload, input_keys)
        del payload
        for w in windows:
            case_ids.append(str(w.case_id))
        for batch, labels in iter_window_batches(
            windows, args.batch_size, args.patch_size,
            to_samples=_multi_window_to_samples, get_label=lambda w: w.label,
            collate_mode=collate_mode,
        ):
            batch = model.batch_to_device(batch)
            # 추론도 lora train 과 동일 bf16 autocast(backward 없음).
            with _maybe_bf16_autocast(device, True):
                out = model.model._encode(batch, task="masked")
                features = _mean_pool(out["encoded"], out["patch_mask"])
                logits = probe(features)
            # bf16 autocast 출력 logits → numpy 가 bfloat16 미지원이라 .float() 필수.
            probs = torch.sigmoid(logits).float().squeeze(-1).cpu().numpy()
            all_labels.append(labels.numpy())
            all_scores.append(probs)
        del windows
        gc.collect()
    if not found:
        return None, None, None
    return np.concatenate(all_labels), np.concatenate(all_scores), case_ids


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
        # data_path 는 단일 통합 .pt (back-compat) 또는 per-(fold,split)[_chunk]
        # prefix 묶음 (ablation runner). n_folds>1 이면 해당 fold 의 chunk 만 로드.
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
            return _payload_to_windows(split_data, input_keys)

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
    parser.add_argument("--fold", type=int, default=0,
                        help="현재 fold 인덱스 (run_eval OOF 집계용 .npz 라벨)")
    parser.add_argument("--n-folds", type=int, default=1,
                        help="전체 fold 수")
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
            model.inject_lora(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
            )
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    sig_str = " + ".join(s.upper() for s in args.input_signals)
    print(f"Mode: {args.mode} | Input: {sig_str} | Window: {args.window_sec}s")

    is_lora = args.mode == "lora"
    load_fold = int(args.fold) if int(args.n_folds) > 1 else None

    # ── 데이터 로드 (Patch A: train/val/test 3-way) — 메모리 절약 로딩 ──
    # OOM 회피: 한 fold 의 train+val+test 를 통째로 fp32 batch 로 미리 빌드하던
    # 기존 경로(프로세스당 ~150 GB)를 제거한다.
    #  · linear_probe + data_path → chunk 단위 스트리밍으로 feature 만 적재
    #    (split 전체를 RAM 에 올리지 않음, peak ≈ chunk 1개 + batch 1개 + 작은 feature).
    #  · 그 외(lora / data_dir / 단일파일 / legacy-no-val) → _load_data 로 windows
    #    적재 후, linear 은 split 별 per-batch 스트리밍 caching + 즉시 free,
    #    lora 는 batch 빌드 후 원본 windows free.
    cached_train: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    cached_val: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    cached_test: list[tuple[torch.Tensor, torch.Tensor]] | None = None
    # lora on-the-fly: fp16 window 만 보유, batch 는 epoch 마다 빌드(미리 안 만듦).
    lora_train_windows: list[MultiSignalWindow] | None = None
    lora_val_windows: list[MultiSignalWindow] | None = None
    lora_test_windows: list[MultiSignalWindow] | None = None
    lora_collate_mode: str | None = None
    lora_stream_test = False  # data_path lora → test 는 끝에 chunk 스트리밍
    patient_ids: list[str]

    def _stat_line(name: str, n_pos: int, n_total: int) -> None:
        print(
            f"  {name}: {n_total} samples "
            f"({n_pos} hypo, {n_pos / max(n_total, 1) * 100:.1f}%)"
        )

    use_stream = (not is_lora) and bool(args.data_path)
    val_stream = None
    if use_stream:
        # val chunk 부재(legacy) 면 dynamic split 가 필요 → 비스트리밍 폴백.
        val_stream = _stream_split_features(args, "val", load_fold, model, device)
        if val_stream is None:
            use_stream = False

    if use_stream:
        print(
            f"\nStreaming prepared data per-chunk (fold={load_fold}) - "
            f"linear_probe (low-RAM)"
        )
        train_stream = _stream_split_features(
            args, "train", load_fold, model, device
        )
        test_stream = _stream_split_features(
            args, "test", load_fold, model, device
        )
        if train_stream is None or test_stream is None:
            print(
                "ERROR: train/test chunks missing in prepared data.",
                file=sys.stderr,
            )
            sys.exit(1)
        cached_train = train_stream["cached"]
        cached_val = val_stream["cached"]
        cached_test = test_stream["cached"]
        # patient(=case)-level grouping id — windows 순서 그대로(정렬 보존).
        patient_ids = test_stream["case_ids"]
        # DDP linear_probe: 추출(per-chunk gather) 완료 → non-rank0 는 여기서 정리·종료
        # (이후 collective 없음, probe 학습·평가·저장은 rank0 전담 → 결과파일 race 없음).
        if use_ddp and not is_main():
            import torch.distributed as dist
            dist.destroy_process_group()
            return
        _stat_line("Train", train_stream["n_pos"], train_stream["n_total"])
        _stat_line("Val  ", val_stream["n_pos"], val_stream["n_total"])
        _stat_line("Test ", test_stream["n_pos"], test_stream["n_total"])
        if not cached_train or not cached_test:
            print("Insufficient data.", file=sys.stderr)
            sys.exit(1)
        if not cached_val:
            print(
                "ERROR: val split is empty — cannot do best-ckpt selection.",
                file=sys.stderr,
            )
            sys.exit(1)
        del train_stream, val_stream, test_stream
        gc.collect()
    elif is_lora:
        # ── lora: fp16 window 만 보유, batch 는 epoch 마다 on-the-fly ──
        # encoder(LoRA)가 epoch 마다 갱신 → feature 캐싱 불가, 매 epoch 실제
        # window forward 필요. fold 전체 batch 를 fp32 padded 로 미리 빌드하던
        # 옛 경로(~150 GB)를 제거하고, fp16 window 만 들고 batch 는 epoch 마다
        # iter_window_batches 로 그 자리에서 빌드해 peak 를 batch 1개로 누른다.
        if args.data_path:
            # chunk 스트리밍으로 train/val window 만 적재(batch 미빌드). test 는
            # 끝에 chunk 스트리밍으로 평가 → 학습 중 RAM 에 안 들고 있는다.
            print(
                f"\nLoading prepared windows (fp16) for lora on-the-fly "
                f"batching (fold={load_fold})"
            )
            lora_train_windows = _load_split_windows_streaming(
                args, "train", load_fold
            )
            lora_val_windows = _load_split_windows_streaming(
                args, "val", load_fold
            )
            if not lora_train_windows:
                print(
                    "ERROR: train chunks missing in prepared data.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not lora_val_windows:
                # legacy(no-val) — train 에서 20% 동적 split (mirror _load_data).
                warnings.warn(
                    "data['val'] not found; falling back to a 20% dynamic split "
                    "of train. Re-run prepare_data.py for a deterministic val "
                    "split.",
                    stacklevel=2,
                )
                rng = np.random.default_rng(args.val_split_seed)
                idx = np.arange(len(lora_train_windows))
                rng.shuffle(idx)
                n_val = max(1, int(len(lora_train_windows) * 0.2))
                val_idx = set(idx[:n_val].tolist())
                new_train = [
                    w for i, w in enumerate(lora_train_windows) if i not in val_idx
                ]
                lora_val_windows = [
                    w for i, w in enumerate(lora_train_windows) if i in val_idx
                ]
                lora_train_windows = new_train
            lora_stream_test = True  # test 는 끝에 스트리밍 평가
            patient_ids = []  # 스트리밍 test eval 에서 window 순서대로 채움
            n_pos_tr = sum(1 for w in lora_train_windows if w.label == 1)
            n_pos_va = sum(1 for w in lora_val_windows if w.label == 1)
            _stat_line("Train", n_pos_tr, len(lora_train_windows))
            _stat_line("Val  ", n_pos_va, len(lora_val_windows))
            print("  Test : (streamed at eval time)")
        else:
            # data_dir / 단일파일 비-prefix smoke 경로 — _load_data 로 3-way 적재.
            # 작은 경로라 test window 도 보유(스트리밍 불필요).
            lora_train_windows, lora_val_windows, lora_test_windows = _load_data(
                args
            )
            n_pos_tr = sum(1 for w in lora_train_windows if w.label == 1)
            n_pos_va = sum(1 for w in lora_val_windows if w.label == 1)
            n_pos_te = sum(1 for w in lora_test_windows if w.label == 1)
            _stat_line("Train", n_pos_tr, len(lora_train_windows))
            _stat_line("Val  ", n_pos_va, len(lora_val_windows))
            _stat_line("Test ", n_pos_te, len(lora_test_windows))
            if not lora_train_windows or not lora_test_windows:
                print("Insufficient data.", file=sys.stderr)
                sys.exit(1)
            if not lora_val_windows:
                print(
                    "ERROR: val split is empty even after fallback — cannot do "
                    "best-ckpt selection.",
                    file=sys.stderr,
                )
                sys.exit(1)
            # test case-order grouping id — 정렬 보존(스트리밍과 동일 순서).
            patient_ids = [str(w.case_id) for w in lora_test_windows]
        # collate_mode 는 학습 시작 전 1회 계산해 재스캔 비용 제거(결과 동일).
        # 분할 전 full windows 로 계산해야 모든 rank 가 같은 mode 를 쓴다.
        lora_collate_mode = infer_collate_mode(
            lora_train_windows, _multi_window_to_samples
        )
        # DDP: train window 를 rank 별로 분할 후 전 rank 최소 길이로 정렬(step 동기화).
        # val/test 는 분할하지 않는다(평가는 rank0 이 full set 으로 수행).
        if use_ddp:
            n_full = len(lora_train_windows)
            lora_train_windows = equalize_shard(shard_for_rank(lora_train_windows))
            if is_main():
                print(
                    f"  [DDP] train shard: {n_full} → {len(lora_train_windows)}"
                    f"/rank × {ddp_world_size()} ranks"
                )
            # world_size 가 train window 수보다 크면 equalize_shard 가 전 rank 를
            # n_min=0 으로 만들어 학습이 조용히 no-op 되고 random-init LoRA 가
            # "best" 로 저장될 수 있다(model-architect 지적). 명시적으로 차단.
            if len(lora_train_windows) == 0:
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
    else:
        # ── linear_probe (non-stream: data_dir / 단일파일 / legacy-no-val) ──
        train_labeled, val_labeled, test_labeled = _load_data(args)

        def _pos_stats(name: str, ws: list[MultiSignalWindow]) -> None:
            n_pos = sum(1 for lw in ws if lw.label == 1)
            _stat_line(name, n_pos, len(ws))

        _pos_stats("Train", train_labeled)
        _pos_stats("Val  ", val_labeled)
        _pos_stats("Test ", test_labeled)

        if not train_labeled or not test_labeled:
            print("Insufficient data.", file=sys.stderr)
            sys.exit(1)
        if not val_labeled:
            print(
                "ERROR: val split is empty even after fallback — cannot do "
                "best-ckpt selection.",
                file=sys.stderr,
            )
            sys.exit(1)

        # test case-order grouping id — window free 전에 미리 추출 (정렬 보존).
        patient_ids = [str(w.case_id) for w in test_labeled]

        # linear_probe: split 별 per-batch 스트리밍 caching + 즉시 free.
        # DDP 면 _cache_features_sharded 가 rank 별 shard 추출→gather→정렬→재배치로
        # 전체(원본 순서) cached 를 rank0 에 구성(균일 bs 배치 = single-GPU 와 동일).
        # 비-DDP 면 _cache_features_from_windows 그대로(byte-identical).
        print("  Caching frozen-encoder features (streaming per-batch)...")
        cached_train = _cache_features_sharded(
            train_labeled, model, device,
            args.batch_size, args.patch_size, progress=True,
        )
        del train_labeled
        gc.collect()
        cached_val = _cache_features_sharded(
            val_labeled, model, device,
            args.batch_size, args.patch_size, progress=True,
        )
        del val_labeled
        gc.collect()
        cached_test = _cache_features_sharded(
            test_labeled, model, device,
            args.batch_size, args.patch_size, progress=True,
        )
        del test_labeled
        gc.collect()
        # DDP linear_probe: 추출(gather) 완료 → non-rank0 정리·종료(이후 collective 없음).
        if use_ddp and not is_main():
            import torch.distributed as dist
            dist.destroy_process_group()
            return

    # ── 학습 + Best-ckpt selection on val (Patch A) ──
    probe = LinearProbe(d_model, n_classes=1)

    if is_lora:
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
        # DDP: encode→pool→probe 를 한 forward 로 묶어 grad all-reduce 가 정상
        # 등록되게 한다. 단일 GPU 면 use_ddp=False → 기존 _encode 직접 호출 경로 유지.
        ddp_module = wrap_lora_ddp(model.model, probe) if use_ddp else None
    else:
        print(f"\nTraining LinearProbe (frozen encoder, d_model={d_model})...")
        probe = probe.to(device)
        probe.train()
        optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)

    criterion = nn.BCEWithLogitsLoss()
    train_losses: list[float] = []
    val_aurocs: list[float] = []
    best_val_auroc = -1.0
    best_epoch = -1
    best_probe_state: dict | None = None
    best_model_state: dict | None = None  # LoRA mode 에서만 저장

    # 캐싱은 위 데이터 로드 단계에서 이미 완료(linear_probe). cached_* 는
    # frozen-encoder feature 라 epoch 간 불변 → encoder forward 반복 제거.
    # lora 는 epoch 마다 lora_train_windows 에서 batch 를 on-the-fly 로 빌드
    # (encoder/LoRA 가 epoch 마다 갱신 → feature 캐싱 불가).

    for epoch in range(args.epochs):
        # ── train one epoch ──
        probe.train()
        if is_lora:
            model.model.train()
            if ddp_module is not None:
                ddp_module.train()
        epoch_loss, n_steps = 0.0, 0

        # lora: window 를 batch_size 씩 잘라 그 자리에서 1 batch 빌드(on-the-fly)
        #       → forward/backward → batch 즉시 폐기 (peak = batch 1개).
        # linear: (features(B,d), labels) 캐시 — 이미 device 상, encoder forward X.
        if is_lora:
            train_iter = iter_window_batches(
                lora_train_windows, args.batch_size, args.patch_size,
                to_samples=_multi_window_to_samples,
                get_label=lambda w: w.label,
                collate_mode=lora_collate_mode,
            )
            # epoch 내 batch 진행 라이브 표시 (tqdm→stderr 즉시 flush).
            n_train_batches = (
                len(lora_train_windows) + args.batch_size - 1
            ) // args.batch_size
            train_iter = tqdm(
                train_iter, total=n_train_batches,
                desc=f"lora train ep{epoch + 1}/{args.epochs}",
                unit="batch", mininterval=2.0,
                disable=not is_main(),  # DDP: rank0 만 진행바 출력
            )
        else:
            train_iter = cached_train
        for first, labels in train_iter:
            target = labels.to(device).unsqueeze(-1)
            # lora: forward+loss(encoder/_mean_pool/probe/BCE) 를 bf16 autocast 로
            #       감싸 가속. backward/optimizer.step 은 autocast **밖**(bf16 은
            #       GradScaler 불필요). linear: nullcontext → fp32 cached feature
            #       경로 불변(재실행 회피).
            with _maybe_bf16_autocast(device, is_lora):
                if is_lora and ddp_module is not None:
                    # DDP: forward 가 LoRATrainModule(encode→pool→probe)를 타야
                    # grad all-reduce 가 등록된다. probe 가 모듈 안에 포함됨.
                    batch = model.batch_to_device(first)
                    logits = ddp_module(batch)
                    loss = criterion(logits, target)
                elif is_lora:
                    batch = model.batch_to_device(first)
                    out = model.model._encode(batch, task="masked")
                    features = _mean_pool(out["encoded"], out["patch_mask"])
                    logits = probe(features)
                    loss = criterion(logits, target)
                else:
                    features = first  # (B, d_model) cached feature
                    logits = probe(features)
                    loss = criterion(logits, target)
            optimizer.zero_grad()
            loss.backward()
            if is_lora:
                nn.utils.clip_grad_norm_(
                    lora_params + list(probe.parameters()), 1.0
                )
            optimizer.step()
            epoch_loss += loss.item()
            n_steps += 1
            if is_lora:
                # on-the-fly batch + activation 즉시 폐기 (host/GPU peak 억제).
                del batch, logits, loss, first
                if ddp_module is None:
                    # 단일 GPU lora elif 분기의 중간 activation(out/features)도 해제
                    # — OOM 하드닝 보존(memory project_downstream_oom_io_hardening).
                    del out, features
        del train_iter

        avg_loss = epoch_loss / max(n_steps, 1)
        train_losses.append(avg_loss)

        # ── val evaluation (DDP: rank0 만 수행) ──
        # all-reduce + optimizer.step 후 전 rank 파라미터가 동일하므로 rank0 평가가
        # 전체를 대표한다. best-ckpt 선택·deepcopy 도 rank0 에서만(저장 주체).
        if is_main():
            if is_lora:
                # val window 를 on-the-fly batch 로 평가(미리 batch 안 만듦).
                val_metrics = _evaluate_lora_windows(
                    model, probe, lora_val_windows, args, device, lora_collate_mode,
                )
            else:
                val_metrics = _eval_probe_cached(probe, cached_val, device)
            val_auroc = float(val_metrics["auroc"])
            val_aurocs.append(val_auroc)

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch
                # deepcopy to detach from current parameters (probe is small — cheap).
                best_probe_state = copy.deepcopy(probe.state_dict())
                if is_lora:
                    # LoRA params live inside model.model — copy entire encoder state.
                    best_model_state = copy.deepcopy(model.model.state_dict())

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
                print(
                    f"  Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}  "
                    f"val_auroc={val_auroc:.4f}  "
                    f"(best={best_val_auroc:.4f}@ep{best_epoch + 1})"
                )

    # DDP(lora): 학습 종료 동기화 후 test/저장은 rank0 전담 → non-rank0 정리·종료.
    # ⚠ linear_probe 에서는 non-rank0 가 feature 추출 직후(위 _stream/_cache 블록) 이미
    #   destroy+return 했으므로, 여기서 무조건 barrier 를 타면 rank0 가 혼자 호출해
    #   peer 를 NCCL timeout(30분)까지 기다리는 hang 이 된다. lora(전 rank 가 함께 학습)
    #   일 때만 동기화한다 — ich/cardiac_arrest 등 다른 task 와 동일 패턴.
    if use_ddp and is_lora:
        import torch.distributed as dist
        dist.barrier()
        if not is_main():
            dist.destroy_process_group()
            return

    # lora: 학습 종료 → train/val window 해제(더 이상 불필요) 후 test 평가.
    # test 는 fresh 스트리밍/보유 window 로 빌드하므로 peak 가 batch 1개로 유지.
    if is_lora:
        lora_train_windows = None
        lora_val_windows = None
        gc.collect()

    # ── Restore best ckpt and run test ONCE (Patch A) ──
    if best_probe_state is None:
        print("WARNING: no best ckpt captured; using last epoch.", file=sys.stderr)
    else:
        probe.load_state_dict(best_probe_state)
        if is_lora and best_model_state is not None:
            model.model.load_state_dict(best_model_state)

    print("\nEvaluating on test set with best-val ckpt...")
    if is_lora:
        if lora_stream_test:
            # data_path: test chunk 를 스트리밍하며 on-the-fly batch eval.
            yt, ys, patient_ids = _evaluate_lora_test_streaming(
                args, load_fold, model, probe, device, lora_collate_mode,
            )
            if yt is None:
                print(
                    "ERROR: test chunks missing in prepared data.",
                    file=sys.stderr,
                )
                sys.exit(1)
            metrics = _compute_metrics(yt, ys)
        else:
            # data_dir/단일파일: 보유한 test window 로 on-the-fly batch eval.
            metrics = _evaluate_lora_windows(
                model, probe, lora_test_windows, args, device, lora_collate_mode,
            )
            lora_test_windows = None
            gc.collect()
    else:
        metrics = _eval_probe_cached(probe, cached_test, device)

    # ── 결과 출력 ──
    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")
    # patient(=case)-level grouping id(`patient_ids`)는 test window 순서 그대로다:
    # linear=cached_test 순서, lora=on-the-fly batch(window) 순서로 누적된
    # y_true/y_score 와 동일 정렬 → _eval_utils.bootstrap_ci 의 환자 단위
    # resampling·dump_fold_predictions 에 그대로 사용 가능.

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

    # fold suffix 를 ROC 파일명에도 적용: 동일 out-dir+mode 로 5-fold 를 동시
    # 실행(run_sharded)할 때 PNG 가 서로 덮어쓰지(torn-file) 않도록.
    # (.json/.npz 는 이미 fold suffix 가 붙어 안전.)
    _roc_suffix = f"_fold{args.fold}" if int(args.n_folds) > 1 else ""
    roc_path = out_dir / f"task1_roc_{args.mode}{_roc_suffix}.png"
    plot_roc_curve(
        y_true, y_score, roc_path,
        title=f"Task 1: Hypotension — {args.mode} ROC",
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
            "input_signals": args.input_signals,
            "lora_rank": args.lora_rank if args.mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.mode == "lora" else None,
            "n_cases": args.n_cases,
            "window_sec": args.window_sec,
            "stride_sec": args.stride_sec,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
            "bootstrap_iters": args.bootstrap_iters,
            "val_split_seed": args.val_split_seed,
        },
    }
    # n_folds>1 이면 fold suffix 를 붙여 fold 마다 별도 저장 (.npz 와 동일 규칙).
    # single split(n_folds=1)은 기존 이름 유지 — 하위호환.
    fold_suffix = f"_fold{args.fold}" if int(args.n_folds) > 1 else ""
    results_path = out_dir / f"task1_results_{args.mode}{fold_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    # fold prediction(.npz) — downstream.run_eval 가 OOF 집계에 사용.
    npz_path = dump_fold_predictions(
        out_dir, task="hypotension", fold_idx=args.fold, n_folds=args.n_folds,
        y_true=y_true, y_score=y_score, patient_ids=patient_ids,
    )
    print(f"Fold predictions: {npz_path}")

    # DDP: rank0 의 프로세스 그룹 정리(non-rank0 는 학습 직후 이미 정리·종료).
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
