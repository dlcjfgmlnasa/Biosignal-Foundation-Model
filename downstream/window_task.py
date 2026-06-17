# -*- coding:utf-8 -*-
"""Window-level classification task 공통 infrastructure.

Hypotension / Shock / Hypoxemia 같이 '윈도우 → binary label' 형식의 task들이 공유:
- MultiSignalWindow 데이터 구조
- PackCollate 기반 배치 생성
- linear_probe / lora 학습 + 평가 루프
- DummyFeatureExtractor (smoke test용)
- _compute_metrics

Patient-level aggregation task (Mortality / Sepsis)는 별도 경로
(shared/aggregator.py) 사용.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import get_global_spatial_id
from data.parser.vitaldb import SIGNAL_TYPES

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_sensitivity_specificity,
)
from downstream.model_wrapper import LinearProbe


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0


@dataclass
class MultiSignalWindow:
    """다중 신호 윈도우 + 라벨."""

    signals: dict[str, np.ndarray]  # {"ecg": (win_samples,), "ppg": ...}
    label: int
    label_value: float
    case_id: str | int


def _multi_window_to_samples(
    mw: MultiSignalWindow, idx: int
) -> list[BiosignalSample]:
    """MultiSignalWindow → 신호별 BiosignalSample 리스트."""
    samples = []
    for ch, (sig_type, signal) in enumerate(mw.signals.items()):
        # sig_type 은 문자열 ("ecg"/"abp"/"ppg"/...). SIGNAL_TYPES 로 int 인덱스 변환 후
        # spatial_id 도 같은 int 인덱스를 사용해야 한다 (Patch C: 이전에 string 으로
        # get_global_spatial_id 호출 → SIGNAL_TYPES 에 없는 키 → fallback 0 위험).
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


def _needed_length(samples: list[BiosignalSample], patch_size: int) -> int:
    """이 윈도우의 모든 신호를 패치 정렬해 이어 붙였을 때 필요한 길이.

    PackCollate 는 각 variate 의 seg_len 을 ``patch_size`` 배수로 FLOOR 한 뒤
    이어 붙이므로(앞 trim 은 start_sample=win_start=0 이라 0), 윈도우 전체를
    담는 데 필요한 폭은 floor 합과 같다. 최소 1 패치는 보장한다.
    """
    total = 0
    for s in samples:
        n = int(s.values.shape[0])
        total += (n // patch_size) * patch_size
    return max(total, patch_size)


def _zero_row(
    max_length: int,
    patch_size: int,
    signal_type: int,
    rate: float,
) -> PackedBatch:
    """신호가 너무 짧아 1 패치도 못 만드는 (사실상 발생하지 않는) 윈도우용 폴백.

    라벨 정렬을 유지하기 위해 1 개의 zero variate(1 패치) 로 구성된 단일 행을
    만든다. extract_features 의 mean-pool 은 patch_mask 가 거의 비어 0 feature 를
    돌려주므로 학습에 해를 주지 않는 degenerate sample 이 된다.
    """
    values = torch.zeros(1, max_length)
    sample_id = torch.zeros(1, max_length, dtype=torch.long)
    variate_id = torch.zeros(1, max_length, dtype=torch.long)
    sample_id[0, :patch_size] = 1
    variate_id[0, :patch_size] = 1
    return PackedBatch(
        values=values,
        sample_id=sample_id,
        variate_id=variate_id,
        lengths=torch.tensor([patch_size], dtype=torch.long),
        sampling_rates=torch.tensor([rate], dtype=torch.float32),
        signal_types=torch.tensor([signal_type], dtype=torch.long),
        spatial_ids=torch.tensor([0], dtype=torch.long),
        padded_lengths=torch.tensor([patch_size], dtype=torch.long),
        start_samples=torch.tensor([0], dtype=torch.long),
    )


def _collate_one_window(
    collate: PackCollate,
    samples: list[BiosignalSample],
    patch_size: int,
    max_length: int,
) -> PackedBatch:
    """한 윈도우의 신호들을 정확히 1 행(B=1)으로 collate 한다.

    한 윈도우의 모든 신호는 같은 ``session_id`` → 같은 그룹 → 하나의 PackUnit →
    하나의 FFD bin 으로 묶이므로 n_rows==1 이 보장된다(any_variate/ci 동일).
    multi-tier truncate 가 모든 variate 를 버리는 극단(모든 신호 < 10s)에서만
    0 행이 나오므로, 그 경우 zero-row 폴백으로 1 행을 보장한다.
    """
    pb = collate(samples)
    if pb.values.shape[0] == 1:
        return pb
    if pb.values.shape[0] == 0:
        st = samples[0].signal_type if samples else 0
        rate = samples[0].sampling_rate if samples else DEFAULT_SR
        return _zero_row(collate.max_length, patch_size, st, rate)
    # 단일 윈도우는 구조상 >1 행이 될 수 없으나, 방어적으로 첫 행만 사용.
    return PackedBatch(
        values=pb.values[:1],
        sample_id=pb.sample_id[:1],
        variate_id=pb.variate_id[:1],
        lengths=pb.lengths,
        sampling_rates=pb.sampling_rates,
        signal_types=pb.signal_types,
        spatial_ids=pb.spatial_ids,
        padded_lengths=pb.padded_lengths,
        start_samples=pb.start_samples,
    )


def _stack_rows(rows: list[PackedBatch]) -> PackedBatch:
    """행마다 B=1 인 PackedBatch 들을 (n_windows, max_length) 로 쌓는다.

    각 행은 같은 ``max_length`` 폭을 갖는다(collate 가 항상 그 폭으로 할당).
    per-variate 평탄 배열은 행 순서대로 concat → PackCollate 가 멀티-행을 직접
    만들 때와 **동일한 row-major 구조**가 되어 모델 forward 가 그대로 처리한다.
    행 i == window i == label i 로 정렬이 보장된다.
    """
    has_padded = rows[0].padded_lengths is not None
    has_starts = rows[0].start_samples is not None
    return PackedBatch(
        values=torch.cat([r.values for r in rows], dim=0),
        sample_id=torch.cat([r.sample_id for r in rows], dim=0),
        variate_id=torch.cat([r.variate_id for r in rows], dim=0),
        lengths=torch.cat([r.lengths for r in rows], dim=0),
        sampling_rates=torch.cat([r.sampling_rates for r in rows], dim=0),
        signal_types=torch.cat([r.signal_types for r in rows], dim=0),
        spatial_ids=torch.cat([r.spatial_ids for r in rows], dim=0),
        padded_lengths=(
            torch.cat([r.padded_lengths for r in rows], dim=0)
            if has_padded else None
        ),
        start_samples=(
            torch.cat([r.start_samples for r in rows], dim=0)
            if has_starts else None
        ),
    )


def make_window_batches(
    windows: list,
    batch_size: int,
    patch_size: int,
    to_samples: Callable[[object, int], list[BiosignalSample]],
    get_label: Callable[[object], float],
    label_dtype: torch.dtype = torch.float32,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    """윈도우 분류용 (batch, labels) 리스트 — **윈도우당 1 행** 보장.

    버그 수정(2026-06-17): 기존 구현은 batch_size 개 윈도우의 모든 신호를 한 번에
    PackCollate(FFD bin-packing) 로 묶어, 여러 윈도우가 1 개 pack 행으로 합쳐졌다.
    그러면 ``extract_features`` 가 행 단위 mean-pool 을 해 B(행 수) ≠ n_windows 가
    되어 logits 와 labels 크기가 어긋났다(예: input [1,1] vs target [8,1]).

    수정: 각 윈도우를 **개별 collate** 해 정확히 1 행을 얻고, 그 행들을 batch 차원
    으로 쌓는다. 결과 PackedBatch 는 FFD 가 윈도우당 1 bin 을 배치했을 때와 구조가
    동일하므로 사전학습 모델 forward 의미는 불변이고, 행 i == window i == label i 로
    per-window 풀링·정렬이 자동 보장된다(linear_probe / lora 양 경로 공통).

    Parameters
    ----------
    windows:
        윈도우 객체 리스트(MultiSignalWindow 또는 dict 등 task 별 표현).
    to_samples:
        ``(window, idx) -> list[BiosignalSample]`` 변환 콜백(task 별 신호 매핑 보존).
    get_label:
        ``window -> float`` 라벨 추출 콜백.
    """
    multi = any(len(to_samples(w, i)) > 1 for i, w in enumerate(windows)) \
        if windows else False
    collate_mode = "any_variate" if multi else "ci"

    batches: list[tuple[PackedBatch, torch.Tensor]] = []
    for i in range(0, len(windows), batch_size):
        chunk = windows[i : i + batch_size]
        chunk_samples = [to_samples(w, i + j) for j, w in enumerate(chunk)]

        # chunk 내 모든 윈도우를 담을 수 있는 공통 폭(가장 큰 윈도우 기준).
        # 같은 max_length 로 윈도우별 collate → 모든 행 폭 동일 → 그대로 stack.
        max_length = max(
            (_needed_length(s, patch_size) for s in chunk_samples),
            default=patch_size,
        )
        collate = PackCollate(
            max_length=max_length,
            collate_mode=collate_mode,
            patch_size=patch_size,
        )

        rows = [
            _collate_one_window(collate, s, patch_size, max_length)
            for s in chunk_samples
        ]
        batch = _stack_rows(rows)
        labels = torch.tensor([get_label(w) for w in chunk], dtype=label_dtype)
        batches.append((batch, labels))
    return batches


def make_batches(
    windows: list[MultiSignalWindow],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    """(batch, labels) 리스트 생성 — 윈도우당 1 행.

    ``max_length`` 는 하위 호환을 위해 시그니처에 남기지만 실제 폭은 윈도우별로
    자동 계산한다(아래 ``make_window_batches`` 참조).
    """
    return make_window_batches(
        windows,
        batch_size,
        patch_size,
        to_samples=_multi_window_to_samples,
        get_label=lambda w: w.label,
    )


class DummyFeatureExtractor:
    """Smoke test용 — 실제 checkpoint 없이 파이프라인 검증."""

    def __init__(self, d_model: int = 128):
        self.d_model = d_model
        self.device = torch.device("cpu")

    def extract_features(
        self, batch: PackedBatch, pool: str = "mean"
    ) -> torch.Tensor:
        b = batch.values.shape[0]
        return torch.randn(b, self.d_model)


def mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    mask_f = patch_mask.unsqueeze(-1).float()
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
    return compute_binary_metrics(
        np.concatenate(all_labels), np.concatenate(all_scores)
    )


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

    lora_params = model.lora_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr},
            {"params": probe.parameters(), "lr": lr},
        ],
        weight_decay=0.01,
    )

    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for batch, labels in train_batches:
            batch = model.batch_to_device(batch)
            out = model.model(batch, task="masked")
            features = mean_pool(out["encoded"], out["patch_mask"])

            logits = probe(features)
            loss = criterion(logits, labels.to(device).unsqueeze(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                lora_params + list(probe.parameters()), gradient_clip
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
        features = mean_pool(out["encoded"], out["patch_mask"])

        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)

    return compute_binary_metrics(
        np.concatenate(all_labels), np.concatenate(all_scores)
    )


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """AUROC/AUPRC + optimal Youden J threshold."""
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


def make_dummy_windows(
    n: int,
    input_signals: list[str],
    win_samples: int = 6000,
    seed: int = 42,
) -> list[MultiSignalWindow]:
    """Dummy smoke test용 — 라벨은 random binary."""
    rng = np.random.default_rng(seed)
    windows = []
    for i in range(n):
        signals = {s: rng.standard_normal(win_samples).astype(np.float32) for s in input_signals}
        windows.append(
            MultiSignalWindow(
                signals=signals,
                label=int(rng.integers(0, 2)),
                label_value=float(rng.uniform(50, 100)),
                case_id=f"dummy_{i}",
            )
        )
    return windows
