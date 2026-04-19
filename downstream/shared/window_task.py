# -*- coding:utf-8 -*-
"""Window-level classification task 공통 infrastructure.

Hypotension / Shock / Hypoxemia 같이 '윈도우 → binary label' 형식의 task들이 공유:
- MultiSignalWindow 데이터 구조
- PackCollate 기반 배치 생성
- linear_probe / lora 학습 + 평가 루프
- DummyFeatureExtractor (smoke test용)
- _compute_metrics

Patient-level aggregation task (Mortality / Sepsis / Extubation)는 별도 경로
(shared/aggregator.py) 사용.
"""
from __future__ import annotations

from dataclasses import dataclass

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
from downstream.shared.model_wrapper import LinearProbe


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


def make_batches(
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
