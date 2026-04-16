# -*- coding:utf-8 -*-
"""ICU Mortality Prediction (hospital_expire_flag).

환자 단위 예측: 여러 10분 윈도우를 Foundation Model로 인코딩한 뒤,
Transformer Aggregator로 시간 순서를 반영하여 환자 수준 표현을 생성한다.

구조:
    ICU Stay (수시간)
    → 10분 윈도우 × K개 슬라이딩
    → Foundation Model Encoder (frozen) → h_1, h_2, ..., h_K  (d_model)
    → [CLS] + h_1..h_K → Transformer Aggregator (학습 가능)
    → CLS output → LinearProbe → mortality 예측

2가지 모드:
  - linear_probe: Frozen encoder + Transformer Aggregator + LinearProbe
  - lora:         Frozen encoder + LoRA + Transformer Aggregator + LinearProbe

사용법:
    python -m downstream.classification.mortality.run \
        --checkpoint best.pt \
        --data-path datasets/processed/mortality/mortality_w600s.pt \
        --mode linear_probe --epochs 30 --max-windows 24
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import get_global_spatial_id

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0

SIGNAL_TYPE_INT: dict[str, int] = {
    "ecg": 0, "abp": 1, "ppg": 2, "cvp": 3, "pap": 6, "icp": 7,
}


# ── Transformer Aggregator ───────────────────────────────────


class TransformerAggregator(nn.Module):
    """시간 순서를 반영하는 Transformer 기반 환자 표현 생성기.

    [CLS] 토큰 + K개 윈도우 representation → self-attention → CLS output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_windows: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_windows + 1, d_model) * 0.02
        )  # +1 for CLS

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        chunk_reprs: torch.Tensor,  # (B, K, d_model)
        mask: torch.Tensor | None = None,  # (B, K) bool, True=valid
    ) -> torch.Tensor:  # (B, d_model)
        B, K, D = chunk_reprs.shape

        # [CLS] + chunk representations
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, chunk_reprs], dim=1)  # (B, K+1, D)

        # Positional embedding
        x = x + self.pos_embed[:, :K + 1, :]

        # Attention mask: CLS는 항상 valid
        if mask is not None:
            cls_mask = torch.ones(B, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, K+1)
            # TransformerEncoder의 src_key_padding_mask는 True=무시
            padding_mask = ~full_mask
        else:
            padding_mask = None

        out = self.encoder(x, src_key_padding_mask=padding_mask)
        return out[:, 0, :]  # CLS token → (B, d_model)


# ── 윈도우 인코딩 헬퍼 ──────────────────────────────────────


def _make_samples_for_window(
    signals: dict[str, torch.Tensor],  # {sig_type: (win_samples,)}
    idx: int,
) -> list[BiosignalSample]:
    samples = []
    for ch, (sig_type, signal) in enumerate(signals.items()):
        stype_int = SIGNAL_TYPE_INT.get(sig_type, 0)
        spatial_id = get_global_spatial_id(stype_int, 0)
        samples.append(
            BiosignalSample(
                values=signal,
                length=len(signal),
                channel_idx=ch,
                recording_idx=idx,
                sampling_rate=DEFAULT_SR,
                n_channels=len(signals),
                win_start=0,
                signal_type=stype_int,
                session_id=f"mort_{idx}",
                spatial_id=spatial_id,
            )
        )
    return samples


def _mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


@torch.no_grad()
def encode_patient_windows(
    model,
    patient: dict,
    patch_size: int,
    max_windows: int,
    use_lora: bool = False,
) -> torch.Tensor:
    """한 환자의 K개 윈도우를 인코딩하여 (K, d_model) 반환."""
    sig_types = list(patient["signals"].keys())
    K = patient["n_windows"]

    # max_windows 제한 (균등 샘플링)
    if K > max_windows:
        indices = np.linspace(0, K - 1, max_windows, dtype=int)
    else:
        indices = np.arange(K)

    multi = len(sig_types) > 1
    collate_mode = "any_variate" if multi else "ci"
    win_samples = patient["signals"][sig_types[0]].shape[1]

    collate = PackCollate(
        max_length=win_samples, collate_mode=collate_mode, patch_size=patch_size
    )

    chunk_reprs = []
    for idx in indices:
        # 이 윈도우의 신호들
        win_signals = {
            st: patient["signals"][st][idx] for st in sig_types
        }
        samples = _make_samples_for_window(win_signals, idx)
        batch = collate(samples)

        if use_lora:
            batch = model.batch_to_device(batch)
            out = model.model(batch, task="masked")
            feat = _mean_pool(out["encoded"], out["patch_mask"])
        else:
            feat = model.extract_features(batch, pool="mean")

        chunk_reprs.append(feat)  # (1, d_model)

    return torch.cat(chunk_reprs, dim=0)  # (K', d_model)


# ── 환자 배치 생성 ───────────────────────────────────────────


def _collate_patients(
    patient_reprs: list[torch.Tensor],  # [(K_i, d_model), ...]
    labels: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """가변 K를 패딩하여 배치를 만든다.

    Returns: (padded_reprs (B, K_max, d), mask (B, K_max), labels (B,))
    """
    K_max = max(r.shape[0] for r in patient_reprs)
    d_model = patient_reprs[0].shape[1]
    B = len(patient_reprs)

    padded = torch.zeros(B, K_max, d_model, device=device)
    mask = torch.zeros(B, K_max, dtype=torch.bool, device=device)

    for i, r in enumerate(patient_reprs):
        K_i = r.shape[0]
        padded[i, :K_i] = r.to(device)
        mask[i, :K_i] = True

    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)
    return padded, mask, labels_t


# ── 학습 ─────────────────────────────────────────────────────


def train_model(
    model,
    aggregator: TransformerAggregator,
    probe: LinearProbe,
    train_patients: list[dict],
    epochs: int,
    lr: float,
    device: torch.device,
    patch_size: int,
    max_windows: int,
    batch_size: int = 8,
    use_lora: bool = False,
    gradient_clip: float = 1.0,
) -> list[float]:
    aggregator = aggregator.to(device)
    probe = probe.to(device)
    aggregator.train()
    probe.train()

    params = list(aggregator.parameters()) + list(probe.parameters())
    if use_lora:
        model.model.train()
        params += model.lora_parameters()

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        # 환자 셔플
        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(train_patients))

        epoch_loss, n_batches = 0.0, 0

        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start: batch_start + batch_size]

            # 1. 각 환자의 윈도우를 인코딩
            patient_reprs = []
            batch_labels = []
            for idx in batch_indices:
                p = train_patients[idx]

                if use_lora:
                    # LoRA 모드: gradient 필요 → no_grad 제거
                    reprs = _encode_with_grad(
                        model, p, patch_size, max_windows
                    )
                else:
                    reprs = encode_patient_windows(
                        model, p, patch_size, max_windows
                    )

                patient_reprs.append(reprs)
                batch_labels.append(p["mortality"])

            # 2. 패딩 + Aggregator
            padded, mask, labels = _collate_patients(
                patient_reprs, batch_labels, device
            )
            patient_repr = aggregator(padded, mask)  # (B, d_model)

            # 3. Probe + Loss
            logits = probe(patient_repr)
            loss = criterion(logits.squeeze(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")

    return losses


def _encode_with_grad(model, patient, patch_size, max_windows):
    """LoRA 모드: gradient를 유지하면서 인코딩."""
    sig_types = list(patient["signals"].keys())
    K = patient["n_windows"]

    if K > max_windows:
        indices = np.linspace(0, K - 1, max_windows, dtype=int)
    else:
        indices = np.arange(K)

    multi = len(sig_types) > 1
    collate_mode = "any_variate" if multi else "ci"
    win_samples = patient["signals"][sig_types[0]].shape[1]
    collate = PackCollate(
        max_length=win_samples, collate_mode=collate_mode, patch_size=patch_size
    )

    chunk_reprs = []
    for idx in indices:
        win_signals = {st: patient["signals"][st][idx] for st in sig_types}
        samples = _make_samples_for_window(win_signals, idx)
        batch = collate(samples)
        batch = model.batch_to_device(batch)
        out = model.model(batch, task="masked")
        feat = _mean_pool(out["encoded"], out["patch_mask"])
        chunk_reprs.append(feat)

    return torch.cat(chunk_reprs, dim=0)


# ── 평가 ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model,
    aggregator: TransformerAggregator,
    probe: LinearProbe,
    test_patients: list[dict],
    device: torch.device,
    patch_size: int,
    max_windows: int,
) -> dict:
    aggregator.to(device).eval()
    probe.to(device).eval()
    if hasattr(model, "model"):
        model.model.eval()

    all_labels, all_scores = [], []

    for p in test_patients:
        reprs = encode_patient_windows(model, p, patch_size, max_windows)
        padded = reprs.unsqueeze(0).to(device)  # (1, K, d_model)
        mask = torch.ones(1, reprs.shape[0], dtype=torch.bool, device=device)

        patient_repr = aggregator(padded, mask)
        logit = probe(patient_repr)
        prob = torch.sigmoid(logit).squeeze().cpu().item()

        all_labels.append(p["mortality"])
        all_scores.append(prob)

    return _compute_metrics(np.array(all_labels), np.array(all_scores))


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


def _load_data(data_path: str) -> tuple[list[dict], list[dict], dict]:
    """환자 단위로 그룹핑된 .pt 파일을 로드한다."""
    print(f"\nLoading data: {data_path}")
    data = torch.load(data_path, weights_only=False)
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    print(f"  Signals: {meta.get('input_signals', '?')}")
    print(f"  Window: {meta.get('window_sec', '?')}s")
    print(f"  Aggregation: {meta.get('aggregation', 'window_level')}")

    return data["train"], data["test"], meta


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICU Mortality Prediction (Patient-Level Transformer Aggregation)"
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
    parser.add_argument("--batch-size", type=int, default=8,
                        help="환자 수 per batch (윈도우 수 아님)")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--max-windows", type=int, default=24,
                        help="환자당 최대 윈도우 수 (초과 시 균등 샘플링)")
    parser.add_argument("--agg-layers", type=int, default=2,
                        help="Transformer Aggregator 레이어 수")
    parser.add_argument("--agg-heads", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── 모델 로드 ──
    from downstream.model_wrapper import DownstreamModelWrapper

    print(f"Loading checkpoint: {args.checkpoint}")
    model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
    d_model = model.d_model
    patch_size = model.patch_size
    print(f"  d_model={d_model}, patch_size={patch_size}")

    use_lora = args.mode == "lora"
    if use_lora:
        model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    # ── 데이터 로드 ──
    train_patients, test_patients, meta = _load_data(args.data_path)

    n_dead_train = sum(1 for p in train_patients if p["mortality"] == 1)
    n_dead_test = sum(1 for p in test_patients if p["mortality"] == 1)
    avg_win_train = np.mean([p["n_windows"] for p in train_patients])
    avg_win_test = np.mean([p["n_windows"] for p in test_patients])
    print(f"  Train: {len(train_patients)} patients "
          f"({n_dead_train} dead, avg {avg_win_train:.1f} windows)")
    print(f"  Test:  {len(test_patients)} patients "
          f"({n_dead_test} dead, avg {avg_win_test:.1f} windows)")
    print(f"  Max windows per patient: {args.max_windows}")

    # ── Aggregator + Probe ──
    aggregator = TransformerAggregator(
        d_model=d_model,
        n_heads=args.agg_heads,
        n_layers=args.agg_layers,
        max_windows=args.max_windows,
    )
    probe = LinearProbe(d_model, n_classes=1)

    n_agg = sum(p.numel() for p in aggregator.parameters())
    n_probe = sum(p.numel() for p in probe.parameters())
    print(f"\n  Aggregator: {n_agg:,} params ({args.agg_layers} layers, "
          f"{args.agg_heads} heads)")
    print(f"  Probe: {n_probe:,} params")
    if use_lora:
        n_lora = sum(p.numel() for p in model.lora_parameters())
        print(f"  LoRA: {n_lora:,} params (rank={args.lora_rank})")

    # ── 학습 ──
    print(f"\nTraining ({args.mode})...")
    train_losses = train_model(
        model, aggregator, probe, train_patients,
        epochs=args.epochs, lr=args.lr, device=device,
        patch_size=patch_size, max_windows=args.max_windows,
        batch_size=args.batch_size, use_lora=use_lora,
    )

    # ── 평가 ──
    print("\nEvaluating...")
    metrics = evaluate_model(
        model, aggregator, probe, test_patients,
        device=device, patch_size=patch_size,
        max_windows=args.max_windows,
    )

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 60}")
    print(f"  ICU Mortality — {args.mode} (Transformer Aggregator)")
    print(f"{'=' * 60}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Prevalence:  {metrics['prevalence']:.3f} "
          f"({metrics['n_positive']}/{metrics['n_total']})")
    print(f"{'=' * 60}")

    roc_path = out_dir / f"mortality_roc_{args.mode}.png"
    plot_roc_curve(y_true, y_score, roc_path,
                   title=f"ICU Mortality — {args.mode} ROC")
    print(f"\nROC curve: {roc_path}")

    results = {
        **metrics, "train_losses": train_losses,
        "config": {
            "task": "icu_mortality_prediction",
            "mode": args.mode,
            "aggregation": "transformer",
            "agg_layers": args.agg_layers,
            "agg_heads": args.agg_heads,
            "max_windows": args.max_windows,
            "data_path": args.data_path,
            "epochs": args.epochs, "lr": args.lr,
        },
    }
    results_path = out_dir / f"mortality_results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
