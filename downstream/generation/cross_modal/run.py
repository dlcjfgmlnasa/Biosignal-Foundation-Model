# -*- coding:utf-8 -*-
"""Task 8: Any-to-Any Cross-modal Prediction.

2가지 모드:
  - zero_shot: 사전학습된 cross_pred_per_type head로 바로 평가 (학습 없음)
  - lora:      LoRA adapter + WaveformRegressionHead 학습 후 평가

Usage
-----
# Dummy test (no checkpoint):
python -m downstream.generation.cross_modal.run --dummy

# Zero-shot evaluation (default scenarios):
python -m downstream.generation.cross_modal.run \
    --checkpoint path/to/best.pt --mode zero_shot

# Zero-shot with specific fixed scenario:
python -m downstream.generation.cross_modal.run \
    --checkpoint path/to/best.pt --mode zero_shot \
    --scenario "ECG+PPG->ABP"

# Zero-shot generalized (random subset → full recovery):
python -m downstream.generation.cross_modal.run \
    --checkpoint path/to/best.pt --mode zero_shot \
    --scenario random_subset \
    --mask-ratio 0.25 0.5 0.75 \
    --n-iterations 3

# LoRA regression:
python -m downstream.generation.cross_modal.run \
    --checkpoint path/to/best.pt --mode lora \
    --data-path outputs/downstream/any_to_any/task8_xxx.pt \
    --scenario "ECG->ABP" --epochs 30 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import SIGNAL_KEY_TO_TYPE
from downstream._save_utils import load_prepared_chunked

# signal_type_key -> signal_type_id : data.spatial_map 의 SSOT(v2) 사용.
# (10 signal_type, spatial_id 폐지, resp_impedance=8 / resp_flow=9.)
SIGNAL_TYPE_IDS: dict[str, int] = SIGNAL_KEY_TO_TYPE

# Mechanism groups for analysis
MECHANISM_GROUPS: dict[str, str] = {
    "ecg": "cardiovascular",
    "abp": "cardiovascular",
    "ppg": "cardiovascular",
    "cvp": "cardiovascular",
    "co2": "respiratory",
    "awp": "respiratory",
    "resp_impedance": "respiratory",
    "resp_flow": "respiratory",
    "pap": "cardiovascular",
    "icp": "neurological",
}


# ── Scenarios ────────────────────────────────────────────────


@dataclass
class Scenario:
    """Input -> target prediction scenario."""

    name: str
    inputs: list[str]  # signal type keys to provide
    target: str  # signal type key to predict
    group_type: str  # "intra" (same mechanism group) or "inter"
    n_inputs: int  # number of input channels (for analysis)


def parse_scenario(scenario_str: str) -> Scenario:
    """Parse scenario string like 'ECG->ABP' or 'ECG+PPG->ABP'.

    Returns a Scenario with automatically determined group_type.
    """
    if "->" not in scenario_str:
        raise ValueError(f"Invalid scenario format: '{scenario_str}'. Use 'ECG->ABP'.")
    left, right = scenario_str.split("->", 1)
    inputs = [s.strip().lower() for s in left.split("+")]
    target = right.strip().lower()

    for s in inputs + [target]:
        if s not in SIGNAL_TYPE_IDS:
            raise ValueError(f"Unknown signal type: '{s}'. Valid: {list(SIGNAL_TYPE_IDS)}")

    # Determine group type
    input_groups = {MECHANISM_GROUPS[s] for s in inputs}
    target_group = MECHANISM_GROUPS[target]
    group_type = "intra" if all(g == target_group for g in input_groups) else "inter"

    return Scenario(
        name=scenario_str.upper(),
        inputs=inputs,
        target=target,
        group_type=group_type,
        n_inputs=len(inputs),
    )


def get_default_scenarios() -> list[Scenario]:
    """Task #12 Cross-Modal Reconstruction default scenarios.

    Paper-confirmed pair definition (docs/paper_task_modality.md).
    Reporting set (2026-06-17 갱신): 강결합 (γ, CROSS_PRED_ALLOWED_PAIRS) 기반,
    "약한(비침습) 신호 → 강한(침습) 신호" 복원 방향:
      - ECG → ABP (비침습 → 침습 동맥압)          (0,1)
      - PPG → ABP (비침습 말초 맥파 → 침습 동맥압) (1,2)
      - ECG+PPG → ABP (비침습 2채널 → 침습 ABP)   (0,1)+(1,2)  ← multi-input
      - AWP → RESP_Flow (ventilator P–Q 인과)     (5,9)

    제외:
      - ECG → PPG: cardiac→peripheral 이나, 임상 활용도 높은 PPG → ABP
        (비침습 BP 추정)로 대체.
      - ABP → PPG: 강 → 약 방향이라 임상 의의 낮음 → ECG+PPG → ABP 로 대체.
      - ABP → ICP: ICP 가 VitalDB 에 없음(평가 소스 부재) → 제외.
      - CO2 → RESP_Impedance: 약결합 + downstream RESP 데이터 소스 부재.
      - ABP → PAP / CVP → PAP: PAP 는 사전학습에서 제외(slot 6 데이터 없음) →
        타깃이 pretrain 에 미등장하여 복원 대상 부적합.
    """
    scenarios = [
        # 강결합 cardiovascular (γ) — 약 → 강 방향
        Scenario("ECG->ABP", ["ecg"], "abp", "intra", 1),
        Scenario("PPG->ABP", ["ppg"], "abp", "intra", 1),
        Scenario("ECG+PPG->ABP", ["ecg", "ppg"], "abp", "intra", 2),  # multi-input
        # 강결합 respiratory (γ, spatial_map (5,9))
        Scenario("AWP->RESP_FLOW", ["awp"], "resp_flow", "intra", 1),
    ]
    return scenarios


# ── Result ───────────────────────────────────────────────────


@dataclass
class AnyToAnyResult:
    """Single scenario evaluation result."""

    scenario_name: str
    inputs: list[str]
    target: str
    group_type: str
    n_inputs: int
    # reconstructed head metrics
    recon_mse: float
    recon_mae: float
    recon_pearson_r: float
    # cross_pred head metrics
    cross_mse: float
    cross_mae: float
    cross_pearson_r: float
    n_patches: int


# ── Batch construction (reuse Task 3 pattern) ───────────────


def build_multivariate_batch(
    signals: dict[str, np.ndarray],
    patch_size: int = 100,
    session_id: str = "eval_0",
    sr: float = 100.0,
) -> PackedBatch:
    """Signal type dict -> multi-variate PackedBatch."""
    samples: list[BiosignalSample] = []

    for ch_idx, (stype_key, signal) in enumerate(signals.items()):
        signal_type = SIGNAL_TYPE_IDS[stype_key]
        n_usable = (len(signal) // patch_size) * patch_size
        if n_usable == 0:
            continue
        signal = signal[:n_usable]

        samples.append(
            BiosignalSample(
                values=torch.tensor(signal, dtype=torch.float32),
                length=n_usable,
                channel_idx=ch_idx,
                recording_idx=0,
                sampling_rate=sr,
                n_channels=len(signals),
                win_start=0,
                signal_type=signal_type,
                session_id=session_id,
                spatial_id=0,
            )
        )

    if not samples:
        raise ValueError("No valid signals for batch construction")

    max_length = sum(s.length for s in samples)
    collate = PackCollate(
        max_length=max_length,
        collate_mode="any_variate",
        patch_size=patch_size,
    )
    return collate(samples)


# ── Evaluation (zero-shot) ──────────────────────────────────


def evaluate_scenario(
    model: torch.nn.Module,
    batch: PackedBatch,
    target_signal_type: int,
    patch_size: int,
    device: torch.device,
) -> dict[str, float] | None:
    """Evaluate a single scenario: target variate reconstruction + cross-modal.

    Returns dict with recon_mse/mae/r, cross_mse/mae/r, n_patches, or None.
    """
    model.eval()
    batch.values = batch.values.to(device)
    batch.sample_id = batch.sample_id.to(device)
    batch.variate_id = batch.variate_id.to(device)

    with torch.no_grad():
        out = model(batch, task="masked")

    reconstructed = out["reconstructed"]  # (B, N, patch_size)
    cross_pred = out.get("cross_pred")  # (B, N, patch_size) or None
    patch_signal_types = out["patch_signal_types"]  # (B, N)
    patch_mask = out["patch_mask"]  # (B, N) bool

    if patch_signal_types is None:
        return None

    # Original patches (normalized scale)
    normalized = ((batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]).squeeze(-1)
    B, L = normalized.shape
    N = L // patch_size
    original_patches = normalized.reshape(B, N, patch_size)

    # Target channel patches
    target_mask = (patch_signal_types == target_signal_type) & patch_mask
    if not target_mask.any():
        return None

    orig = original_patches[target_mask].cpu()  # (M, P)

    # Reconstructed head
    recon_pred = reconstructed[target_mask].cpu()
    recon_mse = F.mse_loss(recon_pred, orig).item()
    recon_mae = F.l1_loss(recon_pred, orig).item()
    recon_r = _pearson_r(recon_pred.reshape(-1), orig.reshape(-1))

    # Cross-modal head
    if cross_pred is not None:
        cross_p = cross_pred[target_mask].cpu()
        cross_mse = F.mse_loss(cross_p, orig).item()
        cross_mae = F.l1_loss(cross_p, orig).item()
        cross_r = _pearson_r(cross_p.reshape(-1), orig.reshape(-1))
    else:
        cross_mse = cross_mae = cross_r = 0.0

    return {
        "recon_mse": recon_mse,
        "recon_mae": recon_mae,
        "recon_pearson_r": recon_r,
        "cross_mse": cross_mse,
        "cross_mae": cross_mae,
        "cross_pearson_r": cross_r,
        "n_patches": int(target_mask.sum().item()),
    }


def _pearson_r(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation coefficient between 1D tensors."""
    x = x - x.mean()
    y = y - y.mean()
    num = (x * y).sum()
    den = (x.norm() * y.norm()).clamp(min=1e-8)
    return (num / den).item()


# ── Synthetic signal generation ──────────────────────────────

_SYNTH_FREQS: dict[str, float] = {
    "ecg": 1.2,
    "abp": 1.1,
    "ppg": 1.0,
    "cvp": 0.9,
    "co2": 0.25,
    "awp": 0.3,
    "pap": 1.0,
    "icp": 0.15,
    "resp_impedance": 0.25,  # 호흡 effort (~15 bpm)
    "resp_flow": 0.25,       # ventilator flow (~15 bpm)
}


def _gen_synthetic(stype: str, n_samples: int = 3000, sr: float = 100.0) -> np.ndarray:
    """Generate synthetic signal for a given type."""
    t = np.arange(n_samples) / sr
    freq = _SYNTH_FREQS.get(stype, 1.0)
    sig = np.sin(2 * np.pi * freq * t)
    sig += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    sig += 0.1 * np.random.randn(n_samples)
    return sig


# ── WaveformRegressionHead ──────────────────────────────────


class WaveformRegressionHead(nn.Module):
    """Predicts target waveform patches from encoder representations.

    Parameters
    ----------
    d_model : int
        Encoder hidden dimension.
    patch_size : int
        Number of samples per patch.
    """

    def __init__(self, d_model: int, patch_size: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, patch_size),
        )

    def forward(
        self,
        encoded: torch.Tensor,  # (B, N, d_model)
    ) -> torch.Tensor:  # (B, N, patch_size)
        return self.head(encoded)


# ── LoRA data utilities ─────────────────────────────────────


def _data_path_available(data_path: str, fold: int | None = None) -> bool:
    """data_path 가 단일 .pt 파일이거나 split-chunk 묶음으로 존재하는지 확인.

    fold 지정 시 ``<stem>_fold{F}_*_chunk*.pt`` 존재 여부로 판단.
    """
    p = Path(data_path)
    if fold is None:
        if p.is_file():
            return True
        if p.suffix != ".pt" and p.with_suffix(".pt").is_file():
            return True
    stem = p.name[:-3] if p.name.endswith(".pt") else p.name
    parent = p.parent if str(p.parent) else Path(".")
    prefix = stem if fold is None else f"{stem}_fold{int(fold)}"
    return any(parent.glob(f"{prefix}_*_chunk*.pt"))


def _load_prepared_data(data_path: str, fold: int | None = None) -> dict:
    """Load a prepared .pt file for LoRA training.

    Expected structure::

        {
            "train": {"signals": {"ecg": (N, T), "abp": (N, T), ...}, "case_ids": [...]},
            ["val": {...},]
            "test":  {"signals": {...}, "case_ids": [...]},
            "metadata": {"signal_types": [...], "window_sec": 30, ...}
        }

    ``fold`` 지정 시 ``<stem>_fold{F}_<split>_chunk*.pt`` 만 로드 (K-fold CV).
    None 이면 비-fold glob + 단일파일 back-compat.
    """
    # 단일 .pt (back-compat) 또는 split-chunked .pt 묶음을 동일 shape 으로 로드.
    data = load_prepared_chunked(data_path, fold=fold)
    meta = data.get("metadata", {})
    print(f"  Loaded: {data_path} (fold={fold})")
    print(f"  Source: {meta.get('source', '?')}, Signals: {meta.get('signal_types', '?')}")
    print(f"  Train: {meta.get('n_train', '?')}, Test: {meta.get('n_test', '?')}")
    return data


def _patchify(signal: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Reshape (T,) or (N, T) signal into patches.

    Parameters
    ----------
    signal : torch.Tensor
        1D ``(T,)`` or 2D ``(N, T)`` waveform.
    patch_size : int

    Returns
    -------
    ``(N_patches,  patch_size)`` or ``(N, N_patches, patch_size)``
    """
    if signal.dim() == 1:
        T = signal.shape[0]
        n_patches = T // patch_size
        return signal[: n_patches * patch_size].reshape(n_patches, patch_size)
    else:
        N, T = signal.shape
        n_patches = T // patch_size
        return signal[:, : n_patches * patch_size].reshape(N, n_patches, patch_size)


def _build_lora_batches(
    split_data: dict,
    scenario: Scenario,
    patch_size: int,
    batch_size: int,
    sr: float = 100.0,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    """Build (input_batch, target_patches) pairs for LoRA training.

    For each sample:
      - Build PackedBatch from INPUT signals only
      - Patchify TARGET signal as ground truth

    Returns list of (PackedBatch, target_patches) tuples.
    target_patches: (B, N_patches_per_input, patch_size)
    """
    signals_dict = split_data["signals"]  # {stype: (N, T) tensor}

    # Validate that required signals exist
    for s in scenario.inputs:
        if s not in signals_dict:
            raise ValueError(f"Input signal '{s}' not found. Available: {list(signals_dict)}")
    if scenario.target not in signals_dict:
        raise ValueError(
            f"Target signal '{scenario.target}' not found. Available: {list(signals_dict)}"
        )

    n_samples = signals_dict[scenario.inputs[0]].shape[0]
    T = signals_dict[scenario.inputs[0]].shape[1]
    n_patches_per_variate = T // patch_size

    batches: list[tuple[PackedBatch, torch.Tensor]] = []

    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        B = end - start

        # Collect all BiosignalSamples for input signals
        all_samples: list[BiosignalSample] = []
        for idx_in_batch in range(B):
            global_idx = start + idx_in_batch
            for ch_idx, stype_key in enumerate(scenario.inputs):
                sig = signals_dict[stype_key][global_idx]  # (T,)
                n_usable = (len(sig) // patch_size) * patch_size
                if n_usable == 0:
                    continue
                all_samples.append(
                    BiosignalSample(
                        values=sig[:n_usable].float(),
                        length=n_usable,
                        channel_idx=ch_idx,
                        recording_idx=idx_in_batch,
                        sampling_rate=sr,
                        n_channels=len(scenario.inputs),
                        win_start=0,
                        signal_type=SIGNAL_TYPE_IDS[stype_key],
                        session_id=f"lora_{global_idx}",
                        spatial_id=0,
                    )
                )

        if not all_samples:
            continue

        # Collate input signals
        max_length = sum(s.length for s in all_samples if s.recording_idx == 0) * 1
        # Actually compute per row: max over all recording_idx groups
        from collections import defaultdict
        row_lengths: dict[int, int] = defaultdict(int)
        for s in all_samples:
            row_lengths[s.recording_idx] += s.length
        max_length = max(row_lengths.values())

        collate_mode = "any_variate" if len(scenario.inputs) > 1 else "ci"
        collate = PackCollate(
            max_length=max_length,
            collate_mode=collate_mode,
            patch_size=patch_size,
        )
        batch = collate(all_samples)

        # Target patches: (B, n_patches_per_variate, patch_size)
        target_sig = signals_dict[scenario.target][start:end]  # (B, T)
        target_patches = _patchify(target_sig, patch_size)  # (B, n_patches, patch_size)

        batches.append((batch, target_patches))

    return batches


def _pool_by_time(
    encoded: torch.Tensor,  # (B, N_total, d_model)
    time_id: torch.Tensor,  # (B, N_total)
    patch_mask: torch.Tensor,  # (B, N_total) bool
) -> torch.Tensor:  # (B, N_time, d_model)
    """Mean pool encoded features by time_id.

    Groups patches with the same time_id and averages them.
    Returns features for each unique time step, sorted by time_id.
    """
    B, N, D = encoded.shape
    device = encoded.device

    # Find the max number of unique time steps
    # time_id for valid patches; invalid patches have arbitrary time_id
    # Mask invalid patches by setting their time_id to a large value
    masked_time = time_id.clone()
    masked_time[~patch_mask] = -1  # mark invalid

    # Get unique sorted time steps per batch (they should be consistent)
    # For simplicity, use the max time_id +1 as N_time
    valid_times = masked_time[patch_mask]
    if valid_times.numel() == 0:
        return encoded[:, :1, :]  # fallback

    max_time = int(valid_times.max().item()) + 1

    pooled = torch.zeros(B, max_time, D, device=device)
    counts = torch.zeros(B, max_time, 1, device=device)

    for b in range(B):
        valid = patch_mask[b]  # (N,)
        if not valid.any():
            continue
        t_ids = time_id[b, valid]  # (M,)
        feats = encoded[b, valid]  # (M, D)
        for t in range(max_time):
            t_mask = t_ids == t
            if t_mask.any():
                pooled[b, t] = feats[t_mask].mean(dim=0)
                counts[b, t] = 1.0

    # Remove time steps with no patches (keep only those present in all batches)
    # For aligned signals this should be all time steps
    valid_time_mask = counts.squeeze(-1).sum(dim=0) > 0  # (max_time,)
    pooled = pooled[:, valid_time_mask]  # (B, N_valid, D)

    return pooled


# ── LoRA training ────────────────────────────────────────────


def train_lora_regression(
    model,  # DownstreamModelWrapper (with LoRA injected)
    head: WaveformRegressionHead,
    train_batches: list[tuple[PackedBatch, torch.Tensor]],
    epochs: int,
    lr: float,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> list[float]:
    """Train LoRA adapters + WaveformRegressionHead.

    Parameters
    ----------
    model : DownstreamModelWrapper with LoRA injected.
    head : WaveformRegressionHead.
    train_batches : list of (PackedBatch, target_patches) tuples.
    epochs : number of training epochs.
    lr : learning rate.
    device : torch device.
    gradient_clip : gradient clipping norm.

    Returns
    -------
    List of per-epoch average losses.
    """
    model.model.train()
    head = head.to(device)
    head.train()

    lora_params = model.lora_parameters()
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": lr},
            {"params": head.parameters(), "lr": lr},
        ],
        weight_decay=0.01,
    )

    losses: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch, target_patches in train_batches:
            batch = model.batch_to_device(batch)
            target_patches = target_patches.to(device)  # (B, N_target, patch_size)

            out = model.model(batch, task="masked", mask_ratio=0.0)
            encoded = out["encoded"]  # (B, N_total, d_model)
            time_id = out["time_id"]  # (B, N_total)
            patch_mask = out["patch_mask"]  # (B, N_total) bool

            # Pool by time position -> (B, N_time, d_model)
            pooled = _pool_by_time(encoded, time_id, patch_mask)

            # Predict target patches
            predicted = head(pooled)  # (B, N_time, patch_size)

            # Align lengths: take min of predicted and target
            n_out = min(predicted.shape[1], target_patches.shape[1])
            predicted = predicted[:, :n_out]
            target = target_patches[:, :n_out]

            loss = F.mse_loss(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                lora_params + list(head.parameters()),
                gradient_clip,
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.6f}")

    return losses


@torch.no_grad()
def evaluate_lora_regression(
    model,  # DownstreamModelWrapper
    head: WaveformRegressionHead,
    test_batches: list[tuple[PackedBatch, torch.Tensor]],
    device: torch.device,
    collect_raw: bool = False,
) -> dict:
    """Evaluate LoRA regression on test set.

    Returns dict with mse, mae, pearson_r, n_patches. ``collect_raw`` 시
    per-window 평탄화 waveform ``raw_y_true``/``raw_y_score`` (N_win, L) 도 포함
    (preds_fold{F}.npz dump 용, forecast/classification 포맷과 동일 키).
    """
    model.model.eval()
    head.to(device).eval()

    all_pred: list[torch.Tensor] = []
    all_target: list[torch.Tensor] = []
    # per-window raw waveform (collect_raw) — 배치별 (B, L) numpy 누적.
    raw_pred_win: list[np.ndarray] = []
    raw_tgt_win: list[np.ndarray] = []

    for batch, target_patches in test_batches:
        batch = model.batch_to_device(batch)
        target_patches = target_patches.to(device)

        out = model.model(batch, task="masked", mask_ratio=0.0)
        encoded = out["encoded"]
        time_id = out["time_id"]
        patch_mask = out["patch_mask"]

        pooled = _pool_by_time(encoded, time_id, patch_mask)
        predicted = head(pooled)

        n_out = min(predicted.shape[1], target_patches.shape[1])
        pred_o = predicted[:, :n_out].cpu()      # (B, n_out, patch_size)
        tgt_o = target_patches[:, :n_out].cpu()  # (B, n_out, patch_size)
        all_pred.append(pred_o.reshape(-1))
        all_target.append(tgt_o.reshape(-1))
        if collect_raw:
            # window 당 평탄화 waveform: (B, n_out*patch_size)
            raw_pred_win.append(pred_o.reshape(pred_o.shape[0], -1).numpy())
            raw_tgt_win.append(tgt_o.reshape(tgt_o.shape[0], -1).numpy())

    pred = torch.cat(all_pred)
    target = torch.cat(all_target)

    mse = F.mse_loss(pred, target).item()
    mae = F.l1_loss(pred, target).item()
    r = _pearson_r(pred, target)
    n_patches = pred.shape[0]

    metrics: dict = {
        "mse": mse,
        "mae": mae,
        "pearson_r": r,
        "n_patches": n_patches,
    }

    if collect_raw and raw_pred_win:
        # 배치 간 L 이 다를 수 있으니 공통 최소 길이로 truncate 후 stack.
        min_L = min(a.shape[1] for a in raw_pred_win)
        y_score = np.concatenate([a[:, :min_L] for a in raw_pred_win], axis=0)
        y_true = np.concatenate([a[:, :min_L] for a in raw_tgt_win], axis=0)
        metrics["raw_y_score"] = y_score.astype(np.float32)  # (N_win, L)
        metrics["raw_y_true"] = y_true.astype(np.float32)    # (N_win, L)

    return metrics


# ── Dummy test ───────────────────────────────────────────────


def run_dummy_test() -> list[AnyToAnyResult]:
    """Run all scenarios with a random (untrained) model."""
    from model import ModelConfig
    from model.biosignal_model import BiosignalFoundationModel

    print("=" * 70)
    print("Task 8: Any-to-Any Prediction - Dummy Test (random model)")
    print("=" * 70)

    config = ModelConfig(d_model=64, num_layers=2, patch_size=100)
    model = BiosignalFoundationModel.from_config(config)
    model.eval()
    device = torch.device("cpu")

    scenarios = get_default_scenarios()
    results: list[AnyToAnyResult] = []

    for sc in scenarios:
        all_types = sc.inputs + [sc.target]
        signals = {stype: _gen_synthetic(stype) for stype in all_types}

        batch = build_multivariate_batch(signals, patch_size=config.patch_size)
        target_type_id = SIGNAL_TYPE_IDS[sc.target]

        metrics = evaluate_scenario(
            model, batch, target_type_id, config.patch_size, device
        )
        if metrics is None:
            print(f"  SKIP: {sc.name} (target not found in batch)")
            continue

        results.append(
            AnyToAnyResult(
                scenario_name=sc.name,
                inputs=sc.inputs,
                target=sc.target,
                group_type=sc.group_type,
                n_inputs=sc.n_inputs,
                recon_mse=round(metrics["recon_mse"], 6),
                recon_mae=round(metrics["recon_mae"], 6),
                recon_pearson_r=round(metrics["recon_pearson_r"], 4),
                cross_mse=round(metrics["cross_mse"], 6),
                cross_mae=round(metrics["cross_mae"], 6),
                cross_pearson_r=round(metrics["cross_pearson_r"], 4),
                n_patches=metrics["n_patches"],
            )
        )

    _print_results_table(results)
    _print_analysis(results)
    return results


# ── Real evaluation (zero-shot) ─────────────────────────────


# ── Generalized Random-Subset Recovery ─────────────────────


def _select_random_subset(
    signal_keys: list[str],
    mask_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Mask ratio에 따라 visible / hidden subset을 무작위로 분할.

    - visible: 모델 입력으로 제공되는 signal keys
    - hidden: 가려져서 예측 target이 되는 signal keys
    - 항상 최소 1개 visible, 최소 1개 hidden (예측할 게 있어야 함)
    """
    import random

    rng = random.Random(seed)
    keys_sorted = sorted(signal_keys)
    n = len(keys_sorted)
    if n < 2:
        return set(keys_sorted), set()

    n_hidden = max(1, min(n - 1, int(round(n * mask_ratio))))
    hidden = set(rng.sample(keys_sorted, n_hidden))
    visible = set(keys_sorted) - hidden
    return visible, hidden


@torch.no_grad()
def evaluate_random_subset_window(
    model: torch.nn.Module,
    all_signals: dict[str, np.ndarray],
    mask_ratio: float,
    patch_size: int,
    device: torch.device,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Random visible subset → hidden recovery 평가 (single window).

    Parameters
    ----------
    all_signals:
        윈도우 내 사용 가능한 모든 signal 파형 (numpy).
    mask_ratio:
        Hidden signal 비율 (0-1). 0.5면 절반 hidden.

    Returns
    -------
    {signal_key: {"mae", "mse", "pearson_r"}} — hidden 각 signal별 metric.
    """
    visible, hidden = _select_random_subset(
        list(all_signals.keys()), mask_ratio, seed=seed,
    )
    if not hidden or not visible:
        return {}

    # Visible signal만으로 batch 구성 → cross_pred_per_type 또는 generate_cross_modal로 hidden 예측
    visible_signals = {k: all_signals[k] for k in visible}
    batch = build_multivariate_batch(visible_signals, patch_size=patch_size)
    batch.values = batch.values.to(device)
    batch.sample_id = batch.sample_id.to(device)
    batch.variate_id = batch.variate_id.to(device)

    results: dict[str, dict[str, float]] = {}
    for target_key in hidden:
        target_type_id = SIGNAL_TYPE_IDS[target_key]

        try:
            out = model.generate_cross_modal(
                batch, target_signal_type=target_type_id, denormalize=True,
            )
        except Exception:
            continue

        pred_waveform = out["waveform"]  # (B, N, patch_size)
        if pred_waveform.numel() == 0:
            continue
        pred_flat = pred_waveform[0].reshape(-1).cpu().numpy()

        # GT: flatten ground-truth signal to matching length
        gt_full = all_signals[target_key]
        min_len = min(len(pred_flat), len(gt_full))
        if min_len < patch_size:
            continue
        pred_flat = pred_flat[:min_len]
        gt_flat = gt_full[:min_len]

        mae = float(np.mean(np.abs(pred_flat - gt_flat)))
        mse = float(np.mean((pred_flat - gt_flat) ** 2))
        if np.std(pred_flat) > 1e-8 and np.std(gt_flat) > 1e-8:
            r = float(np.corrcoef(pred_flat, gt_flat)[0, 1])
        else:
            r = 0.0

        results[target_key] = {"mae": mae, "mse": mse, "pearson_r": r}

    return results


def run_random_subset_eval(
    checkpoint_path: str,
    model_version: str = "v1",
    n_cases: int = 20,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    data_path: str | None = None,
    mask_ratios: list[float] | None = None,
    n_iterations: int = 3,
    device_str: str = "cpu",
    fold: int | None = None,
) -> dict:
    """Random-subset → full recovery zero-shot 평가.

    Paper 1 Physiological Generation 4.2.3 확장:
    고정 시나리오 대비 Any-Variate capability를 random mask로 검증.

    Parameters
    ----------
    mask_ratios:
        평가할 mask 비율 리스트 (e.g., [0.25, 0.5, 0.75]).
    n_iterations:
        각 mask_ratio별 random seed 반복 수 (분산 감소).

    Returns
    -------
    {mask_ratio: {signal: {mae, mse, pearson_r, n_windows}}}
    """
    from downstream.model_wrapper import DownstreamModelWrapper

    if mask_ratios is None:
        mask_ratios = [0.25, 0.50, 0.75]

    print("=" * 70)
    print("Cross-Modal Generalized — Random-Subset → Full Recovery")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  mask_ratios: {mask_ratios}, n_iterations: {n_iterations}")
    print("=" * 70)

    device = torch.device(device_str)
    wrapper = DownstreamModelWrapper(checkpoint_path, model_version, device)
    model = wrapper.model
    patch_size = wrapper.patch_size

    # 데이터 소스 준비: data_path 우선, 없으면 pilot cases
    windows: list[dict[str, np.ndarray]] = []
    if data_path and _data_path_available(data_path, fold=fold):
        data = _load_prepared_data(data_path, fold=fold)
        test_data = data["test"]
        available_sigs = list(test_data["signals"].keys())
        n_samples = test_data["signals"][available_sigs[0]].shape[0]
        for i in range(n_samples):
            w = {s: test_data["signals"][s][i].numpy() for s in available_sigs}
            windows.append(w)
    else:
        from downstream.data_utils import load_pilot_cases

        cases = load_pilot_cases(
            n_cases=n_cases, signal_types=list(SIGNAL_TYPE_IDS.keys()),
        )
        win_samples = int(window_sec * 100.0)
        stride_samples = int(stride_sec * 100.0)
        for case in cases:
            available = [k for k in SIGNAL_TYPE_IDS if k in case.tracks]
            if len(available) < 2:
                continue
            min_len = min(len(case.tracks[t]) for t in available)
            for start in range(0, min_len - win_samples + 1, stride_samples):
                w = {t: case.tracks[t][start:start + win_samples] for t in available}
                windows.append(w)

    if not windows:
        print("  No valid windows. Aborting.")
        return {}

    print(f"  Total windows: {len(windows)}")

    # mask_ratio별로 평가
    all_results: dict[float, dict] = {}
    for mr in mask_ratios:
        per_signal: dict[str, list[dict[str, float]]] = {}
        for it in range(n_iterations):
            seed = 1000 + it  # 반복별 다른 seed
            for w in windows:
                if len(w) < 2:
                    continue
                r = evaluate_random_subset_window(
                    model, w, mr, patch_size, device, seed=seed,
                )
                for sig, m in r.items():
                    per_signal.setdefault(sig, []).append(m)

        # signal별 aggregation
        agg: dict[str, dict[str, float]] = {}
        for sig, metrics_list in per_signal.items():
            if not metrics_list:
                continue
            agg[sig] = {
                "mae": round(float(np.mean([m["mae"] for m in metrics_list])), 6),
                "mse": round(float(np.mean([m["mse"] for m in metrics_list])), 6),
                "pearson_r": round(
                    float(np.mean([m["pearson_r"] for m in metrics_list])), 4,
                ),
                "n_windows": len(metrics_list),
            }
        all_results[mr] = agg

        # 중간 출력
        print(f"\n  [mask_ratio={mr}]")
        print(f"    {'Signal':<6s} {'MAE':>10s} {'MSE':>10s} {'Pearson r':>10s} {'N':>8s}")
        for sig in sorted(agg.keys()):
            m = agg[sig]
            print(
                f"    {sig.upper():<6s} {m['mae']:>10.4f} {m['mse']:>10.4f} "
                f"{m['pearson_r']:>10.4f} {m['n_windows']:>8d}"
            )

    # Overall summary
    print(f"\n{'=' * 70}")
    print("  Summary — MAE as f(mask_ratio)")
    print("=" * 70)
    header = f"  {'Signal':<6s} " + " ".join(
        f"{f'mr={mr:.2f}':>10s}" for mr in mask_ratios
    )
    print(header)
    all_signals_seen = set()
    for agg in all_results.values():
        all_signals_seen.update(agg.keys())
    for sig in sorted(all_signals_seen):
        row = f"  {sig.upper():<6s} "
        for mr in mask_ratios:
            m = all_results[mr].get(sig)
            val = f"{m['mae']:.4f}" if m else "-"
            row += f"{val:>10s} "
        print(row)

    return all_results


def run_checkpoint_eval(
    checkpoint_path: str,
    model_version: str = "v1",
    n_cases: int = 20,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    data_path: str | None = None,
    scenarios: list[Scenario] | None = None,
    device_str: str = "cpu",
    fold: int | None = None,
) -> list[AnyToAnyResult]:
    """Evaluate scenarios with a pretrained checkpoint (zero-shot)."""
    from downstream.model_wrapper import DownstreamModelWrapper

    print("=" * 70)
    print("Task 8: Any-to-Any Prediction — Zero-shot")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  model_version: {model_version}")
    print("=" * 70)

    device = torch.device(device_str)
    wrapper = DownstreamModelWrapper(checkpoint_path, model_version, device)
    model = wrapper.model
    patch_size = wrapper.patch_size

    if scenarios is None:
        scenarios = get_default_scenarios()

    results: list[AnyToAnyResult] = []

    if data_path and _data_path_available(data_path, fold=fold):
        # Evaluate from prepared .pt file
        data = _load_prepared_data(data_path, fold=fold)
        test_data = data["test"]

        for sc in scenarios:
            sc_metrics: list[dict] = []
            needed = set(sc.inputs + [sc.target])
            available = set(test_data["signals"].keys())
            if not needed.issubset(available):
                print(f"  SKIP: {sc.name} (missing signals: {needed - available})")
                continue

            n_samples = test_data["signals"][sc.inputs[0]].shape[0]
            for i in range(n_samples):
                signals_np = {
                    s: test_data["signals"][s][i].numpy() for s in needed
                }
                try:
                    batch = build_multivariate_batch(signals_np, patch_size=patch_size)
                    target_type_id = SIGNAL_TYPE_IDS[sc.target]
                    m = evaluate_scenario(model, batch, target_type_id, patch_size, device)
                    if m is not None:
                        sc_metrics.append(m)
                except Exception:
                    continue

            if sc_metrics:
                results.append(_aggregate_scenario_metrics(sc, sc_metrics))
            else:
                print(f"  SKIP: {sc.name} (no valid windows)")
    else:
        # Evaluate from VitalDB pilot cases
        from downstream.data_utils import load_pilot_cases

        all_needed = set()
        for sc in scenarios:
            all_needed.update(sc.inputs)
            all_needed.add(sc.target)
        cases = load_pilot_cases(n_cases=n_cases, signal_types=list(all_needed))

        for sc in scenarios:
            sc_metrics: list[dict] = []
            needed_types = set(sc.inputs + [sc.target])

            for case in cases:
                if not needed_types.issubset(case.tracks.keys()):
                    continue

                min_len = min(len(case.tracks[t]) for t in needed_types)
                win_samples = int(window_sec * 100.0)
                stride_samples = int(stride_sec * 100.0)

                for start in range(0, min_len - win_samples + 1, stride_samples):
                    signals = {
                        t: case.tracks[t][start : start + win_samples]
                        for t in needed_types
                    }
                    try:
                        batch = build_multivariate_batch(signals, patch_size=patch_size)
                        target_type_id = SIGNAL_TYPE_IDS[sc.target]
                        m = evaluate_scenario(
                            model, batch, target_type_id, patch_size, device
                        )
                        if m is not None:
                            sc_metrics.append(m)
                    except Exception:
                        continue

            if sc_metrics:
                results.append(_aggregate_scenario_metrics(sc, sc_metrics))
            else:
                print(f"  SKIP: {sc.name} (no valid windows)")

    if results:
        _print_results_table(results)
        _print_analysis(results)
    else:
        print("  No valid results.")

    return results


def _aggregate_scenario_metrics(
    sc: Scenario, sc_metrics: list[dict]
) -> AnyToAnyResult:
    """Aggregate per-window metrics into a single AnyToAnyResult."""
    return AnyToAnyResult(
        scenario_name=sc.name,
        inputs=sc.inputs,
        target=sc.target,
        group_type=sc.group_type,
        n_inputs=sc.n_inputs,
        recon_mse=round(np.mean([m["recon_mse"] for m in sc_metrics]), 6),
        recon_mae=round(np.mean([m["recon_mae"] for m in sc_metrics]), 6),
        recon_pearson_r=round(
            np.mean([m["recon_pearson_r"] for m in sc_metrics]), 4
        ),
        cross_mse=round(np.mean([m["cross_mse"] for m in sc_metrics]), 6),
        cross_mae=round(np.mean([m["cross_mae"] for m in sc_metrics]), 6),
        cross_pearson_r=round(
            np.mean([m["cross_pearson_r"] for m in sc_metrics]), 4
        ),
        n_patches=sum(m["n_patches"] for m in sc_metrics),
    )


# ── LoRA regression entry point ─────────────────────────────


def run_lora_regression(
    checkpoint_path: str,
    data_path: str,
    scenario: Scenario,
    model_version: str = "v1",
    epochs: int = 30,
    lr: float = 1e-4,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    batch_size: int = 16,
    device_str: str = "cpu",
    out_dir: str = ".",
    fold: int | None = None,
    n_folds: int = 1,
    dump_preds: bool = True,
) -> dict:
    """Run LoRA regression training and evaluation for a single scenario.

    Parameters
    ----------
    checkpoint_path : str
        Path to pretrained checkpoint.
    data_path : str
        Path to prepared .pt file.
    scenario : Scenario
        Input->target scenario.
    model_version : str
        Model version ("v1" or "v2").
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate.
    lora_rank : int
        LoRA rank.
    lora_alpha : float
        LoRA scaling factor.
    batch_size : int
        Batch size.
    device_str : str
        Device string.
    out_dir : str
        Output directory for results.

    Returns
    -------
    dict with metrics and config.
    """
    from downstream.model_wrapper import DownstreamModelWrapper

    inputs_str = " + ".join(s.upper() for s in scenario.inputs)
    target_str = scenario.target.upper()

    print("=" * 70)
    print(f"Task 8: Any-to-Any — LoRA Regression")
    print(f"  Scenario:   {inputs_str} -> {target_str}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data:       {data_path}")
    print(f"  LoRA:       rank={lora_rank}, alpha={lora_alpha}")
    print(f"  Training:   epochs={epochs}, lr={lr}, batch_size={batch_size}")
    print("=" * 70)

    device = torch.device(device_str)

    # Load model + inject LoRA
    wrapper = DownstreamModelWrapper(checkpoint_path, model_version, device)
    wrapper.inject_lora(rank=lora_rank, alpha=lora_alpha)
    patch_size = wrapper.patch_size

    # Load data (fold-aware)
    data = _load_prepared_data(data_path, fold=fold)

    # Build batches
    print("\nBuilding train batches...")
    train_batches = _build_lora_batches(
        data["train"], scenario, patch_size, batch_size
    )
    print(f"  Train: {len(train_batches)} batches")

    print("Building test batches...")
    test_batches = _build_lora_batches(
        data["test"], scenario, patch_size, batch_size
    )
    print(f"  Test: {len(test_batches)} batches")

    if not train_batches:
        print("ERROR: No train batches.", file=sys.stderr)
        sys.exit(1)

    # Create regression head
    head = WaveformRegressionHead(wrapper.d_model, patch_size)

    n_lora = sum(p.numel() for p in wrapper.lora_parameters())
    n_head = sum(p.numel() for p in head.parameters())
    print(f"\nTrainable params: LoRA={n_lora:,} + Head={n_head:,} = {n_lora + n_head:,}")

    # Train
    print(f"\nTraining...")
    train_losses = train_lora_regression(
        wrapper, head, train_batches, epochs, lr, device
    )

    # Evaluate (raw 수집은 preds dump 시에만)
    print(f"\nEvaluating on test set...")
    metrics = evaluate_lora_regression(
        wrapper, head, test_batches, device, collect_raw=dump_preds,
    )
    # raw 배열은 JSON 직렬화 불가 → 분리.
    raw_y_score = metrics.pop("raw_y_score", None)
    raw_y_true = metrics.pop("raw_y_true", None)

    # Print results
    print(f"\n{'=' * 50}")
    print(f"  Scenario:  {scenario.name}")
    print(f"  MSE:       {metrics['mse']:.6f}")
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  N patches: {metrics['n_patches']}")
    print(f"{'=' * 50}")

    # Save results
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenario_slug = scenario.name.replace("->", "_to_").replace("+", "_")
    results = {
        "scenario": scenario.name,
        "inputs": scenario.inputs,
        "target": scenario.target,
        "group_type": scenario.group_type,
        "metrics": metrics,
        "train_losses": train_losses,
        "config": {
            "mode": "lora",
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "data_path": data_path,
        },
    }

    results_file = out_path / f"task8_lora_{scenario_slug}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # ── Raw prediction dump (ablation aggregator 호환) ──────────────
    # forecast/classification 과 동일한 <out_dir>/preds_fold{idx}.npz 포맷.
    # generation: y_true / y_score 는 float waveform (N_win, L).
    if dump_preds and raw_y_true is not None and raw_y_score is not None:
        n_win = raw_y_true.shape[0]
        # patient_id: test split case_ids (윈도우 순서 보존) 를 정렬해서 매핑.
        case_ids = data.get("test", {}).get("case_ids", [])
        if len(case_ids) >= n_win:
            patient_id = np.array([str(c) for c in case_ids[:n_win]], dtype=str)
        else:
            patient_id = np.array([str(i) for i in range(n_win)], dtype=str)
        preds_path = out_path / f"preds_fold{int(fold) if fold is not None else 0}.npz"
        payload = {
            "y_true": raw_y_true.astype(np.float32),
            "y_score": raw_y_score.astype(np.float32),
            "patient_id": patient_id,
            "fold_idx": np.int64(int(fold) if fold is not None else 0),
            "n_folds": np.int64(int(n_folds)),
            "task": f"cross_modal_{scenario_slug}",
            "classes": np.array([], dtype=str),
            "extra__scenario": np.full((n_win,), scenario.name, dtype=object).astype(str),
            "extra__target": np.full((n_win,), scenario.target, dtype=object).astype(str),
            "extra__group_type": np.full(
                (n_win,), scenario.group_type, dtype=object,
            ).astype(str),
        }
        np.savez_compressed(preds_path, **payload)
        print(f"Predictions saved: {preds_path} "
              f"(y_true={raw_y_true.shape}, y_score={raw_y_score.shape}, "
              f"patients={len(patient_id)})")

    return results


# ── Linear-probe regression entry point ─────────────────────


def train_linear_probe_regression(
    model,  # DownstreamModelWrapper (encoder frozen, no LoRA)
    head: WaveformRegressionHead,
    train_batches: list[tuple[PackedBatch, torch.Tensor]],
    epochs: int,
    lr: float,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> list[float]:
    """Train only WaveformRegressionHead on top of a frozen encoder.

    LoRA 를 주입하지 않고 encoder 를 eval()+no_grad 로 고정한 채 head 만 학습한다
    (frozen linear probe). forward/pooling 로직은 train_lora_regression 과 동일.
    """
    model.model.eval()
    for p in model.model.parameters():
        p.requires_grad_(False)
    head = head.to(device)
    head.train()

    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)

    losses: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for batch, target_patches in train_batches:
            batch = model.batch_to_device(batch)
            target_patches = target_patches.to(device)  # (B, N_target, patch_size)

            # encoder 는 frozen → no_grad 로 feature 추출 (메모리 절감, grad 차단)
            with torch.no_grad():
                out = model.model(batch, task="masked", mask_ratio=0.0)
                encoded = out["encoded"]      # (B, N_total, d_model)
                time_id = out["time_id"]      # (B, N_total)
                patch_mask = out["patch_mask"]  # (B, N_total) bool
                pooled = _pool_by_time(encoded, time_id, patch_mask)

            predicted = head(pooled)  # (B, N_time, patch_size)

            n_out = min(predicted.shape[1], target_patches.shape[1])
            predicted = predicted[:, :n_out]
            target = target_patches[:, :n_out]

            loss = F.mse_loss(predicted, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.6f}")

    return losses


def run_linear_probe_regression(
    checkpoint_path: str,
    data_path: str,
    scenario: Scenario,
    model_version: str = "v1",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    device_str: str = "cpu",
    out_dir: str = ".",
    fold: int | None = None,
    n_folds: int = 1,
    dump_preds: bool = True,
) -> dict:
    """Frozen linear-probe regression for a single cross-modal scenario.

    run_lora_regression 과 동일하나 LoRA 주입 없이 frozen encoder + regression
    head 만 학습한다. 저장/preds dump 포맷은 lora 와 동일 (ablation aggregator 호환).
    """
    from downstream.model_wrapper import DownstreamModelWrapper

    inputs_str = " + ".join(s.upper() for s in scenario.inputs)
    target_str = scenario.target.upper()

    print("=" * 70)
    print(f"Task 8: Any-to-Any — Linear Probe (frozen encoder)")
    print(f"  Scenario:   {inputs_str} -> {target_str}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data:       {data_path}")
    print(f"  Training:   epochs={epochs}, lr={lr}, batch_size={batch_size}")
    print("=" * 70)

    device = torch.device(device_str)

    # Load model (NO LoRA injection → encoder frozen)
    wrapper = DownstreamModelWrapper(checkpoint_path, model_version, device)
    patch_size = wrapper.patch_size

    # Load data (fold-aware)
    data = _load_prepared_data(data_path, fold=fold)

    print("\nBuilding train batches...")
    train_batches = _build_lora_batches(
        data["train"], scenario, patch_size, batch_size
    )
    print(f"  Train: {len(train_batches)} batches")

    print("Building test batches...")
    test_batches = _build_lora_batches(
        data["test"], scenario, patch_size, batch_size
    )
    print(f"  Test: {len(test_batches)} batches")

    if not train_batches:
        print("ERROR: No train batches.", file=sys.stderr)
        sys.exit(1)

    head = WaveformRegressionHead(wrapper.d_model, patch_size)
    n_head = sum(p.numel() for p in head.parameters())
    print(f"\nTrainable params: Head={n_head:,} (encoder frozen)")

    print(f"\nTraining...")
    train_losses = train_linear_probe_regression(
        wrapper, head, train_batches, epochs, lr, device
    )

    print(f"\nEvaluating on test set...")
    metrics = evaluate_lora_regression(
        wrapper, head, test_batches, device, collect_raw=dump_preds,
    )
    raw_y_score = metrics.pop("raw_y_score", None)
    raw_y_true = metrics.pop("raw_y_true", None)

    print(f"\n{'=' * 50}")
    print(f"  Scenario:  {scenario.name}")
    print(f"  MSE:       {metrics['mse']:.6f}")
    print(f"  MAE:       {metrics['mae']:.6f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  N patches: {metrics['n_patches']}")
    print(f"{'=' * 50}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenario_slug = scenario.name.replace("->", "_to_").replace("+", "_")
    results = {
        "scenario": scenario.name,
        "inputs": scenario.inputs,
        "target": scenario.target,
        "group_type": scenario.group_type,
        "metrics": metrics,
        "train_losses": train_losses,
        "config": {
            "mode": "linear_probe",
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "data_path": data_path,
        },
    }

    results_file = out_path / f"task8_linear_probe_{scenario_slug}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_file}")

    # ── Raw prediction dump (ablation aggregator 호환) ──────────────
    if dump_preds and raw_y_true is not None and raw_y_score is not None:
        n_win = raw_y_true.shape[0]
        case_ids = data.get("test", {}).get("case_ids", [])
        if len(case_ids) >= n_win:
            patient_id = np.array([str(c) for c in case_ids[:n_win]], dtype=str)
        else:
            patient_id = np.array([str(i) for i in range(n_win)], dtype=str)
        preds_path = out_path / f"preds_fold{int(fold) if fold is not None else 0}.npz"
        payload = {
            "y_true": raw_y_true.astype(np.float32),
            "y_score": raw_y_score.astype(np.float32),
            "patient_id": patient_id,
            "fold_idx": np.int64(int(fold) if fold is not None else 0),
            "n_folds": np.int64(int(n_folds)),
            "task": f"cross_modal_{scenario_slug}",
            "classes": np.array([], dtype=str),
            "extra__scenario": np.full((n_win,), scenario.name, dtype=object).astype(str),
            "extra__target": np.full((n_win,), scenario.target, dtype=object).astype(str),
            "extra__group_type": np.full(
                (n_win,), scenario.group_type, dtype=object,
            ).astype(str),
        }
        np.savez_compressed(preds_path, **payload)
        print(f"Predictions saved: {preds_path} "
              f"(y_true={raw_y_true.shape}, y_score={raw_y_score.shape}, "
              f"patients={len(patient_id)})")

    return results


# ── Output ───────────────────────────────────────────────────


def _print_results_table(results: list[AnyToAnyResult]) -> None:
    """Print results comparing reconstructed vs cross_pred heads."""
    hdr = (
        f"\n{'Scenario':<22} {'Group':<7} {'#In':<4} "
        f"{'Recon MSE':<11} {'Recon r':<9} "
        f"{'Cross MSE':<11} {'Cross r':<9} "
        f"{'#Patches':<9}"
    )
    print(hdr)
    print("-" * len(hdr.strip()))
    for r in results:
        print(
            f"{r.scenario_name:<22} {r.group_type:<7} {r.n_inputs:<4} "
            f"{r.recon_mse:<11.6f} {r.recon_pearson_r:<9.4f} "
            f"{r.cross_mse:<11.6f} {r.cross_pearson_r:<9.4f} "
            f"{r.n_patches:<9}"
        )


def _print_analysis(results: list[AnyToAnyResult]) -> None:
    """Print mechanism group and input count analysis."""
    if not results:
        return

    print("\n--- Mechanism Group Analysis ---")
    for group in ["intra", "inter"]:
        grp = [r for r in results if r.group_type == group]
        if grp:
            avg_recon_r = np.mean([r.recon_pearson_r for r in grp])
            avg_cross_r = np.mean([r.cross_pearson_r for r in grp])
            print(
                f"  {group:5s}: recon_r={avg_recon_r:.4f}  cross_r={avg_cross_r:.4f}  (n={len(grp)} scenarios)"
            )

    print("\n--- Input Channel Count Effect ---")
    for n_in in sorted(set(r.n_inputs for r in results)):
        grp = [r for r in results if r.n_inputs == n_in and r.group_type == "intra"]
        if grp:
            avg_recon_r = np.mean([r.recon_pearson_r for r in grp])
            avg_cross_r = np.mean([r.cross_pearson_r for r in grp])
            print(
                f"  {n_in}->1: recon_r={avg_recon_r:.4f}  cross_r={avg_cross_r:.4f}  (n={len(grp)} scenarios)"
            )

    # Which head is better?
    recon_wins = sum(1 for r in results if r.recon_pearson_r > r.cross_pearson_r)
    cross_wins = sum(1 for r in results if r.cross_pearson_r > r.recon_pearson_r)
    print("\n--- Head Comparison ---")
    print(f"  reconstructed wins: {recon_wins}/{len(results)}")
    print(f"  cross_pred wins:    {cross_wins}/{len(results)}")


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Task 8: Any-to-Any cross-modal prediction"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--mode",
        type=str,
        default="zero_shot",
        choices=["zero_shot", "linear_probe", "lora"],
        help="zero_shot: pretrained cross_pred head, "
             "linear_probe: frozen encoder + regression head, "
             "lora: LoRA + regression head",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to prepared .pt file (from prepare_data.py)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario string. Fixed: 'ECG->ABP', 'ECG+PPG->ABP'. "
             "Generalized: 'random_subset' (+ --mask-ratio, --n-iterations).",
    )
    parser.add_argument(
        "--mask-ratio", type=float, nargs="+", default=[0.25, 0.50, 0.75],
        help="'random_subset' 시 hidden 비율 리스트 (e.g., 0.25 0.5 0.75). "
             "각 비율마다 전체 signal 대비 얼마나 가리고 나머지로 예측하는지.",
    )
    parser.add_argument(
        "--n-iterations", type=int, default=3,
        help="'random_subset' 시 mask_ratio별 random seed 반복 수 (분산 감소).",
    )
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--epochs", type=int, default=30, help="LoRA training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA alpha")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for LoRA")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/downstream/any_to_any",
        help="Output directory for results",
    )
    parser.add_argument(
        "--dummy", action="store_true", help="Dummy test with random model"
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="K-fold CV fold index. n_folds>1 일 때 해당 fold chunk 를 로드.",
    )
    parser.add_argument(
        "--n-folds", type=int, default=1,
        help="총 fold 수. 1 이면 비-fold/단일 data_path (back-compat), "
             ">=2 면 _fold{F} chunk 를 fold 별로 로드.",
    )
    parser.add_argument(
        "--no-dump-preds", action="store_true",
        help="preds_fold{idx}.npz raw waveform dump 생략 (lora 모드 기본은 dump).",
    )
    args = parser.parse_args()

    # n_folds>1 이면 해당 fold chunk 만 로드, 1 이면 비-fold/단일 (None).
    load_fold = int(args.fold) if int(args.n_folds) > 1 else None

    if args.dummy:
        run_dummy_test()
        return

    if not args.checkpoint:
        print("Error: provide --checkpoint or --dummy", file=sys.stderr)
        sys.exit(1)

    if args.mode == "zero_shot":
        # Random subset (generalized) scenario
        if args.scenario and args.scenario.lower() == "random_subset":
            run_random_subset_eval(
                checkpoint_path=args.checkpoint,
                model_version=args.model_version,
                n_cases=args.n_cases,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                data_path=args.data_path,
                mask_ratios=args.mask_ratio,
                n_iterations=args.n_iterations,
                device_str=args.device,
                fold=load_fold,
            )
            return

        # Fixed-scenario path (기존)
        scenarios = None
        if args.scenario:
            scenarios = [parse_scenario(args.scenario)]

        run_checkpoint_eval(
            checkpoint_path=args.checkpoint,
            model_version=args.model_version,
            n_cases=args.n_cases,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
            data_path=args.data_path,
            scenarios=scenarios,
            device_str=args.device,
            fold=load_fold,
        )

    elif args.mode == "linear_probe":
        if not args.data_path:
            print("Error: --data-path required for linear_probe mode", file=sys.stderr)
            sys.exit(1)
        if not args.scenario:
            print("Error: --scenario required for linear_probe mode (e.g. 'PPG->ABP')",
                  file=sys.stderr)
            sys.exit(1)

        scenario = parse_scenario(args.scenario)
        run_linear_probe_regression(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            scenario=scenario,
            model_version=args.model_version,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device_str=args.device,
            out_dir=args.out_dir,
            fold=load_fold,
            n_folds=args.n_folds,
            dump_preds=not args.no_dump_preds,
        )

    elif args.mode == "lora":
        if not args.data_path:
            print("Error: --data-path required for lora mode", file=sys.stderr)
            sys.exit(1)
        if not args.scenario:
            print("Error: --scenario required for lora mode (e.g. 'ECG->ABP')",
                  file=sys.stderr)
            sys.exit(1)

        scenario = parse_scenario(args.scenario)
        run_lora_regression(
            checkpoint_path=args.checkpoint,
            data_path=args.data_path,
            scenario=scenario,
            model_version=args.model_version,
            epochs=args.epochs,
            lr=args.lr,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            batch_size=args.batch_size,
            device_str=args.device,
            out_dir=args.out_dir,
            fold=load_fold,
            n_folds=args.n_folds,
            dump_preds=not args.no_dump_preds,
        )


if __name__ == "__main__":
    main()
