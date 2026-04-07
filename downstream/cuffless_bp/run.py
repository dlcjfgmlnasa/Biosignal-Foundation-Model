# -*- coding:utf-8 -*-
"""Cuffless BP - 평가 스크립트.

PPG/ECG 입력 -> ABP waveform 복원 -> SBP/DBP 추출.
Zero-shot (forward(masked) → cross_pred) — 추가 학습 없음.

사용법:
    python -m downstream.cuffless_bp.run --dummy
    python -m downstream.cuffless_bp.run --checkpoint path/to/best.pt \
        --data-path outputs/downstream/cuffless_bp/cuffless_bp_mimic3_ppg.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample
from data.parser.vitaldb import SIGNAL_TYPES

PATCH_SIZE = 100
TARGET_SR = 100.0

ABP_SIGNAL_TYPE = SIGNAL_TYPES["abp"]  # 1


# ---- SBP/DBP extraction ----


def extract_sbp_dbp(abp_window: np.ndarray, sr: float = 100.0) -> tuple[float, float] | None:
    """ABP waveform -> (SBP, DBP) in mmHg."""
    from scipy.signal import find_peaks

    if len(abp_window) < int(sr * 1):
        return None

    q75, q25 = np.percentile(abp_window, [75, 25])
    iqr = q75 - q25
    if iqr < 5:
        return None

    min_dist = max(1, int(sr * 0.4))
    peaks, _ = find_peaks(abp_window, prominence=iqr * 0.3, distance=min_dist)
    if len(peaks) < 2:
        return None

    sbp = float(np.median(abp_window[peaks]))

    troughs, _ = find_peaks(-abp_window, prominence=iqr * 0.3, distance=min_dist)
    dbp = float(np.median(abp_window[troughs])) if len(troughs) >= 1 else float(np.min(abp_window))

    return (sbp, dbp)


# ---- BHS Grade ----


def _bhs_grade(errors: np.ndarray) -> str:
    abs_e = np.abs(errors)
    p5 = float((abs_e <= 5).mean())
    p10 = float((abs_e <= 10).mean())
    p15 = float((abs_e <= 15).mean())
    if p5 >= 0.60 and p10 >= 0.85 and p15 >= 0.95:
        return "A"
    if p5 >= 0.50 and p10 >= 0.75 and p15 >= 0.90:
        return "B"
    if p5 >= 0.40 and p10 >= 0.65 and p15 >= 0.85:
        return "C"
    return "D"


# ---- Dummy model ----


class DummyModel:
    def __init__(self, patch_size=100):
        self.patch_size = patch_size

    def eval(self):
        return self

    def __call__(self, batch, task="masked", **kwargs):
        b, l = batch.values.shape
        p = self.patch_size
        n = l // p
        return {
            "reconstructed": torch.randn(b, n, p),
            "cross_pred": torch.randn(b, n, p),
            "patch_signal_types": torch.zeros(b, n, dtype=torch.long),
            "loc": torch.zeros(b, l, 1),
            "scale": torch.ones(b, l, 1),
            "patch_mask": torch.ones(b, n, dtype=torch.bool),
        }


# ---- Batch construction (multi-variate) ----


def _build_multivariate_batch(
    input_signals: dict[str, torch.Tensor],
    target_abp: torch.Tensor,
    patch_size: int,
) -> PackedBatch:
    """Input signals + target ABP -> multi-variate PackedBatch."""
    # Combine: input signals + ABP (ABP will be masked)
    all_signals = {**input_signals, "abp": target_abp}
    samples = []
    for ch_idx, (stype_key, signal) in enumerate(all_signals.items()):
        stype_int = SIGNAL_TYPES.get(stype_key, 0)
        n_usable = (len(signal) // patch_size) * patch_size
        if n_usable == 0:
            continue
        samples.append(BiosignalSample(
            values=signal[:n_usable].float(),
            length=n_usable,
            channel_idx=ch_idx,
            recording_idx=0,
            sampling_rate=TARGET_SR,
            n_channels=len(all_signals),
            win_start=0,
            signal_type=stype_int,
            session_id="cuffless_0",
            spatial_id=0,
        ))

    max_length = sum(s.length for s in samples)
    collate = PackCollate(max_length=max_length, collate_mode="any_variate", patch_size=patch_size)
    return collate(samples)


# ---- Evaluation ----


@torch.no_grad()
def evaluate_cuffless_bp(
    model,
    test_data: dict,
    input_signal_keys: list[str],
    patch_size: int,
    device: str = "cpu",
) -> dict:
    """Cuffless BP: input -> ABP waveform reconstruction -> SBP/DBP."""
    input_tensors = test_data["input_signals"]  # {stype: (N, win)}
    target_abp = test_data["target_abp"]        # (N, win)

    n_samples = target_abp.shape[0]

    waveform_mse, waveform_mae, waveform_r = [], [], []
    sbp_errors, dbp_errors = [], []

    for i in range(n_samples):
        # Build batch
        inp = {k: input_tensors[k][i] for k in input_signal_keys if k in input_tensors}
        abp = target_abp[i]

        try:
            batch = _build_multivariate_batch(inp, abp, patch_size)
        except Exception:
            continue

        batch.values = batch.values.to(device)
        batch.sample_id = batch.sample_id.to(device)
        batch.variate_id = batch.variate_id.to(device)

        out = model(batch, task="masked")

        reconstructed = out["reconstructed"]
        cross_pred = out.get("cross_pred", reconstructed)
        patch_signal_types = out.get("patch_signal_types")
        patch_mask = out["patch_mask"]
        loc = out["loc"]
        scale = out["scale"]

        b, l = batch.values.shape
        n = l // patch_size

        # Find ABP patches
        if patch_signal_types is not None:
            abp_mask = (patch_signal_types == ABP_SIGNAL_TYPE) & patch_mask
        else:
            abp_mask = patch_mask

        if not abp_mask.any():
            continue

        # Original ABP patches (normalized)
        normalized = ((batch.values.unsqueeze(-1) - loc) / scale.clamp(min=1e-8)).squeeze(-1)
        original_patches = normalized[:, :n * patch_size].reshape(b, n, patch_size)

        # Denormalize for SBP/DBP
        loc_patches = loc.squeeze(-1)[:, :n * patch_size].reshape(b, n, patch_size)
        scale_patches = scale.squeeze(-1)[:, :n * patch_size].reshape(b, n, patch_size)

        orig_abp = original_patches[abp_mask].cpu()
        pred_abp = cross_pred[abp_mask].cpu()

        # Waveform metrics (normalized scale)
        orig_flat = orig_abp.numpy().reshape(-1)
        pred_flat = pred_abp.numpy().reshape(-1)

        if len(orig_flat) == 0:
            continue

        waveform_mse.append(float(np.mean((orig_flat - pred_flat) ** 2)))
        waveform_mae.append(float(np.mean(np.abs(orig_flat - pred_flat))))

        if np.std(orig_flat) > 1e-8 and np.std(pred_flat) > 1e-8:
            waveform_r.append(float(np.corrcoef(orig_flat, pred_flat)[0, 1]))

        # Denormalized ABP for SBP/DBP
        abp_loc = loc_patches[abp_mask].cpu()
        abp_scale = scale_patches[abp_mask].cpu()
        orig_denorm = (orig_abp * abp_scale + abp_loc).numpy().reshape(-1)
        pred_denorm = (pred_abp * abp_scale + abp_loc).numpy().reshape(-1)

        orig_bp = extract_sbp_dbp(orig_denorm)
        pred_bp = extract_sbp_dbp(pred_denorm)
        if orig_bp and pred_bp:
            sbp_errors.append(pred_bp[0] - orig_bp[0])
            dbp_errors.append(pred_bp[1] - orig_bp[1])

    results = {
        "waveform_mse": float(np.mean(waveform_mse)) if waveform_mse else 0.0,
        "waveform_mae": float(np.mean(waveform_mae)) if waveform_mae else 0.0,
        "waveform_pearson_r": float(np.mean(waveform_r)) if waveform_r else 0.0,
        "n_waveform": len(waveform_mse),
    }

    if sbp_errors:
        sbp_e = np.array(sbp_errors)
        dbp_e = np.array(dbp_errors)
        results.update({
            "sbp_mae": float(np.mean(np.abs(sbp_e))),
            "dbp_mae": float(np.mean(np.abs(dbp_e))),
            "sbp_me": float(np.mean(sbp_e)),
            "sbp_sde": float(np.std(sbp_e)),
            "dbp_me": float(np.mean(dbp_e)),
            "dbp_sde": float(np.std(dbp_e)),
            "bhs_sbp": _bhs_grade(sbp_e),
            "bhs_dbp": _bhs_grade(dbp_e),
            "n_bp": len(sbp_errors),
        })

    return results


# ---- Main ----


def main() -> None:
    parser = argparse.ArgumentParser(description="Cuffless BP Evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/cuffless_bp")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Model
    if args.dummy:
        print("Using dummy model (pipeline verification)")
        model = DummyModel()
        device = "cpu"
    elif args.checkpoint:
        from downstream.model_wrapper import DownstreamModelWrapper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapper = DownstreamModelWrapper(args.checkpoint, args.model_version, device)
        model = wrapper.model
    else:
        print("ERROR: --checkpoint or --dummy required.", file=sys.stderr)
        sys.exit(1)

    # Data
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading prepared data: {args.data_path}")
        data = torch.load(args.data_path, weights_only=False)
        meta = data["metadata"]
        input_keys = meta["input_signals"]
        test_data = data["test"]
        n_test = test_data["target_abp"].shape[0]
        print(f"  Input: {input_keys}, Test: {n_test} samples")
    elif args.dummy:
        input_keys = ["ppg"]
        test_data = {
            "input_signals": {"ppg": torch.randn(30, 1000)},
            "target_abp": torch.randn(30, 1000),
        }
        print("  Dummy: PPG->ABP, 30 synthetic samples")
    else:
        print("ERROR: --data-path required.", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    print(f"\nEvaluating Cuffless BP ({'+'.join(k.upper() for k in input_keys)} -> ABP)...")
    metrics = evaluate_cuffless_bp(
        model, test_data, input_keys,
        patch_size=PATCH_SIZE, device=device,
    )

    mode_str = "+".join(k.upper() for k in input_keys)
    print(f"\n{'='*55}")
    print(f"  Cuffless BP: {mode_str} -> ABP")
    print(f"{'='*55}")
    print(f"  Waveform MSE:       {metrics['waveform_mse']:.6f}")
    print(f"  Waveform MAE:       {metrics['waveform_mae']:.6f}")
    print(f"  Waveform Pearson r: {metrics['waveform_pearson_r']:.4f}")
    if "sbp_mae" in metrics:
        print(f"  ---")
        print(f"  SBP MAE: {metrics['sbp_mae']:.1f} mmHg (ME={metrics['sbp_me']:.1f}, SDE={metrics['sbp_sde']:.1f})")
        print(f"  DBP MAE: {metrics['dbp_mae']:.1f} mmHg (ME={metrics['dbp_me']:.1f}, SDE={metrics['dbp_sde']:.1f})")
        print(f"  BHS Grade: SBP={metrics['bhs_sbp']}, DBP={metrics['bhs_dbp']}")
        print(f"  AAMI pass: SBP={'PASS' if abs(metrics['sbp_me'])<=5 and metrics['sbp_sde']<=8 else 'FAIL'}, "
              f"DBP={'PASS' if abs(metrics['dbp_me'])<=5 and metrics['dbp_sde']<=8 else 'FAIL'}")
    print(f"{'='*55}")

    results_path = out_dir / f"cuffless_bp_{mode_str}_results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved: {results_path}")


if __name__ == "__main__":
    main()
