# -*- coding:utf-8 -*-
"""Task 8: Any-to-Any Cross-modal Prediction.

임의의 input 신호 조합에서 target 신호를 예측한다.
reconstructed (self-reconstruction head)와 cross_pred (cross-modal head) 모두 평가.

Usage
-----
# Dummy test (no checkpoint):
python -m downstream.any_to_any.run --dummy

# Real evaluation:
python -m downstream.any_to_any.run --checkpoint path/to/best.pt
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from data.collate import PackCollate, PackedBatch
from data.dataset import BiosignalSample

# signal_type_key -> signal_type_id
SIGNAL_TYPE_IDS: dict[str, int] = {
    "ecg": 0, "abp": 1, "ppg": 2, "cvp": 3, "co2": 4, "awp": 5, "pap": 6, "icp": 7,
}

# Mechanism groups for analysis
MECHANISM_GROUPS: dict[str, str] = {
    "ecg": "cardiovascular",
    "abp": "cardiovascular",
    "ppg": "cardiovascular",
    "cvp": "cardiovascular",
    "co2": "respiratory",
    "awp": "respiratory",
    "pap": "cardiovascular",
    "icp": "neurological",
}


# ── Scenarios ────────────────────────────────────────────────

@dataclass
class Scenario:
    """Input -> target prediction scenario."""
    name: str
    inputs: list[str]       # signal type keys to provide
    target: str             # signal type key to predict
    group_type: str         # "intra" (same mechanism group) or "inter"
    n_inputs: int           # number of input channels (for analysis)


def get_default_scenarios() -> list[Scenario]:
    """Default evaluation scenarios spanning various complexities."""
    scenarios = [
        # 1-to-1: Cardiovascular intra-group
        Scenario("ECG->ABP",    ["ecg"],             "abp", "intra", 1),
        Scenario("ABP->ECG",    ["abp"],             "ecg", "intra", 1),
        Scenario("PPG->ABP",    ["ppg"],             "abp", "intra", 1),
        # 2-to-1: Cardiovascular intra-group
        Scenario("ECG+PPG->ABP", ["ecg", "ppg"],     "abp", "intra", 2),
        Scenario("ECG+ABP->PPG", ["ecg", "abp"],     "ppg", "intra", 2),
        # 3-to-1: Rich input
        Scenario("ECG+PPG+CVP->ABP", ["ecg", "ppg", "cvp"], "abp", "intra", 3),
        # Inter-group (baseline - expect low performance)
        Scenario("ECG->CO2",    ["ecg"],             "co2", "inter", 1),
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

        samples.append(BiosignalSample(
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
        ))

    if not samples:
        raise ValueError("No valid signals for batch construction")

    max_length = sum(s.length for s in samples)
    collate = PackCollate(
        max_length=max_length,
        collate_mode="any_variate",
        patch_size=patch_size,
    )
    return collate(samples)


# ── Evaluation ───────────────────────────────────────────────

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

    reconstructed = out["reconstructed"]         # (B, N, patch_size)
    cross_pred = out.get("cross_pred")           # (B, N, patch_size) or None
    patch_signal_types = out["patch_signal_types"]  # (B, N)
    patch_mask = out["patch_mask"]               # (B, N) bool

    if patch_signal_types is None:
        return None

    # Original patches (normalized scale)
    normalized = (
        (batch.values.unsqueeze(-1) - out["loc"]) / out["scale"]
    ).squeeze(-1)
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
        "recon_mse": recon_mse, "recon_mae": recon_mae, "recon_pearson_r": recon_r,
        "cross_mse": cross_mse, "cross_mae": cross_mae, "cross_pearson_r": cross_r,
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
    "ecg": 1.2, "abp": 1.1, "ppg": 1.0, "cvp": 0.9,
    "co2": 0.25, "awp": 0.3, "pap": 1.0, "icp": 0.15,
}


def _gen_synthetic(stype: str, n_samples: int = 3000, sr: float = 100.0) -> np.ndarray:
    """Generate synthetic signal for a given type."""
    t = np.arange(n_samples) / sr
    freq = _SYNTH_FREQS.get(stype, 1.0)
    sig = np.sin(2 * np.pi * freq * t)
    sig += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    sig += 0.1 * np.random.randn(n_samples)
    return sig


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

        metrics = evaluate_scenario(model, batch, target_type_id, config.patch_size, device)
        if metrics is None:
            print(f"  SKIP: {sc.name} (target not found in batch)")
            continue

        results.append(AnyToAnyResult(
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
        ))

    _print_results_table(results)
    _print_analysis(results)
    return results


# ── Real evaluation ──────────────────────────────────────────

def run_checkpoint_eval(
    checkpoint_path: str,
    model_version: str = "v1",
    n_cases: int = 20,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
) -> list[AnyToAnyResult]:
    """Evaluate all scenarios with a pretrained checkpoint."""
    from downstream.model_wrapper import DownstreamModelWrapper
    from downstream.data_utils import load_pilot_cases

    print("=" * 70)
    print(f"Task 8: Any-to-Any Prediction")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  model_version: {model_version}")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper = DownstreamModelWrapper(checkpoint_path, model_version, device)
    model = wrapper.model
    patch_size = wrapper.patch_size

    scenarios = get_default_scenarios()

    # Load all signal types needed across scenarios
    all_needed = set()
    for sc in scenarios:
        all_needed.update(sc.inputs)
        all_needed.add(sc.target)
    cases = load_pilot_cases(n_cases=n_cases, signal_types=list(all_needed))

    results: list[AnyToAnyResult] = []

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
                signals = {t: case.tracks[t][start:start + win_samples] for t in needed_types}

                try:
                    batch = build_multivariate_batch(signals, patch_size=patch_size)
                    target_type_id = SIGNAL_TYPE_IDS[sc.target]
                    m = evaluate_scenario(model, batch, target_type_id, patch_size, device)
                    if m is not None:
                        sc_metrics.append(m)
                except Exception as e:
                    continue

        if sc_metrics:
            # Aggregate across windows
            results.append(AnyToAnyResult(
                scenario_name=sc.name,
                inputs=sc.inputs,
                target=sc.target,
                group_type=sc.group_type,
                n_inputs=sc.n_inputs,
                recon_mse=round(np.mean([m["recon_mse"] for m in sc_metrics]), 6),
                recon_mae=round(np.mean([m["recon_mae"] for m in sc_metrics]), 6),
                recon_pearson_r=round(np.mean([m["recon_pearson_r"] for m in sc_metrics]), 4),
                cross_mse=round(np.mean([m["cross_mse"] for m in sc_metrics]), 6),
                cross_mae=round(np.mean([m["cross_mae"] for m in sc_metrics]), 6),
                cross_pearson_r=round(np.mean([m["cross_pearson_r"] for m in sc_metrics]), 4),
                n_patches=sum(m["n_patches"] for m in sc_metrics),
            ))
        else:
            print(f"  SKIP: {sc.name} (no valid windows)")

    if results:
        _print_results_table(results)
        _print_analysis(results)
    else:
        print("  No valid results.")

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
            print(f"  {group:5s}: recon_r={avg_recon_r:.4f}  cross_r={avg_cross_r:.4f}  (n={len(grp)} scenarios)")

    print("\n--- Input Channel Count Effect ---")
    for n_in in sorted(set(r.n_inputs for r in results)):
        grp = [r for r in results if r.n_inputs == n_in and r.group_type == "intra"]
        if grp:
            avg_recon_r = np.mean([r.recon_pearson_r for r in grp])
            avg_cross_r = np.mean([r.cross_pearson_r for r in grp])
            print(f"  {n_in}->1: recon_r={avg_recon_r:.4f}  cross_r={avg_cross_r:.4f}  (n={len(grp)} scenarios)")

    # Which head is better?
    recon_wins = sum(1 for r in results if r.recon_pearson_r > r.cross_pearson_r)
    cross_wins = sum(1 for r in results if r.cross_pearson_r > r.recon_pearson_r)
    print(f"\n--- Head Comparison ---")
    print(f"  reconstructed wins: {recon_wins}/{len(results)}")
    print(f"  cross_pred wins:    {cross_wins}/{len(results)}")


# ── CLI ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Task 8: Any-to-Any cross-modal prediction")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--n-cases", type=int, default=20)
    parser.add_argument("--window-sec", type=float, default=30.0)
    parser.add_argument("--stride-sec", type=float, default=15.0)
    parser.add_argument("--dummy", action="store_true", help="Dummy test with random model")
    args = parser.parse_args()

    if args.dummy:
        run_dummy_test()
    elif args.checkpoint:
        run_checkpoint_eval(
            checkpoint_path=args.checkpoint,
            model_version=args.model_version,
            n_cases=args.n_cases,
            window_sec=args.window_sec,
            stride_sec=args.stride_sec,
        )
    else:
        print("Error: provide --checkpoint or --dummy")
        sys.exit(1)


if __name__ == "__main__":
    main()
