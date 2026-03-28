# -*- coding:utf-8 -*-
"""Task 4: Cross-modal 예측 (ECG → ABP).

ECG 입력만으로 ABP 파형을 생성하여 Foundation model의
cross-modal generation 능력을 검증한다.

방식: collate_mode="any_variate" → ECG+ABP 다변량 입력 →
ABP 전체 패치를 variate-level 마스킹 → cross_pred 중 ABP 위치 추출.

Usage
-----
# 실제 모델로 평가:
python -m downstream.task4_cross_modal.run --checkpoint outputs/phase2_v1/best.pt --n-cases 30

# 더미 테스트 (checkpoint 없이):
python -m downstream.task4_cross_modal.run --dummy
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.collate import PackedBatch
from data.spatial_map import get_global_spatial_id


# ── 데이터 구조 ───────────────────────────────────────────────


ECG_SIGNAL_TYPE = 0
ABP_SIGNAL_TYPE = 1
TARGET_SR = 100.0


@dataclass
class CrossModalSample:
    """ECG+ABP 동시 존재 윈도우."""
    ecg: np.ndarray     # (win_samples,) at 100Hz
    abp: np.ndarray     # (win_samples,) at 100Hz — ground truth target
    case_id: int


# ── 데이터 준비 ───────────────────────────────────────────────


def find_ecg_abp_cases(
    n_cases: int = 30,
    signal_types: list[str] | None = None,
) -> list:
    """ECG+ABP가 동시 존재하는 pilot case들을 로드한다."""
    from downstream.common.data_utils import load_pilot_cases

    if signal_types is None:
        signal_types = ["ecg", "abp"]

    cases = load_pilot_cases(n_cases=n_cases, signal_types=signal_types)
    # ECG와 ABP가 모두 존재하는 케이스만 필터
    return [c for c in cases if "ecg" in c.tracks and "abp" in c.tracks]


def extract_paired_windows(
    cases: list,
    window_sec: float = 30.0,
    stride_sec: float = 15.0,
    sr: float = TARGET_SR,
) -> list[CrossModalSample]:
    """ECG+ABP가 동시 존재하는 윈도우를 추출한다.

    두 신호의 겹치는 시간 구간에서 슬라이딩 윈도우를 생성한다.
    """
    win_samples = int(window_sec * sr)
    stride_samples = int(stride_sec * sr)
    results: list[CrossModalSample] = []

    for case in cases:
        ecg = case.tracks["ecg"]
        abp = case.tracks["abp"]

        # 겹치는 길이
        overlap_len = min(len(ecg), len(abp))
        if overlap_len < win_samples:
            continue

        for start in range(0, overlap_len - win_samples + 1, stride_samples):
            end = start + win_samples
            results.append(CrossModalSample(
                ecg=ecg[start:end],
                abp=abp[start:end],
                case_id=case.case_id,
            ))

    return results


def create_dummy_samples(n_samples: int = 20, win_samples: int = 3000) -> list[CrossModalSample]:
    """더미 ECG+ABP 데이터를 생성한다 (checkpoint 없이 파이프라인 검증용)."""
    rng = np.random.default_rng(42)
    samples = []
    t = np.arange(win_samples) / TARGET_SR

    for i in range(n_samples):
        hr = rng.uniform(60, 100)  # bpm
        freq = hr / 60.0

        # 단순 합성 ECG: R-peak spike + baseline
        ecg = np.zeros(win_samples, dtype=np.float32)
        for peak_t in np.arange(0, t[-1], 1.0 / freq):
            idx = int(peak_t * TARGET_SR)
            if idx < win_samples:
                # R-peak: sharp gaussian
                window = np.arange(max(0, idx - 5), min(win_samples, idx + 6))
                ecg[window] += np.exp(-0.5 * ((window - idx) / 1.5) ** 2)
        ecg += rng.normal(0, 0.05, win_samples).astype(np.float32)

        # 단순 합성 ABP: 동기화된 펄스파
        abp = np.zeros(win_samples, dtype=np.float32)
        for peak_t in np.arange(0, t[-1], 1.0 / freq):
            idx = int(peak_t * TARGET_SR) + 3  # 약간의 delay
            if idx < win_samples:
                # Systolic wave
                window = np.arange(max(0, idx - 8), min(win_samples, idx + 15))
                abp[window] += 40 * np.exp(-0.5 * ((window - idx) / 4.0) ** 2)
        abp += 80  # diastolic baseline ~80mmHg
        abp += rng.normal(0, 1.0, win_samples).astype(np.float32)

        samples.append(CrossModalSample(ecg=ecg, abp=abp, case_id=9000 + i))

    return samples


# ── PackedBatch 구성 ──────────────────────────────────────────


def build_packed_batch(
    sample: CrossModalSample,
    patch_size: int = 100,
) -> tuple[PackedBatch, torch.Tensor]:
    """단일 CrossModalSample에서 any_variate PackedBatch를 구성한다.

    ECG와 ABP를 하나의 row에 이어붙인다 (collate_mode="any_variate" 패턴).
    ABP variate의 위치를 마스킹하기 위한 abp_variate_mask도 반환한다.

    Returns
    -------
    (batch, abp_patch_mask)
    - batch: PackedBatch with ECG (variate_id=1) + ABP (variate_id=2)
    - abp_patch_mask: (1, N) bool — ABP 패치 위치가 True
    """
    ecg = torch.tensor(sample.ecg, dtype=torch.float32)
    abp = torch.tensor(sample.abp, dtype=torch.float32)

    # 패치 정렬: 각 variate를 patch_size의 배수로 패딩
    def align(t: torch.Tensor) -> torch.Tensor:
        rem = len(t) % patch_size
        if rem > 0:
            pad = patch_size - rem
            t = torch.cat([t, torch.zeros(pad)])
        return t

    ecg_aligned = align(ecg)
    abp_aligned = align(abp)

    n_ecg = len(ecg_aligned)
    n_abp = len(abp_aligned)
    total_len = n_ecg + n_abp

    # values: ECG 이어붙이기 ABP
    values = torch.cat([ecg_aligned, abp_aligned]).unsqueeze(0)  # (1, L)

    # sample_id: 전부 같은 sample (=1)
    sample_id = torch.ones(1, total_len, dtype=torch.long)

    # variate_id: ECG=1, ABP=2
    variate_id = torch.cat([
        torch.full((n_ecg,), 1, dtype=torch.long),
        torch.full((n_abp,), 2, dtype=torch.long),
    ]).unsqueeze(0)  # (1, L)

    # signal_types: per-variate (2 variates)
    signal_types = torch.tensor([ECG_SIGNAL_TYPE, ABP_SIGNAL_TYPE], dtype=torch.long)

    # spatial_ids: per-variate global spatial IDs
    ecg_spatial = get_global_spatial_id(ECG_SIGNAL_TYPE, 1)  # Lead_II
    abp_spatial = get_global_spatial_id(ABP_SIGNAL_TYPE, 1)  # Radial
    spatial_ids = torch.tensor([ecg_spatial, abp_spatial], dtype=torch.long)

    # lengths, padded_lengths
    lengths = torch.tensor([len(ecg), len(abp)], dtype=torch.long)
    padded_lengths = torch.tensor([n_ecg, n_abp], dtype=torch.long)

    batch = PackedBatch(
        values=values,
        sample_id=sample_id,
        variate_id=variate_id,
        lengths=lengths,
        sampling_rates=torch.tensor([TARGET_SR, TARGET_SR]),
        signal_types=signal_types,
        spatial_ids=spatial_ids,
        padded_lengths=padded_lengths,
    )

    # ABP 패치 마스크: patch-level (1, N)
    n_patches_ecg = n_ecg // patch_size
    n_patches_abp = n_abp // patch_size
    n_patches_total = n_patches_ecg + n_patches_abp

    abp_patch_mask = torch.zeros(1, n_patches_total, dtype=torch.bool)
    abp_patch_mask[0, n_patches_ecg:] = True

    return batch, abp_patch_mask


# ── 평가 로직 ────────────────────────────────────────────────


def evaluate_cross_modal(
    wrapper,
    samples: list[CrossModalSample],
    patch_size: int = 100,
    output_dir: Path | None = None,
    max_plots: int = 5,
) -> dict[str, float]:
    """Cross-modal ECG→ABP 평가를 실행한다.

    Parameters
    ----------
    wrapper:
        DownstreamModelWrapper (또는 forward_masked를 가진 객체).
    samples:
        CrossModalSample 리스트.
    patch_size:
        패치 크기 (wrapper.patch_size 사용 권장).
    output_dir:
        시각화 저장 디렉토리. None이면 시각화 생략.
    max_plots:
        저장할 최대 시각화 수.

    Returns
    -------
    dict with keys: mse, mae, pearson_r, n_samples.
    """
    from eval._metrics import (
        compute_mae,
        compute_mse,
        compute_pearson_r,
        plot_reconstruction,
    )

    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    per_sample_metrics: list[dict] = []

    for idx, sample in enumerate(samples):
        batch, abp_patch_mask = build_packed_batch(sample, patch_size=patch_size)

        # Forward: cross_pred에서 ABP 위치 추출
        out = wrapper.forward_masked(batch)

        cross_pred = out["cross_pred"]  # (1, N, patch_size)
        loc = out["loc"]               # (1, L, 1)
        scale = out["scale"]           # (1, L, 1)

        # ABP 패치만 추출
        abp_pred_patches = cross_pred[abp_patch_mask]  # (N_abp, patch_size)

        # Denormalize: 정규화된 cross_pred를 원본 스케일로 복원
        # ABP의 loc/scale은 variate 내 동일 → ABP 영역에서 추출
        n_ecg_samples = (batch.variate_id[0] == 1).sum().item()
        abp_loc = loc[0, n_ecg_samples, 0].item()    # scalar
        abp_scale = scale[0, n_ecg_samples, 0].item()  # scalar

        abp_pred_raw = abp_pred_patches.cpu().numpy().flatten()
        abp_pred_denorm = abp_pred_raw * abp_scale + abp_loc

        # Ground truth: 원본 ABP (이미 원본 스케일)
        abp_true = sample.abp[:len(abp_pred_denorm)]

        all_pred.append(abp_pred_denorm)
        all_true.append(abp_true)

        # Per-sample metrics
        mse = compute_mse(abp_true, abp_pred_denorm)
        mae = compute_mae(abp_true, abp_pred_denorm)
        r = compute_pearson_r(abp_true, abp_pred_denorm)
        per_sample_metrics.append({"mse": mse, "mae": mae, "pearson_r": r})

        # 시각화 (처음 max_plots개)
        if output_dir is not None and idx < max_plots:
            plot_reconstruction(
                original=abp_true,
                reconstructed=abp_pred_denorm,
                save_path=output_dir / f"cross_modal_case{sample.case_id}_win{idx}.png",
                title=f"ECG→ABP Case {sample.case_id} (r={r:.3f})",
                sr=TARGET_SR,
            )

    if not all_pred:
        return {"mse": 0.0, "mae": 0.0, "pearson_r": 0.0, "n_samples": 0}

    # 전체 aggregate
    pred_all = np.concatenate(all_pred)
    true_all = np.concatenate(all_true)

    agg_mse = compute_mse(true_all, pred_all)
    agg_mae = compute_mae(true_all, pred_all)
    agg_r = compute_pearson_r(true_all, pred_all)

    # Per-sample mean
    mean_r = float(np.mean([m["pearson_r"] for m in per_sample_metrics]))

    return {
        "mse": agg_mse,
        "mae": agg_mae,
        "pearson_r": agg_r,
        "pearson_r_per_sample_mean": mean_r,
        "n_samples": len(samples),
    }


# ── 더미 모델 (checkpoint 없이 파이프라인 테스트용) ─────────────


class DummyWrapper:
    """Checkpoint 없이 forward_masked를 시뮬레이션한다.

    cross_pred = 0 + 작은 노이즈를 반환한다 (학습되지 않은 모델 시뮬레이션).
    """

    def __init__(self, patch_size: int = 100) -> None:
        self.patch_size = patch_size

    def forward_masked(self, batch: PackedBatch) -> dict[str, torch.Tensor]:
        values = batch.values  # (1, L)
        B, L = values.shape
        P = self.patch_size
        N = L // P

        # Scaler: per-variate mean/std
        loc = values.mean(dim=-1, keepdim=True).unsqueeze(-1)  # (B, 1, 1)
        scale = values.std(dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-8)
        loc = loc.expand(B, L, 1)
        scale = scale.expand(B, L, 1)

        # Dummy cross_pred: 약간의 노이즈
        cross_pred = torch.randn(B, N, P) * 0.1

        return {
            "cross_pred": cross_pred,
            "reconstructed": cross_pred,
            "loc": loc,
            "scale": scale,
            "patch_mask": torch.ones(B, N, dtype=torch.bool),
            "patch_sample_id": torch.ones(B, N, dtype=torch.long),
            "patch_variate_id": batch.variate_id[:, ::P][:, :N],
            "time_id": torch.arange(N).unsqueeze(0).expand(B, -1),
        }


# ── CLI 진입점 ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 4: Cross-modal ECG→ABP")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="사전학습 checkpoint 경로 (.pt)")
    parser.add_argument("--model-version", type=str, default="v1",
                        choices=["v1", "v2"], help="모델 버전")
    parser.add_argument("--n-cases", type=int, default=30,
                        help="평가할 VitalDB 케이스 수")
    parser.add_argument("--window-sec", type=float, default=30.0,
                        help="윈도우 길이 (초)")
    parser.add_argument("--output-dir", type=str, default="outputs/task4_cross_modal",
                        help="결과 저장 디렉토리")
    parser.add_argument("--max-plots", type=int, default=10,
                        help="저장할 최대 시각화 수")
    parser.add_argument("--dummy", action="store_true",
                        help="더미 모델로 파이프라인 테스트")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 준비 ──
    if args.dummy:
        print("[Task 4] Dummy mode: 합성 데이터 사용")
        samples = create_dummy_samples(n_samples=20, win_samples=int(args.window_sec * TARGET_SR))
    else:
        print(f"[Task 4] VitalDB에서 ECG+ABP 케이스 {args.n_cases}개 로드 중...")
        cases = find_ecg_abp_cases(n_cases=args.n_cases)
        print(f"  ECG+ABP 동시 존재: {len(cases)}개")

        samples = extract_paired_windows(
            cases, window_sec=args.window_sec, stride_sec=args.window_sec / 2,
        )
        print(f"  윈도우 추출: {len(samples)}개")

    if not samples:
        print("[Task 4] ERROR: 평가할 샘플이 없습니다.")
        return

    # ── 모델 로드 ──
    if args.dummy or args.checkpoint is None:
        patch_size = 100
        wrapper = DummyWrapper(patch_size=patch_size)
        print(f"[Task 4] DummyWrapper 사용 (patch_size={patch_size})")
    else:
        from downstream.common.model_wrapper import DownstreamModelWrapper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapper = DownstreamModelWrapper(
            args.checkpoint, model_version=args.model_version, device=device,
        )
        patch_size = wrapper.patch_size
        print(f"[Task 4] 모델 로드 완료: {args.model_version}, d_model={wrapper.d_model}, patch_size={patch_size}")

    # ── 평가 ──
    print(f"[Task 4] Cross-modal ECG→ABP 평가 시작 ({len(samples)} samples)...")
    metrics = evaluate_cross_modal(
        wrapper, samples,
        patch_size=patch_size,
        output_dir=output_dir,
        max_plots=args.max_plots,
    )

    # ── 결과 출력 ──
    print()
    print("=" * 60)
    print("  Task 4: Cross-modal ECG → ABP Results")
    print("=" * 60)
    print(f"  Samples evaluated : {metrics['n_samples']}")
    print(f"  MSE               : {metrics['mse']:.4f}")
    print(f"  MAE               : {metrics['mae']:.4f}")
    print(f"  Pearson r (agg)   : {metrics['pearson_r']:.4f}")
    print(f"  Pearson r (mean)  : {metrics['pearson_r_per_sample_mean']:.4f}")
    print("=" * 60)

    # 결과를 JSON으로 저장
    import json
    results_path = output_dir / "metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
