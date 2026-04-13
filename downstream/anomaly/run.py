# -*- coding:utf-8 -*-
"""Task 6: 이상 탐지 (Anomaly Detection).

정상 구간의 masked reconstruction loss를 기준으로 anomaly scoring.
직관: Foundation model은 정상 패턴을 잘 재구성하므로 reconstruction error가 낮고,
이상 구간(아티팩트, 센서 탈착, 부정맥 등)은 error가 높다.

라벨 전략: quality check PASS = 정상(0), FAIL = 이상(1).
VitalDB parser가 reject한 세그먼트를 anomaly 라벨로 역이용.

Usage
-----
# 실제 모델:
python -m downstream.anomaly.run --checkpoint outputs/phase1_v1/best.pt --signal-type ecg

# 더미 테스트:
python -m downstream.anomaly.run --dummy --signal-type ecg
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from data.collate import PackedBatch
from data.spatial_map import get_global_spatial_id


# ── 상수 ──────────────────────────────────────────────────────

TARGET_SR = 100.0

SIGNAL_TYPE_INDEX: dict[str, int] = {
    "ecg": 0,
    "abp": 1,
    "ppg": 2,
    "cvp": 3,
    "co2": 4,
    "awp": 5,
    "pap": 6,
    "icp": 7,
}


# ── 데이터 구조 ───────────────────────────────────────────────


@dataclass
class AnomalyWindow:
    """이상 탐지용 윈도우."""

    signal: np.ndarray  # (win_samples,) at 100Hz
    signal_type: str
    case_id: int
    label: int  # 0=normal, 1=anomaly
    quality_score: dict  # segment_quality_score 결과


# ── 데이터 준비 ───────────────────────────────────────────────


def prepare_anomaly_windows(
    n_cases: int = 30,
    signal_type: str = "ecg",
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
) -> list[AnomalyWindow]:
    """정상/이상 윈도우를 추출한다.

    quality check PASS → label=0 (정상), FAIL → label=1 (이상).
    """
    from downstream.data_utils import (
        extract_windows,
        load_pilot_cases,
    )
    from data.parser._common import segment_quality_score, domain_quality_check

    cases = load_pilot_cases(n_cases=n_cases, signal_types=[signal_type])

    results: list[AnomalyWindow] = []

    for case in cases:
        # quality_check=False로 모든 윈도우 추출 (FAIL 포함)
        windows = extract_windows(
            case,
            signal_type=signal_type,
            window_sec=window_sec,
            stride_sec=stride_sec,
            quality_check=False,
        )

        for w in windows:
            # 기본 품질 점수
            basic = segment_quality_score(w.signal)
            domain = domain_quality_check(signal_type, w.signal, sr=TARGET_SR)

            # 라벨: 기본 AND 도메인 모두 통과해야 정상
            is_normal = basic["pass"] and domain["pass"]
            label = 0 if is_normal else 1

            results.append(
                AnomalyWindow(
                    signal=w.signal,
                    signal_type=signal_type,
                    case_id=w.case_id,
                    label=label,
                    quality_score={**basic, "domain_pass": domain["pass"]},
                )
            )

    return results


def create_dummy_windows(
    n_normal: int = 40,
    n_anomaly: int = 20,
    win_samples: int = 1000,
    signal_type: str = "ecg",
) -> list[AnomalyWindow]:
    """더미 정상/이상 윈도우를 생성한다."""
    rng = np.random.default_rng(42)
    results: list[AnomalyWindow] = []

    # 정상: 깨끗한 사인파 + 약간의 노이즈
    for i in range(n_normal):
        freq = rng.uniform(0.8, 1.5)
        t = np.arange(win_samples) / TARGET_SR
        sig = np.sin(2 * np.pi * freq * t).astype(np.float32)
        sig += rng.normal(0, 0.05, win_samples).astype(np.float32)
        results.append(
            AnomalyWindow(
                signal=sig,
                signal_type=signal_type,
                case_id=8000 + i,
                label=0,
                quality_score={"pass": True, "flatline_ratio": 0.0},
            )
        )

    # 이상: 스파이크, flatline, 랜덤 노이즈
    for i in range(n_anomaly):
        anomaly_type = i % 3
        if anomaly_type == 0:
            # Spike artifact
            sig = rng.normal(0, 0.1, win_samples).astype(np.float32)
            spike_pos = rng.integers(100, win_samples - 100, size=5)
            sig[spike_pos] = rng.uniform(3, 8, size=5).astype(np.float32)
        elif anomaly_type == 1:
            # Flatline
            sig = np.full(win_samples, 0.5, dtype=np.float32)
            sig += rng.normal(0, 0.001, win_samples).astype(np.float32)
        else:
            # Pure random noise
            sig = rng.normal(0, 1.0, win_samples).astype(np.float32)

        results.append(
            AnomalyWindow(
                signal=sig,
                signal_type=signal_type,
                case_id=9000 + i,
                label=1,
                quality_score={
                    "pass": False,
                    "flatline_ratio": 0.5 if anomaly_type == 1 else 0.0,
                },
            )
        )

    return results


# ── PackedBatch 구성 (CI mode, 단일 variate) ────────────────


def build_ci_batch(
    window: AnomalyWindow,
    patch_size: int = 100,
) -> PackedBatch:
    """단일 윈도우에서 CI mode PackedBatch를 구성한다."""
    sig = torch.tensor(window.signal, dtype=torch.float32)

    # 패치 정렬 패딩
    rem = len(sig) % patch_size
    if rem > 0:
        sig = torch.cat([sig, torch.zeros(patch_size - rem)])

    L = len(sig)
    values = sig.unsqueeze(0)  # (1, L)
    sample_id = torch.ones(1, L, dtype=torch.long)
    variate_id = torch.ones(1, L, dtype=torch.long)

    stype_idx = SIGNAL_TYPE_INDEX.get(window.signal_type, 0)
    signal_types = torch.tensor([stype_idx], dtype=torch.long)
    spatial_id = get_global_spatial_id(stype_idx, 0)  # Unknown
    spatial_ids = torch.tensor([spatial_id], dtype=torch.long)

    return PackedBatch(
        values=values,
        sample_id=sample_id,
        variate_id=variate_id,
        lengths=torch.tensor([len(window.signal)], dtype=torch.long),
        sampling_rates=torch.tensor([TARGET_SR]),
        signal_types=signal_types,
        spatial_ids=spatial_ids,
        padded_lengths=torch.tensor([L], dtype=torch.long),
    )


# ── Anomaly Scoring ──────────────────────────────────────────


def compute_anomaly_scores(
    wrapper,
    windows: list[AnomalyWindow],
    patch_size: int = 100,
    mask_ratio: float = 0.15,
) -> list[float]:
    """각 윈도우의 reconstruction MSE를 anomaly score로 계산한다.

    mask_ratio 비율로 랜덤 패치를 마스킹 → reconstruction → 마스킹 위치 MSE.
    """
    from loss.masked_mse_loss import create_patch_mask

    scores: list[float] = []

    for window in windows:
        batch = build_ci_batch(window, patch_size=patch_size)
        out = wrapper.forward_masked(batch)

        reconstructed = out["reconstructed"]  # (1, N, P)
        patch_mask = out["patch_mask"]  # (1, N) bool

        # 원본 패치 추출 (정규화 후)
        normalized = (
            (batch.values.unsqueeze(-1) - out["loc"]) / out["scale"].clamp(min=1e-8)
        ).squeeze(-1)  # (1, L)
        B, L = normalized.shape
        P = patch_size
        N = L // P
        original_patches = normalized.reshape(B, N, P)  # (1, N, P)

        # 마스킹 생성
        pred_mask = create_patch_mask(patch_mask, mask_ratio=mask_ratio)

        # 마스킹 위치의 MSE
        if pred_mask.any():
            mse = (
                ((reconstructed[pred_mask] - original_patches[pred_mask]) ** 2)
                .mean()
                .item()
            )
        else:
            mse = 0.0

        scores.append(mse)

    return scores


def find_optimal_threshold(
    scores: list[float],
    labels: list[int],
) -> tuple[float, float]:
    """F1을 최대화하는 threshold를 찾는다.

    Returns
    -------
    (best_threshold, best_f1)
    """
    from downstream.metrics import compute_f1

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # 후보 threshold: 정상 분포의 percentile 기반
    normal_scores = scores_arr[labels_arr == 0]
    if len(normal_scores) == 0:
        return 0.0, 0.0

    candidates = np.percentile(normal_scores, [80, 85, 90, 95, 97.5, 99])
    # score 분포의 균등 분할도 추가
    candidates = np.concatenate(
        [
            candidates,
            np.linspace(scores_arr.min(), scores_arr.max(), 20),
        ]
    )

    best_f1 = 0.0
    best_thresh = float(candidates[0])

    for thresh in candidates:
        preds = (scores_arr >= thresh).astype(int)
        f1 = compute_f1(labels_arr, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)

    return best_thresh, best_f1


# ── 시각화 ────────────────────────────────────────────────────


def plot_score_distribution(
    scores: list[float],
    labels: list[int],
    threshold: float,
    save_path: str | Path,
    title: str = "Anomaly Score Distribution",
) -> None:
    """정상/이상 reconstruction MSE 분포 히스토그램을 저장한다."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    normal_scores = scores_arr[labels_arr == 0]
    anomaly_scores = scores_arr[labels_arr == 1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    bins = np.linspace(
        min(scores_arr.min(), 0),
        scores_arr.max() * 1.05,
        50,
    )

    ax.hist(
        normal_scores,
        bins=bins,
        alpha=0.6,
        color="steelblue",
        label=f"Normal (n={len(normal_scores)})",
        density=True,
    )
    ax.hist(
        anomaly_scores,
        bins=bins,
        alpha=0.6,
        color="salmon",
        label=f"Anomaly (n={len(anomaly_scores)})",
        density=True,
    )
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {threshold:.4f}",
    )

    ax.set_xlabel("Reconstruction MSE (anomaly score)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 더미 모델 ────────────────────────────────────────────────


class DummyWrapper:
    """Checkpoint 없이 reconstruction을 시뮬레이션한다.

    정상 입력에는 낮은 노이즈, 비정상 입력에는 높은 노이즈를 반환하여
    파이프라인이 의미 있는 AUROC를 보이도록 한다.
    (실제 모델과 달리 입력을 직접 보고 판단하는 oracle — 파이프라인 검증 전용)
    """

    def __init__(self, patch_size: int = 100) -> None:
        self.patch_size = patch_size

    def forward_masked(self, batch: PackedBatch) -> dict[str, torch.Tensor]:
        values = batch.values  # (1, L)
        B, L = values.shape
        P = self.patch_size
        N = L // P

        # Per-variate normalization (scaler 시뮬레이션)
        loc = values.mean(dim=-1, keepdim=True).unsqueeze(-1).expand(B, L, 1)
        scale = (
            values.std(dim=-1, keepdim=True)
            .unsqueeze(-1)
            .clamp(min=1e-8)
            .expand(B, L, 1)
        )

        # Normalized → patches
        normalized = ((values.unsqueeze(-1) - loc) / scale).squeeze(-1)
        original_patches = normalized.reshape(B, N, P)

        # Reconstruction = original + noise (시뮬레이션)
        # 입력의 분산이 높으면 (비정상) 더 큰 노이즈 추가
        input_var = values.var().item()
        noise_scale = 0.05 + 0.1 * min(input_var, 5.0)
        reconstructed = (
            original_patches + torch.randn_like(original_patches) * noise_scale
        )

        return {
            "reconstructed": reconstructed,
            "cross_pred": reconstructed,
            "loc": loc,
            "scale": scale,
            "patch_mask": torch.ones(B, N, dtype=torch.bool),
            "patch_sample_id": torch.ones(B, N, dtype=torch.long),
            "patch_variate_id": torch.ones(B, N, dtype=torch.long),
            "time_id": torch.arange(N).unsqueeze(0).expand(B, -1),
        }


# ── CLI 진입점 ────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Task 6: Anomaly Detection")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="사전학습 checkpoint 경로 (.pt)"
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="v1",
        choices=["v1", "v2"],
        help="모델 버전",
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        default="ecg",
        choices=["ecg", "abp", "ppg", "cvp", "co2", "awp"],
        help="평가할 신호 타입",
    )
    parser.add_argument(
        "--n-cases", type=int, default=30, help="로드할 VitalDB 케이스 수"
    )
    parser.add_argument(
        "--window-sec", type=float, default=10.0, help="윈도우 길이 (초)"
    )
    parser.add_argument("--mask-ratio", type=float, default=0.15, help="마스킹 비율")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/task6_anomaly",
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--dummy", action="store_true", help="더미 모델로 파이프라인 테스트"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 준비 ──
    if args.dummy:
        print(f"[Task 6] Dummy mode: {args.signal_type} Synthetic data")
        windows = create_dummy_windows(
            n_normal=40,
            n_anomaly=20,
            win_samples=int(args.window_sec * TARGET_SR),
            signal_type=args.signal_type,
        )
    else:
        print(f"[Task 6] VitalDB {args.signal_type} 윈도우 추출 중...")
        windows = prepare_anomaly_windows(
            n_cases=args.n_cases,
            signal_type=args.signal_type,
            window_sec=args.window_sec,
            stride_sec=args.window_sec / 2,
        )

    labels = [w.label for w in windows]
    n_normal = labels.count(0)
    n_anomaly = labels.count(1)
    print(f"  Normal: {n_normal}, Anomaly: {n_anomaly}, Total: {len(windows)}")

    if n_normal == 0 or n_anomaly == 0:
        print("[Task 6] ERROR: 정상 또는 이상 윈도우가 없습니다. 평가 불가.")
        return

    # ── 모델 로드 ──
    if args.dummy or args.checkpoint is None:
        patch_size = 100
        wrapper = DummyWrapper(patch_size=patch_size)
        print(f"[Task 6] DummyWrapper 사용 (patch_size={patch_size})")
    else:
        from downstream.model_wrapper import DownstreamModelWrapper

        device = "cuda" if torch.cuda.is_available() else "cpu"
        wrapper = DownstreamModelWrapper(
            args.checkpoint,
            model_version=args.model_version,
            device=device,
        )
        patch_size = wrapper.patch_size
        print(f"[Task 6] 모델 로드: {args.model_version}, patch_size={patch_size}")

    # ── Anomaly Scoring ──
    print(f"[Task 6] Computing anomaly scores ({len(windows)} windows)...")
    scores = compute_anomaly_scores(
        wrapper,
        windows,
        patch_size=patch_size,
        mask_ratio=args.mask_ratio,
    )

    # ── 메트릭 계산 ──
    from downstream.metrics import compute_auroc, compute_auprc

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    auroc = compute_auroc(labels_arr, scores_arr)
    auprc = compute_auprc(labels_arr, scores_arr)

    # Optimal threshold
    best_thresh, best_f1 = find_optimal_threshold(scores, labels)
    preds_at_thresh = (scores_arr >= best_thresh).astype(int)

    # 정상 분포 통계
    normal_scores = scores_arr[labels_arr == 0]
    anomaly_scores_arr = scores_arr[labels_arr == 1]

    metrics = {
        "signal_type": args.signal_type,
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "auroc": float(auroc) if isinstance(auroc, float) else auroc,
        "auprc": float(auprc) if isinstance(auprc, float) else auprc,
        "f1_at_optimal": best_f1,
        "optimal_threshold": best_thresh,
        "normal_score_mean": float(normal_scores.mean()),
        "normal_score_std": float(normal_scores.std()),
        "anomaly_score_mean": float(anomaly_scores_arr.mean()),
        "anomaly_score_std": float(anomaly_scores_arr.std()),
    }

    # ── 시각화 ──
    from downstream.viz import plot_roc_curve

    plot_score_distribution(
        scores,
        labels,
        best_thresh,
        save_path=output_dir / f"score_dist_{args.signal_type}.png",
        title=f"Anomaly Score Distribution — {args.signal_type.upper()}",
    )
    plot_roc_curve(
        labels_arr,
        scores_arr,
        save_path=output_dir / f"roc_{args.signal_type}.png",
        title=f"Anomaly Detection ROC — {args.signal_type.upper()}",
    )

    # ── 결과 출력 ──
    print()
    print("=" * 60)
    print(f"  Task 6: Anomaly Detection - {args.signal_type.upper()}")
    print("=" * 60)
    print(f"  Normal / Anomaly   : {n_normal} / {n_anomaly}")
    print(f"  AUROC              : {metrics['auroc']:.4f}")
    print(f"  AUPRC              : {metrics['auprc']:.4f}")
    print(f"  F1@optimal         : {metrics['f1_at_optimal']:.4f}")
    print(f"  Optimal threshold  : {metrics['optimal_threshold']:.6f}")
    print(
        f"  Normal MSE (mean)  : {metrics['normal_score_mean']:.6f} +/- {metrics['normal_score_std']:.6f}"
    )
    print(
        f"  Anomaly MSE (mean) : {metrics['anomaly_score_mean']:.6f} +/- {metrics['anomaly_score_std']:.6f}"
    )
    print("=" * 60)

    # JSON 저장
    results_path = output_dir / f"metrics_{args.signal_type}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
