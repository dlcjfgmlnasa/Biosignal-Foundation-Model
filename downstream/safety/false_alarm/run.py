# -*- coding:utf-8 -*-
"""False Alarm Reduction — ECG Signal Quality (PhysioNet/CinC Challenge 2011).

Paper 4.2.5 Clinical Safety & Outcome. Foundation model의 masked reconstruction
error를 quality score(= false alarm likelihood)로 사용한다.
정상 품질 ECG(= true alarm 가능)는 reconstruction error가 낮고, 노이즈/artifact ECG
(= false alarm 유발)는 높다. Zero-shot (추가 학습 없음).

평가 데이터: PhysioNet/CinC Challenge 2011 (12-lead ECG Signal Quality)
- Acceptable (정상 품질) = true signal → reconstruction error 낮을 것
- Unacceptable (artifact/noise) = false alarm 유발 → reconstruction error 높을 것

사용법:
    python -m downstream.safety.false_alarm.run \
        --checkpoint best.pt \
        --data-dir datasets/processed/signal_quality \
        --lead II --device cuda
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
from data.spatial_map import get_global_spatial_id

from downstream.shared.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.shared.viz import plot_roc_curve


# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0
ECG_SIGNAL_TYPE_INT = 0  # ecg


# ── 데이터 로딩 ──────────────────────────────────────────────


def load_records(
    data_dir: str | Path,
    lead: str = "II",
) -> list[dict]:
    """manifest.json에서 레코드를 로드한다.

    Parameters
    ----------
    data_dir: processed 디렉토리 (manifest.json + .pt 파일).
    lead: 사용할 ECG lead (기본 "II").

    Returns
    -------
    list of dict: {signal: ndarray, label: int, record: str, quality_group: int}
    """
    data_path = Path(data_dir)
    manifest_path = data_path / "manifest.json"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    records = []
    for rec_info in manifest["records"]:
        anomaly_label = rec_info.get("anomaly_label", -1)
        quality_group = rec_info.get("quality_group", 0)

        if anomaly_label < 0:
            continue  # 라벨 없는 레코드 건너뜀

        # 요청한 lead의 .pt 파일 찾기
        signal = None
        for sig_info in rec_info.get("signals", []):
            sig_name = sig_info.get("signal_name", "")
            if sig_name == lead:
                pt_path = data_path / sig_info["file"]
                if pt_path.exists():
                    tensor = torch.load(pt_path, weights_only=True)
                    signal = tensor.squeeze(0).numpy()
                break

        if signal is None:
            # 요청 lead가 없으면 아무 lead나 사용
            for sig_info in rec_info.get("signals", []):
                pt_path = data_path / sig_info.get("file", "")
                if pt_path.exists():
                    tensor = torch.load(pt_path, weights_only=True)
                    signal = tensor.squeeze(0).numpy()
                    break

        if signal is None:
            continue

        records.append({
            "signal": signal,
            "label": anomaly_label,  # 0=normal(acceptable), 1=anomaly(unacceptable)
            "quality_group": quality_group,
            "record": rec_info.get("record", ""),
        })

    return records


# ── 배치 생성 ────────────────────────────────────────────────


def make_single_batch(
    signal: np.ndarray,
    idx: int,
    patch_size: int,
) -> PackedBatch:
    """단일 ECG 신호 → CI mode PackedBatch."""
    values = torch.from_numpy(signal).float()

    # 패치 정렬 패딩
    rem = len(values) % patch_size
    if rem > 0:
        values = torch.cat([values, torch.zeros(patch_size - rem)])

    L = len(values)
    values = values.unsqueeze(0)  # (1, L)

    spatial_id = get_global_spatial_id(ECG_SIGNAL_TYPE_INT, 0)

    collate = PackCollate(
        max_length=L, collate_mode="ci", patch_size=patch_size
    )
    sample = BiosignalSample(
        values=values.squeeze(0),
        length=L,
        channel_idx=0,
        recording_idx=idx,
        sampling_rate=DEFAULT_SR,
        n_channels=1,
        win_start=0,
        signal_type=ECG_SIGNAL_TYPE_INT,
        session_id=f"rec_{idx}",
        spatial_id=spatial_id,
    )
    return collate([sample])


# ── Anomaly Scoring ──────────────────────────────────────────


@torch.no_grad()
def compute_reconstruction_scores(
    model,
    records: list[dict],
    patch_size: int,
    mask_ratio: float = 0.5,
    n_trials: int = 5,
) -> list[float]:
    """각 레코드의 masked reconstruction MSE를 anomaly score로 계산한다.

    여러 번 랜덤 마스킹하여 평균 MSE를 사용한다 (안정적 scoring).
    """
    from loss.masked_mse_loss import create_patch_mask

    model.model.eval()
    scores: list[float] = []

    for i, rec in enumerate(records):
        batch = make_single_batch(rec["signal"], i, patch_size)
        out = model.forward_masked(batch)

        reconstructed = out["reconstructed"]  # (B, N, P)
        patch_mask = out["patch_mask"]  # (B, N) bool

        # 원본 패치 추출 (정규화된 값)
        normalized = (
            (batch.values.to(model.device).unsqueeze(-1) - out["loc"])
            / out["scale"].clamp(min=1e-8)
        ).squeeze(-1)
        B, L = normalized.shape
        P = patch_size
        N = L // P
        original_patches = normalized[:, :N * P].reshape(B, N, P)

        # 여러 번 랜덤 마스킹 → 평균 MSE
        trial_mses = []
        for _ in range(n_trials):
            pred_mask = create_patch_mask(patch_mask, mask_ratio=mask_ratio)
            if pred_mask.any():
                mse = (
                    (reconstructed[pred_mask] - original_patches[pred_mask]) ** 2
                ).mean().item()
            else:
                mse = 0.0
            trial_mses.append(mse)

        scores.append(float(np.mean(trial_mses)))

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{len(records)}] record={rec['record']}, "
                  f"label={'anomaly' if rec['label']==1 else 'normal'}, "
                  f"score={scores[-1]:.6f}")

    return scores


# ── 메트릭 계산 ──────────────────────────────────────────────


def _compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> dict:
    """AUROC, AUPRC, optimal threshold 계산."""
    auroc = compute_auroc(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)

    # Optimal threshold (Youden's J)
    best_thresh, best_j = 0.0, -1.0
    thresholds = np.percentile(y_score, np.linspace(1, 99, 99))
    for thresh in thresholds:
        y_pred = (y_score >= thresh).astype(int)
        ss = compute_sensitivity_specificity(y_true, y_pred)
        j = ss["sensitivity"] + ss["specificity"] - 1.0
        if j > best_j:
            best_j = j
            best_thresh = float(thresh)

    y_pred_opt = (y_score >= best_thresh).astype(int)
    ss_opt = compute_sensitivity_specificity(y_true, y_pred_opt)
    f1 = compute_f1(y_true, y_pred_opt, average="macro")

    normal_scores = y_score[y_true == 0]
    anomaly_scores = y_score[y_true == 1]

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1_macro": f1,
        "optimal_threshold": best_thresh,
        "sensitivity": ss_opt["sensitivity"],
        "specificity": ss_opt["specificity"],
        "n_total": len(y_true),
        "n_normal": int((y_true == 0).sum()),
        "n_anomaly": int(y_true.sum()),
        "score_mean_normal": float(normal_scores.mean()) if len(normal_scores) > 0 else 0.0,
        "score_std_normal": float(normal_scores.std()) if len(normal_scores) > 0 else 0.0,
        "score_mean_anomaly": float(anomaly_scores.mean()) if len(anomaly_scores) > 0 else 0.0,
        "score_std_anomaly": float(anomaly_scores.std()) if len(anomaly_scores) > 0 else 0.0,
    }


# ── 시각화 ───────────────────────────────────────────────────


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    save_path: str | Path,
    title: str = "Reconstruction Error Distribution",
) -> None:
    """Normal/Anomaly의 reconstruction MSE 분포 히스토그램."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal_scores = y_score[y_true == 0]
    anomaly_scores = y_score[y_true == 1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(min(y_score.min(), 0), y_score.max() * 1.05, 50)

    ax.hist(normal_scores, bins=bins, alpha=0.6, color="steelblue",
            label=f"Acceptable (n={len(normal_scores)})", density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.6, color="salmon",
            label=f"Unacceptable (n={len(anomaly_scores)})", density=True)
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.4f}")

    ax.set_xlabel("Reconstruction MSE (anomaly score)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── CLI 진입점 ───────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anomaly Detection — ECG Signal Quality (Reconstruction Error)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--data-dir", type=str, default="datasets/processed/signal_quality",
        help="파싱된 데이터 디렉토리 (manifest.json + .pt)",
    )
    parser.add_argument(
        "--lead", type=str, default="II",
        help="사용할 ECG lead (기본 II)",
    )
    parser.add_argument(
        "--mask-ratio", type=float, default=0.5,
        help="마스킹 비율",
    )
    parser.add_argument(
        "--n-trials", type=int, default=5,
        help="랜덤 마스킹 반복 횟수",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ──
    print(f"\nLoading data: {args.data_dir}")
    print(f"  Lead: {args.lead}")
    records = load_records(args.data_dir, args.lead)

    if not records:
        print("ERROR: No records loaded.", file=sys.stderr)
        sys.exit(1)

    n_normal = sum(1 for r in records if r["label"] == 0)
    n_anomaly = sum(1 for r in records if r["label"] == 1)
    print(f"  Total: {len(records)} (Normal={n_normal}, Anomaly={n_anomaly})")

    if n_normal == 0 or n_anomaly == 0:
        print("ERROR: Need both normal and anomaly records.", file=sys.stderr)
        sys.exit(1)

    # ── 모델 로드 ──
    from downstream.shared.model_wrapper import DownstreamModelWrapper

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
    patch_size = model.patch_size
    print(f"  d_model={model.d_model}, patch_size={patch_size}")
    print(f"  Mask ratio: {args.mask_ratio}, Trials: {args.n_trials}")

    # ── Anomaly Scoring (zero-shot) ──
    print(f"\nComputing reconstruction error ({len(records)} records)...")
    scores = compute_reconstruction_scores(
        model, records, patch_size,
        mask_ratio=args.mask_ratio,
        n_trials=args.n_trials,
    )

    # ── 메트릭 계산 ──
    y_true = np.array([r["label"] for r in records])
    y_score = np.array(scores)

    metrics = _compute_metrics(y_true, y_score)

    # ── 결과 출력 ──
    print(f"\n{'=' * 60}")
    print(f"  Anomaly Detection — ECG Signal Quality (Zero-Shot)")
    print(f"  Lead: {args.lead}")
    print(f"{'=' * 60}")
    print(f"  AUROC:              {metrics['auroc']:.4f}")
    print(f"  AUPRC:              {metrics['auprc']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"  Threshold:          {metrics['optimal_threshold']:.6f}")
    print(f"  Sensitivity:        {metrics['sensitivity']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    print(f"  Normal/Anomaly:     {metrics['n_normal']}/{metrics['n_anomaly']}")
    print(f"  Score (Normal):     {metrics['score_mean_normal']:.6f} "
          f"+/- {metrics['score_std_normal']:.6f}")
    print(f"  Score (Anomaly):    {metrics['score_mean_anomaly']:.6f} "
          f"+/- {metrics['score_std_anomaly']:.6f}")
    print(f"{'=' * 60}")

    # ── 시각화 ──
    roc_path = out_dir / "roc_signal_quality.png"
    plot_roc_curve(y_true, y_score, roc_path,
                   title=f"ECG Signal Quality — Lead {args.lead} ROC")
    print(f"\nROC curve: {roc_path}")

    dist_path = out_dir / "score_distribution.png"
    plot_score_distribution(
        y_true, y_score, metrics["optimal_threshold"],
        dist_path, title=f"Reconstruction Error — Lead {args.lead}"
    )
    print(f"Score distribution: {dist_path}")

    # ── JSON 저장 ──
    results = {
        **metrics,
        "config": {
            "mode": "reconstruction_error_zero_shot",
            "dataset": "PhysioNet-Challenge-2011",
            "lead": args.lead,
            "mask_ratio": args.mask_ratio,
            "n_trials": args.n_trials,
            "patch_size": patch_size,
        },
    }
    results_path = out_dir / "results_signal_quality.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
