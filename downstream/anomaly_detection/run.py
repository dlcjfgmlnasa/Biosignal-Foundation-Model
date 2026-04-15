# -*- coding:utf-8 -*-
"""Anomaly Detection — Reconstruction Error 기반 이상 탐지.

Foundation model의 masked reconstruction error를 anomaly score로 사용한다.
정상 신호는 reconstruction error가 낮고, 이상 신호는 높다.
학습 없이(zero-shot) pretrained model만으로 이상 탐지를 수행한다.

평가 데이터: PhysioNet/CinC Challenge 2015 (ICU False Alarm)
- True alarm (실제 부정맥) = anomaly → reconstruction error 높을 것
- False alarm (artifact 등) = 정상 변이 → reconstruction error 낮을 것

사용법:
    python -m downstream.anomaly_detection.run \
        --checkpoint best.pt \
        --data-path datasets/processed/anomaly_detection/false_alarm_ecg_w10s.pt \
        --device cuda

    python -m downstream.anomaly_detection.run \
        --checkpoint best.pt \
        --data-dir datasets/processed/anomaly_detection \
        --input-signals ecg ppg abp --window-sec 10
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
from data.parser.vitaldb import SIGNAL_TYPES

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve


# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0

ALARM_TYPES = [
    "asystole",
    "extreme_bradycardia",
    "extreme_tachycardia",
    "ventricular_tachycardia",
    "ventricular_flutter_fib",
]

SIGNAL_NAME_TO_TYPE: dict[str, str] = {
    "II": "ecg", "I": "ecg", "III": "ecg",
    "V": "ecg", "V1": "ecg", "V2": "ecg", "V5": "ecg",
    "aVR": "ecg", "aVL": "ecg", "aVF": "ecg",
    "MCL": "ecg", "MCL1": "ecg",
    "ABP": "abp", "ART": "abp", "AOBP": "abp",
    "PLETH": "ppg",
}


# ── 데이터 로딩 ──────────────────────────────────────────────


def load_records(
    data_dir: str | Path,
    input_signals: list[str],
    window_sec: float = 10.0,
) -> list[dict]:
    """manifest.json에서 레코드를 로드한다."""
    data_path = Path(data_dir)
    manifest_path = data_path / "manifest.json"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    win_samples = int(window_sec * DEFAULT_SR)
    records = []

    for rec_info in manifest["records"]:
        label = 1 if rec_info["label"] else 0
        alarm_type = rec_info.get("alarm_type", "unknown")

        signals: dict[str, np.ndarray] = {}
        type_loaded: set[str] = set()

        for sig_info in rec_info.get("signals", []):
            sig_type = sig_info.get("signal_type")
            if sig_type is None:
                sig_name = sig_info.get("signal_name", "")
                sig_type = SIGNAL_NAME_TO_TYPE.get(sig_name)
            if sig_type is None or sig_type not in input_signals:
                continue
            if sig_type in type_loaded:
                continue

            pt_path = data_path / sig_info.get("file", "")
            if not pt_path.exists():
                continue

            tensor = torch.load(pt_path, weights_only=True)
            signal = tensor.squeeze(0).numpy()

            if len(signal) > win_samples:
                signal = signal[-win_samples:]

            signals[sig_type] = signal
            type_loaded.add(sig_type)

        if not signals:
            continue

        min_len = min(len(s) for s in signals.values())
        signals = {k: v[-min_len:] for k, v in signals.items()}

        records.append({
            "signals": signals,
            "label": label,
            "alarm_type": alarm_type,
            "record": rec_info.get("record", ""),
        })

    return records


def load_from_pt(data_path: str) -> list[dict]:
    """prepare_data.py로 생성한 .pt에서 전체 레코드를 로드한다.

    Anomaly detection은 train/test split이 불필요하므로 train+test를 합친다.
    """
    data = torch.load(data_path, weights_only=False)
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    print(f"  Input signals: {meta.get('input_signals', '?')}")
    print(f"  Window: {meta.get('window_sec', '?')}s")

    records = []
    for split_name in ["train", "test"]:
        split_data = data.get(split_name, {})
        labels = split_data.get("labels", torch.tensor([]))
        alarm_types = split_data.get("alarm_types", ["unknown"] * len(labels))
        rec_names = split_data.get("records", [f"r{i}" for i in range(len(labels))])
        input_keys = list(split_data.get("signals", {}).keys())

        for i in range(len(labels)):
            signals = {
                k: split_data["signals"][k][i].numpy()
                for k in input_keys
                if k in split_data["signals"]
            }
            records.append({
                "signals": signals,
                "label": int(labels[i].item()),
                "alarm_type": alarm_types[i],
                "record": rec_names[i],
            })

    return records


# ── 배치 생성 ────────────────────────────────────────────────


def _record_to_samples(rec: dict, idx: int) -> list[BiosignalSample]:
    samples = []
    for ch, (sig_type, signal) in enumerate(rec["signals"].items()):
        stype_int = SIGNAL_TYPES.get(sig_type, 0)
        spatial_id = get_global_spatial_id(stype_int, 0)
        samples.append(
            BiosignalSample(
                values=torch.from_numpy(signal).float(),
                length=len(signal),
                channel_idx=ch,
                recording_idx=idx,
                sampling_rate=DEFAULT_SR,
                n_channels=len(rec["signals"]),
                win_start=0,
                signal_type=stype_int,
                session_id=f"rec_{rec['record']}",
                spatial_id=spatial_id,
            )
        )
    return samples


def make_single_batch(
    rec: dict,
    idx: int,
    patch_size: int,
    max_length: int,
) -> PackedBatch:
    """단일 레코드 → PackedBatch."""
    multi = len(rec["signals"]) > 1
    collate_mode = "any_variate" if multi else "ci"
    collate = PackCollate(
        max_length=max_length, collate_mode=collate_mode, patch_size=patch_size
    )
    samples = _record_to_samples(rec, idx)
    return collate(samples)


# ── Anomaly Scoring (Reconstruction Error) ───────────────────


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

    Parameters
    ----------
    model: DownstreamModelWrapper (frozen, eval mode).
    records: 레코드 리스트.
    patch_size: 패치 크기.
    mask_ratio: 마스킹 비율 (0.5 = 50% 패치를 마스킹).
    n_trials: 랜덤 마스킹 반복 횟수 (평균).

    Returns
    -------
    각 레코드의 평균 reconstruction MSE (anomaly score).
    """
    from loss.masked_mse_loss import create_patch_mask

    model.model.eval()
    scores: list[float] = []
    first_sig = next(iter(records[0]["signals"].values()))
    max_length = len(first_sig)

    for i, rec in enumerate(records):
        batch = make_single_batch(rec, i, patch_size, max_length)
        out = model.forward_masked(batch)

        reconstructed = out["reconstructed"]  # (B, N, P)
        patch_mask = out["patch_mask"]  # (B, N) bool

        # 원본 패치 추출 (정규화된 값)
        normalized = (
            (batch.values.to(model.device).unsqueeze(-1) - out["loc"])
            / out["scale"].clamp(min=1e-8)
        ).squeeze(-1)  # (B, L)
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
            print(f"  [{i + 1}/{len(records)}] score={scores[-1]:.6f}")

    return scores


# ── 메트릭 계산 ──────────────────────────────────────────────


def _compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alarm_types: list[str],
) -> dict:
    """전체 + 알람 타입별 메트릭을 계산한다.

    anomaly score가 높을수록 true alarm(anomaly)일 가능성이 높다고 가정.
    """
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

    result = {
        "auroc": auroc,
        "auprc": auprc,
        "f1_macro": f1,
        "optimal_threshold": best_thresh,
        "sensitivity": ss_opt["sensitivity"],
        "specificity": ss_opt["specificity"],
        "n_total": len(y_true),
        "n_anomaly": int(y_true.sum()),
        "n_normal": int((y_true == 0).sum()),
        "score_mean_normal": float(y_score[y_true == 0].mean()) if (y_true == 0).any() else 0.0,
        "score_std_normal": float(y_score[y_true == 0].std()) if (y_true == 0).any() else 0.0,
        "score_mean_anomaly": float(y_score[y_true == 1].mean()) if (y_true == 1).any() else 0.0,
        "score_std_anomaly": float(y_score[y_true == 1].std()) if (y_true == 1).any() else 0.0,
        "y_true": y_true,
        "y_score": y_score,
    }

    # 알람 타입별 메트릭
    alarm_arr = np.array(alarm_types)
    per_alarm = {}
    for atype in ALARM_TYPES:
        mask = alarm_arr == atype
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        ys = y_score[mask]
        n_true = int(yt.sum())
        n_false = int((yt == 0).sum())

        if len(np.unique(yt)) < 2:
            per_alarm[atype] = {"n": int(mask.sum()), "n_true": n_true, "auroc": 0.0}
            continue

        per_alarm[atype] = {
            "n": int(mask.sum()),
            "n_true": n_true,
            "n_false": n_false,
            "auroc": compute_auroc(yt, ys),
            "score_mean_true": float(ys[yt == 1].mean()),
            "score_mean_false": float(ys[yt == 0].mean()),
        }
    result["per_alarm"] = per_alarm

    return result


# ── 시각화 ───────────────────────────────────────────────────


def plot_score_distribution(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
    save_path: str | Path,
    title: str = "Anomaly Score Distribution",
) -> None:
    """True/False alarm의 reconstruction MSE 분포 히스토그램."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normal_scores = y_score[y_true == 0]
    anomaly_scores = y_score[y_true == 1]

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.linspace(
        min(y_score.min(), 0), y_score.max() * 1.05, 50
    )

    ax.hist(normal_scores, bins=bins, alpha=0.6, color="steelblue",
            label=f"False Alarm (n={len(normal_scores)})", density=True)
    ax.hist(anomaly_scores, bins=bins, alpha=0.6, color="salmon",
            label=f"True Alarm (n={len(anomaly_scores)})", density=True)
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
        description="Anomaly Detection — Reconstruction Error Scoring"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="prepare_data.py로 생성한 .pt 파일 경로",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="파싱된 데이터 디렉토리 (manifest.json + .pt)",
    )
    parser.add_argument(
        "--input-signals", nargs="+", default=["ecg", "ppg", "abp"],
        choices=["ecg", "ppg", "abp"],
    )
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument(
        "--mask-ratio", type=float, default=0.5,
        help="마스킹 비율 (높을수록 더 많은 패치를 마스킹하여 복원 난이도 증가)",
    )
    parser.add_argument(
        "--n-trials", type=int, default=5,
        help="랜덤 마스킹 반복 횟수 (평균으로 안정적 scoring)",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 데이터 로드 ──
    if args.data_path and Path(args.data_path).exists():
        print(f"\nLoading prepared data: {args.data_path}")
        records = load_from_pt(args.data_path)
    elif args.data_dir:
        print(f"\nLoading data: {args.data_dir}")
        print(f"  Input signals: {args.input_signals}")
        print(f"  Window: {args.window_sec}s")
        records = load_records(args.data_dir, args.input_signals, args.window_sec)
    else:
        print("ERROR: --data-path or --data-dir required.", file=sys.stderr)
        sys.exit(1)

    if not records:
        print("ERROR: No records loaded.", file=sys.stderr)
        sys.exit(1)

    n_true = sum(1 for r in records if r["label"] == 1)
    n_false = len(records) - n_true
    print(f"  Total: {len(records)} (True alarm={n_true}, False alarm={n_false})")

    # ── 모델 로드 ──
    from downstream.model_wrapper import DownstreamModelWrapper

    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
    patch_size = model.patch_size
    print(f"  d_model={model.d_model}, patch_size={patch_size}")

    sig_str = " + ".join(s.upper() for s in args.input_signals)
    print(f"  Input: {sig_str} | Window: {args.window_sec}s")
    print(f"  Mask ratio: {args.mask_ratio}, Trials: {args.n_trials}")

    # ── Anomaly Scoring (zero-shot, 학습 없음) ──
    print(f"\nComputing reconstruction error scores ({len(records)} records)...")
    scores = compute_reconstruction_scores(
        model, records, patch_size,
        mask_ratio=args.mask_ratio,
        n_trials=args.n_trials,
    )

    # ── 메트릭 계산 ──
    y_true = np.array([r["label"] for r in records])
    y_score = np.array(scores)
    alarm_types = [r["alarm_type"] for r in records]

    metrics = _compute_all_metrics(y_true, y_score, alarm_types)
    y_true_out = metrics.pop("y_true")
    y_score_out = metrics.pop("y_score")

    # ── 결과 출력 ──
    print(f"\n{'=' * 60}")
    print(f"  Anomaly Detection — Reconstruction Error (Zero-Shot)")
    print(f"{'=' * 60}")
    print(f"  AUROC:              {metrics['auroc']:.4f}")
    print(f"  AUPRC:              {metrics['auprc']:.4f}")
    print(f"  F1 (macro):         {metrics['f1_macro']:.4f}")
    print(f"  Threshold:          {metrics['optimal_threshold']:.6f}")
    print(f"  Sensitivity:        {metrics['sensitivity']:.4f}")
    print(f"  Specificity:        {metrics['specificity']:.4f}")
    print(f"  True/False alarm:   {metrics['n_anomaly']}/{metrics['n_normal']}")
    print(f"  Score (True alarm):  {metrics['score_mean_anomaly']:.6f} "
          f"+/- {metrics['score_std_anomaly']:.6f}")
    print(f"  Score (False alarm): {metrics['score_mean_normal']:.6f} "
          f"+/- {metrics['score_std_normal']:.6f}")

    per_alarm = metrics.get("per_alarm", {})
    if per_alarm:
        print(f"\n  {'Alarm Type':<30s}  {'N':>4s}  {'AUROC':>6s}  "
              f"{'True MSE':>10s}  {'False MSE':>10s}")
        print(f"  {'-' * 70}")
        for atype in ALARM_TYPES:
            if atype not in per_alarm:
                continue
            pa = per_alarm[atype]
            auroc_s = f"{pa['auroc']:.3f}" if pa.get("auroc", 0) > 0 else "  N/A"
            true_s = f"{pa.get('score_mean_true', 0):.6f}" if "score_mean_true" in pa else "       N/A"
            false_s = f"{pa.get('score_mean_false', 0):.6f}" if "score_mean_false" in pa else "       N/A"
            print(f"  {atype:<30s}  {pa['n']:>4d}  {auroc_s:>6s}  "
                  f"{true_s:>10s}  {false_s:>10s}")
    print(f"{'=' * 60}")

    # ── 시각화 ──
    roc_path = out_dir / "roc_anomaly_detection.png"
    plot_roc_curve(y_true_out, y_score_out, roc_path,
                   title="Anomaly Detection — Reconstruction Error ROC")
    print(f"\nROC curve: {roc_path}")

    dist_path = out_dir / "score_distribution.png"
    plot_score_distribution(
        y_true_out, y_score_out, metrics["optimal_threshold"],
        dist_path, title="Reconstruction Error Distribution"
    )
    print(f"Score distribution: {dist_path}")

    # ── JSON 저장 ──
    results = {
        **metrics,
        "config": {
            "mode": "reconstruction_error",
            "input_signals": args.input_signals,
            "window_sec": args.window_sec,
            "mask_ratio": args.mask_ratio,
            "n_trials": args.n_trials,
            "patch_size": patch_size,
        },
    }
    results_path = out_dir / "results_anomaly_detection.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
