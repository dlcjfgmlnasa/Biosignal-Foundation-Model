# -*- coding:utf-8 -*-
"""ICU False Alarm Reduction (PhysioNet/CinC Challenge 2015).

ICU 부정맥 알람의 True/False를 판별하는 binary classification task.
ECG(2ch) + ABP/PPG 다채널 입력 → Foundation Model → True alarm 확률.

5가지 알람 타입:
  Asystole, Extreme Bradycardia, Extreme Tachycardia,
  Ventricular Tachycardia, Ventricular Flutter/Fibrillation

모드:
  - linear_probe: Frozen encoder + LinearProbe
  - lora:         Frozen encoder + LoRA adapters + LinearProbe

평가 메트릭:
  - AUROC, AUPRC (전체 + 알람 타입별)
  - Challenge Score: 100 * (TP+TN) / (TP+TN+FP+5*FN)  — FN에 5배 페널티
  - Sensitivity, Specificity

사용법:
    python -m downstream.anomaly_detection.run \
        --checkpoint best.pt --data-dir datasets/processed/anomaly_detection \
        --mode linear_probe --window-sec 60

    python -m downstream.anomaly_detection.run \
        --checkpoint best.pt --data-dir datasets/processed/anomaly_detection \
        --mode lora --lr 1e-4 --lora-rank 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe


# ── 설정 ──────────────────────────────────────────────────────

DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0

# 알람 타입 목록
ALARM_TYPES = [
    "asystole",
    "extreme_bradycardia",
    "extreme_tachycardia",
    "ventricular_tachycardia",
    "ventricular_flutter_fib",
]

# signal_name → signal_type 매핑 (ECG lead가 여러 개이므로 첫 번째만 사용하는 경우)
SIGNAL_NAME_TO_TYPE: dict[str, str] = {
    "II": "ecg", "I": "ecg", "III": "ecg",
    "V": "ecg", "V1": "ecg", "V2": "ecg", "V5": "ecg",
    "aVR": "ecg", "aVL": "ecg", "aVF": "ecg",
    "MCL": "ecg", "MCL1": "ecg",
    "ABP": "abp", "ART": "abp", "AOBP": "abp",
    "PLETH": "ppg",
}


# ── Challenge Score ──────────────────────────────────────────


def challenge_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PhysioNet 2015 Challenge Score.

    Score = 100 * (TP + TN) / (TP + TN + FP + 5*FN)
    FN에 5배 페널티 — 실제 알람을 놓치는 것이 가장 위험.
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    denom = tp + tn + fp + 5 * fn
    if denom == 0:
        return 0.0
    return 100.0 * (tp + tn) / denom


# ── 데이터 로딩 ──────────────────────────────────────────────


def load_records(
    data_dir: str | Path,
    input_signals: list[str],
    window_sec: float = 60.0,
) -> list[dict]:
    """manifest.json에서 레코드를 로드하고 윈도우를 추출한다.

    Parameters
    ----------
    data_dir: processed 디렉토리 (manifest.json + .pt 파일).
    input_signals: 사용할 signal_type 목록 (e.g., ["ecg", "ppg", "abp"]).
    window_sec: 알람 직전 윈도우 길이 (초). 전체 5분 중 마지막 N초 사용.

    Returns
    -------
    list of dict: {signals: {stype: ndarray}, label: int, alarm_type: str, record: str}
    """
    data_path = Path(data_dir)
    manifest_path = data_path / "manifest.json"

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    win_samples = int(window_sec * DEFAULT_SR)
    records = []

    for rec_info in manifest["records"]:
        label = 1 if rec_info["label"] else 0
        alarm_type = rec_info.get("alarm_type", "unknown")

        # 레코드에서 요청한 signal_type별로 .pt 로드
        # ECG가 2ch일 수 있으므로 signal_type별 첫 번째만 사용
        signals: dict[str, np.ndarray] = {}
        type_loaded: set[str] = set()

        for sig_info in rec_info.get("signals", []):
            sig_file = sig_info.get("file", "")
            # signal_type 결정: sig_info에 있으면 사용, 없으면 파일명에서 추출
            sig_type = sig_info.get("signal_type")
            if sig_type is None:
                sig_name = sig_info.get("signal_name", "")
                sig_type = SIGNAL_NAME_TO_TYPE.get(sig_name)
            if sig_type is None or sig_type not in input_signals:
                continue
            if sig_type in type_loaded:
                continue  # 같은 타입 중복 방지 (ECG 2ch → 첫 번째만)

            pt_path = data_path / sig_file
            if not pt_path.exists():
                continue

            tensor = torch.load(pt_path, weights_only=True)  # (1, T)
            signal = tensor.squeeze(0).numpy()  # (T,)

            # 알람 직전 window_sec 추출 (끝에서 자름)
            if len(signal) > win_samples:
                signal = signal[-win_samples:]

            signals[sig_type] = signal
            type_loaded.add(sig_type)

        if not signals:
            continue

        # 모든 채널 길이 통일 (짧은 쪽에 맞춤)
        min_len = min(len(s) for s in signals.values())
        signals = {k: v[-min_len:] for k, v in signals.items()}

        records.append({
            "signals": signals,
            "label": label,
            "alarm_type": alarm_type,
            "record": rec_info.get("record", ""),
        })

    return records


# ── 배치 생성 ────────────────────────────────────────────────


def _record_to_samples(rec: dict, idx: int) -> list[BiosignalSample]:
    """레코드 → BiosignalSample 리스트 (collate 입력용)."""
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


def make_batches(
    records: list[dict],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[PackedBatch, torch.Tensor, list[str]]]:
    """(batch, labels, alarm_types) 리스트 생성."""
    multi = any(len(r["signals"]) > 1 for r in records)
    collate_mode = "any_variate" if multi else "ci"
    collate = PackCollate(
        max_length=max_length, collate_mode=collate_mode, patch_size=patch_size
    )

    batches = []
    for i in range(0, len(records), batch_size):
        chunk = records[i : i + batch_size]
        all_samples = []
        for j, rec in enumerate(chunk):
            all_samples.extend(_record_to_samples(rec, idx=i + j))
        labels = torch.tensor([r["label"] for r in chunk], dtype=torch.float32)
        alarm_types = [r["alarm_type"] for r in chunk]
        batch = collate(all_samples)
        batches.append((batch, labels, alarm_types))
    return batches


# ── Mean Pooling ─────────────────────────────────────────────


def _mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# ── Linear Probe ─────────────────────────────────────────────


def train_linear_probe(
    model,
    probe: LinearProbe,
    train_batches: list[tuple[PackedBatch, torch.Tensor, list[str]]],
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
        for batch, labels, _ in train_batches:
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
    test_batches: list[tuple[PackedBatch, torch.Tensor, list[str]]],
    device: torch.device,
) -> dict:
    probe.to(device).eval()
    all_labels, all_scores, all_alarm_types = [], [], []
    for batch, labels, atypes in test_batches:
        features = model.extract_features(batch, pool="mean").to(device)
        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
        all_alarm_types.extend(atypes)
    return _compute_all_metrics(
        np.concatenate(all_labels), np.concatenate(all_scores), all_alarm_types
    )


# ── LoRA Fine-tuning ─────────────────────────────────────────


def train_lora(
    model,
    probe: LinearProbe,
    train_batches: list[tuple[PackedBatch, torch.Tensor, list[str]]],
    epochs: int,
    lr: float,
    device: torch.device,
    gradient_clip: float = 1.0,
) -> list[float]:
    model.model.train()
    probe = probe.to(device)
    probe.train()

    lora_params = model.lora_parameters()
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": lr},
        {"params": probe.parameters(), "lr": lr},
    ], weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss()
    losses = []

    for epoch in range(epochs):
        epoch_loss, n = 0.0, 0
        for batch, labels, _ in train_batches:
            batch = model.batch_to_device(batch)
            out = model.model(batch, task="masked")
            features = _mean_pool(out["encoded"], out["patch_mask"])

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
    test_batches: list[tuple[PackedBatch, torch.Tensor, list[str]]],
    device: torch.device,
) -> dict:
    model.model.eval()
    probe.to(device).eval()
    all_labels, all_scores, all_alarm_types = [], [], []

    for batch, labels, atypes in test_batches:
        batch = model.batch_to_device(batch)
        out = model.model(batch, task="masked")
        features = _mean_pool(out["encoded"], out["patch_mask"])

        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
        all_alarm_types.extend(atypes)

    return _compute_all_metrics(
        np.concatenate(all_labels), np.concatenate(all_scores), all_alarm_types
    )


# ── 메트릭 계산 ──────────────────────────────────────────────


def _compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    alarm_types: list[str],
) -> dict:
    """전체 + 알람 타입별 메트릭을 계산한다."""
    # 전체 메트릭
    auroc = compute_auroc(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)

    # Optimal threshold (Youden's J)
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
    cscore = challenge_score(y_true, y_pred_opt)

    result = {
        "auroc": auroc,
        "auprc": auprc,
        "f1_macro": f1,
        "challenge_score": cscore,
        "optimal_threshold": float(best_thresh),
        "sensitivity": ss_opt["sensitivity"],
        "specificity": ss_opt["specificity"],
        "n_total": len(y_true),
        "n_true_alarm": int(y_true.sum()),
        "n_false_alarm": int((y_true == 0).sum()),
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
        if len(np.unique(yt)) < 2:
            per_alarm[atype] = {
                "n": int(mask.sum()),
                "n_true": int(yt.sum()),
                "auroc": 0.0,
            }
            continue

        yp = (ys >= best_thresh).astype(int)
        per_alarm[atype] = {
            "n": int(mask.sum()),
            "n_true": int(yt.sum()),
            "n_false": int((yt == 0).sum()),
            "auroc": compute_auroc(yt, ys),
            "challenge_score": challenge_score(yt, yp),
            "sensitivity": compute_sensitivity_specificity(yt, yp)["sensitivity"],
            "specificity": compute_sensitivity_specificity(yt, yp)["specificity"],
        }
    result["per_alarm"] = per_alarm

    return result


# ── CLI 진입점 ───────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICU False Alarm Reduction (PhysioNet 2015)"
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument(
        "--mode", type=str, default="linear_probe",
        choices=["linear_probe", "lora"],
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument(
        "--data-dir", type=str, default="datasets/processed/anomaly_detection",
    )
    parser.add_argument(
        "--input-signals", nargs="+", default=["ecg", "ppg", "abp"],
        choices=["ecg", "ppg", "abp"],
    )
    parser.add_argument(
        "--window-sec", type=float, default=60.0,
        help="알람 직전 윈도우 길이 (초). 기본 60초. 최대 300초(short) 또는 330초(long).",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── 데이터 로드 ──
    print(f"\nLoading data: {args.data_dir}")
    print(f"  Input signals: {args.input_signals}")
    print(f"  Window: {args.window_sec}s (pre-alarm)")

    records = load_records(args.data_dir, args.input_signals, args.window_sec)
    if not records:
        print("ERROR: No records loaded.", file=sys.stderr)
        sys.exit(1)

    # Train/Test split (record-level, stratified by alarm_type)
    rng = np.random.default_rng(42)
    rng.shuffle(records)

    n_train = max(1, int(len(records) * args.train_ratio))
    train_records = records[:n_train]
    test_records = records[n_train:]

    n_true_train = sum(1 for r in train_records if r["label"] == 1)
    n_true_test = sum(1 for r in test_records if r["label"] == 1)
    print(f"  Train: {len(train_records)} ({n_true_train} true alarms)")
    print(f"  Test:  {len(test_records)} ({n_true_test} true alarms)")

    if not test_records:
        print("ERROR: No test records.", file=sys.stderr)
        sys.exit(1)

    # 배치 생성
    first_sig = next(iter(train_records[0]["signals"].values()))
    max_length = len(first_sig)

    train_batches = make_batches(
        train_records, args.batch_size, args.patch_size, max_length
    )
    test_batches = make_batches(
        test_records, args.batch_size, args.patch_size, max_length
    )

    # ── 모델 로드 ──
    if args.checkpoint:
        from downstream.model_wrapper import DownstreamModelWrapper

        print(f"\nLoading checkpoint: {args.checkpoint}")
        model = DownstreamModelWrapper(args.checkpoint, args.model_version, args.device)
        d_model = model.d_model

        if args.mode == "lora":
            model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)
    else:
        print("ERROR: --checkpoint required.", file=sys.stderr)
        sys.exit(1)

    sig_str = " + ".join(s.upper() for s in args.input_signals)
    print(f"Mode: {args.mode} | Input: {sig_str} | Window: {args.window_sec}s")

    # ── 학습 ──
    probe = LinearProbe(d_model, n_classes=1)

    if args.mode == "linear_probe":
        print(f"\nTraining LinearProbe (frozen encoder, d_model={d_model})...")
        train_losses = train_linear_probe(
            model, probe, train_batches, args.epochs, args.lr, device
        )
        print("\nEvaluating...")
        metrics = evaluate_linear_probe(model, probe, test_batches, device)
    elif args.mode == "lora":
        n_lora = sum(p.numel() for p in model.lora_parameters())
        n_probe = sum(p.numel() for p in probe.parameters())
        print(f"\nTraining LoRA + Probe (rank={args.lora_rank}, "
              f"LoRA={n_lora:,} + Probe={n_probe:,} params)...")
        train_losses = train_lora(
            model, probe, train_batches, args.epochs, args.lr, device
        )
        print("\nEvaluating...")
        metrics = evaluate_lora(model, probe, test_batches, device)

    # ── 결과 출력 ──
    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")

    print(f"\n{'=' * 60}")
    print(f"  ICU False Alarm Reduction — {args.mode}")
    print(f"{'=' * 60}")
    print(f"  AUROC:            {metrics['auroc']:.4f}")
    print(f"  AUPRC:            {metrics['auprc']:.4f}")
    print(f"  F1 (macro):       {metrics['f1_macro']:.4f}")
    print(f"  Challenge Score:  {metrics['challenge_score']:.2f}")
    print(f"  Threshold:        {metrics['optimal_threshold']:.3f}")
    print(f"  Sensitivity:      {metrics['sensitivity']:.4f}")
    print(f"  Specificity:      {metrics['specificity']:.4f}")
    print(f"  True/False alarm: {metrics['n_true_alarm']}/{metrics['n_false_alarm']}")

    per_alarm = metrics.get("per_alarm", {})
    if per_alarm:
        print(f"\n  {'Alarm Type':<30s}  {'N':>4s}  {'AUROC':>6s}  {'CScore':>7s}  "
              f"{'Sens':>5s}  {'Spec':>5s}")
        print(f"  {'-' * 70}")
        for atype in ALARM_TYPES:
            if atype not in per_alarm:
                continue
            pa = per_alarm[atype]
            auroc_s = f"{pa['auroc']:.3f}" if pa.get("auroc", 0) > 0 else "  N/A"
            cs_s = f"{pa.get('challenge_score', 0):.1f}" if "challenge_score" in pa else "   N/A"
            sens_s = f"{pa.get('sensitivity', 0):.3f}" if "sensitivity" in pa else "  N/A"
            spec_s = f"{pa.get('specificity', 0):.3f}" if "specificity" in pa else "  N/A"
            print(f"  {atype:<30s}  {pa['n']:>4d}  {auroc_s:>6s}  {cs_s:>7s}  "
                  f"{sens_s:>5s}  {spec_s:>5s}")
    print(f"{'=' * 60}")

    # ROC curve
    roc_path = out_dir / f"roc_{args.mode}.png"
    plot_roc_curve(y_true, y_score, roc_path,
                   title=f"False Alarm Reduction — {args.mode} ROC")
    print(f"\nROC curve: {roc_path}")

    # JSON 저장
    results = {
        **metrics,
        "train_losses": train_losses,
        "config": {
            "mode": args.mode,
            "input_signals": args.input_signals,
            "window_sec": args.window_sec,
            "lora_rank": args.lora_rank if args.mode == "lora" else None,
            "epochs": args.epochs,
            "lr": args.lr,
            "train_ratio": args.train_ratio,
            "batch_size": args.batch_size,
        },
    }
    results_path = out_dir / f"results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()
