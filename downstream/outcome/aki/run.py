# -*- coding:utf-8 -*-
"""Postoperative AKI Prediction (KDIGO Cr 기준).

환자 단위 예측: 수술 중 다채널 신호 (ABP/ECG/PPG/CVP) → postop AKI.

구조 (mortality와 동일 패턴):
    Intraop session (수십 분~수시간)
    → 10분 윈도우 × K개 슬라이딩
    → Foundation Model Encoder (frozen or LoRA) → h_1..h_K
    → [CLS] + h_1..h_K → Transformer Aggregator
    → CLS → LinearProbe → AKI 예측

라벨 모드 (prepare_data.py의 `--label-mode`와 일치):
    binary : Stage ≥1 vs no AKI (BCEWithLogitsLoss)
    stage  : KDIGO 0/1/2/3 (CrossEntropyLoss + macro AUROC OvR)

사용법:
    python -m downstream.outcome.aki.run \
        --checkpoint best.pt \
        --data-path datasets/processed/aki/aki_binary_abp_cvp_ecg_ppg_w600s.pt \
        --mode linear_probe --epochs 30 --max-windows 24
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn

from downstream.metrics import (
    bootstrap_ci,
    compute_auprc,
    compute_auroc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe
from downstream._eval_utils import dump_fold_predictions
from downstream._save_utils import load_prepared_split_chunked
from downstream.aggregator import (
    TransformerAggregator,
    collate_patients,
    encode_patient_windows,
)
from downstream._ddp_utils import (
    ddp_world_size,
    equalize_shard,
    is_main,
    maybe_init_ddp,
    run_aggregator_forward,
    shard_for_rank,
    wrap_aggregator_ddp,
)


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0


# ── 학습 ─────────────────────────────────────────────────────


def train_model(
    model,
    aggregator: TransformerAggregator,
    probe: LinearProbe,
    train_patients: list[dict],
    val_patients: list[dict],
    label_mode: str,
    epochs: int,
    lr: float,
    device: torch.device,
    patch_size: int,
    max_windows: int,
    batch_size: int = 8,
    use_lora: bool = False,
    gradient_clip: float = 1.0,
    ddp_module=None,
    cached_train_override: list | None = None,
    cached_val_override: list | None = None,
) -> tuple[list[float], list[float], dict, int]:
    """학습 + 매 epoch val에서 best AUROC ckpt 추적.

    Returns
    -------
    train_losses, val_aurocs, best_state, best_epoch
    """
    aggregator = aggregator.to(device)
    probe = probe.to(device)
    aggregator.train()
    probe.train()

    params = list(aggregator.parameters()) + list(probe.parameters())
    if use_lora:
        model.model.train()
        params += model.lora_parameters()

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)

    if label_mode == "binary":
        # 클래스 불균형 보정: pos_weight = n_neg / n_pos
        n_pos = sum(1 for p in train_patients if p["label"] == 1)
        n_neg = len(train_patients) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"  BCE pos_weight={pos_weight.item():.2f} (n_pos={n_pos}, n_neg={n_neg})")
    else:
        # stage: 4-class CE
        class_counts = np.bincount(
            [p["label"] for p in train_patients], minlength=4
        )
        weights = 1.0 / np.clip(class_counts, 1, None)
        weights = weights / weights.sum() * 4
        cls_w = torch.tensor(weights, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cls_w)
        print(f"  CE class_weights={weights.round(3).tolist()} "
              f"(counts={class_counts.tolist()})")

    losses: list[float] = []
    val_aurocs: list[float] = []
    best_val_auroc = -float("inf")
    best_state: dict = {}
    best_epoch = 0

    # ── 성능/OOM: frozen encoder(linear_probe)면 train/val 윈도우 인코딩을 1회만 수행하고
    # CPU 에 캐시한다. encoder 가 frozen 이라 매 epoch reprs 가 동일하므로 epoch 루프 안에서
    # (train batch 와 per-epoch val 평가 모두) 인코더 forward 를 반복할 필요가 없다
    # (수치 동일, 중복 재계산·재 I/O 제거). LoRA 는 매 epoch encoder 갱신 → 캐시 불가.
    def _encode_all(
        patients: list[dict],
    ) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
        cache: list[tuple[torch.Tensor, torch.Tensor | None]] = []
        for p in patients:
            reprs, times = encode_patient_windows(
                model, p, patch_size, max_windows,
                use_lora=False, session_prefix="aki",
                return_time_secs=True,
            )
            t = times.detach().cpu() if times is not None else None
            cache.append((reprs.detach().cpu(), t))
        return cache

    # cached_*_override: DDP sharded 추출 경로에서 rank0 가 gather 한 (reprs,times)
    # 캐시 주입(재추출 생략). 단일 GPU 면 None → 기존처럼 내부 추출(불변).
    cached_train: list[tuple[torch.Tensor, torch.Tensor | None]] | None = None
    cached_val: list[tuple[torch.Tensor, torch.Tensor | None]] | None = None
    if not use_lora:
        if cached_train_override is not None:
            cached_train = cached_train_override
            cached_val = cached_val_override
        else:
            cached_train = _encode_all(train_patients)
            cached_val = _encode_all(val_patients)

    for epoch in range(epochs):
        aggregator.train()
        probe.train()
        if use_lora:
            model.model.train()

        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(train_patients))

        epoch_loss, n_batches = 0.0, 0

        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start: batch_start + batch_size]

            if ddp_module is not None:
                # ── DDP 경로: encode→aggregate→probe 를 한 forward 로 묶어 grad
                # all-reduce 가 등록되게 한다 (단일 GPU 경로는 아래 else 그대로).
                # use_time_secs=True 로 aki 의 continuous-time pos-embed 보존. ──
                batch_patients = [train_patients[idx] for idx in batch_indices]
                labels = torch.tensor(
                    [p["label"] for p in batch_patients],
                    dtype=torch.float32, device=device,
                )
                logits = run_aggregator_forward(
                    ddp_module, model, batch_patients, patch_size, max_windows,
                    session_prefix="aki", use_time_secs=True,
                )
            else:
                patient_reprs = []
                patient_times: list[torch.Tensor | None] = []
                batch_labels = []
                for idx in batch_indices:
                    p = train_patients[idx]
                    if cached_train is not None:
                        # frozen encoder → epoch 마다 동일한 캐시 재사용 (인코더 forward 없음)
                        reprs, times = cached_train[idx]
                    else:
                        reprs, times = encode_patient_windows(
                            model, p, patch_size, max_windows,
                            use_lora=use_lora, session_prefix="aki",
                            return_time_secs=True,
                        )
                    patient_reprs.append(reprs)
                    patient_times.append(times)
                    batch_labels.append(p["label"])

                padded, mask, labels, time_secs = collate_patients(
                    patient_reprs, batch_labels, device, time_secs=patient_times
                )
                patient_repr = aggregator(padded, mask, time_secs=time_secs)  # (B, d_model)

                logits = probe(patient_repr)
            if label_mode == "binary":
                loss = criterion(logits.squeeze(-1), labels.float())
            else:
                loss = criterion(logits, labels.long())

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)

        # ── val 평가 + best-ckpt 추적 ──
        # DDP: rank0 만 평가/deepcopy 한다. all-reduce + optimizer.step 후 전 rank
        # 파라미터가 동일하므로 rank0 평가가 전체를 대표한다. 단일 GPU 면 is_main()
        # 이 항상 True → 기존과 동일하게 매 epoch 실행(결과 불변).
        if is_main():
            val_metrics = evaluate_model(
                model, aggregator, probe, val_patients, label_mode,
                device=device, patch_size=patch_size, max_windows=max_windows,
                cached_encodings=cached_val,
            )
            val_metrics.pop("y_true", None)
            val_metrics.pop("y_score", None)
            val_auroc = (
                val_metrics.get("auroc")
                if label_mode == "binary"
                else val_metrics.get("macro_auroc")
            )
            val_auroc = float(val_auroc) if val_auroc is not None else float("nan")
            val_aurocs.append(val_auroc)

            if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch + 1
                best_state = {
                    "aggregator": copy.deepcopy(aggregator.state_dict()),
                    "probe": copy.deepcopy(probe.state_dict()),
                }
                if use_lora and hasattr(model, "model"):
                    best_state["lora"] = copy.deepcopy(model.model.state_dict())

            print(
                f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}  "
                f"val_auroc={val_auroc:.4f}  (best={best_val_auroc:.4f}@ep{best_epoch})"
            )

    if not best_state:
        warnings.warn(
            "No best state captured (val AUROC always NaN). Using last-epoch weights."
        )
        best_state = {
            "aggregator": copy.deepcopy(aggregator.state_dict()),
            "probe": copy.deepcopy(probe.state_dict()),
        }
        if use_lora and hasattr(model, "model"):
            best_state["lora"] = copy.deepcopy(model.model.state_dict())
        best_epoch = epochs

    return losses, val_aurocs, best_state, best_epoch


# ── 평가 ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model,
    aggregator: TransformerAggregator,
    probe: LinearProbe,
    test_patients: list[dict],
    label_mode: str,
    device: torch.device,
    patch_size: int,
    max_windows: int,
    fixed_threshold: float | None = None,
    cached_encodings: list[tuple[torch.Tensor, torch.Tensor | None]] | None = None,
) -> dict:
    aggregator.to(device).eval()
    probe.to(device).eval()
    if hasattr(model, "model"):
        model.model.eval()

    all_labels: list[int] = []
    all_scores: list[np.ndarray] = []  # binary: scalar, stage: (4,) softmax

    for i, p in enumerate(test_patients):
        if cached_encodings is not None:
            # frozen encoder → 미리 인코딩된 (reprs, times) 재사용 (인코더 forward 없음)
            reprs, times = cached_encodings[i]
        else:
            reprs, times = encode_patient_windows(
                model, p, patch_size, max_windows, return_time_secs=True,
            )
        padded = reprs.unsqueeze(0).to(device)  # (1, K, d_model)
        mask = torch.ones(1, reprs.shape[0], dtype=torch.bool, device=device)
        time_secs = times.unsqueeze(0).to(device) if times is not None else None

        patient_repr = aggregator(padded, mask, time_secs=time_secs)
        logit = probe(patient_repr)

        if label_mode == "binary":
            prob = torch.sigmoid(logit).squeeze().cpu().item()
            all_scores.append(np.array([prob]))
        else:
            probs = torch.softmax(logit, dim=-1).squeeze(0).cpu().numpy()
            all_scores.append(probs)

        all_labels.append(p["label"])

    y_true = np.array(all_labels)
    y_score = np.stack(all_scores)  # (N,1) binary or (N,4) stage

    if label_mode == "binary":
        return _compute_metrics_binary(y_true, y_score[:, 0], fixed_threshold)
    return _compute_metrics_stage(y_true, y_score)


# ── 메트릭 ───────────────────────────────────────────────────


def _compute_metrics_binary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fixed_threshold: float | None = None,
) -> dict:
    auroc = compute_auroc(y_true, y_score)
    auprc = compute_auprc(y_true, y_score)

    if fixed_threshold is not None:
        # threshold 는 val 에서 선택된 값을 그대로 사용 (test 셀프튜닝 방지).
        best_thresh = float(fixed_threshold)
    else:
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


def _compute_metrics_stage(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """4-class KDIGO stage 평가: macro AUROC (OvR), accuracy, per-class AUROC."""
    n_classes = 4
    per_class_auroc: list[float] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            per_class_auroc.append(float("nan"))
            continue
        per_class_auroc.append(compute_auroc(y_bin, y_score[:, c]))

    valid = [a for a in per_class_auroc if not np.isnan(a)]
    macro_auroc = float(np.mean(valid)) if valid else float("nan")

    y_pred = y_score.argmax(axis=1)
    acc = float((y_pred == y_true).mean())
    f1 = compute_f1(y_true, y_pred, average="macro")
    counts = np.bincount(y_true, minlength=n_classes).tolist()

    # AKI vs no-AKI binary view (stage>=1 == positive)
    y_true_bin = (y_true >= 1).astype(int)
    y_score_bin = 1.0 - y_score[:, 0]  # P(stage >= 1) = 1 - P(stage=0)
    auroc_bin = (
        compute_auroc(y_true_bin, y_score_bin)
        if 0 < y_true_bin.sum() < len(y_true_bin)
        else float("nan")
    )

    return {
        "macro_auroc": macro_auroc,
        "per_class_auroc": per_class_auroc,
        "accuracy": acc,
        "f1_macro": f1,
        "binary_auroc": auroc_bin,
        "class_counts": counts,
        "n_total": len(y_true),
        "y_true": y_true,
        "y_score": y_score,
    }


# ── 데이터 로딩 ──────────────────────────────────────────────


def _load_data(
    data_path: str, fold: int = 0, n_folds: int = 1, val_split_seed: int = 42
) -> tuple[list[dict], list[dict], list[dict], dict]:
    """Train/val/test 3-way 로드. legacy 2-way 산출물은 train에서 dynamic split.

    단일 통합 .pt (back-compat) 또는 per-(fold,split)[_chunk] prefix 묶음
    (5-fold CV) 양쪽을 처리한다. n_folds>1 이면 해당 fold 의 chunk 만 로드.
    """
    load_fold = int(fold) if int(n_folds) > 1 else None
    print(f"\nLoading data: {data_path} (fold={load_fold})")
    data = load_prepared_split_chunked(data_path, fold=load_fold)
    meta = data.get("metadata", {})
    print(f"  Task:        {meta.get('task', '?')}")
    print(f"  Label mode:  {meta.get('label_mode', '?')}")
    print(f"  Signals:     {meta.get('input_signals', '?')}")
    print(f"  Window:      {meta.get('window_sec', '?')}s")
    print(f"  Postop win:  {meta.get('max_postop_days', '?')} days")

    train_p = data["train"]
    test_p = data["test"]
    val_p = data.get("val")

    if val_p is None:
        warnings.warn(
            "data['val'] missing — falling back to dynamic 20% split of train. "
            "Re-run prepare_data.py to get a stable patient-level val split."
        )
        rng = np.random.default_rng(val_split_seed)
        idx = np.arange(len(train_p))
        rng.shuffle(idx)
        n_val = max(1, len(train_p) // 5)
        val_idx = set(idx[:n_val].tolist())
        val_p = [p for i, p in enumerate(train_p) if i in val_idx]
        train_p = [p for i, p in enumerate(train_p) if i not in val_idx]
        print(f"  (fallback) Val split: {len(val_p)} patients from train")

    return train_p, val_p, test_p, meta


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Postop AKI Prediction (Patient-Level Transformer Aggregation)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--model-version", type=str, default="v1", choices=["v1", "v2"]
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument(
        "--mode", type=str, default="linear_probe",
        choices=["linear_probe", "lora"],
    )
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="환자 수 per batch (윈도우 수 아님)",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument(
        "--max-windows", type=int, default=24,
        help="환자당 최대 윈도우 수 (초과 시 균등 샘플링)",
    )
    parser.add_argument("--agg-layers", type=int, default=2)
    parser.add_argument("--agg-heads", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--fold", type=int, default=0,
                        help="현재 fold 인덱스 (run_eval OOF 집계용 .npz 라벨)")
    parser.add_argument("--n-folds", type=int, default=1, help="전체 fold 수")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--val-split-seed",
        type=int,
        default=42,
        help="Seed for fallback dynamic val split when data['val'] is missing.",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=1000,
        help="Bootstrap iterations for 95% CI on test metrics.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── DDP (B안): torchrun 으로 실행되면 rank 별 GPU 핀 + device 강제. torchrun 이
    # 아니면 maybe_init_ddp()=None → 단일 GPU 경로(기존) 그대로 (결과 불변). ──
    ddp_device = maybe_init_ddp()
    use_ddp = ddp_device is not None
    if use_ddp:
        device = ddp_device
        args.device = str(ddp_device)
        # lora: aggregator DDP. linear_probe: sharded frozen-feature 추출 → gather
        # → rank0 단독 학습. 둘 다 허용(이전엔 linear_probe 차단).
        if is_main():
            print(
                f"[DDP] world_size={ddp_world_size()}  device={device}  "
                f"mode={args.mode}"
            )
    else:
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

    # ── 데이터 로드 (train/val/test 3-way) ──
    train_patients, val_patients, test_patients, meta = _load_data(
        args.data_path, fold=args.fold, n_folds=args.n_folds,
        val_split_seed=args.val_split_seed,
    )
    label_mode = meta.get("label_mode", "binary")
    if label_mode not in {"binary", "stage"}:
        raise ValueError(f"Unknown label_mode in data: {label_mode}")

    if label_mode == "binary":
        n_pos_tr = sum(1 for p in train_patients if p["label"] == 1)
        n_pos_va = sum(1 for p in val_patients if p["label"] == 1)
        n_pos_te = sum(1 for p in test_patients if p["label"] == 1)
        print(f"  Train: {len(train_patients)} patients (AKI={n_pos_tr})")
        print(f"  Val:   {len(val_patients)} patients (AKI={n_pos_va})")
        print(f"  Test:  {len(test_patients)} patients (AKI={n_pos_te})")
    else:
        tr_counts = np.bincount(
            [p["label"] for p in train_patients], minlength=4
        ).tolist()
        va_counts = np.bincount(
            [p["label"] for p in val_patients], minlength=4
        ).tolist()
        te_counts = np.bincount(
            [p["label"] for p in test_patients], minlength=4
        ).tolist()
        print(f"  Train stages: {tr_counts}")
        print(f"  Val stages:   {va_counts}")
        print(f"  Test stages:  {te_counts}")
    avg_win = float(np.mean([p["n_windows"] for p in train_patients]))
    print(f"  Avg windows/patient (train): {avg_win:.1f}")
    print(f"  Max windows per patient: {args.max_windows}")

    # ── DDP 분기 ──
    # lora: train 환자 shard(+equalize) → aggregator DDP. linear_probe: 각 rank 가
    # train/val/test 환자 shard 만 frozen encoder 로 인코딩 → (reprs,times,label,
    # case_id) 한 튜플로 묶어 gather_concat(정렬 co-index 보존) → rank0 단독으로
    # aggregator+probe 학습(val best-ckpt)·test 평가·저장. aki 는 time_secs(continuous
    # pos-embed)와 val 이 있으므로 reprs·times 를 함께 캐싱한다.
    # 단일 GPU(use_ddp=False)면 두 분기 모두 skip → 기존 경로 그대로(불변).
    train_cached: list | None = None
    val_cached: list | None = None
    test_cached: list | None = None
    if use_ddp and use_lora:
        n_full = len(train_patients)
        train_patients = equalize_shard(shard_for_rank(train_patients))
        if is_main():
            print(
                f"  [DDP] train shard: {n_full} → {len(train_patients)}"
                f"/rank × {ddp_world_size()} ranks"
            )
        if len(train_patients) == 0:
            if is_main():
                print(
                    "ERROR: DDP train shard 가 비었습니다 (nproc_per_node 가 train "
                    "환자 수보다 큼). nproc 를 줄이거나 단일 GPU 로 실행하세요.",
                    file=sys.stderr,
                )
            import torch.distributed as dist
            dist.destroy_process_group()
            sys.exit(2)
    elif use_ddp and not use_lora:
        def _extract_shard_tuples(patients):
            # idxs 와 sharded patients 는 동일 stride(shard_for_rank)라 정렬된다.
            idxs = shard_for_rank(list(range(len(patients))))
            out = []
            for gi in idxs:
                p = patients[gi]
                reprs, times = encode_patient_windows(
                    model, p, patch_size, args.max_windows,
                    use_lora=False, session_prefix="aki",
                    return_time_secs=True,
                )
                t = times.detach().cpu() if times is not None else None
                out.append(
                    (gi, reprs.detach().cpu(), t, p["label"], str(p["subject_id"]))
                )
            return out

        if is_main():
            print(f"  [DDP] sharded frozen-feature 추출 × {ddp_world_size()} ranks")
        g_train = gather_concat(_extract_shard_tuples(train_patients))
        g_val = gather_concat(_extract_shard_tuples(val_patients))
        g_test = gather_concat(_extract_shard_tuples(test_patients))
        if not is_main():
            import torch.distributed as dist
            dist.destroy_process_group()
            return
        # rank0: gi 로 정렬해 원순서 복원 → 단일 GPU 와 동일 train/val 순서(동일
        # minibatch, 학습 재현성 보존). co-index(reprs/times/label/case_id)는 한
        # 튜플로 묶여 함께 정렬되므로 절대 어긋나지 않는다.
        def _rebuild(g):
            g.sort(key=lambda t: t[0])
            pats = [
                {"label": lbl, "subject_id": cid, "n_windows": r.shape[0]}
                for (_gi, r, _t, lbl, cid) in g
            ]
            cache = [(r, t) for (_gi, r, t, _lbl, _cid) in g]
            return pats, cache

        train_patients, train_cached = _rebuild(g_train)
        val_patients, val_cached = _rebuild(g_val)
        test_patients, test_cached = _rebuild(g_test)

    # ── Aggregator + Probe ──
    n_classes = 1 if label_mode == "binary" else 4
    aggregator = TransformerAggregator(
        d_model=d_model,
        n_heads=args.agg_heads,
        n_layers=args.agg_layers,
        max_windows=args.max_windows,
    )
    probe = LinearProbe(d_model, n_classes=n_classes)

    n_agg = sum(p.numel() for p in aggregator.parameters())
    n_probe = sum(p.numel() for p in probe.parameters())
    print(f"\n  Aggregator: {n_agg:,} params ({args.agg_layers} layers, "
          f"{args.agg_heads} heads)")
    print(f"  Probe: {n_probe:,} params (n_classes={n_classes})")
    if use_lora:
        n_lora = sum(p.numel() for p in model.lora_parameters())
        print(f"  LoRA: {n_lora:,} params (rank={args.lora_rank})")

    # DDP lora: encode→aggregate→probe 를 한 forward 로 묶어 grad all-reduce 등록.
    # linear_probe DDP 는 rank0 단독 학습이라 wrap 불필요(ddp_module=None).
    # 단일 GPU 면 ddp_module=None → 기존 직접 호출 경로(불변).
    agg_ddp = None
    if use_ddp and use_lora:
        aggregator = aggregator.to(device)
        probe = probe.to(device)
        agg_ddp = wrap_aggregator_ddp(model.model, aggregator, probe)

    # ── 학습 (val에서 best AUROC ckpt 추적) ──
    print(f"\nTraining ({args.mode}, label_mode={label_mode})...")
    train_losses, val_aurocs, best_state, best_epoch = train_model(
        model, aggregator, probe, train_patients, val_patients, label_mode,
        epochs=args.epochs, lr=args.lr, device=device,
        patch_size=patch_size, max_windows=args.max_windows,
        batch_size=args.batch_size, use_lora=use_lora,
        ddp_module=agg_ddp,
        cached_train_override=train_cached, cached_val_override=val_cached,
    )

    # ── DDP lora: 학습 종료 동기화 후 non-rank0 종료 (평가/저장은 rank0 전담).
    # (linear_probe DDP 는 추출 직후 non-rank0 가 이미 종료했다.) ──
    if use_ddp and use_lora:
        import torch.distributed as dist
        dist.barrier()
        if not is_main():
            dist.destroy_process_group()
            return

    # ── Best ckpt 복원 후 test 평가 ──
    print(f"\nRestoring best checkpoint (epoch {best_epoch})...")
    aggregator.load_state_dict(best_state["aggregator"])
    probe.load_state_dict(best_state["probe"])
    if use_lora and "lora" in best_state:
        model.model.load_state_dict(best_state["lora"])

    # binary: operating threshold 를 test 가 아닌 val 에서 선택 (test 셀프튜닝 방지).
    # AUROC/AUPRC 는 threshold-free 라 무관하지만 sens/spec/F1 의 낙관 편향을 제거.
    val_threshold: float | None = None
    if label_mode == "binary":
        val_metrics = evaluate_model(
            model, aggregator, probe, val_patients, label_mode,
            device=device, patch_size=patch_size, max_windows=args.max_windows,
            cached_encodings=val_cached,
        )
        val_threshold = float(val_metrics["optimal_threshold"])
        print(f"  Operating threshold (from val): {val_threshold:.3f}")

    print("Evaluating on test set with best-val ckpt...")
    metrics = evaluate_model(
        model, aggregator, probe, test_patients, label_mode,
        device=device, patch_size=patch_size, max_windows=args.max_windows,
        fixed_threshold=val_threshold,
        cached_encodings=test_cached,
    )

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")
    # patient-level grouping id (test_patients 순서 = 예측 순서).
    patient_ids = [str(p["subject_id"]) for p in test_patients]

    # ── Bootstrap CI on test metrics ──
    print(f"Computing {args.bootstrap_iters}-iter bootstrap 95% CI...")
    ci: dict[str, tuple[float, float]] = {}
    if label_mode == "binary":
        score_1d = y_score[:, 0]
        ci["auroc"] = bootstrap_ci(
            compute_auroc, y_true, score_1d, n_iter=args.bootstrap_iters
        )
        ci["auprc"] = bootstrap_ci(
            compute_auprc, y_true, score_1d, n_iter=args.bootstrap_iters
        )
        thr = float(metrics["optimal_threshold"])
        y_pred = (score_1d >= thr).astype(int)
        ci["f1_macro"] = bootstrap_ci(
            lambda yt, yp: compute_f1(yt, yp, average="macro"),
            y_true, y_pred, n_iter=args.bootstrap_iters,
        )
    else:
        # stage: macro AUROC + binary AUROC (stage>=1)
        y_true_bin = (y_true >= 1).astype(int)
        y_score_bin = 1.0 - y_score[:, 0]
        ci["binary_auroc"] = bootstrap_ci(
            compute_auroc, y_true_bin, y_score_bin, n_iter=args.bootstrap_iters
        )
        y_pred = y_score.argmax(axis=1)
        ci["f1_macro"] = bootstrap_ci(
            lambda yt, yp: compute_f1(yt, yp, average="macro"),
            y_true, y_pred, n_iter=args.bootstrap_iters,
        )

    print(f"\n{'=' * 60}")
    print(f"  Postop AKI ({label_mode}) — {args.mode}")
    print(f"  Best epoch:  {best_epoch}/{args.epochs}")
    val_auroc_best = max(val_aurocs) if val_aurocs else float("nan")
    print(f"  Val AUROC:   {val_auroc_best:.4f}")
    print(f"{'=' * 60}")
    # n_folds>1 이면 fold suffix 를 json·png 에 붙여 같은 out-dir 5-fold 동시/순차
    # 실행 시 torn-write·덮어쓰기를 막는다(.npz 와 동일 규칙). single split 은 기존명.
    # (label_mode 구분은 파일명에 이미 포함 — fold suffix 는 그 뒤에 덧붙인다.)
    fold_suffix = f"_fold{args.fold}" if int(args.n_folds) > 1 else ""
    if label_mode == "binary":
        a_lo, a_hi = ci["auroc"]
        p_lo, p_hi = ci["auprc"]
        f_lo, f_hi = ci["f1_macro"]
        print(f"  AUROC:       {metrics['auroc']:.4f} [{a_lo:.4f}, {a_hi:.4f}]")
        print(f"  AUPRC:       {metrics['auprc']:.4f} [{p_lo:.4f}, {p_hi:.4f}]")
        print(f"  F1 (macro):  {metrics['f1_macro']:.4f} [{f_lo:.4f}, {f_hi:.4f}]")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Prevalence:  {metrics['prevalence']:.3f} "
              f"({metrics['n_positive']}/{metrics['n_total']})")
        roc_path = out_dir / f"aki_roc_{args.mode}{fold_suffix}.png"
        plot_roc_curve(
            y_true, y_score[:, 0], roc_path,
            title=f"Postop AKI — {args.mode} ROC",
        )
        print(f"\nROC curve: {roc_path}")
    else:
        b_lo, b_hi = ci["binary_auroc"]
        f_lo, f_hi = ci["f1_macro"]
        print(f"  Macro AUROC:    {metrics['macro_auroc']:.4f}")
        print(f"  Per-class AUROC: "
              f"{[f'{a:.3f}' for a in metrics['per_class_auroc']]}")
        print(f"  Accuracy:       {metrics['accuracy']:.4f}")
        print(f"  F1 (macro):     {metrics['f1_macro']:.4f} [{f_lo:.4f}, {f_hi:.4f}]")
        print(f"  Binary AUROC:   {metrics['binary_auroc']:.4f} "
              f"[{b_lo:.4f}, {b_hi:.4f}]  (stage≥1)")
        print(f"  Class counts:   {metrics['class_counts']}")
        # 이항 환원 ROC (stage>=1 vs no AKI)
        y_true_bin = (y_true >= 1).astype(int)
        y_score_bin = 1.0 - y_score[:, 0]
        roc_path = out_dir / f"aki_stage_binary_roc_{args.mode}{fold_suffix}.png"
        plot_roc_curve(
            y_true_bin, y_score_bin, roc_path,
            title=f"Postop AKI (stage≥1) — {args.mode} ROC",
        )
        print(f"\nBinary ROC curve: {roc_path}")
    print(f"{'=' * 60}")

    results = {
        **metrics,
        "y_true": y_true.tolist(),
        "y_score": y_score.tolist(),
        "patient_ids": patient_ids,
        "train_losses": train_losses,
        "val_aurocs": val_aurocs,
        "val_auroc_best": val_auroc_best,
        "best_epoch": best_epoch,
        **{f"{k}_ci": list(v) for k, v in ci.items()},
        "config": {
            "task": "postop_aki_prediction",
            "label_mode": label_mode,
            "mode": args.mode,
            "aggregation": "transformer",
            "agg_layers": args.agg_layers,
            "agg_heads": args.agg_heads,
            "max_windows": args.max_windows,
            "data_path": args.data_path,
            "epochs": args.epochs,
            "lr": args.lr,
            "bootstrap_iters": args.bootstrap_iters,
        },
    }
    results_path = out_dir / f"aki_results_{args.mode}_{label_mode}{fold_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")

    # binary: y_score (N,1); stage: (N,4) multi-class → run_eval 가 자동 판별.
    npz_path = dump_fold_predictions(
        out_dir, task=f"aki_{label_mode}", fold_idx=args.fold, n_folds=args.n_folds,
        y_true=y_true, y_score=y_score, patient_ids=patient_ids,
        classes=["0", "1", "2", "3"] if label_mode == "stage" else None,
    )
    print(f"Fold predictions: {npz_path}")

    # DDP: rank0 의 프로세스 그룹 정리(non-rank0 는 학습 직후 이미 정리·종료).
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
