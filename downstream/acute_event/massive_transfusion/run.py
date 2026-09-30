# -*- coding:utf-8 -*-
"""Massive Transfusion Prediction (window-level).

SNUH OR (VitalDB .vital) 기반, 대량수혈 onset(correct_start_time)까지의 절대
잔여시간으로 라벨된 window 를 Foundation model representation 으로 분류한다
(IOH/ICH/CardiacArrest 와 동일한 window-level frozen-probe / LoRA 평가).

2가지 모드:
  - linear_probe: Frozen encoder + LinearProbe (representation 품질 평가, 헤드라인)
  - lora:         Frozen encoder + LoRA adapters + LinearProbe (효율적 fine-tuning)

입력: ABP/PPG 윈도우 → encoder → mean pool → LinearProbe
라벨: masstf+(onset 앞 band) = 1, risk-set matched masstf- = 0

사용법:
    python -m downstream.acute_event.massive_transfusion.run \
        --checkpoint best.pt \
        --data-path F:/Massive_Transfusion_Downstream/massive_transfusion_abp_ppg_w600s_h5min \
        --mode linear_probe --n-folds 5 --fold 0 --epochs 100
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

from data.collate import PackedBatch
from data.dataset import BiosignalSample
from data.spatial_map import SIGNAL_KEY_TO_TYPE, get_global_spatial_id

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe
from downstream.window_task import make_window_batches, iter_window_batches
from downstream._eval_utils import dump_fold_predictions
from downstream._save_utils import load_prepared_split_chunked
from downstream._ddp_utils import (
    ddp_enabled,
    ddp_world_size,
    equalize_shard,
    gather_concat,
    is_main,
    maybe_init_ddp,
    shard_for_rank,
    wrap_lora_ddp,
)


DEFAULT_PATCH_SIZE = 100
DEFAULT_SR = 100.0

# v2: data.spatial_map 의 SSOT(SIGNAL_KEY_TO_TYPE) 사용 (로컬 dict drift 방지).
SIGNAL_TYPE_INT: dict[str, int] = SIGNAL_KEY_TO_TYPE


# ── 배치 생성 ─────────────────────────────────────────────────


def _make_samples(
    signals: dict[str, np.ndarray],
    idx: int,
) -> list[BiosignalSample]:
    samples = []
    for ch, (sig_type, signal) in enumerate(signals.items()):
        stype_int = SIGNAL_TYPE_INT.get(sig_type, 0)
        spatial_id = get_global_spatial_id(stype_int, 0)
        samples.append(
            BiosignalSample(
                values=torch.from_numpy(signal).float(),
                length=len(signal),
                channel_idx=ch,
                recording_idx=idx,
                sampling_rate=DEFAULT_SR,
                n_channels=len(signals),
                win_start=0,
                signal_type=stype_int,
                session_id=f"ca_{idx}",
                spatial_id=spatial_id,
            )
        )
    return samples


def _make_batches(
    windows: list[dict],
    batch_size: int,
    patch_size: int,
    max_length: int,
) -> list[tuple[PackedBatch, torch.Tensor]]:
    return make_window_batches(
        windows,
        batch_size,
        patch_size,
        to_samples=lambda w, idx: _make_samples(w["signals"], idx),
        get_label=lambda w: w["label"],
    )


# ── Mean pooling ─────────────────────────────────────────────


def _mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


# ── Linear Probe (frozen feature 스트리밍 추출) ──────────────


@torch.no_grad()
def _stream_extract_features(
    model, windows, batch_size, patch_size, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """iter_window_batches 로 batch 를 하나씩 빌드·forward·폐기 (peak = batch 1개).

    returns (features (N, d_model) cpu, labels (N,) cpu float).
    """
    feats: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    for batch, labels in iter_window_batches(
        windows, batch_size, patch_size,
        to_samples=lambda w, idx: _make_samples(w["signals"], idx),
        get_label=lambda w: w["label"],
    ):
        f = model.extract_features(batch, pool="mean").detach().cpu()
        feats.append(f)
        labs.append(labels)
    if not feats:
        return torch.empty(0), torch.empty(0)
    return torch.cat(feats, dim=0), torch.cat(labs, dim=0).float()


def _extract_features_maybe_sharded(
    model, windows, batch_size, patch_size, device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """linear_probe feature 추출 (torchrun 병렬화).

    DDP 면 각 rank 가 windows 의 shard 만 스트리밍 추출(peak=batch 1개)한 뒤
    gather_concat 으로 모으고 글로벌 idx argsort 로 원본 순서 복원 → rank 수 무관
    동일 (feature,label) 순서. 비-DDP 면 _stream_extract_features 그대로(byte-identical).
    non-main 은 gather 참여 후 빈 텐서 반환(직후 호출측 종료).
    """
    if not ddp_enabled():
        return _stream_extract_features(model, windows, batch_size, patch_size, device)
    my_windows = shard_for_rank(windows)
    my_gidx = shard_for_rank(list(range(len(windows))))
    feats, labels = _stream_extract_features(
        model, my_windows, batch_size, patch_size, device,
    )
    local = [(torch.tensor(my_gidx, dtype=torch.long), feats, labels)]
    gathered = gather_concat(local)
    if not is_main():
        return torch.empty(0), torch.empty(0)
    parts = [g for g in gathered if g[0].numel() > 0]
    if not parts:
        return torch.empty(0), torch.empty(0)
    idx_all = torch.cat([g[0] for g in parts])
    feats_all = torch.cat([g[1] for g in parts], dim=0)
    labels_all = torch.cat([g[2] for g in parts], dim=0)
    order = torch.argsort(idx_all)
    return feats_all[order], labels_all[order]


@torch.no_grad()
def _eval_probe_cached(probe, features, labels, device):
    """미리 추출된 cached feature 로 probe 평가."""
    probe.to(device).eval()
    logits = probe(features.to(device))
    scores = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
    return _compute_metrics(labels.numpy().astype(int), scores)


# ── LoRA ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_lora(model, probe, test_batches, device):
    model.model.eval()
    probe.to(device).eval()
    all_labels, all_scores = [], []
    for batch, labels in test_batches:
        batch = model.batch_to_device(batch)
        out = model.model(batch, task="masked")
        features = _mean_pool(out["encoded"], out["patch_mask"])
        logits = probe(features)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        all_labels.append(labels.numpy())
        all_scores.append(probs)
    return _compute_metrics(np.concatenate(all_labels), np.concatenate(all_scores))


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


def _load_data(args):
    if not args.data_path:
        print("ERROR: --data-path required", file=sys.stderr)
        sys.exit(1)

    load_fold = int(args.fold) if int(args.n_folds) > 1 else None
    print(f"\nLoading data: {args.data_path} (fold={load_fold})")
    data = load_prepared_split_chunked(args.data_path, fold=load_fold)
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    print(f"  Signals: {meta.get('input_signals', '?')}")
    print(f"  Window: {meta.get('window_sec', '?')}s, "
          f"Horizon: {meta.get('horizon_sec', 0) / 60:.0f}min")

    def _to_windows(split_data):
        windows = []
        labels = split_data["labels"]
        sig_types = list(split_data["signals"].keys())
        subject_ids = split_data.get("subject_ids", None)
        case_ids = split_data.get("case_ids", None)
        for i in range(len(labels)):
            signals = {st: split_data["signals"][st][i].numpy() for st in sig_types}
            windows.append({
                "signals": signals,
                "label": int(labels[i].item()),
                "case_id": case_ids[i] if case_ids is not None else 0,
                "subject_id": subject_ids[i] if subject_ids is not None else "unknown",
            })
        return windows

    train_w = _to_windows(data["train"])
    test_w = _to_windows(data["test"])
    val_data = data.get("val")  # prepare_ca_sweep 가 저장하는 val split
    if val_data is not None:
        val_w = _to_windows(val_data)
        print(f"  val split (from prepare_data): {len(val_w)} windows")
    else:
        # Backward-compat: legacy(no-val) 산출물 — train 에서 20% 동적 split.
        warnings.warn(
            "data['val'] not found; falling back to a 20% dynamic split of train. "
            "Re-run prepare_data.py to get a deterministic val split.",
            stacklevel=2,
        )
        seed = getattr(args, "val_split_seed", 42)
        rng = np.random.default_rng(seed)
        idx = np.arange(len(train_w))
        rng.shuffle(idx)
        n_val = max(1, int(len(train_w) * 0.2))
        vset = set(idx[:n_val].tolist())
        val_w = [w for i, w in enumerate(train_w) if i in vset]
        train_w = [w for i, w in enumerate(train_w) if i not in vset]
        print(f"  val split (dynamic, seed={seed}): {len(val_w)} windows")
    return train_w, val_w, test_w


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Massive Transfusion Prediction (window-level)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="linear_probe",
                        choices=["linear_probe", "lora", "scratch"],
                        help="linear_probe: frozen encoder, lora: LoRA adapters, "
                             "scratch: 동일 구조 random init + 전체 파라미터 학습 "
                             "(사전학습 대조군). scratch 도 --checkpoint 이 필요하나 "
                             "ModelConfig 만 읽고 가중치는 버린다.")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--fold", type=int, default=0,
                        help="standard 모드 fold 인덱스 (run_eval OOF 집계용)")
    parser.add_argument("--n-folds", type=int, default=1, help="전체 fold 수")
    parser.add_argument("--val-split-seed", type=int, default=42,
                        help="legacy(no-val) 산출물에서 동적 val split 시 seed.")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── DDP (B안): torchrun 으로 실행되면 rank 별 GPU 핀 + device 강제.
    # torchrun 이 아니면 maybe_init_ddp()=None → 단일 GPU 경로(기존) 그대로.
    ddp_device = maybe_init_ddp()
    use_ddp = ddp_device is not None
    if use_ddp:
        device = ddp_device
        args.device = str(ddp_device)
        # 두 모드 모두 torchrun 지원: lora=grad all-reduce 데이터 병렬,
        # linear_probe=feature 추출 shard→gather(grad sync 없음). 저장은 rank0 전담.
        if is_main():
            print(f"[DDP] world_size={ddp_world_size()}  device={device}")
    else:
        device = torch.device(args.device)

    from downstream.model_wrapper import DownstreamModelWrapper

    # scratch: 가중치는 버리고 ModelConfig 만 재사용 → 사전학습 모델과 아키텍처
    # 동일한 random-init 대조군 (파라미터 수가 자동으로 일치).
    if args.mode == "scratch":
        print(f"Reading ModelConfig only (weights discarded): {args.checkpoint}")
    else:
        print(f"Loading checkpoint: {args.checkpoint}")
    model = DownstreamModelWrapper(
        args.checkpoint,
        args.model_version,
        args.device,
        init_random=(args.mode == "scratch"),
    )
    d_model = model.d_model

    if args.mode == "lora":
        model.inject_lora(rank=args.lora_rank, alpha=args.lora_alpha)

    train_windows, val_windows, test_windows = _load_data(args)

    n_pos_train = sum(1 for w in train_windows if w["label"] == 1)
    n_pos_val = sum(1 for w in val_windows if w["label"] == 1)
    n_pos_test = sum(1 for w in test_windows if w["label"] == 1)
    print(f"  Train: {len(train_windows)} ({n_pos_train} masstf+, "
          f"{n_pos_train / max(len(train_windows), 1) * 100:.1f}%)")
    print(f"  Val:   {len(val_windows)} ({n_pos_val} masstf+, "
          f"{n_pos_val / max(len(val_windows), 1) * 100:.1f}%)")
    print(f"  Test:  {len(test_windows)} ({n_pos_test} masstf+, "
          f"{n_pos_test / max(len(test_windows), 1) * 100:.1f}%)")

    probe = LinearProbe(d_model, n_classes=1)
    # is_lora = "encoder 를 함께 학습하는 end-to-end 경로" 플래그다(batch 빌드·DDP
    # wrap·best_model_state 저장이 이 경로를 공유). scratch 는 학습 대상 파라미터만
    # 다르므로(전체 vs LoRA adapter) 같은 경로를 탄다. — IOH 러너와 동일 패턴.
    is_scratch = args.mode == "scratch"
    is_lora = args.mode in ("lora", "scratch")

    # ── 준비: linear_probe=frozen feature 캐싱 / lora=batch 빌드 + DDP shard ──
    # IOH/ICH 와 동일하게 매 epoch val AUROC 로 best-ckpt 를 잡고, 마지막에 best-val
    # 시점을 복원해 test 1회. 긴 학습에서도 overfitting 이 test 지표를 안 해친다.
    max_length: int | None = None
    train_features = train_labels = None
    val_features = val_labels = None
    test_features = test_labels = None
    train_batches = None
    val_batches = None
    lora_params = None
    ddp_module = None

    if not is_lora:
        # frozen encoder feature 1회 추출(train/val/test). DDP 면 rank 별 shard 추출
        # → gather(원본 순서 복원) → rank0 만 probe 학습·평가. 단일 GPU 면 동일 경로.
        train_features, train_labels = _extract_features_maybe_sharded(
            model, train_windows, args.batch_size, args.patch_size, device,
        )
        val_features, val_labels = _extract_features_maybe_sharded(
            model, val_windows, args.batch_size, args.patch_size, device,
        )
        test_features, test_labels = _extract_features_maybe_sharded(
            model, test_windows, args.batch_size, args.patch_size, device,
        )
        if use_ddp and not is_main():
            import torch.distributed as dist
            dist.destroy_process_group()
            return
        probe = probe.to(device)
        optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        val_features = val_features.to(device)
        print(f"\nTraining LinearProbe (d_model={d_model}) with val best-ckpt...")
    else:
        # lora: encoder fine-tune → feature 캐싱 불가, pre-built batches 필요.
        # ── DDP(B안): train window 를 rank 별 분할 + 전 rank 최소 길이 정렬(step 동기화).
        #   val/test 는 분할하지 않는다(평가는 rank0 가 full set 으로 수행).
        if use_ddp:
            n_full = len(train_windows)
            train_windows = equalize_shard(shard_for_rank(train_windows))
            if is_main():
                print(
                    f"  [DDP] train shard: {n_full} → {len(train_windows)}"
                    f"/rank × {ddp_world_size()} ranks"
                )
            if len(train_windows) == 0:
                # world_size 가 train window 수보다 크면 빈 shard → random-init LoRA
                # 가 "결과"로 저장될 위험. 명시적으로 차단.
                if is_main():
                    print(
                        "ERROR: DDP train shard 가 비었습니다 (nproc_per_node 가 "
                        "train window 수보다 큼). nproc 를 줄이거나 단일 GPU 로 "
                        "실행하세요.",
                        file=sys.stderr,
                    )
                import torch.distributed as dist
                dist.destroy_process_group()
                sys.exit(2)
        first_sig = next(iter(train_windows[0]["signals"].values()))
        max_length = len(first_sig)
        train_batches = _make_batches(
            train_windows, args.batch_size, args.patch_size, max_length,
        )
        # val batch 는 rank0 만 평가에 쓴다(분할 X — full val set).
        if is_main():
            val_batches = _make_batches(
                val_windows, args.batch_size, args.patch_size, max_length,
            )
        probe = probe.to(device)
        # 학습 대상 encoder 파라미터: lora=adapter 만, scratch=encoder 전체.
        lora_params = (
            [p for p in model.model.parameters() if p.requires_grad]
            if is_scratch
            else model.lora_parameters()
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_params, "lr": args.lr},
                {"params": probe.parameters(), "lr": args.lr},
            ],
            weight_decay=0.01,
        )
        # DDP: encode→pool→probe 를 한 forward 로 묶어 grad all-reduce 등록. 단일 GPU
        # 면 use_ddp=False → ddp_module=None → 기존 _make/forward 경로 그대로(불변).
        ddp_module = wrap_lora_ddp(model.model, probe) if use_ddp else None
        n_enc = sum(p.numel() for p in lora_params)
        if is_scratch:
            print(f"\nTraining from scratch (random init, full encoder={n_enc:,}) "
                  f"with val best-ckpt...")
        else:
            print(f"\nTraining LoRA + Probe (rank={args.lora_rank}, LoRA={n_enc:,}) "
                  f"with val best-ckpt...")

    criterion = nn.BCEWithLogitsLoss()
    train_losses: list[float] = []
    best_val_auroc = -1.0
    best_epoch = -1
    best_probe_state: dict | None = None
    best_model_state: dict | None = None  # lora 에서만

    for epoch in range(args.epochs):
        probe.train()
        if is_lora:
            model.model.train()
            if ddp_module is not None:
                ddp_module.train()
        epoch_loss, n_steps = 0.0, 0

        if not is_lora:
            n = train_features.size(0)
            perm = torch.randperm(n, device=device)
            for i in range(0, n, args.batch_size):
                idx = perm[i: i + args.batch_size]
                logits = probe(train_features[idx])
                loss = criterion(logits, train_labels[idx].unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1
        else:
            for batch, labels in train_batches:
                batch = model.batch_to_device(batch)
                if ddp_module is not None:
                    logits = ddp_module(batch)
                else:
                    out = model.model(batch, task="masked")
                    features = _mean_pool(out["encoded"], out["patch_mask"])
                    logits = probe(features)
                loss = criterion(logits, labels.to(device).unsqueeze(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    lora_params + list(probe.parameters()), 1.0,
                )
                optimizer.step()
                epoch_loss += loss.item()
                n_steps += 1

        avg = epoch_loss / max(n_steps, 1)
        train_losses.append(avg)

        # ── val best-ckpt (DDP: rank0 만; all-reduce 후 전 rank 파라미터 동일) ──
        if is_main():
            probe.eval()
            if not is_lora:
                vm = _eval_probe_cached(probe, val_features, val_labels, device)
            else:
                model.model.eval()
                vm = evaluate_lora(model, probe, val_batches, device)
            val_auroc = float(vm["auroc"])
            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                best_epoch = epoch
                best_probe_state = copy.deepcopy(probe.state_dict())
                if is_lora:
                    best_model_state = copy.deepcopy(model.model.state_dict())
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
                print(
                    f"  Epoch {epoch + 1}/{args.epochs}  loss={avg:.4f}  "
                    f"val_auroc={val_auroc:.4f}  "
                    f"(best={best_val_auroc:.4f}@ep{best_epoch + 1})"
                )

    # DDP(lora): 학습 종료 동기화 후 test/저장은 rank0 전담 → non-rank0 정리·종료.
    # (linear_probe non-main 은 위 feature 추출 직후 이미 종료했다.)
    if use_ddp and is_lora:
        import torch.distributed as dist
        dist.barrier()
        if not is_main():
            dist.destroy_process_group()
            return

    # ── best-val ckpt 복원 후 test 1회 ──
    if best_probe_state is None:
        print("WARNING: no best ckpt captured; using last epoch.", file=sys.stderr)
    else:
        probe.load_state_dict(best_probe_state)
        if is_lora and best_model_state is not None:
            model.model.load_state_dict(best_model_state)

    print("\nEvaluating on test set with best-val ckpt...")
    if not is_lora:
        metrics = _eval_probe_cached(probe, test_features, test_labels, device)
    else:
        test_batches = _make_batches(
            test_windows, args.batch_size, args.patch_size, max_length,
        )
        metrics = evaluate_lora(model, probe, test_batches, device)

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")
    # subject-level grouping id (test_windows 순서 = 예측 순서).
    patient_ids = [str(w.get("subject_id", "unknown")) for w in test_windows]

    print(f"\n{'=' * 60}")
    print(f"  Massive Transfusion Prediction - {args.mode}")
    print(f"{'=' * 60}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Prevalence:  {metrics['prevalence']:.3f} "
          f"({metrics['n_positive']}/{metrics['n_total']})")
    print(f"{'=' * 60}")

    # fold suffix: n_folds>1 동시 실행 시 ROC PNG/JSON 충돌(torn-file) 방지.
    fold_suffix = f"_fold{args.fold}" if int(args.n_folds) > 1 else ""
    roc_path = out_dir / f"massive_transfusion_roc_{args.mode}{fold_suffix}.png"
    plot_roc_curve(y_true, y_score, roc_path,
                   title=f"Massive Transfusion - {args.mode} ROC")
    print(f"\nROC curve: {roc_path}")

    results = {
        **metrics,
        "y_true": y_true.tolist(),
        "y_score": y_score.tolist(),
        "patient_ids": patient_ids,
        "train_losses": train_losses,
        "config": {
            "task": "massive_transfusion_prediction",
            "mode": args.mode,
            "data_path": args.data_path,
            "epochs": args.epochs, "lr": args.lr,
        },
    }
    results_path = out_dir / f"massive_transfusion_results_{args.mode}{fold_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")

    npz_path = dump_fold_predictions(
        out_dir, task="massive_transfusion", fold_idx=args.fold,
        n_folds=args.n_folds, y_true=y_true, y_score=y_score,
        patient_ids=patient_ids,
    )
    print(f"Fold predictions: {npz_path}")

    # DDP: rank0 의 프로세스 그룹 정리(non-rank0 는 학습 직후 이미 정리·종료).
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
