# -*- coding:utf-8 -*-
"""Outcome Prediction — Cardiac Arrest (patient-level, SCOPE).

환자 단위 예측: 여러 5분 윈도우(고정 12h band)를 Foundation Model 로 인코딩한 뒤,
Transformer Aggregator 로 시간 순서를 반영하여 환자 수준 표현을 생성한다.
라벨: arrest 환자(1) vs discharge 환자(0) (death 제외).

구조:
    ICU Stay (고정 12h history band, anchor 12h 전까지)
    → 5분 윈도우 × K개 슬라이딩
    → Foundation Model Encoder (frozen) → h_1, h_2, ..., h_K  (d_model)
    → [CLS] + h_1..h_K → Transformer Aggregator (학습 가능)
    → CLS output → LinearProbe → cardiac-arrest outcome 예측

2가지 모드:
  - linear_probe: Frozen encoder + Transformer Aggregator + LinearProbe
  - lora:         Frozen encoder + LoRA + Transformer Aggregator + LinearProbe

사용법:
    python -m downstream.outcome.cardiac_arrest.run \
        --checkpoint best.pt \
        --data-path datasets/processed/scope_cardiac_arrest_outcome/cardiac_arrest_w300s.pt \
        --mode linear_probe --epochs 30 --max-windows 144 \
        --n-folds 5 --fold 0
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)
from downstream.viz import plot_roc_curve
from downstream.model_wrapper import LinearProbe
from downstream._eval_utils import dump_fold_predictions
from downstream._save_utils import (
    iter_prepared_split_chunks,
    load_prepared_split_chunked,
)
from downstream.aggregator import (
    SIGNAL_TYPE_INT,
    LastKAggregator,
    MeanAggregator,
    TransformerAggregator,
    collate_patients,
    encode_patient_windows,
)
from downstream._ddp_utils import (
    ddp_rank,
    ddp_world_size,
    equalize_shard,
    gather_concat,
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
    epochs: int,
    lr: float,
    device: torch.device,
    patch_size: int,
    max_windows: int,
    batch_size: int = 8,
    use_lora: bool = False,
    gradient_clip: float = 1.0,
    ddp_module=None,
    cached_reprs_override: list[torch.Tensor] | None = None,
) -> list[float]:
    aggregator = aggregator.to(device)
    probe = probe.to(device)
    aggregator.train()
    probe.train()

    params = list(aggregator.parameters()) + list(probe.parameters())
    if use_lora:
        model.model.train()
        params += model.lora_parameters()

    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    losses = []

    # ── 성능/OOM: frozen encoder(linear_probe)면 환자 윈도우 인코딩을 1회만 수행하고
    # CPU 에 캐시한다. encoder 가 frozen 이라 매 epoch reprs 가 동일하므로 epoch 루프
    # 안에서 인코더 forward 를 반복할 필요가 없다 (수치 동일, 중복 재계산·재 I/O 제거).
    # LoRA 는 encoder 가 매 epoch 갱신되어 features 가 바뀌므로 캐시 불가 → 기존 경로 유지.
    # cached_reprs_override: DDP sharded 추출 경로에서 rank0 가 gather 한 reprs 를
    # 그대로 주입(재추출 생략). 단일 GPU 면 None → 기존처럼 내부에서 추출(불변).
    cached_reprs: list[torch.Tensor] | None = None
    if not use_lora:
        if cached_reprs_override is not None:
            cached_reprs = cached_reprs_override
        else:
            cached_reprs = []
            for p in tqdm(train_patients, desc="[LP] encode train",
                          unit="pt", disable=not is_main()):
                reprs = encode_patient_windows(
                    model, p, patch_size, max_windows,
                    use_lora=False, session_prefix="ca_out",
                )
                cached_reprs.append(reprs.detach().cpu())

    epoch_bar = tqdm(range(epochs), desc="[train] epoch",
                     unit="ep", disable=not is_main())
    for epoch in epoch_bar:
        # 환자 셔플
        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(train_patients))

        epoch_loss, n_batches = 0.0, 0

        for batch_start in range(0, len(order), batch_size):
            batch_indices = order[batch_start: batch_start + batch_size]

            if ddp_module is not None:
                # ── DDP 경로: encode→aggregate→probe 를 한 forward 로 묶어 grad
                # all-reduce 가 등록되게 한다 (단일 GPU 경로는 아래 else 그대로). ──
                batch_patients = [train_patients[idx] for idx in batch_indices]
                labels = torch.tensor(
                    [p["label"] for p in batch_patients],
                    dtype=torch.float32, device=device,
                )
                logits = run_aggregator_forward(
                    ddp_module, model, batch_patients, patch_size, max_windows,
                    session_prefix="ca_out",
                )
            else:
                # 1. 각 환자의 윈도우를 인코딩
                patient_reprs = []
                batch_labels = []
                for idx in batch_indices:
                    p = train_patients[idx]
                    if cached_reprs is not None:
                        # frozen encoder → epoch 마다 동일한 캐시된 repr 재사용 (인코더 forward 없음)
                        reprs = cached_reprs[idx]
                    else:
                        reprs = encode_patient_windows(
                            model, p, patch_size, max_windows,
                            use_lora=use_lora, session_prefix="ca_out",
                        )
                    patient_reprs.append(reprs)
                    batch_labels.append(p["label"])

                # 2. 패딩 + Aggregator
                padded, mask, labels, _ = collate_patients(
                    patient_reprs, batch_labels, device
                )
                patient_repr = aggregator(padded, mask)  # (B, d_model)

                # 3. Probe
                logits = probe(patient_repr)

            loss = criterion(logits.squeeze(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        epoch_bar.set_postfix(loss=f"{avg:.4f}")

    return losses


# ── 평가 ─────────────────────────────────────────────────────


@torch.no_grad()
def evaluate_model(
    model,
    aggregator: TransformerAggregator,
    probe: LinearProbe,
    test_patients: list[dict],
    device: torch.device,
    patch_size: int,
    max_windows: int,
    precomputed_reprs: list[torch.Tensor] | None = None,
) -> dict:
    aggregator.to(device).eval()
    probe.to(device).eval()
    if hasattr(model, "model"):
        model.model.eval()

    all_labels, all_scores = [], []

    # precomputed_reprs: DDP sharded 추출 경로에서 rank0 가 gather 한 test reprs
    # (test_patients 와 co-index). 단일 GPU 면 None → 기존처럼 즉시 인코딩(불변).
    for i, p in enumerate(tqdm(test_patients, desc="[eval] encode test",
                               unit="pt", disable=not is_main())):
        if precomputed_reprs is not None:
            reprs = precomputed_reprs[i]
        else:
            reprs = encode_patient_windows(model, p, patch_size, max_windows)
        padded = reprs.unsqueeze(0).to(device)  # (1, K, d_model)
        mask = torch.ones(1, reprs.shape[0], dtype=torch.bool, device=device)

        patient_repr = aggregator(padded, mask)
        logit = probe(patient_repr)
        prob = torch.sigmoid(logit).squeeze().cpu().item()

        all_labels.append(p["label"])
        all_scores.append(prob)

    return _compute_metrics(np.array(all_labels), np.array(all_scores))


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


def _load_data(
    data_path: str, fold: int = 0, n_folds: int = 1,
    max_chunks: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """환자 단위로 그룹핑된 .pt 를 로드한다.

    단일 통합 .pt (back-compat) 또는 per-(fold,split)[_chunk] prefix 묶음
    (ablation runner) 양쪽을 처리한다. n_folds>1 이면 해당 fold 의 chunk 만 로드.
    chunk 들의 ``data`` (patient dict 리스트) 는 split 별로 extend concat 된다.
    max_chunks 가 주어지면 split 당 앞쪽 N chunk 만 로드한다(dry-run 가속).
    """
    load_fold = int(fold) if int(n_folds) > 1 else None
    print(f"\nLoading data: {data_path} (fold={load_fold}, max_chunks={max_chunks})")
    data = load_prepared_split_chunked(
        data_path, fold=load_fold, max_chunks=max_chunks,
    )
    meta = data.get("metadata", {})
    print(f"  Task: {meta.get('task', '?')}")
    print(f"  Signals: {meta.get('input_signals', '?')}")
    print(f"  Window: {meta.get('window_sec', '?')}s")
    print(f"  Aggregation: {meta.get('aggregation', 'window_level')}")

    return data["train"], data["test"], meta


def _subsample_dry_run(
    patients: list[dict], n_per_class: int, seed: int,
) -> list[dict]:
    """dry-run: 클래스별로 최대 n_per_class 명만 남긴다 (양/음성 모두 확보).

    양성이 희소하므로 random 이 아니라 클래스별 stratified 로 뽑아, 소수 표본이라도
    AUROC/AUPRC 가 계산 가능하도록(양·음성 최소 1건 이상) 보장한다.
    """
    rng = np.random.default_rng(seed)
    pos = [p for p in patients if p["label"] == 1]
    neg = [p for p in patients if p["label"] == 0]

    def _take(pool: list[dict], n: int) -> list[dict]:
        if len(pool) <= n:
            return list(pool)
        idx = rng.choice(len(pool), size=n, replace=False)
        return [pool[int(i)] for i in idx]

    return _take(pos, n_per_class) + _take(neg, n_per_class)


def _stream_extract_lp_shard(
    data_path: str,
    load_fold: int | None,
    split: str,
    model,
    patch_size: int,
    max_windows: int,
    use_ddp: bool,
) -> list[tuple[int, torch.Tensor, int, str]]:
    """LP frozen-feature 를 chunk 스트리밍으로 추출 — 전체 raw window 상주 회피.

    chunk 를 하나씩 로드 → 환자별 encode → (gi, repr_cpu, label, subject_id) 저장 →
    chunk 폐기. peak RAM = chunk 1개 + 누적 reprs(환자당 ~KB). DDP 면 각 rank 가
    자기 shard(gi%world==rank, shard_for_rank 와 동일 stride)만 추출 → gather+gi
    정렬 후 단일 GPU 와 byte-identical. reprs 는 encode_patient_windows 가 결정적.
    """
    world = ddp_world_size() if use_ddp else 1
    rank = ddp_rank() if use_ddp else 0
    out: list[tuple[int, torch.Tensor, int, str]] = []
    gi = 0
    for payload in iter_prepared_split_chunks(data_path, load_fold, split):
        patients = payload if isinstance(payload, list) else payload.get("data", [])
        for p in patients:
            if gi % world == rank:
                reprs = encode_patient_windows(
                    model, p, patch_size, max_windows,
                    use_lora=False, session_prefix="ca_out",
                )
                out.append(
                    (gi, reprs.detach().cpu(), int(p["label"]), str(p["subject_id"]))
                )
            gi += 1
        del payload
        patients = None
        gc.collect()
    return out


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cardiac Arrest Outcome Prediction (Patient-Level Transformer Aggregation)"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model-version", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="linear_probe",
                        choices=["linear_probe", "lora"])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="환자 수 per batch (윈도우 수 아님)")
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument("--max-windows", type=int, default=144,
                        help="환자당 최대 윈도우 수 (초과 시 균등 샘플링)")
    parser.add_argument("--agg-layers", type=int, default=2,
                        help="Transformer Aggregator 레이어 수")
    parser.add_argument("--agg-heads", type=int, default=4)
    parser.add_argument("--last-k", type=int, default=36,
                        help="[aggregator=lastk] 관측 후반부 마지막 K개 valid window 만 "
                             "평균 (5min window 기준 36=마지막 3h). event 직전 구간 집중.")
    parser.add_argument("--aggregator", type=str, default="mean",
                        choices=["transformer", "mean", "lastk"],
                        help="환자-수준 집약 방식. transformer(기본): 학습 가능 "
                             "aggregator. mean: 파라미터 없는 masked mean "
                             "(소규모 코호트 aggregator 과적합 방지, cardiac arrest 등).")
    parser.add_argument("--out-dir", type=str, default=".")
    parser.add_argument("--fold", type=int, default=0,
                        help="현재 fold 인덱스 (run_eval OOF 집계용 .npz 라벨)")
    parser.add_argument("--n-folds", type=int, default=1, help="전체 fold 수")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dry-run", action="store_true",
                        help="파이프라인 스모크 테스트: 소수 환자만 추출→학습→평가→저장 "
                             "하여 크래시 여부만 빠르게 확인 (결과는 무의미).")
    parser.add_argument("--dry-run-n", type=int, default=4,
                        help="dry-run 시 split·클래스당 표본 환자 수 (기본 4)")
    parser.add_argument("--dry-run-windows", type=int, default=8,
                        help="dry-run 시 환자당 max_windows 상한 (기본 8, 추출 가속)")
    parser.add_argument("--dry-run-epochs", type=int, default=2,
                        help="dry-run 시 epoch 상한 (기본 2)")
    parser.add_argument("--dry-run-chunks", type=int, default=2,
                        help="dry-run 시 split 당 로드할 chunk 수 (기본 2, 로딩 가속)")
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
        # lora: aggregator DDP(환자 shard·grad all-reduce). linear_probe: frozen
        # feature 를 rank 별 sharded 추출 → gather → rank0 가 aggregator+probe 학습.
        # 둘 다 허용(이전엔 linear_probe 를 막았으나 sharded 추출로 가속 가능).
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

    # ── 데이터 로드 / 특징 추출 ──
    # LP 실측 run 은 chunk 스트리밍 추출로 전체 raw window 상주를 피한다(RAM OOM 회피
    #   — SCOPE 24-48h ICU × DDP rank 별 중복 로드가 수백 GB). dry-run 은 max_chunks 로
    #   이미 소량이라 기존 경로 유지. lora 는 매 epoch encoder 재인코딩이 필요해 raw
    #   window 를 들고 있어야 하므로 기존 경로 유지.
    stream_lp = (not use_lora) and (not args.dry_run)
    cached_train_override: list[torch.Tensor] | None = None
    test_precomputed: list[torch.Tensor] | None = None
    meta: dict = {}

    if stream_lp:
        load_fold = int(args.fold) if int(args.n_folds) > 1 else None
        if is_main():
            print("\n[streaming LP] chunk 별 frozen-feature 추출 (peak RAM=chunk 1개)")
        g_train = _stream_extract_lp_shard(
            args.data_path, load_fold, "train", model, patch_size,
            args.max_windows, use_ddp,
        )
        g_test = _stream_extract_lp_shard(
            args.data_path, load_fold, "test", model, patch_size,
            args.max_windows, use_ddp,
        )
        if use_ddp:
            g_train = gather_concat(g_train)
            g_test = gather_concat(g_test)
            if not is_main():
                import torch.distributed as dist
                dist.destroy_process_group()
                return
        # gi 정렬 → 단일 GPU 와 동일 순서(재현성; window군과 동일 기법).
        g_train.sort(key=lambda t: t[0])
        g_test.sort(key=lambda t: t[0])
        train_patients = [
            {"label": lbl, "subject_id": cid, "n_windows": r.shape[0]}
            for (_gi, r, lbl, cid) in g_train
        ]
        cached_train_override = [r for (_gi, r, _l, _c) in g_train]
        test_patients = [
            {"label": lbl, "subject_id": cid, "n_windows": r.shape[0]}
            for (_gi, r, lbl, cid) in g_test
        ]
        test_precomputed = [r for (_gi, r, _l, _c) in g_test]
    else:
        #   dry-run 이면 split 당 앞쪽 few chunk 만 읽어 수십 GB 전체 로딩을 피한다
        #   (chunk0 에 양성이 포함돼 있어 스모크 테스트에 충분).
        load_max_chunks = args.dry_run_chunks if args.dry_run else None
        train_patients, test_patients, meta = _load_data(
            args.data_path, args.fold, args.n_folds, max_chunks=load_max_chunks,
        )
        # ── dry-run: 추출 전에 환자를 소수로 줄이고 윈도우·epoch 상한을 낮춘다. ──
        if args.dry_run:
            train_patients = _subsample_dry_run(train_patients, args.dry_run_n, seed=0)
            test_patients = _subsample_dry_run(test_patients, args.dry_run_n, seed=1)
            args.max_windows = min(args.max_windows, args.dry_run_windows)
            args.epochs = min(args.epochs, args.dry_run_epochs)
            if is_main():
                print("\n" + "!" * 60)
                print("  DRY-RUN — 파이프라인 스모크 테스트 (결과 무의미, 저장물 폐기)")
                print(f"    train={len(train_patients)} / test={len(test_patients)} patients "
                      f"| max_windows={args.max_windows} | epochs={args.epochs}")
                print("!" * 60)

    n_pos_train = sum(1 for p in train_patients if p["label"] == 1)
    n_pos_test = sum(1 for p in test_patients if p["label"] == 1)
    avg_win_train = np.mean([p["n_windows"] for p in train_patients]) if train_patients else 0.0
    avg_win_test = np.mean([p["n_windows"] for p in test_patients]) if test_patients else 0.0
    print(f"  Train: {len(train_patients)} patients "
          f"({n_pos_train} arrest, avg {avg_win_train:.1f} windows)")
    print(f"  Test:  {len(test_patients)} patients "
          f"({n_pos_test} arrest, avg {avg_win_test:.1f} windows)")
    print(f"  Max windows per patient: {args.max_windows}")

    # ── DDP 분기 (lora / dry-run-LP). stream_lp 는 위에서 추출 완료 → skip. ──
    # lora: train 환자 리스트를 rank 별 분할(+equalize)해 aggregator DDP 로 학습.
    # dry-run-LP: 각 rank 가 환자 shard 만 인코딩 → gather → rank0 단독 학습.
    # 단일 GPU(use_ddp=False)면 두 분기 모두 skip → 기존 경로 그대로(불변).
    if not stream_lp and use_ddp and use_lora:
        n_full = len(train_patients)
        train_patients = equalize_shard(shard_for_rank(train_patients))
        if is_main():
            print(
                f"  [DDP] train shard: {n_full} → {len(train_patients)}"
                f"/rank × {ddp_world_size()} ranks"
            )
        if len(train_patients) == 0:
            # world_size 가 train 환자 수보다 크면 shard 가 비어 학습이 no-op 되고
            # random-init 가중치가 저장될 수 있다 → 명시적으로 차단.
            if is_main():
                print(
                    "ERROR: DDP train shard 가 비었습니다 (nproc_per_node 가 train "
                    "환자 수보다 큼). nproc 를 줄이거나 단일 GPU 로 실행하세요.",
                    file=sys.stderr,
                )
            import torch.distributed as dist
            dist.destroy_process_group()
            sys.exit(2)
    elif not stream_lp and use_ddp and not use_lora:
        # (dry-run-LP 전용 경로 — 실측 LP 는 위 stream_lp 에서 처리)
        # sharded frozen-feature 추출 + gather. global index(gi)를 튜플에 포함해
        # gather 후 gi 로 정렬(argsort)해 **원순서를 복원** → 단일 GPU 와 동일한
        # train_patients 순서가 되어 epoch seed permutation 이 동일 minibatch 를
        # 만든다(학습 재현성 보존, window군과 동일 기법). co-index(reprs↔label↔
        # case_id)는 한 튜플로 묶여 함께 정렬되므로 절대 어긋나지 않는다.
        def _extract_shard_tuples(patients):
            # idxs 와 sharded patients 는 동일 stride(shard_for_rank)라 정렬된다.
            idxs = shard_for_rank(list(range(len(patients))))
            out = []
            for gi in tqdm(idxs, desc="[extract] shard", unit="pt",
                           disable=not is_main()):
                p = patients[gi]
                reprs = encode_patient_windows(
                    model, p, patch_size, args.max_windows,
                    use_lora=False, session_prefix="ca_out",
                )
                out.append(
                    (gi, reprs.detach().cpu(), p["label"], str(p["subject_id"]))
                )
            return out

        if is_main():
            print(f"  [DDP] sharded frozen-feature 추출 × {ddp_world_size()} ranks")
        g_train = gather_concat(_extract_shard_tuples(train_patients))
        g_test = gather_concat(_extract_shard_tuples(test_patients))
        if not is_main():
            import torch.distributed as dist
            dist.destroy_process_group()
            return
        # rank0: gi 로 정렬해 원순서 복원 → 단일 GPU 와 동일 리스트/순서.
        g_train.sort(key=lambda t: t[0])
        g_test.sort(key=lambda t: t[0])
        train_patients = [
            {"label": lbl, "subject_id": cid, "n_windows": r.shape[0]}
            for (_gi, r, lbl, cid) in g_train
        ]
        cached_train_override = [r for (_gi, r, _lbl, _cid) in g_train]
        test_patients = [
            {"label": lbl, "subject_id": cid, "n_windows": r.shape[0]}
            for (_gi, r, lbl, cid) in g_test
        ]
        test_precomputed = [r for (_gi, r, _lbl, _cid) in g_test]

    # ── Aggregator + Probe ──
    if args.aggregator == "mean":
        aggregator = MeanAggregator(d_model=d_model)
    elif args.aggregator == "lastk":
        aggregator = LastKAggregator(d_model=d_model, last_k=args.last_k)
    else:
        aggregator = TransformerAggregator(
            d_model=d_model,
            n_heads=args.agg_heads,
            n_layers=args.agg_layers,
            max_windows=args.max_windows,
        )
    probe = LinearProbe(d_model, n_classes=1)

    n_agg = sum(p.numel() for p in aggregator.parameters())
    n_probe = sum(p.numel() for p in probe.parameters())
    if args.aggregator == "transformer":
        print(f"\n  Aggregator: transformer ({n_agg:,} params, "
              f"{args.agg_layers} layers, {args.agg_heads} heads)")
    elif args.aggregator == "lastk":
        print(f"\n  Aggregator: lastk (non-parametric, last_k={args.last_k})")
    else:
        print(f"\n  Aggregator: mean (non-parametric, {n_agg:,} params)")
    print(f"  Probe: {n_probe:,} params")
    if use_lora:
        n_lora = sum(p.numel() for p in model.lora_parameters())
        print(f"  LoRA: {n_lora:,} params (rank={args.lora_rank})")

    # DDP lora: encode→aggregate→probe 를 한 forward 로 묶어 grad all-reduce 등록.
    # wrap 전에 aggregator/probe 를 device 로 옮긴다(DDP 가 rank0 파라미터 broadcast).
    # linear_probe DDP 는 rank0 단독 학습이라 wrap 불필요(ddp_module=None).
    # 단일 GPU 면 ddp_module=None → train_model 이 기존 직접 호출 경로를 탄다(불변).
    agg_ddp = None
    if use_ddp and use_lora:
        aggregator = aggregator.to(device)
        probe = probe.to(device)
        agg_ddp = wrap_aggregator_ddp(model.model, aggregator, probe)

    # ── 학습 ──
    print(f"\nTraining ({args.mode})...")
    train_losses = train_model(
        model, aggregator, probe, train_patients,
        epochs=args.epochs, lr=args.lr, device=device,
        patch_size=patch_size, max_windows=args.max_windows,
        batch_size=args.batch_size, use_lora=use_lora,
        ddp_module=agg_ddp, cached_reprs_override=cached_train_override,
    )

    # ── DDP lora: 학습 종료 동기화 후 non-rank0 종료 (평가/저장은 rank0 전담).
    # (linear_probe DDP 는 추출 직후 non-rank0 가 이미 종료했다.) ──
    if use_ddp and use_lora:
        import torch.distributed as dist
        dist.barrier()
        if not is_main():
            dist.destroy_process_group()
            return

    # ── 평가 ──
    print("\nEvaluating...")
    metrics = evaluate_model(
        model, aggregator, probe, test_patients,
        device=device, patch_size=patch_size,
        max_windows=args.max_windows,
        precomputed_reprs=test_precomputed,
    )

    y_true = metrics.pop("y_true")
    y_score = metrics.pop("y_score")
    # patient-level grouping id (test_patients 순서 = 예측 순서). 환자당 1 예측이라
    # bootstrap 은 이미 patient-level 이지만, run_eval 통합 인터페이스용으로 저장.
    patient_ids = [str(p["subject_id"]) for p in test_patients]

    print(f"\n{'=' * 60}")
    print(f"  Cardiac Arrest Outcome — {args.mode} ({args.aggregator} aggregator)")
    print(f"{'=' * 60}")
    print(f"  AUROC:       {metrics['auroc']:.4f}")
    print(f"  AUPRC:       {metrics['auprc']:.4f}")
    print(f"  F1 (macro):  {metrics['f1_macro']:.4f}")
    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  Prevalence:  {metrics['prevalence']:.3f} "
          f"({metrics['n_positive']}/{metrics['n_total']})")
    print(f"{'=' * 60}")

    # n_folds>1 이면 fold suffix 를 json·png 에 붙여 같은 out-dir 동시/순차 5-fold
    # 실행 시 torn-write·덮어쓰기를 막는다(.npz 와 동일 규칙). single split 은 기존명.
    fold_suffix = f"_fold{args.fold}" if int(args.n_folds) > 1 else ""
    roc_path = out_dir / f"cardiac_arrest_roc_{args.mode}{fold_suffix}.png"
    plot_roc_curve(y_true, y_score, roc_path,
                   title=f"Cardiac Arrest Outcome — {args.mode} ROC")
    print(f"\nROC curve: {roc_path}")

    results = {
        **metrics,
        "y_true": y_true.tolist(),
        "y_score": y_score.tolist(),
        "patient_ids": patient_ids,
        "train_losses": train_losses,
        "config": {
            "task": "scope_cardiac_arrest_outcome",
            "mode": args.mode,
            "aggregation": "transformer",
            "agg_layers": args.agg_layers,
            "agg_heads": args.agg_heads,
            "max_windows": args.max_windows,
            "data_path": args.data_path,
            "epochs": args.epochs, "lr": args.lr,
        },
    }
    results_path = out_dir / f"cardiac_arrest_results_{args.mode}{fold_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results: {results_path}")

    npz_path = dump_fold_predictions(
        out_dir, task="cardiac_arrest_outcome", fold_idx=args.fold, n_folds=args.n_folds,
        y_true=y_true, y_score=y_score, patient_ids=patient_ids,
    )
    print(f"Fold predictions: {npz_path}")

    # DDP: rank0 의 프로세스 그룹 정리(non-rank0 는 학습 직후 이미 정리·종료).
    if use_ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
