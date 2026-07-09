# -*- coding: utf-8 -*-
"""외부 FM baseline 공용 학습/평가 드라이버 (frozen encoder + linear probe).

CNN baseline(:mod:`downstream.baselines.ioh_cnn`)의 ``run_cnn_baseline`` 과 **같은
계약**(prepared 데이터 로더·평가 유틸·산출물 스키마)을 따르되, "from-scratch ResNet1D
end-to-end 학습" 대신 "**frozen 외부 FM → feature 추출(1회) → LinearProbe 학습**" 을
수행한다. 산출물(preds_fold{f}.npz + fold{f}.json)이 CARMEN·CNN baseline 과 동일해
``downstream.run_eval`` / ``aggregator`` 로 구분 없이 집계된다.

재사용:
  - :func:`downstream.baselines.ioh_cnn._load_split_ram` — prepared chunk 스트리밍 로더
  - :func:`downstream.baselines.ioh_cnn._youden_threshold` — val Youden threshold
  - :class:`downstream.model_wrapper.LinearProbe` — LayerNorm→Dropout→Linear head
  - :mod:`downstream.metrics` / :func:`downstream._eval_utils.dump_fold_predictions`

⚠ CARMEN 본코드(module/model/train/collate/downstream run.py)는 건드리지 않는다.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from downstream.baselines.ioh_cnn import _load_split_ram, _youden_threshold
from downstream.model_wrapper import LinearProbe
from downstream._eval_utils import dump_fold_predictions
from downstream.metrics import (
    bootstrap_ci,
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)


@torch.no_grad()
def _extract_features(encoder, chunks, keys, feat_batch: int) -> np.ndarray:
    """per-chunk fp16 (n,C,L) 리스트 → frozen 임베딩 (N, d) np.float32.

    chunk·row 순서를 보존해 labels/case_ids 정렬과 일치. encoder.encode 가 내부에서
    채널선택·세그먼트·리샘플·정규화·forward·세그먼트평균을 처리한다.
    """
    feats: list[torch.Tensor] = []
    for X in chunks:  # X: (n, C, L) fp16 (RAM 상주)
        n = int(X.shape[0])
        for i in range(0, n, feat_batch):
            xb = X[i : i + feat_batch].to(torch.float32)  # (b, C, L)
            e = encoder.encode(xb, keys)  # (b, d) on device
            feats.append(e.detach().cpu())
    if not feats:
        return np.zeros((0, int(getattr(encoder, "embed_dim", 0))), dtype=np.float32)
    return torch.cat(feats, dim=0).numpy().astype(np.float32)


@torch.no_grad()
def _probe_scores(probe, X: np.ndarray, batch: int, device) -> np.ndarray:
    """LinearProbe 로 sigmoid 확률 (N,) 반환 (X 행 순서 유지)."""
    probe.eval()
    out: list[torch.Tensor] = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i : i + batch], dtype=torch.float32, device=device)
        logits = probe(xb).squeeze(-1)
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy() if out else np.array([])


def run_fm_baseline(args, encoder, task_name: str, tag: str, id_fields=("case_ids",)) -> None:
    """frozen FM feature → LinearProbe → val best-ckpt → test 1회 → npz + fold JSON.

    ``encoder`` 는 :class:`~downstream.baselines.fm.encoders.base.FMEncoder` 인스턴스.
    ``task_name`` = npz task 태그(fold 간 일관되면 됨), ``tag`` = 로그 prefix,
    ``id_fields`` = 환자 grouping id 우선순위(CARMEN OOF bootstrap grouping 정합).
    """
    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    load_fold = int(args.fold) if int(args.n_folds) > 1 else None

    # ── prepared 데이터 로드 (CNN baseline 과 동일 로더) ──
    keys, tr_ch, tr_y, tr_id = _load_split_ram(
        args.data_path, load_fold, "train", args.input_signals, id_fields
    )
    if not keys:
        raise FileNotFoundError(
            f"no train chunks/signals for {args.data_path} (fold={load_fold})"
        )
    kva, va_ch, va_y, va_id = _load_split_ram(
        args.data_path, load_fold, "val", args.input_signals, id_fields
    )
    kte, te_ch, te_y, te_id = _load_split_ram(
        args.data_path, load_fold, "test", args.input_signals, id_fields
    )

    # ── frozen feature 추출 (1회) ──
    print(f"[{tag}] extracting frozen features (encoder={encoder.name}, d={encoder.embed_dim}) ...", flush=True)
    Xtr = _extract_features(encoder, tr_ch, keys, args.feat_batch)
    Xte = _extract_features(encoder, te_ch, keys, args.feat_batch)
    ytr = np.asarray(tr_y, dtype=np.float32)
    yte = np.asarray(te_y, dtype=np.float32)
    case_te = te_id
    if kva:  # val chunk 존재
        Xva = _extract_features(encoder, va_ch, keys, args.feat_batch)
        yva = np.asarray(va_y, dtype=np.float32)
    else:  # legacy: train 20% 동적 split (CNN baseline 과 동일)
        rng = np.random.default_rng(args.val_split_seed)
        perm = rng.permutation(len(Xtr))
        n_val = max(1, int(0.2 * len(Xtr)))
        vi, ti = perm[:n_val], perm[n_val:]
        Xva, yva = Xtr[vi], ytr[vi]
        Xtr, ytr = Xtr[ti], ytr[ti]

    print(f"[{tag}] fold={args.fold}/{args.n_folds} signals={keys}  "
          f"train={len(ytr)}({int(ytr.sum())} pos) val={len(yva)} "
          f"test={len(yte)}({int(yte.sum())} pos)  feat_dim={Xtr.shape[1] if len(Xtr) else encoder.embed_dim}",
          flush=True)

    # ── LinearProbe 학습 (encoder frozen, feature 캐시 위에서만) ──
    probe = LinearProbe(encoder.embed_dim, n_classes=1, dropout_p=args.dropout).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)
    n_pos = float(ytr.sum()); n_neg = float(len(ytr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.as_tensor(ytr, dtype=torch.float32)
    best_auroc, best_state, best_ep, no_imp = -1.0, None, -1, 0
    train_losses: list[float] = []
    val_aurocs: list[float] = []
    rng = np.random.default_rng(args.val_split_seed)
    for ep in range(args.epochs):
        probe.train()
        perm = rng.permutation(len(Xtr_t))
        ep_loss, nb = 0.0, 0
        for i in range(0, len(perm), args.batch_size):
            bidx = perm[i : i + args.batch_size]
            xb = Xtr_t[bidx].to(device, non_blocking=True)
            yb = ytr_t[bidx].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(probe(xb).squeeze(-1), yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()); nb += 1
        train_losses.append(ep_loss / max(nb, 1))
        va = _probe_scores(probe, Xva, args.batch_size, device)
        try:
            auroc = compute_auroc(yva.astype(int), va)
        except Exception:
            auroc = float("nan")
        val_aurocs.append(float(auroc))
        if auroc > best_auroc:
            best_auroc, best_ep, no_imp = auroc, ep, 0
            best_state = copy.deepcopy(probe.state_dict())
        else:
            no_imp += 1
        print(f"  ep{ep+1}/{args.epochs} val_auroc={auroc:.4f} (best={best_auroc:.4f}@{best_ep+1})", flush=True)
        if args.patience > 0 and no_imp >= args.patience:
            print(f"  [early-stop] ep{ep+1}", flush=True); break

    if best_state is not None:
        probe.load_state_dict(best_state)

    # ── test 1회 + val Youden threshold ──
    ys = _probe_scores(probe, Xte, args.batch_size, device)
    va_scores = _probe_scores(probe, Xva, args.batch_size, device)
    thr = _youden_threshold(yva.astype(int), va_scores)
    y_pred = (ys >= thr).astype(int)
    yte_i = yte.astype(int)

    auroc = compute_auroc(yte_i, ys); auprc = compute_auprc(yte_i, ys)
    f1 = compute_f1(yte_i, y_pred, average="macro")
    ss = compute_sensitivity_specificity(yte_i, y_pred)
    auroc_ci = bootstrap_ci(compute_auroc, yte_i, ys, n_iter=args.bootstrap_iters)
    auprc_ci = bootstrap_ci(compute_auprc, yte_i, ys, n_iter=args.bootstrap_iters)
    f1_ci = bootstrap_ci(
        lambda t, p: compute_f1(t, p, average="macro"), yte_i, y_pred, n_iter=args.bootstrap_iters
    )
    print(f"\n[{tag}] TEST AUROC={auroc:.4f} [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]  "
          f"AUPRC={auprc:.4f} [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]  "
          f"prevalence={yte_i.mean():.3f} ({yte_i.sum()}/{len(yte_i)})", flush=True)

    # ── 산출물: CARMEN run.py 와 동일 스키마 ──
    dump_fold_predictions(
        out_dir, task=task_name, fold_idx=args.fold,
        n_folds=args.n_folds, y_true=yte_i, y_score=ys, patient_ids=case_te,
    )
    fold_json = {
        "auroc": float(auroc), "auprc": float(auprc), "f1": float(f1),
        "optimal_threshold": float(thr),
        "sensitivity": float(ss["sensitivity"]), "specificity": float(ss["specificity"]),
        "n_total": int(len(yte_i)), "n_positive": int(yte_i.sum()),
        "prevalence": float(yte_i.mean()),
        "y_true": yte_i.tolist(), "y_score": ys.astype(float).tolist(),
        "patient_ids": [str(c) for c in case_te],
        "val_auroc_best": float(best_auroc), "best_epoch": int(best_ep),
        "auroc_ci": [float(auroc_ci[0]), float(auroc_ci[1])],
        "auprc_ci": [float(auprc_ci[0]), float(auprc_ci[1])],
        "f1_ci": [float(f1_ci[0]), float(f1_ci[1])],
        "train_losses": train_losses, "val_aurocs": val_aurocs,
        "config": {
            "mode": task_name,
            "model": encoder.name,
            "eval": "frozen_linear_probe",
            "input_signals": keys,
            "native_sr": encoder.native_sr,
            "seg_sec": encoder.seg_sec,
            "max_segments": encoder.max_segments,
            "embed_dim": encoder.embed_dim,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "dropout": args.dropout, "patience": args.patience,
            "bootstrap_iters": args.bootstrap_iters, "val_split_seed": args.val_split_seed,
        },
    }
    with open(out_dir / f"fold{args.fold}.json", "w") as f:
        json.dump(fold_json, f, indent=2)
    print(f"  saved preds_fold{args.fold}.npz + fold{args.fold}.json → {out_dir}", flush=True)
