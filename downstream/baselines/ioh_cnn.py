# -*- coding: utf-8 -*-
"""IOH end-to-end CNN baseline — VitalDB IOH 문헌(Lee et al. 등) 스타일 비교군.

CARMEN(frozen linear-probe / LoRA) 과 **같은 prepared 데이터·같은 5-fold·같은 평가**
위에서, 사전학습 없이 **task 에 처음부터 지도학습하는 1D-CNN** 을 돌려 직접 비교한다.
SOTA 비교의 공정한 짝(end-to-end supervised)을 우리 코호트에서 재현하는 용도.

⚠ CARMEN 본코드(module/model/train/collate/downstream run.py)는 건드리지 않는다.
   이 파일은 prepared 데이터 로더(_save_utils)와 평가 유틸(_eval_utils, metrics)만
   재사용하는 **독립 스크립트**다.

입력: 우리 hypotension prepare_data 산출물(per-(fold,split)[_chunk] prefix).
모델: 1D ResNet (multi-channel waveform → binary IOH).
프로토콜: 매 epoch val AUROC best-ckpt → 복원 후 test 1회 → preds_fold{f}.npz 저장
          (CARMEN run.py 와 동일 → aggregator 로 동일 집계 가능).

사용법:
    python -m downstream.baselines.ioh_cnn \
        --data-path .../hypotension_ecg_ppg_abp_w300s_h5min \
        --input-signals ecg ppg abp --n-folds 5 --fold 0 \
        --epochs 100 --lr 1e-3 --batch-size 128 --device cuda \
        --out-dir .../result/main/hypotension_cnn/ecg_ppg_abp_w300s_h5min
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from downstream._save_utils import load_prepared_split_chunked
from downstream._eval_utils import dump_fold_predictions
from downstream.metrics import (
    compute_auroc,
    compute_auprc,
    compute_f1,
    compute_sensitivity_specificity,
)


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Val 에서 Youden's J(=tpr-fpr) 최대 지점의 threshold (test leakage 회피)."""
    from sklearn.metrics import roc_curve

    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, y_score)
    return float(thr[int(np.argmax(tpr - fpr))])


# ── 데이터 ────────────────────────────────────────────────────


def _split_to_xy(split, input_signals):
    """prepared split → (X (N,C,L) float32 cpu, y (N,) float32, case_ids list)."""
    sig = split["signals"]
    keys = [s for s in input_signals if s in sig]
    if not keys:
        raise ValueError(f"none of {input_signals} in data signals {list(sig.keys())}")
    # 각 stype: (N, L) tensor. 채널축으로 stack → (N, C, L).
    chans = [torch.as_tensor(sig[k], dtype=torch.float32) for k in keys]
    X = torch.stack(chans, dim=1)
    X = torch.nan_to_num(X, nan=0.0)  # gap 은 0 (prepare 단계서 이미 0-fill 이지만 안전)
    y = torch.as_tensor(split["labels"], dtype=torch.float32).reshape(-1)
    case_ids = split.get("case_ids", None)
    if case_ids is None:
        case_ids = [str(i) for i in range(len(y))]
    else:
        case_ids = [str(c) for c in (case_ids.tolist() if hasattr(case_ids, "tolist") else case_ids)]
    return X, y, keys, case_ids


def _load(args):
    load_fold = int(args.fold) if int(args.n_folds) > 1 else None
    data = load_prepared_split_chunked(args.data_path, fold=load_fold)
    Xtr, ytr, keys, _ = _split_to_xy(data["train"], args.input_signals)
    val = data.get("val")
    if val is not None:
        Xva, yva, _, _ = _split_to_xy(val, args.input_signals)
    else:  # legacy: train 20% 동적 split
        rng = np.random.default_rng(args.val_split_seed)
        idx = rng.permutation(len(ytr))
        n_val = max(1, int(0.2 * len(ytr)))
        vi, ti = idx[:n_val], idx[n_val:]
        Xva, yva = Xtr[vi], ytr[vi]
        Xtr, ytr = Xtr[ti], ytr[ti]
    Xte, yte, _, case_te = _split_to_xy(data["test"], args.input_signals)
    return (Xtr, ytr), (Xva, yva), (Xte, yte, case_te), keys


# ── 모델: 1D ResNet ───────────────────────────────────────────


class _Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.act = nn.ReLU(inplace=True)
        self.down = (
            nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(cout))
            if (stride != 1 or cin != cout) else nn.Identity()
        )

    def forward(self, x):
        r = self.down(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.act(x + r)


class ResNet1D(nn.Module):
    """waveform (B,C,L) → IOH logit. 큰 stride stem 으로 긴 시퀀스 다운샘플."""

    def __init__(self, in_ch: int, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(width), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        chans = [width, width, width * 2, width * 4, width * 8]
        blocks = []
        for i in range(4):
            stride = 1 if i == 0 else 2
            blocks += [_Block(chans[i], chans[i + 1], stride),
                       _Block(chans[i + 1], chans[i + 1], 1)]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Linear(chans[-1], 1))

    def forward(self, x):  # (B, C, L) → (B, 1)
        return self.head(self.blocks(self.stem(x)))


# ── 학습/평가 ─────────────────────────────────────────────────


def _iter_batches(X, y, bs, shuffle, device):
    n = len(y)
    order = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, bs):
        idx = order[i:i + bs]
        yield X[idx].to(device), y[idx].to(device)


@torch.no_grad()
def _scores(model, X, bs, device):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        out.append(torch.sigmoid(model(X[i:i + bs].to(device)).squeeze(-1)).cpu())
    return torch.cat(out).numpy() if out else np.array([])


def main():
    p = argparse.ArgumentParser(description="IOH end-to-end 1D-CNN baseline")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--input-signals", type=str, nargs="+", default=["ecg", "ppg", "abp"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--val-split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--patience", type=int, default=0, help=">0 이면 val best 미개선 patience epoch 후 early-stop")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default=".")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    (Xtr, ytr), (Xva, yva), (Xte, yte, case_te), keys = _load(args)
    print(f"[IOH-CNN] fold={args.fold}/{args.n_folds}  signals={keys}  "
          f"train={len(ytr)}({int(ytr.sum())} pos) val={len(yva)} test={len(yte)}({int(yte.sum())} pos)",
          flush=True)

    model = ResNet1D(in_ch=len(keys), width=args.width).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # class imbalance: pos_weight = n_neg/n_pos (없으면 1)
    n_pos = float(ytr.sum()); n_neg = float(len(ytr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auroc, best_state, best_ep, no_imp = -1.0, None, -1, 0
    for ep in range(args.epochs):
        model.train()
        for xb, yb in _iter_batches(Xtr, ytr, args.batch_size, True, device):
            loss = crit(model(xb).squeeze(-1), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        va = _scores(model, Xva, args.batch_size, device)
        try:
            auroc = compute_auroc(yva.numpy().astype(int), va)
        except Exception:
            auroc = float("nan")
        if auroc > best_auroc:
            best_auroc, best_ep, no_imp = auroc, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep+1}/{args.epochs} val_auroc={auroc:.4f} (best={best_auroc:.4f}@{best_ep+1})", flush=True)
        if args.patience > 0 and no_imp >= args.patience:
            print(f"  [early-stop] ep{ep+1}", flush=True); break

    if best_state is not None:
        model.load_state_dict(best_state)
    ys = _scores(model, Xte, args.batch_size, device)
    yt = yte.numpy().astype(int)
    # threshold-dependent 지표는 val Youden threshold 로(test leakage 회피).
    va_scores = _scores(model, Xva, args.batch_size, device)
    thr = _youden_threshold(yva.numpy().astype(int), va_scores)
    y_pred = (ys >= thr).astype(int)

    auroc = compute_auroc(yt, ys); auprc = compute_auprc(yt, ys)
    f1 = compute_f1(yt, y_pred, average="macro")
    ss = compute_sensitivity_specificity(yt, y_pred)
    print(f"\n[IOH-CNN] TEST  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
          f"prevalence={yt.mean():.3f} ({yt.sum()}/{len(yt)})", flush=True)

    # ── preds .npz (run_eval 집계용, CARMEN run.py 와 동일) ──
    dump_fold_predictions(
        out_dir, task="hypotension_cnn", fold_idx=args.fold,
        n_folds=args.n_folds, y_true=yt, y_score=ys, patient_ids=case_te,
    )
    # ── fold JSON (CARMEN fold{f}.json 과 동일 스키마 → 같은 방식으로 집계 가능) ──
    fold_json = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "optimal_threshold": float(thr),
        "sensitivity": float(ss["sensitivity"]),
        "specificity": float(ss["specificity"]),
        "n_total": int(len(yt)),
        "n_positive": int(yt.sum()),
        "prevalence": float(yt.mean()),
        "y_true": yt.astype(int).tolist(),
        "y_score": ys.astype(float).tolist(),
        "patient_ids": [str(c) for c in case_te],
        "val_auroc_best": float(best_auroc),
        "best_epoch": int(best_ep),
        "config": {
            "model": "resnet1d",
            "width": args.width,
            "input_signals": keys,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
        },
    }
    json_path = out_dir / f"fold{args.fold}.json"
    with open(json_path, "w") as f:
        json.dump(fold_json, f)
    print(f"  saved preds_fold{args.fold}.npz + fold{args.fold}.json → {out_dir}",
          flush=True)


if __name__ == "__main__":
    main()
