# -*- coding: utf-8 -*-
"""Arrhythmia multi-class end-to-end 1D-CNN baseline (VitalDB, N-class).

CARMEN(frozen / LoRA) 과 **같은 prepared 데이터·5-fold·평가** 위에서 사전학습 없이
task 에 처음부터 지도학습하는 1D-ResNet(CrossEntropy). arrhythmia 는 segment-level
multi-class(현재 4-class: NSR/AF/VA/BradyCond)라 binary CNN(ioh_cnn/arrest_cnn)과 달리
n_classes head + CrossEntropy 를 쓴다. class 수·이름은 prepared metadata 에서 읽어
source-agnostic.

데이터: arrhythmia prepare_data_vitaldb 산출물 (단일 .pt/fold: train/val/test, 각
signals(dict)/labels(long)/patients). 프로토콜: 매 epoch val macro-AUROC best-ckpt →
복원 후 test 1회. fold JSON 은 arrhythmia run.py 와 동일 스키마(y_true/y_probs/
patient_ids/per_class_auroc/confusion_matrix/...) → 같은 aggregator 로 3-way 비교.

사용법:
    python -m downstream.baselines.arrhythmia_cnn \
        --data-path .../arrhythmia_ecg_ppg --input-signals ecg ppg \
        --n-folds 5 --fold 0 --epochs 100 --patience 10 \
        --lr 1e-3 --batch-size 128 --device cuda \
        --out-dir .../result/main/arrhythmia_cnn
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
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix,
)

from downstream.baselines.ioh_cnn import ResNet1D


# ── 데이터 (arrhythmia 단일 .pt/fold) ─────────────────────────


def _split_xy(split, input_signals):
    sig = split["signals"]
    keys = [s for s in input_signals if s in sig]
    if not keys:
        keys = list(sig.keys())
    X = torch.stack([torch.as_tensor(sig[k], dtype=torch.float32) for k in keys], dim=1)
    X = torch.nan_to_num(X, nan=0.0)  # (N, C, L)
    y = torch.as_tensor(split["labels"], dtype=torch.long).reshape(-1)
    pats = split.get("patients", None)
    pats = [str(p) for p in pats] if pats is not None else [str(i) for i in range(len(y))]
    return X, y, pats, keys


def _load(args):
    p = Path(args.data_path)
    if not p.exists():
        cand = Path(f"{args.data_path}_fold{int(args.fold)}.pt")
        if cand.exists():
            p = cand
    if not p.exists():
        print(f"ERROR: data not found: {args.data_path} (또는 _fold{args.fold}.pt)",
              file=sys.stderr)
        sys.exit(1)
    data = torch.load(p, weights_only=False)
    meta = data.get("metadata", {})
    classes = meta.get("class_names") or meta.get("classes")
    Xtr, ytr, ptr, keys = _split_xy(data["train"], args.input_signals)
    val = data.get("val")
    if val is not None and len(val.get("labels", [])) > 0:
        Xva, yva, pva, _ = _split_xy(val, args.input_signals)
    else:  # legacy fallback: train 20% split
        rng = np.random.default_rng(args.val_split_seed)
        idx = rng.permutation(len(ytr)); n_val = max(1, int(0.2 * len(ytr)))
        vi, ti = idx[:n_val], idx[n_val:]
        Xva, yva = Xtr[vi], ytr[vi]; Xtr, ytr = Xtr[ti], ytr[ti]
    Xte, yte, pte, _ = _split_xy(data["test"], args.input_signals)
    if classes is None:
        classes = [f"c{i}" for i in range(int(ytr.max().item()) + 1)]
    return (Xtr, ytr), (Xva, yva), (Xte, yte, pte), keys, list(classes)


# ── 학습/평가 ─────────────────────────────────────────────────


def _iter(X, y, bs, shuffle, device):
    order = torch.randperm(len(y)) if shuffle else torch.arange(len(y))
    for i in range(0, len(y), bs):
        idx = order[i:i + bs]
        yield X[idx].to(device), y[idx].to(device)


@torch.no_grad()
def _probs(model, X, bs, device):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        logits = model(X[i:i + bs].to(device))
        out.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0,))


def _macro_auroc(yt, yp):
    return float(roc_auc_score(yt, yp, multi_class="ovr", average="macro"))


def _metrics(yt, yp, classes):
    K = len(classes)
    pred = yp.argmax(1)
    per_auroc, per_stats = {}, {}
    for c in range(K):
        bt = (yt == c).astype(int)
        per_auroc[classes[c]] = (float(roc_auc_score(bt, yp[:, c]))
                                 if 0 < bt.sum() < len(bt) else float("nan"))
        bp = (pred == c).astype(int)
        tp = int((bp & bt).sum()); fn = int(((1 - bp) & bt).sum())
        tn = int(((1 - bp) & (1 - bt)).sum()); fp = int((bp & (1 - bt)).sum())
        per_stats[classes[c]] = {
            "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        }
    cm = confusion_matrix(yt, pred, labels=list(range(K)))
    return {
        "accuracy": float(accuracy_score(yt, pred)),
        "macro_f1": float(f1_score(yt, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, pred, average="weighted", zero_division=0)),
        "per_class_auroc": per_auroc,
        "per_class_stats": per_stats,
        "confusion_matrix": cm.tolist(),
        "class_distribution": {classes[i]: int((yt == i).sum()) for i in range(K)},
    }


def main():
    p = argparse.ArgumentParser(description="Arrhythmia multi-class 1D-CNN baseline")
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--input-signals", type=str, nargs="+", default=["ecg", "ppg"])
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--val-split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--patience", type=int, default=10, help=">0: val macro-AUROC 무개선 patience epoch 후 early-stop")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default=".")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    (Xtr, ytr), (Xva, yva), (Xte, yte, pte), keys, classes = _load(args)
    K = len(classes)
    ytr_np, yva_np, yte_np = ytr.numpy(), yva.numpy(), yte.numpy()
    print(f"[ARR-CNN] fold={args.fold}/{args.n_folds} signals={keys} classes={classes} "
          f"train={len(ytr)} val={len(yva)} test={len(yte)}", flush=True)

    model = ResNet1D(in_ch=len(keys), width=args.width, n_classes=K).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # class imbalance: inverse-freq weight (CrossEntropy)
    counts = np.bincount(ytr_np, minlength=K).astype(float)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)) / K, dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=w)

    best_auroc, best_state, best_ep, no_imp = -1.0, None, -1, 0
    train_losses, val_aurocs = [], []
    for ep in range(args.epochs):
        model.train(); ep_loss, nb = 0.0, 0
        for xb, yb in _iter(Xtr, ytr, args.batch_size, True, device):
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss.item()); nb += 1
        train_losses.append(ep_loss / max(nb, 1))
        va = _probs(model, Xva, args.batch_size, device)
        try:
            auroc = _macro_auroc(yva_np, va)
        except Exception:
            auroc = float("nan")
        val_aurocs.append(float(auroc))
        if auroc > best_auroc:
            best_auroc, best_ep, no_imp = auroc, ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_imp += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep{ep+1}/{args.epochs} val_macroAUROC={auroc:.4f} "
                  f"(best={best_auroc:.4f}@{best_ep+1})", flush=True)
        if args.patience > 0 and no_imp >= args.patience:
            print(f"  [early-stop] ep{ep+1}", flush=True); break

    if best_state is not None:
        model.load_state_dict(best_state)
    ys = _probs(model, Xte, args.batch_size, device)
    m = _metrics(yte_np, ys, classes)
    macro_auroc = float(np.nanmean(list(m["per_class_auroc"].values())))
    print(f"\n[ARR-CNN] TEST macro-AUROC={macro_auroc:.4f} acc={m['accuracy']:.4f} "
          f"macroF1={m['macro_f1']:.4f}", flush=True)
    for c, a in m["per_class_auroc"].items():
        print(f"    {c}: AUROC={a:.4f}", flush=True)

    fold_json = {
        **m,
        "y_true": yte_np.astype(int).tolist(),
        "y_probs": ys.astype(float).tolist(),
        "patient_ids": [str(x) for x in pte],
        "val_auroc_best": float(best_auroc),
        "best_epoch": int(best_ep),
        "train_losses": train_losses,
        "val_aurocs": val_aurocs,
        "config": {
            "model": "resnet1d", "n_classes": K, "classes": classes,
            "width": args.width, "input_signals": keys,
            "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
            "patience": args.patience,
        },
    }
    with open(out_dir / f"fold{args.fold}.json", "w") as f:
        json.dump(fold_json, f, indent=2)
    print(f"  saved fold{args.fold}.json → {out_dir}", flush=True)


if __name__ == "__main__":
    main()
