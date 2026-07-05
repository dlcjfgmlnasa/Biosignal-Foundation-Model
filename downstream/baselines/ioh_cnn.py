# -*- coding: utf-8 -*-
"""IOH end-to-end CNN baseline — VitalDB IOH 문헌(Lee et al. 등) 스타일 비교군.

CARMEN(frozen linear-probe / LoRA) 과 **같은 prepared 데이터·같은 5-fold·같은 평가**
위에서, 사전학습 없이 **task 에 처음부터 지도학습하는 1D-CNN** 을 돌려 직접 비교한다.
SOTA 비교의 공정한 짝(end-to-end supervised)을 우리 코호트에서 재현하는 용도.

⚠ CARMEN 본코드(module/model/train/collate/downstream run.py)는 건드리지 않는다.
   이 파일은 prepared 데이터 로더(_save_utils)와 평가 유틸(_eval_utils, metrics)만
   재사용하는 **독립 스크립트**다.

메모리(OOM) 하드닝 — RAM-resident fp16 스트리밍
------------------------------------------------
기존 구현은 ``load_prepared_split_chunked`` 로 split 의 모든 chunk 를 concat 한 뒤
``torch.stack`` 으로 (N,C,L) **fp32** 텐서를 세 split 동시에 만들어(원본 dict 도 동시
참조) RAM 피크가 폭발했다 → 대형 IOH 코호트에서 커널 OOM-kill.

본 구현은 ``iter_prepared_split_chunks`` 로 chunk 를 하나씩 스트리밍해 각 chunk 를
**fp16 (n,C,L) 텐서로 RAM 에 리스트 보관**한다(원본 즉시 해제, fp32 미승격, 2x concat
없음 → 피크 = split 1개 분량의 fp16). 학습·평가는 그 리스트를 인덱싱하는
:class:`_RamWindowDataset` + ``DataLoader`` 로 배치를 만들고, fp32 캐스팅은 **배치를
GPU 로 올릴 때만** 수행한다. 로컬 디스크가 없어도 되고, 데이터가 RAM 에 있어 학습 중
디스크 I/O 가 없다(GPU 가 데이터 대기로 노는 현상 방지).

⚠ RAM 사용량 ≈ split 전체 fp16 (예: 300s×100Hz×3ch, ~32만 window ≈ 57 GB/fold).
   fold 를 GPU 별로 **동시 실행하면 그만큼 RAM 이 배로** 든다(run.sh 의 NUM_GPUS).
   컨테이너 RAM 한도에 맞춰 동시 fold 수를 정할 것.

입력: 우리 hypotension prepare_data 산출물(per-(fold,split)[_chunk] prefix).
모델: 1D ResNet (multi-channel waveform → binary IOH).
프로토콜: 매 epoch val AUROC best-ckpt → 복원 후 test 1회 → preds_fold{f}.npz 저장
          (CARMEN run.py 와 동일 → aggregator 로 동일 집계 가능).

사용법:
    python -m downstream.baselines.ioh_cnn \
        --data-path .../hypotension_ecg_ppg_abp_w300s_h5min \
        --input-signals ecg ppg abp --n-folds 5 --fold 0 \
        --epochs 100 --lr 1e-3 --batch-size 128 --device cuda \
        --num-workers 4 \
        --out-dir .../result/main/hypotension_cnn/ecg_ppg_abp_w300s_h5min
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from downstream._save_utils import iter_prepared_split_chunks
from downstream._eval_utils import dump_fold_predictions
from downstream.metrics import (
    bootstrap_ci,
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


# ── 데이터: RAM-resident fp16 스트리밍 (OOM 하드닝, 디스크 불필요) ──────


def _load_split_ram(data_path, fold, split, input_signals, id_fields=("case_ids",)):
    """한 split 의 chunk 를 하나씩 스트리밍 → per-chunk fp16 (n,C,L) 텐서 리스트.

    원본 numpy 는 즉시 해제, fp32 승격/2x concat 없음 → 피크 = chunk 1개 + 누적 fp16.
    반환: (keys, chunks(list[Tensor fp16]), labels(np float32), ids(list[str])).
    chunk 가 없으면 keys=[] 로 반환.

    ``id_fields`` 는 grouping id 를 찾을 payload 키의 **우선순위** 다. IOH 는
    ``("case_ids",)`` (기존과 동일), ICH/arrest 는 ``("subject_ids","case_ids")`` 로
    환자단위 subject_ids 를 우선 사용한다 (CARMEN run.py 의 OOF bootstrap grouping 정합).
    """
    keys: list[str] | None = None
    chunks: list[torch.Tensor] = []
    labels: list[np.ndarray] = []
    case_ids: list[str] = []

    for ci, payload in enumerate(iter_prepared_split_chunks(data_path, fold, split)):
        sig = payload.get("signals") if isinstance(payload, dict) else None
        if not sig:
            continue
        k = [s for s in input_signals if s in sig]
        if not k:
            raise ValueError(
                f"none of {input_signals} in data signals {list(sig.keys())}"
            )
        if keys is None:
            keys = k
        # 채널축 stack → (n, C, L) fp16. chunk 1개분만 추가로 잡는다.
        chans = [torch.as_tensor(sig[s]).to(torch.float16) for s in keys]
        X = torch.nan_to_num(torch.stack(chans, dim=1), nan=0.0)  # (n, C, L) fp16
        chunks.append(X.contiguous())

        y = np.asarray(payload["labels"]).reshape(-1).astype(np.float32)
        labels.append(y)
        cid = None
        for fld in id_fields:
            v = payload.get(fld)
            if v is not None:
                cid = v
                break
        if cid is None:
            cid = [f"{split}_{ci}_{r}" for r in range(len(y))]
        else:
            cid = [str(c) for c in (cid.tolist() if hasattr(cid, "tolist") else cid)]
        case_ids.extend(cid)

        del chans, sig, payload
        gc.collect()

    if keys is None:
        return [], [], np.array([], dtype=np.float32), []
    lab = np.concatenate(labels) if labels else np.array([], dtype=np.float32)
    return keys, chunks, lab, case_ids


class _RamWindowDataset(Dataset):
    """RAM 에 상주하는 per-chunk fp16 텐서 리스트 위의 flat window 인덱스.

    ``__getitem__`` 은 해당 window 만 fp32 로 캐스팅해 반환(전체 fp32 사본 없음).
    Linux fork(num_workers>0) 에서 chunks 는 copy-on-write 로 공유되어 worker 가
    복제하지 않는다. ``subset`` 으로 index/label/case 부분집합(legacy val-split)만 노출.
    """

    def __init__(self, chunks, labels, case_ids, subset=None):
        self.chunks = chunks
        counts = [int(c.shape[0]) for c in chunks]
        flat = [(ci, r) for ci, n in enumerate(counts) for r in range(n)]
        labels = np.asarray(labels, dtype=np.float32)
        case_ids = list(case_ids) if case_ids is not None else [str(i) for i in range(len(flat))]
        if subset is not None:
            subset = list(subset)
            flat = [flat[i] for i in subset]
            labels = labels[subset]
            case_ids = [case_ids[i] for i in subset]
        self.index = flat
        self.labels = labels
        self.case_ids = case_ids

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i: int):
        ci, r = self.index[i]
        x = self.chunks[ci][r].to(torch.float32)  # (C, L) fp16→fp32 (window 1개만)
        return x, torch.tensor(self.labels[i], dtype=torch.float32)


def _make_datasets(args, id_fields=("case_ids",)):
    """train/val/test Dataset + keys 를 RAM-resident 스트리밍으로 구성."""
    load_fold = int(args.fold) if int(args.n_folds) > 1 else None
    ktr, tr_ch, tr_y, tr_id = _load_split_ram(
        args.data_path, load_fold, "train", args.input_signals, id_fields
    )
    if not ktr:
        raise FileNotFoundError(
            f"no train chunks/signals for {args.data_path} (fold={load_fold})"
        )
    kva, va_ch, va_y, va_id = _load_split_ram(
        args.data_path, load_fold, "val", args.input_signals, id_fields
    )
    kte, te_ch, te_y, te_id = _load_split_ram(
        args.data_path, load_fold, "test", args.input_signals, id_fields
    )

    train_ds = _RamWindowDataset(tr_ch, tr_y, tr_id)
    if kva:  # val chunk 존재
        val_ds = _RamWindowDataset(va_ch, va_y, va_id)
    else:  # legacy: train 20% 동적 split (index 만 나눔 — 데이터 재적재 없음)
        n = len(train_ds)
        rng = np.random.default_rng(args.val_split_seed)
        perm = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        vi, ti = perm[:n_val], perm[n_val:]
        val_ds = _RamWindowDataset(tr_ch, tr_y, tr_id, subset=vi)
        train_ds = _RamWindowDataset(tr_ch, tr_y, tr_id, subset=ti)
    test_ds = _RamWindowDataset(te_ch, te_y, te_id)
    return train_ds, val_ds, test_ds, ktr


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
    """waveform (B,C,L) → logit(s). 큰 stride stem 으로 긴 시퀀스 다운샘플.

    n_classes=1(기본): binary logit (기존 IOH/arrest 등, BCEWithLogits). n_classes>1:
    multiclass logits (arrhythmia 5-class 등, CrossEntropy). 그 외 구조 불변.
    """

    def __init__(self, in_ch: int, width: int = 64, n_classes: int = 1):
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
                                  nn.Linear(chans[-1], n_classes))

    def forward(self, x):  # (B, C, L) → (B, n_classes)
        return self.head(self.blocks(self.stem(x)))


# ── 학습/평가 ─────────────────────────────────────────────────


@torch.no_grad()
def _scores(model, loader, device):
    """loader(shuffle=False) 순서대로 sigmoid 확률 반환 (dataset.labels 와 정렬)."""
    model.eval()
    amp = (device.type == "cuda")
    out = []
    for xb, _ in loader:
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(xb.to(device, non_blocking=True)).squeeze(-1)
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy() if out else np.array([])


def add_common_args(p, default_signals):
    """세 task(IOH/ICH/arrest) CNN baseline 공용 CLI 인자."""
    p.add_argument("--data-path", type=str, required=True)
    p.add_argument("--input-signals", type=str, nargs="+", default=list(default_signals))
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--val-split-seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--patience", type=int, default=0, help=">0 이면 val best 미개선 patience epoch 후 early-stop")
    p.add_argument("--bootstrap-iters", type=int, default=1000, help="per-fold test CI bootstrap iter (CARMEN 동일)")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 수 (Linux fork 시 RAM 데이터 COW 공유)")
    p.add_argument("--cache-dir", type=str, default="", help="(deprecated no-op) RAM-resident 방식이라 로컬 캐시 불필요")
    p.add_argument("--keep-cache", action="store_true", help="(deprecated no-op)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default=".")
    return p


def run_cnn_baseline(args, task_name, tag, id_fields=("case_ids",)):
    """1D-ResNet end-to-end 지도학습 → val best-ckpt → test 1회 → npz + fold JSON.

    IOH/ICH/arrest 공용. ``task_name`` 은 npz task 태그, ``tag`` 는 로그 prefix,
    ``id_fields`` 는 환자 grouping id 우선순위(IOH=case_ids, ICH/arrest=subject_ids).
    """
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    n_gpus = torch.cuda.device_count() if device.type == "cuda" else 1
    # 입력 길이가 고정(L)이라 cudnn 이 최적 conv 알고리즘을 캐시 → conv 가속.
    torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, keys = _make_datasets(args, id_fields)
    ytr = train_ds.labels.astype(int)
    yva = val_ds.labels.astype(int)
    yte = test_ds.labels.astype(int)
    case_te = test_ds.case_ids
    print(f"[{tag}] fold={args.fold}/{args.n_folds}  signals={keys}  "
          f"train={len(ytr)}({int(ytr.sum())} pos) val={len(yva)} "
          f"test={len(yte)}({int(yte.sum())} pos)  [RAM-resident fp16]", flush=True)

    pin = (device.type == "cuda")
    pw = (args.num_workers > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=pin,
                          persistent_workers=pw, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=pin,
                        persistent_workers=pw)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=pin,
                         persistent_workers=pw)

    model = ResNet1D(in_ch=len(keys), width=args.width).to(device)
    if n_gpus > 1:
        # 한 fold 를 GPU n_gpus 개로: DataParallel(단일 프로세스 → 데이터는 RAM 에 1벌).
        # global batch 를 GPU 축으로 분할 → 각 GPU 는 batch//n_gpus 를 처리.
        model = nn.DataParallel(model)
        print(f"[{tag}] DataParallel across {n_gpus} GPUs "
              f"(global batch {args.batch_size} → ~{args.batch_size // n_gpus}/GPU)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    _unwrap = lambda m: m.module if isinstance(m, nn.DataParallel) else m
    # class imbalance: pos_weight = n_neg/n_pos (없으면 1)
    n_pos = float(ytr.sum()); n_neg = float(len(ytr) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    amp = (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    best_auroc, best_state, best_ep, no_imp = -1.0, None, -1, 0
    # CARMEN run.py 와 동일한 학습곡선 저장 (fold JSON 스키마 일치).
    train_losses: list[float] = []
    val_aurocs: list[float] = []
    for ep in range(args.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = crit(model(xb).squeeze(-1), yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_loss += float(loss.item()); nb += 1
        train_losses.append(ep_loss / max(nb, 1))
        va = _scores(model, val_dl, device)
        try:
            auroc = compute_auroc(yva, va)
        except Exception:
            auroc = float("nan")
        val_aurocs.append(float(auroc))
        if auroc > best_auroc:
            best_auroc, best_ep, no_imp = auroc, ep, 0
            best_state = copy.deepcopy(_unwrap(model).state_dict())
        else:
            no_imp += 1
        print(f"  ep{ep+1}/{args.epochs} val_auroc={auroc:.4f} "
              f"(best={best_auroc:.4f}@{best_ep+1})", flush=True)
        if args.patience > 0 and no_imp >= args.patience:
            print(f"  [early-stop] ep{ep+1}", flush=True); break

    if best_state is not None:
        _unwrap(model).load_state_dict(best_state)
    ys = _scores(model, test_dl, device)
    # threshold-dependent 지표는 val Youden threshold 로(test leakage 회피).
    va_scores = _scores(model, val_dl, device)
    thr = _youden_threshold(yva, va_scores)
    y_pred = (ys >= thr).astype(int)

    auroc = compute_auroc(yte, ys); auprc = compute_auprc(yte, ys)
    f1 = compute_f1(yte, y_pred, average="macro")
    ss = compute_sensitivity_specificity(yte, y_pred)
    # per-fold test CI — CARMEN run.py 와 동일: metrics.bootstrap_ci(window-level,
    # AUROC/AUPRC=score, F1=Youden-threshold binarized pred), 1000 iter.
    auroc_ci = bootstrap_ci(compute_auroc, yte, ys, n_iter=args.bootstrap_iters)
    auprc_ci = bootstrap_ci(compute_auprc, yte, ys, n_iter=args.bootstrap_iters)
    f1_ci = bootstrap_ci(
        lambda t, p: compute_f1(t, p, average="macro"),
        yte, y_pred, n_iter=args.bootstrap_iters,
    )
    print(f"\n[{tag}] TEST  AUROC={auroc:.4f} [{auroc_ci[0]:.4f}, {auroc_ci[1]:.4f}]  "
          f"AUPRC={auprc:.4f} [{auprc_ci[0]:.4f}, {auprc_ci[1]:.4f}]  "
          f"prevalence={yte.mean():.3f} ({yte.sum()}/{len(yte)})", flush=True)

    # ── preds .npz (run_eval 집계용, CARMEN run.py 와 동일) ──
    dump_fold_predictions(
        out_dir, task=task_name, fold_idx=args.fold,
        n_folds=args.n_folds, y_true=yte, y_score=ys, patient_ids=case_te,
    )
    # ── fold JSON — CARMEN hypotension run.py 의 results dict 와 **동일 키·구조** ──
    #   (auroc/auprc/f1/optimal_threshold/sensitivity/specificity/n_total/n_positive/
    #    prevalence + y_true/y_score/patient_ids + val_auroc_best/best_epoch +
    #    auroc_ci/auprc_ci/f1_ci + train_losses/val_aurocs + config). 같은 집계·표
    #    파이프라인이 CARMEN 결과와 구분 없이 처리한다.
    fold_json = {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "optimal_threshold": float(thr),
        "sensitivity": float(ss["sensitivity"]),
        "specificity": float(ss["specificity"]),
        "n_total": int(len(yte)),
        "n_positive": int(yte.sum()),
        "prevalence": float(yte.mean()),
        "y_true": yte.astype(int).tolist(),
        "y_score": ys.astype(float).tolist(),
        "patient_ids": [str(c) for c in case_te],
        "val_auroc_best": float(best_auroc),
        "best_epoch": int(best_ep),
        "auroc_ci": [float(auroc_ci[0]), float(auroc_ci[1])],
        "auprc_ci": [float(auprc_ci[0]), float(auprc_ci[1])],
        "f1_ci": [float(f1_ci[0]), float(f1_ci[1])],
        "train_losses": train_losses,
        "val_aurocs": val_aurocs,
        "config": {
            "mode": task_name,
            "model": "resnet1d",
            "width": args.width,
            "input_signals": keys,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "bootstrap_iters": args.bootstrap_iters,
            "val_split_seed": args.val_split_seed,
        },
    }
    # vault 수집 규칙과 동일한 fold{f}.json (CARMEN 결과도 vault 에선 이 이름).
    json_path = out_dir / f"fold{args.fold}.json"
    with open(json_path, "w") as f:
        json.dump(fold_json, f, indent=2)
    print(f"  saved preds_fold{args.fold}.npz + fold{args.fold}.json → {out_dir}",
          flush=True)


def main():
    p = add_common_args(
        argparse.ArgumentParser(description="IOH end-to-end 1D-CNN baseline"),
        default_signals=["ecg", "ppg", "abp"],
    )
    args = p.parse_args()
    # IOH 는 window 당 case 1개(수술 case) → case_ids 로 grouping (기존과 동일).
    run_cnn_baseline(args, task_name="hypotension_cnn", tag="IOH-CNN",
                     id_fields=("case_ids",))


if __name__ == "__main__":
    main()
