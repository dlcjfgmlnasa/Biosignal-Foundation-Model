# -*- coding: utf-8 -*-
"""IOH end-to-end CNN baseline — VitalDB IOH 문헌(Lee et al. 등) 스타일 비교군.

CARMEN(frozen linear-probe / LoRA) 과 **같은 prepared 데이터·같은 5-fold·같은 평가**
위에서, 사전학습 없이 **task 에 처음부터 지도학습하는 1D-CNN** 을 돌려 직접 비교한다.
SOTA 비교의 공정한 짝(end-to-end supervised)을 우리 코호트에서 재현하는 용도.

⚠ CARMEN 본코드(module/model/train/collate/downstream run.py)는 건드리지 않는다.
   이 파일은 prepared 데이터 로더(_save_utils)와 평가 유틸(_eval_utils, metrics)만
   재사용하는 **독립 스크립트**다.

메모리(OOM) 하드닝
------------------
기존 구현은 ``load_prepared_split_chunked`` 로 train/val/test 의 **모든 chunk 를
concat 해 RAM 에 올린 뒤** ``torch.stack`` 으로 (N,C,L) fp32 텐서를 세 split 동시에
만들어 RAM 피크가 폭발했다(네트워크 마운트 대형 코호트에서 커널 OOM-kill).

본 구현은 ``iter_prepared_split_chunks`` 로 **chunk 를 하나씩** 스트리밍하여 각 chunk
를 로컬 ``.npy`` (fp16, (n,C,L)) 로 저장하고(피크 = chunk 1개), 학습·평가는 그 로컬
파일들을 ``mmap`` 으로 여는 :class:`_NpyWindowDataset` + ``DataLoader`` 로 배치 단위
lazy 로드한다. RAM 사용은 데이터셋 크기와 무관하게 (chunk 몇 개 + 배치) 수준으로 고정.

입력: 우리 hypotension prepare_data 산출물(per-(fold,split)[_chunk] prefix).
모델: 1D ResNet (multi-channel waveform → binary IOH).
프로토콜: 매 epoch val AUROC best-ckpt → 복원 후 test 1회 → preds_fold{f}.npz 저장
          (CARMEN run.py 와 동일 → aggregator 로 동일 집계 가능).

사용법:
    python -m downstream.baselines.ioh_cnn \
        --data-path .../hypotension_ecg_ppg_abp_w300s_h5min \
        --input-signals ecg ppg abp --n-folds 5 --fold 0 \
        --epochs 100 --lr 1e-3 --batch-size 128 --device cuda \
        --num-workers 4 --cache-dir /local/scratch \
        --out-dir .../result/main/hypotension_cnn/ecg_ppg_abp_w300s_h5min
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from downstream._save_utils import iter_prepared_split_chunks
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


# ── 데이터: 스트리밍 캐시 + mmap Dataset (OOM 하드닝) ────────────────


def _cache_split(data_path, fold, split, input_signals, cache_dir):
    """한 split 의 chunk 를 하나씩 스트리밍해 로컬 ``.npy`` (fp16, (n,C,L)) 로 저장.

    RAM 피크 = chunk 1개 분량. 반환: (keys, npy_paths, counts, labels, case_ids).
    chunk 가 없으면 keys=[] 로 반환(호출측이 legacy val-split 등 처리).
    """
    keys: list[str] | None = None
    paths: list[str] = []
    counts: list[int] = []
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
        # 채널축 stack → (n, C, L) fp16. chunk 1개만 잡는다.
        chans = [torch.as_tensor(sig[s]).to(torch.float16) for s in keys]
        X = torch.stack(chans, dim=1)
        X = torch.nan_to_num(X, nan=0.0).numpy()
        pth = str(Path(cache_dir) / f"{split}_chunk{ci}.npy")
        np.save(pth, X)
        paths.append(pth)
        counts.append(int(X.shape[0]))

        y = np.asarray(payload["labels"]).reshape(-1).astype(np.float32)
        labels.append(y)
        cid = payload.get("case_ids")
        if cid is None:
            cid = [f"{split}_{ci}_{r}" for r in range(len(y))]
        else:
            cid = [str(c) for c in (cid.tolist() if hasattr(cid, "tolist") else cid)]
        case_ids.extend(cid)

        del X, chans, sig, payload
        gc.collect()

    if keys is None:
        return [], [], [], np.array([], dtype=np.float32), []
    lab = np.concatenate(labels) if labels else np.array([], dtype=np.float32)
    return keys, paths, counts, lab, case_ids


class _NpyWindowDataset(Dataset):
    """per-chunk 로컬 ``.npy`` (mmap) 위의 flat window 인덱스.

    RAM = (worker 당 최근 chunk 몇 개 mmap 핸들 + 배치). 데이터셋 크기와 무관.
    ``subset`` 으로 index/label/case 를 부분집합(예: legacy val-split)만 노출한다.
    """

    def __init__(self, npy_paths, counts, labels, case_ids, subset=None):
        self.paths = list(npy_paths)
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
        self._cache: dict[int, np.memmap] = {}

    def __len__(self):
        return len(self.index)

    def _mm(self, ci: int) -> np.memmap:
        mm = self._cache.get(ci)
        if mm is None:
            if len(self._cache) >= 4:  # 작은 LRU (worker 당)
                self._cache.pop(next(iter(self._cache)))
            mm = np.load(self.paths[ci], mmap_mode="r")
            self._cache[ci] = mm
        return mm

    def __getitem__(self, i: int):
        ci, r = self.index[i]
        x = np.array(self._mm(ci)[r], dtype=np.float32)  # (C, L), mmap 밖으로 복사
        return torch.from_numpy(x), torch.tensor(self.labels[i], dtype=torch.float32)


def _make_datasets(args, cache_dir):
    """train/val/test Dataset + keys 를 스트리밍 캐시로 구성."""
    load_fold = int(args.fold) if int(args.n_folds) > 1 else None
    ktr, tr_p, tr_c, tr_y, tr_id = _cache_split(
        args.data_path, load_fold, "train", args.input_signals, cache_dir
    )
    if not ktr:
        raise FileNotFoundError(
            f"no train chunks/signals for {args.data_path} (fold={load_fold})"
        )
    kva, va_p, va_c, va_y, va_id = _cache_split(
        args.data_path, load_fold, "val", args.input_signals, cache_dir
    )
    kte, te_p, te_c, te_y, te_id = _cache_split(
        args.data_path, load_fold, "test", args.input_signals, cache_dir
    )

    train_ds = _NpyWindowDataset(tr_p, tr_c, tr_y, tr_id)
    if kva:  # val chunk 존재
        val_ds = _NpyWindowDataset(va_p, va_c, va_y, va_id)
    else:  # legacy: train 20% 동적 split (index 만 나눔 — 데이터 재적재 없음)
        n = len(train_ds)
        rng = np.random.default_rng(args.val_split_seed)
        perm = rng.permutation(n)
        n_val = max(1, int(0.2 * n))
        vi, ti = perm[:n_val], perm[n_val:]
        val_ds = _NpyWindowDataset(tr_p, tr_c, tr_y, tr_id, subset=vi)
        train_ds = _NpyWindowDataset(tr_p, tr_c, tr_y, tr_id, subset=ti)
    test_ds = _NpyWindowDataset(te_p, te_c, te_y, te_id)
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


@torch.no_grad()
def _scores(model, loader, device):
    """loader(shuffle=False) 순서대로 sigmoid 확률 반환 (dataset.labels 와 정렬)."""
    model.eval()
    out = []
    for xb, _ in loader:
        out.append(torch.sigmoid(model(xb.to(device)).squeeze(-1)).cpu())
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
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 수 (네트워크 마운트 병렬 read)")
    p.add_argument("--cache-dir", type=str, default="", help="chunk .npy 캐시 위치(비우면 out-dir/_cache_fold{f}). 로컬 디스크 권장")
    p.add_argument("--keep-cache", action="store_true", help="종료 후 캐시 .npy 유지(기본 삭제)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default=".")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / f"_cache_fold{args.fold}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        train_ds, val_ds, test_ds, keys = _make_datasets(args, cache_dir)
        ytr = train_ds.labels.astype(int)
        yva = val_ds.labels.astype(int)
        yte = test_ds.labels.astype(int)
        case_te = test_ds.case_ids
        print(f"[IOH-CNN] fold={args.fold}/{args.n_folds}  signals={keys}  "
              f"train={len(ytr)}({int(ytr.sum())} pos) val={len(yva)} "
              f"test={len(yte)}({int(yte.sum())} pos)  [streaming/mmap]", flush=True)

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
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        # class imbalance: pos_weight = n_neg/n_pos (없으면 1)
        n_pos = float(ytr.sum()); n_neg = float(len(ytr) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_auroc, best_state, best_ep, no_imp = -1.0, None, -1, 0
        for ep in range(args.epochs):
            model.train()
            for xb, yb in train_dl:
                loss = crit(model(xb.to(device)).squeeze(-1), yb.to(device))
                opt.zero_grad(); loss.backward(); opt.step()
            va = _scores(model, val_dl, device)
            try:
                auroc = compute_auroc(yva, va)
            except Exception:
                auroc = float("nan")
            if auroc > best_auroc:
                best_auroc, best_ep, no_imp = auroc, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                no_imp += 1
            if (ep + 1) % 5 == 0 or ep == 0:
                print(f"  ep{ep+1}/{args.epochs} val_auroc={auroc:.4f} "
                      f"(best={best_auroc:.4f}@{best_ep+1})", flush=True)
            if args.patience > 0 and no_imp >= args.patience:
                print(f"  [early-stop] ep{ep+1}", flush=True); break

        if best_state is not None:
            model.load_state_dict(best_state)
        ys = _scores(model, test_dl, device)
        # threshold-dependent 지표는 val Youden threshold 로(test leakage 회피).
        va_scores = _scores(model, val_dl, device)
        thr = _youden_threshold(yva, va_scores)
        y_pred = (ys >= thr).astype(int)

        auroc = compute_auroc(yte, ys); auprc = compute_auprc(yte, ys)
        f1 = compute_f1(yte, y_pred, average="macro")
        ss = compute_sensitivity_specificity(yte, y_pred)
        print(f"\n[IOH-CNN] TEST  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
              f"prevalence={yte.mean():.3f} ({yte.sum()}/{len(yte)})", flush=True)

        # ── preds .npz (run_eval 집계용, CARMEN run.py 와 동일) ──
        dump_fold_predictions(
            out_dir, task="hypotension_cnn", fold_idx=args.fold,
            n_folds=args.n_folds, y_true=yte, y_score=ys, patient_ids=case_te,
        )
        # ── fold JSON (CARMEN fold{f}.json 과 동일 스키마 → 같은 방식으로 집계 가능) ──
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
    finally:
        if not args.keep_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
