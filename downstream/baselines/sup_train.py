# -*- coding: utf-8 -*-
"""지도학습 기준선(1D-ResNet · CARMEN-scratch) 공통 학습기 — 창 배열 입력, 환자 단위 5-fold OOF.

과제별 feature 스크립트는 창을 저장하지 않으므로, 창 덤프(``moment_tok.DumpEncoder``)나
준비 데이터에서 만든 표준 입력 하나로 모든 과제를 같은 규칙으로 학습한다.

입력 (``--data <prefix>``):
    ``<prefix>_X.npy``   (N, C, L) float16 — 없는 채널은 NaN
    ``<prefix>_meta.npz`` y (N,) · pid (N,) · keys (C,) [· fold (N,) 0..4]
출력 (``--out``): ``preds.npz`` — pred (N,) 또는 (N, K) · y · pid · fold (OOF, 모든 행)

규칙 (두 모델 공통):
    - fold 가 없으면 환자 단위 5-fold(seed 고정). 각 fold 의 학습 환자 중 10% 를 val 로 떼어
      val loss 최저 epoch 의 가중치로 test 를 예측(early stopping patience).
      epoch 당 학습 창은 최대 ``--epoch-size`` 개 무작위 부분집합(대형 코호트 학습 시간 상한),
      val 은 최대 ``--val-max`` 개 고정 부분집합.
    - binary=BCE · multiclass=CE · regression=MSE(학습 fold 평균·표준편차로 표준화 후 복원).
    - 1D-ResNet: 창·채널별 z-score, 없는 채널 0. ``downstream/baselines/ioh_cnn.ResNet1D`` 그대로.
    - CARMEN-scratch: 사전학습 ckpt 의 **ModelConfig 만** 쓰고 random init
      (``DownstreamModelWrapper(init_random=True)``), 전체 파라미터 학습. CARMEN downstream 과 같이
      채널마다 독립 인코딩 → 토큰 mean-pool → 채널 concat(없는 채널 0) → 선형 head.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

# v2 signal_type (data/spatial_map.py). PAP 는 재학습 전이라 ABP 타입으로 임시 입력.
V2 = {"ecg": 0, "abp": 1, "tono": 1, "pap": 1, "ppg": 2, "cvp": 3, "co2": 4, "awp": 5,
      "icp": 6, "resp": 7, "resp_impedance": 7, "chest": 7, "abd": 7, "resp_flow": 8, "air": 8}


def patient_folds(pid: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    u = np.unique(pid); rng = np.random.default_rng(seed); rng.shuffle(u)
    f = {p: i % k for i, p in enumerate(u)}
    return np.array([f[p] for p in pid])


def split_val(pid_tr: np.ndarray, frac: float = 0.1, seed: int = 0) -> np.ndarray:
    u = np.unique(pid_tr); rng = np.random.default_rng(seed); rng.shuffle(u)
    return np.isin(pid_tr, u[: max(1, int(round(len(u) * frac)))])


class ResNetNet(nn.Module):
    def __init__(self, C: int, K: int, level: bool = False):
        super().__init__()
        from downstream.baselines.ioh_cnn import ResNet1D

        self.net = ResNet1D(C, width=64, n_classes=K)
        self.level = level
        if level:  # 창 정규화로 지워지는 절대 수준(창 평균·log 표준편차)을 head 에 되돌림 — CARMEN loc/scale cond 와 대칭
            self.stat_bn = nn.BatchNorm1d(2 * C)
            self.fc = nn.Linear(self.net.head[-1].in_features + 2 * C, K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, C, L) raw, NaN = 없는 채널
        mu = torch.nanmean(x, -1, keepdim=True)
        sd = torch.sqrt(torch.nanmean((x - mu) ** 2, -1, keepdim=True)).clamp_min(1e-6)
        xn = torch.nan_to_num((x - mu) / sd)
        if not self.level:
            return self.net(xn)
        h = self.net.head[1](self.net.head[0](self.net.blocks(self.net.stem(xn))))
        st = torch.nan_to_num(torch.cat([mu, sd.log()], 1).squeeze(-1))  # (B, 2C), 없는 채널 0
        return self.fc(torch.cat([h, self.stat_bn(st)], 1))


class ScratchNet(nn.Module):
    def __init__(self, ckpt: str, keys: list[str], K: int, device):
        super().__init__()
        from downstream.model_wrapper import DownstreamModelWrapper

        w = DownstreamModelWrapper(ckpt, model_version="v2", device=device, init_random=True)
        self.w, self.enc = w, w.model
        self.P = int(w.model.patch_size)
        self.keys = keys
        self.d = int(w.model.d_model)
        self.head = nn.Linear(self.d * len(keys), K)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        """CARMEN downstream 과 같은 방식: 채널마다 독립 시퀀스로 인코딩 → 토큰 mean-pool →
        채널 순서대로 concat (없는 채널은 0)."""
        from data.dataset import BiosignalSample
        from downstream.window_task import iter_window_batches

        xs = x.float().cpu().numpy()
        B = len(xs)
        feat = torch.zeros(B, self.d * len(self.keys), device=self.w.device)
        for c, k in enumerate(self.keys):
            rows = [i for i in range(B) if not np.isnan(xs[i, c]).all()]
            if not rows:
                continue
            st = V2[k]

            def to_samples(i, _k, c=c, st=st):
                return [BiosignalSample(values=torch.from_numpy(np.nan_to_num(xs[i, c])), length=xs.shape[2], channel_idx=0,
                                        recording_idx=i, sampling_rate=100.0, n_channels=1, win_start=0, signal_type=st,
                                        session_id=f"w{i}", spatial_id=st)]

            (batch, order), = list(iter_window_batches(rows, len(rows), self.P, to_samples=to_samples,
                                                       get_label=lambda i: float(i)))
            out = self.enc._encode(self.w.batch_to_device(batch), task="masked")
            m = out["patch_mask"].unsqueeze(-1).float()
            f = ((out["encoded"] * m).sum(1) / m.sum(1).clamp_min(1.0)).float()  # (len(rows), d), 배치 행 순서
            feat[order.long().to(feat.device), c * self.d:(c + 1) * self.d] = f  # label = 원래 행 번호
        return self.head(feat)


def run(args):
    dev = torch.device("cuda")
    X = np.load(f"{args.data}_X.npy", mmap_mode="r")
    M = np.load(f"{args.data}_meta.npz", allow_pickle=True)
    y, pid, keys = M["y"], M["pid"].astype(str), [str(k) for k in M["keys"]]
    fold = M["fold"].astype(int) if "fold" in M.files else patient_folds(pid)
    task = args.task
    K = int(np.nanmax(y)) + 1 if task == "multiclass" else 1
    N = len(y)
    pred = np.full((N, K) if task == "multiclass" else N, np.nan, np.float32)
    os.makedirs(args.out, exist_ok=True)
    print(f"{args.model} {task} N={N} C={X.shape[1]} L={X.shape[2]} keys={keys} 환자 {len(np.unique(pid))}", flush=True)

    for f in range(5):
        if args.folds and f not in args.folds:
            continue
        te = np.where(fold == f)[0]; trall = np.where((fold != f) & np.isfinite(y))[0]  # 라벨 결측 행은 학습·val 제외(예측은 함)
        isv = split_val(pid[trall], seed=f); va, tr = trall[isv], trall[~isv]
        rng = np.random.default_rng(f)
        if len(va) > args.val_max:  # val 은 고정 부분집합
            va = np.sort(rng.choice(va, args.val_max, replace=False))
        yt = y.astype(np.float32).copy()
        if task == "regression":
            mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-6); yt = (yt - mu) / sd
        yt = np.nan_to_num(yt)
        torch.manual_seed(f)
        net = (ResNetNet(X.shape[1], K, level=args.model == "resnet_lvl") if args.model.startswith("resnet") else ScratchNet(args.ckpt, keys, K, dev)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-2)
        lossf = nn.CrossEntropyLoss() if task == "multiclass" else (nn.BCEWithLogitsLoss() if task == "binary" else nn.MSELoss())

        def batches(idx, shuffle):  # 배치 안은 정렬(memmap 읽기), 라벨도 같은 ii 로 → 순서 일치
            idx = np.random.default_rng().permutation(idx) if shuffle else idx
            for s in range(0, len(idx), args.bs):
                ii = np.sort(idx[s:s + args.bs])
                yield ii, torch.from_numpy(np.asarray(X[ii], np.float32)).to(dev)

        def infer(idx):
            net.eval(); out = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for ii, xb in batches(idx, False):
                    out.append(net(xb).float().cpu())
            return torch.cat(out)

        def loss_on(logits, ii):
            t = torch.from_numpy(yt[ii]).to(dev)
            return lossf(logits, t.long()) if task == "multiclass" else lossf(logits[:, 0], t)

        best, best_state, bad = np.inf, None, 0
        for ep in range(args.epochs):
            net.train(); t0, tl, n = time.time(), 0.0, 0
            tr_ep = rng.choice(tr, min(len(tr), args.epoch_size), replace=False)  # epoch 당 창 상한
            for ii, xb in batches(tr_ep, True):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = loss_on(net(xb).float(), ii)
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                tl += loss.item() * len(ii); n += len(ii)
            vl = loss_on(infer(va).to(dev), va).item()
            print(f"  fold{f} ep{ep} train {tl / n:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if vl < best - 1e-4:
                best, bad = vl, 0; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= args.patience:
                    break
        net.load_state_dict(best_state)
        p = infer(te)
        if task == "multiclass":
            pred[te] = torch.softmax(p, -1).numpy()
        elif task == "binary":
            pred[te] = torch.sigmoid(p[:, 0]).numpy()
        else:
            pred[te] = p[:, 0].numpy() * sd + mu
        np.savez(f"{args.out}/preds_fold{f}.npz", pred=pred[te], y=y[te], pid=pid[te], idx=te, best_val=best)
        print(f"fold{f} done best val {best:.4f}", flush=True)
        del net, opt; torch.cuda.empty_cache()
    np.savez(f"{args.out}/preds.npz", pred=pred, y=y, pid=pid, fold=fold)
    print("SUP-DONE", args.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--task", choices=["binary", "multiclass", "regression"], required=True)
    ap.add_argument("--model", choices=["resnet", "resnet_lvl", "scratch"], required=True)
    ap.add_argument("--ckpt", help="scratch: ModelConfig 를 읽을 사전학습 ckpt")
    ap.add_argument("--epochs", type=int, default=50); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--bs", type=int, default=None); ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--epoch-size", type=int, default=None, help="epoch 당 학습 창 상한(무작위 부분집합). 기본 resnet 5만·scratch 2만")
    ap.add_argument("--val-max", type=int, default=None, help="val 창 상한(고정 부분집합). 기본 resnet 2만·scratch 5천")
    a = ap.parse_args()
    a.bs = a.bs or (256 if a.model.startswith("resnet") else 16)
    a.lr = a.lr or (1e-3 if a.model.startswith("resnet") else 1e-4)
    a.epoch_size = a.epoch_size or (50000 if a.model.startswith("resnet") else 20000)
    a.val_max = a.val_max or (20000 if a.model.startswith("resnet") else 5000)
    run(a)
