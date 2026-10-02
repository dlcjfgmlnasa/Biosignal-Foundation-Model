# -*- coding: utf-8 -*-
"""동결 CARMEN feature + LinearProbe — sup_train 과 같은 창 배열 입력·같은 환자 fold 로 사전학습 ckpt 끼리 비교.

사전학습 설계 비교(예: RespImp 게이팅 ``r7_*``)용. 창 배열 하나를 여러 ckpt 로 인코딩해 두고,
채널 조합마다 같은 fold·같은 probe 레시피로 학습해 ckpt 간 paired ΔAUROC 를 낸다.

    feat : ``<data>_X.npy`` (N, C, L) → 채널마다 독립 인코딩 → 유효 patch mean-pool
           → ``<out>`` npz: feat (N, C, d) float16 · have (N, C) bool · keys
    probe: feature 세트(``tag=npz``) × 채널 조합(``name=k1,k2``) 마다 환자 단위 5-fold OOF.
           입력 = 채널 embedding concat(없는 채널 0) + 채널 보유 지시변수 (논문 Downstream 절과 동일).
           LinearProbe(LayerNorm→Dropout(0.1)→Linear) · Adam 1e-3 · BCE · batch 512 · val AUROC 최고 epoch.
           fold·val 분할은 sup_train 과 같다(meta 의 fold, 없으면 patient_folds seed 0 / split_val seed=fold).

예:
    python -m downstream.baselines.frozen_probe feat --data $S/sleep --keys air chest abd --ckpt CK --out F_a.npz
    python -m downstream.baselines.frozen_probe probe --data $S/sleep --feat a=F_a.npz b=F_b.npz \
        --sets resp=chest,abd air=air all=air,chest,abd --pairs b-a --out res.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from downstream.baselines.sup_eval import boot
from downstream.baselines.sup_train import V2, patient_folds, split_val


@torch.no_grad()
def feat(args):
    from data.dataset import BiosignalSample
    from downstream.model_wrapper import DownstreamModelWrapper
    from downstream.window_task import iter_window_batches

    X = np.load(f"{args.data}_X.npy", mmap_mode="r")
    M = np.load(f"{args.data}_meta.npz", allow_pickle=True)
    allk = [str(k) for k in M["keys"]]
    w = DownstreamModelWrapper(args.ckpt, model_version="v2", device="cuda")
    enc, P, d = w.model.eval(), int(w.model.patch_size), int(w.model.d_model)
    N, L = X.shape[0], X.shape[2]
    F = np.zeros((N, len(args.keys), d), np.float16)
    have = np.zeros((N, len(args.keys)), bool)
    print(f"feat N={N} L={L} keys={args.keys} patch={P} d={d} ckpt={args.ckpt}", flush=True)

    for j, k in enumerate(args.keys):
        c, st = allk.index(k), V2[k]
        for s in range(0, N, args.chunk):  # 청크 단위로 memmap 을 읽는다
            xs = np.asarray(X[s:s + args.chunk, c], np.float32)  # (n, L)
            rows = np.where(~np.isnan(xs).all(1))[0]
            if not len(rows):
                continue

            def to_samples(i, _k, xs=xs, st=st):
                return [BiosignalSample(values=torch.from_numpy(np.nan_to_num(xs[i])), length=L, channel_idx=0,
                                        recording_idx=int(i), sampling_rate=100.0, n_channels=1, win_start=0,
                                        signal_type=st, session_id=f"w{i}", spatial_id=st)]

            for batch, order in iter_window_batches(list(rows), args.bs, P, to_samples=to_samples,
                                                    get_label=lambda i: float(i)):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = enc._encode(w.batch_to_device(batch), task="masked")
                m = out["patch_mask"].unsqueeze(-1).float()
                f = ((out["encoded"].float() * m).sum(1) / m.sum(1).clamp_min(1.0)).cpu().numpy()
                ii = s + order.long().numpy()  # label = 청크 안 행 번호
                F[ii, j] = f
                have[ii, j] = True
            print(f"  {k} {min(s + args.chunk, N)}/{N}", flush=True)
    np.savez(args.out, feat=F, have=have, keys=np.array(args.keys))
    print("FEAT-DONE", args.out, flush=True)


def fit_probe(Xtr, ytr, Xva, yva, Xte, seed, max_epochs=2000, patience=20):
    from downstream.model_wrapper import LinearProbe

    torch.manual_seed(seed)
    dev = Xtr.device
    probe = LinearProbe(Xtr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    best, best_state, bad = -1.0, None, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(max_epochs):
        probe.train()
        for b in torch.randperm(len(Xtr), generator=g).split(512):
            b = b.to(dev)
            loss = lossf(probe(Xtr[b])[:, 0], ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        probe.eval()
        with torch.no_grad():
            a = roc_auc_score(yva, probe(Xva)[:, 0].float().cpu().numpy())
        if a > best + 1e-4:
            best, bad = a, 0
            best_state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    probe.load_state_dict(best_state); probe.eval()
    with torch.no_grad():
        return torch.sigmoid(probe(Xte)[:, 0]).float().cpu().numpy(), best


def probe(args):
    M = np.load(f"{args.data}_meta.npz", allow_pickle=True)
    y, pid = M["y"].astype(np.float32), M["pid"].astype(str)
    fold = M["fold"].astype(int) if "fold" in M.files else patient_folds(pid)
    sets = {s.split("=")[0]: s.split("=")[1].split(",") for s in args.sets}
    feats = dict(s.split("=", 1) for s in args.feat)
    res, oof = {}, {}
    for tag, path in feats.items():
        z = np.load(path, allow_pickle=True)
        keys = [str(k) for k in z["keys"]]
        for sname, ks in sets.items():
            idx = [keys.index(k) for k in ks]
            Xall = np.concatenate([z["feat"][:, idx].reshape(len(y), -1).astype(np.float32),
                                   z["have"][:, idx].astype(np.float32)], 1)
            use = z["have"][:, idx].any(1) & np.isfinite(y)  # 채널 하나라도 있는 창만
            Xg = torch.from_numpy(Xall).cuda()
            pred = np.full(len(y), np.nan, np.float32)
            for f in range(5):
                te = np.where((fold == f) & use)[0]; trall = np.where((fold != f) & use)[0]
                isv = split_val(pid[trall], seed=f); va, tr = trall[isv], trall[~isv]
                pred[te], vbest = fit_probe(Xg[tr], torch.from_numpy(y[tr]).cuda(), Xg[va], y[va], Xg[te], seed=f)
                print(f"  {tag}/{sname} fold{f} val AUROC {vbest:.4f}", flush=True)
            ok = np.isfinite(pred)
            a, p = roc_auc_score(y[ok], pred[ok]), average_precision_score(y[ok], pred[ok])
            ca, cp = boot(y[ok], pred[ok], pid[ok], roc_auc_score, n=args.boot), boot(y[ok], pred[ok], pid[ok], average_precision_score, n=args.boot)
            res[f"{tag}/{sname}"] = dict(n=int(ok.sum()), auroc=a, auroc_ci=list(ca), auprc=p, auprc_ci=list(cp))
            oof[(tag, sname)] = pred
            print(f"{tag}/{sname:<8} n={ok.sum()} AUROC {a:.3f} ({ca[0]:.3f}–{ca[1]:.3f})  AUPRC {p:.3f} ({cp[0]:.3f}–{cp[1]:.3f})", flush=True)
            del Xg; torch.cuda.empty_cache()

    # paired Δ (b − a): 같은 행·환자 cluster bootstrap
    for pr in args.pairs or []:
        a_, b_ = pr.split("-")[1], pr.split("-")[0]
        for sname in sets:
            pa, pb = oof[(a_, sname)], oof[(b_, sname)]
            ok = np.isfinite(pa) & np.isfinite(pb)
            for mname, fn in (("auroc", roc_auc_score), ("auprc", average_precision_score)):
                d = fn(y[ok], pb[ok]) - fn(y[ok], pa[ok])
                ci = boot(y[ok], np.stack([pb[ok], pa[ok]], 1), pid[ok], lambda yy, ss, fn=fn: fn(yy, ss[:, 0]) - fn(yy, ss[:, 1]), n=args.boot)
                sig = "*" if ci[0] > 0 or ci[1] < 0 else " "
                res[f"d_{mname}/{sname}/{pr}"] = dict(delta=d, ci=list(ci))
                print(f"Δ{mname.upper()} {sname:<8} {b_} − {a_}  {d:+.3f} ({ci[0]:+.3f},{ci[1]:+.3f}){sig}", flush=True)
    json.dump(res, open(args.out, "w"), indent=1)
    print("PROBE-DONE", args.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["feat", "probe"])
    ap.add_argument("--data", required=True, help="sup 입력 prefix (<prefix>_X.npy · <prefix>_meta.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt"); ap.add_argument("--keys", nargs="+", help="feat: 인코딩할 채널 (meta keys 중)")
    ap.add_argument("--bs", type=int, default=256); ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--feat", nargs="+", help="probe: tag=feat.npz")
    ap.add_argument("--sets", nargs="+", help="probe: name=k1,k2 채널 조합")
    ap.add_argument("--pairs", nargs="*", help="probe: b-a (b − a) paired Δ")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    feat(a) if a.cmd == "feat" else probe(a)
