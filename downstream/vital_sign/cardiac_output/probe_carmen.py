# -*- coding: utf-8 -*-
"""CARMEN frozen probe for ECG+PPG -> CO (cardiac output, L/min).

Phase 1 of the CO feasibility (Phase 0 = from-scratch on KGPU, see
scratchpad/co_phase0_v2.py). Runs on KHDP where the CARMEN checkpoint lives
(KGPU 반출 미승인). Data (ECG+PPG windows + CO labels) is pre-prepared OFF-server
via vitaldb (KHDP 폐쇄망이라 vitaldb 불가) into an .npz and uploaded.

두 arm 을 **동일 데이터·동일 split·동일 eval** 로 비교한다:
  - carmen  : frozen CARMEN encoder + attention-pool head (BPRegressionHead, n_out=1)
  - scratch : from-scratch 1D-CNN (Phase 0 와 동일 구조)

핵심 질문(유일하게 남은 것): 사전학습 feature 가 **within-patient trending**
(residual r / temporal 4-quadrant)을 from-scratch 위로 유의하게 끌어올리나?
절대 CO 는 Phase 0 에서 이미 mean-null 과 구별 불가(park 후보).

Eval (estimator-hardened): patient-level MAE + patient-cluster bootstrap CI,
slope β, within-patient residual r (all + EV1000-only), temporal 4-quadrant,
Bland-Altman. window-level 은 참고만.

Usage (KHDP)
------------
    python -m downstream.vital_sign.cardiac_output.probe_carmen \
        --data /path/to/co_data_w30.npz \
        --checkpoint /home/coder/workspace/k-mimic-/bio_fm/outputs/main/phase2/\
kmimic_phase2_k2/checkpoints/checkpoint_phase2_av_epoch049_final.pt \
        --model-version v2 --arm both --epochs 40 --device cuda \
        --out /home/coder/workspace/k-mimic-/bio_fm/result/main/cardiac_output/co_probe.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR = 100


# ── Eval helpers (Phase 0 v2 와 동일) ────────────────────────────


def within_resid_r(pred, true, pid, sel=None):
    pr, tr = pred.astype(float).copy(), true.astype(float).copy()
    for u in np.unique(pid):
        m = pid == u
        pr[m] -= pr[m].mean()
        tr[m] -= tr[m].mean()
    keep = np.ones(len(pred), bool) if sel is None else sel
    pr, tr = pr[keep], tr[keep]
    if len(pr) < 3 or pr.std() < 1e-8 or tr.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pr, tr)[0, 1])


def four_quadrant_temporal(pred, true, pid, ws):
    dp, dt = [], []
    for u in np.unique(pid):
        m = np.where(pid == u)[0]
        if len(m) < 2:
            continue
        o = m[np.argsort(ws[m])]  # temporal order within patient
        dp.append(np.diff(pred[o]))
        dt.append(np.diff(true[o]))
    if not dp:
        return float("nan"), 0
    dp, dt = np.concatenate(dp), np.concatenate(dt)
    z = 0.15 * np.max(np.abs(dt)) if dt.size else 0
    keep = np.abs(dt) >= z
    if keep.sum() == 0:
        return float("nan"), 0
    return float((np.sign(dp[keep]) == np.sign(dt[keep])).mean()), int(keep.sum())


def patient_agg(pred, true, pid):
    ups = np.unique(pid)
    pa = np.array([pred[pid == u].mean() for u in ups])
    ta = np.array([true[pid == u].mean() for u in ups])
    return pa, ta


def boot_ci(pa_pred, pa_true, mu, rng, n=2000):
    k = len(pa_pred)
    mm, dd = [], []
    for _ in range(n):
        s = rng.integers(0, k, k)
        m = np.mean(np.abs(pa_pred[s] - pa_true[s]))
        nl = np.mean(np.abs(pa_true[s] - mu))
        mm.append(m)
        dd.append(nl - m)
    q = lambda a: [round(float(np.percentile(a, 2.5)), 3), round(float(np.percentile(a, 97.5)), 3)]
    return {"patient_mae_ci": q(mm), "delta_null_minus_model_ci": q(dd),
            "prob_beat_null": round(float(np.mean(np.array(dd) > 0)), 3)}


def evaluate(pred, yte, pidte, wste, devte, mu):
    pa, ta = patient_agg(pred, yte, pidte)
    p_mae, p_null = float(np.mean(np.abs(pa - ta))), float(np.mean(np.abs(ta - mu)))
    beta = float(np.polyfit(ta, pa, 1)[0]) if np.std(ta) > 1e-6 else 0.0
    ci = boot_ci(pa, ta, mu, np.random.default_rng(0))
    ev = devte == "EV1000/CO"
    fq, nfq = four_quadrant_temporal(pred, yte, pidte, wste)
    d = pred - yte
    return {
        "patient_level": {"mae": round(p_mae, 3), "null_mae": round(p_null, 3),
                          "beat_null": bool(p_mae < p_null), "slope_beta": round(beta, 3), **ci},
        "within_patient": {"residual_r_all": round(within_resid_r(pred, yte, pidte), 3),
                           "residual_r_ev1000": (round(within_resid_r(pred, yte, pidte, ev), 3)
                                                 if ev.sum() > 5 else None),
                           "four_quadrant_temporal": (None if fq != fq else round(fq, 3)), "n_fq": nfq},
        "window_level": {"mae": round(float(np.mean(np.abs(pred - yte))), 3),
                         "null_mae": round(float(np.mean(np.abs(yte - mu))), 3)},
        "bland_altman": {"bias": round(float(d.mean()), 3),
                         "loa": [round(float(d.mean() - 1.96 * d.std()), 2),
                                 round(float(d.mean() + 1.96 * d.std()), 2)]},
    }


# ── CARMEN arm ───────────────────────────────────────────────────


def build_batches(idxs, X, y, pid, ws, dev, patch_size, batch_size, sr=SR):
    """(ECG ch0=type0, PPG ch1=type2) any_variate PackedBatch + 병렬 라벨/pid/ws/dev."""
    from data.collate import PackCollate
    from data.dataset import BiosignalSample
    from data.spatial_map import SIGNAL_KEY_TO_TYPE

    out = []
    keys = ["ecg", "ppg"]
    for s in range(0, len(idxs), batch_size):
        chunk = idxs[s:s + batch_size]
        samples, yb, pb, wb, db = [], [], [], [], []
        rec = 0
        for gi in chunk:
            added = False
            for ch, key in enumerate(keys):
                sig = X[gi, ch].astype(np.float32)
                n_us = (len(sig) // patch_size) * patch_size
                if n_us == 0:
                    continue
                samples.append(BiosignalSample(
                    values=torch.tensor(sig[:n_us], dtype=torch.float32), length=n_us,
                    channel_idx=ch, recording_idx=rec, sampling_rate=sr, n_channels=2,
                    win_start=0, signal_type=SIGNAL_KEY_TO_TYPE[key],
                    session_id=f"co_{gi}", spatial_id=0))
                added = True
            if not added:
                continue
            yb.append(float(y[gi])); pb.append(str(pid[gi])); wb.append(int(ws[gi])); db.append(str(dev[gi]))
            rec += 1
        if not samples:
            continue
        rl = defaultdict(int)
        for smp in samples:
            rl[smp.recording_idx] += smp.length
        collate = PackCollate(max_length=max(rl.values()), collate_mode="any_variate", patch_size=patch_size)
        out.append((collate(samples), torch.tensor(yb, dtype=torch.float32),
                    np.array(pb), np.array(wb, dtype=np.int64), np.array(db)))
    return out


def run_carmen(X, y, pid, ws, dev, tr_idx, te_idx, ckpt, mv, epochs, lr, device):
    from downstream.model_wrapper import DownstreamModelWrapper
    from downstream.generation.cross_modal.run import BPRegressionHead

    dev_t = torch.device(device)
    wrapper = DownstreamModelWrapper(ckpt, mv, device)
    fm = wrapper.model
    ps = wrapper.patch_size
    print(f"  CARMEN loaded: d_model={wrapper.d_model} patch_size={ps}", flush=True)

    tr_b = build_batches(tr_idx, X, y, pid, ws, dev, ps, 16)
    te_b = build_batches(te_idx, X, y, pid, ws, dev, ps, 16)
    ytr = torch.cat([b[1] for b in tr_b]).numpy()
    mu, sd = float(ytr.mean()), float(ytr.std()) + 1e-6

    # frozen feature cache (encoder 고정 → 1회 forward)
    cache = []
    with torch.no_grad():
        for b, lab, *_ in tr_b:
            o = fm(wrapper.batch_to_device(b), task="masked", mask_ratio=0.0)
            cache.append((o["encoded"].detach().cpu(), o["patch_mask"].detach().cpu(), lab))

    head = BPRegressionHead(wrapper.d_model, n_out=1).to(dev_t)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    for ep in range(epochs):
        head.train(); tot = 0.0
        for enc, mask, lab in cache:
            pred = head(enc.to(dev_t), mask.to(dev_t)).squeeze(-1)
            loss = F.mse_loss(pred, (lab.to(dev_t) - mu) / sd)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"  [carmen] ep{ep+1} loss={tot/len(cache):.4f}", flush=True)

    head.eval()
    preds, ys, pids, wss, devs = [], [], [], [], []
    with torch.no_grad():
        for b, lab, pb, wb, db in te_b:
            o = fm(wrapper.batch_to_device(b), task="masked", mask_ratio=0.0)
            pr = head(o["encoded"], o["patch_mask"]).squeeze(-1).cpu().numpy() * sd + mu
            preds.append(pr); ys.append(lab.numpy()); pids.append(pb); wss.append(wb); devs.append(db)
    pred = np.concatenate(preds)
    return evaluate(pred, np.concatenate(ys), np.concatenate(pids),
                    np.concatenate(wss), np.concatenate(devs), mu)


# ── from-scratch arm (Phase 0 와 동일 구조) ──────────────────────


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, 32, 7, 2, 3), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run_scratch(X, y, pid, ws, dev, tr_idx, te_idx, epochs, lr, device):
    dev_t = torch.device(device)
    # per-window z-score (Phase 0 와 동일)
    def norm(a):
        a = a.astype(np.float32)
        m = a.mean(axis=2, keepdims=True); s = a.std(axis=2, keepdims=True) + 1e-6
        return (a - m) / s
    Xtr = torch.tensor(norm(X[tr_idx])).to(dev_t)
    Xte = torch.tensor(norm(X[te_idx])).to(dev_t)
    ytr = torch.tensor(y[tr_idx], dtype=torch.float32).to(dev_t)
    mu, sd = ytr.mean().item(), ytr.std().item() + 1e-6
    model = CNN().to(dev_t)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-2)
    n, bs = len(Xtr), 128
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev_t); tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            loss = F.mse_loss(model(Xtr[idx]), (ytr[idx] - mu) / sd)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        if (ep + 1) % 15 == 0:
            print(f"  [scratch] ep{ep+1} loss={tot/(n//bs+1):.4f}", flush=True)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).cpu().numpy() * sd + mu
    return evaluate(pred, y[te_idx], pid[te_idx], ws[te_idx], dev[te_idx], mu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="co_data_w30.npz (X,y,pid,ws,dev)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model-version", default="v2")
    ap.add_argument("--arm", default="both", choices=["carmen", "scratch", "both"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="co_probe_result.json")
    a = ap.parse_args()
    t0 = time.time()

    d = np.load(a.data, allow_pickle=True)
    X, y, pid, ws, dev = d["X"], d["y"].astype(np.float32), d["pid"].astype(str), d["ws"], d["dev"].astype(str)
    print(f"DATA X={X.shape} patients={len(np.unique(pid))} CO mean={y.mean():.2f} "
          f"devices={dict(zip(*np.unique(dev, return_counts=True)))}", flush=True)

    rng = np.random.default_rng(a.seed)
    pats = np.unique(pid); rng.shuffle(pats)
    trp = set(pats[:int(len(pats) * 0.8)])
    tr_idx = np.array([i for i in range(len(y)) if pid[i] in trp])
    te_idx = np.array([i for i in range(len(y)) if pid[i] not in trp])
    print(f"split: train {len(tr_idx)}win / test {len(te_idx)}win "
          f"({len(set(pid[te_idx]))} test patients)", flush=True)

    res = {"n_win": int(len(y)), "n_patients": int(len(pats)),
           "n_test_patients": int(len(set(pid[te_idx]))),
           "co_mean": round(float(y.mean()), 2), "co_std": round(float(y.std()), 2),
           "window_s": int(d["win_s"]) if "win_s" in d else None}
    if a.arm in ("scratch", "both"):
        print("\n=== from-scratch arm ===", flush=True)
        res["scratch"] = run_scratch(X, y, pid, ws, dev, tr_idx, te_idx, max(a.epochs, 50), a.lr, a.device)
    if a.arm in ("carmen", "both"):
        print("\n=== CARMEN frozen arm ===", flush=True)
        res["carmen"] = run_carmen(X, y, pid, ws, dev, tr_idx, te_idx,
                                   a.checkpoint, a.model_version, a.epochs, a.lr, a.device)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    print("\nRESULT", json.dumps(res), flush=True)
    json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
