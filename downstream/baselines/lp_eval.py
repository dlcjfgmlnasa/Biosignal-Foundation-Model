# -*- coding:utf-8 -*-
"""lp_eval — index_eval2 와 인자·fold·지표·출력이 같고 probe 만 run.py LinearProbe 레시피로 바꾼 판 (09-28 사용자: probe 전부 LinearProbe).

probe = LayerNorm→Dropout(0.1)→Linear, Adam lr 1e-3, batch 512, ≤2000 epoch(patience 200·≤300k step), val 최고 epoch 복원.
  라벨이 0/1 → BCE·val AUROC / 연속 → 표준화 라벨 MSE·val MAE. fold k test · fold (k+1)%5 val · 나머지 train.
환자 단위 5-fold LinearProbe OOF → 환자 클러스터 bootstrap, paired Δ.
지표: MAE, pooled r, within-patient r(환자 평균 제거), |err|≤tol, AUROC(label ≥ thr).
--patient-agg: 창 단위로 학습하되 환자별 예측·라벨 평균으로 평가(나이처럼 환자 내 상수인 라벨).
--require: 모델에 쓰지 않아도 유한해야 하는 키(비교 집합 고정용).
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ap = argparse.ArgumentParser()
ap.add_argument("--feat", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--models", nargs="+", required=True)
ap.add_argument("--refs", nargs="+", default=[])
ap.add_argument("--direct", nargs="*", default=[])
ap.add_argument("--require", nargs="*", default=[])
ap.add_argument("--where", nargs="*", default=[], help="key:lo:hi — 이 범위 행만 평가(예: t_rel:0.25:0.75)")
ap.add_argument("--thr", type=float, default=None)
ap.add_argument("--tol", type=float, default=2.0)
ap.add_argument("--unit", default="")
ap.add_argument("--boot", type=int, default=2000)  # 논문 Methods: patient-cluster bootstrap 2,000회
ap.add_argument("--patient-agg", action="store_true")
ap.add_argument("--pid-key", default="pid", help="fold·bootstrap 묶음 키(예: PWDB 연령군 age)")
ap.add_argument("--pid-map", default=None, help="VitalDB cases.csv — pid '*_<caseid>' 를 subjectid 로 치환")
ap.add_argument("--out", required=True)
ap.add_argument("--lt", action="store_true", help="양성 = label < thr (AUROC·AUPRC 점수 부호 반전)")
args = ap.parse_args()

z = np.load(args.feat, allow_pickle=True)
y = z[args.label].astype(np.float64)
pid = np.array([str(v) for v in z[args.pid_key]])
if args.pid_map:
    import csv
    cmap = {r["caseid"]: r["subjectid"] for r in csv.DictReader(open(args.pid_map, encoding="utf-8-sig"))}
    pid = np.array([f"vdb_{cmap.get(str(p).split('_')[-1], str(p))}" for p in pid])
N = len(y)


def feat(spec):
    return np.concatenate([np.asarray(z[k], dtype=np.float64).reshape(N, -1) for k in spec.split("+")], 1)


models = {}
for m in args.models:
    name, spec = m.split("=", 1)
    models[name] = None if spec == "null" else feat(spec)
direct = {f"direct:{k}": np.asarray(z[k], dtype=np.float64) for k in args.direct}

mask = np.isfinite(y)
for X in list(models.values()) + list(direct.values()) + [feat(k) for k in args.require]:
    if X is not None:
        mask &= np.isfinite(X).reshape(N, -1).all(1)
for w in args.where:
    k_, lo_, hi_ = w.split(":")
    v_ = np.asarray(np.asarray(z[k_], dtype=np.float64), dtype=float).reshape(N, -1)[:, 0]
    mask &= np.isfinite(v_) & (v_ >= float(lo_)) & (v_ <= float(hi_))
up = np.unique(pid[mask])
rng = np.random.default_rng(0)
fold_of = dict(zip(rng.permutation(up), np.arange(len(up)) % 5))
fold = np.array([fold_of.get(p, -1) for p in pid])
print(f"{args.label}: windows {mask.sum()}/{N}  patients {len(up)}  median {np.median(y[mask]):.3f} "
      f"IQR [{np.percentile(y[mask], 25):.3f},{np.percentile(y[mask], 75):.3f}]"
      + (f"  pos(>={args.thr:g}) {np.mean(y[mask] >= args.thr) * 100:.1f}%" if args.thr is not None else "")
      + ("  [patient-agg]" if args.patient_agg else ""), flush=True)


BIN = set(np.unique(y[mask]).tolist()) <= {0.0, 1.0}


def lp(xtr, ytr, xva, yva, xte, epochs=2000, patience=200, max_steps=300000, lr=1e-3, bs=512, seed=42):
    """run.py LinearProbe 레시피. BIN 이면 BCE·val AUROC, 아니면 표준화 MSE·val MAE."""
    import torch, torch.nn as nn
    torch.manual_seed(seed); dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = (0.0, 1.0) if BIN else (float(ytr.mean()), float(ytr.std() or 1.0))
    T = lambda a: torch.tensor(np.asarray(a, np.float32), device=dev)
    Xtr, Ytr, Xva, Xte = T(xtr), T((ytr - mu) / sd), T(xva), T(xte)
    m = nn.Sequential(nn.LayerNorm(Xtr.shape[1]), nn.Dropout(0.1), nn.Linear(Xtr.shape[1], 1)).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=lr); crit = nn.BCEWithLogitsLoss() if BIN else nn.MSELoss()
    best, state, bad, steps = -np.inf, None, 0, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xtr), device=dev)
        for i in range(0, len(Xtr), bs):
            j = perm[i:i + bs]; opt.zero_grad(); crit(m(Xtr[j]).squeeze(1), Ytr[j]).backward(); opt.step(); steps += 1
        m.eval()
        with torch.no_grad(): pv = m(Xva).squeeze(1).cpu().numpy()
        v = (roc_auc_score(yva, pv) if 0 < yva.sum() < len(yva) else -np.inf) if BIN else -np.mean(np.abs(pv * sd + mu - yva))
        if v > best: best, state, bad = v, {a: t.clone() for a, t in m.state_dict().items()}, 0
        else: bad += 1
        if bad >= patience or steps >= max_steps: break
    m.load_state_dict(state); m.eval()
    with torch.no_grad(): o = m(Xte).squeeze(1)
    return (torch.sigmoid(o) if BIN else o * sd + mu).cpu().numpy()


def oof(X):
    pred = np.full(N, np.nan)
    for k in range(5):
        te = (fold == k) & mask; va = (fold == (k + 1) % 5) & mask; tr = (fold != k) & (fold != (k + 1) % 5) & mask
        if X is None:
            pred[te] = y[tr | va].mean(); continue
        pred[te] = lp(X[tr], y[tr], X[va], y[va], X[te])
    return pred


preds = {k: oof(X) for k, X in models.items()}
preds.update(direct)
idx = np.where(mask)[0]
_, gcode = np.unique(pid[idx], return_inverse=True)
yy = y[idx]
P = {k: v[idx] for k, v in preds.items()}
G = gcode.max() + 1
if args.patient_agg:  # 환자별 평균으로 축약 → 단위 = 환자
    cnt = np.bincount(gcode, minlength=G)
    yy = np.bincount(gcode, yy, G) / cnt
    P = {k: np.bincount(gcode, v, G) / cnt for k, v in P.items()}
    gcode = np.arange(G)
members = [[] for _ in range(G)]
for i, gc in enumerate(gcode):
    members[gc].append(i)
members = [np.array(m) for m in members]


def metrics(p, yv, gv):
    e = p - yv
    out = {"mae": float(np.abs(e).mean()), "r": float(np.corrcoef(p, yv)[0, 1]) if p.std() > 0 else 0.0,
           "tol": float(np.mean(np.abs(e) <= args.tol))}
    g_, inv = np.unique(gv, return_inverse=True)
    c = np.bincount(inv)
    pc = p - (np.bincount(inv, p) / c)[inv]; yc = yv - (np.bincount(inv, yv) / c)[inv]
    out["within_r"] = float(np.corrcoef(pc, yc)[0, 1]) if pc.std() > 1e-12 and yc.std() > 1e-12 else 0.0
    if args.thr is not None:
        lab = (yv < args.thr) if args.lt else (yv >= args.thr); s = -p if args.lt else p
        ok = 0 < lab.sum() < len(lab)
        out["auroc"] = float(roc_auc_score(lab, s)) if ok else np.nan
        out["auprc"] = float(average_precision_score(lab, s)) if ok else np.nan
    return out


point = {k: metrics(v, yy, gcode) for k, v in P.items()}
br = np.random.default_rng(1)
boots = []
for _ in range(args.boot):
    pick = br.integers(0, G, G)
    ii = np.concatenate([members[p] for p in pick])
    gg = np.concatenate([np.full(len(members[p]), j) for j, p in enumerate(pick)])  # 복원추출 중복 환자 = 별개 클러스터
    boots.append((ii, gg))
bs = {k: [metrics(v[ii], yy[ii], gg) for ii, gg in boots] for k, v in P.items()}
mets = ["mae", "r", "within_r", "tol"] + (["auroc", "auprc"] if args.thr is not None else [])
res = {"label": args.label, "n_windows": int(mask.sum()), "n_patients": int(G), "patient_agg": args.patient_agg, "models": {}}
for k in P:
    row = {m: [point[k][m], float(np.nanpercentile([b[m] for b in bs[k]], 2.5)), float(np.nanpercentile([b[m] for b in bs[k]], 97.5))] for m in mets}
    for ref in args.refs:
        if ref == k or ref not in P:
            continue
        for m in ("mae", "r", "within_r") + (("auroc",) if args.thr is not None else ()):
            d = np.array([bs[k][i][m] - bs[ref][i][m] for i in range(args.boot)])
            d = d[np.isfinite(d)]
            pv = float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))) if len(d) else np.nan  # 양측 bootstrap p
            row[f"d_{m}_vs_{ref}"] = [point[k][m] - point[ref][m], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), pv]
    res["models"][k] = row

f3 = lambda v: f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"
hdr = f"{'model':<26}{'MAE ' + args.unit:>26}{'r':>22}{'within-r':>22}{'|err|<=' + format(args.tol, 'g'):>22}" + (f"{'AUROC':>22}{'AUPRC':>22}" if args.thr is not None else "")
print(hdr)
for k, row in res["models"].items():
    print(f"{k:<26}{f3(row['mae']):>26}{f3(row['r']):>22}{f3(row['within_r']):>22}{f3(row['tol']):>22}" + (f"{f3(row['auroc']):>22}{f3(row['auprc']):>22}" if args.thr is not None else ""))
print("\npaired Δ (model − ref), patient-cluster bootstrap 95% CI (* = excludes 0)")
for k, row in res["models"].items():
    for key, v in row.items():
        if key.startswith("d_"):
            sig = "*" if (v[1] > 0 or v[2] < 0) else " "
            print(f"  {k:<26}{key:<32}{v[0]:+.3f} [{v[1]:+.3f},{v[2]:+.3f}]{sig}")
json.dump(res, open(args.out, "w"), indent=1)
np.savez(args.out.rsplit(".", 1)[0] + "_oof.npz", y=yy, g=gcode, idx=idx, pid=pid[idx], **{f"p_{k}": v for k, v in P.items()})
print("LP-EVAL-DONE")
