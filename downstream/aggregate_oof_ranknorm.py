# -*- coding: utf-8 -*-
"""Rank-normalized OOF pooling + patient-cluster bootstrap CI.

draft Table 3 의 headline 수치를 생성하는 스크립트다.

**Provenance**: 원본은 세션 scratchpad(`agg_all2.py`)에만 존재해 소실 위험이 있었다.
2026-07-10 저장소로 이관. 로직은 원본과 동일하며, 하드코딩된 `ROOT` 만 환경변수로
override 가능하게 했다(서버 실행용).

왜 rank-norm 인가
-----------------
fold 별 score 스케일이 어긋나면(miscalibration) raw concat pooled AUROC 가 fold-mean
보다 눌린다 (ICH 5min: raw 0.815 vs fold-mean 0.850). pooling **전에** fold 별로
``rankdata(score)/(n+1)`` 을 적용하면 pooled ≈ fold-mean 으로 수렴한다. AUROC 는
fold 내 monotonic 변환에 불변이므로 per-fold 값은 바뀌지 않고 pooled 만 교정된다.

⚠ `downstream/_eval_utils.py` 의 ``evaluate_oof()`` 는 ``concat_oof()`` 로 **raw
concat** 하므로 이 스크립트와 다른 값을 낸다. Table 3 대표값은 여기서 나온
``pooled RANK`` 이다.

사용법
------
    python -m downstream.aggregate_oof_ranknorm
    RESULT_ROOT=/home/coder/workspace/k-mimic-/bio_fm/result python -m downstream.aggregate_oof_ranknorm

``ROOT`` 하위를 재귀 탐색해 ``fold0.json`` 이 있는 모든 디렉토리를 집계한다.
각 ``fold{f}.json`` 은 ``y_true`` + (``y_score`` | ``y_probs``) + (``patient_ids`` |
``patient_id``) 를 담아야 한다.
"""
import json, glob, os, sys
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = os.environ.get(
    "RESULT_ROOT", r"C:/Projects/Biosignal-Foundation-Model/obsidian/Result"
)

def auroc(yt, ys):
    if ys.ndim == 1:
        return roc_auc_score(yt, ys)
    vals = []
    for k in range(ys.shape[1]):
        yk = (yt == k).astype(int)
        if 0 < yk.sum() < len(yk):
            vals.append(roc_auc_score(yk, ys[:, k]))
    return float(np.mean(vals))

def auprc(yt, ys):
    if ys.ndim == 1:
        return average_precision_score(yt, ys)
    vals = [average_precision_score((yt == k).astype(int), ys[:, k]) for k in range(ys.shape[1])]
    return float(np.mean(vals))

def ranknorm_fold(ys):
    if ys.ndim == 1:
        return rankdata(ys, method="average") / (len(ys) + 1.0)
    out = np.empty_like(ys, dtype=float)
    for k in range(ys.shape[1]):
        out[:, k] = rankdata(ys[:, k], method="average") / (len(ys) + 1.0)
    return out

def boot_all(yt, raw, rn, pid, n_iter=1000, seed=42, ci=0.95):
    """One resample loop -> AUROC/AUPRC for both raw & rank-norm scores."""
    pt = dict(raw_au=auroc(yt, raw), raw_ap=auprc(yt, raw),
              rn_au=auroc(yt, rn), rn_ap=auprc(yt, rn))
    upids = np.unique(pid)
    p2i = {p: np.where(pid == p)[0] for p in upids}
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in ["raw_au", "raw_ap", "rn_au", "rn_ap"]}
    for _ in range(n_iter):
        s = rng.choice(upids, size=len(upids), replace=True)
        idx = np.concatenate([p2i[p] for p in s])
        y = yt[idx]; r = raw[idx]; n = rn[idx]
        try:
            acc["raw_au"].append(auroc(y, r)); acc["raw_ap"].append(auprc(y, r))
            acc["rn_au"].append(auroc(y, n));  acc["rn_ap"].append(auprc(y, n))
        except Exception:
            continue
    a = (1 - ci) / 2
    out = {}
    for k in acc:
        arr = np.array([v for v in acc[k] if np.isfinite(v)])
        out[k] = (pt[k], float(np.percentile(arr, 100*a)), float(np.percentile(arr, 100*(1-a))))
    return out

def compute(base, root=ROOT):
    """한 leaf(fold*.json 디렉토리)의 headline 지표를 dict 로 반환.

    ``run()`` 이 출력하던 값과 동일하되, 인쇄 대신 구조화 결과를 돌려줘
    ``rank_results.py`` 등에서 재사용할 수 있게 분리한 함수다. fold json 이
    없으면 ``None``. 반환 dict 의 ``rn_au``/``rn_ap``/``raw_au``/``raw_ap`` 는
    ``(point, ci_lo, ci_hi)`` 튜플이다 (Table 3 대표값 = ``rn_au``).
    """
    folds = sorted(glob.glob(os.path.join(base, "fold*.json")))
    if not folds:
        return None
    yt_all=[]; raw_all=[]; rn_all=[]; pid_all=[]; per_au=[]; per_ap=[]; npos=0
    for i, f in enumerate(folds):
        d = json.load(open(f))
        yt = np.array(d["y_true"])
        skey = "y_score" if "y_score" in d else "y_probs"
        ys = np.array(d[skey], dtype=float)
        pk = "patient_ids" if "patient_ids" in d else ("patient_id" if "patient_id" in d else None)
        # fold prefix: OOF CV 에서 환자는 정확히 한 fold 의 test 에만 등장하므로
        # prefix 를 붙여도 cluster 정의가 깨지지 않는다 (fold 간 ID 충돌만 방지).
        pid = np.array([f"f{i}_{p}" for p in d[pk]]) if pk else np.array([f"f{i}_{j}" for j in range(len(yt))])
        per_au.append(auroc(yt, ys)); per_ap.append(auprc(yt, ys))
        yt_all.append(yt); raw_all.append(ys); rn_all.append(ranknorm_fold(ys)); pid_all.append(pid)
        npos += int((yt > 0).sum())
    yt=np.concatenate(yt_all); raw=np.concatenate(raw_all); rn=np.concatenate(rn_all); pid=np.concatenate(pid_all)
    o = boot_all(yt, raw, rn, pid)
    rel = os.path.relpath(base, root).replace("\\", "/")
    return dict(
        rel=rel, n=int(len(yt)), pos=int(npos),
        patients=int(len(np.unique(pid))), prev=float((yt > 0).mean()),
        n_folds=len(folds),
        fold_mean_au=float(np.mean(per_au)), fold_sd_au=float(np.std(per_au)),
        fold_mean_ap=float(np.mean(per_ap)), fold_sd_ap=float(np.std(per_ap)),
        raw_au=o["raw_au"], raw_ap=o["raw_ap"], rn_au=o["rn_au"], rn_ap=o["rn_ap"],
    )


def run(base):
    r = compute(base)
    if r is None:
        return
    print("### %s" % r["rel"])
    print("  n=%d pos=%d patients=%d prev=%.4f" % (r["n"], r["pos"], r["patients"], r["prev"]))
    print("  fold-mean   AUROC=%.3f±%.3f  AUPRC=%.3f±%.3f" % (r["fold_mean_au"], r["fold_sd_au"], r["fold_mean_ap"], r["fold_sd_ap"]))
    print("  pooled RAW  AUROC=%.3f (%.3f-%.3f)  AUPRC=%.3f (%.3f-%.3f)" % (*r["raw_au"], *r["raw_ap"]))
    print("  pooled RANK AUROC=%.3f (%.3f-%.3f)  AUPRC=%.3f (%.3f-%.3f)" % (*r["rn_au"], *r["rn_ap"]))
    print(flush=True)


def main():
    dirs = sorted(set(os.path.dirname(p) for p in glob.glob(os.path.join(ROOT, "**", "fold0.json"), recursive=True)))
    for b in dirs:
        run(b)


if __name__ == "__main__":
    main()
