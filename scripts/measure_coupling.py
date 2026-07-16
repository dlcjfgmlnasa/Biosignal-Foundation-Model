# -*- coding: utf-8 -*-
"""경험적 cross-modal coupling prior 실측 — 전체 9 signal_type (KHDP shards/processed).

manifest_full.jsonl 을 직접 읽어 같은 (subject, session) 안에서 시간(start_sample)
이 겹치는 recording 들을 정렬, 쌍별 A-patch -> B-patch 선형(Ridge) 예측 test R² 를
coupling weight W[A,B] 로 측정한다. z-score(패치별) 후 SHAPE 결합 측정.

- self-contained (numpy/torch/sklearn 만). repo import 불필요.
- 여러 소스 corpus 지원: --manifest 를 여러 개 주면 합산 (K-MIMIC + SNUH).
- file_ref 에 '#' (sharded 가상경로) 이면 스킵 → processed/ 레이아웃(실제 .pt) 필요.
  sharded 전용 corpus 면 먼저 processed 레이아웃(per-recording .pt)을 가리킬 것.

사용:
  python scripts/measure_coupling.py \
      --manifest /path/to/processed/<ds1>/manifest_full.jsonl \
                 /path/to/processed/<ds2>/manifest_full.jsonl \
      --out coupling_full.json --max-subjects 800
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

PATCH = 200  # 2s @ 100Hz (CARMEN patch)
SIGNAL_TYPE_NAMES = {
    0: "ecg", 1: "abp", 2: "ppg", 3: "cvp", 4: "co2",
    5: "awp", 6: "icp", 7: "resp_imp", 8: "resp_flow",
}


def zscore(x):  # (n, PATCH)
    m = x.mean(-1, keepdims=True)
    s = x.std(-1, keepdims=True) + 1e-6
    return (x - m) / s


def load_signal(path):
    """.pt -> (T,) float32. 다채널이면 channel 0 (primary lead)."""
    t = torch.load(path, weights_only=True)
    if hasattr(t, "numpy"):
        t = t.numpy()
    t = np.asarray(t, dtype=np.float32)
    if t.ndim > 1:
        t = t[0]
    return t


def collect_recordings(manifest_paths):
    """여러 manifest_full.jsonl -> {(subject,session): [rec,...]}."""
    groups = defaultdict(list)
    n_subj = 0
    for mp in manifest_paths:
        mp = Path(mp)
        base = mp.parent
        with open(mp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                subj = json.loads(line)
                sid = subj["subject_id"]
                n_subj += 1
                for sess in subj.get("sessions", []):
                    ss = str(sess.get("session_id", ""))
                    for rec in sess.get("recordings", []):
                        fref = rec.get("file", "")
                        if not fref or "#" in fref:  # sharded 가상경로 스킵
                            continue
                        groups[(str(mp), sid, ss)].append({
                            "st": int(rec["signal_type"]),
                            "start": int(rec.get("start_sample", 0)),
                            "n": int(rec["n_timesteps"]),
                            "path": str(base / sid / fref),
                        })
    return groups, n_subj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True,
                    help="manifest_full.jsonl 경로(들)")
    ap.add_argument("--out", default="coupling_full.json")
    ap.add_argument("--max-subjects", type=int, default=800,
                    help="처리할 최대 (subject,session) 그룹 수")
    ap.add_argument("--max-patch-per-pair-session", type=int, default=40)
    ap.add_argument("--cap-per-pair", type=int, default=20000, help="쌍별 patch 상한")
    ap.add_argument("--ridge-alpha", type=float, default=1.0)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    groups, n_subj = collect_recordings(args.manifest)
    print(f"[coupling] subjects={n_subj}  groups={len(groups)}", flush=True)

    aligned = defaultdict(lambda: [[], []])  # (A,B) -> [Xa_list, Xb_list]
    counts_st = defaultdict(int)
    n_grp = 0
    for key, recs in groups.items():
        if n_grp >= args.max_subjects:
            break
        n_grp += 1
        # 그룹 내 신호를 한 번씩만 로드 (path 캐시)
        cache = {}
        for r in recs:
            counts_st[r["st"]] += 1
        for ia in range(len(recs)):
            for ib in range(len(recs)):
                a, b = recs[ia], recs[ib]
                if a["st"] == b["st"]:
                    continue
                pair = (a["st"], b["st"])
                if sum(len(x) for x in aligned[pair][0]) >= args.cap_per_pair:
                    continue
                s = max(a["start"], b["start"])
                e = min(a["start"] + a["n"], b["start"] + b["n"])
                if e - s < PATCH:
                    continue
                try:
                    if a["path"] not in cache:
                        cache[a["path"]] = load_signal(a["path"])
                    if b["path"] not in cache:
                        cache[b["path"]] = load_signal(b["path"])
                    sa = cache[a["path"]][s - a["start"]: e - a["start"]]
                    sb = cache[b["path"]][s - b["start"]: e - b["start"]]
                except Exception:  # noqa: BLE001
                    continue
                L = (min(len(sa), len(sb)) // PATCH) * PATCH
                if L < PATCH:
                    continue
                Pa = sa[:L].reshape(-1, PATCH)
                Pb = sb[:L].reshape(-1, PATCH)
                valid = np.isfinite(Pa).all(1) & np.isfinite(Pb).all(1)
                idx = np.nonzero(valid)[0]
                if len(idx) == 0:
                    continue
                if len(idx) > args.max_patch_per_pair_session:
                    idx = rng.choice(idx, args.max_patch_per_pair_session, replace=False)
                aligned[pair][0].append(zscore(Pa[idx]))
                aligned[pair][1].append(zscore(Pb[idx]))
        cache.clear()
        if n_grp % 100 == 0:
            print(f"[coupling] groups {n_grp}/{min(len(groups), args.max_subjects)}", flush=True)

    print(f"[coupling] recording count per signal_type: "
          f"{ {SIGNAL_TYPE_NAMES.get(k, k): v for k, v in sorted(counts_st.items())} }", flush=True)
    print(f"{'pair':16s} {'R2(A->B)':>9s} {'n_patch':>9s}", flush=True)
    results = {}
    for (a, b), (Xs, Ys) in sorted(aligned.items()):
        if not Xs:
            continue
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        if len(X) < 100:
            continue
        Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3, random_state=42)
        r = Ridge(alpha=args.ridge_alpha).fit(Xtr, Ytr)
        r2 = float(r2_score(Yte, r.predict(Xte)))
        an = SIGNAL_TYPE_NAMES.get(a, a)
        bn = SIGNAL_TYPE_NAMES.get(b, b)
        results[f"{an}->{bn}"] = {"r2": round(r2, 4), "n": int(len(X)), "a": a, "b": b}
        print(f"{an+'->'+bn:16s} {r2:9.4f} {len(X):9d}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"[coupling] saved {args.out} ({len(results)} pairs)", flush=True)


if __name__ == "__main__":
    main()
