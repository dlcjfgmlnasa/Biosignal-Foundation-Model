# -*- coding: utf-8 -*-
"""lp_eval 결과 JSON 의 paired Δ p값을 모아 Benjamini–Hochberg 보정 (논문 Methods: 결과 절 단위·지표별).

lp_eval 은 비교마다 ``models[<model>]["d_<metric>_vs_<ref>"] = [Δ, lo, hi, p]`` (양측 bootstrap p) 를 저장한다.
같은 결과 절(section)에 속한 JSON 들의 같은 지표 p값을 한 묶음으로 BH 보정하고, q < alpha 를 유의로 표시한다.

    python -m downstream.baselines.bh_agg \
        --files detection=xgen/lp/af_af.json detection=xgen/lp/cinc_ev.json prediction=xgen/lp/etco2_3m.json \
        --out bh.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def bh(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q값 (단조 보정 포함)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    if m == 0:
        return p
    o = np.argsort(p)
    q = p[o] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[o] = np.minimum(q, 1.0)
    return out


def collect(files: list[str]) -> list[dict]:
    rows = []
    for spec in files:
        sec, path = spec.split("=", 1)
        res = json.load(open(path, encoding="utf-8"))
        task = res.get("label", os.path.basename(path))
        for model, row in res["models"].items():
            for key, v in row.items():
                if not key.startswith("d_") or len(v) < 4 or v[3] is None or not np.isfinite(v[3]):
                    continue
                metric, ref = key[2:].split("_vs_", 1)
                rows.append(dict(section=sec, file=os.path.basename(path), task=task, model=model, ref=ref,
                                 metric=metric, delta=v[0], lo=v[1], hi=v[2], p=v[3]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="section=lp_eval 결과.json")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = collect(a.files)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["section"], r["metric"])].append(i)
    for idx in groups.values():
        for i, q in zip(idx, bh([rows[i]["p"] for i in idx])):
            rows[i]["q"] = float(q)
            rows[i]["sig"] = bool(q < a.alpha)

    for (sec, met), idx in sorted(groups.items()):
        print(f"== {sec} · {met} (BH, m={len(idx)})")
        for i in idx:
            r = rows[i]
            print(f"  {r['file']:<24}{r['model']:<16}vs {r['ref']:<14}Δ {r['delta']:+.3f} [{r['lo']:+.3f},{r['hi']:+.3f}]"
                  f"  p {r['p']:.4f}  q {r['q']:.4f}{' *' if r['sig'] else ''}")
    json.dump(rows, open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("BH-DONE", a.out)


if __name__ == "__main__":
    # 자가검증: R p.adjust(c(.01,.04,.03,.005), "BH") = .02 .04 .04 .02
    assert np.allclose(bh([0.01, 0.04, 0.03, 0.005]), [0.02, 0.04, 0.04, 0.02])
    main()
