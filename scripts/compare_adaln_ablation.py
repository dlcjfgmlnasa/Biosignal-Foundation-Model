"""Compare baseline vs AdaLN ablation training logs.

Usage:
    python scripts/compare_adaln_ablation.py \\
        --baseline outputs/ablation_adaln/baseline/training_log.csv \\
        --adaln    outputs/ablation_adaln/adaln_dc16/training_log.csv
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path


def load(p: str | Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[int] = set()
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                ep = int(r["epoch"])
            except (KeyError, ValueError):
                continue
            if ep in seen:
                continue
            seen.add(ep)
            parsed: dict = {}
            for k, v in r.items():
                try:
                    parsed[k] = float(v)
                except (TypeError, ValueError):
                    parsed[k] = v
            rows.append(parsed)
    rows.sort(key=lambda d: d["epoch"])
    return rows


def col(rows: list[dict], k: str) -> list[float]:
    return [r[k] for r in rows if isinstance(r.get(k), float)]


def fmt(x: float | None, w: int = 9, prec: int = 4) -> str:
    return f"{x:>{w}.{prec}f}" if x is not None else " " * w


def summarize(rows: list[dict], label: str) -> dict:
    vt = col(rows, "val_total")
    vm = col(rows, "val_masked")
    vn = col(rows, "val_next")
    n = max(1, len(vt))
    last_k = min(3, n)
    return {
        "label": label,
        "n_epochs": len(rows),
        "best_val_total": min(vt) if vt else None,
        "best_epoch": (vt.index(min(vt)) if vt else None),
        "final_val_total": vt[-1] if vt else None,
        "final_val_masked": vm[-1] if vm else None,
        "final_val_next": vn[-1] if vn else None,
        "lastk_val_total": st.mean(vt[-last_k:]) if vt else None,
        "lastk_val_masked": st.mean(vm[-last_k:]) if vm else None,
        "lastk_val_next": st.mean(vn[-last_k:]) if vn else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--adaln", required=True)
    args = ap.parse_args()

    base_rows = load(args.baseline)
    ada_rows = load(args.adaln)
    if not base_rows or not ada_rows:
        raise SystemExit("Empty training_log.csv on one or both sides — runs may not have started.")

    n = min(len(base_rows), len(ada_rows))

    print(f"{'Ep':>3} | {'base_train':>10} {'ada_train':>10} {'D':>7}"
          f" | {'base_val':>9} {'ada_val':>9} {'D':>7}"
          f" | {'b_mask':>8} {'a_mask':>8} | {'b_next':>8} {'a_next':>8}")
    print("-" * 110)
    for i in range(n):
        b, a = base_rows[i], ada_rows[i]
        bt, at = b.get("train_total"), a.get("train_total")
        bv, av = b.get("val_total"), a.get("val_total")
        bm, am = b.get("val_masked"), a.get("val_masked")
        bn, an = b.get("val_next"), a.get("val_next")
        d_train = (at - bt) if (isinstance(bt, float) and isinstance(at, float)) else None
        d_val = (av - bv) if (isinstance(bv, float) and isinstance(av, float)) else None
        ds = lambda x: f"{x:+7.3f}" if isinstance(x, float) else " " * 7
        print(f"{i:3d} | {fmt(bt, 10)} {fmt(at, 10)} {ds(d_train)}"
              f" | {fmt(bv)} {fmt(av)} {ds(d_val)}"
              f" | {fmt(bm, 8)} {fmt(am, 8)} | {fmt(bn, 8)} {fmt(an, 8)}")

    print()
    print("Summary")
    print("-" * 70)
    base_s = summarize(base_rows, "baseline")
    ada_s = summarize(ada_rows, "adaln_dc16")
    for s in (base_s, ada_s):
        print(
            f"  {s['label']:<12}: epochs={s['n_epochs']:2d}"
            f"  best_val={fmt(s['best_val_total'])} @ ep {s['best_epoch']}"
            f"  final_val={fmt(s['final_val_total'])}"
            f"  final_mask={fmt(s['final_val_masked'])}"
            f"  final_next={fmt(s['final_val_next'])}"
        )

    print()
    print("Delta (adaln - baseline; negative = AdaLN better)")
    pairs = [
        ("best_val_total", base_s["best_val_total"], ada_s["best_val_total"]),
        ("final_val_total", base_s["final_val_total"], ada_s["final_val_total"]),
        ("final_val_masked", base_s["final_val_masked"], ada_s["final_val_masked"]),
        ("final_val_next", base_s["final_val_next"], ada_s["final_val_next"]),
        ("lastk_val_masked", base_s["lastk_val_masked"], ada_s["lastk_val_masked"]),
        ("lastk_val_next", base_s["lastk_val_next"], ada_s["lastk_val_next"]),
    ]
    for name, b, a in pairs:
        if isinstance(b, float) and isinstance(a, float):
            rel = (a - b) / b * 100 if b else 0.0
            print(f"  {name:<18}: {a - b:+.4f}  ({rel:+.1f}%)")
        else:
            print(f"  {name:<18}: n/a")


if __name__ == "__main__":
    main()