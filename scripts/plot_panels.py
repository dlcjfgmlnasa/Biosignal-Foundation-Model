# -*- coding:utf-8 -*-
"""코퍼스 그림 개별 패널 EPS 렌더 (hours 막대 / gender 원그래프).

``figure_corpus_stats.json`` (scripts/figure_corpus_stats.py 산출) 을 읽어
독립 패널을 각각 EPS/PNG 로 저장한다. 메인 합본(plot_corpus_figure.py)과 별개로,
논문에서 패널을 따로 쓰고 싶을 때 사용.

  --panel hours   : 모달리티별 시간(hours) 막대 — 데이터 '양'의 정확한 지표
  --panel gender  : 성별 원그래프 (EMR patients.csv 필요)
  --panel both    : 둘 다

사용:
    python -m scripts.plot_panels --stats figure_corpus_stats.json --panel both --fmt eps
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["ps.fonttype"] = 42   # EPS/PDF Type42 임베드 (투고용)
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

MOD_ORDER = ["ECG", "ABP", "PPG", "CVP", "CO2", "AWP", "ICP",
             "RESP_Impedance", "RESP_Flow"]
MOD_COLORS = {
    "ECG": "#e15759", "ABP": "#f28e2b", "PPG": "#4e79a7", "CVP": "#76b7b2",
    "CO2": "#59a14f", "AWP": "#edc948", "ICP": "#b07aa1",
    "RESP_Impedance": "#9c755f", "RESP_Flow": "#ff9da7",
}
GENDER_COLORS = {"M": "#4e79a7", "MALE": "#4e79a7", "남": "#4e79a7",
                 "F": "#e8a0b8", "FEMALE": "#e8a0b8", "여": "#e8a0b8"}


def plot_hours(st: dict, out: Path, pct: bool = False) -> None:
    per = {m["name"]: m["hours"] for m in st["per_modality"]}
    names = [n for n in MOD_ORDER if n in per]
    total_h = st["total_hours"]
    if pct:
        vals = [per[n] / total_h * 100 for n in names]
        ylabel, vfmt = "% of total hours", lambda v: f"{v:.1f}"
    else:
        vals = [per[n] / 1e3 for n in names]   # ×10³ h
        ylabel, vfmt = "Hours (×10³)", lambda v: f"{v:.0f}"
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    bars = ax.bar(range(len(names)), vals,
                  color=[MOD_COLORS[n] for n in names])
    ax.set_title("Hours per Modality", fontsize=15, weight="bold")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("RESP_Impedance", "RespImp")
                        .replace("RESP_Flow", "RespFlow").replace("CO2", "CO₂")
                        for n in names], rotation=45, ha="right", fontsize=10)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, vfmt(v),
                ha="center", va="bottom", fontsize=8, color="#333")
    ax.spines[["top", "right"]].set_visible(False)
    note = (f"Total {total_h/1e3:,.0f}×10³ h" if not pct
            else f"Total {total_h/1e3:,.0f}×10³ h = 100%")
    ax.text(0.98, 0.95, note, transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#666")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"[saved] {out}")


def plot_gender(st: dict, out: Path) -> bool:
    emr = st.get("emr", {})
    g = emr.get("gender_wave_cohort") or emr.get("gender_all")
    if not g:
        print("[skip] gender: EMR 성별 데이터 없음 (emr.available=false). "
              "figure_corpus_stats.py 를 --emr-dir 와 함께 돌려 patients.csv 를 "
              "읽어야 합니다. 가짜 수치로 그리지 않습니다.")
        return False
    # 정규화 (대소문자/표기 흡수)
    norm: dict[str, int] = {}
    for k, v in g.items():
        kk = (k or "?").strip().upper()
        label = "Male" if kk in ("M", "MALE", "남") else \
                "Female" if kk in ("F", "FEMALE", "여") else (k or "?")
        norm[label] = norm.get(label, 0) + int(v)
    labels = list(norm.keys())
    vals = list(norm.values())
    total = sum(vals)
    colors = [GENDER_COLORS.get(l.upper(), "#bbbbbb") if l == "Male"
              else ("#e8a0b8" if l == "Female" else "#bbbbbb") for l in labels]
    fig, ax = plt.subplots(figsize=(4.6, 4.6))
    ax.set_title("Sex Distribution", fontsize=15, weight="bold")
    scope = "wave cohort" if emr.get("gender_wave_cohort") else "all deceased"
    ax.pie(vals, labels=labels, colors=colors,
           autopct=lambda p: f"{p:.1f}%\n({int(round(p*total/100)):,})",
           wedgeprops=dict(width=0.42, edgecolor="white"),
           textprops=dict(fontsize=11), startangle=90)
    ax.text(0, -1.25, f"N = {total:,}  ({scope})", ha="center", fontsize=10,
            color="#666")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print(f"[saved] {out}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", required=True, type=Path)
    ap.add_argument("--panel", choices=["hours", "gender", "both"], default="both")
    ap.add_argument("--fmt", default="eps", help="출력 포맷 (eps/pdf/png)")
    ap.add_argument("--prefix", default="panel", help="출력 파일 prefix")
    ap.add_argument("--pct", action="store_true", help="hours 막대를 % 로")
    args = ap.parse_args()

    st = json.loads(args.stats.read_text(encoding="utf-8"))
    if args.panel in ("hours", "both"):
        plot_hours(st, Path(f"{args.prefix}_hours.{args.fmt}"), pct=args.pct)
    if args.panel in ("gender", "both"):
        plot_gender(st, Path(f"{args.prefix}_gender.{args.fmt}"))


if __name__ == "__main__":
    main()
