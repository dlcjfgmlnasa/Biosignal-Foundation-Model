# -*- coding:utf-8 -*-
"""연령×성별 인구 피라미드 (population pyramid).

patients.csv (또는 동등 파일) + 선택적 subject_id 필터(파형 코호트)로
5세 단위 age band × sex 교차표를 만들어, 남(좌)·여(우) 등지는 가로 막대로 렌더.
de-id 된 anchor_age (5세 밴드 문자열) 를 그대로 사용.

사용:
    python -m scripts.plot_age_sex_pyramid --patients patients.csv \
        --ids wave_ids.json --out pyramid.eps
    # 필터 없이 전체 코호트
    python -m scripts.plot_age_sex_pyramid --patients patients.csv --out pyramid.eps
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["pdf.fonttype"] = 42
import matplotlib.pyplot as plt

# 5세 밴드 정렬 순서 (아래=어림 → 위=많음)
BAND_ORDER = ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34",
              "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69",
              "70-74", "75-79", "80-84", "85-89", "90 years or older"]
BAND_LABEL = {b: ("90+" if "90" in b else b) for b in BAND_ORDER}
M_COLOR, F_COLOR = "#4e79a7", "#e8a0b8"


def load(patients: Path, ids_path: Path | None):
    keep = None
    if ids_path:
        keep = set(json.loads(ids_path.read_text()))
    seen, recs = set(), []
    with open(patients, encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            sid = re.sub(r"\D", "", r.get("subject_id", ""))
            if not sid:
                continue
            sid = str(int(sid))
            if keep is not None and sid not in keep:
                continue
            if sid in seen:
                continue
            seen.add(sid)
            recs.append(r)
    return recs


def render(recs, out: Path):
    cross = defaultdict(lambda: {"M": 0, "F": 0})
    for r in recs:
        band = (r.get("anchor_age") or "").strip()
        sex = (r.get("sex") or "").strip().upper()
        if band in BAND_LABEL and sex in ("M", "F"):
            cross[band][sex] += 1
    bands = [b for b in BAND_ORDER if b in cross]
    males = [cross[b]["M"] for b in bands]
    females = [cross[b]["F"] for b in bands]
    n = len(bands)
    y = range(n)
    tot_m, tot_f = sum(males), sum(females)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(y, [-m for m in males], color=M_COLOR, label=f"Male (n={tot_m:,})")
    ax.barh(y, females, color=F_COLOR, label=f"Female (n={tot_f:,})")
    ax.set_yticks(list(y))
    ax.set_yticklabels([BAND_LABEL[b] for b in bands], fontsize=9)
    ax.set_xlabel("Patients")
    ax.set_title("Age × Sex Distribution", fontsize=15, weight="bold")
    # x축 음수 → 절대값 라벨
    xmax = max(max(males), max(females))
    step = 20 if xmax <= 120 else 50
    ticks = list(range(-((xmax // step + 1) * step), (xmax // step + 1) * step + 1, step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(abs(t)) for t in ticks], fontsize=9)
    ax.axvline(0, color="#888", lw=0.8)
    ax.legend(loc="lower right", fontsize=10, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.0, 1.005, f"N = {tot_m + tot_f:,}", transform=ax.transAxes,
            fontsize=9, color="#666")
    fig.tight_layout()
    fig.savefig(out, facecolor="white", dpi=150)
    print(f"[saved] {out}  (M={tot_m}, F={tot_f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", required=True, type=Path)
    ap.add_argument("--ids", type=Path, default=None,
                    help="파형 코호트 subject_id JSON 리스트 (없으면 전체)")
    ap.add_argument("--out", type=Path, default=Path("pyramid.eps"))
    a = ap.parse_args()
    render(load(a.patients, a.ids), a.out)


if __name__ == "__main__":
    main()
