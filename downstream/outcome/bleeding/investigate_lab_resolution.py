#!/usr/bin/env python
"""VitalDB lab_data.csv 의 Hb/Hct 시간 해상도 조사.

목적: 출혈/대량수혈 "직전 예측(B)" 라벨이 lab_data 의 Hb/Hct 급락 시점으로
      만들어질 수 있는지 — 케이스당 측정 횟수·측정 간격 분포로 타당성 점검.

서버 실행 예:
    python -m downstream.outcome.bleeding.investigate_lab_resolution \
        --lab-csv /home/coder/workspace/datasets/vitaldb_open/1.0.0/lab_data.csv \
        --clinical-csv /home/coder/workspace/datasets/vitaldb_open/1.0.0/clinical_data.csv

clinical-csv 는 선택 — 주면 opnd(수술 종료) 기준 intraop 구간만 따로 집계.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict


# VitalDB lab name 후보 (소문자 비교). 실제 표기는 --dump-names 로 먼저 확인 권장.
HB_NAMES = {"hb", "hgb", "hemoglobin"}
HCT_NAMES = {"hct", "hematocrit"}


def load_case_opspan(clinical_csv: str) -> dict[int, tuple[float, float]]:
    """clinical_data.csv → {caseid: (opstart_sec, opend_sec)} (가능한 컬럼만)."""
    span: dict[int, tuple[float, float]] = {}
    with open(clinical_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        start_col = "opstart" if "opstart" in cols else None
        end_col = "opend" if "opend" in cols else None
        for row in reader:
            try:
                cid = int(row["caseid"])
            except (ValueError, TypeError, KeyError):
                continue
            try:
                s = float(row[start_col]) if start_col else 0.0
                e = float(row[end_col]) if end_col else float("inf")
            except (ValueError, TypeError):
                continue
            span[cid] = (s, e)
    return span


def percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    idx = min(len(sorted_vals) - 1, int(q * (len(sorted_vals) - 1)))
    return sorted_vals[idx]


def summarize(name: str, counts: list[int], gaps_min: list[float]) -> None:
    n_cases = len(counts)
    if n_cases == 0:
        print(f"\n[{name}] 측정 케이스 0건 — lab name 표기 확인 필요 (--dump-names)")
        return
    counts_s = sorted(counts)
    ge2 = sum(1 for c in counts if c >= 2)
    ge3 = sum(1 for c in counts if c >= 3)
    print(f"\n[{name}]")
    print(f"  측정된 케이스 수      : {n_cases}")
    print(f"  케이스당 측정 횟수    : median={percentile(counts_s,0.5):.0f} "
          f"p25={percentile(counts_s,0.25):.0f} p75={percentile(counts_s,0.75):.0f} "
          f"max={counts_s[-1]}")
    print(f"  ≥2회 측정 케이스      : {ge2} ({ge2/n_cases*100:.1f}%)  "
          f"≥3회: {ge3} ({ge3/n_cases*100:.1f}%)")
    if gaps_min:
        gaps_s = sorted(gaps_min)
        print(f"  연속 측정 간격(분)    : median={percentile(gaps_s,0.5):.1f} "
              f"p10={percentile(gaps_s,0.1):.1f} p25={percentile(gaps_s,0.25):.1f} "
              f"p75={percentile(gaps_s,0.75):.1f}")
        le15 = sum(1 for g in gaps_min if g <= 15)
        le30 = sum(1 for g in gaps_min if g <= 30)
        print(f"  ≤15분 간격 비율       : {le15/len(gaps_min)*100:.1f}%   "
              f"≤30분: {le30/len(gaps_min)*100:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab-csv", required=True)
    ap.add_argument("--clinical-csv", default=None,
                    help="주면 intraop 구간(opstart~opend)만 따로 집계")
    ap.add_argument("--dump-names", action="store_true",
                    help="lab name 별 row 수 상위 50개만 출력하고 종료")
    args = ap.parse_args()

    if args.dump_names:
        name_count: dict[str, int] = defaultdict(int)
        with open(args.lab_csv, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                name_count[str(row.get("name", "")).strip().lower()] += 1
        print("lab name (row 수 상위 50):")
        for nm, c in sorted(name_count.items(), key=lambda x: -x[1])[:50]:
            print(f"  {nm:20s} {c}")
        return

    span = load_case_opspan(args.clinical_csv) if args.clinical_csv else {}

    # name → {caseid: [dt_sec, ...]}
    times: dict[str, dict[int, list[float]]] = {"Hb": defaultdict(list),
                                                "Hct": defaultdict(list)}
    with open(args.lab_csv, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            nm = str(row.get("name", "")).strip().lower()
            key = "Hb" if nm in HB_NAMES else "Hct" if nm in HCT_NAMES else None
            if key is None:
                continue
            try:
                cid = int(row["caseid"])
                dt = float(row["dt"])
            except (ValueError, TypeError, KeyError):
                continue
            times[key][cid].append(dt)

    for key in ("Hb", "Hct"):
        # 전체 + intraop-only 두 가지로 집계
        for scope in (("all", None), ("intraop", span)):
            label, sp = scope
            if label == "intraop" and not sp:
                continue
            counts: list[int] = []
            gaps: list[float] = []
            for cid, ts in times[key].items():
                tt = sorted(ts)
                if sp:
                    s, e = sp.get(cid, (0.0, float("inf")))
                    tt = [t for t in tt if s <= t <= e]
                if not tt:
                    continue
                counts.append(len(tt))
                gaps.extend((tt[i + 1] - tt[i]) / 60.0 for i in range(len(tt) - 1))
            summarize(f"{key} ({label})", counts, gaps)


if __name__ == "__main__":
    main()
