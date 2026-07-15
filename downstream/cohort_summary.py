# -*- coding:utf-8 -*-
"""Downstream task cohort 요약 (Patients | Events | Prevalence).

두 가지 소스를 합쳐 13개 task 표를 채운다.

A) PREPARED (analysis cohort) — `outputs/downstream/**/*.pt` 를 자동 발견해 집계.
   .pt 스키마: {train,test}[ {signals, labels, label_values, case_ids, subject_ids} ]
              + metadata{task, source, ...}.  ← 논문에 바로 쓰는 최종 cohort.
   window-level(윈도우 단위) + patient-level(subject_ids 단위) 둘 다 보고.

B) SOURCE cohort (BigQuery CSV) — MIMIC 임상 task 의 원본 cohort (파형 매칭 전).
   prepared 가 아직 없을 때의 상한선 참고치.

REGISTRY 로 13개 task 전체(데이터 소스/label 규칙/상태)를 한눈에 나열한다.
VitalDB task(hypotension/hypoxemia/etco2/aki)는 prepared 가 생기면 A 에서 자동 집계됨.
"""
from __future__ import annotations

import csv
import glob
import sys
from collections import defaultdict

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ----------------------------------------------------------------------------
# REGISTRY: 13 task (project_paper_task_modality)
#   db: 데이터 소스 | kind: binary/multiclass/generation | rule: label 규칙
# ----------------------------------------------------------------------------
REGISTRY = {
    "cardiac_arrest":          dict(db="MIMIC", kind="binary",     rule="cardiac arrest within horizon (5/15/30min)"),
    "intracranial_hypertension": dict(db="MIMIC", kind="binary",   rule="ICP>20mmHg sustained, future horizon"),
    "arrhythmia":              dict(db="MIMIC-ExtPPG", kind="multiclass", rule="rhythm class (SR/AF/AFLT/STACH/SBRAD)"),
    "sepsis":                  dict(db="MIMIC", kind="binary",     rule="Sepsis-3"),
    "mortality":               dict(db="MIMIC", kind="binary",     rule="in-hospital death (hospital_expire_flag)"),
    "vent_need":               dict(db="MIMIC", kind="binary",     rule="ventilation within 24h"),
    "extubation":              dict(db="MIMIC", kind="binary",     rule="extubation failure / reintub within 48h"),
    "hypotension":             dict(db="VitalDB", kind="binary",   rule="MAP<threshold sustained (intraop)"),
    "hypoxemia":               dict(db="VitalDB", kind="binary",   rule="SpO2<threshold sustained (intraop)"),
    "etco2_abnormal":          dict(db="VitalDB", kind="binary",   rule="etCO2 abnormal sustained (intraop)"),
    "aki":                     dict(db="VitalDB", kind="binary",   rule="postop AKI (clinical/lab)"),
    "cross_modal":             dict(db="VitalDB", kind="generation", rule="cross-modal reconstruction (no prevalence)"),
    "forecasting":             dict(db="MIMIC", kind="generation", rule="waveform forecasting (no prevalence)"),
}

# B) source cohort CSV (task -> (path, label col))
CSV_TASKS = {
    "cardiac_arrest": ("downstream/acute_event/cardiac_arrest/bquxjob_3b8c6a77_19e2034b913.csv", "cardiac_arrest"),
    "sepsis":         ("downstream/outcome/sepsis/bquxjob_93e3c7c_19d8f609070.csv", "sepsis3"),
    "mortality":      ("downstream/outcome/mortality/bquxjob_6a8255f2_19d9042b214.csv", "hospital_expire_flag"),
    "vent_need":      ("downstream/outcome/vent_need/bquxjob_6b9a6adb_19e203a5e97.csv", "vent_within_24h"),
    "extubation":     ("downstream/outcome/extubation/bquxjob_4f1dd355_19e203e8e82.csv", "extub_fail_48h"),
}

PREPARED_GLOB = "outputs/downstream/**/*.pt"


def _to01(v) -> int:
    try:
        return 1 if int(float(v)) != 0 else 0
    except (ValueError, TypeError):
        return 0


# ---- A) prepared .pt 집계 --------------------------------------------------
def summarize_prepared(pt_path: str):
    """반환: dict 또는 None(라벨 없음/생성형)."""
    obj = torch.load(pt_path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return None
    meta = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}
    # split 합치기 (train/val/test 등 라벨 가진 split 전부)
    labels, subjects = [], []
    for split, payload in obj.items():
        if not isinstance(payload, dict) or "labels" not in payload:
            continue
        lab = payload["labels"]
        if hasattr(lab, "tolist"):
            lab = lab.tolist()
        sids = payload.get("subject_ids") or payload.get("case_ids") or []
        if hasattr(sids, "tolist"):
            sids = sids.tolist()
        labels.extend(int(x) for x in lab)
        subjects.extend(str(s) for s in sids)
    if not labels:
        return None
    n_win = len(labels)
    evt_win = sum(1 for x in labels if x != 0)
    # patient-level (subject_ids 길이가 맞을 때만)
    pat = None
    if len(subjects) == n_win:
        by_subj = defaultdict(int)
        for s, y in zip(subjects, labels):
            by_subj[s] = max(by_subj[s], 1 if y != 0 else 0)
        pat = (len(by_subj), sum(by_subj.values()))
    return dict(source=meta.get("source", "?"), task=meta.get("task", "?"),
                n_win=n_win, evt_win=evt_win,
                n_pat=(pat[0] if pat else None), evt_pat=(pat[1] if pat else None))


def task_of_path(p: str) -> str:
    p = p.replace("\\", "/")
    return p.split("/")[2]  # outputs/downstream/<task>/file.pt


# ---- B) cohort CSV 집계 ----------------------------------------------------
def summarize_csv(path: str, label: str, id_col: str = "subject_id"):
    by_subj = defaultdict(int)
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            by_subj[row[id_col]] = max(by_subj[row[id_col]], _to01(row.get(label, "0")))
    n = len(by_subj)
    e = sum(by_subj.values())
    return n, e, (e / n if n else 0.0)


def main() -> None:
    # ---- A) PREPARED (analysis cohort) ----
    prepared = sorted(glob.glob(PREPARED_GLOB, recursive=True))
    prepared_tasks = set()
    print("## A) Prepared analysis cohort (outputs/downstream/*.pt)\n")
    if prepared:
        print("| Task | config | Source | Win patients | Win events | Win prev | Pat-level prev |")
        print("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for p in prepared:
            task = task_of_path(p)
            cfg = p.replace("\\", "/").split("/")[-1].replace(".pt", "")
            r = summarize_prepared(p)
            if r is None:
                print(f"| {task} | {cfg} | — | — | — | (generation/무라벨) | — |")
                continue
            prepared_tasks.add(task)
            patprev = (f"{r['evt_pat']}/{r['n_pat']} = {r['evt_pat']/r['n_pat']:.1%}"
                       if r["n_pat"] else "—")
            print(f"| {task} | {cfg} | {r['source']} | {r['n_win']:,} | {r['evt_win']:,} "
                  f"| {r['evt_win']/r['n_win']:.1%} | {patprev} |")
    else:
        print("(prepared .pt 없음)")

    # ---- B) SOURCE cohort (CSV) ----
    print("\n## B) Source cohort (BigQuery CSV, 파형매칭 전 상한선)\n")
    print("| Task | Label | Patients | Events | Prevalence |")
    print("| --- | --- | ---: | ---: | ---: |")
    for task, (path, label) in CSV_TASKS.items():
        n, e, prev = summarize_csv(path, label)
        print(f"| {task} | `{label}` | {n:,} | {e:,} | {prev:.1%} |")

    # ---- C) REGISTRY (전체 13 task 상태) ----
    print("\n## C) 전체 task registry / 상태\n")
    print("| Task | DB | Kind | Label rule | 상태 |")
    print("| --- | --- | --- | --- | --- |")
    for task, info in REGISTRY.items():
        if task in prepared_tasks:
            status = "✅ prepared 집계됨(A)"
        elif task in CSV_TASKS:
            status = "🟡 source cohort만(B), prepared 대기"
        elif info["kind"] == "generation":
            status = "— prevalence 해당없음"
        else:
            status = "⬜ prepared 대기 (prepare_data 실행 필요)"
        print(f"| {task} | {info['db']} | {info['kind']} | {info['rule']} | {status} |")


if __name__ == "__main__":
    main()
