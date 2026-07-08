# -*- coding:utf-8 -*-
"""False Ventricular-Tachycardia Alarm Reduction (VTaC) — 데이터 준비.

PhysioNet ``vtac`` (1.0) 벤치마크: ICU 침상 모니터가 울린 심실빈맥(VT) 알람이
**진짜 VT(True)** 인지 **오알람(False)** 인지 이진 분류한다 (false alarm reduction).
VT 는 acute arrhythmic event 이므로, 세그먼트 내 실제 VT event 존재 여부를 판정하는
Acute Event Detection 과제로 다룬다.

- 파형 소스: ``waveforms/<record>/<event>.hea + .dat`` (WFDB, 250Hz, 90000 sample =
    360s). 각 세그먼트는 **알람 전 5분 + 알람 후 1분** → 알람 시각 = 300s 고정.
    채널: ECG lead ≥2 (II·V·I·III·aVR…) + PLETH(PPG) + (일부) ABP.

입력 window 는 VTaC 원논문(Lehman et al., NeurIPS 2023) 설정을 따른다:
    real-time     : 알람 직전 10초 (pre-alarm only)      → win=10, post=0
    retrospective : 알람 전 10초 + 후 5초 = 15초 (기본)  → win=15, post=5
retrospective(15s, post-alarm 5초 포함)가 논문 대표 설정이라 기본값으로 둔다.
- 라벨 소스: ``event_labels.csv`` (record, event, decision ∈ {True, False}).
    품질 필터링 후 최종 5,037 event (True 1,441 = 28.6%). Uncertain/Rejected 제외.
- 공식 분할: ``benchmark_data_split.csv`` (split ∈ {train, val, test}, record 단위).

⚠ 알람 시각 정렬 보존이 핵심 — MIMIC 파서의 ``_apply_pipeline`` 은 "가장 긴 NaN-free
구간"을 뽑아 시각 정렬을 깨므로 사용하지 않는다. arrhythmia 파서와 동일하게 **알람
주변 윈도우를 먼저 잘라(native 250Hz) per-window cleaning** (range → spike → step/
motion → median → notch(60Hz) → bandpass/lowpass → 100Hz resample → lenient quality)
후 저장한다. cleaning primitive·SIGNAL_CONFIGS 는 pretrain 파서(``data/parser/
vitaldb.py``)를 그대로 재사용해 분포 정합을 맞춘다 (US 60Hz 전원 → notch 60Hz 일치).
부정맥 보호를 위해 ECG ``domain_quality_check`` (HR/규칙성)는 적용하지 않는다.

CARMEN 은 modality 당 단일 채널(channel-independent)이므로 ECG 는 우선순위(II→V→I…)로
lead 1개만 취한다.

분할 (``--split-mode``):
    cv        : Stratified record-level 5-fold CV (기본, 타 classification task 정합,
                run.py OOF pooled bootstrap 용) → ``vtac_<mode>_fold{0..4}.pt``
    benchmark : VTaC 공식 train/val/test split (published baseline 비교용) →
                ``vtac_<mode>_fold0.pt`` (n_folds=1)

사용법:
    python -m downstream.acute_event.vtac.prepare_data \
        --data-dir "C:/workspace/New folder/physionet.org/files/vtac/1.0" \
        --input-signals ecg ppg \
        --out-dir outputs/downstream/vtac --workers 16
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# pretrain 파서 cleaning 헬퍼 재사용 (수정 금지 — read-only import) ──────────────
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    _apply_filter,
    _apply_median_filter,
    _apply_notch_filter,
    _apply_range_check,
    _fill_short_nan_gaps,
)
from data.parser._common import resample_to_target, segment_quality_score
from downstream._kfold_utils import (
    stratified_kfold_patient_splits,
    summarize_splits,
)

# ── 설정 ──────────────────────────────────────────────────────

TARGET_SR: float = 100.0
SEGMENT_SEC: float = 360.0     # 세그먼트 전체 길이 (5min pre + 1min post)
ALARM_SEC: float = 300.0       # 세그먼트 내 알람 시각 (고정)

# 이진 라벨: 0 = False alarm(오알람), 1 = True VT(진짜 심실빈맥)
CLASS_NAMES = ["False", "True"]
N_CLASSES = len(CLASS_NAMES)
DECISION_TO_LABEL: dict[str, int] = {"false": 0, "true": 1}

# WFDB sig_name → 우리 signal_type. 12-lead ECG 는 모두 ecg 로 통합(단일 modality).
_ECG_LEADS = {
    "I", "II", "III", "AVR", "AVL", "AVF",
    "V", "V1", "V2", "V3", "V4", "V5", "V6",
    "MCL", "MCL1",
}
# ECG lead 선택 우선순위 (limb lead II 우선 → precordial → 나머지)
_ECG_PRIORITY = [
    "II", "V", "V5", "V1", "V2", "V3", "V4", "V6",
    "I", "III", "AVR", "AVL", "AVF", "MCL", "MCL1",
]
_PPG_NAMES = {"PLETH"}
_ABP_NAMES = ["ABP", "ART"]   # 우선순위 순


def _sig_name_to_type(name: str) -> str | None:
    n = name.strip().upper()
    if n in _ECG_LEADS:
        return "ecg"
    if n in _PPG_NAMES:
        return "ppg"
    if n in _ABP_NAMES:
        return "abp"
    return None


# ── CSV 파싱 ───────────────────────────────────────────────────


def load_event_labels(csv_path: str) -> dict[str, tuple[str, int]]:
    """event_labels.csv → {event_id: (record, label01)}. True/False 만 채택."""
    out: dict[str, tuple[str, int]] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ev = (row.get("event") or "").strip()
            rec = (row.get("record") or "").strip()
            dec = (row.get("decision") or "").strip().lower()
            if not ev or dec not in DECISION_TO_LABEL:
                continue
            out[ev] = (rec, DECISION_TO_LABEL[dec])
    return out


def load_benchmark_split(csv_path: str) -> dict[str, str]:
    """benchmark_data_split.csv → {event_id: split ∈ {train,val,test}}."""
    out: dict[str, str] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            ev = (row.get("event") or "").strip()
            sp = (row.get("split") or "").strip().lower()
            if ev and sp in ("train", "val", "test"):
                out[ev] = sp
    return out


# ── 파형 로딩 + 알람정렬 per-window cleaning ───────────────────


def _pick_channels(
    sig_name: list[str],
    input_signals: tuple[str, ...],
) -> dict[str, int] | None:
    """rec.sig_name → {signal_type: channel_idx}. 필수 신호 하나라도 없으면 None.

    ECG 는 우선순위에 따라 lead 1개만 선택(단일 modality). 나머지는 첫 매칭 채널.
    """
    upper = [s.strip().upper() for s in sig_name]
    name_to_idx: dict[str, int] = {}
    for i, n in enumerate(upper):
        name_to_idx.setdefault(n, i)   # 동일 이름 중복 시 첫 채널

    picked: dict[str, int] = {}
    for stype in input_signals:
        if stype == "ecg":
            idx = next((name_to_idx[n] for n in _ECG_PRIORITY if n in name_to_idx), None)
        elif stype == "ppg":
            idx = next((name_to_idx[n] for n in _PPG_NAMES if n in name_to_idx), None)
        elif stype == "abp":
            idx = next((name_to_idx[n] for n in _ABP_NAMES if n in name_to_idx), None)
        else:
            idx = None
        if idx is None:
            return None
        picked[stype] = idx
    return picked


def _interp_nan(seg: np.ndarray) -> np.ndarray | None:
    """잔여 NaN 선형 보간 (양끝 NaN=nearest). 전부 NaN 이면 None."""
    mask = np.isnan(seg)
    if not mask.any():
        return seg
    if mask.all():
        return None
    idx = np.arange(len(seg))
    seg = seg.copy()
    seg[mask] = np.interp(idx[mask], idx[~mask], seg[~mask])
    return seg


def _clean_window(
    seg: np.ndarray,
    stype: str,
    native_sr: float,
    win_sec: float,
    max_nan_frac: float,
) -> np.ndarray | None:
    """알람 주변 native 윈도우 → cleaning → 100Hz → lenient quality.

    ⚠ VT alarm 판별 과제이므로 **aggressive artifact 검출기(electrocautery spike /
    step / motion)는 적용하지 않는다** — VT 의 고진폭·빠른 QRS·저관류 PPG 를 아티팩트로
    오인해 NaN 마스킹하면 정작 판별 대상 morphology 를 파괴하기 때문(arrhythmia 파서가
    ``domain_quality_check`` 를 끈 것과 동일 논리). 분포 정합에 필요한 range check ·
    median · notch(60Hz) · bandpass/lowpass · 100Hz resample 만 유지한다.

    range check 로 생긴 짧은 out-of-range 결측은 선형 보간하되, 결측 비율이
    ``max_nan_frac`` 를 넘으면(실제 disconnect/포화) window 전체를 drop.
    """
    cfg = SIGNAL_CONFIGS[stype]
    seg = np.asarray(seg, dtype=np.float64)

    seg = _fill_short_nan_gaps(seg, max(1, int(0.1 * native_sr)))
    if cfg.valid_range is not None:
        seg, _ = _apply_range_check(seg, cfg.valid_range)
    seg = _fill_short_nan_gaps(seg, max(1, int(0.2 * native_sr)))

    nan_frac = float(np.isnan(seg).mean())
    if nan_frac > max_nan_frac:      # 실제 disconnect/포화 → drop
        return None
    seg = _interp_nan(seg)           # 잔여 결측 선형 보간 (필터 가능하게)
    if seg is None:
        return None

    if cfg.median_kernel > 0:
        seg = _apply_median_filter(seg, kernel_size=cfg.median_kernel)
    if cfg.notch_freq is not None:
        seg = _apply_notch_filter(seg, freq=cfg.notch_freq, sr=native_sr)
    seg = _apply_filter(seg, cfg, native_sr)

    if native_sr != TARGET_SR:
        seg = resample_to_target(seg, orig_sr=native_sr, target_sr=TARGET_SR)
    target_len = int(win_sec * TARGET_SR)
    if len(seg) < target_len:
        return None
    seg = seg[:target_len]

    q = segment_quality_score(
        seg,
        max_flatline_ratio=0.99,
        max_clip_ratio=0.8,
        max_high_freq_ratio=100.0,   # bypass — VT hf 변동 허용
        min_amplitude=0.0,
        max_amplitude=0.0,
        min_high_freq_ratio=0.0,
    )
    if not q["pass"]:
        return None
    return seg.astype(np.float32)


# ── per-event worker ───────────────────────────────────────────


@dataclass
class _EventJob:
    event_id: str
    record: str
    label: int
    rec_path: str          # .hea/.dat 확장자 없는 경로
    input_signals: tuple[str, ...]
    win_sec: float
    post_alarm_sec: float
    max_nan_frac: float


def _process_event(job: _EventJob) -> dict | None:
    """한 event → labeled 윈도우 신호 샘플 (실패 시 None).

    반환: {"record", "event", "label", "signals":{st:(T,)f32}}
    """
    import wfdb   # worker 프로세스에서 lazy import

    try:
        rec = wfdb.rdrecord(job.rec_path)
    except Exception:
        return None
    if rec.p_signal is None or rec.sig_len == 0:
        return None

    fs = float(rec.fs)
    picked = _pick_channels(list(rec.sig_name), job.input_signals)
    if picked is None:
        return None

    # 알람 주변 윈도우 (native domain): [alarm - pre, alarm + post]
    pre_sec = job.win_sec - job.post_alarm_sec
    alarm_i = int(round(ALARM_SEC * fs))
    s0 = alarm_i - int(round(pre_sec * fs))
    s1 = alarm_i + int(round(job.post_alarm_sec * fs))
    if s0 < 0 or s1 > rec.sig_len:
        return None

    sigs: dict[str, np.ndarray] = {}
    for stype, ch in picked.items():
        raw = rec.p_signal[s0:s1, ch]
        cleaned = _clean_window(raw, stype, fs, job.win_sec, job.max_nan_frac)
        if cleaned is None:
            return None
        sigs[stype] = cleaned

    return {
        "record": job.record,
        "event": job.event_id,
        "label": job.label,
        "signals": sigs,
    }


# ── 저장 (arrhythmia run.py 스키마와 정합) ─────────────────────


def _split_dict(
    samples: list[dict],
    input_signals: tuple[str, ...],
    win_sec: float,
) -> dict:
    target_len = int(win_sec * TARGET_SR)
    signals: dict[str, torch.Tensor] = {}
    for st in input_signals:
        if samples:
            arr = np.stack([s["signals"][st].astype(np.float16) for s in samples])
            signals[st] = torch.from_numpy(arr)          # (N, T) fp16
        else:
            signals[st] = torch.empty((0, target_len), dtype=torch.float16)
    return {
        "signals": signals,
        "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
        "patients": [s["record"] for s in samples],       # record = 분할 키
        "rhythms": [CLASS_NAMES[s["label"]] for s in samples],
        "segment_ids": [s["event"] for s in samples],
    }


def save_fold(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    input_signals: tuple[str, ...],
    out_dir: str,
    fold_idx: int,
    n_folds: int,
    win_sec: float,
    split_mode: str,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "train": _split_dict(train, input_signals, win_sec),
        "val": _split_dict(val, input_signals, win_sec),
        "test": _split_dict(test, input_signals, win_sec),
        "metadata": {
            "task": "vtac_false_vt_alarm",
            "source": "PhysioNet VTaC 1.0",
            "classes": CLASS_NAMES,
            "class_names": CLASS_NAMES,
            "n_classes": N_CLASSES,
            "input_signals": list(input_signals),
            "signal_type_map": {i: s for i, s in enumerate(input_signals)},
            "sampling_rate": TARGET_SR,
            "segment_sec": SEGMENT_SEC,
            "alarm_sec": ALARM_SEC,
            "win_sec": win_sec,
            "signal_dtype": "float16",
            "split_mode": split_mode,
            "split": (
                "VTaC official train/val/test (record-level)"
                if split_mode == "benchmark"
                else (
                    f"stratified record-level {n_folds}-fold CV: "
                    f"fold {fold_idx} test, {(fold_idx + 1) % n_folds} val, rest train"
                )
            ),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "fold_idx": fold_idx,
            "n_folds": n_folds,
        },
    }
    mode_str = "_".join(input_signals)
    save_path = out_path / f"vtac_{mode_str}_fold{fold_idx}.pt"
    torch.save(save_dict, save_path)
    print(f"  Saved: {save_path} ({save_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return save_path


def _print_split_stats(name: str, samples: list[dict]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return
    dist = Counter(s["label"] for s in samples)
    n_rec = len({s["record"] for s in samples})
    print(f"  {name}: {len(samples)} events ({n_rec} records)")
    for cid, cname in enumerate(CLASS_NAMES):
        cnt = dist.get(cid, 0)
        print(f"    {cname}({cid}): {cnt} ({100 * cnt / len(samples):.1f}%)")


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="VTaC False VT Alarm (binary) — Data Preparation"
    )
    p.add_argument("--data-dir", type=str, required=True,
                   help="physionet.org/files/vtac/1.0 (waveforms/ + event_labels.csv "
                        "+ benchmark_data_split.csv)")
    p.add_argument("--out-dir", type=str, default="outputs/downstream/vtac")
    p.add_argument("--input-signals", nargs="+", default=["ecg", "ppg"],
                   choices=["ecg", "ppg", "abp"],
                   help="ecg·ppg 는 거의 전 event 존재, abp 요구 시 소수 event 로 축소")
    p.add_argument("--win-sec", type=float, default=15.0,
                   help="윈도우 길이(초). VTaC 원논문: retrospective 15 / real-time 10")
    p.add_argument("--post-alarm-sec", type=float, default=5.0,
                   help="알람 이후 포함 길이(초). 윈도우=[alarm-(win-post), alarm+post]. "
                        "retrospective=5, real-time=0")
    p.add_argument("--max-nan-frac", type=float, default=0.2,
                   help="윈도우 결측 허용 상한(초과 시 drop). 이하는 선형 보간")
    p.add_argument("--split-mode", choices=["cv", "benchmark"], default="cv",
                   help="cv=stratified record-level 5-fold / benchmark=VTaC 공식 split")
    p.add_argument("--n-folds", type=int, default=5, help="cv 모드 fold 수")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--limit", type=int, default=None, help="앞 N event 만 (smoke test)")
    args = p.parse_args()

    input_signals = tuple(args.input_signals)
    data_dir = Path(args.data_dir)
    wf_dir = data_dir / "waveforms"
    labels_csv = str(data_dir / "event_labels.csv")
    split_csv = str(data_dir / "benchmark_data_split.csv")

    print("=" * 60)
    print(f"  VTaC False VT Alarm: {' + '.join(s.upper() for s in input_signals)}")
    print(f"  Data:       {data_dir}")
    print(f"  Window:     {args.win_sec}s  (alarm@{ALARM_SEC:.0f}s, "
          f"[-{args.win_sec - args.post_alarm_sec:.0f}s, +{args.post_alarm_sec:.0f}s])")
    print(f"  Split mode: {args.split_mode}")
    print("=" * 60)

    # 1. 라벨 + (필요 시) 공식 split 로드, .hea 존재하는 event 만 job 구성
    print("\n[1/4] Building event jobs...")
    ev_labels = load_event_labels(labels_csv)
    bench_split = load_benchmark_split(split_csv) if args.split_mode == "benchmark" else {}

    jobs: list[_EventJob] = []
    n_missing_file = 0
    n_missing_split = 0
    for ev, (rec, label) in sorted(ev_labels.items()):
        if args.split_mode == "benchmark" and ev not in bench_split:
            n_missing_split += 1
            continue
        rec_path = wf_dir / rec / ev
        if not (wf_dir / rec / f"{ev}.hea").exists():
            n_missing_file += 1          # 아직 다운로드 안 된 event → skip
            continue
        jobs.append(_EventJob(
            event_id=ev, record=rec, label=label, rec_path=str(rec_path),
            input_signals=input_signals, win_sec=args.win_sec,
            post_alarm_sec=args.post_alarm_sec, max_nan_frac=args.max_nan_frac,
        ))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"  labeled events: {len(ev_labels)}, jobs: {len(jobs)} "
          f"(missing .hea: {n_missing_file}"
          + (f", not in official split: {n_missing_split}" if bench_split else "")
          + ")")
    if not jobs:
        print("ERROR: no downloaded events found. 다운로드 진행 중이면 완료 후 재실행.",
              file=sys.stderr)
        sys.exit(1)

    # 2. 병렬 파형 로딩 + cleaning
    print(f"\n[2/4] Loading waveforms ({args.workers} workers)...")
    samples: list[dict] = []
    done = 0
    n_jobs = len(jobs)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_event, j): j for j in jobs}
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception as exc:
                print(f"  [WARN] event {futs[fut].event_id} 실패: {exc}", file=sys.stderr)
                res = None
            if res is not None:
                samples.append(res)
            if done % 200 == 0 or done == n_jobs:
                dist = Counter(s["label"] for s in samples)
                print(f"  [{done}/{n_jobs}] kept={len(samples)} "
                      f"(False:{dist.get(0, 0)} True:{dist.get(1, 0)})")

    if not samples:
        print("ERROR: no samples passed cleaning/quality.", file=sys.stderr)
        sys.exit(1)

    dist = Counter(s["label"] for s in samples)
    n_rec = len({s["record"] for s in samples})
    print(f"\n  Final: {len(samples)} events, {n_rec} records")
    for cid, cname in enumerate(CLASS_NAMES):
        c = dist.get(cid, 0)
        print(f"    {cname}({cid}): {c} ({100 * c / len(samples):.1f}%)")

    # 3-4. 분할 + 저장
    if args.split_mode == "benchmark":
        print("\n[4/4] VTaC official split + saving...")
        by_split: dict[str, list[dict]] = defaultdict(list)
        for s in samples:
            by_split[bench_split[s["event"]]].append(s)
        train, val, test = by_split["train"], by_split["val"], by_split["test"]
        _print_split_stats("    Train", train)
        _print_split_stats("    Val", val)
        _print_split_stats("    Test", test)
        save_fold(train, val, test, input_signals, args.out_dir,
                  fold_idx=0, n_folds=1, win_sec=args.win_sec, split_mode="benchmark")
    else:
        print(f"\n[4/4] Stratified {args.n_folds}-fold record-level CV + saving...")
        rec_labels: dict[str, list[int]] = defaultdict(list)
        for s in samples:
            rec_labels[s["record"]].append(s["label"])
        record_ids = sorted(rec_labels.keys())
        print(f"  Total records: {len(record_ids)}")

        splits = stratified_kfold_patient_splits(
            record_ids, dict(rec_labels), n_folds=args.n_folds, seed=args.seed,
        )
        print(summarize_splits(splits, dict(rec_labels)))

        for fold_idx, (tr, va, te) in enumerate(splits):
            print(f"\n  [Fold {fold_idx}]")
            train = [s for s in samples if s["record"] in tr]
            val = [s for s in samples if s["record"] in va]
            test = [s for s in samples if s["record"] in te]
            _print_split_stats("    Train", train)
            _print_split_stats("    Val", val)
            _print_split_stats("    Test", test)
            save_fold(train, val, test, input_signals, args.out_dir,
                      fold_idx, args.n_folds, win_sec=args.win_sec, split_mode="cv")

    print("\n" + "=" * 60)
    print(f"  Done → {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
