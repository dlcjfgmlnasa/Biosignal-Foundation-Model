# -*- coding:utf-8 -*-
"""Arrhythmia Classification - 데이터 준비 (VitalDB-Arrhythmia + VitalDB Open).

PhysioNet ``vitaldb-arrhythmia`` (annotation only) + VitalDB Open ``.vital``
(waveform) 을 엮어 30초 segment 의 현재 rhythm class 를 분류한다 (5-class).

- 라벨 소스: ``Annotation_Files/Annotation_file_<case_id>.csv``
    R-peak ``time_second`` 별 ``rhythm_label`` (10종) + ``bad_signal_quality``.
    마취과 전문의 5명 검증 (Cohen's κ 0.930).
- 파형 소스: ``vitaldb_open/.../vital_files/<caseid:04d>.vital``
    ``SNUADC/ECG_II`` (ECG) + ``SNUADC/PLETH`` (PPG), 500Hz → 100Hz.
- 환자 단위 분할 키: metadata 의 ``subjectid`` (case_id 아님 — 같은 환자의 다른
  수술이 train/test 로 새는 것 방지).

10종 rhythm_label → 5-class collapse (임상 다중분류):
    0 NSR  : N
    1 AFIB : AFIB/AFL
    2 VENT : VT, Patterned Ventricular Ectopy
    3 SVT  : SVTA, Patterned Atrial Ectopy, WAP/MAT
    4 COND : SND, AVB
    (drop): Noise, '', Unclassifiable + bad_signal_quality 윈도우

신호 전처리는 pretrain 파서(``data/parser/vitaldb.py``)의 cleaning 헬퍼를 그대로
재사용해 분포 정합을 맞춘다 (range → spike → step/motion → notch → bandpass/lowpass
→ 100Hz resample). 단, ECG ``domain_quality_check`` (HR·규칙성)는 **적용하지 않는다**
— 부정맥 윈도우를 "비정상"으로 떨궈 정작 분류 대상을 버리기 때문. basic quality
(flatline/saturation) 만 lenient threshold 로 적용한다.

분할: Stratified patient-level 5-fold CV (downstream/_kfold_utils).

사용법:
    python -m downstream.acute_event.arrhythmia.prepare_data_vitaldb \
        --arrhythmia-dir ~/workspace/datasets/vitaldb-arrhythmia/1.0.0 \
        --vital-dir      ~/workspace/datasets/vitaldb_open/1.0.0/vital_files \
        --input-signals ecg ppg \
        --out-dir <OUT> --workers 16
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
    SIGNAL_TYPES,
    TARGET_SR,
    TRACK_MAP,
    _apply_filter,
    _apply_median_filter,
    _apply_notch_filter,
    _apply_range_check,
    _detect_electrocautery,
    _detect_motion_artifact,
    _detect_step_change,
    _fill_short_nan_gaps,
)
from data.parser._common import resample_to_target, segment_quality_score
from downstream._kfold_utils import (
    stratified_kfold_patient_splits,
    summarize_splits,
)

# ── 설정 ──────────────────────────────────────────────────────

SEGMENT_SEC: float = 30.0

# 10종 rhythm_label (annotation 실제 문자열) → 5-class collapse
RHYTHM_COLLAPSE: dict[str, int] = {
    "N": 0,                                # NSR
    "AFIB/AFL": 1,                         # AFIB (fib + flutter)
    "VT": 2,                               # Ventricular
    "Patterned Ventricular Ectopy": 2,
    "SVTA": 3,                             # Supraventricular
    "Patterned Atrial Ectopy": 3,
    "WAP/MAT": 3,
    "SND": 4,                              # Conduction
    "AVB": 4,
    # drop (매핑 없음): "Noise", "", "Unclassifiable"
}
CLASS_NAMES = ["NSR", "AFIB", "VENT", "SVT", "COND"]
N_CLASSES = len(CLASS_NAMES)

# input-signals → vital track 우선순위 (TRACK_MAP 키)
PREFERRED_TRACKS: dict[str, tuple[str, ...]] = {
    "ecg": ("SNUADC/ECG_II", "SNUADC/ECG_V5", "SNUADC/ECG_I", "Solar8000/ECG_II"),
    "ppg": ("SNUADC/PLETH", "Solar8000/PLETH"),
    "abp": ("SNUADC/ART", "SNUADC/FEM"),
}


@dataclass
class _Beat:
    t: float           # R-peak time_second (recording start 기준)
    cls: int | None    # collapsed class (None = drop label)
    bad: bool          # bad_signal_quality


# ── annotation 파싱 ────────────────────────────────────────────


def load_case_to_subject(metadata_csv: str) -> dict[int, str]:
    """metadata.csv → {case_id: subjectid}. subjectid 가 환자 분할 키."""
    out: dict[int, str] = {}
    with open(metadata_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                cid = int(row["case_id"])
            except (KeyError, ValueError):
                continue
            out[cid] = str(row.get("subjectid", "")).strip() or f"case{cid}"
    return out


def parse_annotation(annot_csv: str) -> list[_Beat]:
    """Annotation_file_<case>.csv → 시간순 beat 리스트."""
    beats: list[_Beat] = []
    with open(annot_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                t = float(row["time_second"])
            except (KeyError, ValueError):
                continue
            rl = (row.get("rhythm_label") or "").strip()
            bad = (row.get("bad_signal_quality") or "").strip().lower() == "true"
            beats.append(_Beat(t=t, cls=RHYTHM_COLLAPSE.get(rl), bad=bad))
    beats.sort(key=lambda b: b.t)
    return beats


def windowize(
    beats: list[_Beat],
    win_sec: float,
    stride_sec: float,
    min_beats: int,
) -> list[tuple[float, int]]:
    """beat 시퀀스 → 고정 윈도우 라벨 목록 [(t_start, class), ...].

    윈도우 채택 조건:
      - 윈도우 내 beat 수 >= min_beats (brady 보호 위해 낮게)
      - 모든 beat 의 collapsed class 가 동일 (purity)
      - drop-label(None) beat 이 하나도 없음 (Noise/Unclassifiable 혼입 차단)
      - bad_signal_quality beat 이 하나도 없음
    """
    if not beats:
        return []
    times = np.array([b.t for b in beats])
    t0, t1 = float(times[0]), float(times[-1])
    out: list[tuple[float, int]] = []
    t = t0
    while t + win_sec <= t1 + 1e-6:
        lo = np.searchsorted(times, t, side="left")
        hi = np.searchsorted(times, t + win_sec, side="left")
        win = beats[lo:hi]
        if len(win) >= min_beats:
            classes = {b.cls for b in win}
            if (
                None not in classes
                and len(classes) == 1
                and not any(b.bad for b in win)
            ):
                out.append((t, next(iter(classes))))
        t += stride_sec
    return out


# ── 파형 로딩 + cleaning (pretrain 파서 재사용) ────────────────


def _pick_track(available: list[str], stype: str) -> str | None:
    """available track 중 해당 signal type 으로 매핑되는 첫 트랙 선택."""
    for cand in PREFERRED_TRACKS.get(stype, ()):
        if cand in available:
            return cand
    # fallback: TRACK_MAP 으로 stype 일치하는 아무 트랙
    for trk in available:
        m = TRACK_MAP.get(trk)
        if m is not None and m[0] == stype:
            return trk
    return None


def _load_clean_track(vf, track_name: str, stype: str) -> tuple[np.ndarray, float] | None:
    """단일 트랙 → full-resolution cleaned 배열 (native_sr 유지, 윈도우 슬라이스 전).

    pretrain process_vital 의 full-track 단계(Step0~2c)를 그대로 적용:
    fill_short_nan_gaps → range_check → spike → step(abp/ppg) → motion(ppg).
    필터/리샘플은 절대 시각 정합을 위해 윈도우 슬라이스 후 수행한다.
    """
    cfg = SIGNAL_CONFIGS[stype]
    try:
        trk = vf.find_track(track_name)
        native_sr = trk.srate if trk is not None and trk.srate > 0 else 500.0
        data = vf.to_numpy(track_name, interval=1.0 / native_sr)
    except Exception:
        return None
    if data is None or len(data) == 0:
        return None
    data = np.asarray(data).flatten().astype(np.float64)

    data = _fill_short_nan_gaps(data, max(1, int(0.1 * native_sr)))
    if cfg.valid_range is not None:
        data, _ = _apply_range_check(data, cfg.valid_range)
    if cfg.spike_detection:
        data, _ = _detect_electrocautery(
            data, native_sr, threshold_std=cfg.spike_threshold_std
        )
    if stype in ("abp", "ppg"):
        data, _ = _detect_step_change(data, native_sr)
    if stype == "ppg":
        data, _ = _detect_motion_artifact(data, native_sr)
    return data, float(native_sr)


def _extract_window(
    full: np.ndarray,
    native_sr: float,
    stype: str,
    t_start: float,
    win_sec: float,
) -> np.ndarray | None:
    """cleaned full-track 에서 [t, t+win] 윈도우 → 필터 → 100Hz → quality.

    부정맥 보호: ECG domain_quality_check (HR/규칙성) 미적용. lenient basic quality
    (flatline/saturation) 만 적용 (pretrain lenient_quality 와 동일 threshold).
    """
    cfg = SIGNAL_CONFIGS[stype]
    s0 = int(round(t_start * native_sr))
    n = int(round(win_sec * native_sr))
    seg = full[s0:s0 + n]
    if len(seg) < n:
        return None
    if np.isnan(seg).any():   # 실제 disconnect/artifact (>100ms) → 필터 불가, drop
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
        max_high_freq_ratio=100.0,   # bypass — 부정맥 hf 변동 허용
        min_amplitude=0.0,
        max_amplitude=0.0,
        min_high_freq_ratio=0.0,
    )
    if not q["pass"]:
        return None
    return seg.astype(np.float32)


# ── per-case worker ───────────────────────────────────────────


@dataclass
class _CaseJob:
    case_id: int
    subject_id: str
    annot_csv: str
    vital_path: str
    input_signals: tuple[str, ...]
    win_sec: float
    stride_sec: float
    min_beats: int
    max_per_case_per_class: int


def _process_case(job: _CaseJob) -> list[dict]:
    """한 case → labeled 윈도우 신호 샘플 리스트.

    반환: [{"subject", "case_id", "label", "segment_id", "signals":{st:(T,)f32}}]
    """
    import vitaldb  # worker 프로세스에서 lazy import

    beats = parse_annotation(job.annot_csv)
    windows = windowize(beats, job.win_sec, job.stride_sec, job.min_beats)
    if not windows:
        return []

    try:
        vf = vitaldb.VitalFile(job.vital_path)
        available = vf.get_track_names()
    except Exception:
        return []

    # signal type 별 트랙 선택 + full-track cleaning (1회)
    cleaned: dict[str, tuple[np.ndarray, float]] = {}
    for stype in job.input_signals:
        trk = _pick_track(available, stype)
        if trk is None:
            return []  # 필수 신호 없으면 이 case 전체 skip
        loaded = _load_clean_track(vf, trk, stype)
        if loaded is None:
            return []
        cleaned[stype] = loaded

    per_class: Counter = Counter()
    out: list[dict] = []
    for t_start, label in windows:
        if per_class[label] >= job.max_per_case_per_class:
            continue
        sigs: dict[str, np.ndarray] = {}
        ok = True
        for stype in job.input_signals:
            full, native_sr = cleaned[stype]
            w = _extract_window(full, native_sr, stype, t_start, job.win_sec)
            if w is None:
                ok = False
                break
            sigs[stype] = w
        if not ok:
            continue
        out.append({
            "subject": job.subject_id,
            "case_id": job.case_id,
            "label": label,
            "segment_id": f"{job.case_id}_{t_start:.1f}",
            "signals": sigs,
        })
        per_class[label] += 1
    return out


# ── 저장 (run.py _load_data 스키마와 정합) ─────────────────────


def _split_dict(samples: list[dict], input_signals: tuple[str, ...]) -> dict:
    target_len = int(SEGMENT_SEC * TARGET_SR)
    signals: dict[str, torch.Tensor] = {}
    for st in input_signals:
        if samples:
            arr = np.stack([s["signals"][st].astype(np.float16) for s in samples])
            signals[st] = torch.from_numpy(arr)         # (N, T) fp16
        else:
            signals[st] = torch.empty((0, target_len), dtype=torch.float16)
    return {
        "signals": signals,
        "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
        "patients": [s["subject"] for s in samples],
        "rhythms": [CLASS_NAMES[s["label"]] for s in samples],
        "segment_ids": [s["segment_id"] for s in samples],
    }


def save_fold(
    train: list[dict],
    val: list[dict],
    test: list[dict],
    input_signals: tuple[str, ...],
    out_dir: str,
    fold_idx: int,
    n_folds: int,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "train": _split_dict(train, input_signals),
        "val": _split_dict(val, input_signals),
        "test": _split_dict(test, input_signals),
        "metadata": {
            "task": "arrhythmia_5class",
            "source": "VitalDB-Arrhythmia + VitalDB Open",
            "classes": CLASS_NAMES,
            "class_names": CLASS_NAMES,
            "n_classes": N_CLASSES,
            "rhythm_collapse": RHYTHM_COLLAPSE,
            "input_signals": list(input_signals),
            "signal_type_map": {SIGNAL_TYPES[s]: s for s in input_signals},
            "sampling_rate": TARGET_SR,
            "segment_sec": SEGMENT_SEC,
            "signal_dtype": "float16",
            "split": (
                f"stratified patient(subjectid)-level {n_folds}-fold CV: "
                f"fold {fold_idx} test, {(fold_idx + 1) % n_folds} val, rest train"
            ),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "fold_idx": fold_idx,
            "n_folds": n_folds,
        },
    }
    mode_str = "_".join(input_signals)
    save_path = out_path / f"arrhythmia_{mode_str}_fold{fold_idx}.pt"
    torch.save(save_dict, save_path)
    print(f"  Saved: {save_path} ({save_path.stat().st_size / 1024 / 1024:.2f} MB)")
    return save_path


def _print_split_stats(name: str, samples: list[dict]) -> None:
    if not samples:
        print(f"  {name}: 0 samples")
        return
    dist = Counter(s["label"] for s in samples)
    n_pt = len({s["subject"] for s in samples})
    print(f"  {name}: {len(samples)} samples ({n_pt} patients)")
    for cid, cname in enumerate(CLASS_NAMES):
        cnt = dist.get(cid, 0)
        print(f"    {cname}({cid}): {cnt} ({100 * cnt / len(samples):.1f}%)")


# ── CLI ───────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Arrhythmia 5-class (VitalDB-Arrhythmia + VitalDB Open)"
    )
    p.add_argument("--arrhythmia-dir", type=str, required=True,
                   help="vitaldb-arrhythmia/1.0.0 (metadata.csv + Annotation_Files/)")
    p.add_argument("--vital-dir", type=str, required=True,
                   help="vitaldb_open/.../vital_files (<caseid:04d>.vital)")
    p.add_argument("--out-dir", type=str, default="outputs/downstream/arrhythmia")
    p.add_argument("--input-signals", nargs="+", default=["ecg", "ppg"],
                   choices=["ecg", "ppg", "abp"])
    p.add_argument("--win-sec", type=float, default=SEGMENT_SEC)
    p.add_argument("--stride-sec", type=float, default=15.0,
                   help="윈도우 stride (초). <win-sec 이면 overlap (같은 환자라 leakage 없음)")
    p.add_argument("--min-beats", type=int, default=10,
                   help="윈도우 최소 beat 수 (brady 보호 위해 낮게)")
    p.add_argument("--max-per-case-per-class", type=int, default=40,
                   help="case·class 당 윈도우 상한 (한 긴 case 의 NSR 독식 방지)")
    p.add_argument("--max-per-class", type=int, default=None,
                   help="전역 class 당 윈도우 상한 (None=무제한, balance 강화용)")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    p.add_argument("--limit", type=int, default=None, help="앞 N case 만 (smoke test)")
    args = p.parse_args()

    input_signals = tuple(args.input_signals)
    arr_dir = Path(args.arrhythmia_dir)
    annot_dir = arr_dir / "Annotation_Files"
    vital_dir = Path(args.vital_dir)
    meta_csv = str(arr_dir / "metadata.csv")

    print("=" * 60)
    print(f"  Arrhythmia 5-class (VitalDB): {' + '.join(s.upper() for s in input_signals)}")
    print(f"  Annotation: {annot_dir}")
    print(f"  Vital:      {vital_dir}")
    print(f"  Classes:    {CLASS_NAMES}")
    print("=" * 60)

    # 1. case → subject 매핑 + job 구성
    print("\n[1/4] Building case jobs...")
    case2subj = load_case_to_subject(meta_csv)
    jobs: list[_CaseJob] = []
    n_missing = 0
    for cid, subj in sorted(case2subj.items()):
        annot = annot_dir / f"Annotation_file_{cid}.csv"
        vital = vital_dir / f"{cid:04d}.vital"
        if not annot.exists() or not vital.exists():
            n_missing += 1
            continue
        jobs.append(_CaseJob(
            case_id=cid, subject_id=subj, annot_csv=str(annot), vital_path=str(vital),
            input_signals=input_signals, win_sec=args.win_sec, stride_sec=args.stride_sec,
            min_beats=args.min_beats, max_per_case_per_class=args.max_per_case_per_class,
        ))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"  cases: {len(jobs)} (missing annot/vital: {n_missing}), "
          f"subjects: {len({j.subject_id for j in jobs})}")

    # 2. 병렬 파형 로딩 (네트워크 마운트 → ProcessPool 필수)
    print(f"\n[2/4] Loading waveforms ({args.workers} workers)...")
    samples: list[dict] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_case, j): j for j in jobs}
        for fut in as_completed(futs):
            done += 1
            try:
                res = fut.result()
            except Exception as exc:
                print(f"  [WARN] case {futs[fut].case_id} 실패: {exc}", file=sys.stderr)
                res = []
            samples.extend(res)
            if done % 25 == 0 or done == len(jobs):
                dist = Counter(s["label"] for s in samples)
                print(f"  [{done}/{len(jobs)}] samples={len(samples)} "
                      f"per-class={{{', '.join(f'{CLASS_NAMES[k]}:{dist.get(k,0)}' for k in range(N_CLASSES))}}}")

    if not samples:
        print("ERROR: no samples loaded.", file=sys.stderr)
        sys.exit(1)

    # 3. 전역 class 상한 (옵션) — seeded shuffle 후 cap
    if args.max_per_class is not None:
        rng = np.random.default_rng(args.seed)
        by_cls: dict[int, list[dict]] = defaultdict(list)
        for s in samples:
            by_cls[s["label"]].append(s)
        capped: list[dict] = []
        for cid, lst in by_cls.items():
            rng.shuffle(lst)
            capped.extend(lst[:args.max_per_class])
        samples = capped
    dist = Counter(s["label"] for s in samples)
    print(f"\n  Final: {len(samples)} samples")
    for cid, cname in enumerate(CLASS_NAMES):
        print(f"    {cname}({cid}): {dist.get(cid, 0)}")

    # 4. Stratified patient(subjectid)-level K-fold + 저장
    print(f"\n[4/4] Stratified {args.n_folds}-fold patient-level CV + saving...")
    pt_labels: dict[str, list[int]] = defaultdict(list)
    for s in samples:
        pt_labels[s["subject"]].append(s["label"])
    patient_ids = sorted(pt_labels.keys())
    print(f"  Total patients (subjectid): {len(patient_ids)}")

    splits = stratified_kfold_patient_splits(
        patient_ids, dict(pt_labels), n_folds=args.n_folds, seed=args.seed,
    )
    print(summarize_splits(splits, dict(pt_labels)))

    for fold_idx, (tr, va, te) in enumerate(splits):
        print(f"\n  [Fold {fold_idx}]")
        train = [s for s in samples if s["subject"] in tr]
        val = [s for s in samples if s["subject"] in va]
        test = [s for s in samples if s["subject"] in te]
        _print_split_stats("    Train", train)
        _print_split_stats("    Val", val)
        _print_split_stats("    Test", test)
        save_fold(train, val, test, input_signals, args.out_dir, fold_idx, args.n_folds)

    print("\n" + "=" * 60)
    print(f"  Done: {args.n_folds} fold(s) → {args.out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
