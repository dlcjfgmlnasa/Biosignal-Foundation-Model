# -*- coding:utf-8 -*-
"""Cuffless BP - 데이터 준비 스크립트.

비침습 신호(PPG/ECG)로부터 ABP waveform을 복원하는 cross-modal generation task.
MIMIC-III Waveform에서 시간 정렬된 다채널 데이터를 추출한다.

3가지 입력 모드:
    PPG -> ABP waveform
    ECG -> ABP waveform
    ECG + PPG -> ABP waveform

평가: 복원된 ABP waveform의 MSE/MAE/Pearson r + peak detection으로 SBP/DBP 오차.

사용법:
    # PPG -> ABP (5 cases)
    python -m downstream.cuffless_bp.prepare_data \
        --input-signals ppg --n-cases 5

    # ECG + PPG -> ABP
    python -m downstream.cuffless_bp.prepare_data \
        --input-signals ecg ppg --n-cases 10

    # ECG -> ABP, 시각화 포함
    python -m downstream.cuffless_bp.prepare_data \
        --input-signals ecg --n-cases 5 --visualize
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

TARGET_SR: float = 100.0


# ---- MIMIC-III 로더 (시간 정렬 다채널) ----


def _load_mimic3_aligned(
    n_cases: int,
    input_signals: list[str],
    min_duration_sec: float = 600.0,
    manifest_path: str | None = None,
    scan_patients: int = 500,
) -> list[dict]:
    """MIMIC-III에서 입력 신호 + ABP가 시간 정렬된 데이터를 로드한다.

    Returns: list of {"case_id", "patient_id", "signals": {"ppg": array, "abp": array, ...}}
    """
    import wfdb
    from data.parser.mimic3_waveform import (
        scan_abp_records, load_manifest, _apply_pipeline,
        MIMIC3_NATIVE_SR,
    )

    required_types = set(input_signals) | {"abp"}

    # manifest 로드 또는 스캔
    if manifest_path and Path(manifest_path).exists():
        records = load_manifest(manifest_path)
    else:
        print(f"  Scanning {scan_patients} patients for ABP records...")
        records = scan_abp_records(max_records=scan_patients, verbose=False)

    # 필요 채널 모두 있는 레코드 필터링
    filtered = []
    for r in records:
        has_all = True
        if "ecg" in required_types and not r.has_ecg:
            has_all = False
        if "ppg" in required_types and not r.has_ppg:
            has_all = False
        if has_all:
            filtered.append(r)

    if not filtered:
        print(f"  WARNING: No records with all required signals {required_types}")
        print(f"  ABP: {len(records)}, +ECG: {sum(1 for r in records if r.has_ecg)}, "
              f"+PPG: {sum(1 for r in records if r.has_ppg)}")
        return []

    filtered = filtered[:n_cases]
    print(f"  Found {len(filtered)} records with {required_types}")

    cases = []
    for i, info in enumerate(filtered):
        print(f"  [{i+1}/{len(filtered)}] {info.record_name}...", end=" ")
        t0 = time.time()

        try:
            hdr = wfdb.rdheader(info.record_name, pn_dir=info.pn_dir)
        except Exception as e:
            print(f"SKIP (header: {e})")
            continue

        if not hasattr(hdr, "seg_name") or not hdr.seg_name:
            print("SKIP (no segments)")
            continue

        # 채널 매핑
        ch_map = {"abp": info.abp_channel}
        if "ecg" in required_types and info.ecg_channel:
            ch_map["ecg"] = info.ecg_channel
        if "ppg" in required_types and info.ppg_channel:
            ch_map["ppg"] = info.ppg_channel

        # 모든 채널이 동시에 존재하는 가장 긴 세그먼트 찾기
        best_seg = None
        best_len = 0

        for seg_name, seg_len in zip(hdr.seg_name, hdr.seg_len):
            if seg_name == "~" or seg_name.endswith("_layout") or seg_len <= 0:
                continue
            try:
                seg_hdr = wfdb.rdheader(seg_name, pn_dir=info.pn_dir)
                if seg_hdr.sig_name is None:
                    continue
                all_present = all(ch in seg_hdr.sig_name for ch in ch_map.values())
                if all_present and seg_len > best_len:
                    best_seg = seg_name
                    best_len = seg_len
            except Exception:
                continue

        if best_seg is None or best_len < int(min_duration_sec * MIMIC3_NATIVE_SR):
            print(f"SKIP (no aligned segment >= {min_duration_sec/60:.0f}min)")
            continue

        # 세그먼트 로드
        try:
            seg = wfdb.rdrecord(best_seg, pn_dir=info.pn_dir)
        except Exception as e:
            print(f"SKIP (read: {e})")
            continue

        if seg.p_signal is None:
            print("SKIP (no signal)")
            continue

        # 각 채널 추출 + 전처리
        signals = {}
        valid = True
        for stype, ch_name in ch_map.items():
            if ch_name not in seg.sig_name:
                valid = False
                break
            ch_idx = seg.sig_name.index(ch_name)
            raw = seg.p_signal[:, ch_idx].astype(np.float64)
            processed = _apply_pipeline(raw, stype, MIMIC3_NATIVE_SR)
            if processed is None or len(processed) < int(min_duration_sec * TARGET_SR):
                valid = False
                break
            signals[stype] = processed

        if not valid or "abp" not in signals:
            print("SKIP (preprocessing failed)")
            continue

        # 동일 길이로 자르기
        min_len = min(len(s) for s in signals.values())
        signals = {k: v[:min_len] for k, v in signals.items()}

        elapsed = time.time() - t0
        dur_min = min_len / TARGET_SR / 60
        print(f"OK ({dur_min:.1f}min, {len(signals)} ch) [{elapsed:.1f}s]")

        cases.append({
            "case_id": info.record_name,
            "patient_id": info.patient_id,
            "signals": signals,
        })

    return cases


# ---- 윈도우 추출 ----


def extract_waveform_pairs(
    cases: list[dict],
    input_signals: list[str],
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
) -> list[dict]:
    """시간 정렬된 다채널 데이터에서 (input_window, target_abp_window) 쌍을 추출한다.

    Returns: list of {"inputs": {stype: array}, "target_abp": array, "case_id", "win_start_sec"}
    """
    win_samples = int(window_sec * TARGET_SR)
    stride_samples = int(stride_sec * TARGET_SR)

    pairs = []
    for case in cases:
        signals = case["signals"]
        abp = signals["abp"]
        n_total = len(abp)

        for start in range(0, n_total - win_samples + 1, stride_samples):
            end = start + win_samples

            # 입력 윈도우
            inputs = {}
            for stype in input_signals:
                if stype in signals:
                    inputs[stype] = signals[stype][start:end]

            if not inputs:
                continue

            # 타겟 ABP 윈도우
            target_abp = abp[start:end]

            # NaN 체크
            if any(np.isnan(v).any() for v in inputs.values()):
                continue
            if np.isnan(target_abp).any():
                continue

            pairs.append({
                "inputs": inputs,
                "target_abp": target_abp,
                "case_id": case["case_id"],
                "win_start_sec": start / TARGET_SR,
            })

    return pairs


# ---- SBP/DBP peak detection ----


def extract_sbp_dbp(abp_window: np.ndarray, sr: float = 100.0) -> tuple[float, float] | None:
    """ABP waveform에서 SBP(systolic)와 DBP(diastolic)를 추출한다.

    Returns: (SBP, DBP) in mmHg, or None if detection fails.
    """
    from scipy.signal import find_peaks

    if len(abp_window) < int(sr * 1):
        return None

    # Systolic peaks (SBP)
    q75, q25 = np.percentile(abp_window, [75, 25])
    iqr = q75 - q25
    if iqr < 5:  # 최소 5mmHg 진폭
        return None

    min_dist = max(1, int(sr * 0.4))  # 최소 0.4초 간격 (150bpm)
    peaks, _ = find_peaks(abp_window, prominence=iqr * 0.3, distance=min_dist)

    if len(peaks) < 2:
        return None

    sbp = float(np.median(abp_window[peaks]))

    # Diastolic troughs (DBP)
    inv = -abp_window
    troughs, _ = find_peaks(inv, prominence=iqr * 0.3, distance=min_dist)

    if len(troughs) < 1:
        dbp = float(np.min(abp_window))
    else:
        dbp = float(np.median(abp_window[troughs]))

    return (sbp, dbp)


# ---- 저장 ----


def save_dataset(
    train_pairs: list[dict],
    test_pairs: list[dict],
    input_signals: list[str],
    window_sec: float,
    out_dir: str,
) -> Path:
    """윈도우 쌍을 .pt로 저장한다."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_tensors(pairs: list[dict]) -> dict:
        if not pairs:
            return {}

        sig_tensors = {}
        for stype in input_signals:
            arrs = [p["inputs"][stype] for p in pairs if stype in p["inputs"]]
            if arrs:
                sig_tensors[stype] = torch.stack([torch.from_numpy(a).float() for a in arrs])

        target_abp = torch.stack([torch.from_numpy(p["target_abp"]).float() for p in pairs])

        # SBP/DBP 추출
        sbp_list, dbp_list = [], []
        for p in pairs:
            result = extract_sbp_dbp(p["target_abp"])
            if result:
                sbp_list.append(result[0])
                dbp_list.append(result[1])
            else:
                sbp_list.append(float("nan"))
                dbp_list.append(float("nan"))

        return {
            "input_signals": sig_tensors,
            "target_abp": target_abp,           # (N, win_samples)
            "sbp": torch.tensor(sbp_list, dtype=torch.float32),  # (N,)
            "dbp": torch.tensor(dbp_list, dtype=torch.float32),  # (N,)
            "case_ids": [p["case_id"] for p in pairs],
        }

    save_dict = {
        "train": _to_tensors(train_pairs),
        "test": _to_tensors(test_pairs),
        "metadata": {
            "task": "cuffless_bp",
            "source": "MIMIC-III Waveform",
            "input_signals": input_signals,
            "target": "ABP waveform (continuous)",
            "window_sec": window_sec,
            "sampling_rate": TARGET_SR,
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
        },
    }

    mode_str = "_".join(input_signals)
    filename = f"cuffless_bp_mimic3_{mode_str}.pt"
    save_path = out_path / filename
    torch.save(save_dict, save_path)

    file_size_mb = save_path.stat().st_size / (1024 * 1024)
    print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")
    return save_path


# ---- 통계 ----


def print_stats(name: str, pairs: list[dict]) -> None:
    if not pairs:
        print(f"  {name}: 0 pairs")
        return

    n = len(pairs)
    sbps, dbps = [], []
    for p in pairs:
        result = extract_sbp_dbp(p["target_abp"])
        if result:
            sbps.append(result[0])
            dbps.append(result[1])

    print(f"  {name}: {n} pairs")
    if sbps:
        print(f"    SBP: {np.mean(sbps):.1f} +/- {np.std(sbps):.1f} mmHg "
              f"[{np.min(sbps):.0f}, {np.max(sbps):.0f}]")
        print(f"    DBP: {np.mean(dbps):.1f} +/- {np.std(dbps):.1f} mmHg "
              f"[{np.min(dbps):.0f}, {np.max(dbps):.0f}]")
    else:
        print(f"    SBP/DBP extraction failed for all pairs")


# ---- 시각화 ----


def _visualize(pairs: list[dict], input_signals: list[str], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping.")
        return

    print("\nGenerating visualizations...")

    # 1. 입력 vs 타겟 ABP 비교 (3개 예시)
    n_show = min(3, len(pairs))
    n_rows = len(input_signals) + 1  # input channels + ABP target
    fig, axes = plt.subplots(n_rows, n_show, figsize=(5 * n_show, 3 * n_rows), squeeze=False)

    for col in range(n_show):
        p = pairs[col]
        t = np.arange(len(p["target_abp"])) / TARGET_SR

        for row, stype in enumerate(input_signals):
            ax = axes[row, col]
            sig = p["inputs"][stype]
            ax.plot(t, sig, linewidth=0.6, color="tab:blue")
            if col == 0:
                ax.set_ylabel(f"{stype.upper()}")
            ax.set_title(f"Input: {stype.upper()}" if row == 0 else "")

        # ABP target
        ax = axes[-1, col]
        abp = p["target_abp"]
        ax.plot(t, abp, linewidth=0.6, color="tab:red")
        if col == 0:
            ax.set_ylabel("ABP (mmHg)")
        ax.set_xlabel("Time (s)")

        result = extract_sbp_dbp(abp)
        if result:
            ax.set_title(f"Target ABP (SBP={result[0]:.0f}, DBP={result[1]:.0f})")
        else:
            ax.set_title("Target ABP")

    mode_str = "+".join(s.upper() for s in input_signals)
    fig.suptitle(f"Cuffless BP: {mode_str} -> ABP Waveform", fontsize=13, y=1.02)
    plt.tight_layout()
    path1 = out_dir / "cuffless_bp_examples.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # 2. SBP/DBP 분포
    sbps, dbps = [], []
    for p in pairs:
        result = extract_sbp_dbp(p["target_abp"])
        if result:
            sbps.append(result[0])
            dbps.append(result[1])

    if sbps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.hist(sbps, bins=40, edgecolor="black", alpha=0.7, color="tab:red")
        ax1.set_title(f"SBP Distribution (n={len(sbps)})")
        ax1.set_xlabel("SBP (mmHg)")
        ax1.set_ylabel("Count")

        ax2.hist(dbps, bins=40, edgecolor="black", alpha=0.7, color="tab:blue")
        ax2.set_title(f"DBP Distribution (n={len(dbps)})")
        ax2.set_xlabel("DBP (mmHg)")
        ax2.set_ylabel("Count")

        plt.tight_layout()
        path2 = out_dir / "cuffless_bp_distribution.png"
        fig.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path2}")


# ---- 메인 ----


def prepare_cuffless_bp(
    input_signals: list[str] | None = None,
    n_cases: int = 5,
    window_sec: float = 10.0,
    stride_sec: float = 5.0,
    train_ratio: float = 0.7,
    out_dir: str = "outputs/downstream/cuffless_bp",
    manifest_path: str | None = None,
    scan_patients: int = 500,
    visualize: bool = False,
) -> Path:
    if input_signals is None:
        input_signals = ["ppg"]

    min_duration_sec = window_sec + stride_sec
    mode_str = " + ".join(s.upper() for s in input_signals)

    print(f"{'='*60}")
    print(f"  Cuffless BP: {mode_str} -> ABP Waveform")
    print(f"  Window: {window_sec}s, Stride: {stride_sec}s")
    print(f"{'='*60}")

    # 1. 데이터 로드
    print(f"\n[1/4] Loading aligned multi-channel data...")
    cases = _load_mimic3_aligned(
        n_cases, input_signals, min_duration_sec,
        manifest_path, scan_patients,
    )

    if not cases:
        print("ERROR: No valid cases.", file=sys.stderr)
        sys.exit(1)

    # 2. Train/Test 분할 (patient 단위)
    print(f"\n[2/4] Splitting by patient (ratio={train_ratio})...")
    rng = np.random.default_rng(42)
    patient_ids = list({c["patient_id"] for c in cases})
    rng.shuffle(patient_ids)
    n_train = max(1, int(len(patient_ids) * train_ratio))
    train_patients = set(patient_ids[:n_train])

    train_cases = [c for c in cases if c["patient_id"] in train_patients]
    test_cases = [c for c in cases if c["patient_id"] not in train_patients]
    print(f"  Train: {len(train_cases)} cases ({len(train_patients)} patients)")
    print(f"  Test:  {len(test_cases)} cases ({len(patient_ids) - len(train_patients)} patients)")

    # 3. 윈도우 추출
    print(f"\n[3/4] Extracting waveform pairs...")
    train_pairs = extract_waveform_pairs(train_cases, input_signals, window_sec, stride_sec)
    test_pairs = extract_waveform_pairs(test_cases, input_signals, window_sec, stride_sec)

    print_stats("Train", train_pairs)
    print_stats("Test", test_pairs)

    if not train_pairs and not test_pairs:
        print("ERROR: No pairs extracted.", file=sys.stderr)
        sys.exit(1)

    # 4. 저장
    print(f"\n[4/4] Saving...")
    save_path = save_dataset(train_pairs, test_pairs, input_signals, window_sec, out_dir)

    if visualize:
        all_pairs = train_pairs + test_pairs
        _visualize(all_pairs, input_signals, Path(out_dir))

    total = len(train_pairs) + len(test_pairs)
    print(f"\n{'='*60}")
    print(f"  Cuffless BP data ready: {total} pairs")
    print(f"  File: {save_path}")
    print(f"{'='*60}")
    return save_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cuffless BP - Data Preparation (PPG/ECG -> ABP Waveform)",
    )
    parser.add_argument("--input-signals", nargs="+", default=["ppg"],
                        choices=["ppg", "ecg"],
                        help="Input signal types (target is always ABP)")
    parser.add_argument("--n-cases", type=int, default=5)
    parser.add_argument("--window-sec", type=float, default=10.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--out-dir", type=str, default="outputs/downstream/cuffless_bp")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--scan-patients", type=int, default=500)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    prepare_cuffless_bp(
        input_signals=args.input_signals,
        n_cases=args.n_cases,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        train_ratio=args.train_ratio,
        out_dir=args.out_dir,
        manifest_path=args.manifest,
        scan_patients=args.scan_patients,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
