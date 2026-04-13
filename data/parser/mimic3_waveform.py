# -*- coding:utf-8 -*-
"""MIMIC-III Waveform Database → downstream 전처리 스크립트.

PhysioNet MIMIC-III Waveform Matched Subset (open access)에서
ABP 포함 레코드를 스트리밍으로 읽고, 전처리 후 .pt로 저장한다.

Hypotension Prediction downstream task의 external evaluation 데이터로 사용.

사용법:
    # ABP manifest 스캔 (헤더만 읽어 ABP 레코드 목록 생성)
    python -m data.parser.mimic3_waveform scan --max-records 100

    # 소수 케이스 전처리 + 저장
    python -m data.parser.mimic3_waveform parse --n-cases 5

    # 시각화 포함
    python -m data.parser.mimic3_waveform parse --n-cases 3 --visualize
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from data.parser._common import (
    resample_to_target,
)
from data.parser.vitaldb import (
    SIGNAL_CONFIGS,
    _apply_filter,
    _apply_median_filter,
    _apply_range_check,
    _detect_electrocautery,
    _extract_nan_free_segments,
)


# ── 상수 ──────────────────────────────────────────────────────

TARGET_SR: float = 100.0
MIMIC3_NATIVE_SR: float = 125.0  # MIMIC-III Waveform 공통 sampling rate

# PhysioNet 경로 (Matched Subset -MIMIC-III Clinical DB 연결 가능)
PN_DB = "mimic3wdb-matched/1.0"

# ABP 채널명 후보 (MIMIC-III에서 사용되는 이름들)
ABP_CHANNEL_NAMES = {"ABP", "ART"}

# ECG 채널명 후보
ECG_CHANNEL_NAMES = {"II", "I", "III", "V", "aVR", "aVL", "aVF", "MCL1"}

# PPG 채널명 후보
PPG_CHANNEL_NAMES = {"PLETH"}


# ── 데이터 구조 ──────────────────────────────────────────────────


@dataclass
class MimicRecordInfo:
    """MIMIC-III 레코드 메타정보."""

    record_name: str  # e.g., "p000020-2183-04-28-17-47"
    pn_dir: str  # e.g., "mimic3wdb-matched/1.0/p00/p000020"
    patient_id: str  # e.g., "p000020"
    has_abp: bool = False
    has_ecg: bool = False
    has_ppg: bool = False
    abp_channel: str = ""  # 실제 ABP 채널명 ("ABP" or "ART")
    ecg_channel: str = ""
    ppg_channel: str = ""
    n_segments: int = 0
    total_samples: int = 0  # ABP 신호 총 샘플 수


@dataclass
class MimicCaseData:
    """전처리된 MIMIC-III 케이스 데이터."""

    record_name: str
    patient_id: str
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    # signals: {"abp": (n_timesteps,), "ecg": (n_timesteps,), ...} at TARGET_SR
    numerics: dict[str, np.ndarray] = field(default_factory=dict)
    # numerics: {"ABPMean": (n_samples,), "ABPSys": ..., "ABPDias": ...}


# ── Manifest 스캔 ────────────────────────────────────────────


def scan_abp_records(
    max_records: int = 0,
    save_path: str | None = None,
    verbose: bool = True,
) -> list[MimicRecordInfo]:
    """MIMIC-III Matched Subset에서 ABP 포함 레코드를 스캔한다.

    Parameters
    ----------
    max_records : 스캔할 최대 환자 수. 0이면 전체.
    save_path : manifest JSON 저장 경로. None이면 저장 안 함.
    verbose : 진행 상황 출력.

    Returns
    -------
    ABP 포함 레코드 정보 리스트.
    """
    import wfdb

    if verbose:
        print(f"Scanning MIMIC-III Matched Subset ({PN_DB})...")

    # 최상위 레코드 목록 가져오기
    all_records = wfdb.get_record_list(PN_DB)
    if verbose:
        print(f"  Total records in DB: {len(all_records)}")

    # 환자별 그룹화 (p00/p000020 형태)
    patient_dirs: dict[str, str] = {}  # patient_id -> pn_dir
    for rec in all_records:
        parts = rec.split("/")
        if len(parts) >= 2:
            patient_id = parts[1]
            pn_dir = f"{PN_DB}/{parts[0]}/{parts[1]}"
            patient_dirs[patient_id] = pn_dir

    if verbose:
        print(f"  Unique patients: {len(patient_dirs)}")

    if max_records > 0:
        patient_items = list(patient_dirs.items())[:max_records]
    else:
        patient_items = list(patient_dirs.items())

    abp_records: list[MimicRecordInfo] = []
    n_scanned = 0
    n_errors = 0

    for patient_id, pn_dir in patient_items:
        n_scanned += 1
        try:
            # 환자 디렉토리의 sub-records 가져오기
            sub_records = wfdb.get_record_list(pn_dir)
            if not sub_records:
                continue

            # master waveform record 찾기
            # 형태: "p000020-2183-04-28-17-47" (환자ID-날짜-시각)
            # 제외: numerics("n" suffix), 개별 segment("_0001" 등)
            wave_records = [
                r
                for r in sub_records
                if r.startswith(patient_id) and not r.endswith("n")
            ]

            for rec_name in wave_records:
                try:
                    # layout header에서 채널 확인
                    hdr = wfdb.rdheader(rec_name, pn_dir=pn_dir)

                    # Multi-segment인 경우 layout에서 채널 확인
                    if hasattr(hdr, "seg_name") and hdr.seg_name:
                        layout_name = hdr.seg_name[0]
                        if layout_name and not layout_name.startswith("~"):
                            layout = wfdb.rdheader(layout_name, pn_dir=pn_dir)
                            sig_names = layout.sig_name if layout.sig_name else []
                        else:
                            sig_names = []
                    elif hdr.sig_name:
                        sig_names = hdr.sig_name
                    else:
                        continue

                    # ABP 채널 확인
                    abp_ch = ""
                    for ch in sig_names:
                        if ch in ABP_CHANNEL_NAMES:
                            abp_ch = ch
                            break

                    if not abp_ch:
                        continue

                    # ECG/PPG 채널 확인
                    ecg_ch = ""
                    for ch in sig_names:
                        if ch in ECG_CHANNEL_NAMES:
                            ecg_ch = ch
                            break

                    ppg_ch = ""
                    for ch in sig_names:
                        if ch in PPG_CHANNEL_NAMES:
                            ppg_ch = ch
                            break

                    info = MimicRecordInfo(
                        record_name=rec_name,
                        pn_dir=pn_dir,
                        patient_id=patient_id,
                        has_abp=True,
                        has_ecg=bool(ecg_ch),
                        has_ppg=bool(ppg_ch),
                        abp_channel=abp_ch,
                        ecg_channel=ecg_ch,
                        ppg_channel=ppg_ch,
                        n_segments=len(hdr.seg_name) if hasattr(hdr, "seg_name") else 1,
                    )
                    abp_records.append(info)

                except Exception:
                    n_errors += 1
                    continue

        except Exception:
            n_errors += 1
            continue

        if verbose and n_scanned % 50 == 0:
            print(
                f"  Scanned {n_scanned}/{len(patient_items)} patients, "
                f"found {len(abp_records)} ABP records ({n_errors} errors)"
            )

    if verbose:
        print(
            f"\nScan complete: {len(abp_records)} ABP records from "
            f"{n_scanned} patients ({n_errors} errors)"
        )

    # manifest 저장
    if save_path:
        manifest = [
            {
                "record_name": r.record_name,
                "pn_dir": r.pn_dir,
                "patient_id": r.patient_id,
                "abp_channel": r.abp_channel,
                "ecg_channel": r.ecg_channel,
                "ppg_channel": r.ppg_channel,
                "has_ecg": r.has_ecg,
                "has_ppg": r.has_ppg,
                "n_segments": r.n_segments,
            }
            for r in abp_records
        ]
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(manifest, f, indent=2)
        if verbose:
            print(f"Manifest saved: {save_path}")

    return abp_records


def load_manifest(path: str) -> list[MimicRecordInfo]:
    """저장된 manifest JSON에서 레코드 정보를 로드한다."""
    with open(path) as f:
        data = json.load(f)
    return [
        MimicRecordInfo(
            record_name=d["record_name"],
            pn_dir=d["pn_dir"],
            patient_id=d["patient_id"],
            has_abp=True,
            has_ecg=d.get("has_ecg", False),
            has_ppg=d.get("has_ppg", False),
            abp_channel=d["abp_channel"],
            ecg_channel=d.get("ecg_channel", ""),
            ppg_channel=d.get("ppg_channel", ""),
            n_segments=d.get("n_segments", 0),
        )
        for d in data
    ]


# ── 단일 레코드 로드 + 전처리 ────────────────────────────────


def load_and_preprocess_record(
    info: MimicRecordInfo,
    signal_types: list[str] | None = None,
    min_duration_s: float = 60.0,
    verbose: bool = False,
) -> MimicCaseData | None:
    """MIMIC-III 레코드를 스트리밍으로 읽고 전처리한다.

    Parameters
    ----------
    info : 레코드 메타정보.
    signal_types : 추출할 signal types. None이면 ["abp"].
    min_duration_s : 최소 유효 신호 길이 (초).
    verbose : 진행 상황 출력.

    Returns
    -------
    MimicCaseData 또는 None (유효 데이터 없으면).
    """
    import wfdb

    if signal_types is None:
        signal_types = ["abp"]

    # 채널 매핑
    channel_map: dict[str, str] = {}  # signal_type -> actual channel name
    if "abp" in signal_types and info.abp_channel:
        channel_map["abp"] = info.abp_channel
    if "ecg" in signal_types and info.ecg_channel:
        channel_map["ecg"] = info.ecg_channel
    if "ppg" in signal_types and info.ppg_channel:
        channel_map["ppg"] = info.ppg_channel

    if "abp" not in channel_map:
        return None  # ABP 필수

    case = MimicCaseData(
        record_name=info.record_name,
        patient_id=info.patient_id,
    )

    try:
        # Master header 읽기
        hdr = wfdb.rdheader(info.record_name, pn_dir=info.pn_dir)
    except Exception as e:
        if verbose:
            print(f"  Error reading header {info.record_name}: {e}")
        return None

    # Multi-segment 레코드 처리
    if hasattr(hdr, "seg_name") and hdr.seg_name:
        segments_data = _load_multi_segment(
            hdr,
            info.pn_dir,
            channel_map,
            min_duration_s,
            verbose,
        )
    else:
        # Single-segment
        segments_data = _load_single_segment(
            info.record_name,
            info.pn_dir,
            channel_map,
            verbose,
        )

    if not segments_data:
        return None

    # 각 signal type별로 가장 긴 세그먼트를 전처리
    for stype, raw_signal in segments_data.items():
        processed = _apply_pipeline(raw_signal, stype, MIMIC3_NATIVE_SR)
        if processed is not None and len(processed) >= int(min_duration_s * TARGET_SR):
            case.signals[stype] = processed

    if "abp" not in case.signals:
        return None

    # Numerics 로드 시도 (MAP/SBP/DBP)
    try:
        case.numerics = _load_numerics(info, verbose)
    except Exception:
        pass  # numerics 없어도 ABP 파형에서 MAP 계산 가능

    return case


def _load_multi_segment(
    hdr,
    pn_dir: str,
    channel_map: dict[str, str],
    min_duration_s: float,
    verbose: bool,
) -> dict[str, np.ndarray]:
    """Multi-segment 레코드에서 각 signal type의 가장 긴 연속 데이터를 추출."""
    import wfdb

    # signal type별로 모든 세그먼트의 데이터를 수집
    collected: dict[str, list[np.ndarray]] = {st: [] for st in channel_map}

    for seg_name, seg_len in zip(hdr.seg_name, hdr.seg_len):
        # gap 또는 layout 세그먼트 스킵
        if seg_name == "~" or seg_name.endswith("_layout") or seg_len <= 0:
            continue

        try:
            seg = wfdb.rdrecord(seg_name, pn_dir=pn_dir)
        except Exception:
            continue

        if seg.p_signal is None or seg.sig_name is None:
            continue

        for stype, ch_name in channel_map.items():
            if ch_name in seg.sig_name:
                ch_idx = seg.sig_name.index(ch_name)
                data = seg.p_signal[:, ch_idx].astype(np.float64)
                # NaN이 너무 많으면 스킵
                nan_ratio = np.isnan(data).sum() / len(data)
                if nan_ratio < 0.5 and len(data) > int(
                    min_duration_s * MIMIC3_NATIVE_SR
                ):
                    collected[stype].append(data)

    # 각 signal type에서 가장 긴 세그먼트 선택
    result: dict[str, np.ndarray] = {}
    for stype, segments in collected.items():
        if segments:
            longest = max(segments, key=len)
            result[stype] = longest

    return result


def _load_single_segment(
    record_name: str,
    pn_dir: str,
    channel_map: dict[str, str],
    verbose: bool,
) -> dict[str, np.ndarray]:
    """Single-segment 레코드에서 데이터를 추출."""
    import wfdb

    try:
        seg = wfdb.rdrecord(record_name, pn_dir=pn_dir)
    except Exception:
        return {}

    if seg.p_signal is None or seg.sig_name is None:
        return {}

    result: dict[str, np.ndarray] = {}
    for stype, ch_name in channel_map.items():
        if ch_name in seg.sig_name:
            ch_idx = seg.sig_name.index(ch_name)
            data = seg.p_signal[:, ch_idx].astype(np.float64)
            result[stype] = data

    return result


def _load_numerics(info: MimicRecordInfo, verbose: bool) -> dict[str, np.ndarray]:
    """Numerics 레코드에서 MAP/SBP/DBP를 로드한다."""
    import wfdb

    numerics_name = info.record_name + "n"
    try:
        hdr = wfdb.rdheader(numerics_name, pn_dir=info.pn_dir)
    except Exception:
        return {}

    # Multi-segment numerics 처리
    if hasattr(hdr, "seg_name") and hdr.seg_name:
        all_data: dict[str, list[float]] = {}
        target_channels = {"ABPMean", "ABPSys", "ABPDias"}

        for seg_name, seg_len in zip(hdr.seg_name, hdr.seg_len):
            if seg_name == "~" or seg_name.endswith("_layout") or seg_len <= 0:
                continue
            try:
                seg = wfdb.rdrecord(seg_name, pn_dir=info.pn_dir)
            except Exception:
                continue
            if seg.p_signal is None or seg.sig_name is None:
                continue

            for ch in target_channels:
                if ch in seg.sig_name:
                    ch_idx = seg.sig_name.index(ch)
                    vals = seg.p_signal[:, ch_idx].flatten()
                    if ch not in all_data:
                        all_data[ch] = []
                    all_data[ch].extend(vals.tolist())

        return {k: np.array(v) for k, v in all_data.items()}

    return {}


def _apply_pipeline(
    data: np.ndarray,
    stype: str,
    native_sr: float,
) -> np.ndarray | None:
    """VitalDB 파서와 동일한 전처리 파이프라인 적용.

    Range check → Spike detection → NaN segment extraction →
    Median → Filter → Resample (125→100Hz).
    """
    cfg = SIGNAL_CONFIGS.get(stype)
    if cfg is None:
        return None

    # Step 1: Range check
    if cfg.valid_range is not None:
        data, _ = _apply_range_check(data, cfg.valid_range)

    # Step 2: Spike detection
    if cfg.spike_detection:
        data, _ = _detect_electrocautery(
            data, native_sr, threshold_std=cfg.spike_threshold_std
        )

    # Step 3: NaN-free segments (가장 긴 것 사용)
    min_samples = int(60.0 * native_sr)
    segments = _extract_nan_free_segments(data, min_samples)
    if not segments:
        return None

    segment = max(segments, key=len)

    # Step 4: Median → Filter
    if cfg.median_kernel > 0:
        segment = _apply_median_filter(segment, kernel_size=cfg.median_kernel)
    segment = _apply_filter(segment, cfg, native_sr)

    # Step 5: Resample to TARGET_SR (125 → 100 Hz)
    if native_sr != TARGET_SR:
        segment = resample_to_target(segment, orig_sr=native_sr, target_sr=TARGET_SR)

    return segment


# ── 배치 파싱 + 저장 ──────────────────────────────────────────


def parse_and_save(
    n_cases: int = 5,
    signal_types: list[str] | None = None,
    manifest_path: str | None = None,
    out_dir: str = "outputs/downstream/mimic3",
    visualize: bool = False,
    verbose: bool = True,
) -> Path:
    """MIMIC-III에서 n_cases개 ABP 레코드를 파싱하여 .pt로 저장한다.

    Parameters
    ----------
    n_cases : 파싱할 케이스 수.
    signal_types : 추출할 signal types. None이면 ["abp"].
    manifest_path : 기존 manifest JSON 경로. None이면 실시간 스캔.
    out_dir : 저장 디렉토리.
    visualize : True이면 전처리 결과 시각화.
    verbose : 진행 상황 출력.

    Returns
    -------
    저장된 .pt 파일 경로.
    """
    if signal_types is None:
        signal_types = ["abp"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── 1. Manifest 로드 또는 스캔 ──
    if manifest_path and Path(manifest_path).exists():
        if verbose:
            print(f"[1/4] Loading manifest: {manifest_path}")
        records = load_manifest(manifest_path)
    else:
        if verbose:
            print(f"[1/4] Scanning for ABP records (max {n_cases * 3} patients)...")
        # ABP 포함 레코드를 찾기 위해 넉넉하게 스캔
        records = scan_abp_records(
            max_records=n_cases * 3,
            save_path=str(out_path / "mimic3_abp_manifest.json"),
            verbose=verbose,
        )

    if not records:
        print("ERROR: No ABP records found.", file=sys.stderr)
        sys.exit(1)

    # n_cases개만 선택
    records = records[:n_cases]
    if verbose:
        print(f"\n[2/4] Processing {len(records)} records...")

    # ── 2. 각 레코드 로드 + 전처리 ──
    cases: list[MimicCaseData] = []
    for i, info in enumerate(records):
        if verbose:
            print(
                f"  [{i + 1}/{len(records)}] {info.record_name} "
                f"(ABP={info.abp_channel}, ECG={info.ecg_channel or 'N/A'})..."
            )
        t0 = time.time()
        case = load_and_preprocess_record(info, signal_types, verbose=verbose)
        elapsed = time.time() - t0

        if case and case.signals:
            cases.append(case)
            abp_len = len(case.signals.get("abp", []))
            if verbose:
                print(
                    f"    OK -ABP: {abp_len} samples "
                    f"({abp_len / TARGET_SR:.1f}s at {TARGET_SR}Hz) [{elapsed:.1f}s]"
                )
        else:
            if verbose:
                print(f"    SKIP -no valid data [{elapsed:.1f}s]")

    if not cases:
        print("ERROR: No valid cases after preprocessing.", file=sys.stderr)
        sys.exit(1)

    # ── 3. .pt 저장 ──
    if verbose:
        print(f"\n[3/4] Saving {len(cases)} cases to .pt...")

    save_dict = {
        "cases": [],
        "metadata": {
            "source": "MIMIC-III Waveform Matched Subset",
            "pn_db": PN_DB,
            "n_cases": len(cases),
            "signal_types": signal_types,
            "sampling_rate": TARGET_SR,
            "native_sr": MIMIC3_NATIVE_SR,
        },
    }

    for case in cases:
        case_dict = {
            "record_name": case.record_name,
            "patient_id": case.patient_id,
            "signals": {
                stype: torch.from_numpy(sig).float()
                for stype, sig in case.signals.items()
            },
        }
        if case.numerics:
            case_dict["numerics"] = {
                k: torch.from_numpy(v).float() for k, v in case.numerics.items()
            }
        save_dict["cases"].append(case_dict)

    save_path = out_path / "mimic3_abp_data.pt"
    torch.save(save_dict, save_path)
    file_size_mb = save_path.stat().st_size / (1024 * 1024)

    if verbose:
        print(f"  Saved: {save_path} ({file_size_mb:.2f} MB)")

    # ── 4. 통계 출력 ──
    if verbose:
        print("\n[4/4] Statistics:")
        total_duration = 0.0
        for case in cases:
            abp = case.signals.get("abp")
            if abp is not None:
                dur = len(abp) / TARGET_SR
                total_duration += dur
                print(
                    f"  {case.record_name}: ABP {dur:.0f}s "
                    f"({dur / 60:.1f}min), range=[{abp.min():.1f}, {abp.max():.1f}] mmHg"
                )

        print(
            f"\n  Total: {len(cases)} cases, {total_duration:.0f}s "
            f"({total_duration / 3600:.1f}h)"
        )
        print(f"  File: {save_path} ({file_size_mb:.2f} MB)")

    # ── 시각화 ──
    if visualize:
        _visualize_cases(cases, out_path)

    return save_path


# ── 시각화 ────────────────────────────────────────────────────


def _visualize_cases(cases: list[MimicCaseData], out_dir: Path) -> None:
    """전처리된 케이스들의 ABP 파형을 시각화한다."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping visualization.")
        return

    print("\nGenerating visualizations...")

    # 1. 각 케이스의 ABP 파형 (처음 60초)
    n_cases = len(cases)
    fig, axes = plt.subplots(n_cases, 1, figsize=(14, 3 * n_cases), squeeze=False)
    fig.suptitle(
        "MIMIC-III ABP Waveforms (first 60s, preprocessed)", fontsize=14, y=1.02
    )

    for i, case in enumerate(cases):
        ax = axes[i, 0]
        abp = case.signals.get("abp")
        if abp is None:
            continue

        # 처음 60초만 표시
        n_show = min(len(abp), int(60 * TARGET_SR))
        t = np.arange(n_show) / TARGET_SR
        ax.plot(t, abp[:n_show], linewidth=0.5, color="tab:red")
        ax.axhline(
            y=65,
            color="orange",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="MAP=65 (threshold)",
        )
        ax.set_ylabel("ABP (mmHg)")
        ax.set_title(f"{case.record_name} (patient: {case.patient_id})")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlim(0, 60)

    axes[-1, 0].set_xlabel("Time (s)")
    plt.tight_layout()
    path1 = out_dir / "mimic3_abp_waveforms.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path1}")

    # 2. MAP 분포 (10초 윈도우 평균)
    all_maps = []
    for case in cases:
        abp = case.signals.get("abp")
        if abp is None:
            continue
        win_size = int(10 * TARGET_SR)
        for start in range(0, len(abp) - win_size + 1, win_size):
            win_map = float(np.mean(abp[start : start + win_size]))
            all_maps.append(win_map)

    if all_maps:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # MAP histogram
        ax1.hist(all_maps, bins=50, edgecolor="black", alpha=0.7, color="tab:blue")
        ax1.axvline(
            x=65, color="red", linestyle="--", linewidth=2, label="MAP=65 threshold"
        )
        n_hypo = sum(1 for m in all_maps if m < 65)
        ax1.set_title(
            f"MAP Distribution (10s windows)\n"
            f"Total: {len(all_maps)}, Hypotension: {n_hypo} "
            f"({n_hypo / len(all_maps) * 100:.1f}%)"
        )
        ax1.set_xlabel("MAP (mmHg)")
        ax1.set_ylabel("Count")
        ax1.legend()

        # MAP over time (첫 번째 케이스)
        first_abp = cases[0].signals.get("abp")
        if first_abp is not None:
            win_maps = []
            win_times = []
            for start in range(0, len(first_abp) - win_size + 1, win_size):
                win_maps.append(float(np.mean(first_abp[start : start + win_size])))
                win_times.append(start / TARGET_SR / 60)  # minutes
            ax2.plot(
                win_times, win_maps, "o-", markersize=2, linewidth=1, color="tab:red"
            )
            ax2.axhline(y=65, color="orange", linestyle="--", linewidth=1)
            ax2.fill_between(
                win_times, 0, 65, alpha=0.1, color="red", label="Hypotension zone"
            )
            ax2.set_title(f"MAP over time -{cases[0].record_name}")
            ax2.set_xlabel("Time (min)")
            ax2.set_ylabel("MAP (mmHg)")
            ax2.legend(fontsize=8)

        plt.tight_layout()
        path2 = out_dir / "mimic3_map_distribution.png"
        fig.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path2}")

    # 3. 품질 통계 요약
    print("\n  MAP Statistics:")
    if all_maps:
        maps_arr = np.array(all_maps)
        print(f"    Mean: {maps_arr.mean():.1f} mmHg")
        print(f"    Std:  {maps_arr.std():.1f} mmHg")
        print(f"    Min:  {maps_arr.min():.1f}, Max: {maps_arr.max():.1f}")
        print(
            f"    Hypotension (MAP<65): {n_hypo}/{len(all_maps)} "
            f"({n_hypo / len(all_maps) * 100:.1f}%)"
        )


# ── CLI ──────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MIMIC-III Waveform → downstream 전처리",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan 서브커맨드
    scan_parser = subparsers.add_parser("scan", help="ABP 레코드 manifest 스캔")
    scan_parser.add_argument("--max-records", type=int, default=100)
    scan_parser.add_argument("--out-dir", type=str, default="outputs/downstream/mimic3")

    # parse 서브커맨드
    parse_parser = subparsers.add_parser("parse", help="레코드 파싱 + 전처리 + 저장")
    parse_parser.add_argument("--n-cases", type=int, default=5)
    parse_parser.add_argument("--signal-types", nargs="+", default=["abp"])
    parse_parser.add_argument("--manifest", type=str, default=None)
    parse_parser.add_argument(
        "--out-dir", type=str, default="outputs/downstream/mimic3"
    )
    parse_parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    if args.command == "scan":
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        scan_abp_records(
            max_records=args.max_records,
            save_path=str(out_dir / "mimic3_abp_manifest.json"),
        )

    elif args.command == "parse":
        parse_and_save(
            n_cases=args.n_cases,
            signal_types=args.signal_types,
            manifest_path=args.manifest,
            out_dir=args.out_dir,
            visualize=args.visualize,
        )


if __name__ == "__main__":
    main()
