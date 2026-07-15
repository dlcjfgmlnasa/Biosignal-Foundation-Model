# -*- coding:utf-8 -*-
"""In-Hospital Mortality (24h Landmark) — Pass 1: 원격 헤더 스캔 → 다운로드 목록 빌드.

`02_mortality_landmark_cohort.sql` 로 뽑은 cohort CSV + matched-subset 인덱스
(`RECORDS-waveforms`) 를 받아, **실제 신호(.dat)를 한 바이트도 받지 않고** 원격
헤더만 파싱해서 다운로드 대상을 확정한다.

왜 이렇게 하는가 (기존 `build_records_to_download.py` + `download_waveforms.py` 결함):
  1. 구 코드는 레코드 **시작시각**만으로 창 포함 여부를 판정했다
     (`icu_in <= rec_dt <= icu_out`). ICU 입실 전 시작해 입실 후로 이어지는
     레코드가 통째로 탈락한다. → 여기서는 헤더의 `sig_len/fs` 로 **종료시각**을
     구해 구간 겹침(overlap)으로 판정한다.
  2. 구 코드는 ICU stay **전체**를 받았다. landmark 는 첫 24h 만 필요하다.
     게다가 matched-subset 레코드 하나가 90시간을 넘기도 한다 →
     **세그먼트 단위로 24h 창에 겹치는 것만** 골라 받는다 (용량 대폭 절감).
  3. 채널 요건(최소 ECG)은 clinical DB 로 판정 불가. layout 세그먼트 헤더의
     signal name union 으로 게이팅한다.
  4. 헤더를 디스크에 미리 떨궈 놓으면 `.hea` 존재만 보고 skip 하는 다운로더
     함정(거짓 "완료")과 충돌한다. → Pass 1 은 **원격 파싱만** 하고 아무것도
     쓰지 않는다. 실제 파일 쓰기는 Pass 2 (`download_landmark.py`) 가 전담한다.

출력:
  - ``landmark_manifest.json``  환자별 라벨·레코드·세그먼트·채널·attrition 전체
  - ``FILES-to-download``       내려받을 파일의 상대경로 목록 (Pass 2 입력)
  - ``RECORDS-to-download``     레코드 경로 목록 (기존 도구 호환용)

사용법:
    python -m downstream.outcome.mortality.build_landmark_records \
        --cohort-csv downstream/outcome/mortality/bquxjob_landmark.csv \
        --records-file downstream/outcome/sepsis/RECORDS-waveforms \
        --out-dir downstream/outcome/mortality \
        --landmark-hours 24 --workers 12
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path


PN_DB = "mimic3wdb-matched"

# WFDB sig_name → signal_type. prepare_data.py 의 MIMIC_SIGNAL_MAP 과 동일 (SSOT 복제).
MIMIC_SIGNAL_MAP: dict[str, str] = {
    "II": "ecg",
    "V": "ecg",
    "ABP": "abp",
    "ART": "abp",
    "PLETH": "ppg",
    "CO2": "co2",
    "RESP": "resp_impedance",
}

# 채널 요건: 최소 ECG 1채널. ECG+ABP+PPG 를 모두 가지면 "ideal" 로 표기(제외 아님).
REQUIRED_MODALITIES: set[str] = {"ecg"}
IDEAL_MODALITIES: set[str] = {"ecg", "abp", "ppg"}

# cohort CSV 필수 컬럼 — 하나라도 없으면 구 SQL(01_) 로 뽑은 CSV 다.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject_id", "icu_intime", "icu_outtime", "hospital_expire_flag",
    "mortality_28d", "died_in_window", "insufficient_obs",
)


# ────────────────────────────────────────────────────────────────
# 파싱 유틸
# ────────────────────────────────────────────────────────────────
def parse_datetime_str(s: str) -> datetime | None:
    """BigQuery/ISO 계열 datetime 문자열 파싱."""
    if not s or not s.strip():
        return None
    s = s.strip().replace("T", " ").split(".")[0].replace(" UTC", "")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def parse_record_datetime(record_path: str) -> tuple[int, datetime | None]:
    """`pXX/pXXXXXX/pXXXXXX-YYYY-MM-DD-hh-mm` → (subject_id, 시작시각)."""
    parts = record_path.strip().split("/")
    if len(parts) < 3:
        return 0, None
    try:
        sid = int(parts[1][1:])
    except ValueError:
        return 0, None
    m = re.match(r"p\d+-(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})", parts[-1])
    if m is None:
        return sid, None
    try:
        return sid, datetime(*(int(g) for g in m.groups()))  # type: ignore[arg-type]
    except ValueError:
        return sid, None


def load_waveform_index(records_file: Path) -> dict[int, list[str]]:
    """RECORDS-waveforms → {subject_id: [record_path, ...]}"""
    index: dict[int, list[str]] = {}
    with open(records_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sid, _ = parse_record_datetime(line)
            if sid:
                index.setdefault(sid, []).append(line)
    return index


# ────────────────────────────────────────────────────────────────
# Cohort
# ────────────────────────────────────────────────────────────────
@dataclass
class Attrition:
    """CONSORT 감쇠 카운터 — 제외 사유별 환자 수."""
    n_cohort: int = 0                 # CSV 행 수 (= 성인 · 환자당 첫 ICU stay)
    n_died_in_window: int = 0         # 관측창 내 사망 (leakage)
    n_insufficient_obs: int = 0       # 24h 관측 미충족
    n_eligible: int = 0               # landmark 자격
    n_no_waveform: int = 0            # matched subset 에 waveform 없음
    n_no_overlap: int = 0             # landmark 창에 겹치는 세그먼트 없음
    n_no_required_channel: int = 0    # ECG 부재
    n_header_error: int = 0           # 원격 헤더 파싱 실패
    n_final: int = 0                  # 최종 코호트
    n_final_ideal: int = 0            # 그 중 ECG+ABP+PPG 모두 보유

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


@dataclass
class Patient:
    subject_id: int
    icustay_id: str
    icu_intime: datetime
    label: int                        # hospital_expire_flag (primary)
    label_28d: int                    # mortality_28d (aux)
    records: list[dict] = field(default_factory=list)


def load_landmark_cohort(cohort_csv: Path, attr: Attrition) -> list[Patient]:
    """cohort CSV → landmark 자격 환자. 제외 사유를 attr 에 누적한다."""
    with open(cohort_csv, newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            sys.exit(
                f"ERROR: cohort CSV 에 컬럼이 없습니다: {missing}\n"
                f"       구 SQL(01_mortality_cohort.sql) 로 뽑은 CSV 로 보입니다.\n"
                f"       sql/02_mortality_landmark_cohort.sql 을 BigQuery 에서\n"
                f"       재실행해 새 CSV 를 export 하세요."
            )

        patients: list[Patient] = []
        for row in reader:
            attr.n_cohort += 1
            died = row["died_in_window"] == "1"
            insuf = row["insufficient_obs"] == "1"
            if died:
                attr.n_died_in_window += 1
            if insuf:
                attr.n_insufficient_obs += 1
            if died or insuf:
                continue

            t0 = parse_datetime_str(row["icu_intime"])
            if t0 is None:
                attr.n_insufficient_obs += 1
                continue

            attr.n_eligible += 1
            patients.append(Patient(
                subject_id=int(row["subject_id"]),
                icustay_id=row.get("icustay_id", ""),
                icu_intime=t0,
                label=int(row["hospital_expire_flag"]),
                label_28d=int(row["mortality_28d"]),
            ))
    return patients


# ────────────────────────────────────────────────────────────────
# 원격 헤더 스캔 (디스크에 아무것도 쓰지 않는다)
# ────────────────────────────────────────────────────────────────
def _rdheader(name: str, pn_dir: str, retries: int = 3):
    """네트워크 재시도를 붙인 wfdb.rdheader."""
    import wfdb
    last: Exception | None = None
    for _ in range(retries):
        try:
            return wfdb.rdheader(name, pn_dir=pn_dir)
        except Exception as e:  # noqa: BLE001 — 네트워크/404 모두 재시도 후 상위 보고
            last = e
    raise RuntimeError(f"rdheader 실패 {pn_dir}/{name}: {last}")


def scan_record(
    record_path: str,
    t0: datetime,
    t1: datetime,
) -> dict | None:
    """레코드 하나를 원격 헤더로 스캔 → landmark 창에 겹치는 세그먼트만 추린다.

    반환 None = 창에 겹치는 세그먼트 없음.
    반환 dict 의 ``channels`` 는 layout 세그먼트가 선언한 signal name union.
    """
    pXX, pid, rec_name = record_path.split("/")
    pn_dir = f"{PN_DB}/{pXX}/{pid}"

    hdr = _rdheader(rec_name, pn_dir=pn_dir)
    fs = float(hdr.fs)

    # 시작시각: 헤더의 base_datetime 우선, 없으면 파일명.
    base = getattr(hdr, "base_datetime", None)
    if base is None:
        _, base = parse_record_datetime(record_path)
    if base is None:
        raise RuntimeError(f"시작시각 불명: {record_path}")

    files: list[str] = [f"{rec_name}.hea"]
    segments: list[dict] = []
    channels: set[str] = set()

    seg_names = getattr(hdr, "seg_name", None)
    if seg_names is None:
        # 단일 세그먼트 레코드 — 헤더 자체가 채널을 선언한다.
        rec_end = base + timedelta(seconds=hdr.sig_len / fs)
        if rec_end <= t0 or base >= t1:
            return None
        channels.update(hdr.sig_name or [])
        files.append(f"{rec_name}.dat")
        segments.append({"name": rec_name, "start": base.isoformat(),
                         "dur_sec": hdr.sig_len / fs})
    else:
        # multi-segment: layout 헤더가 채널 union 을 선언, '~' 는 gap(파일 없음).
        layout = next((s for s in seg_names if s.endswith("_layout")), None)
        if layout:
            files.append(f"{layout}.hea")
            lhdr = _rdheader(layout, pn_dir=pn_dir)
            channels.update(lhdr.sig_name or [])

        offset = 0
        for name, seg_len in zip(seg_names, hdr.seg_len):
            seg_start = base + timedelta(seconds=offset / fs)
            seg_end = base + timedelta(seconds=(offset + seg_len) / fs)
            offset += seg_len
            if name == "~" or seg_len == 0 or name.endswith("_layout"):
                continue
            if seg_end <= t0 or seg_start >= t1:   # 구간 겹침 판정 (시작시각 only 아님)
                continue
            files.extend([f"{name}.hea", f"{name}.dat"])
            segments.append({"name": name, "start": seg_start.isoformat(),
                             "dur_sec": seg_len / fs})

    if not segments:
        return None

    modalities = {MIMIC_SIGNAL_MAP[c] for c in channels if c in MIMIC_SIGNAL_MAP}
    return {
        "record": record_path,
        "start": base.isoformat(),
        "fs": fs,
        "files": [f"{pXX}/{pid}/{fn}" for fn in files],
        "segments": segments,
        "channels": sorted(channels),
        "modalities": sorted(modalities),
    }


def scan_patient(patient: Patient, rec_paths: list[str], landmark_hours: float) -> dict:
    """환자의 모든 레코드를 스캔하고 채널 요건을 적용한다."""
    t0 = patient.icu_intime
    t1 = t0 + timedelta(hours=landmark_hours)

    kept: list[dict] = []
    errors = 0
    for rp in rec_paths:
        try:
            info = scan_record(rp, t0, t1)
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"  header error {rp}: {e}", file=sys.stderr)
            continue
        if info is not None:
            kept.append(info)

    modalities: set[str] = set()
    for info in kept:
        modalities.update(info["modalities"])

    return {"patient": patient, "records": kept,
            "modalities": modalities, "errors": errors}


# ────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pass 1: 원격 헤더 스캔으로 landmark 다운로드 목록 빌드."
    )
    parser.add_argument("--cohort-csv", type=str, required=True,
                        help="02_mortality_landmark_cohort.sql 로 export 한 CSV")
    parser.add_argument("--records-file", type=str,
                        default="downstream/outcome/sepsis/RECORDS-waveforms",
                        help="matched subset 전체 waveform 인덱스")
    parser.add_argument("--out-dir", type=str,
                        default="downstream/outcome/mortality")
    parser.add_argument("--landmark-hours", type=float, default=24.0,
                        help="ICU intime 이후 관측창 길이 (기본 24h)")
    parser.add_argument("--workers", type=int, default=12,
                        help="헤더 스캔 병렬도 (헤더는 KB 단위라 높여도 안전)")
    parser.add_argument("--limit", type=int, default=None,
                        help="앞 N명만 스캔 (테스트용)")
    args = parser.parse_args()

    try:
        import wfdb  # noqa: F401
    except ImportError:
        sys.exit("ERROR: wfdb 필요.  pip install wfdb")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    attr = Attrition()

    print("Loading waveform index...")
    wf_index = load_waveform_index(Path(args.records_file))
    print(f"  {len(wf_index)} subjects with waveforms")

    print("Loading landmark cohort...")
    patients = load_landmark_cohort(Path(args.cohort_csv), attr)
    print(f"  cohort rows           : {attr.n_cohort}")
    print(f"  excl. died in window  : {attr.n_died_in_window}")
    print(f"  excl. insufficient obs: {attr.n_insufficient_obs}")
    print(f"  landmark eligible     : {attr.n_eligible}")

    # waveform 보유 환자만
    with_wf = [p for p in patients if p.subject_id in wf_index]
    attr.n_no_waveform = len(patients) - len(with_wf)
    print(f"  with waveform         : {len(with_wf)} "
          f"(no waveform: {attr.n_no_waveform})")

    if args.limit:
        with_wf = with_wf[: args.limit]
        print(f"  [--limit] {len(with_wf)} 명만 스캔")

    print(f"\n원격 헤더 스캔 ({args.workers} workers, .dat 미다운로드)...")
    final: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(scan_patient, p, wf_index[p.subject_id], args.landmark_hours): p
            for p in with_wf
        }
        for fut in as_completed(futs):
            res = fut.result()
            p: Patient = res["patient"]
            attr.n_header_error += res["errors"]

            if not res["records"]:
                attr.n_no_overlap += 1
            elif not REQUIRED_MODALITIES.issubset(res["modalities"]):
                attr.n_no_required_channel += 1
            else:
                p.records = res["records"]
                attr.n_final += 1
                if IDEAL_MODALITIES.issubset(res["modalities"]):
                    attr.n_final_ideal += 1
                final.append({
                    "subject_id": p.subject_id,
                    "icustay_id": p.icustay_id,
                    "icu_intime": p.icu_intime.isoformat(),
                    "landmark_end": (p.icu_intime
                                     + timedelta(hours=args.landmark_hours)).isoformat(),
                    "mortality": p.label,
                    "mortality_28d": p.label_28d,
                    "modalities": sorted(res["modalities"]),
                    "records": p.records,
                })

            done += 1
            if done % 200 == 0 or done == len(with_wf):
                print(f"  [{done}/{len(with_wf)}] kept={attr.n_final}")

    final.sort(key=lambda d: d["subject_id"])

    # ── 출력 ────────────────────────────────────────────────
    all_files = sorted({f for pt in final for r in pt["records"] for f in r["files"]})
    all_records = sorted({r["record"] for pt in final for r in pt["records"]})

    (out_dir / "FILES-to-download").write_text("\n".join(all_files) + "\n")
    (out_dir / "RECORDS-to-download").write_text("\n".join(all_records) + "\n")

    n_death = sum(pt["mortality"] for pt in final)
    n_death28 = sum(pt["mortality_28d"] for pt in final)
    prevalence = 100.0 * n_death / len(final) if final else 0.0
    prevalence28 = 100.0 * n_death28 / len(final) if final else 0.0
    n_dat = sum(1 for f in all_files if f.endswith(".dat"))

    manifest = {
        "task": "in_hospital_mortality_landmark",
        "landmark_hours": args.landmark_hours,
        "time_zero": "icu_intime",
        "label": "hospital_expire_flag",
        "label_aux": "mortality_28d",
        "required_modalities": sorted(REQUIRED_MODALITIES),
        "ideal_modalities": sorted(IDEAL_MODALITIES),
        "attrition": attr.as_dict(),
        "prevalence_pct": round(prevalence, 2),
        "prevalence_28d_pct": round(prevalence28, 2),
        "n_patients": len(final),
        "n_deaths": n_death,
        "n_records": len(all_records),
        "n_files": len(all_files),
        "n_dat_files": n_dat,
        "patients": final,
    }
    (out_dir / "landmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str)
    )

    print(f"\n{'=' * 62}")
    print("  CONSORT attrition")
    print(f"    성인·첫 ICU stay        : {attr.n_cohort}")
    print(f"    − 관측창 내 사망         : {attr.n_died_in_window}")
    print(f"    − 24h 관측 미충족        : {attr.n_insufficient_obs}")
    print(f"    = landmark 자격          : {attr.n_eligible}")
    print(f"    − waveform 없음          : {attr.n_no_waveform}")
    print(f"    − 창에 겹치는 신호 없음   : {attr.n_no_overlap}")
    print(f"    − ECG 채널 없음          : {attr.n_no_required_channel}")
    print(f"    = 최종 코호트            : {attr.n_final}")
    print(f"        그 중 ECG+ABP+PPG    : {attr.n_final_ideal}")
    if attr.n_header_error:
        print(f"    (헤더 파싱 실패 레코드   : {attr.n_header_error})")
    print(f"\n  Prevalence (in-hospital) : {prevalence:.2f}%  "
          f"({n_death}/{len(final)})")
    print(f"  Prevalence (28-day)      : {prevalence28:.2f}%  "
          f"({n_death28}/{len(final)})")
    print(f"\n  다운로드 대상: {len(all_records)} 레코드 / {len(all_files)} 파일 "
          f"({n_dat} .dat)")
    print(f"  → {out_dir / 'FILES-to-download'}")
    print(f"  → {out_dir / 'landmark_manifest.json'}")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
