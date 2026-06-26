# -*- coding:utf-8 -*-
"""사전학습 코퍼스(K-MIMIC-MORTAL) 특성화 그림용 통계 집계.

Tongren-style "dataset characterization" 패널(환자수·recording수·모달리티별 바·
연령 분포·신호 파라미터 표·길이 분포 도넛)에 들어갈 모든 수치를 한 번에 뽑아
``figure_corpus_stats.json`` 으로 저장한다. ``--plot`` 시 matplotlib 멀티패널 PNG 도 렌더.

두 소스를 결합한다:
  (A) 파싱된 manifest_full.jsonl  → 모달리티별 recs/subjects/tokens/hours, 총량,
      환자수(=학습이 실제로 본 코호트), recording 길이(분) 분포.
      remap_record_v2 적용(PAP drop · RESP Imp/Flow 분기 · ICP 7→6) → 학습 파이프라인과 동일.
  (B) raw EMR CSV (patients.csv / admissions.csv, 파드에 read-only 마운트)
      → 연령·성별·ICU LOS·사망 플래그. 가능하면 파형 코호트로 한정, 아니면 전체 deceased 코호트.

사용 (서버 파드에서):
    python -m scripts.figure_corpus_stats \\
        --data-dir /home/coder/workspace/updown/bio_fm/data/train/k_mimic_full_v2 \\
        --emr-dir  datasets/K-MIMIC-MORTAL/1.0.0/EMR \\
        --plot

EMR 이 없거나 경로 모르면 --emr-dir 생략 → (A) 만 집계.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.spatial_map import SIGNAL_TYPE_NAMES, remap_record_v2  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_PATCH = 200
DEFAULT_SR = 100.0

# (A) recording-length 도넛 bin (분) — 참조 그림의 "Spacing Between Slices" 대응
LEN_BINS_MIN = [(0, 5), (5, 15), (15, 60), (60, 1e9)]
LEN_BIN_LABELS = ["<5", "5-15", "15-60", ">60"]
# 연령 도넛/히스토 bin
AGE_BINS = [(0, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 200)]
AGE_BIN_LABELS = ["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]


def _iter_records(sm: dict):
    """subject manifest → recording dict (K-MIMIC 중첩/flat 둘 다)."""
    sessions = sm.get("sessions")
    if isinstance(sessions, list) and sessions:
        for sess in sessions:
            if isinstance(sess, dict):
                for r in sess.get("recordings", []):
                    if isinstance(r, dict):
                        yield r
        return
    recs = sm.get("recordings") or sm.get("files", [])
    for r in recs:
        if isinstance(r, dict):
            yield r


def scan_manifest(data_dir: Path, patch: int, sr: float) -> dict:
    full = data_dir / "manifest_full.jsonl"
    if not full.exists():
        sys.exit(f"[ERROR] {full} 없음")

    st_recs: Counter = Counter()
    st_subj: dict[int, set] = defaultdict(set)
    st_samp: dict[int, int] = defaultdict(int)
    len_hist = Counter()              # recording 길이(분) bin → count
    sessions_total = 0
    recs_raw = pap_drop = 0
    total_samp = 0
    subjects: set[str] = set()
    wave_subject_ids: set[str] = set()  # 파형 코호트 subject_id (EMR join 용)

    with open(full, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                sm = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = sm.get("subject_id") or ""
            subjects.add(sid)
            wave_subject_ids.add(sid)
            sess = sm.get("sessions")
            if isinstance(sess, list):
                sessions_total += len(sess)

            for r in _iter_records(sm):
                recs_raw += 1
                old_st = r.get("signal_type")
                sp = r.get("spatial_ids") or []
                n_t = r.get("n_timesteps", r.get("length", 0))
                n_ch = r.get("n_channels", len(sp) if sp else 1)
                if not isinstance(n_t, int) or n_t <= 0:
                    continue
                remapped = remap_record_v2(old_st, [int(s) for s in sp] if sp else [0] * n_ch)
                if remapped is None:
                    pap_drop += 1
                    continue
                new_st, new_sp = remapped
                if new_st not in SIGNAL_TYPE_NAMES:
                    continue
                eff_ch = len(new_sp) if new_sp else (n_ch or 1)
                samp = n_t * eff_ch
                st_recs[new_st] += 1
                st_subj[new_st].add(sid)
                st_samp[new_st] += samp
                total_samp += samp
                # 길이(분): 채널당 timestep 기준
                minutes = n_t / sr / 60.0
                for (lo, hi), lab in zip(LEN_BINS_MIN, LEN_BIN_LABELS):
                    if lo <= minutes < hi:
                        len_hist[lab] += 1
                        break

    per_mod = []
    for st in sorted(st_recs):
        samp = st_samp[st]
        per_mod.append({
            "signal_type": st,
            "name": SIGNAL_TYPE_NAMES[st],
            "recordings": st_recs[st],
            "subjects": len(st_subj[st]),
            "hours": round(samp / sr / 3600, 1),
            "tokens": samp // patch,
        })
    return {
        "n_patients": len(subjects),
        "n_sessions": sessions_total,
        "n_recordings_raw": recs_raw,
        "pap_dropped": pap_drop,
        "total_samples": total_samp,
        "total_tokens": total_samp // patch,
        "total_hours": round(total_samp / sr / 3600, 1),
        "per_modality": per_mod,
        "recording_len_min_hist": {l: len_hist.get(l, 0) for l in LEN_BIN_LABELS},
        "_wave_subject_ids": wave_subject_ids,
        "patch_size": patch,
        "sampling_rate": sr,
    }


def _find_col(header: list[str], cands: list[str]) -> str | None:
    low = {h.lower(): h for h in header}
    for c in cands:
        if c in low:
            return low[c]
    return None


def _norm_id(s: str) -> str:
    """'VDB_1603' / '1603' / '1603.0' → '1603' (join 키 표준화)."""
    m = re.search(r"(\d+)", str(s))
    return str(int(m.group(1))) if m else str(s)


def scan_emr(emr_dir: Path, wave_ids: set[str]) -> dict:
    out: dict = {"available": False}
    pat = emr_dir / "patients.csv"
    adm = emr_dir / "admissions.csv"
    if not pat.exists():
        out["note"] = f"{pat} 없음 — EMR 집계 생략"
        return out

    wave_norm = {_norm_id(x) for x in wave_ids}
    ages: dict[str, float] = {}      # subject_id(norm) → age
    genders: dict[str, str] = {}
    with open(pat, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        hdr = rd.fieldnames or []
        id_c = _find_col(hdr, ["subject_id", "patient_id", "pid"])
        age_c = _find_col(hdr, ["anchor_age", "age", "admission_age"])
        sex_c = _find_col(hdr, ["gender", "sex"])
        out["patients_header"] = hdr
        for row in rd:
            sid = _norm_id(row.get(id_c, "")) if id_c else ""
            if age_c:
                try:
                    ages[sid] = float(row[age_c])
                except (ValueError, TypeError, KeyError):
                    pass
            if sex_c:
                genders[sid] = (row.get(sex_c) or "").strip()

    def _age_stats(idset: set[str] | None) -> dict | None:
        vals = [a for s, a in ages.items() if (idset is None or s in idset)]
        vals = [a for a in vals if a and a > 0]
        if not vals:
            return None
        vals.sort()
        n = len(vals)
        hist = {l: 0 for l in AGE_BIN_LABELS}
        for a in vals:
            for (lo, hi), lab in zip(AGE_BINS, AGE_BIN_LABELS):
                if lo <= a < hi:
                    hist[lab] += 1
                    break
        return {
            "n": n,
            "mean": round(sum(vals) / n, 1),
            "median": vals[n // 2],
            "min": vals[0],
            "max": vals[-1],
            "hist": hist,
        }

    matched = wave_norm & set(ages.keys())
    out["available"] = True
    out["n_emr_patients"] = len(ages)
    out["n_wave_matched_emr"] = len(matched)
    out["age_all_deceased"] = _age_stats(None)
    out["age_wave_cohort"] = _age_stats(matched) if matched else None
    if genders:
        gw = Counter(genders[s] for s in (matched or set(ages)) if s in genders)
        out["gender_wave_cohort" if matched else "gender_all"] = dict(gw)

    # admissions: 사망 플래그(검증용)
    if adm.exists():
        with open(adm, encoding="utf-8", newline="") as fh:
            rd = csv.DictReader(fh)
            hdr = rd.fieldnames or []
            exp_c = _find_col(hdr, ["hospital_expire_flag", "death_flag"])
            n_adm = 0
            n_exp = 0
            subj = set()
            id_c = _find_col(hdr, ["subject_id", "patient_id"])
            for row in rd:
                n_adm += 1
                if id_c:
                    subj.add(_norm_id(row.get(id_c, "")))
                if exp_c and _safe01(row.get(exp_c)):
                    n_exp += 1
            out["admissions"] = {
                "n_admissions": n_adm, "n_subjects": len(subj),
                "n_expired_adm": n_exp,
                "expired_rate": round(n_exp / n_adm, 3) if n_adm else None,
            }
    return out


def _safe01(v) -> int:
    try:
        return 1 if int(float(v)) != 0 else 0
    except (ValueError, TypeError):
        return 0


def make_plot(stats: dict, emr: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("K-MIMIC-MORTAL (SNUH) — Pretraining Corpus", fontsize=16, weight="bold")

    # 1) 모달리티별 recordings 바
    ax1 = fig.add_subplot(2, 3, 1)
    mods = stats["per_modality"]
    ax1.bar([m["name"] for m in mods], [m["recordings"] for m in mods], color="#7fb3d5")
    ax1.set_title("Recordings per modality")
    ax1.tick_params(axis="x", rotation=60)

    # 2) 모달리티별 tokens 바
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.bar([m["name"] for m in mods], [m["tokens"] / 1e6 for m in mods], color="#82c99a")
    ax2.set_title("Tokens per modality (M)")
    ax2.tick_params(axis="x", rotation=60)

    # 3) 연령 분포
    ax3 = fig.add_subplot(2, 3, 3)
    age = (emr.get("age_wave_cohort") or emr.get("age_all_deceased")) if emr.get("available") else None
    if age:
        ax3.bar(list(age["hist"].keys()), list(age["hist"].values()), color="#e8a0b8")
        ax3.set_title(f"Age (mean {age['mean']}, median {age['median']})")
        ax3.tick_params(axis="x", rotation=45)
    else:
        ax3.text(0.5, 0.5, "Age N/A\n(EMR 미제공)", ha="center", va="center")
        ax3.set_title("Age Distribution")

    # 4) recording 길이 도넛
    ax4 = fig.add_subplot(2, 3, 4)
    lh = stats["recording_len_min_hist"]
    ax4.pie(list(lh.values()), labels=[f"{k}m" for k in lh], autopct="%1.1f%%")
    ax4.set_title("Recording length (min)")

    # 5) 모달리티별 시간(hours) 도넛
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.pie([m["hours"] for m in mods], labels=[m["name"] for m in mods], autopct="%1.0f%%")
    ax5.set_title("Hours per modality")

    # 6) 요약 텍스트
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    lines = [
        f"Patients : {stats['n_patients']:,}",
        f"Sessions : {stats['n_sessions']:,}",
        f"Recordings : {stats['n_recordings_raw']:,}",
        f"Total tokens : {stats['total_tokens'] / 1e9:.3f} B",
        f"Total hours : {stats['total_hours']:,.0f}",
        f"Sampling : {stats['sampling_rate']:g} Hz | patch {stats['patch_size']} "
        f"({stats['patch_size'] / stats['sampling_rate']:.0f}s/token)",
        f"Modalities : {len(mods)}",
    ]
    if emr.get("admissions"):
        a = emr["admissions"]
        lines.append(f"Admissions : {a['n_admissions']:,} "
                     f"(expired {a['expired_rate']:.0%})")
    ax6.text(0.02, 0.95, "\n".join(lines), va="top", fontsize=12, family="monospace")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    print(f"[plot] saved → {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="manifest_full.jsonl 있는 파싱 디렉토리")
    ap.add_argument("--emr-dir", type=Path, default=None,
                    help="patients.csv/admissions.csv 있는 EMR 디렉토리 (없으면 생략)")
    ap.add_argument("--patch-size", type=int, default=DEFAULT_PATCH)
    ap.add_argument("--sampling-rate", type=float, default=DEFAULT_SR)
    ap.add_argument("--out", type=Path, default=Path("figure_corpus_stats.json"))
    ap.add_argument("--plot", action="store_true", help="멀티패널 PNG 도 렌더")
    args = ap.parse_args()

    print("[1/2] manifest 스캔...")
    stats = scan_manifest(args.data_dir, args.patch_size, args.sampling_rate)
    wave_ids = stats.pop("_wave_subject_ids")

    emr = {"available": False}
    if args.emr_dir:
        print("[2/2] EMR 스캔...")
        emr = scan_emr(args.emr_dir, wave_ids)

    # ---- 텍스트 리포트 ----
    print("\n" + "=" * 64)
    print("  K-MIMIC-MORTAL 코퍼스 통계 (그림용)")
    print("=" * 64)
    print(f"  Patients      : {stats['n_patients']:,}")
    print(f"  Sessions      : {stats['n_sessions']:,}")
    print(f"  Recordings    : {stats['n_recordings_raw']:,}  (PAP dropped {stats['pap_dropped']:,})")
    print(f"  Total tokens  : {stats['total_tokens']:,} ({stats['total_tokens']/1e9:.3f}B)")
    print(f"  Total hours   : {stats['total_hours']:,.0f}")
    print(f"\n  {'mod':<16}{'recs':>12}{'subj':>8}{'hours':>11}{'tokens':>14}")
    for m in stats["per_modality"]:
        print(f"  {m['name']:<16}{m['recordings']:>12,}{m['subjects']:>8,}"
              f"{m['hours']:>11,.1f}{m['tokens']:>14,}")
    if emr.get("available"):
        age = emr.get("age_wave_cohort") or emr.get("age_all_deceased")
        scope = "wave cohort" if emr.get("age_wave_cohort") else "all deceased"
        print(f"\n  EMR matched (wave∩EMR): {emr.get('n_wave_matched_emr')}/{stats['n_patients']}")
        if age:
            print(f"  Age [{scope}] n={age['n']:,} mean={age['mean']} "
                  f"median={age['median']} min={age['min']} max={age['max']}")
            print(f"    hist: {age['hist']}")
    else:
        print(f"\n  [EMR] {emr.get('note', '미집계 (--emr-dir 미지정)')}")

    # ---- JSON 저장 ----
    dump = {**stats, "emr": emr}
    args.out.write_text(json.dumps(dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] saved → {args.out}")

    if args.plot:
        make_plot(stats, emr, args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
