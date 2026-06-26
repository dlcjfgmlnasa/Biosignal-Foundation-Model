# -*- coding:utf-8 -*-
"""K-MIMIC-MORTAL 사전학습 코퍼스 특성화 그림 (Tongren-style 패널).

``scripts/figure_corpus_stats.py`` 가 뽑은 ``figure_corpus_stats.json`` 을 입력으로
받아 참조 그림과 동일한 레이아웃의 멀티패널 PNG 를 렌더한다. JSON 이 없으면
메모리 실측치(2026-05-25 파싱 완료분) FALLBACK 으로 로컬 미리보기 가능.

레이아웃 (참조 Tongren (c) 패널 대응):
  상단: [모달리티 hero] [Patients] [Sessions/Recordings] [Age Distribution(2행 span)]
  중단: [Multi-modal Waveform 예시(2칸 span)]      [모달리티별 막대]
  하단: [Recording length 도넛] [Modality hours 도넛] [Acquisition Parameters 표(2칸)]

사용:
    # 서버에서 figure_corpus_stats.py 돌려 JSON 받은 뒤
    python -m scripts.plot_corpus_figure --stats figure_corpus_stats.json --out corpus_figure.png
    # JSON 없이 메모리 실측치로 미리보기
    python -m scripts.plot_corpus_figure --out corpus_figure_preview.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, FancyBboxPatch

# ── 모달리티 고정 색/순서 (9축) ───────────────────────────────
MOD_ORDER = ["ECG", "ABP", "PPG", "CVP", "CO2", "AWP", "ICP",
             "RESP_Impedance", "RESP_Flow"]
MOD_COLORS = {
    "ECG": "#e15759", "ABP": "#f28e2b", "PPG": "#4e79a7", "CVP": "#76b7b2",
    "CO2": "#59a14f", "AWP": "#edc948", "ICP": "#b07aa1",
    "RESP_Impedance": "#9c755f", "RESP_Flow": "#ff9da7",
}
CARDIO = {"ECG", "ABP", "PPG", "CVP", "ICP"}   # AWP/CO2/RESP = respiratory side

# ── FALLBACK: 메모리 실측치 (verify_manifest_full 2026-05-25) ──
#   hours 기준이 SOT, tokens = hours * 3600 * SR / patch = hours * 1800 (@100Hz, patch200)
FALLBACK = {
    "n_patients": 725,
    "n_sessions": 194_148,
    "n_recordings_raw": 4_612_049,
    "pap_dropped": 4_485,
    "total_tokens": 1_203_186_657,
    "total_hours": 668_437,
    "patch_size": 200,
    "sampling_rate": 100.0,
    "per_modality": [
        # name, recordings, subjects, hours
        ("ECG", 740_000, 713, 160_000),
        ("ABP", 1_200_000, 675, 158_000),
        ("PPG", 1_680_000, 707, 163_000),
        ("CVP", 197_000, 195, 17_700),
        ("CO2", 108_000, 131, 30_300),
        ("AWP", 238_000, 247, 49_300),
        ("ICP", 43_000, 23, 1_870),
        ("RESP_Impedance", 236_000, 351, 37_800),
        ("RESP_Flow", 168_000, 248, 50_900),
    ],
    # recording-length 분포는 매니페스트 실측 필요 — 미보유 시 illustrative
    "recording_len_min_hist": None,
    "emr": {"available": False},
}


def load_stats(path: Path | None) -> tuple[dict, bool]:
    """JSON 로드 → 표준 dict. 없으면 FALLBACK. (dict, is_real) 반환."""
    if path and path.exists():
        d = json.loads(path.read_text(encoding="utf-8"))
        per = [(m["name"], m["recordings"], m["subjects"], m["hours"])
               for m in d["per_modality"]]
        return ({
            "n_patients": d["n_patients"], "n_sessions": d["n_sessions"],
            "n_recordings_raw": d["n_recordings_raw"],
            "pap_dropped": d.get("pap_dropped", 0),
            "total_tokens": d["total_tokens"], "total_hours": d["total_hours"],
            "patch_size": d["patch_size"], "sampling_rate": d["sampling_rate"],
            "per_modality": per,
            "recording_len_min_hist": d.get("recording_len_min_hist"),
            "emr": d.get("emr", {"available": False}),
        }, True)
    return FALLBACK, False


# ── 작은 시각요소 헬퍼 ────────────────────────────────────────
def draw_person(ax, x, y, s, color):
    ax.add_patch(Circle((x, y + 0.55 * s), 0.22 * s, color=color, zorder=3))
    ax.add_patch(Wedge((x, y), 0.42 * s, 20, 160, color=color, zorder=3))


def synth_wave(kind: str, n=600):
    t = np.linspace(0, 4 * np.pi, n)
    if kind == "ECG":
        y = np.zeros(n)
        for c in np.linspace(0, n, 6, endpoint=False):
            idx = int(c)
            y += 1.0 * np.exp(-((np.arange(n) - idx) ** 2) / 6)       # R
            y -= 0.18 * np.exp(-((np.arange(n) - idx + 12) ** 2) / 8)  # Q
            y -= 0.22 * np.exp(-((np.arange(n) - idx - 12) ** 2) / 8)  # S
            y += 0.12 * np.exp(-((np.arange(n) - idx - 45) ** 2) / 120)  # T
        return y
    if kind == "ABP":
        base = 0.5 + 0.45 * np.maximum(0, np.sin(t))
        dicrotic = 0.08 * np.maximum(0, np.sin(t - 0.6)) ** 4
        return base + dicrotic
    if kind == "PPG":
        return 0.5 + 0.4 * (np.maximum(0, np.sin(t)) ** 1.5)
    if kind == "CO2":  # capnography: 사각에 가까운 plateau
        s = np.sin(t)
        return 0.5 + 0.4 * np.tanh(3 * s)
    return np.sin(t)


# ── 메인 렌더 ────────────────────────────────────────────────
def render(st: dict, is_real: bool, out: Path) -> None:
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.05],
                          hspace=0.45, wspace=0.32,
                          left=0.04, right=0.97, top=0.90, bottom=0.05)
    tag = "" if is_real else "  [PREVIEW — fallback numbers]"
    fig.suptitle("(c) K-MIMIC-MORTAL  (Pretraining Corpus, SNUH)" + tag,
                 fontsize=18, weight="bold", x=0.04, ha="left")

    mods = st["per_modality"]
    name2hours = {m[0]: m[3] for m in mods}
    name2recs = {m[0]: m[1] for m in mods}
    mods_sorted = [m for nm in MOD_ORDER for m in mods if m[0] == nm]

    # (1) 모달리티 hero — 3 trace 미니 파형
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_title("Multi-modal ICU Waveforms", fontsize=12, weight="bold")
    for i, k in enumerate(["ECG", "ABP", "PPG"]):
        y = synth_wave(k)
        ax.plot(np.linspace(0, 1, len(y)), 0.66 - 0.30 * i + 0.10 * (y - y.mean()),
                color=MOD_COLORS[k], lw=1.4)
        ax.text(-0.02, 0.66 - 0.30 * i, k, fontsize=9, ha="right", va="center",
                color=MOD_COLORS[k], weight="bold")
    ax.set_xlim(-0.18, 1.02); ax.set_ylim(0, 1)

    # (2) Patients
    ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
    ax.set_title("Patients", fontsize=12, weight="bold")
    for i, x in enumerate([0.30, 0.5, 0.70]):
        draw_person(ax, x, 0.50, 0.34, "#e8a0b8")
    ax.text(0.5, 0.16, f"{st['n_patients']:,}", fontsize=26, weight="bold",
            ha="center", va="center")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # (3) Sessions / Recordings
    ax = fig.add_subplot(gs[0, 2]); ax.axis("off")
    ax.set_title("Recordings", fontsize=12, weight="bold")
    ax.text(0.5, 0.62, f"{st['n_sessions']:,}", fontsize=24, weight="bold",
            ha="center", va="center", color="#4e79a7")
    ax.text(0.5, 0.40, "sessions", fontsize=11, ha="center", color="#4e79a7")
    ax.text(0.5, 0.18, f"{st['n_recordings_raw']/1e6:.1f}M raw recordings",
            fontsize=11, ha="center", color="#666")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # (4) Age Distribution — 2행 span, 수평 히스토
    ax = fig.add_subplot(gs[0:2, 3])
    ax.set_title("Age Distribution", fontsize=13, weight="bold")
    emr = st.get("emr", {})
    age = (emr.get("age_wave_cohort") or emr.get("age_all_deceased")) if emr.get("available") else None
    if age:
        labels = list(age["hist"].keys())
        vals = list(age["hist"].values())
        ax.barh(range(len(labels)), vals, color="#e8a0b8")
        ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
        box = (f"Total: {age['n']:,}\nMean Age: {age['mean']}\n"
               f"Median Age: {age['median']}\nMin Age: {age['min']}\n"
               f"Max Age: {age['max']}")
        ax.text(0.97, 0.97, box, transform=ax.transAxes, va="top", ha="right",
                fontsize=11, bbox=dict(boxstyle="round", fc="white", ec="#333"))
        ax.set_xlabel("Patients")
    else:
        ax.barh(range(7), [3, 8, 14, 16, 13, 9, 5], color="#eed6df")  # illustrative
        ax.set_yticks(range(7))
        ax.set_yticklabels(["<30", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"])
        ax.text(0.5, 0.5, "EMR pending\n(run figure_corpus_stats\n--emr-dir on pod)",
                transform=ax.transAxes, ha="center", va="center", fontsize=12,
                color="#b00", weight="bold")
        ax.set_xlabel("Patients (illustrative)")

    # (5) Multi-modal waveform 예시 — 2칸 span
    ax = fig.add_subplot(gs[1, 0:2]); ax.axis("off")
    ax.set_title("Multi-modal Waveform Examples", fontsize=12, weight="bold", loc="left")
    show = ["ECG", "ABP", "PPG", "CO2"]
    for i, k in enumerate(show):
        y = synth_wave(k)
        off = 3 - i
        ax.plot(np.linspace(0, 1, len(y)), off + 0.42 * (y - y.mean()) / (np.ptp(y) + 1e-9),
                color=MOD_COLORS[k], lw=1.3)
        ax.text(-0.01, off, k, fontsize=10, ha="right", va="center",
                color=MOD_COLORS[k], weight="bold")
    ax.set_xlim(-0.12, 1.02); ax.set_ylim(-0.5, 3.6)
    ax.text(1.0, -0.4, "100 Hz · illustrative traces", fontsize=8, ha="right", color="#888")

    # (6) 모달리티별 막대 (recordings)
    ax = fig.add_subplot(gs[1, 2])
    ax.set_title("Recordings per modality", fontsize=12, weight="bold")
    names = [m[0] for m in mods_sorted]
    recs = [m[1] / 1e3 for m in mods_sorted]
    ax.bar(range(len(names)), recs, color=[MOD_COLORS[n] for n in names])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("RESP_", "R-") for n in names], rotation=55,
                       ha="right", fontsize=8)
    ax.set_ylabel("×10³ recordings")

    # (7) Recording length 도넛
    ax = fig.add_subplot(gs[2, 0])
    ax.set_title("Recording length (min)", fontsize=12, weight="bold")
    lh = st.get("recording_len_min_hist")
    if lh:
        labels = list(lh.keys()); vals = list(lh.values())
        note = ""
    else:
        labels = ["<5", "5-15", "15-60", ">60"]; vals = [22, 31, 30, 17]
        note = " (illustrative)"
    ax.pie(vals, labels=[f"{l}m" for l in labels], autopct="%1.0f%%",
           colors=["#4e79a7", "#f28e2b", "#59a14f", "#e8a0b8"],
           wedgeprops=dict(width=0.42), textprops=dict(fontsize=9))
    ax.text(0.5, -0.08, note, transform=ax.transAxes, ha="center", fontsize=8, color="#888")

    # (8) Modality hours 도넛
    ax = fig.add_subplot(gs[2, 1])
    ax.set_title("Hours per modality", fontsize=12, weight="bold")
    hv = [name2hours[n] for n in names]
    ax.pie(hv, labels=[n.replace("RESP_", "R-") for n in names],
           colors=[MOD_COLORS[n] for n in names],
           wedgeprops=dict(width=0.42), textprops=dict(fontsize=7))

    # (9) Acquisition Parameters 표 — 2칸 span
    ax = fig.add_subplot(gs[2, 2:4]); ax.axis("off")
    ax.set_title("Signal Acquisition Parameters", fontsize=12, weight="bold")
    spp = st["n_sessions"] / max(st["n_patients"], 1)
    hpp = st["total_hours"] / max(st["n_patients"], 1)
    cardio_h = sum(h for n, h in name2hours.items() if n in CARDIO)
    respi_h = sum(h for n, h in name2hours.items() if n not in CARDIO)
    rows = [
        ("Sampling rate", f"{st['sampling_rate']:g} Hz"),
        ("Tokenization", f"patch {st['patch_size']} "
                         f"({st['patch_size']/st['sampling_rate']:.0f} s / token)"),
        ("Modalities", f"{len(mods)}  (cardio {len(CARDIO & set(names))} / respi {len(set(names)-CARDIO)})"),
        ("Total tokens", f"{st['total_tokens']/1e9:.3f} B"),
        ("Total duration", f"{st['total_hours']:,.0f} h  (~{st['total_hours']/24/365:.0f} patient-yr)"),
        ("Sessions / patient", f"{spp:,.0f}  (mean)"),
        ("Hours / patient", f"{hpp:,.0f}  (~{hpp/24:.0f} d, mean)"),
        ("Cardio : Respi (h)", f"{cardio_h/(cardio_h+respi_h)*100:.0f} : "
                               f"{respi_h/(cardio_h+respi_h)*100:.0f}"),
    ]
    tbl = ax.table(cellText=rows, colLabels=["Parameter", "Value"],
                   cellLoc="left", colLoc="left", loc="center",
                   colWidths=[0.42, 0.58])
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.45)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#ddd")
        if r == 0:
            cell.set_facecolor("#e8a0b8"); cell.set_text_props(weight="bold")
        elif r % 2:
            cell.set_facecolor("#fbeef3")

    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[saved] {out}  (source={'JSON' if is_real else 'FALLBACK preview'})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stats", type=Path, default=None,
                    help="figure_corpus_stats.json (없으면 메모리 실측 FALLBACK)")
    ap.add_argument("--out", type=Path, default=Path("corpus_figure.png"))
    args = ap.parse_args()
    st, is_real = load_stats(args.stats)
    render(st, is_real, args.out)


if __name__ == "__main__":
    main()
