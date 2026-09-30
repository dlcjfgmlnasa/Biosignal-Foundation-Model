# -*- coding: utf-8 -*-
"""경험적 커플링 probe (합의된 방법). SNUH/K-MIMIC 공용, shard 기반.

지표 (모두 window(10s) 내 = 환자 내 계산 → median 집계, between-patient 분산에 강건):
  - bestlag_corr : 생리 window(±300ms) 내 |corr| 최대 정렬 상관  ← 주지표(정보 존재 하한)
  - coherence    : magnitude-squared coherence 대역평균 (delay-invariant, 비선형 일부 포함)
  - lag0_corr    : 위상 정렬 없는 co-located 상관        ← phase-sensitivity 병기
  - phase_sens   : bestlag_corr - lag0_corr             ← 위상 가변성(=attention 필요도)
lag 는 per-patch 자유탐색 금지 → window 단위 단일 lag(±300ms), 과적합 회피.
corr 은 대칭이므로 pair 는 undirected. remap_record_v2 로 disk→v2 변환.
수렴 확인: n_win(윈도우 수) 충분(>수천)한 쌍만 신뢰.

실행: KMIMIC_SHARD_DIR=/home/coder/workspace/k-mimic-/bio_fm/data/train/k_mimic_shard_500mb \
      python coupling_probe.py
결과: COPY START~END 블록 + probe_<tag>.npz 저장.
"""
import sys, glob, os, itertools
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import numpy as np, torch
from collections import defaultdict, Counter
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, coherence as _coh

SHARD_DIR = os.environ.get("KMIMIC_SHARD_DIR",
    "/home/coder/workspace/k-mimic-/bio_fm/data/train/k_mimic_shard_500mb")
TAG = os.environ.get("TAG", "kmimic")
SR = 100; PATCH = 200; WIN = 1000
LAGMAX = 30                     # ±300ms 생리 window
NAMES = ["ecg","abp","ppg","cvp","co2","awp","icp","resp_imp","resp_flow"]
NST = 9
MAX_SHARDS = int(os.environ.get("MAX_SHARDS", "80"))
MAX_WIN = int(os.environ.get("MAX_WIN", "6000"))   # pair 당 window 상한(수렴 충분)
ONLY = os.environ.get("ONLY", "")                   # "0-1,1-2" 형태면 그 쌍만(빠른 검증)

# ── remap (SSOT 우선) ──
try:
    sys.path.insert(0, os.getcwd())
    from data.spatial_map import remap_record_v2 as _rm
    def remap_st(st, sp):
        r = _rm(int(st), list(sp) if sp is not None else None); return None if r is None else r[0]
    REMAP_SRC = "SSOT"
except Exception:
    def remap_st(st, sp):
        st = int(st)
        if st == 6: return None
        if st == 8:
            f = int(sp[0]) if sp is not None and len(sp) else 0
            return {1:7,2:8}.get(f,7)
        return {0:0,1:1,2:2,3:3,4:4,5:5,7:6}.get(st, st if 0<=st<NST else None)
    REMAP_SRC = "inline"

bC, aC = butter(2, [0.12/50, 8.0/50], "band")       # 호흡(0.15-0.5)+심장(0.7-3.5) 모두 통과
def bp(x): return filtfilt(bC, aC, np.nan_to_num(x - np.nanmean(x)))
freqs = rfftfreq(WIN, 1/SR); physio = (freqs >= 0.12) & (freqs <= 4.0)
def live(w):
    w = np.asarray(w, float)
    if not np.isfinite(w).all() or w.std() < 1e-6: return False
    P = np.abs(rfft(w - w.mean()))**2; t = P.sum()
    return t > 0 and (P[physio].sum()/t) > 0.15
def best_lag_corr(a, b):        # a,b: bandpassed z-not-needed
    lag0 = np.corrcoef(a, b)[0, 1]
    best = (0, lag0)
    for lg in range(-LAGMAX, LAGMAX+1, 2):
        if lg >= 0: c = np.corrcoef(a[:WIN-lg], b[lg:])[0, 1]
        else:       c = np.corrcoef(a[-lg:], b[:WIN+lg])[0, 1]
        if abs(c) > abs(best[1]): best = (lg, c)
    return lag0, best[1], best[0]
_cohband = None
def coh_band(a, b):
    global _cohband
    f, C = _coh(a, b, fs=SR, nperseg=256)
    if _cohband is None: _cohband = (f >= 0.3) & (f <= 6.0)
    return float(np.nanmean(C[_cohband]))

want_pairs = None
if ONLY:
    want_pairs = set()
    for tok in ONLY.split(","):
        x, y = tok.split("-"); want_pairs.add(tuple(sorted((int(x), int(y)))))

shard_files = sorted(glob.glob(os.path.join(SHARD_DIR, "shard_*.pt")))
if not shard_files:   # 여러 달 하위디렉토리면 recursive
    shard_files = sorted(glob.glob(os.path.join(SHARD_DIR, "**", "shard_*.pt"), recursive=True))
shard_files = shard_files[:MAX_SHARDS]
assert shard_files, f"no shards in {SHARD_DIR}"

acc = defaultdict(lambda: {"lag0": [], "best": [], "coh": []})
raw_ct = Counter(); resp8_sp = Counter(); v2_ct = Counter()
for si, sf in enumerate(shard_files):
    pairs_done = [p for p in acc if len(acc[p]["best"]) >= MAX_WIN]
    if want_pairs and all(len(acc[p]["best"]) >= MAX_WIN for p in want_pairs): break
    try: sh = torch.load(sf, weights_only=False, map_location="cpu")
    except Exception as e: print(f"skip {os.path.basename(sf)}: {e}", flush=True); continue
    grp = defaultdict(lambda: defaultdict(list))
    for k, r in sh.items():
        ost = int(r["signal_type"]); raw_ct[ost] += 1
        sp = r.get("spatial_ids", None); sp = list(sp) if sp is not None else None
        if ost == 8 and sp: resp8_sp[tuple(sorted(set(int(x) for x in sp)))] += 1
        v2 = remap_st(ost, sp)
        if v2 is None or not (0 <= v2 < NST): continue
        v2_ct[v2] += 1; grp[r["subject_id"]][v2].append(r)
    for subj, byst in grp.items():
        for i, j in itertools.combinations(sorted(t for t in byst if byst[t]), 2):
            if want_pairs and (i, j) not in want_pairs: continue
            if len(acc[(i,j)]["best"]) >= MAX_WIN: continue
            for ri in byst[i]:
                for rj in byst[j]:
                    s = max(ri["start_sample"], rj["start_sample"])
                    en = min(ri["start_sample"]+ri["n_timesteps"], rj["start_sample"]+rj["n_timesteps"])
                    if en - s < WIN: continue
                    vi = np.asarray(ri["values"], float).flatten()[s-ri["start_sample"]:en-ri["start_sample"]]
                    vj = np.asarray(rj["values"], float).flatten()[s-rj["start_sample"]:en-rj["start_sample"]]
                    for w in range(0, min(len(vi), len(vj))-WIN, WIN):
                        wi, wj = vi[w:w+WIN], vj[w:w+WIN]
                        if not (live(wi) and live(wj)): continue
                        Ai, Aj = bp(wi), bp(wj)
                        l0, lb, _ = best_lag_corr(Ai, Aj)
                        acc[(i,j)]["lag0"].append(l0); acc[(i,j)]["best"].append(lb)
                        acc[(i,j)]["coh"].append(coh_band(wi, wj))
                    if len(acc[(i,j)]["best"]) >= MAX_WIN: break
                if len(acc[(i,j)]["best"]) >= MAX_WIN: break
    del sh
    print(f"[{si+1}/{len(shard_files)}] {os.path.basename(sf)} conv={len([p for p in acc if len(acc[p]['best'])>=MAX_WIN])}", flush=True)

rows = []
for (i, j), d in acc.items():
    n = len(d["best"])
    if n < 300: continue
    rows.append((i, j, n, float(np.median(np.abs(d["lag0"]))), float(np.median(np.abs(d["best"]))),
                 float(np.median(d["coh"]))))
np.savez(f"probe_{TAG}.npz", rows=np.array(rows, dtype=object))

print("\n########## COPY START ##########")
print(f"# tag={TAG} remap={REMAP_SRC} shards={len(shard_files)}")
print(f"# raw(disk) st: {dict(sorted(raw_ct.items()))}")
print(f"# disk8(RESP) spatial: {dict(resp8_sp)}  v2 st: {dict(sorted(v2_ct.items()))}")
san = next((r for r in rows if (r[0],r[1])==(0,1)), None)
if san: print(f"# [SANITY ecg-abp] bestlag_corr={san[4]:+.3f} lag0={san[3]:+.3f} n={san[2]}  <-강해야 정상")
print(f"# {'pair':<20}{'n_win':>7}{'lag0':>8}{'bestlag':>9}{'coher':>8}{'phase_sens':>11}")
for i,j,n,l0,lb,co in sorted(rows, key=lambda r:-r[4]):
    print(f"# {NAMES[i]+'-'+NAMES[j]:<20}{n:>7}{l0:>8.3f}{lb:>9.3f}{co:>8.3f}{lb-l0:>11.3f}")
# 커플링 weight = bestlag_corr(주지표), max-normalized
mx = max((r[4] for r in rows), default=1.0) or 1.0
print("COUPLING = {  # undirected bestlag_corr, max-normalized")
for i,j,n,l0,lb,co in sorted(rows, key=lambda r:-r[4]):
    if lb/mx >= 0.05:
        print(f"    ({i},{j}): {lb/mx:.3f},  # {NAMES[i]}-{NAMES[j]}  coh={co:.2f} phase_sens={lb-l0:+.2f}")
print("}")
print("########## COPY END ##########")
