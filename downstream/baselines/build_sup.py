# -*- coding: utf-8 -*-
"""과제별 sup_train 입력 빌더 — 평가 산출물(feature npz 등)과 **같은 행 순서**의 창 배열.

``build_sup.py <task> --out <prefix>`` → ``<prefix>_X.npy`` (N, C, L) float16 (없는 채널 NaN) +
``<prefix>_meta.npz`` (y, pid, keys[, fold]). 행 i 는 해당 과제 평가 파일의 행 i 와 같은 창이라,
학습된 OOF 예측을 그 파일에 1차원 feature 로 붙여 CARMEN·MOMENT 와 같은 평가기로 채점한다.
경로는 노드(``$ORCH_HOME``) 산출물 기준.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

H = os.environ.get("ORCH_HOME", "")
F = f"{H}/final"


def save(out, X, y, pid, keys, fold=None):
    np.save(f"{out}_X.npy", X.astype(np.float16))
    extra = {} if fold is None else {"fold": np.asarray(fold)}
    np.savez(f"{out}_meta.npz", y=np.asarray(y, np.float32), pid=np.asarray(pid).astype(str), keys=np.array(keys), **extra)
    print(f"saved {out}: X {X.shape} y mean {np.nanmean(y):.3f} 환자 {len(np.unique(pid))} keys {keys}", flush=True)


STYPE_ID = {"ecg": 0, "abp": 1, "ppg": 2, "cvp": 3, "co2": 4, "awp": 5}  # DumpEncoder 는 signal_type 번호로 저장


def _dump_path(prefix, s):
    p = f"{prefix}_{s}.npz"
    return p if os.path.exists(p) else f"{prefix}_{STYPE_ID[s]}.npz"


def from_dump(feat, stypes, dump_prefix, idx_key="dump_{}_mean"):
    """DumpEncoder 행 번호(feat[idx_key]) → (N, C, L) 창. 없는 행 NaN."""
    bufs = [np.load(_dump_path(dump_prefix, s), allow_pickle=True)["x"] for s in stypes]
    N = len(np.asarray(feat[idx_key.format(stypes[0])]))
    L = max(len(b[0]) for b in bufs)
    X = np.full((N, len(stypes), L), np.nan, np.float16)
    for c, (s, b) in enumerate(zip(stypes, bufs)):
        ix = np.asarray(feat[idx_key.format(s)]).reshape(N, -1)[:, 0]
        ok = np.isfinite(ix)
        for i in np.where(ok)[0]:
            x = b[int(ix[i])]; X[i, c, : len(x)] = x
    return X


def arrhythmia(out):
    import torch
    Xs, ys, ps, fs = [], [], [], []
    for f in range(5):  # fm_arr.py 와 같은 순서 (fold test 순차)
        d = torch.load(f"{H}/xgen/arr_ecgppg/arrhythmia_ecg_ppg_fold{f}.pt", weights_only=False)["test"]
        keys = list(d["signals"].keys())
        Xs.append(torch.stack([d["signals"][k] for k in keys], 1).half().numpy()); ys.append(d["labels"].long().numpy())
        ps.append(np.asarray(d["patients"]).astype(str)); fs.append(np.full(len(ys[-1]), f))
    save(out, np.concatenate(Xs), np.concatenate(ys), np.concatenate(ps), keys, np.concatenate(fs))


def ap(out):
    """파형 MAP r3 순서. prep2 fold test 순차(=r2 순서)로 읽고 (G, 출현순번) 키로 r3 재정렬."""
    import collections
    from downstream._save_utils import load_prepared_split_chunked
    B = f"{H}/xgen/bpmap"
    Xs = []
    for f in range(5):
        t = load_prepared_split_chunked(f"{B}/prep2/bp_ecg_ppg_w30s", fold=f, splits=("test",))["test"]
        Xs.append(t["signals"]["ppg"].numpy().astype(np.float16))
    X2 = np.concatenate(Xs)
    a = np.load(f"{B}/r3/feat_fmbpmap/_meta.npz", allow_pickle=True); b = np.load(f"{B}/r2/feat_fmbpmap/_meta.npz", allow_pickle=True)
    assert len(X2) == len(b["G"])

    def keys(G):
        k = collections.Counter(); o = []
        for g in G:
            o.append((g, k[g])); k[g] += 1
        return o
    kb = {k: i for i, k in enumerate(keys(b["G"]))}; idx = np.array([kb[k] for k in keys(a["G"])])
    save(out, X2[idx][:, None, :], a["Y"][:, 2], a["G"], ["ppg"], a["FO"])


def etco2(out):
    z = np.load(f"{F}/etco2/feat.npz", allow_pickle=True)
    save(out, from_dump(z, ["co2", "awp"], f"{F}/etco2/dump"), z["lab3"], z["pid"], ["co2", "awp"])


def co(out):
    z = np.load(f"{F}/co/feat_A2_co_pap.npz", allow_pickle=True)
    X = from_dump({"dump_abp_mean": z["dump_mean"]}, ["abp"], f"{F}/co/dump")  # PAP 는 재학습 전 ABP 타입(1) 으로 덤프됨
    save(out, X, (np.asarray(z["value"]) < 4.0).astype(float), z["pid"], ["pap"])


def af(out):
    d = np.load(f"{F}/af/dump.npz", allow_pickle=True); z = np.load(f"{H}/xgen/re23_af/af_feat.npz", allow_pickle=True)
    assert (d["pid"] == z["pid"]).all(), "AF 덤프 순서가 af_feat 와 다름"
    save(out, d["x"][:, None], z["y"], z["pid"], ["ppg"])


def svv(out):
    z = np.load(f"{F}/svv/svv_feat.npz", allow_pickle=True)
    save(out, np.load(f"{F}/svv/dump_ppg.npy")[:, None], z["svv"], z["pid"], ["ppg"])


def ppv(out):
    z = np.load(f"{F}/ppv/ppv_feat_pvi_svv.npz", allow_pickle=True); z0 = np.load(f"{F}/ppv/ppv_feat.npz", allow_pickle=True)
    assert (z["pid"] == z0["pid"]).all() and np.allclose(z["win_start"], z0["win_start"]), "PVI 부착 후 행 순서 변경"
    save(out, np.load(f"{F}/ppv/dump_ppg.npy")[:, None], z["ppv"], z["pid"], ["ppg"])


def rr(out):
    import torch
    Z = torch.load(f"{H}/ext/m3probe/rr_feat_moment.pt", weights_only=False)
    X = np.load(f"{H}/ext/m3probe/rr_dump_ecgppg.npy")
    save(out, X, np.asarray(Z["y"]), np.asarray(Z["sid"]), ["ecg", "ppg"])


def age(out):
    O = f"{H}/xgen/re23_aurora_moment"; z = np.load(f"{O}/aurora_feat.npz", allow_pickle=True)
    idx = {f"dump_{st}_mean": z[f"dump_{k}_mean"] for st, k in (("ecg", "ecg"), ("ppg", "ppg"), ("abp", "tono"))}  # 덤프 파일은 stype 이름
    save(out, from_dump(idx, ["ecg", "ppg", "abp"], f"{O}/dump"), z["age"], z["pid"], ["ecg", "ppg", "abp"])


def sex(out):
    O = f"{H}/xgen/re23_vdb_moment"; z = np.load(f"{O}/feat.npz", allow_pickle=True)
    sx = {int(r["caseid"]): r["sex"] for r in csv.DictReader(open(f"{H}/aki/csv/clinical_data.csv", encoding="utf-8-sig"))}
    cid = np.asarray(z["cid"])
    y = np.array([{"m": 1.0, "f": 0.0}.get(str(sx.get(int(c), "")).strip().lower(), np.nan) for c in cid])
    X = from_dump({f"dump_{m}_mean": z[f"dumpidx_{m}"] for m in ("ecg", "abp", "ppg")}, ["ecg", "abp", "ppg"], f"{O}/dump")
    save(out, X, y, cid, ["ecg", "abp", "ppg"])


def sleep(out):
    D = f"{H}/ext/challenge2018"; recs = sorted(glob.glob(f"{D}/feat_moment/*.npz"))
    Xs, ys, ps = [], [], []
    for r in recs:  # c18_assemble 과 같은 순서(파일명 정렬 · 레코드 내 창 순서)
        z = np.load(r, allow_pickle=True); w = np.load(f"{D}/win_dump/{os.path.basename(r)}")
        Xs.append(np.stack([w[k] for k in ("x_ecg", "x_air", "x_chest", "x_abd")], 1)); ys.append(z["ev"]); ps += [os.path.basename(r)[:-4]] * len(z["ev"])
    save(out, np.concatenate(Xs), np.concatenate(ys), ps, ["ecg", "air", "chest", "abd"])


def outcome(task, out, channels=("ecg", "ppg", "abp", "resp")):
    D = f"{H}/ext/m3prep/{task}"; M = np.load(f"{D}/meta.npz", allow_pickle=True); CH = list(M["channels"])
    N = len(M["pid"]); L = np.load(f"{D}/{channels[0]}.npy", mmap_mode="r").shape[1]
    X = np.lib.format.open_memmap(f"{out}_X.npy", mode="w+", dtype=np.float16, shape=(N, len(channels), L))
    for c, ch in enumerate(channels):
        A = np.load(f"{D}/{ch}.npy", mmap_mode="r"); pres = M["present"][:, CH.index(ch)]
        for s in range(0, N, 4096):
            blk = np.array(A[s:s + 4096], dtype=np.float16); blk[~pres[s:s + 4096]] = np.nan; X[s:s + 4096, c] = blk
    X.flush()
    y = M["label"].astype(np.float32)
    keep = f"{D}/keep_72h.csv" if task == "cardiac_arrest" else None
    if keep and os.path.exists(keep):  # 평가 코호트 밖 환자는 학습에서 제외(라벨 NaN) — CARMEN probe 와 같은 학습 모집단
        ks = {int(r[next(iter(r))]) for r in csv.DictReader(open(keep))}
        y[~np.isin(M["pid"], list(ks))] = np.nan
        print(f"keep 코호트 {len(ks)}명 · 학습 창 {np.isfinite(y).sum()}", flush=True)
    np.savez(f"{out}_meta.npz", y=y, pid=M["pid"].astype(str), keys=np.array(channels), fold=M["fold"])
    print(f"saved {out}: X {X.shape} 환자 {len(np.unique(M['pid']))}", flush=True)


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("task"); ap_.add_argument("--out", required=True); a = ap_.parse_args()
    fn = {"arrhythmia": arrhythmia, "ap": ap, "etco2": etco2, "co": co, "af": af, "svv": svv, "ppv": ppv, "rr": rr,
          "age": age, "sex": sex, "sleep": sleep}
    if a.task in fn:
        fn[a.task](a.out)
    else:
        outcome(a.task, a.out)
