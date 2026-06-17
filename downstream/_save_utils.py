# -*- coding:utf-8 -*-
"""Downstream prepare_data 공용 저장 헬퍼.

`torch.stack([torch.from_numpy(a).float() for a in arrs])` 패턴은 중간 list +
contiguous 출력 두 벌을 잡아 600s × N-channel windows 에서 ~70 GB 피크를 유발한다.
출력 텐서를 미리 할당하고 numpy 참조를 즉시 해제하며 fp16 으로 저장하면
디스크와 RAM 피크를 모두 절반 이하로 낮출 수 있다 (run.py loader 는
`torch.from_numpy(...).float()` 으로 다시 fp32 캐스팅하므로 학습 코드는 무수정).
"""

from __future__ import annotations

import gc
import re
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # tqdm 미설치 시 no-op 폴백
    def tqdm(it, **_kwargs):  # type: ignore[no-redef]
        return it


def stack_arrays_destructive(
    arrs: list[np.ndarray],
    signal_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """torch.stack 의 2x peak 없이 (N, T) 텐서 생성. arrs 는 in-place 로 비워진다."""
    if not arrs:
        return torch.empty((0,), dtype=signal_dtype)
    n = len(arrs)
    T = int(arrs[0].shape[0])
    out = torch.empty((n, T), dtype=signal_dtype)
    for i in range(n):
        a = arrs[i]
        out[i].copy_(torch.from_numpy(a))
        arrs[i] = None  # free numpy buffer immediately
    arrs.clear()
    return out


def consume_input_signals(
    samples: list[Any],
    input_signals: Iterable[str],
    signal_dtype: torch.dtype = torch.float16,
    attr: str = "input_signals",
) -> dict[str, torch.Tensor]:
    """샘플 객체의 `attr` (dict[str, np.ndarray]) 을 stype 별 텐서로 합친다.

    각 샘플에서 `dict.pop(stype)` 으로 numpy 를 빼내 즉시 해제한다.
    stype 별로 길이가 다른 샘플은 (기존 동작과 동일하게) 해당 stype 에서만 제외된다.
    """
    sig_tensors: dict[str, torch.Tensor] = {}
    for stype in input_signals:
        # 길이는 첫 등장 샘플 기준
        T = None
        for s in samples:
            d = getattr(s, attr)
            if stype in d:
                T = int(d[stype].shape[0])
                break
        if T is None:
            continue
        n = sum(1 for s in samples if stype in getattr(s, attr))
        out = torch.empty((n, T), dtype=signal_dtype)
        i = 0
        for s in samples:
            d = getattr(s, attr)
            arr = d.pop(stype, None)
            if arr is None:
                continue
            out[i].copy_(torch.from_numpy(arr))
            i += 1
        sig_tensors[stype] = out
    gc.collect()
    return sig_tensors


def stack_window_dicts_destructive(
    windows: list[dict],
    signal_dtype: torch.dtype = torch.float16,
) -> dict[str, torch.Tensor]:
    """patient-level pack: list[dict[stype -> np.ndarray]] → dict[stype -> (K, T)].

    각 dict 에서 stype 을 pop 하여 numpy 를 즉시 해제한다.
    """
    if not windows:
        return {}
    sig_types = sorted(windows[0].keys())
    sig_tensors: dict[str, torch.Tensor] = {}
    for st in sig_types:
        K = len(windows)
        # 첫 sample 에서 길이
        T = int(windows[0][st].shape[0])
        out = torch.empty((K, T), dtype=signal_dtype)
        for k in range(K):
            arr = windows[k].pop(st, None)
            if arr is None:
                # mismatch — pad with zeros (shouldn't happen if shapes uniform)
                continue
            out[k].copy_(torch.from_numpy(arr))
        sig_tensors[st] = out
    return sig_tensors


def consume_gap_masks(
    samples: list[Any],
    input_signals: Iterable[str],
    attr: str = "input_gap_masks",
) -> dict[str, torch.Tensor]:
    """consume_input_signals 의 gap_mask 버전.

    np.stack 의 2x peak 없이 (N, T) bool 텐서 생성. 각 sample 의 attr 에서
    pop 하여 numpy 즉시 해제.
    """
    sig_tensors: dict[str, torch.Tensor] = {}
    for stype in input_signals:
        T = None
        for s in samples:
            d = getattr(s, attr, None) or {}
            if stype in d:
                T = int(d[stype].shape[0])
                break
        if T is None:
            continue
        n = sum(1 for s in samples
                if stype in (getattr(s, attr, None) or {}))
        out = torch.empty((n, T), dtype=torch.bool)
        i = 0
        for s in samples:
            d = getattr(s, attr, None) or {}
            arr = d.pop(stype, None)
            if arr is None:
                continue
            out[i].copy_(torch.from_numpy(arr))
            i += 1
        sig_tensors[stype] = out
    gc.collect()
    return sig_tensors


def stack_attr_destructive(
    samples: list[Any],
    attr: str,
    signal_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """samples 의 attr (np.ndarray) 들을 (N, T) 텐서로 합치고 attr 을 None 으로 비운다."""
    if not samples:
        return torch.empty((0,), dtype=signal_dtype)
    n = len(samples)
    T = int(getattr(samples[0], attr).shape[0])
    out = torch.empty((n, T), dtype=signal_dtype)
    for i, s in enumerate(samples):
        a = getattr(s, attr)
        out[i].copy_(torch.from_numpy(a))
        setattr(s, attr, None)
    return out


# ── chunked save/load (sepsis-style split chunking) ──────────────────────


def _concat_values(vals: list[Any]) -> Any:
    """동일 구조 값 리스트를 key별 재귀 concat.

    - torch.Tensor  -> dim 0 으로 torch.cat (빈 텐서 제외)
    - dict           -> 키별 재귀 concat (e.g. signals={stype: (N,T)})
    - list           -> extend
    - 그 외 scalar   -> 첫 값 그대로
    """
    first = vals[0]
    if isinstance(first, torch.Tensor):
        non_empty = [v for v in vals if v.numel() > 0]
        if not non_empty:
            return first
        if len(non_empty) == 1:
            return non_empty[0]
        return torch.cat(non_empty, dim=0)
    if isinstance(first, dict):
        out_d: dict[Any, Any] = {}
        all_keys: list[Any] = []
        for v in vals:
            for kk in v.keys():
                if kk not in all_keys:
                    all_keys.append(kk)
        for kk in all_keys:
            out_d[kk] = _concat_values([v[kk] for v in vals if kk in v])
        return out_d
    if isinstance(first, list):
        out_l: list[Any] = []
        for v in vals:
            out_l.extend(v)
        return out_l
    return first


def _concat_payloads(payloads: list[dict]) -> dict:
    """split chunk payload(dict) 리스트를 단일 dict 로 병합 (key별 concat)."""
    if not payloads:
        return {}
    merged: dict[str, Any] = {}
    keys: list[str] = []
    for pl in payloads:
        for k in pl.keys():
            if k not in keys:
                keys.append(k)
    for k in keys:
        merged[k] = _concat_values([pl[k] for pl in payloads if k in pl])
    return merged


def _split_count(split_dict: dict) -> int:
    """split payload 에서 샘플 수(N) 추정 (첫 텐서/리스트의 길이)."""
    for v in split_dict.values():
        if isinstance(v, torch.Tensor):
            return int(v.shape[0]) if v.dim() > 0 else 0
        if isinstance(v, dict):
            for vv in v.values():
                if isinstance(vv, torch.Tensor):
                    return int(vv.shape[0]) if vv.dim() > 0 else 0
        if isinstance(v, list):
            return len(v)
    return 0


def load_prepared_chunked(
    data_path: str | Path,
    fold: int | None = None,
) -> dict:
    """단일 .pt 또는 split-chunked .pt 묶음을 통일된 dict 로 로드.

    Back-compat: ``fold is None`` 이고 ``data_path`` 가 실제 파일이면 그대로
    ``torch.load`` 한다.

    Chunked: ``data_path`` (보통 ``<base>.pt`` 또는 확장자 없는 base) 의 stem 을
    prefix 로 하는 chunk 파일들을 split 별로 모아 텐서/리스트를 concat 한다.
    반환 shape 은 기존 단일 파일과 동일하다::

        {"train": {...}, ["val": {...},] "test": {...}, "metadata": {...}}

    Parameters
    ----------
    fold : int | None
        None  → 비-fold glob ``<stem>_<split>_chunk*.pt`` (+ 단일파일 back-compat).
        int K → fold glob ``<stem>_fold{K}_<split>_chunk*.pt`` 만 로드 (K-fold CV).

    cross_modal(`signals` nested dict + `case_ids`) 와
    forecast(`context`/`target` flat 텐서 + `case_ids`) 양쪽을 처리한다.
    """
    p = Path(data_path)

    # 1) 단일 파일 우선 (back-compat). fold 미지정 시에만.
    if fold is None:
        if p.is_file():
            return torch.load(p, weights_only=False)
        if p.suffix != ".pt":
            single = p.with_suffix(".pt")
            if single.is_file():
                return torch.load(single, weights_only=False)

    # 2) chunk 집계: prefix 를 기준으로 <prefix>_<split>_chunk<idx>.pt glob
    stem = p.name[:-3] if p.name.endswith(".pt") else p.name
    parent = p.parent if str(p.parent) else Path(".")
    prefix = stem if fold is None else f"{stem}_fold{int(fold)}"

    chunk_files = sorted(parent.glob(f"{prefix}_*_chunk*.pt"))

    # split 별로 (chunk_idx, file) 수집
    pat = re.compile(r"(.+)_chunk(\d+)$")
    fold_split = re.compile(r"^fold\d+_")  # fold 파일 split 토큰
    splits: dict[str, list[tuple[int, Path]]] = {}
    for f in chunk_files:
        name = f.name[:-3]  # strip .pt
        rest = name[len(prefix) + 1:]  # "{split}_chunk{idx}"
        m = pat.match(rest)
        if not m:
            continue
        split_name = m.group(1)
        # fold 미지정인데 fold 파일이 glob 에 섞였으면 제외 (n_folds>=2 산출물).
        if fold is None and fold_split.match(split_name):
            continue
        cidx = int(m.group(2))
        splits.setdefault(split_name, []).append((cidx, f))

    if not splits:
        looked = parent / (prefix + "_*_chunk*.pt")
        raise FileNotFoundError(
            f"No single file or chunk files found for: {data_path} "
            f"(fold={fold}, looked for {looked})"
        )

    out: dict[str, Any] = {}
    metadata: dict[str, Any] | None = None
    for split_name, items in splits.items():
        items.sort(key=lambda x: x[0])
        payloads: list[dict] = []
        for _, f in items:
            ck = torch.load(f, weights_only=False)
            if metadata is None and isinstance(ck.get("metadata"), dict):
                metadata = dict(ck["metadata"])
            payloads.append(
                {k: v for k, v in ck.items() if k not in ("split", "metadata")}
            )
        out[split_name] = _concat_payloads(payloads)

    # metadata 정리: chunk 고유 키 제거 + 집계 카운트 보강 (fold_idx/n_folds 는 보존)
    metadata = metadata or {}
    for k in ("chunk_idx", "n_windows", "split"):
        metadata.pop(k, None)
    for split_name in out:
        metadata[f"n_{split_name}"] = _split_count(out[split_name])
    out["metadata"] = metadata
    return out


def load_prepared_split_chunked(
    data_path: str | Path,
    fold: int | None = None,
    splits: Iterable[str] = ("train", "val", "test"),
) -> dict:
    """분류 prepare_data 의 per-(fold,split)[_chunk] 산출물을 통일 dict 로 로드.

    분류 task(hypotension/mortality/ich)의 ``save_split_dataset`` 은 split 마다
    ``{"split": str, "data": <split_payload>, "metadata": {...}}`` 형태로 따로
    저장한다 (forecast 의 flat 텐서와 달리 실제 payload 가 ``"data"`` 로 한 겹
    wrap 된다). 이 함수는 ``"data"`` 를 벗기고 같은 (fold, split) 의 모든 chunk 를
    concat 하여, 기존 run.py 가 기대하는::

        {"train": <payload>, ["val": <payload>,] "test": <payload>, "metadata": {...}}

    형태로 돌려준다.

    payload 는 task 에 따라 dict (hypotension/ich: ``signals``/``labels``/... 키) 또는
    list (mortality: patient dict 리스트) 일 수 있으며, 양쪽 모두 :func:`_concat_values`
    로 dim0/extend concat 된다.

    Back-compat
    -----------
    ``data_path`` 가 실제 .pt 파일이면 (prefix 가 아니라 통합 산출물이면) 그대로
    ``torch.load`` 하여 반환한다 — 기존 단일파일 경로 무수정.

    Parameters
    ----------
    fold : int | None
        None  → 비-fold glob ``<base>_<split>*.pt`` (+ 단일파일 back-compat).
        int K → fold glob ``<base>_fold{K}_<split>*.pt`` 만 로드 (K-fold CV).
    splits : Iterable[str]
        탐색할 split 이름들. 존재하지 않는 split (예: val 없는 legacy) 은 건너뛴다.
    """
    p = Path(data_path)

    # 1) 실제 파일이면 단일 통합 산출물 — 그대로 로드 (back-compat).
    if p.is_file():
        return torch.load(p, weights_only=False)
    if fold is None and p.suffix != ".pt":
        single = p.with_suffix(".pt")
        if single.is_file():
            return torch.load(single, weights_only=False)

    # 2) prefix 모드: <base>[_fold{K}]_<split>*.pt 를 split 별로 glob → concat.
    base = p.name[:-3] if p.name.endswith(".pt") else p.name
    parent = p.parent if str(p.parent) else Path(".")
    prefix = base if fold is None else f"{base}_fold{int(fold)}"

    out: dict[str, Any] = {}
    metadata: dict[str, Any] | None = None
    for split_name in splits:
        files = sorted(parent.glob(f"{prefix}_{split_name}*.pt"))
        if not files:
            continue
        payloads: list[Any] = []
        # 네트워크 마운트에서 chunk 순차 torch.load 는 분 단위 — 진행률 표시.
        for f in tqdm(
            files,
            desc=f"load {split_name} (fold={fold})",
            unit="chunk",
            mininterval=0.5,
        ):
            ck = torch.load(f, weights_only=False)
            if metadata is None and isinstance(ck.get("metadata"), dict):
                metadata = dict(ck["metadata"])
            # split payload 는 "data" 로 wrap 됨. 누락 시 chunk 전체에서 추출.
            payloads.append(ck["data"] if "data" in ck else ck)
        out[split_name] = _concat_values(payloads)

    if not out:
        looked = parent / (prefix + "_<split>*.pt")
        raise FileNotFoundError(
            f"No single file or split-chunk files found for: {data_path} "
            f"(fold={fold}, looked for {looked})"
        )

    # metadata 정리: chunk 고유 키 제거 (fold_idx/n_folds 는 보존).
    metadata = metadata or {}
    for k in ("chunk_idx", "n_windows", "n_samples", "n_patients", "n_dead", "split"):
        metadata.pop(k, None)
    out["metadata"] = metadata
    return out


def add_signal_dtype_arg(
    parser, default: str = "float16"
) -> Callable[[str], torch.dtype]:
    """argparse 에 --signal-dtype 추가 후 매핑 함수 반환."""
    parser.add_argument(
        "--signal-dtype", type=str, default=default,
        choices=["float16", "float32"],
        help="Storage dtype for waveform tensors. fp16 halves disk/RAM peak; "
             "run.py auto-casts back to fp32 at load time.",
    )
    return lambda s: {"float16": torch.float16, "float32": torch.float32}[s]
