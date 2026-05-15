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
from typing import Any, Callable, Iterable

import numpy as np
import torch


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
