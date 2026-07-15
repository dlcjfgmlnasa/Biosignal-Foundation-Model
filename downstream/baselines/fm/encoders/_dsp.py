# -*- coding: utf-8 -*-
"""외부 FM upstream 전처리(필터) 재현용 DSP 헬퍼.

각 upstream 레포는 encoder 에 넣기 전 자체 필터를 적용한다. 어댑터의 기본
per-channel z-norm 만 쓰면 논문 스펙과 어긋나므로, 모델별 ``_preprocess`` override
에서 아래 함수들로 upstream 파이프라인을 재현한다.

동작 지점: :meth:`FMEncoder.encode` 는 리샘플 → ``_preprocess`` 순서이므로, 여기 들어오는
텐서는 이미 **모델 네이티브 SR** 이다. 따라서 필터 설계 주파수도 native_sr 기준이다.

부수 효과 (측정): upstream 의 40Hz 저역통과는 100Hz→native 선형보간이 만든 상위대역
imaging 아티팩트도 함께 제거한다. 필터를 켜면 100Hz 경로와 native-SR 경로의 임베딩
cosine 이 ECGFounder 0.624→0.982, ST-MEM 0.942→0.998 로 수렴한다.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from scipy.signal import butter, cheby2, filtfilt, iirnotch, medfilt, sosfiltfilt


def _apply_rowwise(x: torch.Tensor, fn: Callable[[np.ndarray], np.ndarray]) -> torch.Tensor:
    """(b, C, L) 텐서의 각 (b,C) 행에 numpy 필터 fn 을 적용하고 원 device/dtype 로 복귀."""
    device, dtype = x.device, x.dtype
    arr = x.detach().cpu().to(torch.float64).numpy()
    b, c, length = arr.shape
    flat = arr.reshape(b * c, length)
    out = np.empty_like(flat)
    for i, row in enumerate(flat):
        out[i] = fn(row)
    out = np.ascontiguousarray(out.reshape(b, c, length))
    return torch.from_numpy(out).to(device=device, dtype=dtype)


def notch(x: torch.Tensor, fs: float, freq: float, q: float = 30.0) -> torch.Tensor:
    """전원 잡음 제거 (iirnotch + filtfilt). upstream ECGFounder ``util.filter_bandpass`` 와 동일."""
    b, a = iirnotch(freq, q, fs)
    return _apply_rowwise(x, lambda r: filtfilt(b, a, r))


def bandpass(x: torch.Tensor, fs: float, low: float, high: float, order: int = 4) -> torch.Tensor:
    """Butterworth bandpass + filtfilt (zero-phase)."""
    b, a = butter(order, [low, high], btype="bandpass", fs=fs)
    return _apply_rowwise(x, lambda r: filtfilt(b, a, r))


def sos_filter(x: torch.Tensor, fs: float, cutoff: float, btype: str, order: int = 5) -> torch.Tensor:
    """SOS Butterworth + sosfiltfilt. upstream ST-MEM ``util.transforms.SOSFilter`` 와 동일."""
    sos = butter(order, cutoff, btype=btype, fs=fs, output="sos")
    return _apply_rowwise(x, lambda r: sosfiltfilt(sos, r))


def remove_baseline_median(x: torch.Tensor, fs: float, frac: float = 0.4) -> torch.Tensor:
    """median filter 로 baseline 추정 후 차감. upstream ECGFounder 와 동일 (kernel = 0.4·fs, 홀수)."""
    kernel = int(frac * fs) + 1
    if kernel % 2 == 0:
        kernel += 1
    return _apply_rowwise(x, lambda r: r - medfilt(r, kernel_size=kernel))


def amplitude_norm_q95(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """upstream BIOT 정규화: ``x / (quantile(|x|, 0.95) + eps)``. **평균을 빼지 않아 DC 가 보존된다.**"""
    q = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True)
    return x / (q + eps)


def pyppg_bandpass(x: torch.Tensor, fs: float, fl: float = 0.5, fh: float = 12.0,
                   order: int = 4, rs: float = 20.0) -> torch.Tensor:
    """upstream PaPaGei 의 PPG 필터 (pyPPG ``preproc.Preprocessing``): Chebyshev II bandpass
    + (fs≥75 이면) 0.02·fs 이동평균 스무딩. 근거: pyPPG ``preproc.py`` L19-28."""
    b, a = cheby2(order, rs, [fl, fh], btype="bandpass", fs=fs)
    win = int(round(fs * 0.02))
    if fs >= 75 and win >= 1:
        bm = np.ones(win) / win

        def _fn(r):
            r = filtfilt(b, a, r)
            return filtfilt(bm, [1.0], r)
    else:
        def _fn(r):
            return filtfilt(b, a, r)

    return _apply_rowwise(x, _fn)
