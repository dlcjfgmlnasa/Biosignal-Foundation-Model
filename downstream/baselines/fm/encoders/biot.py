# -*- coding: utf-8 -*-
"""BIOT (NeurIPS 2023, ycq091044) frozen encoder 어댑터.

범용 biosignal transformer(EEG 사전학습, cross-data). 채널을 STFT 토큰화해 채널·위치
임베딩과 함께 linear-attention transformer 로 인코딩 → 시간축 평균 → (b, emb_size=256).
**다채널** 모델이라 task 가 제공하는 모든 신호 채널을 함께 사용한다.

Upstream: https://github.com/ycq091044/BIOT  (model/biot.py → BIOTEncoder)
가중치   : repo 의 pretrained-models/*.ckpt (또는 HF braindecode/BIOT)
           EEG-PREST-16-channels.ckpt / EEG-SHHS+PREST-18-channels.ckpt /
           EEG-six-datasets-18-channels.ckpt

네이티브 200Hz, n_fft=200·hop=100(=1s 토큰·0.5s hop). n_channels·emb_size 는 체크포인트
의 ``channel_tokens.weight`` shape 에서 추론해 아키텍처를 정확히 맞춘다.

주의: BIOT 사전학습 채널 임베딩은 EEG montage 의미다. ECG/ABP/PPG 를 앞쪽 채널 슬롯에
매핑해 쓰는 것은 BIOT 의 cross-data 설계 취지에 부합하나(도메인 이질), 그 한계는
README 에 명시한다.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch

from downstream.baselines.fm.encoders import _dsp, resolve_repo_root
from downstream.baselines.fm.encoders.base import FMEncoder


def _strip_prefix(state: dict, prefix: str) -> dict:
    if any(k.startswith(prefix) for k in state):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    return state


def _load_upstream_biot(root: str):
    """upstream ``model/biot.py`` 를 파일 경로에서 직접 로드해 ``BIOTEncoder`` 를 반환.

    BIOT 레포의 최상위 패키지명이 ``model`` 이라 CARMEN 의 ``model/`` 패키지와 충돌한다.
    runner 가 ``_fm_common`` → CARMEN ``model`` 을 먼저 import 하므로 ``sys.modules['model']``
    이 선점되어, sys.path 앞에 레포 루트를 넣어도 ``from model.biot import ...`` 는
    ModuleNotFoundError 가 난다. 따라서 패키지 경로를 거치지 않고 파일에서 직접 로드한다
    (``biot.py`` 는 패키지 내부 상대 import 가 없어 안전하며, ``model/__init__.py`` 를
    건너뛰어 sparcnet 등 불필요한 의존성도 피한다).
    """
    path = os.path.join(root, "model", "biot.py")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"BIOT 레포에 model/biot.py 없음: {path}")
    mod_name = "_upstream_biot"
    if mod_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
    return sys.modules[mod_name].BIOTEncoder


class BIOTEncoderWrapper(FMEncoder):
    name = "biot"
    native_sr = 200.0
    seg_sec = 10.0
    supported_modalities = None  # 다채널: 제공된 모든 채널 사용
    max_channels: int | None = None  # 체크포인트 n_channels 로 로드 후 clamp

    def _build_and_load(self) -> None:
        root = resolve_repo_root(self.third_party_root, "FM_BIOT_ROOT")
        BIOTEncoder = _load_upstream_biot(root)

        ckpt = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        state = _strip_prefix(state, "biot.")  # wrapper 로 저장된 경우 대비
        # 아키텍처 파라미터를 체크포인트에서 추론.
        ct = state.get("channel_tokens.weight")
        if ct is not None:
            n_channels, emb_size = int(ct.shape[0]), int(ct.shape[1])
        else:
            n_channels, emb_size = 18, 256
        model = BIOTEncoder(
            emb_size=emb_size, heads=8, depth=4,
            n_channels=n_channels, n_fft=200, hop_length=100,
        )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[biot] WARN missing: {list(missing)[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[biot] WARN unexpected: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        self.model = model
        self.embed_dim = emb_size
        self.max_channels = n_channels  # 입력 채널이 pretrained 슬롯 수를 넘지 않도록 clamp

    def _select(self, x: torch.Tensor, keys: list[str]) -> torch.Tensor:
        # 다채널: 전 채널 사용하되 체크포인트 채널 슬롯 수로 clamp.
        if self.max_channels is not None and x.shape[1] > self.max_channels:
            x = x[:, : self.max_channels, :]
        return x.contiguous()

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # upstream utils.py 의 모든 로더가 쓰는 정규화: x / (q95(|x|) + 1e-8).
        # 평균을 빼지 않아 DC/baseline 이 보존되며, 이것이 STFT bin-0·저주파에 직접 반영된다.
        # (기본 z-norm 은 평균을 빼고 std 로 나눠 진폭이 ~2배 커진다 → 사전학습 분포 이탈)
        return _dsp.amplitude_norm_q95(x)

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (b, emb_size)
