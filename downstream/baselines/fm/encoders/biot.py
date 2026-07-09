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

import torch

from downstream.baselines.fm.encoders import add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder


def _strip_prefix(state: dict, prefix: str) -> dict:
    if any(k.startswith(prefix) for k in state):
        return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}
    return state


class BIOTEncoderWrapper(FMEncoder):
    name = "biot"
    native_sr = 200.0
    seg_sec = 10.0
    supported_modalities = None  # 다채널: 제공된 모든 채널 사용
    max_channels: int | None = None  # 체크포인트 n_channels 로 로드 후 clamp

    def _build_and_load(self) -> None:
        add_repo_to_path(self.third_party_root, "FM_BIOT_ROOT")
        from model.biot import BIOTEncoder  # type: ignore

        ckpt = torch.load(self.weights_path, map_location="cpu")
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

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # (b, emb_size)
