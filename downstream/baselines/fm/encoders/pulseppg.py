# -*- coding: utf-8 -*-
"""Pulse-PPG (UbiComp 2025, maxxu05) frozen encoder 어댑터.

PPG 전용 FM(field-trained). 네이티브 **50Hz**(eval 데이터가 ``_ppg_50Hz``). RelCon 으로
학습된 ResNet1D encoder(``pulseppg.nets.ResNet1D.ResNet1D_Net.Net``, finalpool="max")를
그대로 frozen feature extractor 로 사용 — ``net(x)`` 가 임베딩을 직접 반환한다.

Upstream: https://github.com/maxxu05/pulseppg  (pulseppg/ 패키지)
가중치   : Zenodo 10.5281/zenodo.17270930 → download_model.sh 로 unzip
           (체크포인트 dict 의 "net" 키에 encoder state_dict 저장)

사전학습은 4분 입력이지만 temporal pooling 으로 임의 길이 처리 가능 → 여기서는
``seg_sec`` (기본 60s) 단위로 잘라 세그먼트 평균한다.
"""
from __future__ import annotations

import torch

from downstream.baselines.fm.encoders import add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder

_NET_KW = dict(
    in_channels=1,
    base_filters=128,
    kernel_size=11,
    stride=2,
    groups=1,
    n_block=12,
    finalpool="max",
)


class PulsePPGEncoder(FMEncoder):
    name = "pulseppg"
    native_sr = 50.0
    seg_sec = 60.0
    supported_modalities = frozenset({"ppg"})
    max_channels = 1

    def _build_and_load(self) -> None:
        add_repo_to_path(self.third_party_root, "FM_PULSEPPG_ROOT")
        from pulseppg.nets.ResNet1D.ResNet1D_Net import Net  # type: ignore

        model = Net(**_NET_KW)
        ckpt = torch.load(self.weights_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("net") or ckpt.get("state_dict") or ckpt
        else:
            state = ckpt
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[pulseppg] WARN missing: {list(missing)[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[pulseppg] WARN unexpected: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        self.model = model
        self.embed_dim = 0  # base._infer_embed_dim 이 dummy forward 로 추론

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # finalpool="max" → (b, C_final)
