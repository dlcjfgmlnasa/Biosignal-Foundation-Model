# -*- coding: utf-8 -*-
"""CSFM — Cardiac Sensing Foundation Model (Nat. Mach. Intell. 2026, Gu et al.) 어댑터.

**멀티모달**(ECG + PPG) cardiac FM. 1.7M 개인 데이터로 self-supervised 사전학습.
CARMEN 과 SR·모달리티가 가장 가까운 핵심 비교군(둘 다 ECG+PPG, **100Hz**).

Upstream: https://github.com/guxiao0822/Cardiac-Sensing-FM  (network/model.py)
가중치   : **Academic Access Agreement 필요** (xiao.gu@eng.ox.ac.uk, README 참조).

규격(README):
  - 입력 `[channels, time]`, **100Hz** (우리 데이터와 동일 → 리샘플 불필요).
  - signal_size=2500 (=25s), patch_size=25.
  - channel-type 인덱스: 0-11 = 12-lead ECG {I,II,III,aVR,aVL,aVF,V1-V6}, 12 = PPG.
    모니터링/wearable ECG 는 **Lead II 로 취급 → channel=1**. 우리는 ecg→1, ppg→12.
  - feature 추출: `to_latent=Identity` 라 ``mlp_head`` 를 Identity 로 교체하면 forward 가
    CLS latent (b, dim) 을 반환. dim = 768(tiny/base) / 1024(large).

variant 는 env ``FM_CSFM_VARIANT`` 로 선택 (tiny|base|large, 기본 base).
"""
from __future__ import annotations

import os

import torch
from torch import nn

from downstream.baselines.fm.encoders import add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder

# 우리 modality key → CSFM channel-type index.
_CSFM_CHANNEL = {"ecg": 1, "ppg": 12}  # ecg=Lead II, ppg


class CSFMEncoder(FMEncoder):
    name = "csfm"
    native_sr = 100.0  # 우리 소스와 동일 → 리샘플 no-op
    seg_sec = 25.0     # signal_size 2500 / 100Hz
    supported_modalities = frozenset({"ecg", "ppg"})  # 멀티모달
    max_channels = None

    def _build_and_load(self) -> None:
        add_repo_to_path(self.third_party_root, "FM_CSFM_ROOT")
        from network.model import CSFM_model  # type: ignore

        variant = os.environ.get("FM_CSFM_VARIANT", "base").lower()
        model = CSFM_model(variant)
        ckpt = torch.load(self.weights_path, map_location="cpu")
        if isinstance(ckpt, dict):
            state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        else:
            state = ckpt
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # mlp_head 는 어차피 Identity 로 교체하므로 mlp_head.* missing 은 무해.
        leftover = [m for m in missing if not m.startswith("mlp_head.")]
        if leftover:
            print(f"[csfm] WARN missing (mlp_head 외): {leftover[:8]}{'...' if len(leftover) > 8 else ''}")
        if unexpected:
            print(f"[csfm] WARN unexpected: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        model.mlp_head = nn.Identity()  # 분류 head 제거 → forward 가 CLS latent 반환
        self.model = model
        # dim 은 pos-embedding 마지막 축에서 직접 읽는다(variant 무관, 견고).
        self.embed_dim = int(model.ts_pos_embedding.shape[-1])
        self._cur_channels: list[int] = [_CSFM_CHANNEL["ecg"]]

    def _select(self, x: torch.Tensor, keys: list[str]) -> torch.Tensor:
        # ecg/ppg 채널만 순서대로 선택하고, 각 채널의 CSFM channel-type index 를 기록.
        idxs = [i for i, k in enumerate(keys) if k in _CSFM_CHANNEL]
        if not idxs:
            raise ValueError(
                f"[csfm] 지원 modality {{ecg, ppg}} 중 입력 {keys} 에 없음 → 평가 불가."
            )
        self._cur_channels = [_CSFM_CHANNEL[keys[i]] for i in idxs]
        return x[:, idxs, :].contiguous()

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        channel = torch.as_tensor(self._cur_channels, dtype=torch.long, device=x.device)
        out = self.model(x, channel, text=None, task="cls")  # mlp_head=Identity → (b, dim)
        return out

    def _infer_embed_dim(self) -> int:
        return self.embed_dim
