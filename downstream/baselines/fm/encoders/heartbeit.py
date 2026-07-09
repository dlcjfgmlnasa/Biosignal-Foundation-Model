# -*- coding: utf-8 -*-
"""HeartBEiT (npj Digital Medicine 2023, Mount Sinai) frozen encoder 어댑터.

ECG 전용 **이미지 기반** FM: ECG 파형을 이미지로 렌더링해 BEiT(Vision Transformer,
DALL-E tokenizer 로 masked image modeling 사전학습)로 인코딩한다. 임베딩 = BEiT
pooled representation (base=768-d).

⚠ 가중치 접근: HeartBEiT 체크포인트는 저자(Mount Sinai)에게 **접근 신청**이 필요하다
   (공개 무료 다운로드 아님). 획득한 HuggingFace 형식 디렉토리를 ``--weights`` 로 준다.
⚠ 다른 FM 4종(waveform)과 달리 입력이 이미지라, 파형→이미지 렌더링을 여기서 수행한다.
   렌더링은 표준 ECG 용지 렌더가 아닌 **경량 line-raster**(이식성 우선)이며, 저자 원본
   렌더와 화소 단위로 동일하지 않다 — 결과 해석 시 유의(README 명시).

Upstream: https://arxiv.org/abs/2212.14040 / https://www.nature.com/articles/s41746-023-00840-9
의존성   : transformers (BeitModel). ``pip install transformers``.
"""
from __future__ import annotations

import torch

from downstream.baselines.fm.encoders.base import FMEncoder

# BEiT 기본 이미지 정규화 (mean=std=0.5).
_IMG_MEAN = 0.5
_IMG_STD = 0.5


class HeartBEiTEncoder(FMEncoder):
    name = "heartbeit"
    native_sr = 500.0
    seg_sec = 10.0  # 표준 10s ECG
    supported_modalities = frozenset({"ecg"})
    max_channels = 1
    img_size = 224

    def _build_and_load(self) -> None:
        try:
            from transformers import BeitModel  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError("HeartBEiT 는 transformers(BeitModel) 필요 — pip install transformers") from e
        # weights_path = 획득한 HF BEiT 디렉토리(또는 hub id). MIM 체크포인트면 base encoder 로드.
        model = BeitModel.from_pretrained(self.weights_path)
        self.model = model
        self.embed_dim = int(model.config.hidden_size)

    def _render(self, x: torch.Tensor) -> torch.Tensor:
        """z-norm 된 파형 (b,1,L) → BEiT 입력 이미지 (b,3,H,W).

        각 이미지 열(w)에 파형 1점을 매핑해 선 궤적을 그린다(경량 raster). 진폭은
        ±3σ clip 후 [0,1] 스케일, y 축 반전(위=고전위). 3채널 복제 + BEiT mean/std 정규화.
        """
        b, _, L = x.shape
        H = W = self.img_size
        v = x.squeeze(1)  # (b, L)
        v = torch.clamp(v, -3.0, 3.0) / 6.0 + 0.5  # → [0,1]
        idx = torch.linspace(0, L - 1, W, device=x.device).round().long()
        cols = v[:, idx]  # (b, W)
        ys = ((1.0 - cols) * (H - 1)).round().long().clamp(0, H - 1)  # (b, W)
        img = torch.zeros(b, H, W, device=x.device)
        br = torch.arange(b, device=x.device).unsqueeze(1).expand(b, W)
        wr = torch.arange(W, device=x.device).unsqueeze(0).expand(b, W)
        img[br, ys, wr] = 1.0
        img = img.unsqueeze(1).repeat(1, 3, 1, 1)  # (b,3,H,W)
        return (img - _IMG_MEAN) / _IMG_STD

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self._render(x)
        out = self.model(pixel_values=pixel_values)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output  # (b, hidden)
        return out.last_hidden_state.mean(dim=1)  # (b, hidden)

    def _infer_embed_dim(self) -> int:
        return self.embed_dim  # config.hidden_size 로 이미 설정
