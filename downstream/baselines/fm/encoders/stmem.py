# -*- coding: utf-8 -*-
"""ST-MEM (ICLR 2024, VUNO) frozen encoder 어댑터.

ECG 전용. Spatio-temporal masked ECG modeling 으로 사전학습된 ViT. 네이티브 250Hz,
seq_len=2250(=9s), 12-lead. ``num_classes=None`` 이면 forward 가 lead·patch 평균 →
LayerNorm 한 (b, width) 표현을 직접 반환한다(width=768 base / 384 small).

Upstream: https://github.com/vuno/ST-MEM  (models/encoder/st_mem_vit.py)
가중치   : repo README 의 Google Drive 링크(encoder-only / encoder+decoder). encoder-only 사용.

우리 ECG 는 단일 lead 이므로 **num_leads=1** 로 생성하고 사전학습 lead 임베딩 12개 중
하나만 로드한다(기본 index 1 = Lead II, 수술장·ICU 모니터링 ECG 의 표준 lead).

⚠ num_leads=12 로 생성해 단일 lead 를 넣으면 안 된다. upstream ``forward_encoding`` 은
``lead_embeddings`` 를 slicing 없이 전부 stack 하므로 (b,1,n+2,w) + (b,12,n+2,w) 브로드캐스트
가 일어나 **단일 lead 가 12개 lead 임베딩과 함께 12배 복제**되고, 이어지는
``rearrange(..., c=num_leads)`` 때문에 lead 별 SEP 토큰 제거도 어긋난다. 예외 없이
(b, width) 가 나오므로 오작동을 알아채기 어렵다.
"""
from __future__ import annotations

import torch

from downstream.baselines.fm.encoders import _dsp, add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder

# 사전학습 lead 임베딩 중 우리 단일 lead 에 매핑할 인덱스 (0=I, 1=II, 2=III, ...).
LEAD_INDEX = 1  # Lead II


def _unwrap_state(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict", "encoder"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _select_lead_embedding(state: dict, lead_index: int) -> dict:
    """12개 ``lead_embeddings.*`` 중 lead_index 하나만 남겨 ``lead_embeddings.0`` 으로 remap."""
    n_leads = sum(1 for k in state if k.startswith("lead_embeddings."))
    if n_leads == 0:
        return state
    if not 0 <= lead_index < n_leads:
        raise ValueError(f"LEAD_INDEX={lead_index} 가 체크포인트 lead 수 {n_leads} 범위 밖")
    out = {}
    for k, v in state.items():
        if not k.startswith("lead_embeddings."):
            out[k] = v
        elif int(k.split(".")[1]) == lead_index:
            out["lead_embeddings.0"] = v
    return out


class STMEMEncoder(FMEncoder):
    name = "stmem"
    native_sr = 250.0
    seg_sec = 9.0  # seq_len 2250 / 250Hz
    supported_modalities = frozenset({"ecg"})
    max_channels = 1

    def _build_and_load(self) -> None:
        add_repo_to_path(self.third_party_root, "FM_STMEM_ROOT")
        from models.encoder.st_mem_vit import st_mem_vit_base  # type: ignore

        # num_classes=None → head=Identity → forward 가 (b, width) 표현 반환.
        model = st_mem_vit_base(num_leads=1, num_classes=None, seq_len=2250, patch_size=75)
        state = _unwrap_state(torch.load(self.weights_path, map_location="cpu", weights_only=False))
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        state = _select_lead_embedding(state, LEAD_INDEX)
        missing, unexpected = model.load_state_dict(state, strict=False)
        leftover = [m for m in missing if not m.startswith("head.")]
        if leftover:
            print(f"[stmem] WARN missing (head 외): {leftover[:8]}{'...' if len(leftover) > 8 else ''}")
        if unexpected:
            print(f"[stmem] WARN unexpected: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        self.model = model
        self.embed_dim = int(model.width)  # 768 (base)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # upstream util.transforms: HighpassFilter(0.67) → LowpassFilter(40) → Standardize.
        # 단일 lead 이므로 Standardize(axis=(-1,-2)) 는 per-channel z-score 와 동일하다.
        x = _dsp.sos_filter(x, self.native_sr, 0.67, btype="highpass", order=5)
        x = _dsp.sos_filter(x, self.native_sr, 40.0, btype="lowpass", order=5)
        return super()._preprocess(x)

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # num_classes=None → (b, width)
