# -*- coding: utf-8 -*-
"""ST-MEM (ICLR 2024, VUNO) frozen encoder 어댑터.

ECG 전용. Spatio-temporal masked ECG modeling 으로 사전학습된 ViT. 네이티브 250Hz,
seq_len=2250(=9s), 12-lead. ``num_classes=None`` 이면 forward 가 lead·patch 평균 →
LayerNorm 한 (b, width) 표현을 직접 반환한다(width=768 base / 384 small).

Upstream: https://github.com/vuno/ST-MEM  (models/encoder/st_mem_vit.py)
가중치   : repo README 의 Google Drive 링크(encoder-only / encoder+decoder). encoder-only 사용.

우리 ECG 는 단일 lead 이므로 (b,1,2250) 로 입력되어 lead-0 임베딩을 사용한다(12-lead
모델에 단일 lead 를 넣는 표준 근사 — README 에 명시). num_leads=12 로 생성해 사전학습
lead 임베딩 전부를 로드한다.
"""
from __future__ import annotations

import torch

from downstream.baselines.fm.encoders import add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder


def _unwrap_state(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict", "encoder"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


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
        model = st_mem_vit_base(num_leads=12, num_classes=None, seq_len=2250, patch_size=75)
        state = _unwrap_state(torch.load(self.weights_path, map_location="cpu"))
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        leftover = [m for m in missing if not m.startswith("head.")]
        if leftover:
            print(f"[stmem] WARN missing (head 외): {leftover[:8]}{'...' if len(leftover) > 8 else ''}")
        if unexpected:
            print(f"[stmem] WARN unexpected: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        self.model = model
        self.embed_dim = int(model.width)  # 768 (base)

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)  # num_classes=None → (b, width)
