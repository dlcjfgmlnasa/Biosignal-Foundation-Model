# -*- coding: utf-8 -*-
"""PaPaGei-S (ICLR 2025, Nokia Bell Labs) frozen encoder 어댑터.

PPG 전용 FM. 네이티브 125Hz, 10s = 1250 samples, single-channel. ResNet1DMoE
encoder → 512-d 임베딩(model 출력 tuple 의 [0]).

Upstream: https://github.com/Nokia-Bell-Labs/papagei-foundation-model
가중치   : Zenodo record 13983110 → ``papagei_s.pt`` (weights/ 하위 권장)

모델 생성은 upstream README 방식과 동일(``models.resnet.ResNet1DMoE``). 가중치 로드는
upstream ``linearprobing.utils.load_model_without_module_prefix`` 와 같은 동작(``module.``
prefix 제거)을 여기서 직접 수행한다 — upstream 헬퍼는 ``torch.load(path)`` 를 map_location
없이 호출해 CUDA 스토리지로 저장된 ``papagei_s.pt`` 를 CPU 머신에서 로드하지 못하고,
GPU 머신에서도 ``--device`` 를 무시하고 항상 GPU0 에 올린다. 헬퍼를 쓰지 않으면
pandas/joblib/sklearn 등 불필요한 import 체인도 피할 수 있다.

전처리: upstream 순서(z-score → pyPPG bandpass)를 재현한다. upstream example 은
``torch_ecg Normalize(z-score)`` → ``preprocess_one_ppg_signal`` (pyPPG ``preproc.Preprocessing``:
Chebyshev II bandpass order4/rs20 [0.5,12] + 0.02·fs 이동평균) → resample 순이다. bandpass 를
z-score **뒤**에 두어(그리고 이후 재정규화하지 않아) upstream 과 동일한 분산 스케일을 유지한다.
어댑터가 이미 native 125Hz 로 리샘플한 뒤 이 함수가 호출되므로 필터 설계 SR 은 native_sr 이다.
"""
from __future__ import annotations

import torch

from downstream.baselines.fm.encoders import _dsp, add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder

def _unwrap_state(ckpt):
    """checkpoint 가 wrapper dict 인 경우 state_dict 를 꺼낸다 (papagei_s.pt 는 평평한 형태)."""
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


_MOE_KW = dict(
    base_filters=32,
    kernel_size=3,
    stride=2,
    groups=1,
    n_block=18,
    n_classes=512,
    n_experts=3,
)


class PaPaGeiEncoder(FMEncoder):
    name = "papagei"
    native_sr = 125.0
    seg_sec = 10.0
    supported_modalities = frozenset({"ppg"})
    max_channels = 1

    def _build_and_load(self) -> None:
        add_repo_to_path(self.third_party_root, "FM_PAPAGEI_ROOT")
        from models.resnet import ResNet1DMoE  # type: ignore

        model = ResNet1DMoE(in_channels=1, **_MOE_KW)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=False)
        state = _unwrap_state(state)
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[papagei] WARN missing: {list(missing)[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[papagei] WARN unexpected: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        self.model = model
        self.embed_dim = 512  # n_classes

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # upstream 순서: z-score → pyPPG Chebyshev II bandpass(+이동평균). 이후 재정규화 없음.
        x = super()._preprocess(x)  # per-channel z-score (torch_ecg Normalize 와 동일)
        return _dsp.pyppg_bandpass(x, self.native_sr)

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        emb = out[0] if isinstance(out, (tuple, list)) else out
        return emb  # (b, 512)
