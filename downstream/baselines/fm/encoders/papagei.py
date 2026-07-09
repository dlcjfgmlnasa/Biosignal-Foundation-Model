# -*- coding: utf-8 -*-
"""PaPaGei-S (ICLR 2025, Nokia Bell Labs) frozen encoder 어댑터.

PPG 전용 FM. 네이티브 125Hz, 10s = 1250 samples, single-channel. ResNet1DMoE
encoder → 512-d 임베딩(model 출력 tuple 의 [0]).

Upstream: https://github.com/Nokia-Bell-Labs/papagei-foundation-model
가중치   : Zenodo record 13983110 → ``papagei_s.pt`` (weights/ 하위 권장)

빌드/로드는 upstream README 방식과 동일:
    from models.resnet import ResNet1DMoE
    from linearprobing.utils import load_model_without_module_prefix
    model = ResNet1DMoE(in_channels=1, base_filters=32, kernel_size=3, stride=2,
                        groups=1, n_block=18, n_classes=512, n_experts=3)
    model = load_model_without_module_prefix(model, "weights/papagei_s.pt")

정규화: upstream 은 Butterworth bandpass 후 세그먼트화하지만, 여기서는 이식성·배치처리
위해 base 의 per-channel z-norm 을 기본 사용한다(대부분의 probing 파이프라인과 정합).
"""
from __future__ import annotations

import torch

from downstream.baselines.fm.encoders import add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder

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
        from linearprobing.utils import load_model_without_module_prefix  # type: ignore

        model = ResNet1DMoE(in_channels=1, **_MOE_KW)
        model = load_model_without_module_prefix(model, self.weights_path)
        self.model = model
        self.embed_dim = 512  # n_classes

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        emb = out[0] if isinstance(out, (tuple, list)) else out
        return emb  # (b, 512)
