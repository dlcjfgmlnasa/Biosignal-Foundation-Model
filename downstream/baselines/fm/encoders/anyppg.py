# -*- coding: utf-8 -*-
"""AnyPPG (arXiv 2511.01747, PKUDigitalHealth) frozen encoder 어댑터.

PPG 전용 FM (ECG-guided 사전학습, 100k+ 시간). ECGFounder 와 같은 랩·같은 `Net1D`
계열이라 통합이 가장 단순하다. 네이티브 125Hz, 10s = 1250 samples, single-channel,
z-score 정규화 → 512-d 임베딩.

Upstream: https://github.com/PKUDigitalHealth/AnyPPG  (resnet1d.Net1D)
가중치   : 레포 내 공개 ``load_anyppg/anyppg_ckpt.pth`` (별도 신청 불필요).

빌드/로드는 upstream demo 와 동일한 Net1D config 로 생성 후 state_dict 로드.
demo 기준 ``model(x)`` 가 (b, 512) 임베딩을 직접 반환한다.
"""
from __future__ import annotations

import torch

from downstream.baselines.fm.encoders import add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder

# upstream demo 의 Net1D config (AnyPPG README).
_NET1D_KW = dict(
    in_channels=1,
    base_filters=64,
    ratio=1.0,
    filter_list=[64, 160, 160, 400, 400, 512],
    m_blocks_list=[2, 2, 2, 3, 3, 1],
    kernel_size=3,
    stride=2,
    groups_width=16,
    use_bn=True,
    use_do=True,
    verbose=False,
)


class AnyPPGEncoder(FMEncoder):
    name = "anyppg"
    native_sr = 125.0
    seg_sec = 10.0
    supported_modalities = frozenset({"ppg"})
    max_channels = 1

    def _build_and_load(self) -> None:
        add_repo_to_path(self.third_party_root, "FM_ANYPPG_ROOT")
        from resnet1d import Net1D  # type: ignore

        model = Net1D(**_NET1D_KW)
        ckpt = torch.load(self.weights_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[anyppg] WARN missing: {list(missing)[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            print(f"[anyppg] WARN unexpected: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        self.model = model
        self.embed_dim = 512  # filter_list[-1]

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # demo: model(x) → (b, 512). return_features 형태(tuple)면 feature 쪽 사용.
        if isinstance(out, (tuple, list)):
            return out[-1]
        return out
