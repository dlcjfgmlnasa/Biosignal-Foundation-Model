# -*- coding: utf-8 -*-
"""ECGFounder (NEJM AI 2025, PKUDigitalHealth) frozen encoder 어댑터.

ECG 전용 FM. 네이티브 500Hz, 10s = 5000 samples, single-lead(in_channels=1) 사용.
임베딩 = Net1D 의 ``deep_features`` (마지막 stage global-avg-pool, 1024-d).

Upstream: https://github.com/PKUDigitalHealth/ECGFounder  (net1d.py)
가중치   : https://huggingface.co/PKUDigitalHealth/ECGFounder
           1-lead → ``1_lead_ECGFounder.pth`` (in_channels=1)
           12-lead→ ``12_lead_ECGFounder.pth`` (in_channels=12)
우리 downstream 은 ECG 를 단일 modality 로 다루므로 **1-lead** 체크포인트를 쓴다.

빌드/로드 방식은 upstream ``finetune_model.ft_1lead_ECGFounder`` 와 동일:
Net1D 생성 → checkpoint['state_dict'] 에서 ``dense.*`` 제외 → load_state_dict(strict=False).
여기서는 분류 head 를 붙이지 않고 return_features 로 임베딩만 추출한다.
"""
from __future__ import annotations

import torch

from downstream.baselines.fm.encoders import add_repo_to_path
from downstream.baselines.fm.encoders.base import FMEncoder

# upstream Net1D 하이퍼파라미터 (finetune_model.py 의 ECGFounder 설정과 일치).
_NET1D_KW = dict(
    base_filters=64,
    ratio=1,
    filter_list=[64, 160, 160, 400, 400, 1024, 1024],
    m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
    kernel_size=16,
    stride=2,
    groups_width=16,
    use_bn=False,
    use_do=False,
)


class ECGFounderEncoder(FMEncoder):
    name = "ecgfounder"
    native_sr = 500.0
    seg_sec = 10.0
    supported_modalities = frozenset({"ecg"})
    max_channels = 1  # single-lead 체크포인트

    def _build_and_load(self) -> None:
        add_repo_to_path(self.third_party_root, "FM_ECGFOUNDER_ROOT")
        from net1d import Net1D  # type: ignore

        model = Net1D(in_channels=1, n_classes=1, return_features=True, **_NET1D_KW)
        ckpt = torch.load(self.weights_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        state = {k: v for k, v in state.items() if not k.startswith("dense.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # dense.* (분류 head) 만 missing 이어야 정상.
        leftover = [m for m in missing if not m.startswith("dense.")]
        if leftover:
            print(f"[ecgfounder] WARN missing (dense 외): {leftover[:8]}{'...' if len(leftover) > 8 else ''}")
        if unexpected:
            print(f"[ecgfounder] WARN unexpected: {list(unexpected)[:8]}{'...' if len(unexpected) > 8 else ''}")
        model.return_features = True
        self.model = model
        self.embed_dim = 1024  # filter_list[-1]

    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        # Net1D(return_features=True) → (logits, deep_features); 임베딩만 사용.
        _, deep_features = self.model(x)
        return deep_features  # (b, 1024)
