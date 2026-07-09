# -*- coding: utf-8 -*-
"""외부 FM baseline encoder registry + upstream import 헬퍼.

각 encoder wrapper 는 공개 레포 코드를 런타임에 import 한다. 레포 경로는
``--third-party-root`` (CLI) 또는 모델별 환경변수로 지정한다:

    ECGFounder : FM_ECGFOUNDER_ROOT   (레포 루트, net1d.py 포함)
    PaPaGei    : FM_PAPAGEI_ROOT      (레포 루트, models/·preprocessing/ 포함)
    Pulse-PPG  : FM_PULSEPPG_ROOT     (레포 루트, pulseppg/ 패키지 포함)
    BIOT       : FM_BIOT_ROOT         (레포 루트, model/biot.py 포함)
    CSFM       : FM_CSFM_ROOT         (레포 루트, network/model.py 포함) + FM_CSFM_VARIANT(tiny|base|large)
    HeartBEiT  : FM_HEARTBEIT_ROOT    (선택, transformers 로컬 로드 시)
    AnyPPG     : FM_ANYPPG_ROOT       (레포 루트, resnet1d.py 포함)
    ST-MEM     : FM_STMEM_ROOT        (레포 루트, models/ 패키지 포함)

``--third-party-root`` 가 주어지면 그 경로를, 아니면 위 환경변수를 sys.path 에 추가한다.
"""
from __future__ import annotations

import importlib
import os
import sys


def add_repo_to_path(third_party_root: str | None, env_var: str, subdir: str | None = None) -> str:
    """upstream 레포 경로를 sys.path 에 추가하고 그 경로를 반환.

    우선순위: 명시 인자(third_party_root) > 환경변수(env_var). subdir 지정 시
    루트 하위 경로도 함께 추가(패키지 내부 상대 import 대응).
    """
    root = third_party_root or os.environ.get(env_var)
    if not root:
        raise RuntimeError(
            f"upstream 레포 경로 미지정 — --third-party-root 또는 ${env_var} 를 설정하세요 "
            f"(README 의 setup 참조)."
        )
    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"레포 경로 없음: {root} (${env_var})")
    for p in ([root, os.path.join(root, subdir)] if subdir else [root]):
        if p not in sys.path:
            sys.path.insert(0, p)
    return root


# name → "module:ClassName" (지연 import 로 미사용 encoder 의존성 회피)
_REGISTRY = {
    "ecgfounder": "downstream.baselines.fm.encoders.ecgfounder:ECGFounderEncoder",
    "papagei": "downstream.baselines.fm.encoders.papagei:PaPaGeiEncoder",
    "pulseppg": "downstream.baselines.fm.encoders.pulseppg:PulsePPGEncoder",
    "biot": "downstream.baselines.fm.encoders.biot:BIOTEncoderWrapper",
    "csfm": "downstream.baselines.fm.encoders.csfm:CSFMEncoder",
    "heartbeit": "downstream.baselines.fm.encoders.heartbeit:HeartBEiTEncoder",
    "anyppg": "downstream.baselines.fm.encoders.anyppg:AnyPPGEncoder",
    "stmem": "downstream.baselines.fm.encoders.stmem:STMEMEncoder",
}

ENCODER_NAMES = tuple(_REGISTRY.keys())


def build_encoder(name: str, **kwargs):
    """registry 에서 encoder 를 지연 import 하여 인스턴스화."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown encoder {name!r}; choices={ENCODER_NAMES}")
    mod_path, cls_name = _REGISTRY[name].split(":")
    cls = getattr(importlib.import_module(mod_path), cls_name)
    return cls(**kwargs)
