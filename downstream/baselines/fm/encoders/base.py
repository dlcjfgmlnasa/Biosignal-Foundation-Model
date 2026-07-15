# -*- coding: utf-8 -*-
"""외부 사전학습 Foundation Model(FM) baseline 용 공통 encoder 인터페이스.

CARMEN 과 **같은 prepared 데이터·같은 5-fold·같은 평가** 위에서, 공개된 외부 FM
(BIOT / ECGFounder / PaPaGei / Pulse-PPG / ST-MEM / AnyPPG) 을 **frozen feature extractor**
로 붙여 linear-probe 로 비교하기 위한 어댑터 계층이다.

핵심 문제: 우리 prepared window 는 **100Hz, (n, C, L)** 인데 각 FM 은 네이티브 SR·입력
길이·모달리티가 제각각이다(ECGFounder 500Hz·10s·ECG, PaPaGei 125Hz·10s·PPG,
Pulse-PPG 50Hz·PPG, BIOT 200Hz·다채널). :class:`FMEncoder` 가 아래를 공통 처리한다.

1. **채널 선택** — 모델이 지원하는 modality 채널만 골라낸다(ECG 전용/PPG 전용/다채널).
2. **세그먼트 분할** — 긴 window 를 모델 네이티브 입력 길이(초) 단위로 잘라
   (최대 ``max_segments`` 개, 균등 샘플링) 각 세그먼트를 인코딩 후 **평균**한다
   (PaPaGei 논문의 segment-then-aggregate 와 동일한 관행).
3. **리샘플링** — 소스 100Hz → 모델 네이티브 SR (선형 보간, ECGFounder 관행과 동일).
4. **정규화** — 세그먼트별 per-channel z-norm(기본). 모델별 override 가능.

⚠ 이 계층은 downstream 평가 전용이며 CARMEN 본코드(module/model/train/collate)를
   전혀 건드리지 않는다. upstream 레포·가중치는 런타임에 import/load 하며, 부재 시
   명확한 에러로 안내한다(README 참조).
"""
from __future__ import annotations

import abc

import numpy as np
import torch
import torch.nn.functional as F

# 우리 prepared window 의 고정 SR (downstream/window_task.py TARGET_SR 와 동일).
SOURCE_SR = 100.0


class FMEncoder(abc.ABC):
    """Frozen 외부 FM encoder 의 공통 base.

    서브클래스는 :meth:`_build_and_load` (모델 생성 + 가중치 로드, ``self.model``·
    ``self.embed_dim`` 설정) 와 :meth:`_forward_native` (네이티브 텐서 → (b, d) 임베딩)
    를 구현한다. 나머지(채널선택·세그먼트·리샘플·정규화)는 base 가 처리한다.

    클래스 속성:
        name:                 encoder 식별자 (CLI --encoder 값).
        native_sr:            모델 네이티브 sampling rate (Hz).
        seg_sec:              모델 네이티브 입력 세그먼트 길이 (초).
        supported_modalities: 이 모델이 다루는 modality key 집합
                              (예: {"ecg"}, {"ppg"}). ``None`` 이면 제공된 모든 채널
                              을 함께 사용(다채널, BIOT).
        max_channels:         선택할 최대 채널 수 (단일 lead 모델은 1). ``None`` = 무제한.
    """

    name: str = "base"
    native_sr: float = 100.0
    seg_sec: float = 10.0
    supported_modalities: frozenset[str] | None = None
    max_channels: int | None = None

    def __init__(
        self,
        weights_path: str,
        device: str | torch.device = "cuda",
        max_segments: int = 8,
        third_party_root: str | None = None,
    ) -> None:
        self.weights_path = weights_path
        self.device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
        self.max_segments = int(max_segments)
        self.third_party_root = third_party_root
        self.model: torch.nn.Module | None = None
        self.embed_dim: int = 0
        self._build_and_load()
        if self.model is not None:
            self.model.to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad_(False)
        if not self.embed_dim:
            self.embed_dim = self._infer_embed_dim()

    # ── 서브클래스 구현부 ────────────────────────────────────────
    @abc.abstractmethod
    def _build_and_load(self) -> None:
        """upstream 모델 생성 + 가중치 로드 → ``self.model``, ``self.embed_dim`` 설정."""

    @abc.abstractmethod
    def _forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """네이티브 세그먼트 (b, C, seg_native) → 임베딩 (b, embed_dim).

        입력은 이미 device 위·모델 네이티브 SR·정규화 완료 상태다.
        """

    # ── 공통 전처리 ─────────────────────────────────────────────
    def _select(self, x: torch.Tensor, keys: list[str]) -> torch.Tensor:
        """(b, C, L) 에서 이 모델이 지원하는 modality 채널만 선택.

        supported_modalities=None → 전 채널(다채널). 그 외 → 매칭 채널 중 앞
        ``max_channels`` 개. 매칭 채널이 없으면 명확한 에러(=이 task 는 이 baseline
        평가 불가, README 의 modality coverage matrix 참조).
        """
        if self.supported_modalities is None:
            return x
        idxs = [i for i, k in enumerate(keys) if k in self.supported_modalities]
        if not idxs:
            raise ValueError(
                f"[{self.name}] 지원 modality {set(self.supported_modalities)} 중 "
                f"입력 신호 {keys} 에 없음 → 이 task 는 {self.name} 로 평가 불가."
            )
        if self.max_channels is not None:
            idxs = idxs[: self.max_channels]
        return x[:, idxs, :].contiguous()

    def _segment(self, x: torch.Tensor, seg_src: int) -> list[torch.Tensor]:
        """(b, C, L) → 세그먼트 리스트, 각 (b, C, seg_src). 소스 SR 기준.

        L < seg_src 면 zero-pad 한 1개. ``max_segments <= 0`` (또는 full 세그먼트 수 이상)
        이면 **겹침 없이 전 구간**을 연속 세그먼트로 커버한다(권장 — CARMEN 은 window 전체를
        pool 하므로 부분 커버는 외부 FM 을 불리하게 만든다). full 세그먼트 수가 양의
        max_segments 를 초과할 때만 [0, L-seg_src] 구간에서 균등 시작점을 골라 그만큼만 쓴다.
        """
        L = x.shape[-1]
        if L < seg_src:
            return [F.pad(x, (0, seg_src - L))]
        n_full = L // seg_src
        if self.max_segments <= 0 or n_full <= self.max_segments:
            starts = [i * seg_src for i in range(n_full)]
        else:
            starts = np.linspace(0, L - seg_src, self.max_segments).round().astype(int).tolist()
        return [x[..., s : s + seg_src].contiguous() for s in starts]

    def _resample(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        """소스 세그먼트 (b, C, seg_src) → (b, C, target_len) 선형보간 리샘플."""
        if x.shape[-1] == target_len:
            return x
        return F.interpolate(x, size=target_len, mode="linear", align_corners=False)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """네이티브 세그먼트 정규화 (기본: per-channel z-norm). 모델별 override 가능."""
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mu) / sd

    def _infer_embed_dim(self) -> int:
        """네이티브 zero-세그먼트 1개를 forward 해 임베딩 차원을 추론(embed_dim 미설정 시)."""
        seg_native = int(round(self.seg_sec * self.native_sr))
        c = 1 if (self.supported_modalities is not None) else 1
        dummy = torch.zeros(1, c, seg_native, device=self.device)
        with torch.no_grad():
            out = self._forward_native(self._preprocess(dummy))
        return int(out.shape[-1])

    # ── 공개 API (runner 가 호출) ───────────────────────────────
    @torch.no_grad()
    def encode(self, x: torch.Tensor, keys: list[str]) -> torch.Tensor:
        """prepared window 배치 (b, C, L) @100Hz → 임베딩 (b, embed_dim).

        채널선택 → 세그먼트 분할 → (세그먼트별) 리샘플·정규화·forward → 세그먼트 평균.
        입력은 CPU/GPU 어느쪽이든 되며 float 로 캐스팅해 처리한다.
        """
        x = x.to(torch.float32)
        x = self._select(x, keys)  # (b, Cs, L)
        seg_src = int(round(self.seg_sec * SOURCE_SR))
        seg_native = int(round(self.seg_sec * self.native_sr))
        segs = self._segment(x, seg_src)
        embs = []
        for s in segs:
            s = self._resample(s, seg_native)
            s = self._preprocess(s)
            e = self._forward_native(s.to(self.device))  # (b, d)
            embs.append(e.float())
        return torch.stack(embs, dim=0).mean(dim=0)  # (b, d)
