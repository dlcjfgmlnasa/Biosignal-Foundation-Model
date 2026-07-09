# -*- coding:utf-8 -*-
"""Patient-level Transformer Aggregator + 윈도우 인코딩 헬퍼.

여러 10분 윈도우의 representation을 [CLS] + self-attention으로 aggregate하여
환자 단위 예측에 사용된다 (Mortality / Sepsis / Cardiac Arrest 등).

구조:
    ICU Stay / Patient
    → K개 윈도우 슬라이딩
    → Foundation Model Encoder (frozen 또는 LoRA) → h_1, ..., h_K  (d_model)
    → [CLS] + h_1..h_K → TransformerAggregator → CLS → Probe → 예측

Public API
----------
- TransformerAggregator: nn.Module — [CLS] + K windows → CLS repr
- mean_pool: (B, N, d), (B, N) → (B, d)
- encode_patient_windows: 한 환자의 K 윈도우 → (K, d)
- collate_patients: 가변 K 패딩 → (B, K_max, d), mask, labels
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from data.collate import PackCollate
from data.dataset import BiosignalSample
from data.spatial_map import SIGNAL_KEY_TO_TYPE, get_global_spatial_id

from downstream._gap_mask import sample_to_patch_mask


DEFAULT_SR = 100.0

# 환자 단위 task에서 공통으로 쓰는 signal_type 매핑.
# v2: data.spatial_map 의 SSOT(SIGNAL_KEY_TO_TYPE) 를 그대로 사용한다.
# 구 로컬 dict 는 0~7 만 있어 "resp"/"resp_impedance"/"resp_flow" 가
# .get(...,0) → ECG 로 조용히 폴백되는 잠재 버그가 있었다(8/9 누락).
SIGNAL_TYPE_INT: dict[str, int] = SIGNAL_KEY_TO_TYPE


def _time_sinusoidal_embedding(
    time_secs: torch.Tensor,  # (B, K)
    d_model: int,
    base: float = 10000.0,
) -> torch.Tensor:  # (B, K, d_model)
    """Continuous-time positional embedding (Vaswani-style on real time).

    time_secs: 각 윈도우의 시작 시각 (초). 동일 환자 내에서 상대시간이면 충분.
    """
    half = d_model // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=time_secs.dtype, device=time_secs.device) / half)
    )  # (half,)
    angles = time_secs.unsqueeze(-1) * inv_freq  # (B, K, half)
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (B, K, 2*half)
    if emb.shape[-1] < d_model:
        # d_model이 홀수일 경우 0으로 pad
        pad = torch.zeros(
            *emb.shape[:-1], d_model - emb.shape[-1],
            dtype=emb.dtype, device=emb.device,
        )
        emb = torch.cat([emb, pad], dim=-1)
    return emb


class TransformerAggregator(nn.Module):
    """시간 순서를 반영하는 Transformer 기반 환자 표현 생성기.

    [CLS] 토큰 + K개 윈도우 representation → self-attention → CLS output.

    Positional embedding 모드 (`pos_mode`):
      - "time": time_secs 기반 sinusoidal (continuous, 갭 인지)
      - "index": K개 학습 가능한 임베딩 (legacy, backward compat)
      - "auto" (default): forward에서 time_secs가 주어지면 time, 아니면 index
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        max_windows: int = 128,
        pos_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.pos_mode = pos_mode
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # Index-based pos_embed은 fallback용 항상 보유 (CLS 포함 max_windows+1)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_windows + 1, d_model) * 0.02
        )
        # CLS 전용 학습 임베딩 (time mode에서 CLS 위치 표시)
        self.cls_pos_embed = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(
        self,
        chunk_reprs: torch.Tensor,  # (B, K, d_model)
        mask: torch.Tensor | None = None,  # (B, K) bool, True=valid
        time_secs: torch.Tensor | None = None,  # (B, K) float, 윈도우 시작 시각(초)
    ) -> torch.Tensor:  # (B, d_model)
        b, k, _ = chunk_reprs.shape

        # [CLS] + chunk representations
        cls = self.cls_token.expand(b, -1, -1)  # (B, 1, D)
        x = torch.cat([cls, chunk_reprs], dim=1)  # (B, K+1, D)

        # Positional embedding 선택
        use_time = (
            self.pos_mode == "time"
            or (self.pos_mode == "auto" and time_secs is not None)
        )
        if use_time:
            assert time_secs is not None, "time_secs required for pos_mode='time'"
            time_pe = _time_sinusoidal_embedding(time_secs, self.d_model)  # (B, K, D)
            cls_pe = self.cls_pos_embed.expand(b, -1, -1)  # (B, 1, D)
            pe = torch.cat([cls_pe, time_pe], dim=1)  # (B, K+1, D)
            x = x + pe
        else:
            x = x + self.pos_embed[:, : k + 1, :]

        # Attention mask: CLS는 항상 valid
        if mask is not None:
            cls_mask = torch.ones(b, 1, dtype=torch.bool, device=mask.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)  # (B, K+1)
            # TransformerEncoder의 src_key_padding_mask는 True=무시
            padding_mask = ~full_mask
        else:
            padding_mask = None

        out = self.encoder(x, src_key_padding_mask=padding_mask)
        return out[:, 0, :]  # CLS token → (B, d_model)


class MeanAggregator(nn.Module):
    """파라미터 없는 masked mean 기반 환자 표현 생성기.

    :class:`TransformerAggregator` 의 drop-in 대체 (동일 forward 시그니처).
    K개 윈도우 representation 을 mask 기준 평균해 (B, d_model) 을 낸다. 학습 가능한
    aggregation 파라미터가 없어 소규모 코호트(예: cardiac arrest ~74명)에서 aggregator
    자체의 과적합을 구조적으로 배제한다. 윈도우 순서/시간은 무시하므로 ``time_secs`` 는
    받되 사용하지 않는다. ``n_heads``/``n_layers``/``max_windows``/``pos_mode`` 등
    TransformerAggregator 전용 인자는 ``**kwargs`` 로 흡수해 무시한다(교체 호환).
    """

    def __init__(self, d_model: int, **kwargs) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        chunk_reprs: torch.Tensor,  # (B, K, d_model)
        mask: torch.Tensor | None = None,  # (B, K) bool, True=valid
        time_secs: torch.Tensor | None = None,  # 무시 (순서 무관)
    ) -> torch.Tensor:  # (B, d_model)
        if mask is None:
            return chunk_reprs.mean(dim=1)
        m = mask.unsqueeze(-1).to(chunk_reprs.dtype)  # (B, K, 1)
        return (chunk_reprs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class LastKAggregator(nn.Module):
    """관측 후반부(event 에 가장 가까운) 마지막 K개 valid window 만 masked-mean.

    :class:`TransformerAggregator` drop-in (동일 forward 시그니처), **파라미터 없음**.
    window 는 시간 순(index 0=가장 이른 관측, 마지막 valid=event 에 가장 가까움)으로
    쌓이므로, "마지막 K개 valid window"는 event 직전 구간에 해당한다. 악화 신호가
    관측 후반에 몰릴 때 전체 mean 의 희석을 피한다. ``last_k <= 0`` 이면 전체 mean 과 동일.
    ``time_secs``·transformer 전용 인자는 무시(교체 호환).
    """

    def __init__(self, d_model: int, last_k: int = 36, **kwargs) -> None:
        super().__init__()
        self.d_model = d_model
        self.last_k = last_k

    def forward(
        self,
        chunk_reprs: torch.Tensor,  # (B, K, d_model)
        mask: torch.Tensor | None = None,  # (B, K) bool, True=valid
        time_secs: torch.Tensor | None = None,  # 무시
    ) -> torch.Tensor:  # (B, d_model)
        b, k, _ = chunk_reprs.shape
        if mask is None:
            mask = torch.ones(b, k, dtype=torch.bool, device=chunk_reprs.device)
        if self.last_k is not None and self.last_k > 0:
            # rev_cum[i] = 위치 i 포함 오른쪽(더 늦은) valid window 개수.
            # 뒤에서부터 last_k 번째까지(=rev_cum<=last_k)만 선택.
            valid_f = mask.to(chunk_reprs.dtype)
            rev_cum = torch.cumsum(valid_f.flip(dims=[1]), dim=1).flip(dims=[1])
            sel = mask & (rev_cum <= self.last_k)
        else:
            sel = mask
        m = sel.unsqueeze(-1).to(chunk_reprs.dtype)
        return (chunk_reprs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def mean_pool(
    encoded: torch.Tensor,  # (B, N, d_model)
    patch_mask: torch.Tensor,  # (B, N)
) -> torch.Tensor:  # (B, d_model)
    """Patch mask 기준 평균 pooling."""
    mask_f = patch_mask.unsqueeze(-1).float()
    return (encoded * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)


def _make_samples_for_window(
    signals: dict[str, torch.Tensor],  # {sig_type: (win_samples,)}
    idx: int,
    session_prefix: str = "patient",
) -> list[BiosignalSample]:
    """한 윈도우의 다채널 신호들을 BiosignalSample 리스트로 변환."""
    samples = []
    for ch, (sig_type, signal) in enumerate(signals.items()):
        stype_int = SIGNAL_TYPE_INT.get(sig_type, 0)
        spatial_id = get_global_spatial_id(stype_int, 0)
        samples.append(
            BiosignalSample(
                values=signal,
                length=len(signal),
                channel_idx=ch,
                recording_idx=idx,
                sampling_rate=DEFAULT_SR,
                n_channels=len(signals),
                win_start=0,
                signal_type=stype_int,
                session_id=f"{session_prefix}_{idx}",
                spatial_id=spatial_id,
            )
        )
    return samples


def _gap_patch_mask_for_window(
    batch,
    gap_masks: dict,  # {sig_type: (K, win_samples) bool}
    idx: int,
    sig_types: list[str],
    patch_size: int,
) -> torch.Tensor | None:
    """packed batch(단일 윈도우)의 gap 을 patch-level (1, N) bool 로 변환.

    `batch.variate_id` 로 각 variate 의 실제 packed 위치를 읽어 sample-level gap 을
    동일 좌표에 채운 뒤 `sample_to_patch_mask` 로 다운샘플한다. per-window encode 는
    start_sample=win_start=0 이라 trim=0 이고 stride=patch_size(non-overlap)이므로
    sample_to_patch_mask 의 patch 그리드가 모델 패치와 정확히 정렬된다.
    gap 이 없으면 None (extra_content_mask 미적용 → 기존 동작과 동일).
    """
    var_ids = batch.variate_id  # (1, T) long, 0=padding, 1..n=variate (packed 순서)
    t = var_ids.shape[1]
    sample_gap = torch.zeros(1, t, dtype=torch.bool)
    # packed 순서 = PackCollate 의 group_samples.sort(key=(signal_type, channel_idx)).
    sorted_sts = sorted(sig_types, key=lambda st: SIGNAL_TYPE_INT.get(st, 0))
    for v in [int(x) for x in torch.unique(var_ids) if int(x) > 0]:
        if v - 1 >= len(sorted_sts):
            continue
        st = sorted_sts[v - 1]
        if st not in gap_masks:
            continue
        pos = var_ids[0] == v  # 해당 variate 가 차지한 연속 구간
        seg_len = int(pos.sum())
        if seg_len == 0:
            continue
        gm = torch.as_tensor(gap_masks[st][idx], dtype=torch.bool)[:seg_len]
        sample_gap[0, pos] = gm
    if not bool(sample_gap.any()):
        return None
    return sample_to_patch_mask(sample_gap, patch_size)  # (1, N)


def encode_patient_windows(
    model,
    patient: dict,
    patch_size: int,
    max_windows: int,
    use_lora: bool = False,
    session_prefix: str = "patient",
    return_time_secs: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]:
    """한 환자의 K개 윈도우를 인코딩하여 (K, d_model) 반환.

    Parameters
    ----------
    model:
        DownstreamModelWrapper — extract_features 또는 batch_to_device + model
    patient:
        {"signals": {sig_type: (K, win_samples)}, "n_windows": K,
         "start_secs": (K,) optional time meta, ...}
    patch_size:
        Encoder의 patch_size (보통 100 또는 200)
    max_windows:
        K가 너무 크면 균등 샘플링으로 축소
    use_lora:
        True이면 gradient 활성 경로 (LoRA fine-tune 시). False이면 no_grad.
    session_prefix:
        BiosignalSample.session_id prefix (task별 구분용; 기본 "patient")
    return_time_secs:
        True이면 (chunk_reprs, time_secs) 튜플 반환. patient에 start_secs 없으면
        time_secs는 None.
    """
    sig_types = list(patient["signals"].keys())
    k = patient["n_windows"]

    # max_windows 제한 (균등 샘플링)
    if k > max_windows:
        indices = np.linspace(0, k - 1, max_windows, dtype=int)
    else:
        indices = np.arange(k)

    multi = len(sig_types) > 1
    collate_mode = "any_variate" if multi else "ci"
    win_samples = patient["signals"][sig_types[0]].shape[1]

    # 한 윈도우의 모든 variate가 같은 row 에 packing 되어야 multi-modal repr 이 된다.
    # max_length 를 단일 variate 길이로 잡으면 첫 variate(최저 signal_type)가 row 를
    # 가득 채워 offset==max_length 에서 break → 나머지 채널이 통째로 누락된다.
    # variate 수만큼 확보해 모든 채널이 한 row 에 들어가게 한다 (ci 모드는 ×1 = 동일).
    pack_max_length = win_samples * len(sig_types)

    collate = PackCollate(
        max_length=pack_max_length, collate_mode=collate_mode, patch_size=patch_size
    )

    # gap_masks 가 저장돼 있으면 0-fill 된 gap 구간을 patch-level [MASK] 로 교체
    # (사전학습 mask_token mechanism 재사용). 없거나 gap 0 이면 기존 동작.
    gap_masks = patient.get("gap_masks")
    has_gap = isinstance(gap_masks, dict) and len(gap_masks) > 0
    device = getattr(model, "device", None)

    grad_ctx = torch.enable_grad() if use_lora else torch.no_grad()
    chunk_reprs = []
    with grad_ctx:
        for idx in indices:
            win_signals = {st: patient["signals"][st][idx] for st in sig_types}
            samples = _make_samples_for_window(win_signals, idx, session_prefix)
            batch = collate(samples)

            gap_patch = None
            if has_gap:
                gap_patch = _gap_patch_mask_for_window(
                    batch, gap_masks, int(idx), sig_types, patch_size
                )
                if gap_patch is not None and device is not None:
                    gap_patch = gap_patch.to(device)

            if use_lora:
                batch = model.batch_to_device(batch)
                out = model.model(batch, task="masked", extra_content_mask=gap_patch)
                feat = mean_pool(out["encoded"], out["patch_mask"])
            else:
                feat = model.extract_features(
                    batch, pool="mean", gap_mask_patch=gap_patch
                )

            chunk_reprs.append(feat)  # (1, d_model)

    out_reprs = torch.cat(chunk_reprs, dim=0)  # (K', d_model)
    if not return_time_secs:
        return out_reprs

    start_secs_full = patient.get("start_secs")
    if start_secs_full is None:
        return out_reprs, None
    if isinstance(start_secs_full, torch.Tensor):
        start_secs_full = start_secs_full.float()
    else:
        start_secs_full = torch.as_tensor(start_secs_full, dtype=torch.float32)
    sub = start_secs_full[indices]  # (K',)
    return out_reprs, sub


def collate_patients(
    patient_reprs: list[torch.Tensor],  # [(K_i, d_model), ...]
    labels: list[int],
    device: torch.device,
    time_secs: list[torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """가변 K를 패딩하여 배치를 만든다.

    Parameters
    ----------
    time_secs : list of (K_i,) tensor or None per patient. None이면 시간 정보 미제공.

    Returns
    -------
    (padded_reprs (B, K_max, d), mask (B, K_max), labels (B,),
     time_secs (B, K_max) or None)
    """
    k_max = max(r.shape[0] for r in patient_reprs)
    d_model = patient_reprs[0].shape[1]
    b = len(patient_reprs)

    padded = torch.zeros(b, k_max, d_model, device=device)
    mask = torch.zeros(b, k_max, dtype=torch.bool, device=device)

    for i, r in enumerate(patient_reprs):
        k_i = r.shape[0]
        padded[i, :k_i] = r.to(device)
        mask[i, :k_i] = True

    labels_t = torch.tensor(labels, dtype=torch.float32, device=device)

    times_t: torch.Tensor | None = None
    if time_secs is not None and any(t is not None for t in time_secs):
        times_t = torch.zeros(b, k_max, dtype=torch.float32, device=device)
        for i, t in enumerate(time_secs):
            if t is None:
                continue
            k_i = t.shape[0]
            times_t[i, :k_i] = t.to(device)

    return padded, mask, labels_t, times_t
