# -*- coding:utf-8 -*-
"""Biosignal Foundation Model.

Scaler → PatchEmbedding → SpatialEmbedding → TransformerEncoder → Head 파이프라인.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from functools import partial

import torch
from torch import nn

from data.collate import PackedBatch
from loss.masked_mse_loss import create_patch_mask
from model._config import ModelConfig
from module.packed_scaler import PackedStdScaler, PackedScaler
from module.patch import PatchEmbedding
from module.position import (
    BinaryAttentionBias,
    QueryKeyProjection,
    RotaryProjection,
)
from module.transformer import TransformerEncoder


class BlockNextHead(nn.Module):
    """Shared trunk + K horizon-specific heads for Block Next Prediction.

    각 position의 encoded vector를 공유 non-linear trunk로 변환한 뒤,
    K개 독립 Linear head가 horizon별 미래 patch를 예측한다.

    입력:  ``(B, N, d_model)``
    출력:  ``(B, N, K, patch_size)`` — k번째 head가 t+k 패치 예측

    Parameters
    ----------
    d_model:
        입력 차원.
    patch_size:
        출력 patch 크기 (샘플 수).
    block_size:
        K — 예측할 future patch 수.
    d_inner:
        trunk 내부 차원. ``None``이면 ``d_model``.
    """

    def __init__(
        self,
        d_model: int,
        patch_size: int,
        block_size: int,
        d_inner: int | None = None,
    ) -> None:
        super().__init__()
        d_inner = d_inner if d_inner is not None else d_model
        self.block_size = block_size
        self.patch_size = patch_size

        self.trunk = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(),
        )
        self.heads = nn.ModuleList(
            [nn.Linear(d_inner, patch_size) for _ in range(block_size)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_model)
        h = self.trunk(x)  # (B, N, d_inner)
        outs = [head(h) for head in self.heads]  # list of (B, N, patch_size)
        return torch.stack(outs, dim=2)  # (B, N, K, patch_size)


class BiosignalFoundationModel(nn.Module):
    """생체신호 파운데이션 모델 — 모든 신호를 raw patch reconstruction.

    모든 signal type에 대해 동일하게 raw patch 복원을 수행한다.
    ``_encode()``로 공통 인코딩 파이프라인(Scaler → Patchify → Project →
    SpatialEmbed → LocScale → Encoder)을 분리하여 서브클래스에서 확장 가능.

    Parameters
    ----------
    d_model:
        트랜스포머 임베딩 차원.
    num_layers:
        트랜스포머 인코더 레이어 수.
    patch_size:
        패치 크기 (time-step 수).
    stride:
        패치 보폭 (overlapping 시). ``None``이면 ``patch_size``와 동일.
    num_heads:
        어텐션 헤드 수. ``None``이면 ``d_model // 64``.
    num_groups:
        GQA 그룹 수. ``None``이면 ``num_heads`` (MHA).
    use_glu:
        Gated Linear Unit FFN 사용 여부.
    use_moe:
        Mixture of Experts 사용 여부.
    use_rope:
        Rotary Position Embedding 사용 여부.
    use_var_attn_bias:
        BinaryAttentionBias (variate 간 bias) 사용 여부.
    scaler:
        입력 정규화 스케일러. ``None``이면 ``PackedStdScaler``.
    dropout_p:
        드롭아웃 확률.
    num_signal_types:
        신호 타입(modality) 수. v2: 9 (2026-06-23 PAP 제거 후 연속 번호)
        (ecg=0, abp=1, ppg=2, cvp=3, co2=4, awp=5, icp=6,
        resp_impedance=7, resp_flow=8).
    use_spatial_embed:
        단일 modality(signal_type) 임베딩 사용 여부.
        (이름은 하위 호환을 위해 유지 — v2에서 의미를 "modality embedding"으로
        재정의. spatial_id 소분류 임베딩은 폐지됨.)
    next_block_size:
        Block Next Prediction에서 각 position이 병렬 예측하는 future patch 수 (K).
        각 position n에서 encoded_causal[n]으로부터 n+1, n+2, ..., n+K 시점의 raw patch를
        non-autoregressive하게 동시 예측한다.
    """

    def __init__(
        self,
        d_model: int,
        num_layers: int,
        patch_size: int,
        stride: int | None = None,
        num_heads: int | None = None,
        num_groups: int | None = None,
        use_glu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        num_experts_per_token: int = 2,
        use_rope: bool = True,
        use_var_attn_bias: bool = True,
        scaler: PackedScaler | None = None,
        dropout_p: float = 0.0,
        # v2 단일 modality embedding: ECG0,ABP1,PPG2,CVP3,CO24,AWP5,ICP6,
        # RESP_Impedance7, RESP_Flow8 (2026-06-23 PAP 제거 후 9종 연속).
        num_signal_types: int = 9,
        use_spatial_embed: bool = True,
        next_block_size: int = 4,
        next_head_d_inner: int | None = None,
        dual_cross_head: bool = False,
        d_cond: int = 16,
        patch_local_norm: bool = False,   # 안 A: scaler 를 패치 단위로
        cond_local_trend: bool = False,   # 안 B: 조건에 패치 국소 편차 추가
        # none|dev|lag|deriv|multi|patchls|patchstat|patchstat_abs|patchabs
        # patchabs 는 창 loc/scale 을 패치 절대 통계로 대체(2열).
        cond_trend_mode: str = "none",
        gate_absolute_only: bool = False,  # 게이팅을 절대항(창 loc/scale)에만 적용
        # masked 경로에서 마스킹된 패치의 패치단위 통계를 가린다 (정답 누출 차단).
        mask_cond_trend: bool = True,

        use_lscnorm: bool = True,
        gate_unitless_cond: bool = False,
        enrich_cond_peak: bool = False,
        gated_cond_signal_types: list[int] | tuple[int, ...] | None = None,
        cond_transform: str = "none",
        cond_dropout_prob: float = 0.0,
        cond_dropout_signal_types: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.num_signal_types = num_signal_types
        # d_cond: AdaLN modulation 입력 차원 (override 가능 hyperparameter).
        self.d_cond = d_cond
        # 창 단위 정규화는 조건 경로에 시간 해상도를 주지 않는다(2026-09-19 확인).
        # A: 값 자체를 패치 단위로 정규화 / B: 값은 그대로 두고 조건에만 국소 편차 추가.
        self.patch_local_norm = patch_local_norm
        self.cond_local_trend = cond_local_trend
        # 조건 벡터 설계 sweep. cond_local_trend=True 는 mode='dev' 와 동일(하위호환).
        self.cond_trend_mode = ("dev" if cond_local_trend else cond_trend_mode)
        # 부분 게이팅. PPG·ECG 의 **절대** loc/scale 은 기기 AGC 지문이라 막아야
        # 하지만, 창 scale 로 나눈 **상대** 패치 통계는 이득이 약분되어 불변이다
        # (실측: gain ×0.21~×3.7 에서 소수점 4자리까지 동일). 통째로 막으면
        # 생리 정보(PPG 박동 진폭의 호흡성 변이 = PPV/SVV 의 실체)까지 버린다.
        self.gate_absolute_only = gate_absolute_only
        self.mask_cond_trend = mask_cond_trend
        # Conditioning redesign (2026-07-15): A=게이팅, B=peak enrichment (기본 off,
        # config로 on).
        self.gate_unitless_cond = gate_unitless_cond
        self.enrich_cond_peak = enrich_cond_peak
        # loc/scale 이 물리적 영점에 추적되지 않는 modality (게이팅 대상):
        #   ECG=0, PPG=2, RESP_Imp=7.
        # PPG·RESP_Imp 는 unitless (device gain 이 진폭을 임의로 결정).
        # ECG 는 명목상 mV 이나 (a) highpass 로 patch mean 이 항등적 0 → loc 정보량 없음,
        # (b) 진폭은 환자별 상수에 가까워 (within/between rCV 비 0.065) 생리가 아닌 지문으로
        # 작동한다. 무차원 형태((max-min)/std, max/|min|) 로도 환자 내 변동이 없어 기각.
        # 근거: SNUH OR raw probe 2026-08-03 (모달리티당 60-644명, shard 5개).
        # 비게이팅 6종 = ABP·CVP·ICP·CO2·AWP·RespFlow (모두 물리적 영점 기준 절대값).
        #
        # 2026-09-08: modality 별 분해 실험을 위해 config 로 override 가능하게 뺐다.
        # (ECG·PPG 를 한꺼번에 끄면 어느 쪽이 기여했는지 알 수 없다.)
        # None 이면 기존 기본값 (0, 2, 7) 유지 — 구 config/checkpoint 호환.
        self._gated_signal_types = (
            tuple(gated_cond_signal_types)
            if gated_cond_signal_types is not None
            else (0, 2, 7)
        )

        # cond 입력 변환 (_transform_cond 참조). permod 용 modality 별 scale median 은
        # VitalDB 5,115 case 실측값 (2026-09-08, modality 당 27만~179만 표본).
        # 코퍼스에 없던 ICP·RESP_Imp·RESP_Flow 는 1.0 (= 변환 없음) 으로 둔다.
        self.cond_transform = cond_transform
        self.cond_dropout_prob = float(cond_dropout_prob)
        # D': dropout 적용 modality 제한 (None = 전부). calibrated modality 보호용.
        self._cond_dropout_signal_types = (
            tuple(cond_dropout_signal_types)
            if cond_dropout_signal_types is not None
            else None
        )
        self.register_buffer(
            "_cond_scale_median",
            torch.tensor(
                [0.221, 12.133, 12.389, 3.340, 12.949, 4.945, 1.0, 1.0, 1.0]
            ),
            persistent=False,
        )

        # 1. Scaler (point-level)
        # 안 A: patch_local_norm 이면 scaler 그룹 키에 패치 인덱스를 포함시킨다.
        # patch_size(=stride 아님) 기준 — PatchEmbedding 이 stride 로 자르므로
        # overlapping 이면 근사지만, 현 설정은 stride == patch_size 다.
        self.scaler = scaler or PackedStdScaler(
            patch_len=(patch_size if patch_local_norm else 0))

        # 2. Patch Embedding
        self.patch_embed = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model,
            stride=stride,
        )
        # RoPE position interpolation 토글 (overlapping-stride 추론 전용, 기본 on).
        # stride==patch_size(학습/비중첩)에선 무의미(PI 분기 자체가 비활성).
        # ablation(naive-overlap vs PI-overlap)에서 wrapper가 False로 끌 수 있다.
        self.rope_pi = True

        # 3. Transformer Encoder
        num_heads = num_heads or d_model // 64

        var_attn_bias_layer: Callable | None = None
        if use_var_attn_bias:
            var_attn_bias_layer = partial(BinaryAttentionBias)

        time_qk_proj_layer: Callable | None = None
        if use_rope:
            time_qk_proj_layer = partial(
                QueryKeyProjection,
                proj_layer=partial(RotaryProjection),
            )

        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_groups=num_groups,
            use_glu=use_glu,
            use_moe=use_moe,
            num_experts=num_experts,
            num_experts_per_token=num_experts_per_token,
            var_attn_bias_layer=var_attn_bias_layer,
            time_qk_proj_layer=time_qk_proj_layer,
            dropout_p=dropout_p,
            d_cond=self.d_cond,
        )

        # 4. Modality Embedding (v2: 단일 signal_type 임베딩)
        # spatial_id 소분류 임베딩은 폐지 — modality(signal_type) 단위 단일 임베딩만 사용.
        # (use_spatial_embed 이름은 하위 호환을 위해 유지, 의미는 modality embedding으로 재정의.)
        self.use_spatial_embed = use_spatial_embed
        if use_spatial_embed:
            self.signal_type_embed = nn.Embedding(num_signal_types, d_model)

        # 5. Loc/Scale AdaLN Conditioning (환자별 절대 레벨 정보 보존)
        # (loc, scale) 2D scalar → d_cond conditioning vector → encoder 모든
        # layer의
        # LSCNorm modulation 입력. MLP(2 → d_cond → d_cond) — non-linearity로
        # expressiveness 확보.
        # B(enrich_cond_peak): 입력이 [loc,scale](2) 또는 [loc,scale,max,min](4).
        cond_in_dim = 4 if enrich_cond_peak else 2
        # 추세 항 차원. 대부분 dev=(패치 국소평균 − 창 loc)/창 scale 에서 파생된다.
        _mode = "dev" if cond_local_trend else cond_trend_mode
        if _mode in ("patchabs", "patchabs_rel"):
            # 창 통계를 **대체**하는 유일한 모드. 총 2열([패치평균, 패치std] 물리단위).
            cond_in_dim = 2
        else:
            cond_in_dim += {"none": 0, "dev": 1, "lag": 6, "deriv": 4,
                            "multi": 5, "patchls": 2, "patchstat": 4,
                            "patchstat_abs": 4, "abs_res": 2}[_mode]
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in_dim, self.d_cond),
            nn.SiLU(),
            nn.Linear(self.d_cond, self.d_cond),
        )

        # 6. Reconstruction Head (자기 variate 복원)
        self.head = nn.Linear(d_model, patch_size)

        # 7. Block Next-Patch Prediction Head (공유 trunk + K개 horizon-specific
        # head)
        # - trunk: 모든 horizon 공통 non-linear 변환 (Linear+GELU)
        # - heads: K개 독립 Linear projection (각 horizon 전용)
        # 한 Linear(d, K*P)보다 non-linearity + horizon 분업으로 장거리 예측 품질 향상.
        self.next_block_size = next_block_size
        self.next_head = BlockNextHead(
            d_model=d_model,
            patch_size=patch_size,
            block_size=next_block_size,
            d_inner=next_head_d_inner,
        )

        # 8. Cross-Modal Prediction Heads (target signal type별 독립 Linear)
        self.cross_heads = nn.ModuleDict(
            {
                str(st): nn.Linear(d_model, patch_size)
                for st in range(num_signal_types)
            }
        )

        # 8b. 예측형(t→t+1) cross-modal head. 동시점 head 와 출력을 공유하면
        # 같은 텐서가 t 타깃과 t+1 타깃을 동시에 맞춰야 해서 타협값만 학습된다.
        # dual_cross_head=False 면 생성하지 않고 forward 도 기존과 동일하다.
        self.dual_cross_head = bool(dual_cross_head)
        if self.dual_cross_head:
            self.cross_next_heads = nn.ModuleDict(
                {
                    str(st): nn.Linear(d_model, patch_size)
                    for st in range(num_signal_types)
                }
            )

        # 10. Learnable [MASK] Token
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 11. (Ablation) LSCNorm 비활성화 — cond_proj + 모든 LSCNorm.modulation 을
        # zero-freeze 하여 plain RMSNorm 과 forward 동등 (gamma=0, beta=0 고정).
        # 모델 구조는 그대로 두고 파라미터만 동결하여 checkpoint 호환성 유지.
        self.use_lscnorm = use_lscnorm
        if not use_lscnorm:
            self._disable_lscnorm_modulation()

    def _disable_lscnorm_modulation(self) -> None:
        """Ablation: cond_proj 와 모든 LSCNorm.modulation 을 0 으로 고정.

        결과: encoder LSCNorm 출력 = norm(x) * (1+0) + 0 = norm(x) = plain RMSNorm.
        """
        from module.norm import LSCNorm

        # cond_proj 출력을 항상 0 으로 (Linear(0)=bias=0, SiLU(0)=0, Linear(0)=bias=0)
        for p in self.cond_proj.parameters():
            p.data.zero_()
            p.requires_grad = False

        # 모든 LSCNorm.modulation 을 0 으로 freeze
        for m in self.modules():
            if isinstance(m, LSCNorm):
                m.modulation.weight.data.zero_()
                m.modulation.bias.data.zero_()
                m.modulation.weight.requires_grad = False
                m.modulation.bias.requires_grad = False

    @staticmethod
    def _sample_variate_drop(
        p_sid: torch.Tensor,  # (B, N)
        p_vid: torch.Tensor,  # (B, N)
        patch_mask: torch.Tensor,  # (B, N)
        drop_prob: float,
    ) -> torch.Tensor | None:
        """Complete Variate Dropout: **packing unit 별로** 하나의 variate를
        attention에서 제거.

        Returns (B, N) bool mask — True = dropped from attention.
        다변량(2+ variates)인 unit 에서만 작동. None if no drop.

        ⚠️ ``variate_id`` 는 packing unit 마다 1 부터 재시작하므로 반드시 ``p_sid`` 로
        스코핑해야 한다. 행 전체에서 ``p_vid == chosen`` 을 비교하면 같은 행에 packing 된
        **서로 다른 환자의 k번째 variate 가 함께 제거**된다.
        """
        b, n = p_vid.shape
        drop_mask = torch.zeros(b, n, dtype=torch.bool, device=p_vid.device)
        any_dropped = False
        for bi in range(b):
            valid_idx = patch_mask[bi].nonzero(as_tuple=True)[0]
            if len(valid_idx) == 0:
                continue
            row_sids = p_sid[bi, valid_idx]
            for uid in row_sids.unique():
                if torch.rand(1).item() >= drop_prob:
                    continue
                u_idx = valid_idx[row_sids == uid]
                u_vids = p_vid[bi, u_idx]
                unique_vids = u_vids[u_vids > 0].unique()
                if len(unique_vids) < 2:
                    continue  # 단일 variate unit → dropout 불가
                # 랜덤으로 하나 선택
                chosen = unique_vids[
                    torch.randint(len(unique_vids), (1,)).item()
                ]
                drop_mask[bi, u_idx[u_vids == chosen]] = True
                any_dropped = True
        return drop_mask if any_dropped else None

    def _transform_cond(
        self,
        # (B, N, C) — C=2 [loc,scale] 또는 4 [loc,scale,max,min]
        cond_stats: torch.Tensor,
        patch_signal_types: torch.Tensor | None,  # (B, N)
    ) -> torch.Tensor:
        """cond 입력의 modality 간 동적범위를 줄인다.

        ``none``     : 원값 (기존 동작).
        ``logscale`` : scale 계열만 ``log1p`` — 58.6배 → 4.1 차이. loc 은 그대로.
        ``logboth``  : loc 은 signed-log, scale 계열은 ``log1p``.
        ``permod``   : scale 계열을 modality 별 median 으로 나눠 ≈1 로 맞춘다.
                       log 과 달리 modality **내부** 변동을 선형으로 보존한다.
        """
        mode = self.cond_transform
        if mode == "none":
            return cond_stats

        loc = cond_stats[..., :1]
        rest = cond_stats[..., 1:]  # scale (+ max, min)

        if mode == "logscale":
            rest = torch.log1p(rest.clamp(min=0.0))
        elif mode == "logboth":
            loc = torch.sign(loc) * torch.log1p(loc.abs())
            rest = torch.log1p(rest.clamp(min=0.0))
        elif mode == "permod":
            if patch_signal_types is not None:
                med = self._cond_scale_median[patch_signal_types].unsqueeze(
                    -1
                )  # (B,N,1)
                rest = rest / med.clamp(min=1e-6)
        else:
            raise ValueError(f"unknown cond_transform: {mode}")
        return torch.cat([loc, rest], dim=-1)

    @classmethod
    def from_config(cls, config: ModelConfig) -> BiosignalFoundationModel:
        """ModelConfig로부터 모델 인스턴스를 생성한다."""
        import inspect

        valid_params = set(
            inspect.signature(cls.__init__).parameters.keys()
        ) - {"self"}
        kwargs = {
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.name in valid_params
        }
        # 코드 드리프트 방어(2026-09-19): 예전에 노드 코드가 낡아 conditioning 키 4개를
        # **조용히** 버렸고, state_dict 는 0 missing/0 unexpected 로 로드되어 경고조차
        # 없었다. 기본값과 다른 값이 버려지는 경우에만 즉시 실패시킨다 —
        # 기본값 그대로인 신규 키는 구 코드로도 재현이 되므로 통과시킨다.
        _dropped = {
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.name not in valid_params
            and getattr(config, f.name) != f.default
        }
        if _dropped:
            raise ValueError(
                "ModelConfig 키가 모델 signature 에 없어 무시됨 — 코드 버전 불일치: "
                f"{_dropped}"
            )
        return cls(**kwargs)

    # ── Encode Pipeline ────────────────────────────────────────────

    def _encode(
        self,
        batch: PackedBatch,
        task: str = "masked",
        mask_ratio: float = 0.0,
        block_mask: bool = False,
        block_size_min: int = 3,
        block_size_max: int = 8,
        variate_mask_prob: float = 0.0,
        variate_drop_prob: float = 0.0,
        extra_content_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """공통 인코딩 파이프라인: Scaler → Patchify → Project → SpatialEmbed → LocScale
        → Encoder.

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        task:
            ``"masked"``: 양방향 attention.
            ``"next_pred"``: causal attention.
            ``"both"``: 양방향 + causal 동시 (encoder 2회 호출, DDP single forward 호환).

        Returns
        -------
        dict with keys:
            ``encoded``: ``(B, N, d_model)`` — 양방향 인코딩된 패치 표현
            (task="both"/"masked").
            ``encoded_causal``: ``(B, N, d_model)`` — causal 인코딩 (task="both"일
            때만).
            ``patches``: ``(B, N, patch_size)`` — raw patches.
            ``patch_signal_types``: ``(B, N)`` — 패치별 signal type.
            ``patch_spatial_ids``: ``(B, N)`` — 패치별 spatial ID.
            ``loc``: ``(B, L, 1)`` — per-variate 위치.
            ``scale``: ``(B, L, 1)`` — per-variate 스케일.
            ``patch_mask``: ``(B, N)`` — 유효 패치 마스크.
            ``patch_sample_id``: ``(B, N)`` — 패치별 sample_id.
            ``patch_variate_id``: ``(B, N)`` — 패치별 variate_id.
            ``time_id``: ``(B, N)`` — 패치별 시간 인덱스.
        """
        # 1. Scaler: point-level 정규화
        values = batch.values.unsqueeze(-1)  # (B, L, 1)
        loc, scale, gmax, gmin = self.scaler(
            values,
            sample_id=batch.sample_id,
            variate_id=batch.variate_id,
        )
        normalized = ((values - loc) / scale.clamp(min=1e-8)).squeeze(
            -1
        )  # (B, L)

        # 2. Patchify (projection 전 raw patches 추출)
        patches, p_sid, p_vid, time_id, patch_mask = self.patch_embed.patchify(
            normalized, batch.sample_id, batch.variate_id
        )
        # patches: (B, N, patch_size)

        b = patches.shape[0]
        device = patches.device

        # 3. global_var_idx 계산 — CNN stem과 modality embedding 모두에서 재사용
        patch_signal_types: torch.Tensor | None = None
        # v2: spatial_id 임베딩은 폐지됐으나, data/downstream 호환을 위해 plumbing은 유지.
        # patch_spatial_ids는 출력 dict에만 실려 downstream에서 참조 가능 (모델 내부 미사용).
        patch_spatial_ids: torch.Tensor | None = None

        if hasattr(batch, "spatial_ids") and batch.spatial_ids is not None:
            per_row_max_var = p_vid.max(dim=-1).values  # (B,)
            var_offsets = torch.zeros(b, dtype=torch.long, device=device)
            if b > 1:
                var_offsets[1:] = per_row_max_var[:-1].cumsum(dim=0)
            global_var_idx = var_offsets.unsqueeze(-1) + (p_vid - 1)  # (B, N)
            global_var_idx = global_var_idx.clamp(min=0)

            patch_signal_types = batch.signal_types.to(device)[
                global_var_idx
            ]  # (B, N)
            # spatial_ids는 v2에서 [0] 보존 필드 — 임베딩에 쓰지 않고 plumbing만 유지.
            patch_spatial_ids = batch.spatial_ids.to(device)[
                global_var_idx
            ]  # (B, N)

            # 절대 시간 기반 abs_time_id 계산 (cross-modal 매칭 전용)
            # time_id(상대적)는 RoPE용으로 유지, abs_time_id는 cross-modal loss용
            #
            # 같은 sample_id 내에서 절대 시간의 최소값을 빼서
            # 버킷 내 상대 offset으로 변환 → patch_size 단위 양자화.
            # → 같은 물리적 시간대의 다른 variate 패치가 동일 abs_time_id를 가짐.
            abs_time_id = time_id  # fallback
            if (
                hasattr(batch, "start_samples")
                and batch.start_samples is not None
            ):
                patch_start = batch.start_samples.to(device)[
                    global_var_idx
                ]  # (B, N)
                abs_time = patch_start + time_id * self.patch_size  # (B, N)
                # patch_size 단위로 양자화 — 같은 물리적 시간의 패치가 정확히 매칭
                abs_time_id = abs_time // self.patch_size  # (B, N)
                abs_time_id[~patch_mask] = 0

        # 4. Projection (linear 또는 CNN stem) — patch content 표현만 생성
        patch_embed = self.patch_embed.project(patches, patch_signal_types)
        # patch_embed: (B, N, d_model)

        # 패딩 마스크 (p_vid==0은 패딩 토큰)
        valid_token = (p_vid > 0).unsqueeze(-1)  # (B, N, 1)

        # 5-6. Conditioning Embedding 계산 (signal_type modality + loc + scale)
        # mask_token이 patch content를 덮어써도 conditioning은 살아남도록 별도 계산.
        # 이후 mask 적용 후 합산하여 마스킹된 위치도 자기 신호 종류/레벨 정보를 유지.
        cond = torch.zeros_like(patch_embed)
        if self.use_spatial_embed and patch_signal_types is not None:
            # v2: 단일 modality(signal_type) 임베딩만 더함. spatial_id 임베딩 폐지.
            sig_emb = self.signal_type_embed(
                patch_signal_types
            )  # (B, N, d_model)
            cond = cond + sig_emb

        n = patch_embed.shape[1]
        stride = self.patch_embed.stride
        patch_starts = torch.arange(n, device=device) * stride  # (N,)
        patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
        patch_loc = loc[:, patch_starts, :]  # (B, N, 1)
        patch_scale = scale[:, patch_starts, :]  # (B, N, 1)

        # AdaLN conditioning 입력 (B: enrich 시 winsorized max/min 추가)
        if self.enrich_cond_peak:
            patch_max = gmax[:, patch_starts, :]  # (B, N, 1)
            patch_min = gmin[:, patch_starts, :]  # (B, N, 1)
            cond_stats = torch.cat(
                [patch_loc, patch_scale, patch_max, patch_min], dim=-1
            )  # (B, N, 4)
        else:
            cond_stats = torch.cat(
                [patch_loc, patch_scale], dim=-1
            )  # (B, N, 2)

        # 안 B(cond_local_trend): 조건에 시간 해상도를 부여한다.
        # 값 정규화(창 단위)는 건드리지 않으므로 토큰이 보는 파형은 불변 —
        # 사전학습 동역학을 보존한 채 창 내부 추세만 조건으로 흘린다.
        # 무차원 상대값이라 modality 간 동적범위(ECG 0.221 vs CO2 12.95) 문제를
        # 새로 만들지 않는다.
        n_cond_extra = 0  # 패치 단위 추가 열 개수 (마스킹 대상)
        if self.cond_trend_mode != "none":
            # patches 는 이미 정규화된 값((values-loc)/scale)의 패치이므로,
            # 그 평균이 곧 (패치 국소 평균 - 창 loc)/창 scale 이다.
            dev = patches.mean(dim=-1, keepdim=True)  # (B, N, 1)

            def _lag(k: int) -> torch.Tensor:
                """k 패치 이전 값. **sample 경계를 넘지 않는다** — packed batch 는
                한 행에 여러 window 가 이어 붙으므로, 막지 않으면 다른 환자의 창을
                넘겨다보게 된다. 경계에서는 현재 값으로 대체(causal padding)."""
                sh = torch.roll(dev, shifts=k, dims=1)
                same = p_sid == torch.roll(p_sid, shifts=k, dims=1)  # (B, N)
                same[:, :k] = False                                  # wrap-around 차단
                return torch.where(same.unsqueeze(-1), sh, dev)

            # ⚠ lag 은 **패치 개수** 기준이라 시간 span 이 창 길이와 무관하게 고정이다
            # (patch 50 @100Hz = 0.5 s/패치). 파운데이션 모델에는 이게 맞다 — 창 비례로
            # 하면 같은 조건 차원이 과제마다(부정맥 30s / 저혈압 300s / ICH 1200s) 다른
            # 의미를 갖게 되어 사전학습과 다운스트림이 어긋난다. 대신 범위를 **로그
            # 간격으로 넓게** 잡아, 짧은 창에서는 뒤쪽 lag 이 sample 경계에 걸려 자기
            # 값으로 자동 degrade 되도록 한다.
            m = self.cond_trend_mode
            if m == "patchstat_abs":
                # 보정된 채널(ABP·CVP·PAP·CO2·AWP·RespFlow)에는 패치 통계를
                # **절대 단위**로 준다. 저혈압(MAP<65 mmHg)·두개내압(ICP>22) 처럼
                # 라벨이 절대 역치인 과제에서, 상대값만 주면 모델이
                # `창loc + 상대편차 x 창scale` 을 곱셈으로 복원해야 하는데
                # 조건 MLP 가 곱셈을 잘 못 한다.
                # 무단위 채널(PPG·ECG·RespImp)은 절대값이 장비 지문이므로
                # 상대값 그대로 둔다 — gate_unitless_cond 와 같은 구분이다.
                pm = dev
                ps = patches.std(dim=-1, keepdim=True)
                pmax = patches.max(dim=-1, keepdim=True).values
                pmin = patches.min(dim=-1, keepdim=True).values
                if patch_signal_types is not None and self._gated_signal_types:
                    unit = torch.zeros_like(patch_signal_types, dtype=torch.bool)
                    for st in self._gated_signal_types:
                        unit |= patch_signal_types == st
                    calib = (~unit).unsqueeze(-1).to(patches.dtype)  # (B,N,1)
                else:
                    calib = torch.ones_like(dev)
                sc, lo = patch_scale.to(patches.dtype), patch_loc.to(patches.dtype)
                extra = [
                    torch.where(calib > 0, pm * sc + lo, pm),
                    torch.where(calib > 0, ps * sc, ps),
                    torch.where(calib > 0, pmax * sc + lo, pmax),
                    torch.where(calib > 0, pmin * sc + lo, pmin),
                ]
            elif m in ("patchabs", "patchabs_rel"):
                # 창 loc/scale 을 **대체**한다(덧붙이지 않는다). 패치 평균·std 를
                # 물리 단위 그대로 준다 — 창 통계는 패치 평균들을 attention 으로
                # 모으면 복원되므로 중복이다. 4열 -> 2열.
                # 장점: ① 창 길이에 불변(상대값은 창 자신의 통계 기준이라 길이가
                # 바뀌면 의미가 변한다) ② 절대 보정 정보를 유지하면서 시간 해상도
                # 확보. 대가: modality 간 동적범위(ECG 0.221 vs CO2 12.95, 58.6배)가
                # 전 열에 적용되므로 cond_transform 을 같이 볼 것.
                #
                # patchabs_rel: **같은 두 칸**에 보정 채널은 절대(mmHg), 무단위
                # 채널(PPG·RespImp)은 상대값을 넣는다. 무단위 채널의 절대값은
                # 장비 이득일 뿐이라 의미가 없지만, 창 scale 로 나눈 상대값은
                # AGC 에 불변이므로(gain x0.21~x3.7 에도 소수 4자리까지 동일)
                # 생리 정보가 남는다. 0 으로 채우는 열이 없어 전면 게이팅보다
                # 열을 아낀다. 이 모드는 게이팅이 아무것도 0 으로 만들면 안 되므로
                # gate_absolute_only=true 와 함께 쓴다(아래 n_abs=0 참조).
                sc, lo = patch_scale.to(patches.dtype), patch_loc.to(patches.dtype)
                pm = dev
                ps = patches.std(dim=-1, keepdim=True)
                if m == "patchabs":
                    extra = [pm * sc + lo, ps * sc]
                else:
                    unit = torch.zeros_like(patch_signal_types, dtype=torch.bool)                         if patch_signal_types is not None else None
                    if unit is not None:
                        for st in self._gated_signal_types:
                            unit |= patch_signal_types == st
                        calib = (~unit).unsqueeze(-1).to(patches.dtype)
                        extra = [torch.where(calib > 0, pm * sc + lo, pm),
                                 torch.where(calib > 0, ps * sc, ps)]
                    else:
                        extra = [pm * sc + lo, ps * sc]
                cond_stats = cond_stats[..., :0]   # 창 통계 제거
            elif m == "abs_res":
                # 창 loc/scale 을 **남긴 채**(patchabs 와 달리) 패치 통계를
                # 물리 단위로 준다. 4열 = [창loc, 창scale, 패치평균(mmHg),
                # 패치std(mmHg)] — patchls 와 열 수가 같고 절대 정보가 추가된다.
                #
                # 동기: 토큰은 창 단위 z-정규화를 거쳐 **절대 수준이 제거**되므로
                # ABP·CVP·CO2·ICP 같은 보정 채널에서 절대압은 조건 경로가
                # 유일한 통로다. patchabs 는 창 앵커를 버려 저혈압에서 최악이었고
                # (0.842), patchls 는 패치 열이 무차원이라 절대 정보가 창 단위
                # 하나뿐이다. 둘의 중간이다.
                #
                # 무단위 채널(PPG·RespImp)은 절대값이 장비 이득일 뿐이므로
                # 상대값으로 남긴다(전면 게이팅 시에는 어차피 전부 0).
                sc, lo = patch_scale.to(patches.dtype), patch_loc.to(patches.dtype)
                pm = dev
                ps = patches.std(dim=-1, keepdim=True)
                if patch_signal_types is not None and self._gated_signal_types:
                    unit = torch.zeros_like(patch_signal_types, dtype=torch.bool)
                    for st in self._gated_signal_types:
                        unit |= patch_signal_types == st
                    calib = (~unit).unsqueeze(-1).to(patches.dtype)
                    extra = [torch.where(calib > 0, pm * sc + lo, pm),
                             torch.where(calib > 0, ps * sc, ps)]
                else:
                    extra = [pm * sc + lo, ps * sc]
            elif m == "patchls":
                # 가장 단순한 형태: 정규화는 창 단위 하나, 조건은 기존 [loc, scale]
                # 쌍을 **패치 단위**로. max/min 없이 이것만으로 충분한지 확인한다.
                extra = [dev, patches.std(dim=-1, keepdim=True)]
            elif m == "patchstat":
                # 정규화는 창 단위 하나로 유지(토큰이 추세를 그대로 담음) 하고,
                # 기존 조건 통계 세트(loc·scale·max·min)를 **패치 단위**로 계산한다.
                # 각 패치가 자기 자신만 기술하므로 lag 이 없고 → 창 길이 의존성도,
                # sample 경계 처리도 필요 없다. 패치 간 비교는 attention 이 한다.
                # 전부 창 scale 로 나눈 무차원 상대값이라 modality 간 동적범위
                # (ECG 0.221 vs CO2 12.95) 문제를 새로 만들지 않는다.
                extra = [dev,
                         patches.std(dim=-1, keepdim=True),    # 국소 진폭
                         patches.max(dim=-1, keepdim=True).values,
                         patches.min(dim=-1, keepdim=True).values]
            elif m == "dev":
                extra = [dev]
            elif m == "lag":                       # 0.5s ~ 128s 어디서 왔는가
                extra = [dev] + [_lag(k) for k in (1, 4, 16, 64, 256)]
            elif m == "deriv":                     # 수준 + 단기·중기·장기 기울기
                extra = [dev, dev - _lag(2), dev - _lag(16), dev - _lag(128)]
            elif m == "multi":                     # 다중 스케일 평균 + 인과적 최소값
                short = torch.stack([_lag(k) for k in (0, 1, 2, 3)], 0).mean(0)
                mid = torch.stack([_lag(k) for k in range(0, 32, 4)], 0).mean(0)
                long_ = torch.stack([_lag(k) for k in range(0, 512, 32)], 0).mean(0)
                cmin = torch.stack(
                    [_lag(k) for k in (0, 2, 8, 32, 128, 512)], 0).min(0).values
                extra = [dev, short, mid, long_, cmin]
            else:
                raise ValueError(f"unknown cond_trend_mode: {m}")
            n_cond_extra = len(extra)
            cond_stats = torch.cat(
                [cond_stats] + [e.to(cond_stats.dtype) for e in extra], dim=-1)

        # cond 입력 스케일 정규화 (2026-09-08).
        # cond_proj 는 loc/scale 을 **물리 단위 원값** 그대로 받는데, 실측(VitalDB
        # 5,115 case, modality 당 27만~179만 표본) 결과 scale 의 modality 간 동적범위가
        # **58.6배** 였다 (ECG 0.221 vs CO2 12.95). Linear 하나가 이를 다 받으면 큰
        # modality 가 gradient 를 지배하고 ECG 는 사실상 묻힌다. loc 은 ECG(고역통과로
        # median 0) 를 빼면 8배라 문제 없어 기본적으로 건드리지 않는다.
        # 부분 게이팅: 절대항(창 loc/scale, enrich 시 max/min 포함)만 0 으로.
        # 상대 패치 통계는 그대로 흘려보낸다.
        if (self.gate_absolute_only and self.gate_unitless_cond
                and patch_signal_types is not None):
            _g = torch.zeros_like(patch_signal_types, dtype=torch.bool)
            for st in self._gated_signal_types:
                _g |= patch_signal_types == st
            # patchabs 계열은 창 단위 절대 prefix 가 없다(전 열이 패치 단위이며,
            # patchabs_rel 은 무단위 채널에 이미 상대값을 넣었다) -> 0 으로 만들 열이 없다.
            n_abs = 0 if self.cond_trend_mode in ("patchabs", "patchabs_rel")                 else (4 if self.enrich_cond_peak else 2)
            keep = torch.ones_like(cond_stats)
            keep[..., :n_abs] = (~_g).unsqueeze(-1).to(cond_stats.dtype)
            cond_stats = cond_stats * keep

        # cond 통계 -> ada_cond 변환을 **클로저**로 둔다. masked 경로는 마스킹된
        # 패치의 패치단위 통계를 가린 별도의 cond 를 써야 하기 때문이다(아래 참조).
        # dropout 난수는 두 경로가 공유하도록 여기서 한 번만 뽑는다.
        keep_tok: torch.Tensor | None = None
        if self.training and self.cond_dropout_prob > 0.0:
            # D(cond_dropout_prob): 학습 중에만, **variate 단위로** cond 를 통째로 제거.
            # 패치 단위로 떨어뜨리면 같은 변량의 남은 패치에서 그대로 복원되므로 규제가
            # 안 된다. cond 가 코호트/장비 공변량 지름길로 굳는 것을 막는다.
            n_var = int(p_vid.max().item()) + 1
            keep = (
                torch.rand(b, n_var, device=device) >= self.cond_dropout_prob
            ).to(torch.float32)  # (B, V)
            keep[:, 0] = 0.0  # p_vid==0 은 패딩
            keep_tok = keep.gather(1, p_vid.clamp(min=0))  # (B, N)
            # D': 지정된 modality 만 dropout 대상. 나머지(calibrated)는 항상 keep.
            if (
                self._cond_dropout_signal_types is not None
                and patch_signal_types is not None
            ):
                elig = torch.zeros_like(
                    patch_signal_types, dtype=torch.bool
                )  # (B, N)
                for st in self._cond_dropout_signal_types:
                    elig |= patch_signal_types == st
                keep_tok = torch.where(
                    elig, keep_tok, torch.ones_like(keep_tok)
                )

        def _build_ada(cs: torch.Tensor) -> torch.Tensor:
            """(B, N, C) cond 통계 -> (B, N, d_cond) AdaLN 조건 벡터."""
            ada = self.cond_proj(
                self._transform_cond(cs, patch_signal_types)
            )  # (B, N, d_cond)

            # A(gate_unitless_cond): 물리적 영점에 추적되지 않는 modality
            # (ECG·PPG·RESP_Imp)의 loc/scale conditioning 차단 (device/환자 지문).
            # 마스크는 cond_proj 출력에 — 입력에 걸면 bias 때문에 의미 깨짐.
            # NOTE: 게이팅해도 LSCNorm.modulation 의 bias 는 살아 있으므로, 해당
            # modality 는 입력 비의존적 affine 만 갖는 표준 정규화로 환원된다.
            if self.gate_unitless_cond and patch_signal_types is not None:
                gated = torch.zeros_like(
                    patch_signal_types, dtype=torch.bool
                )  # (B, N)
                for st in self._gated_signal_types:
                    gated |= patch_signal_types == st
                if not self.gate_absolute_only:
                    ada = ada * (~gated).unsqueeze(-1)
                # gate_absolute_only 면 위에서 cond_stats 의 절대 열만 이미 0 이 되었고,
                # 출력 마스킹은 하지 않는다. bias 가 남는 것이 **의도**다 — 상대 열로부터
                # 의미 있는 modulation 을 만들어야 하기 때문. (전면 게이팅 시에는 반대로
                # 출력에 걸어야 bias 까지 죽는다.)
            if keep_tok is not None:
                ada = ada * keep_tok.unsqueeze(-1).to(ada.dtype)
            return ada * valid_token  # 패딩 위치는 0으로

        ada_cond = _build_ada(cond_stats)
        cond = cond * valid_token  # signal_type + spatial_id만 token에 더해짐

        # 7. Pred Mask 생성 (random/block/variate-level)
        pred_mask: torch.Tensor | None = None
        if mask_ratio > 0 and task in ("masked", "both"):
            pred_mask = create_patch_mask(
                patch_mask,
                mask_ratio=mask_ratio,
                patch_variate_id=p_vid if variate_mask_prob > 0 else None,
                variate_mask_prob=variate_mask_prob,
                block_mask=block_mask,
                block_size_min=block_size_min,
                block_size_max=block_size_max,
                # unit 스코핑은 variate-level 마스킹이 켜진 경우에만 활성 (Phase 2).
                # Phase 1 (variate_mask_prob=0) 은 구 동작 그대로 → 기존 ckpt 재현 가능.
                patch_sample_id=p_sid if variate_mask_prob > 0 else None,
            )

        # 8. Base Attention Mask: 같은 sample 내에서만 attend + 유효 패치만
        base_attn_mask = (
            (p_sid.unsqueeze(-1) == p_sid.unsqueeze(-2))
            & patch_mask.unsqueeze(-2)
            & patch_mask.unsqueeze(-1)
        )  # (B, N, n)

        # 8.5. Complete Variate Dropout: attention에서 variate를 물리적으로 제거
        # → 학습 시 "해당 variate 없이 cross-pred" 시나리오를 경험
        # → zero-shot cross-modal generation의 train-inference gap 해소
        drop_mask: torch.Tensor | None = None
        if (
            variate_drop_prob > 0
            and self.training
            and task in ("masked", "both")
        ):
            drop_mask = self._sample_variate_drop(
                p_sid, p_vid, patch_mask, variate_drop_prob
            )  # (B, N) bool — True = attention에서 제거
            if drop_mask is not None:
                keep = ~drop_mask  # (B, N)
                # attention에서 제거: dropped 토큰은 attend 못하고, attend 받지도 못함
                base_attn_mask = (
                    base_attn_mask & keep.unsqueeze(-1) & keep.unsqueeze(-2)
                )

        # 9. Encoder 입력 빌드 헬퍼
        # patch content를 mask_token으로 교체(content_mask 위치) → conditioning 합산.
        # 이렇게 해야 마스킹/드롭된 위치도 signal_type·spatial·loc·scale 정보가 유지됨.
        def _make_input(content_mask: torch.Tensor | None) -> torch.Tensor:
            if content_mask is None:
                x = patch_embed
            else:
                mt = self.mask_token.expand_as(patch_embed)
                x = torch.where(content_mask.unsqueeze(-1), mt, patch_embed)
            return x + cond

        # 10. Task에 따른 Encoder 호출
        result: dict[str, torch.Tensor] = {
            "patches": patches,
            "patch_signal_types": patch_signal_types,
            "patch_spatial_ids": patch_spatial_ids,
            "loc": loc,
            "scale": scale,
            "patch_mask": patch_mask,
            "patch_sample_id": p_sid,
            "patch_variate_id": p_vid,
            "time_id": time_id,  # 상대적 (RoPE용)
            "abs_time_id": abs_time_id,  # 절대적 (cross-modal 매칭용)
            "pred_mask": pred_mask,
        }

        # MoE 라우팅에서 padded 토큰 제외용 — (B, N) bool
        token_valid = p_vid > 0
        # RoPE position: overlapping-stride 추론 시 정수 ordinal(time_id)은 물리간격을
        # 과대평가한다(인접 토큰이 stride/patch_size 배로 촘촘한데 ordinal 차는 여전히 1).
        # → position interpolation으로 교정: physical position = time_id *
        # stride/patch_size.
        # 비중첩(stride==patch_size)이면 정수 time_id 그대로 → 학습/기존 경로 byte-identical.
        # 원본 정수 time_id는 result dict·abs_time_id용으로 그대로 보존한다.
        if self.patch_embed.stride < self.patch_size and self.rope_pi:
            rope_time_id = time_id.to(torch.float32) * (
                self.patch_embed.stride / self.patch_size
            )
        else:
            rope_time_id = time_id
        encoder_kwargs = dict(
            var_id=p_vid,
            time_id=rope_time_id,
            token_mask=token_valid,
            cond=ada_cond,
            # RoPE는 상대 time_id(overlapping 시 PI 적용); token_mask는 MoE aux_loss용;
            # cond는 AdaLN용
        )
        use_causal = task in ("next_pred", "both")

        # causal mask (next_pred, both에서 공유)
        # ⚠️ packed 인덱스 하한삼각(torch.tril)은 any-variate packing 에서 인과성을
        # 보장하지 못한다. 한 unit 안의 신호가 variate-major 로 놓이므로
        # (ECG t0..t2 → PPG t0..t2 → ABP t0..t2), PPG@t0 의 packed 인덱스가 ECG@t2
        # 보다
        # 커서 "미래" ECG 를 참조하게 된다 (3-variate 예시에서 causal 허용칸의 18%).
        # → 물리 시간(abs_time_id) 기준으로 비교한다. 같은 시각(t_j == t_i)의 cross-modal
        #   참조는 허용하고 미래만 차단한다.
        # Phase 1 (channel-independent) 은 블록마다 variate 가 하나뿐이라 packed 순서 =
        # 시간 순서 → tril 과 결과가 동일하다 (기존 checkpoint 재현 가능).
        if use_causal:
            causal_ok = abs_time_id.unsqueeze(-1) >= abs_time_id.unsqueeze(
                -2
            )  # (B, N, N) — t_i >= t_j
            causal_mask = base_attn_mask & causal_ok  # (B, N, N)

        # bidirectional 입력: pred_mask | drop_mask 위치를 mask_token으로 교체
        bi_content_mask = drop_mask
        if pred_mask is not None:
            bi_content_mask = (
                pred_mask
                if bi_content_mask is None
                else (pred_mask | bi_content_mask)
            )
        # Downstream gap masking: 데이터 prep 단계에서 NaN→0 채운 patch 위치를
        # mask_token 으로 교체 (downstream finetune 전용 통로).
        if extra_content_mask is not None:
            bi_content_mask = (
                extra_content_mask
                if bi_content_mask is None
                else (extra_content_mask | bi_content_mask)
            )

        # [정답 누출 차단] 패치단위 통계(평균·std·max·min)는 **그 패치 자신**을
        # 기술하므로, 마스킹된 위치에 그대로 흘리면 재구성 정답을 넘겨주는 셈이다.
        # 창 단위 loc/scale(앞 n_abs 열)은 전 패치 공통값이라 패치를 특정하지 못하므로
        # 그대로 둔다(기존 instance norm 계약 유지).
        # 실측 2026-09-19: 이 4통계만 아는 최적 선형 예측기가 정규화 패치 MSE 를 기준의
        # 44.0% 로 낮추고, 학습 masked loss 비가 44.2% 로 일치했다 — 손실 개선분이 전부
        # 누출이었다. causal(next-pred) 경로는 위치 i 가 i+1 의 cond 를 못 보므로
        # 가리지 않는다.
        bi_kwargs = encoder_kwargs
        if (
            self.mask_cond_trend
            and n_cond_extra > 0
            and bi_content_mask is not None
        ):
            n_abs = cond_stats.shape[-1] - n_cond_extra
            cs = cond_stats.clone()
            cs[..., n_abs:] = torch.where(
                bi_content_mask.unsqueeze(-1),
                torch.zeros_like(cs[..., n_abs:]),
                cs[..., n_abs:],
            )
            bi_kwargs = {**encoder_kwargs, "cond": _build_ada(cs)}

        if task == "both":
            result["encoded"] = self.encoder(
                _make_input(bi_content_mask),
                attn_mask=base_attn_mask,
                **bi_kwargs,
            )
            # causal: drop_mask만 적용 (causal attention이 미래 정보 차단하므로
            # pred_mask는 불필요).
            result["encoded_causal"] = self.encoder(
                _make_input(drop_mask),
                attn_mask=causal_mask,
                **encoder_kwargs,
            )
        elif task == "next_pred":
            result["encoded"] = self.encoder(
                _make_input(drop_mask),
                attn_mask=causal_mask,
                **encoder_kwargs,
            )
        else:  # "masked"
            result["encoded"] = self.encoder(
                _make_input(bi_content_mask),
                attn_mask=base_attn_mask,
                **bi_kwargs,
            )

        return result

    # ── Forward ────────────────────────────────────────────────────

    def forward(
        self,
        batch: PackedBatch,
        task: str = "masked",  # "masked" 또는 "next_pred"
        mask_ratio: float = 0.0,
        block_mask: bool = False,
        block_size_min: int = 3,
        block_size_max: int = 8,
        variate_mask_prob: float = 0.0,
        variate_drop_prob: float = 0.0,
        extra_content_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        enc = self._encode(
            batch,
            task=task,
            mask_ratio=mask_ratio,
            block_mask=block_mask,
            block_size_min=block_size_min,
            block_size_max=block_size_max,
            variate_mask_prob=variate_mask_prob,
            variate_drop_prob=variate_drop_prob,
            extra_content_mask=extra_content_mask,
        )

        encoded = enc[
            "encoded"
        ]  # bidirectional (or sole encoding for single-task)
        patch_signal_types = enc["patch_signal_types"]  # (B, N) or None

        out_dict: dict[str, torch.Tensor] = {
            "encoded": encoded,
            "patches": enc["patches"],
            "patch_signal_types": patch_signal_types,
            "loc": enc["loc"],
            "scale": enc["scale"],
            "patch_mask": enc["patch_mask"],
            "patch_sample_id": enc["patch_sample_id"],
            "patch_variate_id": enc["patch_variate_id"],
            "time_id": enc["abs_time_id"],  # cross-modal 매칭용 (절대 시간)
            "pred_mask": enc["pred_mask"],
        }

        # ── Masked Reconstruction ──
        if task in ("masked", "both"):
            out_dict["reconstructed"] = self.head(
                encoded
            )  # (B, N, patch_size)
            # Per-target-type cross-modal prediction (separate heads)
            cross_pred_per_type = torch.stack(
                [
                    self.cross_heads[str(st)](encoded)
                    for st in range(self.num_signal_types)
                ],
                dim=2,
            )  # (B, N, num_signal_types, patch_size)
            out_dict["cross_pred_per_type"] = cross_pred_per_type
            if self.dual_cross_head:
                out_dict["cross_next_pred_per_type"] = torch.stack(
                    [
                        self.cross_next_heads[str(st)](encoded)
                        for st in range(self.num_signal_types)
                    ],
                    dim=2,
                )  # (B, N, num_signal_types, patch_size)

        # ── Block Next-Patch Prediction ──
        # encoded_causal[n] → K개의 future raw patches (n+1, ..., n+K) 병렬 예측.
        # BlockNextHead (shared trunk + K heads)가 바로 (B, N, K, P) 반환.
        if task in ("next_pred", "both"):
            encoded_for_next = enc.get(
                "encoded_causal", encoded
            )  # (B, N, d_model)
            out_dict["next_pred"] = self.next_head(
                encoded_for_next
            )  # (B, N, K, P)

        return out_dict

    # ── Inference API ──────────────────────────────────────────────

    @torch.no_grad()
    def extract_features(self, batch: PackedBatch) -> dict[str, torch.Tensor]:
        """Downstream task용 feature 추출 (양방향 attention).

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.

        Returns
        -------
        dict with keys:
            ``encoded``, ``patch_mask``, ``loc``, ``scale``,
            ``patch_sample_id``, ``patch_variate_id``.
        """
        self.eval()
        out = self.forward(batch, task="masked")
        out.pop("reconstructed", None)
        out.pop("cross_pred_per_type", None)
        return out

    @torch.no_grad()
    def generate_cross_modal(
        self,
        batch: PackedBatch,
        target_signal_type: int,
        denormalize: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Zero-shot cross-modal waveform generation (Virtual Token Injection).

        입력 batch의 source signal로부터 target signal type의 waveform을 생성한다.
        target variate에 [MASK] 가상 토큰을 주입하여, 학습 시 variate dropout과
        동일한 상황을 재현한다.

        Parameters
        ----------
        batch:
            Source signal만 포함된 PackedBatch.
        target_signal_type:
            생성할 target signal type (0=ECG, 1=ABP, 2=PPG, ...).
        denormalize:
            ``True``이면 source의 loc/scale로 denormalize (approximate).

        Returns
        -------
        dict with keys:
            ``waveform``: ``(B, N, patch_size)`` — 생성된 target waveform.
            ``patch_mask``: ``(B, N)`` — 유효 패치 마스크.
        """
        self.eval()

        # Forward (mask_ratio=0 → 마스킹 없이 순수 source 정보만 사용)
        out = self.forward(batch, task="masked", mask_ratio=0.0)

        cross_pred_per_type = out["cross_pred_per_type"]  # (B, N, T, P)
        target_pred = cross_pred_per_type[
            :, :, target_signal_type, :
        ]  # (B, N, P)

        if denormalize:
            loc = out["loc"]  # (B, L, 1)
            scale = out["scale"]  # (B, L, 1)
            p = self.patch_size
            stride = self.patch_embed.stride
            n = target_pred.shape[1]
            patch_starts = torch.arange(n, device=loc.device) * stride
            patch_starts = patch_starts.clamp(max=loc.shape[1] - 1)
            patch_loc = loc[:, patch_starts, :]  # (B, N, 1)
            patch_scale = scale[:, patch_starts, :]  # (B, N, 1)
            target_pred = target_pred * patch_scale + patch_loc

        return {
            "waveform": target_pred,
            "patch_mask": out["patch_mask"],
        }

    @torch.no_grad()
    def forecast(
        self,
        batch: PackedBatch,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Block next-patch 예측 (non-autoregressive).

        각 position n에서 미래 K개 패치를 동시에 예측한다.

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        denormalize:
            ``True``이면 scaler의 loc/scale로 원본 스케일 복원.

        Returns
        -------
        torch.Tensor
            ``(B, N, K, patch_size)`` block prediction map.
        """
        self.eval()
        out = self.forward(batch, task="next_pred")
        pred = out["next_pred"]  # (B, N, K, patch_size)

        if denormalize:
            loc = out["loc"]  # (B, L, 1)
            scale = out["scale"]  # (B, L, 1)
            p = self.patch_size
            patch_loc = loc[:, ::p, :]  # (B, N_approx, 1)
            patch_scale = scale[:, ::p, :]  # (B, N_approx, 1)
            n = pred.shape[1]
            patch_loc = patch_loc[:, :n, :]  # (B, N, 1)
            patch_scale = patch_scale[:, :n, :]  # (B, N, 1)
            # Broadcast over K dimension
            pred = pred * patch_scale.unsqueeze(2) + patch_loc.unsqueeze(2)

        return pred

    @torch.no_grad()
    def generate(
        self,
        batch: PackedBatch,
        n_steps: int,
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Block-autoregressive 다단계 생성.

        Block Next Prediction head가 1-shot에 K개 패치를 내놓으므로, 매 forward마다
        K개를 모두 취해 입력에 append → 다시 forward → 반복. ``collate_mode="ci"``
        (single-variate-per-row) 전제.

        Parameters
        ----------
        batch:
            PackCollate로 생성된 PackedBatch.
        n_steps:
            생성할 패치 수.
        denormalize:
            ``True``이면 최종 출력을 원본 스케일로 복원.

        Returns
        -------
        torch.Tensor
            ``(n_steps, B, patch_size)`` generated patches.
        """
        self.eval()
        p = self.patch_size
        k = self.next_block_size

        out = self.forward(batch, task="next_pred")
        loc = out["loc"]  # (B, L, 1)
        scale = out["scale"]  # (B, L, 1)
        cached_loc = loc[:, 0:1, :]  # (B, 1, 1)
        cached_scale = scale[:, 0:1, :]  # (B, 1, 1)

        generated: list[torch.Tensor] = []
        pred = out["next_pred"]  # (B, N, K, patch_size)
        patch_mask = out["patch_mask"]  # (B, N)
        b = pred.shape[0]
        last_valid_idx = patch_mask.sum(dim=-1) - 1  # (B,)
        last_valid_idx = last_valid_idx.clamp(min=0)
        arange_b = torch.arange(b, device=pred.device)
        block = pred[arange_b, last_valid_idx]  # (B, K, patch_size)

        # 한 번의 forward에서 나오는 K patches를 순서대로 append.
        for j in range(k):
            if len(generated) >= n_steps:
                break
            generated.append(block[:, j, :])  # (B, patch_size)

        while len(generated) < n_steps:
            # block의 K patches를 모두 입력에 append — 다음 forward에서 새 예측.
            for j in range(k):
                batch = _append_patch_to_batch(batch, block[:, j, :], p)

            out = self.forward(batch, task="next_pred")
            pred = out["next_pred"]  # (B, N, K, patch_size)
            patch_mask = out["patch_mask"]
            last_valid_idx = patch_mask.sum(dim=-1) - 1
            last_valid_idx = last_valid_idx.clamp(min=0)
            block = pred[arange_b, last_valid_idx]  # (B, K, patch_size)
            for j in range(k):
                if len(generated) >= n_steps:
                    break
                generated.append(block[:, j, :])

        result = torch.stack(
            generated[:n_steps], dim=0
        )  # (n_steps, B, patch_size)

        if denormalize:
            dl = cached_loc.squeeze(-1).permute(1, 0)  # (1, B)
            ds = cached_scale.squeeze(-1).permute(1, 0)  # (1, B)
            result = result * ds.unsqueeze(-1) + dl.unsqueeze(-1)

        return result


def _append_patch_to_batch(
    batch: PackedBatch,
    new_patch: torch.Tensor,  # (B, patch_size)
    patch_size: int,
) -> PackedBatch:
    """PackedBatch에 새 패치를 append한다.

    Single-variate-per-row 가정. max_length 초과 시 우측 패딩 확장.

    Parameters
    ----------
    batch:
        기존 PackedBatch.
    new_patch:
        추가할 패치. ``(B, patch_size)``.
    patch_size:
        패치 크기.

    Returns
    -------
    PackedBatch
        새 패치가 append된 PackedBatch.
    """
    b, seq_len = batch.values.shape
    device = batch.values.device

    valid_mask = batch.sample_id > 0  # (B, L)
    valid_lengths = valid_mask.sum(dim=-1)  # (B,)

    new_end = valid_lengths + patch_size  # (B,)
    max_new_end = new_end.max().item()

    if max_new_end > seq_len:
        pad_size = max_new_end - seq_len
        batch = PackedBatch(
            values=torch.cat(
                [batch.values, torch.zeros(b, pad_size, device=device)], dim=-1
            ),
            sample_id=torch.cat(
                [
                    batch.sample_id,
                    torch.zeros(b, pad_size, dtype=torch.long, device=device),
                ],
                dim=-1,
            ),
            variate_id=torch.cat(
                [
                    batch.variate_id,
                    torch.zeros(b, pad_size, dtype=torch.long, device=device),
                ],
                dim=-1,
            ),
            lengths=batch.lengths,
            sampling_rates=batch.sampling_rates,
            signal_types=batch.signal_types,
            spatial_ids=batch.spatial_ids,
            padded_lengths=batch.padded_lengths,
        )

    for i in range(b):
        start = valid_lengths[i].item()
        end = start + patch_size
        batch.values[i, start:end] = new_patch[i]
        batch.sample_id[i, start:end] = (
            batch.sample_id[i, start - 1] if start > 0 else 1
        )
        batch.variate_id[i, start:end] = (
            batch.variate_id[i, start - 1] if start > 0 else 1
        )

    return batch
