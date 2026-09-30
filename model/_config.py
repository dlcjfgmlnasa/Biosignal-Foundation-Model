# -*- coding:utf-8 -*-
"""모델 설정 데이터클래스.

BiosignalFoundationModel의 모든 아키텍처 파라미터를 하나의 dataclass로 통합하여
실험 재현성과 checkpoint 직렬화를 보장한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class ModelConfig:
    """BiosignalFoundationModel 아키텍처 설정.

    Parameters
    ----------
    d_model:
        트랜스포머 임베딩 차원.
    num_layers:
        트랜스포머 인코더 레이어 수.
    patch_size:
        패치 크기 (time-step 수).
    stride:
        패치 보폭. ``None``이면 ``patch_size``와 동일 (non-overlapping).
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
    use_spatial_embed:
        단일 modality(signal_type) 임베딩 사용 여부.
        (이름은 하위 호환을 위해 유지 — v2에서 의미를 "modality embedding"으로
        재정의. spatial_id 소분류 임베딩은 폐지됨.)
    dropout_p:
        드롭아웃 확률.
    num_signal_types:
        신호 타입(modality) 수. v2: 9 (2026-06-23 PAP 제거 후 연속 번호)
        (ECG0, ABP1, PPG2, CVP3, CO24, AWP5, ICP6,
        RESP_Impedance7, RESP_Flow8).
    next_block_size:
        Block Next Prediction에서 각 position이 병렬 예측하는 future patch 수 (K).
    """

    # Architecture
    d_model: int = 64
    num_layers: int = 2
    patch_size: int = 100
    stride: int | None = None
    num_heads: int | None = None
    num_groups: int | None = None

    # Features
    use_glu: bool = True
    use_moe: bool = False
    num_experts: int = 8
    num_experts_per_token: int = 2
    use_rope: bool = True
    use_var_attn_bias: bool = True
    use_spatial_embed: bool = True
    dropout_p: float = 0.0

    # Signal types (v2: single modality embedding)
    # ECG0, ABP1, PPG2, CVP3, CO24, AWP5, ICP6,
    # RESP_Impedance7, RESP_Flow8 — 2026-06-23 PAP 제거 후 9종 연속 번호.
    num_signal_types: int = 9
    # NOTE: num_spatial_ids는 v2에서 폐지됨 (spatial_id 소분류 임베딩 제거).
    # 구 yaml/checkpoint에 남은 num_spatial_ids 키는 from_dict가 무시한다.

    # Task
    next_block_size: int = (
        4  # Block Next Prediction (K future patches per position)
    )
    next_head_d_inner: int | None = (
        None  # BlockNextHead 내부 차원. None이면 d_model 사용
    )

    # Contrastive
    contrastive_proj_dim: int = 0  # 0=비활성, >0=projection head 출력 차원

    # 예측형(t→t+1) cross-modal 전용 head. dual_coupling 과 짝으로 쓴다.
    dual_cross_head: bool = False

    # AdaLN conditioning (loc/scale을 multiplicative gate로 모든 layer에 주입)
    # use_lscnorm=True: LSCNorm(RMSNorm + AdaLN modulation, default).
    # use_lscnorm=False: 동등 forward 를 위해 cond_proj·modulation 을 zero-freeze 하여
    #   plain RMSNorm 과 출력 동일하게 만든다 (ablation 용도, 모델 구조는 보존).
    use_lscnorm: bool = True
    d_cond: int = (
        16  # AdaLN cond vector 차원 (ablation에서 16 채택, override 가능)
    )

    # Conditioning redesign (2026-07-15) — 기본 False(opt-in): 구 checkpoint 로딩
    # 호환.
    # 재사전학습 config(yaml)에서만 True 로 켠다.
    # A: PPG(2)·RESP_Imp(7) 의 device-arbitrary loc/scale conditioning 차단 (ckpt
    # 호환).
    # 2026-09-19: 창 단위 정규화가 조건 경로에 시간 해상도를 주지 않는 문제 대응.
    # A) 값 자체를 패치 단위로 정규화 (scaler 그룹 키에 패치 인덱스 추가).
    #    토큰이 보는 파형이 바뀌므로 ckpt 비호환 → from-scratch.
    patch_local_norm: bool = False
    # B) 값 정규화는 그대로 두고 cond 에 (패치 국소 평균 − 창 loc)/창 scale 추가.
    #    cond_proj 입력 2→3 이라 ckpt 비호환 → from-scratch. 토큰은 불변.
    cond_local_trend: bool = False
    # 조건 벡터 설계 sweep (2026-09-19). 모두 dev=(패치 국소평균−창 loc)/창 scale 파생.
    #   none  : 기존 [loc, scale]
    #   dev   : +dev                      (안 B)
    #   lag   : +dev, dev_{i-1,4,16,64,256}   0.5s~128s (patch 50 @100Hz)
    #   deriv : +dev, Δ(2), Δ(16), Δ(128)    단기·중기·장기 기울기
    #   multi : +dev, 단기/중기/장기 평균, 인과적 최소값(~256s)
    #   patchstat     : +패치별 [평균, 표준편차, max, min] (무차원 상대값)
    #   patchstat_abs : 위와 같되 **보정 채널은 절대 단위**(mmHg 등).
    #                   라벨이 절대 역치인 과제(저혈압 MAP<65, ICP>22)를 겨냥.
    #                   무단위 채널(PPG·ECG·RespImp)은 상대값 유지.
    #               하나로 두고 기존 통계 세트만 패치 해상도로. lag 이 없어
    #               창 길이 의존성·경계 처리가 필요 없다.
    # lag 은 패치 개수 기준 = **고정 시간 span**. 창 길이가 달라도 의미가
    # 유지되며, 짧은 창에서는 sample 경계에 걸려 자기 값으로 degrade 된다.
    # cond_local_trend=True 는 'dev' 와 동일(하위호환).
    cond_trend_mode: str = "none"
    # 게이팅을 **절대항(창 loc/scale)에만** 적용하고 상대 패치 통계는 살린다.
    # PPG AGC 이득은 창 scale 로 나누면 약분되므로 상대값은 장비 불변(실측 확인).
    gate_absolute_only: bool = False
    # masked reconstruction 경로에서 **마스킹된 패치의** 패치단위 통계
    # (평균·std·max·min) 을 0 으로 가린다. 가리지 않으면 재구성 정답을
    # 그대로 넘겨주는 누출이다 (2026-09-19 실측: 손실 개선분 전부가 누출).
    # causal(next-pred) 경로는 미래 cond 를 못 보므로 가리지 않는다.
    mask_cond_trend: bool = True
    gate_unitless_cond: bool = False
    # B: cond 입력을 [loc,scale]→[loc,scale,max,min] 확장 (cond_proj 2→4, ckpt
    # 비호환→from-scratch).
    enrich_cond_peak: bool = False
    #  A': 게이팅 대상 signal_type 목록. None 이면 기본값 (0=ECG, 2=PPG, 7=RESP_Imp).
    #  modality 별 기여 분해용 — 예: [2] 면 PPG 만, [0] 이면 ECG 만 차단.
    gated_cond_signal_types: list[int] | None = None
    #  C: cond 입력 스케일 변환. 실측상 scale 의 modality 간 동적범위가 58.6배라
    #  (ECG 0.221 vs CO2 12.95) raw Linear 입력으로는 작은 modality 가 묻힌다.
    #  none(기존) | logscale | logboth | permod
    cond_transform: str = "none"
    #  D: 학습 중 variate 단위로 cond 를 통째로 떨어뜨리는 확률 (0=비활성, inference 무영향).
    #  근거: cond 가 코호트/장비 공변량 지름길로 쓰이면 파형 경로가 덜 학습된다.
    #  일부 variate 의 cond 를 무작위로 지워 파형만으로도 서게 만든다 (CFG 학습과 동형).
    cond_dropout_prob: float = 0.0
    #  D': cond dropout 을 적용할 signal_type 목록. None 이면 전 modality (기존 K4 동작).
    #  근거(2026-09-10): K4 는 ABP cond(=MAP·맥압, IOH 라벨 그 자체)까지 25% 지워 IOH-ABP 가
    #  covariate null 아래로 붕괴(-0.039). calibrated modality 는 불가침이라는 게이팅 원칙과
    #  맞추려면 unitless(ECG=0·PPG=2·RESP_Imp=7)에만 걸어야 한다.
    cond_dropout_signal_types: list[int] | None = None

    def to_dict(self) -> dict:
        """Checkpoint 저장용 직렬화."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        """dict에서 ModelConfig 복원.

        알 수 없는 키는 무시하여 이전 버전 checkpoint와 호환한다.
        """
        valid_keys = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})
