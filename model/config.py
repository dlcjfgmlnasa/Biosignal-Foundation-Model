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
        signal_type + spatial_id 이중 임베딩 사용 여부.
    dropout_p:
        드롭아웃 확률.
    num_signal_types:
        신호 타입 수 (ECG, ABP, EEG, PPG, CVP, CO2, AWP).
    num_spatial_ids:
        글로벌 spatial ID 수.
    max_horizon:
        Next-patch prediction 최대 예측 거리.
    use_cnn_stem:
        Modality-specific 1D-CNN stem 사용 여부.
    stem_hidden_channels:
        CNN stem 중간 채널 수.
    stem_num_layers:
        CNN stem Conv1d 레이어 수.
    stem_kernel_size:
        CNN stem 커널 크기.
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
    use_rope: bool = True
    use_var_attn_bias: bool = True
    use_spatial_embed: bool = True
    dropout_p: float = 0.0

    # Signal types
    num_signal_types: int = 7
    num_spatial_ids: int = 55

    # Task
    max_horizon: int = 1

    # CNN Stem
    use_cnn_stem: bool = False
    stem_hidden_channels: int = 64
    stem_num_layers: int = 3
    stem_kernel_size: int = 3

    # Contrastive
    contrastive_proj_dim: int = 0  # 0=비활성, >0=projection head 출력 차원

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
