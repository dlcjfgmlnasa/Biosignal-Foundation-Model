# 모델 아키텍처 보고서

> **Biosignal Foundation Model** — 수술중 생체신호(ECG, ABP, EEG, PPG, CVP, CO2, AWP) 범용 사전학습 모델의 아키텍처 상세 문서.

---

## 1. 전체 구조 개요

모델은 **Scaler -> PatchEmbedding -> SpatialEmbedding -> TransformerEncoder -> TaskHead** 파이프라인으로 구성된다. 단일 Transformer Encoder가 멀티태스크(Masked Patch Modeling, Next-Patch Prediction)를 수행하며, attention mask 전환으로 양방향/인과적(causal) 인코딩을 지원한다.

### 1.1 Forward 데이터 흐름

```
PackedBatch.values (B, L)
  -> PackedStdScaler -> normalized (B, L), loc/scale (B, L, 1)
  -> PatchEmbedding.patchify -> patches (B, N, P), metadata (sample_id, variate_id, time_id, patch_mask)
  -> [V2: EEG 패치 -> ModalityCNNStem -> stem_features (B, N, d_model), stop-grad target 저장]
  -> PatchEmbedding.project -> embedded (B, N, d_model)
  -> + signal_type_embed + spatial_id_embed  (Dual Additive Embedding)
  -> + loc_proj + scale_proj                  (Loc/Scale Injection)
  -> TransformerEncoder (GQA + GLU FFN + RoPE + VarBias)
  -> Task Heads:
      masked:    head -> (B, N, P),  cross_head -> (B, N, P)
      next_pred: next_head + horizon_embed -> (B, N, P)
      both:      위 전부 출력 (단일 forward, DDP 호환)
      V2 추가:   eeg_recon_head -> (B, N, d_model) [EEG 위치만 유효]
```

### 1.2 입력 형식

입력은 `PackedBatch` (NamedTuple)로, FFD bin-packing collate(`PackCollate`)가 여러 sample/variate를 하나의 row에 밀집 패킹한다.

| 필드 | Shape | 설명 |
|------|-------|------|
| `values` | `(B, L)` | 1D 연속 신호값 |
| `sample_id` | `(B, L)` long | 환자/샘플 구분 (0=패딩) |
| `variate_id` | `(B, L)` long | variate(채널) 구분 (0=패딩) |
| `signal_types` | `(total_variates,)` long | 전역 signal type 매핑 |
| `spatial_ids` | `(total_variates,)` long | 전역 spatial ID 매핑 |

---

## 2. module/ 상세

### 2.1 `module/norm.py` — RMSNorm

Root Mean Square Layer Normalization. LayerNorm의 mean-centering을 생략하고 RMS 기반 정규화만 수행한다. 계산량이 적고 Transformer에서 LayerNorm과 동등 이상의 성능을 보인다.

```python
class RMSNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        eps: float = 1e-5,
        weight: bool = True,           # 학습 가능한 gamma 스케일
        dtype: torch.dtype | None = None,
    ) -> None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (*batch, *normalized_shape) -> (*batch, *normalized_shape)
        # output = x * rsqrt(mean(x^2) + eps) * gamma
```

**설계 의도**: 프로젝트 전반에서 Transformer의 기본 정규화 레이어로 사용된다. Q/K norm, encoder layer pre-norm, 최종 출력 norm 모두 RMSNorm 기반이다.

---

### 2.2 `module/attention.py` — GQA, MHA, MQA

#### GroupedQueryAttention (GQA)

핵심 어텐션 모듈. Q는 전체 헤드 수만큼, K/V는 그룹 수만큼 projection하여 그룹 내 헤드가 K/V를 공유한다. `num_groups=num_heads`이면 MHA, `num_groups=1`이면 MQA.

```python
class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        dim: int,                      # 입출력 차원 (= d_model)
        num_heads: int,                # 어텐션 헤드 수
        num_groups: int,               # K/V 그룹 수
        bias: bool = True,
        norm_layer = RMSNorm,          # Q/K norm
        softmax_scale: float | None = None,  # 1/sqrt(head_dim)
        attn_dropout_p: float = 0.0,
        var_attn_bias: Callable | None = None,   # BinaryAttentionBias
        time_attn_bias: Callable | None = None,
        var_qk_proj: Callable | None = None,
        time_qk_proj: Callable | None = None,    # RoPE
    ) -> None

    def forward(
        self,
        query: torch.Tensor,    # (*batch, q_len, dim)
        key: torch.Tensor,      # (*batch, kv_len, dim)
        value: torch.Tensor,    # (*batch, kv_len, dim)
        attn_mask: torch.Tensor | None = None,    # (*batch, q_len, kv_len) bool
        query_var_id: torch.Tensor | None = None,  # (*batch, q_len) long
        kv_var_id: torch.Tensor | None = None,      # (*batch, kv_len) long
        query_time_id: torch.Tensor | None = None,  # (*batch, q_len) long
        kv_time_id: torch.Tensor | None = None,     # (*batch, kv_len) long
    ) -> torch.Tensor:  # (*batch, q_len, dim)
```

**내부 shape 흐름**:

```
Q: (*, q_len, dim) -> q_proj -> (*, q_len, dim) -> reshape (*, group, hpg, q_len, head_dim) -> q_norm
K: (*, kv_len, dim) -> k_proj -> (*, kv_len, group*head_dim) -> reshape (*, group, 1, kv_len, head_dim) -> k_norm -> expand (*, group, hpg, kv_len, head_dim)
V: (*, kv_len, dim) -> v_proj -> (*, kv_len, group*head_dim) -> reshape (*, group, 1, kv_len, head_dim) -> expand (*, group, hpg, kv_len, head_dim)
-> QK projection (RoPE 등)
-> attn_mask 갱신 (variate bias 합산)
-> F.scaled_dot_product_attention
-> reshape (*, q_len, dim) -> out_proj
```

**RoPE 적용 방식**: `time_qk_proj`로 `QueryKeyProjection(proj_layer=RotaryProjection)`을 전달하면, Q/K에 time_id 기반 RoPE가 적용된다. `partial_factor`로 head_dim의 일부분에만 적용 가능하다.

#### MultiHeadAttention (MHA), MultiQueryAttention (MQA)

GQA의 특수 케이스로 구현된 서브클래스:

```python
class MultiHeadAttention(GroupedQueryAttention):
    # num_groups = num_heads (각 헤드 독립 K/V)

class MultiQueryAttention(GroupedQueryAttention):
    # num_groups = 1 (모든 헤드가 단일 K/V 공유)
```

---

### 2.3 `module/ffn.py` — FeedForward, GLU FFN, MoE FFN

#### FeedForward

표준 2-layer FFN (fc1 -> activation -> fc2).

```python
class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int | None = None,    # 기본 4 * in_dim
        out_dim: int | None = None,        # 기본 in_dim
        activation = F.gelu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ) -> None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> (..., out_dim)
        # gelu(fc1(x)) -> fc2
```

#### GatedLinearUnitFeedForward (GLU FFN)

SiLU-gated FFN. `hidden_dim = ceil(2/3 * 4 * in_dim / 8) * 8`로 파라미터 수를 표준 FFN과 유사하게 조절한다. LLaMA/Gemma 등 최신 Transformer 표준.

```python
class GatedLinearUnitFeedForward(FeedForward):
    # forward: silu(fc_gate(x)) * fc1(x) -> fc2
    # hidden_dim = adjust_hidden_dim(4 * in_dim) = (int(dim * 2/3) + 7) // 8 * 8
```

#### MoEFeedForward (Mixture of Experts)

Switch Transformer 방식의 learned linear gate. 토큰별 top-k expert를 선택하고 softmax 가중치로 출력을 합산한다.

```python
class MoEFeedForward(nn.Module):
    def __init__(
        self,
        num_experts: int,               # 전문가 수 (기본 32)
        num_experts_per_token: int,      # 토큰당 활성 전문가 (기본 2)
        in_dim: int,
        hidden_dim: int | None = None,
        out_dim: int | None = None,
        activation = F.silu,
        bias: bool = True,
        ffn_dropout_p: float = 0.0,
    ) -> None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., in_dim) -> (..., dim)
        # gate_logits = gate(x)  # (T, E)
        # top-k selection -> weighted expert dispatch
        # self.aux_loss: load balancing loss (E * sum(f_i * P_i))
```

**라우팅 모니터링**: `get_routing_stats()` 메서드로 expert_load, routing_entropy, max_min_ratio 조회 가능. `_expert_counts`, `_routing_entropy` 속성이 매 forward마다 갱신된다.

---

### 2.4 `module/patch.py` — PatchEmbedding

연속 신호를 고정 크기 패치 단위로 분할하고 d_model 차원으로 투영한다.

```python
class PatchEmbedding(nn.Module):
    def __init__(
        self,
        patch_size: int,           # 패치 시간 길이
        d_model: int,              # 출력 임베딩 차원
        stride: int | None = None, # None이면 non-overlapping
        bias: bool = True,
        stem: nn.Module | None = None,  # CNN stem (있으면 linear proj 대체)
    ) -> None

    def patchify(
        self,
        values: torch.Tensor,       # (B, L)
        sample_id: torch.Tensor,     # (B, L) long
        variate_id: torch.Tensor,    # (B, L) long
    ) -> tuple[
        torch.Tensor,  # patches:    (B, N, P)
        torch.Tensor,  # sample_id:  (B, N) long
        torch.Tensor,  # variate_id: (B, N) long
        torch.Tensor,  # time_id:    (B, N) long  -- variate 내 순서 인덱스
        torch.Tensor,  # patch_mask: (B, N) bool  -- True=유효
    ]

    def project(
        self,
        patches: torch.Tensor,                  # (B, N, P)
        patch_signal_types: torch.Tensor | None, # (B, N) long
    ) -> torch.Tensor:  # (B, N, d_model)
        # stem이 있으면 ModalityCNNStem, 없으면 nn.Linear(P, d_model)
```

**Non-overlapping**: `L % P == 0` 필수. `values.reshape(B, N, P)`로 분할.

**Overlapping**: `stride < patch_size` 시 `values.unfold(-1, P, S)` 사용. 패치 유효 조건: P개 위치 모두 동일 (sample_id, variate_id).

**time_id 계산**: 같은 (sample_id, variate_id) 조합의 연속 패치에 대해 0부터 시작하는 순차 인덱스 부여. RoPE의 위치 정보로 사용되며, cross-modal 페어링(동일 시간 인덱스의 다른 variate 간 매칭)에도 활용된다.

---

### 2.5 `module/packed_scaler.py` — PackedStdScaler, PackedAbsMeanScaler

Packed batch에서 (sample_id, variate_id) 그룹별 정규화를 수행한다. scatter_add 기반 O(L) 구현으로 효율적이다.

#### PackedStdScaler (Z-score)

```python
class PackedStdScaler(PackedScaler):
    def __init__(self, correction: int = 1, minimum_scale: float = 1e-5) -> None

    def forward(
        self,
        target: torch.Tensor,    # (B, L, D)
        observed_mask=None, sample_id=None, variate_id=None,
    ) -> tuple[
        torch.Tensor,  # loc:   (B, L, D) -- 그룹별 평균
        torch.Tensor,  # scale: (B, L, D) -- 그룹별 표준편차 (Bessel 보정)
    ]
```

**설계 의도**: 서로 다른 환자/채널이 하나의 row에 패킹되므로, (sample_id, variate_id) 조합을 group key로 변환하여 per-variate 정규화를 수행한다. 패딩 위치(sample_id=0)는 loc=0, scale=1로 초기화.

#### PackedAbsMeanScaler (절대 평균)

```python
class PackedAbsMeanScaler(PackedScaler):
    def __init__(self, minimum_scale: float = 1e-5) -> None
    # loc = 0, scale = mean(|x|) per group
```

---

### 2.6 `module/transformer.py` — TransformerEncoder, TransformerEncoderLayer

#### TransformerEncoderLayer

Pre-norm 구조의 단일 레이어: RMSNorm -> Self-Attention -> Residual -> RMSNorm -> FFN -> Residual.

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,           # GLU FFN 또는 MoE FFN
        norm1: nn.Module | None,    # RMSNorm
        norm2: nn.Module | None,    # RMSNorm
        post_attn_dropout_p: float = 0.0,
        pre_norm: bool = True,
    ) -> None

    def forward(
        self,
        x: torch.Tensor,        # (*batch, time_len, dim)
        attn_mask=None,          # (*batch, time_len, time_len) bool
        var_id=None,             # (*batch, time_len) long
        time_id=None,            # (*batch, time_len) long
    ) -> torch.Tensor:           # (*batch, time_len, dim)
```

#### TransformerEncoder

N개의 TransformerEncoderLayer를 스택하고 최종 RMSNorm을 적용한다.

```python
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: int | None = None,    # 기본 d_model // 64
        num_groups: int | None = None,   # 기본 num_heads (MHA)
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer = RMSNorm,
        activation = F.silu,
        use_moe: bool = False,           # MoE FFN 사용
        use_glu: bool = True,            # GLU FFN 사용
        use_qk_norm: bool = True,        # Q/K RMSNorm
        var_attn_bias_layer=None,        # BinaryAttentionBias
        time_qk_proj_layer=None,         # RoPE
        d_ff: int | None = None,         # FFN hidden dim
    ) -> None

    def forward(
        self,
        x: torch.Tensor,     # (*batch, time_len, dim)
        attn_mask=None,       # (*batch, time_len, time_len) bool
        var_id=None,          # (*batch, time_len) long
        time_id=None,         # (*batch, time_len) long
    ) -> torch.Tensor:        # (*batch, time_len, dim)
        # for layer in self.layers: x = layer(x, ...)
        # return self.norm(x)
```

---

### 2.7 `module/position/` — RoPE, BinaryAttentionBias

#### RotaryProjection (RoPE)

시간 위치 인코딩. `time_id` 기반으로 Q/K에 회전 변환을 적용한다. cos/sin 캐시를 자동 확장하여 가변 길이 시퀀스를 지원한다.

```python
class RotaryProjection(Projection):
    def __init__(
        self,
        proj_width: int,           # 짝수 필수 (= head_dim)
        num_heads: int,
        num_groups: int,
        max_len: int = 512,        # 자동 확장
        base: int = 10000,
    ) -> None

    def forward(
        self, x: torch.Tensor, seq_id: torch.Tensor | None
    ) -> torch.Tensor:
        # cos(m*theta) * x + sin(m*theta) * rotate(x)
```

#### QueryKeyProjection

Q/K에 RoPE 등의 projection을 적용하는 래퍼. `partial_factor`로 head_dim의 일부분에만 적용 가능.

```python
class QueryKeyProjection(nn.Module):
    def __init__(
        self,
        dim: int, num_heads: int, num_groups: int,
        proj_layer: type[Projection],     # RotaryProjection
        partial_factor: tuple[float, float] | None = None,
    ) -> None
```

#### BinaryAttentionBias

같은 variate_id 여부에 따라 학습 가능한 2개의 bias를 적용한다. 같은 채널 내 토큰은 stronger attention, 다른 채널 간 토큰은 weaker (또는 그 반대를 학습).

```python
class BinaryAttentionBias(AttentionBias):
    def __init__(self, dim: int, num_heads: int, num_groups: int) -> None
        # self.emb = nn.Embedding(2, num_heads)
        # emb[0]: 다른 variate 간 bias, emb[1]: 같은 variate 내 bias
```

---

### 2.8 `module/cnn_stem.py` — Conv1dStem, ModalityCNNStem

#### Conv1dStem

단일 modality용 1D-CNN 패치 임베딩. `padding="same"` + `AdaptiveAvgPool1d(1)` 구조.

```python
class Conv1dStem(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_channels: int = 64,
        num_layers: int = 3,       # 최소 2
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # (M, P) -> unsqueeze(1) -> (M, 1, P)
        # -> Conv1d layers (GELU activation) -> (M, d_model, P)
        # -> AdaptiveAvgPool1d(1) -> (M, d_model, 1)
        # -> squeeze -> (M, d_model)
```

**구조**: `Conv1d(1, H, k, same) -> GELU -> [Conv1d(H, H, k, same) -> GELU] * (L-2) -> Conv1d(H, d_model, 1) -> AvgPool1d(1)`

#### ModalityCNNStem

신호 타입별 전용 Conv1dStem을 보유하고, per-patch signal_type에 따라 vectorized dispatch한다.

```python
class ModalityCNNStem(nn.Module):
    def __init__(
        self,
        num_signal_types: int,     # 7 (ECG, ABP, EEG, PPG, CVP, CO2, AWP)
        d_model: int,
        hidden_channels: int = 64,
        num_layers: int = 3,
        kernel_size: int = 3,
    ) -> None

    def forward(
        self,
        patches: torch.Tensor,      # (B, N, P)
        signal_types: torch.Tensor,  # (B, N) long
    ) -> torch.Tensor:               # (B, N, d_model)
```

---

## 3. model/ 상세

### 3.1 `model/config.py` — ModelConfig

모든 아키텍처 하이퍼파라미터를 하나의 dataclass로 통합한다. checkpoint 직렬화 (`to_dict()`/`from_dict()`) 및 YAML 저장을 지원한다.

```python
@dataclass
class ModelConfig:
    # Architecture
    d_model: int = 64
    num_layers: int = 2
    patch_size: int = 100
    stride: int | None = None
    num_heads: int | None = None     # 기본 d_model // 64
    num_groups: int | None = None    # 기본 num_heads (MHA)

    # Features
    use_glu: bool = True
    use_moe: bool = False
    use_rope: bool = True
    use_var_attn_bias: bool = True
    use_spatial_embed: bool = True
    dropout_p: float = 0.0

    # Signal types
    num_signal_types: int = 7
    num_spatial_ids: int = 12

    # Task
    max_horizon: int = 1

    # CNN Stem
    use_cnn_stem: bool = False
    stem_hidden_channels: int = 64
    stem_num_layers: int = 3
    stem_kernel_size: int = 3

    # Contrastive
    contrastive_proj_dim: int = 0    # 0=비활성
```

### 3.2 `model/biosignal_model.py` — BiosignalFoundationModel (Deprecated)

초기 구현. 현재는 V1/V2로 대체되어 deprecated 상태. `_encode()` 분리 없이 단일 forward에 모든 로직이 포함된다.

### 3.3 `model/v1.py` — BiosignalFoundationModelV1

V1은 모든 signal type에 대해 동일하게 raw patch reconstruction을 수행한다. `_encode()`로 공통 인코딩 파이프라인을 분리하여 V2에서 상속 확장한다.

#### _encode() 파이프라인

```python
def _encode(self, batch: PackedBatch, task: str = "masked") -> dict[str, torch.Tensor]:
    # 1. Scaler: (B, L) -> normalized (B, L), loc/scale (B, L, 1)
    # 2. Patchify: (B, L) -> patches (B, N, P), metadata
    # 3. global_var_idx 계산: spatial_ids/signal_types -> patch-level 매핑
    # 4. Project: patches (B, N, P) -> embedded (B, N, d_model)
    # 5. Spatial Embed: + signal_type_embed + spatial_id_embed
    # 6. Loc/Scale Injection: + loc_proj(loc) + scale_proj(scale)
    # 7. Attention Mask: same-sample mask & validity mask
    # 8. Encoder: task별 mask 전환 후 TransformerEncoder 호출
    #    - "masked": bidirectional
    #    - "next_pred": causal (lower triangular)
    #    - "both": encoder 2회 호출 (bi + causal), DDP single forward 호환
```

#### 멀티태스크 지원

| Task | Attention | Head | 출력 |
|------|-----------|------|------|
| `"masked"` | 양방향 | `head` (d_model -> P), `cross_head` (d_model -> P) | `reconstructed`, `cross_pred` |
| `"next_pred"` | Causal | `next_head` + `horizon_embed` | `next_pred` |
| `"both"` | 양방향 + Causal | 전부 | 전부 (encoder 2회) |

**Horizon Embedding**: next-patch prediction의 예측 거리(H)를 조건화한다. `horizon_embed(H-1)`을 encoded에 더하여 "H-step 앞 예측"을 모델에 알린다. `max_horizon`까지 지원.

#### signal_type + spatial_id 이중 임베딩 (Dual Additive Embedding)

```python
self.signal_type_embed = nn.Embedding(7, d_model)   # ECG(0), ABP(1), EEG(2), PPG(3), CVP(4), CO2(5), AWP(6)
self.spatial_id_embed = nn.Embedding(12, d_model)    # 전역 sensor location ID

# forward:
embedded = embedded + (sig_emb + spa_emb) * valid_mask  # 패딩 위치 보호
```

**설계 의도**: signal_type은 신호의 물리적 특성(파형 형태, 주파수 대역)을 인코딩하고, spatial_id는 센서 위치(예: lead I vs lead II, EEG F3 vs C4)를 인코딩한다. Additive 방식으로 합산하여 토큰에 "어떤 종류의 신호가 어디서 측정되었는가"를 알린다.

#### Loc/Scale Injection

per-variate 정규화로 제거된 절대 레벨 정보를 보존한다.

```python
self.loc_proj = nn.Linear(1, d_model)    # 평균값 -> d_model
self.scale_proj = nn.Linear(1, d_model)  # 표준편차 -> d_model

# forward: 각 패치 시작점의 loc/scale을 샘플링하여 embedded에 더함
embedded = embedded + (loc_proj(loc) + scale_proj(scale)) * valid_mask
```

**설계 의도**: Z-score 정규화는 상대적 패턴을 학습하기 좋지만, 절대 혈압 80mmHg vs 120mmHg 같은 임상적으로 중요한 레벨 정보가 손실된다. 이를 선형 투영으로 d_model 공간에 주입하여 보존한다.

#### Task Head 구성

| Head | 파라미터 | 입력 -> 출력 | 용도 |
|------|---------|-------------|------|
| `head` | `nn.Linear(d_model, P)` | `(B, N, d_model) -> (B, N, P)` | 자기 variate 복원 (MPM) |
| `next_head` | `nn.Linear(d_model, P)` | `(B, N, d_model) -> (B, N, P)` | same-variate next-patch 예측 |
| `cross_head` | `nn.Linear(d_model, P)` | `(B, N, d_model) -> (B, N, P)` | cross-modal 예측 |
| `horizon_embed` | `nn.Embedding(H, d_model)` | `int -> (d_model,)` | horizon conditioning |
| `contrastive_proj` | `MLP(d_model -> d_model -> proj_dim)` | `(B, N, d_model) -> (B, N, proj_dim)` | InfoNCE contrastive (선택적) |

#### Forward 출력 구조

```python
{
    "encoded":           (B, N, d_model),  # Transformer 출력
    "patches":           (B, N, P),        # raw patches (loss target)
    "patch_signal_types":(B, N),           # 패치별 signal type
    "reconstructed":     (B, N, P),        # masked recon (task=masked/both)
    "cross_pred":        (B, N, P),        # cross-modal pred (task=masked/both)
    "next_pred":         (B, N, P),        # next-patch pred (task=next_pred/both)
    "contrastive_z":     (B, N, proj_dim), # contrastive proj (선택적)
    "loc":               (B, L, 1),
    "scale":             (B, L, 1),
    "patch_mask":        (B, N),           # True=유효
    "patch_sample_id":   (B, N),
    "patch_variate_id":  (B, N),
    "time_id":           (B, N),
}
```

#### Inference API

| 메서드 | 설명 | 출력 Shape |
|--------|------|-----------|
| `extract_features(batch)` | 양방향 feature 추출 (downstream용) | dict (encoded 등) |
| `forecast(batch, horizon, denormalize)` | 단일-step 예측 + denormalize | `(B, N, P)` |
| `generate(batch, n_steps, denormalize)` | autoregressive 다단계 생성 | `(n_steps, B, P)` |

---

## 4. V1 vs V2 설계

### 4.1 핵심 차이

| 항목 | V1 | V2 |
|------|----|----|
| **EEG reconstruction target** | raw patch (`patch_size` 차원) | CNN stem 출력 (`d_model` 차원, stop-grad) |
| **EEG reconstruction head** | `head` (d_model -> P) | `eeg_recon_head` (d_model -> d_model) |
| **기타 신호** | raw patch 복원 동일 | 동일 |
| **Loss 복잡도** | 단일 MSE | MPM + EEG stem MSE |
| **추가 출력** | 없음 | `eeg_reconstructed`, `eeg_recon_target`, `eeg_mask` |

### 4.2 V2 EEG 복원 전략

**배경**: EEG는 비정상적(non-stationary) 확률 과정에 가까워 raw time-domain MSE 복원이 blurry mean으로 수렴한다. 유의미한 정보가 주파수 대역 파워에 인코딩되어 있어, raw 복원은 임상적으로 무의미하다.

**접근**: data2vec simplified 방식. CNN stem이 raw patch에서 저수준 특징을 추출하고, encoder는 stem 출력을 reconstruction target으로 사용한다. Stop-gradient (`.detach()`)로 encoder가 stem을 통해 trivial solution을 학습하는 것을 방지한다.

```python
class BiosignalFoundationModelV2(BiosignalFoundationModelV1):
    def __init__(self, ...):
        super().__init__(...)
        self.eeg_recon_head = nn.Linear(d_model, d_model)  # EEG 전용

    def forward(self, batch, task, horizon):
        enc = self._encode(batch, task=task)
        # ...V1 heads 동일...
        if patch_signal_types is not None:
            eeg_mask = patch_signal_types == 2  # EEG
            eeg_reconstructed = self.eeg_recon_head(encoded)   # (B, N, d_model)
            eeg_recon_target = stem_output.detach()            # stop-grad
```

### 4.3 V2 추가 출력

```python
{
    # ...V1의 모든 키...
    "eeg_reconstructed": (B, N, d_model),  # EEG head 출력 (전체 위치)
    "eeg_recon_target":  (B, N, d_model),  # stem 출력 detached (전체 위치)
    "eeg_mask":          (B, N),           # True=EEG 패치
}
```

---

## 5. 주요 하이퍼파라미터

### 5.1 현재 학습 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `d_model` | 128 | Transformer 임베딩 차원 |
| `num_layers` | 8 | Transformer 레이어 수 |
| `patch_size` | 100 | 패치 크기 (100Hz에서 1초) |
| `num_heads` | 2 (= 128 // 64) | 어텐션 헤드 수 |
| `num_groups` | 2 (= num_heads) | GQA 그룹 수 (MHA) |
| `head_dim` | 64 (= d_model / num_heads) | 헤드 차원 |
| `use_glu` | True | GLU FFN |
| `use_rope` | True | Rotary Position Embedding |
| `use_var_attn_bias` | True | Binary variate attention bias |
| `use_cnn_stem` | True | Modality-specific CNN stem |
| `max_horizon` | 5 | 최대 예측 거리 |
| `num_signal_types` | 7 | ECG, ABP, EEG, PPG, CVP, CO2, AWP |
| `num_spatial_ids` | 12 | 전역 sensor location IDs |

### 5.2 파라미터 수 계산 (추정)

**GLU FFN hidden_dim**: `(int(4 * 128 * 2/3) + 7) // 8 * 8 = 344`

| 구성요소 | 파라미터 수 (per layer) |
|---------|----------------------|
| GQA (Q/K/V proj + out_proj) | 128\*128 + 128\*64\*2 + 128\*128 = 49,152 |
| Q/K RMSNorm | 64 * 2 = 128 |
| GLU FFN (fc1 + fc_gate + fc2) | 128\*344 + 128\*344 + 344\*128 = 132,096 |
| Layer Norms (x2) | 128 * 2 = 256 |
| **Layer 합계** | ~181,632 |

| 구성요소 | 파라미터 수 (전체) |
|---------|------------------|
| Encoder (8 layers) | ~1,453,056 |
| Final RMSNorm | 128 |
| PatchEmbedding (proj or stem) | stem: ~7\*Conv1dStem |
| Signal type / Spatial ID embed | 7\*128 + 12\*128 = 2,432 |
| Loc/Scale proj | (1\*128+128)\*2 = 516 |
| Heads (head, next, cross) | (128\*100)\*3 = 38,400 |
| Horizon embed | 5\*128 = 640 |
| BinaryAttentionBias (x8) | 2\*2\*8 = 32 |
| RoPE | 0 (no learnable params) |
| **총 추정** | **~6.4M** (CI Base config) |

### 5.3 스케일 정의 (계획)

| 스케일 | d_model | num_layers | 추정 파라미터 |
|--------|---------|------------|-------------|
| Small | 64 | 2 | ~36K |
| Base | 128 | 8 | ~6.4M |
| Large | 256 | 12 | TBD |

---

## 6. DDP 호환 설계

- `task="both"`: 단일 `model.forward()` 호출 내에서 encoder를 2회 호출(bidirectional + causal)하여 DDP unused parameters 문제를 해결한다.
- `find_unused_parameters=True`, `static_graph=True` 설정으로 DDP wrapper가 gradient sync를 자동 처리한다.
- Manual `dist.all_reduce` gradient sync는 제거되었다.
