# Biosignal Foundation Model Wiki

> 수술중 모니터링(VitalDB, K-MIMIC) 생체신호를 위한 Self-Supervised Foundation Model의 전체 설계, 구현, 학습, 평가를 상세히 기술한다.

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [모델 아키텍처](#2-모델-아키텍처)
3. [데이터 파이프라인](#3-데이터-파이프라인)
4. [사전학습 Phase 1 (Channel-Independent)](#4-사전학습-phase-1-channel-independent)
5. [사전학습 Phase 2 (Any-Variate Cross-Modal)](#5-사전학습-phase-2-any-variate-cross-modal)
6. [Loss 함수 상세](#6-loss-함수-상세)
7. [Cross-Modal Prediction](#7-cross-modal-prediction)
8. [Downstream Tasks](#8-downstream-tasks)
9. [시각화](#9-시각화)
10. [실행 가이드](#10-실행-가이드)

---

## 1. 프로젝트 개요

### 1.1 목표

수술중 및 ICU 모니터링 장비에서 수집되는 **다양한 생체신호**(ECG, ABP, PPG, CVP, CO2, AWP, PAP, ICP)를 **단일 Transformer 기반 모델**로 통합 표현학습하고, 3개 카테고리에 걸친 9개 다운스트림 임상 태스크 — **Acute Event Detection** (Intraoperative Hypotension Prediction, Arrhythmia Classification, Intracranial Hypertension), **Outcome Prediction** (Sepsis, Mortality, Cardiac Arrest, Postoperative Acute Kidney Injury), **Physiological Generation** (Cross-Modal Reconstruction, Waveform Forecasting) — 에 활용할 수 있는 범용 foundation model을 구축한다.

### 1.2 대상 신호 (8종)

| signal_type | 코드 | 설명 | 데이터 소스 | 주파수 범위 |
|---|---|---|---|---|
| ECG | 0 | 심전도 | VitalDB, K-MIMIC | 0.5~40Hz (bandpass) |
| ABP | 1 | 동맥혈압 | VitalDB, K-MIMIC | DC~15Hz (lowpass) |
| PPG | 2 | 맥파 | VitalDB, K-MIMIC | DC~8Hz (lowpass) |
| CVP | 3 | 중심정맥압 | VitalDB, K-MIMIC | DC~10Hz (lowpass) |
| CO2 | 4 | 호기말CO2 | VitalDB | DC~5Hz (lowpass) |
| AWP | 5 | 기도압 | VitalDB | DC~20Hz (lowpass) |
| PAP | 6 | 폐동맥압 | K-MIMIC | DC~15Hz (lowpass) |
| ICP | 7 | 두개내압 | K-MIMIC | DC~10Hz (lowpass) |

### 1.3 사전학습 전략 (2-Phase Curriculum)

```
Phase 1: Channel-Independent (CI)
  - 각 채널을 독립적으로 학습 (collate_mode="ci")
  - Masked Patch Modeling (MPM) + Next-Patch Prediction
  - 단일 신호의 시간적 패턴 학습

Phase 2: Any-Variate Cross-Modal (AV)
  - 다변량 세션에서 교차 모달 학습 (collate_mode="any_variate")
  - Phase 1 checkpoint에서 시작
  - Cross-Modal MSE + Contrastive Loss + Variate Masking/Dropout
  - 신호 간 생리학적 관계 학습 (Virtual Sensing)
```

### 1.4 데이터 소스

- **VitalDB Open (SNUADC/)**: 수술실(OR), 500Hz, ~6,388 cases, ECG/ABP/PPG/CVP/CO2/AWP
- **K-MIMIC-MORTAL (SNUADCM/)**: ICU, 500Hz, ECG/ABP/PPG/CVP/PAP/ICP

모든 신호는 전처리 후 **100Hz**로 통일 리샘플링되어 `.pt` 파일로 저장된다.

---

## 2. 모델 아키텍처

### 2.1 전체 파이프라인

```
Input PackedBatch
    |
    v
[1] PackedStdScaler           -- per-variate Z-score 정규화 (loc, scale)
    |
    v
[2] PatchEmbedding.patchify() -- (B, L) -> (B, N, P) raw patches 추출
    |
    v
[3] PatchEmbedding.project()  -- Residual MLP: (B, N, P) -> (B, N, d_model)
    |
    v
[4] Dual Spatial Embedding    -- signal_type_embed + spatial_id_embed (additive)
    |
    v
[5] Loc/Scale AdaLN Conditioning  -- cond_proj([loc, scale]) -> ada_cond (B, N, d_cond)
                                     ada_cond는 모든 transformer layer의 AdaRMSNorm으로 주입됨
    |
    v
[6] [MASK] Token Replacement  -- pred_mask=True인 패치를 learnable mask_token으로 교체
    |
    v
[7] Attention Mask 구성        -- sample_id 기반 + causal mask (task별)
    |
    v
[8] TransformerEncoder         -- GQA + GLU FFN + RoPE + BinaryAttentionBias
    |
    v
[9] Task-specific Heads:
    ├── head (Linear)          -> reconstructed (B, N, P)        -- 자기 variate 복원
    ├── next_head (Linear)     -> next_pred (B, N, P)            -- next-patch 예측
    ├── cross_heads (ModuleDict) -> cross_pred_per_type (B, N, T, P) -- target별 cross-modal
    └── contrastive_proj (MLP) -> contrastive_z (B, N, proj_dim)  -- InfoNCE용 projection
```

**핵심 설계 원칙**:
- **단일 Encoder, 다중 Head**: 양방향(masked) + causal(next_pred) attention을 `task="both"`로 동시 수행
- **No Separate Decoder**: causal mask은 main encoder에 적용, post-hoc decoder 없음
- **Learnable [MASK] Token**: 마스킹된 패치의 content를 learnable token으로 교체하여 정보 누출 방지

### 2.2 ModelConfig (하이퍼파라미터)

```python
@dataclass
class ModelConfig:
    d_model: int = 64           # 트랜스포머 임베딩 차원
    num_layers: int = 2         # 트랜스포머 레이어 수
    patch_size: int = 100       # 패치 크기 (100Hz에서 1초 = 100 samples)
    stride: int | None = None   # 패치 보폭 (None = patch_size, non-overlapping)
    num_heads: int | None = None  # 어텐션 헤드 수 (None = d_model // 64)
    num_groups: int | None = None # GQA 그룹 수 (None = num_heads = MHA)
    use_glu: bool = True        # Gated Linear Unit FFN
    use_moe: bool = False       # Mixture of Experts
    num_experts: int = 8
    num_experts_per_token: int = 2
    use_rope: bool = True       # Rotary Position Embedding
    use_var_attn_bias: bool = True  # BinaryAttentionBias
    use_spatial_embed: bool = True  # Dual Embedding
    dropout_p: float = 0.0
    num_signal_types: int = 7   # 신호 타입 수
    num_spatial_ids: int = 12   # 전역 spatial ID 수
    max_horizon: int = 1        # next-pred 최대 horizon
    contrastive_proj_dim: int = 0  # 0=비활성, >0=contrastive head 활성
    d_cond: int = 16            # AdaLN cond vector 차원 (loc/scale -> AdaRMSNorm)
```

**Base Config (Phase 2, ~10M params)**:
```yaml
d_model: 256
num_layers: 12
patch_size: 200       # 100Hz x 2초 = 200 samples/patch
num_heads: 8          # head_dim = 256/8 = 32
use_glu: true
use_rope: true
use_var_attn_bias: true
dropout_p: 0.1
max_horizon: 3
contrastive_proj_dim: 128
```

### 2.3 컴포넌트 상세

#### 2.3.1 PackedStdScaler (Z-score 정규화)

파일: `module/packed_scaler.py`

```
입력: (B, L, 1) raw values + sample_id + variate_id
출력: loc (B, L, 1), scale (B, L, 1)
```

- **(sample_id, variate_id) 그룹별** 평균/표준편차 계산
- `scatter_add` 기반 O(L) 구현 (O(L^2) pairwise mask 대체)
- Bessel 보정 (`correction=1`), `minimum_scale=1e-5`
- 패딩 위치(`sample_id==0`)는 `loc=0, scale=1`로 초기화

다른 변형:
- `PackedAbsMeanScaler`: 절대 평균 기반 스케일링
- `PackedNOPScaler`: 정규화 없음 (loc=0, scale=1)

#### 2.3.2 PatchEmbedding (패치 토큰화)

파일: `module/patch.py`

```
patchify(): (B, L) -> (B, N, P) raw patches
                    + patch_sample_id (B, N)
                    + patch_variate_id (B, N)
                    + time_id (B, N)    -- variate 내 순서 인덱스
                    + patch_mask (B, N) -- True=유효

project():  (B, N, P) -> (B, N, d_model) via ResidualMLP
```

**ResidualMLP (TimesFM style)**:
```python
class ResidualMLP:
    # MLP(in_dim -> hidden=2*in_dim -> out_dim) + skip(in_dim -> out_dim)
    forward(x) = MLP(x) + skip(x)
```

- Non-overlapping: `stride == patch_size`, `values.reshape(B, N, P)`
- Overlapping: `stride < patch_size`, `values.unfold(-1, P, S)`
- **time_id 계산**: `(sample_id, variate_id)` 조합이 같은 연속 패치에 0부터 순차 인덱스 부여

#### 2.3.3 Dual Spatial Embedding

```python
signal_type_embed = nn.Embedding(num_signal_types, d_model)  # ECG=0, ABP=1, ...
spatial_id_embed = nn.Embedding(num_spatial_ids, d_model)     # Lead_II=1, Radial=1, ...

# Additive injection:
embedded = embedded + (sig_emb + spa_emb) * valid_token
```

**Spatial Map 체계** (`data/spatial_map.py`):

```
ECG(0):  Unknown=0, Lead_II=1, Lead_V5=2     -> global: 0, 1, 2
ABP(1):  Unknown=0, Radial=1, Femoral=2       -> global: 3, 4, 5
PPG(2):  Unknown=0, Finger=1                   -> global: 6, 7
CVP(3):  Unknown=0                              -> global: 8
CO2(4):  Unknown=0                              -> global: 9
AWP(5):  Unknown=0                              -> global: 10
PAP(6):  Unknown=0                              -> global: 11
ICP(7):  Unknown=0                              -> global: 12

TOTAL_SPATIAL_IDS = 13
```

`get_global_spatial_id(signal_type, local_id)` -> 전역 고유 ID

#### 2.3.4 Loc/Scale AdaLN Conditioning

Z-score 정규화로 제거된 **환자별 절대 레벨 정보**(혈압 mmHg, ECG mV 등)를 모든 transformer layer에 multiplicative gating으로 주입한다.

```python
# (loc, scale) 2D scalar -> d_cond conditioning vector
cond_proj = nn.Sequential(
    nn.Linear(2, d_cond),
    nn.SiLU(),
    nn.Linear(d_cond, d_cond),
)
loc_scale = torch.cat([patch_loc, patch_scale], dim=-1)  # (B, N, 2)
ada_cond = cond_proj(loc_scale)                          # (B, N, d_cond)

# 매 transformer layer의 RMSNorm을 AdaRMSNorm으로 대체:
#   AdaRMSNorm(x; cond) = RMSNorm(x) * (1 + γ(cond)) + β(cond)
# γ, β는 Linear(d_cond, 2*d_model)로 예측. Zero-init으로 안전 도입.
```

**왜 additive embedding이 아닌 AdaLN인가**:
- Additive: input embedding 1회 더해짐 → 깊은 layer에서 정보 희석
- AdaLN: 모든 layer norm에 cond가 직접 곱해짐 → 정보 보존 + gradient 강제 흐름
- 30 epoch ablation에서 -3.3% val_total 우세 → adopted as default
- `d_cond` (default 16) 는 hyperparameter로 조정 가능

#### 2.3.5 TransformerEncoder

파일: `module/transformer.py`

```
TransformerEncoder:
  layers: N x TransformerEncoderLayer
  norm: RMSNorm(d_model)   -- 최종 normalization

TransformerEncoderLayer:
  Pre-norm 구조:
    x = x + SA(Norm1(x))
    x = x + FFN(Norm2(x))
```

#### 2.3.6 GroupedQueryAttention (GQA)

파일: `module/attention.py`

```
Q: (B, group, hpg, q_len, head_dim)    -- 전체 헤드
K: (B, group, 1, kv_len, head_dim)     -- 그룹 수만큼 (broadcast)
V: (B, group, 1, kv_len, head_dim)     -- 그룹 수만큼 (broadcast)
```

- **MHA**: `num_groups == num_heads` (각 헤드가 독립 K/V)
- **MQA**: `num_groups == 1` (모든 헤드가 단일 K/V 공유)
- **GQA**: `1 < num_groups < num_heads` (그룹 내 K/V 공유)

기능:
- **Q/K Norm**: `RMSNorm(head_dim)` -- 학습 안정성
- **RoPE** (`RotaryProjection`): time_id 기반 회전 위치 인코딩 -> **시간축 위치 정보**
- **BinaryAttentionBias**: variate_id 기반 same/different variate bias -> **채널 관계 학습**
- `F.scaled_dot_product_attention` 사용 (FlashAttention 지원)

#### 2.3.7 Feed-Forward Networks

파일: `module/ffn.py`

**Standard FFN**:
```
fc1(in_dim -> hidden_dim=4*d) -> activation -> fc2(hidden_dim -> out_dim)
```

**GatedLinearUnitFeedForward (GLU)**:
```
hidden_dim = round_up_to_8(4 * d_model * 2/3)
output = activation(gate(x)) * fc1(x)  -- element-wise gating
fc2(output)
```
- SiLU 활성화 함수
- hidden_dim = `(int(4 * d_model * 2/3) + 7) // 8 * 8`

**MoEFeedForward (Mixture of Experts)**:
```
gate: Linear(d_model -> num_experts, bias=False)
experts: num_experts x GatedLinearUnitFeedForward

Forward:
  1. gate_logits = gate(x)                -- (T, E)
  2. weights, selected = topk(gate_logits, K)  -- (T, K)
  3. 각 expert에 할당된 토큰 실행 후 가중합
  4. aux_loss = E * sum(f_i * P_i)        -- load balancing
```

라우팅 모니터링: `get_routing_stats()` -> expert_load, routing_entropy, max_min_ratio

#### 2.3.8 RMSNorm / AdaRMSNorm

파일: `module/norm.py`

**RMSNorm** (QK norm 등 standalone norm에서 사용):
```python
output = x * rsqrt(mean(x^2) + eps) * gamma

# gamma: learnable scale parameter (optional)
# eps: 1e-5 (기본)
```

LayerNorm 대비 mean subtraction 생략으로 계산 효율 향상.

**AdaRMSNorm** (Encoder의 모든 layer norm에서 사용 — AdaLN conditioning):
```python
output = RMSNorm(x) * (1 + gamma(cond)) + beta(cond)

# gamma, beta: Linear(d_cond, 2*d_model)에서 chunk으로 분리
# Zero-init: nn.init.zeros_(modulation.weight/bias)
#   -> 학습 시작 시 plain RMSNorm과 동일 -> 안전한 도입
# cond: (B, N, d_cond) — cond_proj([loc, scale])의 출력
```

- DiT (ICCV 2023)의 AdaLN-Zero 변형 채택
- 매 layer에 conditioning이 직접 작용 → 정보 희석 없음
- forward가 항상 cond를 거쳐가므로 모델이 cond를 silently ignore 못함

#### 2.3.9 Horizon Embedding

```python
horizon_embed = nn.Embedding(max_horizon, d_model)

# Forward:
h_emb = horizon_embed(horizon - 1)  # 0-indexed
encoded_h = encoded_causal + h_emb  # additive
next_pred = next_head(encoded_h)     # (B, N, P)
```

**Curriculum Transition**: 새로운 horizon embed는 이전 최고 horizon embed에서 초기화
```python
horizon_embed.weight[new_h] = horizon_embed.weight[prev_max_h - 1].clone()
```

#### 2.3.10 Task-specific Heads

```python
# 1. Reconstruction Head (자기 variate 복원)
head = nn.Linear(d_model, patch_size)  # -> (B, N, P)

# 2. Next-Patch Prediction Head
next_head = nn.Linear(d_model, patch_size)  # -> (B, N, P)

# 3. Cross-Modal Prediction Heads (target signal type별 독립)
cross_heads = nn.ModuleDict({
    str(st): nn.Linear(d_model, patch_size)
    for st in range(num_signal_types)
})
# -> (B, N, num_signal_types, P) via stack

# 4. Contrastive Projection Head (SimCLR-style)
contrastive_proj = nn.Sequential(
    nn.Linear(d_model, d_model),
    nn.GELU(),
    nn.Linear(d_model, contrastive_proj_dim),
)  # -> (B, N, proj_dim)
```

### 2.4 Attention Mask 구성

```python
# Base mask: 같은 sample 내에서만 attend + 유효 패치만
base_attn_mask = (p_sid.unsqueeze(-1) == p_sid.unsqueeze(-2))
               & patch_mask.unsqueeze(-2)
               & patch_mask.unsqueeze(-1)
# shape: (B, N, N)

# Causal mask (next_pred, both):
causal_mask = base_attn_mask & tril(ones(N, N))

# Variate Dropout mask:
# drop_mask = True인 variate의 토큰은 attend 불가 + content를 mask_token으로 교체
base_attn_mask = base_attn_mask & keep.unsqueeze(-1) & keep.unsqueeze(-2)
```

### 2.5 Inference API

```python
# 1. Feature 추출 (downstream용)
model.extract_features(batch) -> {"encoded": (B, N, d_model), "patch_mask": ...}

# 2. Zero-shot Cross-Modal Generation (Virtual Token Injection)
model.generate_cross_modal(batch, target_signal_type=1)
# -> {"waveform": (B, N, P), "patch_mask": (B, N)}

# 3. 단일-step Forecasting
model.forecast(batch, horizon=1) -> (B, N, P)

# 4. Autoregressive Multi-step Generation
model.generate(batch, n_steps=10) -> (n_steps, B, P)
```

---

## 3. 데이터 파이프라인

### 3.1 데이터 소스

| 소스 | 위치 | 환경 | 신호 | 원본 SR | 비고 |
|---|---|---|---|---|---|
| VitalDB Open | SNUADC/ | OR (수술실) | ECG, ABP, PPG, CVP, CO2, AWP | 500Hz (ECG/ABP/PPG/CVP), 62.5Hz (CO2/AWP) | ~6,388 cases |
| K-MIMIC | SNUADCM/ | ICU | ECG, ABP, PPG, CVP, PAP, ICP | 500Hz | 추가 확장 |

#### 3.1.1 K-MIMIC Ward(병상) 분포

K-MIMIC 전체 .vital 파일 (`kmimic_vital_all.txt`, 228,608 건) 의 ward 별 분포. ICU 종류에 따라 임상적으로 다른 모니터링 패턴을 보이므로 학습 데이터의 modality coverage 도 ward 구성에 직접 의존한다.

| Ward | 파일수 | % | 누적% | 임상적 특징 | 주요 모니터링 |
|---|---|---|---|---|---|
| **MICU** (Medical ICU) | 77,818 | 34.0% | 34.0% | 내과 — sepsis, 호흡부전, 심부전 | ECG, PPG, CO2, ABP (sepsis 시 CVP) |
| **SICU2** (Surgical ICU 2) | 35,742 | 15.6% | 49.7% | 수술 후 — 일반/major surgery | ECG, ABP, CVP, ART |
| **PICU** (Pediatric ICU) | 32,644 | 14.3% | 64.0% | 소아 — 비침습 위주 | ECG, PPG, RR (CVP 드뭄) |
| **EICU** (Emergency ICU) | 27,635 | 12.1% | 76.0% | 응급 입원 직전, 짧은 stay | ECG, PPG, ABP (제한적) |
| **SICU1** (Surgical ICU 1) | 20,535 | 9.0% | 85.0% | 수술 후 — SICU2 와 분리 운영 | ECG, ABP, CVP, ART |
| **CPICU** (Cardio Pediatric ICU) | 17,392 | 7.6% | 92.6% | 심장 (소아/cardiothoracic) | ECG, ABP, CVP, PAP, ART |
| **NICU** (Neonatal ICU) | 12,478 | 5.5% | 98.1% | 신생아 — micro-monitor | ECG, PPG, RR (CVP 거의 없음) |
| **CCU** (Coronary Care Unit) | 4,364 | 1.9% | 100.0% | 관상동맥/AMI | ECG 위주 |
| **합계** | **228,608** | 100% | | | |

**Modality coverage 함의:**

- **ECG / PPG / RR**: 거의 모든 ward 에서 측정 → 학습 데이터 풍부
- **ABP / ART (동맥)**: SICU + CPICU + MICU 일부 = 약 25-35% 환자
- **CVP**: SICU + CPICU 환자 일부 + MICU sepsis 케이스 = 약 10-20% 환자 (실측 진단 결과 SNUADCM/CVP 트랙은 약 20-30% 파일에서 valid CVP 파형 존재, 나머지는 transducer 미연결 sentinel)
- **PAP / ICP**: 매우 희소, CPICU/NCU 일부 case 만

**파일 수준 sentinel 비율 (실측, n=50 진단):**

SNUADCM/CVP 트랙 보유 파일 중 `valid_range=(-5, 40)` mmHg 안에 분포하는 정상 파형은 약 20-30%. 나머지는 monitor disconnect default (127.93, 129.88, 209.96 등 다양한 stuck value) 로 채워져 있어 `valid_range` filter 가 자동으로 NaN 마스킹 → quality gate 로 작동한다. 이는 데이터 손상이 아니라 **CVP 라인이 환자에게 실제 잡히지 않은 임상 reality** 의 반영이다 (CVC 적응증 = vasopressor, 대량 수액, CRRT, TPN 등).

### 3.2 전처리 파이프라인 (`data/parser/vitaldb.py`)

모든 신호에 동일한 단계를 적용하되, 신호별 파라미터(`SignalConfig`)가 다르다:

```
1. Range Check      -- 물리적 유효 범위 밖 → NaN
2. Spike Detection   -- 전기소작기 등 아티팩트 → NaN (MAD 기반)
3. Motion Artifact   -- PPG 전용: envelope + gradient → NaN
4. NaN-free Segments -- 연속 유효 구간 추출 (최소 60초)
5. Median Filter     -- 임펄스 노이즈 제거 (ABP, PPG: kernel=5)
6. Notch Filter      -- 전원 주파수 제거 (ECG: 60Hz)
7. Bandpass/Lowpass   -- 신호별 주파수 대역 필터링
8. Resample → 100Hz  -- 모든 신호를 TARGET_SR=100Hz로 통일
9. Quality Check      -- segment_quality_score + domain_quality_check
10. Save .pt          -- float32 텐서, subject별 디렉토리, manifest.json
```

**신호별 SignalConfig 예시**:
```python
"ecg": SignalConfig(
    valid_range=(-5.0, 5.0),
    filter_type="bandpass", filter_freq=(0.5, 40.0),
    notch_freq=60.0,
    spike_detection=True, spike_threshold_std=10.0,
    min_amplitude=0.3,
    min_high_freq_ratio=0.05,  # QRS 없으면 불량
)

"abp": SignalConfig(
    valid_range=(20.0, 300.0),
    filter_type="lowpass", filter_freq=(0.0, 15.0),
    spike_detection=True, spike_threshold_std=6.0,
    median_kernel=5,
    min_amplitude=10.0,
)
```

### 3.3 BiosignalDataset

파일: `data/dataset.py`

```python
class BiosignalDataset(Dataset[BiosignalSample]):
    """Channel-Independent lazy-loading + sliding window."""

    def __init__(
        self,
        manifest: Sequence[RecordingManifest],
        window_seconds: float = 30.0,   # 윈도우 길이 (초)
        stride_seconds: float = None,   # None = window_seconds (비중첩)
        cache_size: int = 8,            # LRU 캐시 크기
        crop_ratio_range: tuple = None, # (0.5, 1.0) = 랜덤 길이 crop
        patch_size: int = None,         # crop 시 패치 배수 정렬
    )
```

**동작 원리**:
1. `RecordingManifest`로 메타데이터만 관리 (path, n_channels, n_timesteps, sampling_rate, ...)
2. `__getitem__` 호출 시 **on-demand** 로딩 + **LRU 캐시**로 반복 로드 방지
3. CI 패러다임: 모든 채널을 개별 `BiosignalSample`로 풀어헤침
4. Sliding window: `window_seconds` x `sampling_rate` 샘플 단위로 분할

**데이터 형식 지원**:
- `.pt` (PyTorch, mmap 가능)
- `.h5` (HDF5, `subject.h5#dataset_name`)
- `.zarr` (Zarr, float16 -> float32)

**Random Crop**: 학습 시 윈도우 내에서 `crop_ratio_range=(min, max)` 비율로 랜덤 잘라냄. `patch_size` 배수 정렬.

### 3.4 PackCollate (Bin-Packing Collate)

파일: `data/collate.py`

```python
class PackCollate:
    def __init__(
        self,
        max_length: int,             # 출력 행 너비
        collate_mode: str,           # "ci" 또는 "any_variate"
        patch_size: int = None,      # 패치 정렬
        stride: int = None,
        slot_size: int = 60000,      # cross-modal 그루핑 슬롯
    )
```

#### 3.4.1 CI 모드 (`collate_mode="ci"`)

```
각 BiosignalSample을 독립적인 그루핑 키로 → 채널 간 그루핑 없음
→ FFD bin-packing으로 행 채움
→ 각 행: 여러 독립 채널의 시계열이 이어져 패킹됨
```

- 그루핑 키: `(i,)` (고유 인덱스)
- Phase 1에서 사용

#### 3.4.2 Any-Variate 모드 (`collate_mode="any_variate"`)

```
같은 세션의 동시 시간대 채널을 하나의 그룹으로 묶음
→ 그룹 내 variate들을 이어 붙임
→ FFD bin-packing으로 행 채움
→ 각 행: 다변량 세션의 채널들이 패킹됨 (cross-modal pair 가능)
```

- 그루핑 키: `(session_id, time_slot)` 또는 `(recording_idx, win_start)`
- `slot_size = 60000` samples = 600초 (10분)
- Phase 2에서 사용

#### 3.4.3 FFD Bin-Packing 알고리즘

```
First-Fit Decreasing (FFD):
  1. 모든 PackUnit을 total_length 내림차순 정렬
  2. 각 unit을 첫 번째 여유 있는 bin에 배치
  3. 없으면 새 bin 생성

Min-heap 기반 최적화:
  - heap: (-remaining, bin_idx, version)
  - 버전 번호로 오래된 항목 lazy 제거
```

#### 3.4.4 PackedBatch 출력

```python
@dataclass
class PackedBatch:
    values: torch.Tensor        # (B, max_length) -- 패킹된 신호 값
    sample_id: torch.Tensor     # (B, max_length) -- 행 내 1-based sample ID (0=padding)
    variate_id: torch.Tensor    # (B, max_length) -- 행 내 1-based variate ID (0=padding)
    lengths: torch.Tensor       # (total_variates,)
    sampling_rates: torch.Tensor  # (total_variates,)
    signal_types: torch.Tensor  # (total_variates,) -- 신호 타입 코드
    spatial_ids: torch.Tensor   # (total_variates,) -- 전역 spatial_id
    padded_lengths: torch.Tensor | None
    start_samples: torch.Tensor | None  # (total_variates,) -- 절대 시작 sample
```

#### 3.4.5 Patch 정렬 패딩

```
실제 길이가 patch_size 배수가 아닐 경우:
  padded_seg_len = P + ceil(max(0, seg_len - P) / S) * S

공통 시간 그리드 정렬:
  abs_start를 patch_size 배수로 올림하여 모든 variate의 패치 경계가
  동일 절대 시간에 정렬 → cross-modal 매칭 정확도 보장
```

### 3.5 Manifest 구조

```
datasets/processed/
  VDB_0001/
    manifest.json    -- subject 메타데이터
    VDB_0001_S0_ecg_1_seg0_0.pt  -- (1, T) float32 @ 100Hz
    VDB_0001_S0_abp_1_seg0_0.pt
    ...
  VDB_0002/
    manifest.json
    ...
```

`manifest.json` 예시:
```json
{
  "subject_id": "VDB_0001",
  "sessions": [
    {
      "session_id": "VDB_0001_S0",
      "recordings": [
        {
          "file": "VDB_0001_S0_ecg_1_seg0_0.pt",
          "signal_type": 0,
          "n_channels": 1,
          "n_timesteps": 360000,
          "sampling_rate": 100.0,
          "spatial_ids": [1],
          "start_sample": 0
        }
      ]
    }
  ]
}
```

---

## 4. 사전학습 Phase 1 (Channel-Independent)

파일: `train/1_channel_independency.py`

### 4.1 목적

각 채널을 **독립적으로** 학습하여, 단일 신호의 **시간적 패턴**(파형 형태, 주기, 진폭 변화)을 표현하는 범용 인코더를 구축한다.

### 4.2 학습 구성

| 설정 | 값 | 비고 |
|---|---|---|
| `collate_mode` | `"ci"` | 채널 독립 패킹 |
| `task` | `"both"` | 양방향 + causal 동시 |
| `mask_ratio` | 0.15 | 15% 패치 마스킹 |
| `alpha` (MPM) | 1.0 | Masked reconstruction |
| `beta` (NextPred) | 1.0 | Next-patch prediction |
| `gamma` (CrossModal) | 0.0 | 비활성 |
| `delta` (Contrastive) | 0.0 | 비활성 |
| `variate_mask_prob` | 0.0 | 비활성 |
| `lr` | 1e-3 | Adam optimizer |
| `n_epochs` | 70 | |
| `warmup_epochs` | 5 | Cosine warmup |

### 4.3 Loss 조합

```
L_total = alpha * L_MPM + beta * L_NextPred
```

- **L_MPM**: 마스킹된 패치 위치에서 Peak-Weighted MSE
- **L_NextPred**: same-variate next-patch prediction

### 4.4 Horizon Curriculum

`max_horizon > 1`이고 `horizon_curriculum=True`일 때:

```
n_epochs를 3단계로 분할:
  전반 40% (epoch 0~27): H = 1 (1-step 예측만)
  중반 30% (epoch 28~48): H ≤ ceil(max_h * 0.6) (중거리 예측 추가)
  후반 30% (epoch 49~69): H ≤ max_h (전체 range)
```

- 학습 시 매 배치마다 `h = (batch_idx % max_horizon) + 1`로 순환
- 새 horizon embed는 이전 최고 horizon embed에서 초기화 (cold-start 방지)
- `horizon_weight = 1.0 / horizon` (먼 예측일수록 loss 가중치 감소)

### 4.5 기타 기능

- **RecordingLocalitySampler**: 같은 레코딩의 윈도우를 연속 yield → LRU 캐시 히트율 극대화, 네트워크 디스크 I/O 병목 해소
- **Random Crop**: `crop_ratio_range=(min, max)` -- 학습 데이터 다양성 증가
- **Early Stopping**: validation loss 기준, `patience` epochs
- **DDP 지원**: `torchrun --nproc_per_node=2 -m train.1_channel_independency`
- **AMP**: `--use_amp` 플래그로 Mixed Precision 활성

---

## 5. 사전학습 Phase 2 (Any-Variate Cross-Modal)

파일: `train/2_any_variate.py`

### 5.1 목적

Phase 1에서 학습된 단일 신호 표현을 기반으로, **다변량 세션에서 신호 간 생리학적 관계**를 학습한다. 이를 통해:
- 한 신호에서 다른 신호의 waveform을 예측 (Virtual Sensing)
- 신호 간 공유 표현 정렬 (Contrastive Alignment)
- 누락 신호 zero-shot 생성 (Variate Dropout → Cross-Modal Generation)

### 5.2 학습 구성

| 설정 | 값 | 비고 |
|---|---|---|
| `collate_mode` | `"any_variate"` | 다변량 패킹 |
| `task` | `"both"` | 양방향 + causal 동시 |
| `mask_ratio` | 0.4 | 40% 패치 마스킹 (Phase 1 대비 높음) |
| `alpha` (MPM) | 0.7 | Phase 1 대비 낮춤 |
| `beta` (NextPred) | 0.3 | Phase 1 대비 낮춤 |
| `gamma` (CrossModal) | 0.5 | **활성** |
| `delta` (Contrastive) | 0.5 | **활성** |
| `variate_mask_prob` | 0.3 | 30% 확률로 전체 variate 마스킹 |
| `variate_drop_prob` | 0.1 | 10% 확률로 variate를 attention에서 완전 제거 |
| `block_mask` | true | 연속 블록 마스킹 |
| `block_size_min/max` | 2/3 | 블록 크기 (패치 수) |
| `lr` | 2e-4 | Phase 1 대비 낮은 LR (fine-tuning) |
| `n_epochs` | 50 | |
| `window_seconds` | 600.0 | 10분 윈도우 (다변량) |
| `contrastive_proj_dim` | 128 | |
| `peak_alpha` | 1.0 | Peak-Weighted MSE 활성 |
| `lambda_spec` | 0.2 | Spectral Loss 활성 |

### 5.3 Phase 1 Checkpoint 로드

```python
# Phase 1 checkpoint에서 config 우선 사용 → 아키텍처 불일치 방지
ckpt_model_config = ModelConfig.from_dict(ckpt_state["config"])
# Phase 2에서 추가/변경되는 파라미터만 덮어쓰기
ckpt_model_config.contrastive_proj_dim = config.model_config.contrastive_proj_dim
ckpt_model_config.max_horizon = config.model_config.max_horizon
```

- `contrastive_proj_dim`: Phase 1에서는 0, Phase 2에서 128로 활성화 → 새 파라미터는 랜덤 초기화
- `max_horizon`: Phase 2에서 변경 가능 (새 horizon embed는 이전 것에서 초기화)

### 5.4 Loss 조합

```
L_total = alpha * L_MPM
        + beta * (L_NextPred + gamma * L_CrossModal)
        + delta * L_Contrastive
        + aux_weight * L_MoE_balance
```

### 5.5 Variate-Level Masking

`variate_mask_prob > 0`일 때 `create_patch_mask()`에서 활성:

```python
if random() < variate_mask_prob:
    # 다변량 행에서 랜덤으로 하나의 variate 선택
    # 해당 variate의 모든 패치를 마스킹 → [MASK] 토큰으로 교체
    # → 다른 variate 정보만으로 해당 variate의 waveform을 복원해야 함
```

**목적**: Virtual Sensing -- ABP가 마스킹되면 ECG + PPG 정보만으로 ABP waveform을 복원하도록 학습

### 5.6 Variate Dropout (Complete Variate Dropout)

```python
def _sample_variate_drop(p_sid, p_vid, patch_mask, drop_prob):
    """행별로 하나의 variate를 attention에서 완전 제거"""
    # 1. drop_prob 확률로 활성화
    # 2. 다변량(2+ variates) 행에서만 작동
    # 3. 랜덤 variate 선택 → attention mask에서 제거 + content를 mask_token으로 교체
```

**목적**: Zero-shot cross-modal generation의 **train-inference gap 해소**
- 학습 시: "해당 variate 없이 cross-pred" 시나리오를 경험
- 추론 시: `generate_cross_modal()`에서 target variate가 없는 상태로 예측

### 5.7 Block Masking

```python
# 연속 블록 단위로 마스킹 → 보간 기반 복원 방지
# → 장기 시간적 의존성 학습을 강제

block_size = randint(block_size_min, block_size_max)  # 2~3 패치
# 배치된 블록 영역 + 양옆 1패치 gap을 run에서 제거하여 인접 블록 방지
```

---

## 6. Loss 함수 상세

### 6.1 CombinedLoss 전체 구조

파일: `loss/criterion.py`

```python
L_total = alpha * L_MPM
        + beta * (L_NextPred + gamma * L_CrossModal)
        + delta * L_Contrastive
```

```python
class CombinedLoss(nn.Module):
    masked_loss_fn: MaskedPatchLoss     # alpha
    next_loss_fn: NextPredictionLoss    # beta (내부에 gamma)
    contrastive_loss_fn: CrossModalContrastiveLoss  # delta
```

### 6.2 Masked Patch Loss (MPM)

파일: `loss/masked_mse_loss.py`

```python
class MaskedPatchLoss:
    def forward(reconstructed, original_patches, pred_mask):
        pred_m = reconstructed[pred_mask]   # (M, P)
        target_m = original_patches[pred_mask]  # (M, P)
        return compute_patch_loss(pred_m, target_m, peak_alpha, lambda_spec)
```

#### 6.2.1 Peak-Weighted MSE

```python
def compute_peak_weighted_mse(pred, target, peak_alpha):
    # peak_alpha > 0일 때:
    abs_target = target.abs()                     # (M, P)
    max_abs = abs_target.amax(dim=-1, keepdim=True)  # (M, 1)
    weight = 1.0 + peak_alpha * (abs_target / max_abs)  # (M, P)
    return (weight * (pred - target)^2).mean()
```

**수식**:
```
w_i = 1 + alpha * (|target_i| / max(|target|))
L_peak_mse = mean(w_i * (pred_i - target_i)^2)
```

- `peak_alpha=0`: 일반 MSE
- `peak_alpha=1`: R-peak, systolic peak 등 임상적으로 중요한 고진폭 샘플에 2배 가중치

#### 6.2.2 Multi-Resolution STFT Loss (Spectral)

```python
def _multi_resolution_stft_loss(pred, target, n_ffts=(16, 32, 64)):
    for n_fft in valid_ffts:
        hop = n_fft // 4
        pred_stft = torch.stft(pred, n_fft, hop, window=hann)
        target_stft = torch.stft(target, n_fft, hop, window=hann)

        pred_mag = pred_stft.abs()
        target_mag = target_stft.abs()

        # Spectral Convergence: ||target - pred||_F / ||target||_F
        sc = norm(target_mag - pred_mag, "fro") / (norm(target_mag, "fro") + 1e-8)

        # Log-magnitude L1
        log_mag = |log1p(pred_mag) - log1p(target_mag)|.mean()

        loss += sc + log_mag

    return loss / len(valid_ffts)
```

**목적**: 시간-주파수 구조를 다중 스케일로 비교하여 주파수 특성 복원 강화

#### 6.2.3 총합 Patch Loss

```
L_patch = L_peak_mse + lambda_spec * L_stft
```

### 6.3 Next-Patch Prediction Loss

파일: `loss/next_prediction_loss.py`

#### 6.3.1 Same-Variate Loss

```python
def _same_variate_loss(next_pred, original_patches, patch_mask, ..., horizon):
    target_next = original_patches[:, horizon:, :]  # (B, N-H, P)
    pred_next = next_pred[:, :-horizon, :]          # (B, N-H, P)

    valid = patch_mask[:, :-horizon] & patch_mask[:, horizon:]
          & (sample_id[:, :-horizon] == sample_id[:, horizon:])
          & (variate_id[:, :-horizon] == variate_id[:, horizon:])

    loss = compute_patch_loss(pred_next[valid], target_next[valid]) * (1/horizon)
```

- **Horizon weight**: `1/h` -- 먼 예측일수록 loss 가중치 감소
- **유효성 조건**: 같은 sample, 같은 variate 내에서만 예측 (variate 경계를 넘지 않음)

#### 6.3.2 Cross-Modal Loss

```python
def _cross_modal_loss(cross_pred_per_type, original_patches, ..., time_id, patch_signal_types):
    # 같은 (sample_id, time_id)에서 서로 다른 variate_id 쌍 매칭
    group_key = batch_idx * (s * k) + sample_id * k + time_id  # (B, N)
    cross_mask = same_group & diff_variate & both_valid & non_pad

    # CROSS_PRED_ALLOWED_PAIRS 필터
    allowed = allowed_lut[st_i, st_j]
    cross_mask = cross_mask & allowed

    # Target-conditioned prediction:
    # source 패치 i에서 target 패치 j의 signal type에 해당하는 cross_head 출력 선택
    target_st = patch_signal_types[b_idx, j_idx]
    pred_p = cross_pred_per_type[b_idx, i_idx, target_st]  # (K, P)
    target_p = original_patches[b_idx, j_idx]               # (K, P)
```

### 6.4 Cross-Modal Contrastive Loss (InfoNCE)

파일: `loss/contrastive_loss.py`

```python
class CrossModalContrastiveLoss:
    log_temperature: nn.Parameter  # learnable, CLIP-style

    def forward(z, patch_mask, patch_sample_id, patch_variate_id, time_id):
        z = F.normalize(z, dim=-1)  # L2 normalize

        temp = log_temperature.exp().clamp(0.01, 1.0)
        sim = bmm(z, z.T) / temp  # (B, N, N) similarity matrix

        # Positive: same (sample_id, time_id), different variate_id
        pos_mask = same_group & diff_var & valid_pair

        # InfoNCE: -log(sum_pos / sum_all)
        log_numer = logsumexp(sim[pos_mask])
        log_denom = logsumexp(sim[valid & ~self])
        loss = -(log_numer - log_denom)
```

**수식**:
```
L_InfoNCE = -1/|A| * sum_{a in A} log( sum_{p in P(a)} exp(sim(a,p)/tau) / sum_{k in K(a)} exp(sim(a,k)/tau) )
```

여기서:
- `A`: positive pair가 있는 유효 anchor
- `P(a)`: anchor `a`의 positive (같은 시간, 다른 variate)
- `K(a)`: anchor `a`의 모든 유효 패치 (자기 제외)
- `tau`: learnable temperature (초기 0.07)

**특성**:
- Contrastive는 **전체 신호 쌍**에 적용 (CROSS_PRED_ALLOWED_PAIRS 무관)
- Cross-Modal MSE는 **whitelist 쌍만** 적용
- 단일 variate만 존재하는 행은 자동 스킵

### 6.5 MoE Auxiliary Loss

```python
# Switch Transformer load balancing
aux_loss = num_experts * sum(f_i * P_i)
# f_i: expert i에 할당된 토큰 비율
# P_i: expert i의 평균 gate 확률
```

- `aux_loss_weight = 0.01`
- `use_moe=True`일 때만 활성

---

## 7. Cross-Modal Prediction

### 7.1 CROSS_PRED_ALLOWED_PAIRS

파일: `data/spatial_map.py`

생리학적으로 waveform 복원이 가능한(인과 관계가 있는) 신호 쌍만 cross-modal MSE prediction에 허용한다.

```python
CROSS_PRED_ALLOWED_PAIRS = {
    # ── Arterial-Cardiac ──
    (0, 1),  # ECG <-> ABP     심박 주기, pulse transit time
    (0, 2),  # ECG <-> PPG     심박 주기, peripheral pulse
    (1, 2),  # ABP <-> PPG     동맥 맥파 (거의 동형)

    # ── Right Heart / Central Hemodynamics ──
    (3, 6),  # CVP <-> PAP     우심방압 <-> 폐동맥압, 우심실 전후부하

    # ── Cerebral Perfusion ──
    (1, 7),  # ABP <-> ICP     CPP = MAP - ICP, 뇌자동조절

    # ── Respiratory ──
    (4, 5),  # CO2 <-> AWP     호흡 역학
}
```

**생리학적 근거**:
- **ECG-ABP**: 심장 수축 → 동맥 혈압파, 동일 심박 주기, R-peak에서 systolic peak까지의 pulse transit time
- **ECG-PPG**: 심장 수축 → 말초 맥파, PPG의 onset은 ECG R-peak 후 일정 시간
- **ABP-PPG**: 동맥 혈압파의 말초 전달, waveform morphology가 매우 유사
- **CVP-PAP**: 우심방압과 폐동맥압은 우심실의 전부하/후부하 관계
- **ABP-ICP**: 뇌관류압(CPP) = MAP - ICP, 뇌자동조절 기전
- **CO2-AWP**: 호기말 CO2와 기도압은 환기 역학으로 동기화

**제외된 쌍 (ECG-CVP, ECG-PAP, ECG-ICP 등)**:
- 전기 신호 → 유체압력의 **정보 단절**: 심박 동기화는 되지만 waveform morphology 복원이 불가능

### 7.2 Mechanism Group 체계

```python
MECHANISM_GROUP = {
    0: 0,  # ECG → Cardiovascular
    1: 0,  # ABP → Cardiovascular
    2: 0,  # PPG → Cardiovascular
    3: 0,  # CVP → Cardiovascular
    4: 1,  # CO2 → Respiratory
    5: 1,  # AWP → Respiratory
    6: 0,  # PAP → Cardiovascular
    7: 0,  # ICP → Cardiovascular
}
```

시각화에서 Cardiovascular/Respiratory 그룹으로 분리하여 표시한다.

### 7.3 Separate Cross Heads 구조

```python
cross_heads = nn.ModuleDict({
    "0": nn.Linear(d_model, patch_size),  # -> ECG target
    "1": nn.Linear(d_model, patch_size),  # -> ABP target
    "2": nn.Linear(d_model, patch_size),  # -> PPG target
    "3": nn.Linear(d_model, patch_size),  # -> CVP target
    "4": nn.Linear(d_model, patch_size),  # -> CO2 target
    "5": nn.Linear(d_model, patch_size),  # -> AWP target
    "6": nn.Linear(d_model, patch_size),  # -> PAP target
    "7": nn.Linear(d_model, patch_size),  # -> ICP target
})
```

**Target-Conditioned Prediction**:
```
source patch (ECG, time_id=5)에서:
  - cross_heads["1"](encoded) → ABP target 예측
  - cross_heads["2"](encoded) → PPG target 예측
  - 각 target type별 독립적인 Linear head 사용
```

Forward에서:
```python
cross_pred_per_type = torch.stack([
    cross_heads[str(st)](encoded)
    for st in range(num_signal_types)
], dim=2)  # (B, N, num_signal_types, patch_size)
```

### 7.4 abs_time_id (절대 시간 매칭)

```python
# Cross-modal 매칭용 절대 시간 인덱스
abs_time = start_samples[global_var_idx] + time_id * patch_size  # (B, N)
abs_time_id = abs_time // patch_size  # patch_size 단위 양자화
```

- `time_id` (상대적): RoPE position encoding용 -- variate 내 순서
- `abs_time_id` (절대적): cross-modal loss 매칭용 -- 물리적 시간 일치

### 7.5 Zero-Shot Cross-Modal Generation

```python
@torch.no_grad()
def generate_cross_modal(batch, target_signal_type, denormalize=True):
    """입력 batch의 source signal로부터 target waveform 생성"""
    out = model.forward(batch, task="masked", mask_ratio=0.0)
    # mask_ratio=0 → 순수 source 정보만 사용

    target_pred = out["cross_pred_per_type"][:, :, target_signal_type, :]
    # denormalize with source의 loc/scale (approximate)
```

학습 시 `variate_drop_prob`로 variate를 attention에서 완전 제거하는 시나리오를 경험했기 때문에, 추론 시 target variate가 없는 상태에서도 합리적인 예측이 가능하다.

---

## 8. Downstream Tasks

### 8.1 Task Taxonomy (3 카테고리 / 9 Task)

> **2026-04-27 확정.** 모든 다운스트림 평가는 다음 3 카테고리 / 9 task로 정렬된다. 디렉토리는 `downstream/{category}/{task}/`.

#### Acute Event Detection
1. **Intraoperative Hypotension Prediction** — `downstream/acute_event/hypotension/`. 미래 5~15분 후 MAP < 65 mmHg 지속 발생 예측. Binary classification.
2. **Arrhythmia Classification** — `downstream/acute_event/arrhythmia/`. PTB-XL 기반 다중 클래스 부정맥 분류.
3. **Intracranial Hypertension** — `downstream/acute_event/intracranial_hypertension/`. ICP 상승 이벤트(>=20 mmHg 지속) 예측. Binary classification.

#### Outcome Prediction
4. **Sepsis Prediction** — `downstream/outcome/sepsis/`. Sepsis-3 기준, 6시간 prediction horizon. Binary classification.
5. **Mortality Prediction** — `downstream/outcome/mortality/`. ICU 입원 중 사망 예측. Binary classification.
6. **Cardiac Arrest Prediction** — `downstream/outcome/cardiac_arrest/`. 심정지 사전 예측. Binary classification.
7. **Postoperative Acute Kidney Injury Prediction** — `downstream/outcome/aki/`. KDIGO Cr 기반 수술 후 AKI 예측. Binary / ordinal.

#### Physiological Generation
8. **Cross-Modal Reconstruction** — `downstream/generation/cross_modal/`. 한 신호로부터 다른 신호의 waveform을 zero-shot 또는 frozen-head 추론으로 생성 (예: ECG→ABP, ABP→ICP). 사전학습된 `cross_heads` 직접 평가.
9. **Waveform Forecasting** — `downstream/generation/intra_modal_forecast/`. 동일 variate 내 미래 패치 예측. 사전학습된 `next_head` 직접 평가.

> **Out of scope**: 초기 초안에서 다뤘던 anomaly detection, BIS regression, 임의 imputation, "Any-to-Any" generation은 본 평가에서 제외되었다. Generation 카테고리는 위 두 task로 통합된다.

### 8.2 모델 래퍼 (DownstreamModelWrapper)


파일: `downstream/model_wrapper.py`

```python
class DownstreamModelWrapper(nn.Module):
    def __init__(self, checkpoint_path, model_version, device):
        # 1. Checkpoint 로드 → config 복원 → 모델 생성
        # 2. State dict 로드 (strict=False)
        # 3. Encoder freeze + eval 모드

    def freeze_encoder(self):
        self.model.requires_grad_(False)

    def unfreeze_encoder(self):
        self.model.requires_grad_(True)

    def extract_features(self, batch, pool="mean"):
        # pool="mean": patch_mask 기반 mean pooling → (B, d_model)
        # pool="none": 패치 레벨 → (B, N, d_model)
```

### 8.3 LinearProbe

```python
class LinearProbe(nn.Module):
    def __init__(self, d_model, n_classes, dropout_p=0.1):
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout_p),
            nn.Linear(d_model, n_classes),
        )
```

### 8.4 세 가지 학습 모드

| 모드 | Encoder | Head | LR | 목적 |
|---|---|---|---|---|
| `linear_probe` | **Frozen** | LinearProbe | 1e-3 | Representation 품질 평가 |
| `finetune` | **Unfrozen** | LinearProbe | encoder: lr*0.1, probe: lr | 실제 임상 성능 |
| `from_scratch` | **Random Init** | LinearProbe | 1e-4 | 사전학습 이득 측정 |

### 8.5 예시: Intraoperative Hypotension Prediction (Task 1)

파일: `downstream/acute_event/hypotension/`

#### 8.5.1 태스크 정의

- **목표**: 미래 5~15분 후 MAP < 65 mmHg (>= 1분 지속) 발생 예측
- **Label 소스**: 항상 ABP (미래 구간의 MAP)
- **Input 소스**: 선택 가능 -- `abp`, `ecg`, `ppg`, `ecg_ppg`

#### 8.5.2 데이터 준비 (`prepare_data.py`)

```python
def extract_forecast_samples(cases, input_signals, window_sec, stride_sec, horizon_sec):
    """시간 정렬된 다채널 데이터에서 (input, future_label) 쌍 추출"""

    # Input window: [start, start + win_samples)
    # Future label: ABP의 [start + win_samples, start + win_samples + horizon_samples)

    # MAP 계산: 10초 윈도우별 평균
    # 1분 지속 = 6개 연속 10초 윈도우에서 MAP < 65
    label = _has_sustained_hypotension(future_maps, threshold=65, min_consecutive=6)
```

**Sweep 실험 설계**:
```bash
python -m downstream.acute_event.hypotension.prepare_data \
    --data-dir vitaldb_pt_test \
    --input-signals abp ecg ppg \
    --window-secs 30 60 300 600 \
    --horizon-mins 5 10 15
# → 4 windows x 3 horizons = 12 데이터셋 생성
```

#### 8.5.3 실행 (`run.py`)

```python
# 다중 신호 입력 → any_variate collate → foundation model → mean pool → LinearProbe
collate_mode = "any_variate" if len(signals) > 1 else "ci"

# 학습
train_linear_probe(model, probe, train_batches, epochs, lr, device)

# 평가
metrics = evaluate_linear_probe(model, probe, test_batches, device)
# AUROC, AUPRC, optimal_threshold (Youden's J), sensitivity, specificity
```

### 8.6 평가 메트릭

파일: `downstream/metrics.py`

#### 분류 메트릭:
- **AUROC**: Binary/Multi-class (one-vs-rest), trapezoidal rule
- **AUPRC**: Average Precision, step-function integration
- **F1 Score**: macro/weighted
- **Sensitivity/Specificity**: binary, optimal threshold (Youden's J)

#### 회귀/복원 메트릭:
- **MSE, MAE, MAPE**: 기본 오차 메트릭
- **Pearson r**: 상관 계수
- **Bland-Altman**: bias, limits of agreement

---

## 9. 시각화

### 9.1 Phase 1 시각화 (`train/visualize.py`)

매 `viz_every` 에폭마다:
- **Reconstruction Figure**: 원본 vs 마스킹된 패치의 복원 결과
- **Next-Pred Figure**: 1-step 예측 vs 실제 다음 패치

### 9.2 Phase 2 전용 시각화 (`train/visualize_phase2.py`)

#### Cross-Modal Prediction Figure

```
상단: [Cardiovascular] ECG, ABP, PPG, CVP, PAP, ICP
  - 각 쌍(예: ECG<->ABP)마다 2행:
    Row 1: B 원본 + A→B 예측 (MSE, r 표시)
    Row 2: A 원본 + B→A 예측 (MSE, r 표시)

하단: [Respiratory] CO2, AWP
  - CO2<->AWP 쌍: 동일 형식
```

- CROSS_PRED_ALLOWED_PAIRS에 정의된 whitelist 쌍만 표시
- signal type별 색상 코딩 (ECG=red, ABP=blue, PPG=green, ...)
- denormalization 적용 (원본 스케일 복원)

**색상 체계**:
```python
SIGNAL_TYPE_COLORS = {
    0: "#e41a1c",  # ECG - red
    1: "#377eb8",  # ABP - blue
    2: "#4daf4a",  # PPG - green
    3: "#984ea3",  # CVP - purple
    4: "#ff7f00",  # CO2 - orange
    5: "#a65628",  # AWP - brown
    6: "#f781bf",  # PAP - pink
    7: "#999999",  # ICP - gray
}
```

---

## 10. 실행 가이드

### 10.1 환경 설정

```bash
# 가상환경 활성화
source .venv/Scripts/activate

# 핵심 의존성
# torch 2.10.0, einops, mne, matplotlib, pyyaml
```

### 10.2 데이터 전처리

```bash
# VitalDB .vital 파일 → .pt 변환
python -m data.parser.vitaldb \
    --raw datasets/raw/vitaldb \
    --out datasets/processed \
    --workers 4

# 트랙 탐색 (dry-run)
python -m data.parser.vitaldb \
    --raw datasets/raw/vitaldb \
    --discover --max-files 3
```

### 10.3 Phase 1 학습

```bash
# 단일 GPU
python -m train.1_channel_independency \
    --d_model 64 --num_layers 2 --patch_size 100 \
    --batch_size 16 --lr 1e-3 --n_epochs 70 \
    --alpha 1.0 --beta 1.0 \
    --data_dir datasets/processed \
    --output_dir outputs/phase1_ci

# 멀티 GPU (DDP)
torchrun --nproc_per_node=2 -m train.1_channel_independency \
    --d_model 256 --num_layers 12 --patch_size 200 \
    --batch_size 16 --lr 8e-4 --n_epochs 70

# YAML config 사용
python -m train.1_channel_independency --config configs/phase1_base.yaml

# Smoke test (1 batch만 실행)
python -m train.1_channel_independency --dry-run

# Resume from checkpoint
python -m train.1_channel_independency --resume outputs/phase1/checkpoints/epoch_050.pt
```

### 10.4 Phase 2 학습

```bash
# 단일 GPU (Phase 1 checkpoint 자동 탐색)
python -m train.2_any_variate

# Phase 1 checkpoint 명시
python -m train.2_any_variate \
    --resume outputs/phase1/base/checkpoints/best.pt

# YAML config 사용 (권장)
python -m train.2_any_variate --config configs/phase2_base.yaml

# 2-GPU DDP
torchrun --nproc_per_node=2 launch_phase2.py \
    --config configs/phase2_base.yaml

# Smoke test
python -m train.2_any_variate --config configs/phase2_base.yaml --dry-run
```

### 10.5 Downstream 실행

```bash
# 1. 데이터 준비 (Hypotension Sweep)
python -m downstream.acute_event.hypotension.prepare_data \
    --data-dir vitaldb_pt_test \
    --input-signals abp ecg ppg \
    --window-secs 30 60 300 600 \
    --horizon-mins 5 10 15 \
    --out-dir outputs/downstream/hypotension

# 2. 실행 — Linear Probe
python -m downstream.acute_event.hypotension.run \
    --checkpoint outputs/phase2/base/checkpoints/best.pt \
    --mode linear_probe \
    --data-path outputs/downstream/hypotension/task1_hypotension_abp_w30s_h5min.pt \
    --epochs 20 --lr 1e-3 --device cuda:0

# 3. 실행 — Fine-tuning
python -m downstream.acute_event.hypotension.run \
    --checkpoint outputs/phase2/base/checkpoints/best.pt \
    --mode finetune \
    --data-path outputs/downstream/hypotension/task1_hypotension_ecg_ppg_w600s_h5min.pt \
    --epochs 30 --lr 1e-4 --device cuda:0

# 4. 실행 — From Scratch (사전학습 이득 비교)
python -m downstream.acute_event.hypotension.run \
    --checkpoint outputs/phase2/base/checkpoints/best.pt \
    --mode from_scratch \
    --data-path outputs/downstream/hypotension/task1_hypotension_abp_w30s_h5min.pt \
    --epochs 30 --lr 1e-4 --device cuda:0

# 5. Pipeline 검증 (dummy 모드)
python -m downstream.acute_event.hypotension.run \
    --dummy --data-dir vitaldb_pt_test \
    --input-signals abp --window-sec 30
```

### 10.6 서버 환경

```
GPU 서버: 2x L40S 48GB (KHDP 분석환경)
  - 8 vCPU, 96GB RAM, 100GB 디스크
  - Public repo만 pull 가능

git push 후 서버에서:
  cd /home/coder/workspace/updown/bio_fm
  git pull origin master
  torchrun --nproc_per_node=2 launch_phase2.py --config configs/phase2_base.yaml
```

### 10.7 Checkpoint 구조

```python
{
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": model_config.to_dict(),   # ModelConfig 직렬화
    "epoch": int,
    "loss": float,
    "phase": str,      # "phase1_ci" or "phase2_av"
    "tag": str,        # "best", "final", or epoch number
}
```

저장 위치: `{output_dir}/{exp_name}/checkpoints/{phase}_{tag}.pt`

---

## 부록: 프로젝트 디렉토리 구조

```
Biosignal-Foundation-Model/
  model/
    _config.py              # ModelConfig dataclass
    biosignal_model.py      # BiosignalFoundationModel (main model)
    checkpoint.py           # save/load checkpoint
  module/
    attention.py            # GQA, MHA, MQA
    ffn.py                  # FFN, GLU, MoE
    norm.py                 # RMSNorm
    packed_scaler.py        # PackedStdScaler, PackedAbsMeanScaler
    patch.py                # PatchEmbedding, ResidualMLP
    position.py             # RoPE, BinaryAttentionBias, QueryKeyProjection
  data/
    dataset.py              # BiosignalDataset, RecordingManifest
    collate.py              # PackCollate, PackedBatch
    spatial_map.py          # SPATIAL_MAP, CROSS_PRED_ALLOWED_PAIRS
    sampler.py              # RecordingLocalitySampler
    parser/
      vitaldb.py            # VitalDB -> .pt 변환
      _common.py            # 공통 QC, 리샘플링
  loss/
    masked_mse_loss.py      # MaskedPatchLoss, create_patch_mask, Peak-Weighted MSE
    next_prediction_loss.py # NextPredictionLoss (same-variate + cross-modal)
    contrastive_loss.py     # CrossModalContrastiveLoss (InfoNCE)
    criterion.py            # CombinedLoss
  train/
    train_utils.py          # TrainConfig, train_one_epoch, utilities
    1_channel_independency.py # Phase 1 script
    2_any_variate.py        # Phase 2 script
    visualize.py            # Phase 1 visualization
    visualize_phase2.py     # Phase 2 visualization (cross-modal)
  downstream/
    model_wrapper.py        # DownstreamModelWrapper, LinearProbe
    data_utils.py           # Pilot case loading, windowing, labeling
    metrics.py              # AUROC, AUPRC, F1, Bland-Altman
    viz.py                  # ROC curve plotting
    acute_event/
      hypotension/          # Intraoperative Hypotension Prediction
      arrhythmia/           # Arrhythmia Classification (PTB-XL)
      intracranial_hypertension/  # Intracranial Hypertension
    outcome/
      sepsis/               # Sepsis Prediction
      mortality/            # Mortality Prediction
      cardiac_arrest/       # Cardiac Arrest Prediction
      aki/                  # Postoperative Acute Kidney Injury Prediction
    generation/
      cross_modal/          # Cross-Modal Reconstruction
      intra_modal_forecast/ # Waveform Forecasting
  configs/
    phase2_base.yaml        # Phase 2 base configuration
  docs/
    wiki.md                 # 이 문서
```
