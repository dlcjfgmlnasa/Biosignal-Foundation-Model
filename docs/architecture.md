# Model Architecture

## Overview

Biosignal Foundation Model은 수술중/ICU 환경의 7종 생체신호를 통합 처리하는
Patch-based Transformer 구조이다.

### 지원 신호 (7종)

| ID | Signal | 환경 | Sampling Rate | 단위 |
|----|--------|------|---------------|------|
| 0 | ECG | OR/ICU | 500Hz -> 100Hz | mV |
| 1 | ABP | OR/ICU | 500Hz -> 100Hz | mmHg |
| 2 | EEG | OR | 128Hz -> 100Hz | uV |
| 3 | PPG | OR/ICU | 500Hz -> 100Hz | NU |
| 4 | CVP | OR | 500Hz -> 100Hz | mmHg |
| 5 | CO2 | OR | 62.5Hz -> 100Hz | mmHg |
| 6 | AWP | OR | 62.5Hz -> 100Hz | hPa |

모든 신호는 100Hz로 통일 리샘플링된다.

## 파이프라인

```
Raw Signal (100Hz)
    |
    v
PackedStdScaler  ──> loc (mu), scale (sigma) 저장
    |                      |
    v                      v
Instance Normalization     Loc/Scale Injection
    |                      (Linear(1, d_model) -> additive)
    v                      |
PatchEmbedding (1D)        |
    |                      |
    v                      v
Patch Tokens + Positional  + loc_emb + scale_emb
    |
    v
Signal Type Embedding + Spatial Embedding (Dual Additive)
    |
    v
TransformerEncoder (GQA + GLU FFN + RoPE + MoE)
    |
    +──> extract_features()      -> (B, N, d_model)  [downstream]
    |
    +──> Reconstruction Head     -> reconstructed     [masked prediction]
    |
    +──> Cross-modal Head        -> cross_pred        [cross-variate prediction]
    |
    +──> Next-Patch Head         -> next_pred         [causal prediction]
    |
    +──> generate()              -> autoregressive    [forecasting]
```

## 핵심 구성 요소

### Instance Normalization with Loc/Scale Injection

1. **정규화**: (sample_id, variate_id) 그룹별 Z-score
   - 환자/센서 간 스케일 편차 제거
   - 모델이 파형의 형태적 동역학(morphological dynamics) 학습에 집중

2. **Loc/Scale Injection**: 정규화로 소실되는 절대 스케일 정보를 재주입
   - `loc_proj = Linear(1, d_model)`, `scale_proj = Linear(1, d_model)`
   - 패치 토큰에 additive injection
   - MOMENT(ICML 2024)의 한계 해결: vertically shifted 시계열 구분 가능

3. **Loss**: 전적으로 normalized scale에서 계산
   - Denormalize는 추론(`forecast()`, `generate()`) 시에만

### Dual Additive Embedding

- **Signal Type Embedding**: 7종 신호 대분류 (ECG=0, ABP=1, ...)
- **Spatial Embedding**: 채널 소분류 (Lead II=1, Lead V5=2, ...)
- 두 임베딩을 additive로 결합

### TransformerEncoder

- **GQA** (Grouped Query Attention): KV head 공유로 메모리 효율
- **GLU FFN** (Gated Linear Unit): SwiGLU activation
- **RoPE** (Rotary Position Embedding): 상대적 위치 인코딩
- **MoE** (Mixture of Experts): 선택적 expert routing

### Multi-task 학습 (단일 Encoder)

| Task | Attention | Head | Loss |
|------|-----------|------|------|
| `task="masked"` | 양방향 (bidirectional) | Reconstruction + Cross-modal | MPM (Masked Patch Modeling) |
| `task="next_pred"` | 인과적 (causal) | Next-Patch | Next-Patch Prediction |
| `task="both"` | 양방향 + 인과적 | 모두 | Combined Loss |

### 2-Phase Curriculum

- **Phase 1 (Channel Independent)**: 단일 채널씩 학습. MPM + Next-Pred.
  - `collate_mode="ci"`, `variate_mask_prob=0`
- **Phase 2 (Any-Variate)**: 다채널 동시 학습. Cross-modal loss 추가.
  - `collate_mode="any_variate"`, `variate_mask_prob>0`
  - Phase 1 checkpoint에서 이어서 학습

### Block Masking

연속 패치 블록 단위 마스킹으로 인접 패치 보간 방지:
- `block_size_min=2, block_size_max=4` (2~4초 연속)
- 블록 배치 후 양옆 1패치 gap 확보
- Fallback 랜덤 패치도 인접 마스킹 제외

### Combined Loss

```
L = alpha * MPM + beta * (NextPred + gamma * CrossModal) + delta * Contrastive
```

- **MPM (Masked Patch Modeling)**: 마스킹된 패치 복원 MSE + gradient loss + spectral loss
- **NextPredictionLoss**: same-variate next-patch + cross-modal next-patch
- **CrossModalContrastiveLoss**: InfoNCE 기반, CLIP-style learnable temperature

## 모델 API

### `extract_features(batch)`
- Downstream classification/regression용
- 반환: `encoded` (B, N, d_model), `patch_mask`, `loc`, `scale`
- loc/scale injection이 적용되어 representation에 amplitude 정보 내장

### `forward(batch, task="masked")`
- Zero-shot reconstruction / cross-modal prediction
- 반환: `reconstructed`, `cross_pred`, `encoded`, `patch_signal_types` 등

### `generate(batch, n_steps)`
- Autoregressive 다단계 생성 (horizon=1 반복)
- `collate_mode="ci"` 전제
- 반환: `(n_steps, B, patch_size)` generated patches

## 설정 예시

```yaml
# Model
d_model: 256
num_heads: 8
num_kv_heads: 2      # GQA
num_layers: 6
patch_size: 100       # 1초 @ 100Hz
ffn_dim_multiplier: 4
use_moe: false
use_rope: true
dropout: 0.1

# Masking
block_mask: true
block_size_min: 2
block_size_max: 4
mask_ratio: 0.15
variate_mask_prob: 0.0  # Phase 1. Phase 2에서 >0
```
