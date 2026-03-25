# Model Architect 서브 플랜

> **지침**: 이 파일은 **Model Architect** 역할의 서브 에이전트 전용 계획표이다.
> 작업을 시작하기 전 이 파일을 읽고, 할당된 `[ ]` 작업을 수행한 뒤 성공하면 `[x]`로 상태를 업데이트하라.
> 작업 전 반드시 `.plans/master_plan.md`의 **데이터 규약** (섹션 2)을 확인하라.

---

## 담당 파일

| 파일 | 역할 |
|------|------|
| `model/biosignal_model.py` | `BiosignalFoundationModel` — 전체 파이프라인 조립 |
| `model/checkpoint.py` | `save_checkpoint`, `load_checkpoint` |
| `module/attention.py` | `GroupedQueryAttention`, `MultiHeadAttention`, `MultiQueryAttention` |
| `module/transformer.py` | `TransformerEncoderLayer`, `TransformerEncoder` |
| `module/patch.py` | `PatchEmbedding` (사용), `MultiResolutionPatchEmbedding` (코드 보존, 미사용) |
| `module/packed_scaler.py` | `PackedScaler`, `PackedStdScaler`, `PackedAbsMeanScaler` |
| `module/ffn.py` | `FeedForward`, `GatedLinearUnitFeedForward`, `MoEFeedForward` |
| `module/norm.py` | `RMSNorm` |
| `module/position/` | `RotaryProjection`, `BinaryAttentionBias`, `QueryKeyProjection` |
| `tests/test_integration.py` | 모델 통합 테스트 |
| `tests/test_attention.py` | 어텐션 테스트 |
| `tests/test_transformer.py` | 트랜스포머 테스트 |
| `tests/test_patch.py` | 패치 토큰화 테스트 |

---

## 1. 기존 구현 현황 (완료)

### 1.1 모듈 레벨
- [x] `RMSNorm` — Root Mean Square Layer Normalization (`eps`, learnable `weight`)
- [x] `GroupedQueryAttention` — GQA with Q/K norm, variate/time bias, variate/time projection
  - `dim`, `num_heads`, `num_groups`, `bias`, `softmax_scale`
  - `var_attn_bias: BinaryAttentionBias`, `time_qk_proj: QueryKeyProjection(RotaryProjection)`
- [x] `MultiHeadAttention` — GQA의 `num_groups=num_heads` 특수 케이스
- [x] `MultiQueryAttention` — GQA의 `num_groups=1` 특수 케이스
- [x] `FeedForward` — Standard `fc1 → gelu → fc2`
- [x] `GatedLinearUnitFeedForward` — `(fc1 * silu(fc_gate)) → fc2`, hidden dim = `ceil(4d * 2/3 / 8) * 8`
- [x] `MoEFeedForward` — Centroid-based routing, `num_experts`, `num_experts_per_token`, top-k selection
- [x] `TransformerEncoderLayer` — Pre/Post-norm, residual connection
- [x] `TransformerEncoder` — Stacked layers, shared/unshared bias/projection, optional MoE centroid buffer
- [x] `PatchEmbedding` — Non-overlapping (`reshape`) + Overlapping (`unfold`) + validity mask — **현재 사용 중**
- [x] `MultiResolutionPatchEmbedding` — Per-variate patch_size, `nn.ModuleDict` projections — **[비활성]** 단일 patch_size 전략으로 전환. 구현 코드는 `module/patch.py`에 보존하나 `BiosignalFoundationModel`에서 사용하지 않음.
- [x] `PackedAbsMeanScaler` — `loc=0`, `scale=mean(|x|)` per (sample_id, variate_id) group
- [x] `PackedStdScaler` — Z-score per group (`loc=mean`, `scale=std`)
- [x] `RotaryProjection` — RoPE with auto-extend `max_len`
- [x] `BinaryAttentionBias` — `nn.Embedding(2, num_heads)` for same/different variate
- [x] `QueryKeyProjection` — Wrapper with `partial_factor` support

### 1.2 모델 레벨

> **[설계 결정 — 단일 patch_size]**
> `BiosignalFoundationModel`은 단일 고정 `patch_size`만 사용한다.
> 모든 신호가 전처리 단계에서 `target_sampling_rate = 100Hz`로 resampling되므로
> `MultiResolutionPatchEmbedding` 및 `variate_patch_sizes` 분기 로직은 불필요하다.
> 모델 생성자에서 `patch_embed = PatchEmbedding(patch_size, d_model)`만 사용한다.

- [x] `BiosignalFoundationModel` — Scaler → PatchEmbedding → SpatialEmbed → TransformerEncoder → Head
  - 입력: `PackedBatch`
  - 출력 dict:
    - `encoded`: `(B, N, d_model)` — 항상
    - `reconstructed`: `(B, N, patch_size)` — task="masked"
    - `cross_pred`: `(B, N, patch_size)` — task="masked" (cross-modal 예측)
    - `next_pred`: `(B, N, patch_size)` — task="next_pred"
    - `loc`, `scale`: `(B, L, 1)` — 항상
    - `patch_mask`: `(B, N)` bool — 항상
    - `patch_sample_id`, `patch_variate_id`: `(B, N)` long — 항상
    - `time_id`: `(B, N)` long — 항상 (cross-modal 페어링용)

### 1.3 Task Head 구조
- [x] `self.head = nn.Linear(d_model, patch_size)` — 자기 variate 복원 (MPM)
- [x] `self.next_head = nn.Linear(d_model, patch_size)` — same-variate next-patch prediction
- [x] `self.cross_head = nn.Linear(d_model, patch_size)` — cross-modal 예측 (다른 variate)
- [x] `self.horizon_embed = nn.Embedding(max_horizon, d_model)` — random horizon conditioning

### 1.4 Inference API
- [x] `extract_features(batch)` — 양방향 attention feature 추출 (reconstructed 제거)
- [x] `forecast(batch, horizon, denormalize)` — 단일-step next-patch prediction + denormalize
- [x] `generate(batch, n_steps, denormalize)` — autoregressive 다단계 생성
- [x] `_append_patch_to_batch()` — PackedBatch에 새 패치 append 헬퍼

### 1.5 Checkpoint
- [x] `save_checkpoint()` — model + optimizer + epoch + config + extra 저장
- [x] `load_checkpoint()` — state_dict 로드 + 메타 반환

---

## 2. MoE (Mixture of Experts) 검증

### 2.1 라우팅 안정화
- [ ] `MoEFeedForward` centroid 초기화 전략 검토
  - 현재: `TransformerEncoder`에서 `nn.Parameter(torch.empty(num_layers, 32, d_model))` 버퍼 할당
  - 문제: 초기화 방법이 명시되지 않음 (Xavier? Kaiming? Random?)
- [ ] `num_experts_per_token` 파라미터에 따른 라우팅 편향(routing collapse) 방지 전략
  - Load balancing loss 추가 고려
- [ ] MoE 전용 테스트 케이스 추가 (`tests/test_ffn.py` 또는 `tests/test_integration.py`)
  - Expert utilization 분포 확인
  - Gradient flow through expert routing

### 2.2 MoE + 모델 통합
- [ ] `BiosignalFoundationModel`에 `use_moe=True` 옵션 검증
  - `TransformerEncoder(use_moe=True)` → centroid buffer 생성 확인
  - Forward pass에서 centroid가 정상 전달되는지 확인
- [ ] MoE 활성화 시 파라미터 수 계산 (`total_params`, `active_params`) 리포팅

---

## 3. Decoder / Task Head 설계

### 3.1 현재 Head 구조 (구현 완료)
```python
# model/biosignal_model.py
self.head = nn.Linear(d_model, patch_size)        # 자기 variate 복원 (MPM)
self.next_head = nn.Linear(d_model, patch_size)   # same-variate 다음 패치 예측
self.cross_head = nn.Linear(d_model, patch_size)  # cross-modal 예측 (다른 variate)
self.horizon_embed = nn.Embedding(max_horizon, d_model)  # horizon conditioning
```

### 3.2 추가 Head 구현 (미구현)
- [ ] `module/head.py` 생성
- [ ] `ForecastingHead` — multi-step 예측
  ```python
  class ForecastingHead(nn.Module):
      def __init__(self, d_model: int, patch_size: int, horizon: int):
          self.proj = nn.Linear(d_model, patch_size * horizon)
      def forward(self, encoded: torch.Tensor) -> torch.Tensor:
          # return: (B, N, horizon, patch_size)
  ```
- [ ] Head 단위 테스트 (`tests/test_head.py`)

> **[분류 태스크 규약]** 분류(수면 스테이징, 이상 탐지 등)는 Prototypical Network 기반 거리 분류만 허용.
> `nn.Linear` 등 학습 가능한 classification head 추가 엄격 금지.
> 분류 평가 파이프라인은 `plan_eval.md` Section 5 참조. Model Architect는 분류 head를 담당하지 않음.

---

## 4. 모델 스케일링 실험

### 4.1 설정 체계화
- [ ] `model/config.py` 생성 — 모델 설정 데이터클래스
  ```python
  @dataclass
  class ModelConfig:
      d_model: int = 64
      num_layers: int = 2
      patch_size: int = 128           # 단일 고정값 (100Hz 기준 1.28s)
      target_sampling_rate: float = 100.0  # 전처리 resampling 목표 Hz
      stride: int | None = None
      num_heads: int | None = None    # None → d_model // 64
      num_groups: int | None = None   # None → num_heads (MHA)
      use_glu: bool = True
      use_moe: bool = False
      use_rope: bool = True
      use_var_attn_bias: bool = True
      dropout_p: float = 0.0
      d_ff: int | None = None         # None → 4 * d_model
      max_horizon: int = 5
  ```
- [ ] `BiosignalFoundationModel.from_config(config: ModelConfig)` 팩토리 추가

### 4.2 스케일 정의
- [ ] Small: `d_model=64, num_layers=2, patch_size=128` (~36K params)
- [ ] Base: `d_model=256, num_layers=6, patch_size=128` (~?)
- [ ] Large: `d_model=512, num_layers=12, patch_size=128` (~?)
- [ ] 각 스케일의 파라미터 수 계산 및 문서화

---

## 5. 아키텍처 개선 후보

### 5.1 Attention 개선
- [ ] Flash Attention 2 호환성 확인 (`F.scaled_dot_product_attention` 백엔드)
- [ ] Sliding Window Attention (긴 시퀀스 효율화)

### 5.2 Position Encoding 개선
- [ ] Learnable Absolute PE (baseline 비교용)
- [ ] ALiBi (Attention with Linear Biases) 구현
- [ ] RoPE `partial_factor` 최적 비율 탐색

### 5.3 Normalization 개선
- [ ] Pre-norm vs Post-norm 성능 비교 (현재 `pre_norm=True`)
- [ ] DeepNorm (deep network 안정화)

---

## 6. 핵심 참조 코드

### 모델 생성 패턴
```python
from model import BiosignalFoundationModel

model = BiosignalFoundationModel(
    d_model=64,
    num_layers=2,
    patch_size=128,
    use_rope=True,
    use_var_attn_bias=True,
    max_horizon=5,
)
# model.scaler       → PackedAbsMeanScaler
# model.patch_embed   → PatchEmbedding(128, 64)
# model.encoder       → TransformerEncoder(64, 2, ...)
# model.head          → nn.Linear(64, 128)        (reconstruction)
# model.next_head     → nn.Linear(64, 128)        (next-patch prediction)
# model.cross_head    → nn.Linear(64, 128)        (cross-modal prediction)
# model.horizon_embed → nn.Embedding(5, 64)       (horizon conditioning)
```

### Forward 흐름
```python
# task="masked" (양방향 attention)
out = model(batch, task="masked")
out["encoded"]           # (B, N, d_model=64) — 인코딩된 표현
out["reconstructed"]     # (B, N, patch_size=128) — 복원된 패치
out["cross_pred"]        # (B, N, patch_size=128) — cross-modal 예측
out["loc"]               # (B, L, 1) — 정규화 역변환용 위치
out["scale"]             # (B, L, 1) — 정규화 역변환용 스케일
out["patch_mask"]        # (B, N) — 유효 패치 마스크
out["patch_sample_id"]   # (B, N) — 패치별 sample_id
out["patch_variate_id"]  # (B, N) — 패치별 variate_id
out["time_id"]           # (B, N) — 패치별 시간 인덱스

# task="next_pred" (causal attention)
out = model(batch, task="next_pred", horizon=3)
out["next_pred"]         # (B, N, patch_size=128) — horizon 조건부 예측
# cross_pred, reconstructed는 없음
```

### TransformerEncoder 내부 흐름
```python
encoder(
    embedded,          # (B, N, d_model)
    attn_mask=attn_mask,  # (B, N, N) bool — same sample & valid patch
    var_id=p_vid,      # (B, N) long — BinaryAttentionBias 용
    time_id=time_id,   # (B, N) long — RoPE 용 (variate 내 패치 순서)
)
```

### Attention 내부 ID 매핑
```
var_id   → BinaryAttentionBias: 같은 variate vs 다른 variate 바이어스
time_id  → RotaryProjection (RoPE): 시간적 위치 인코딩
attn_mask → 다른 sample 간 attention 차단 + 패딩 마스킹
```

### 테스트 현황
```bash
pytest tests/ -v  # 200 tests passed
# 주요 테스트: test_integration.py (39 tests)
#   - TestCrossModalLoss (5 tests): cross-modal loss 검증
#   - TestMaskedPatchLoss (2 tests): MaskedPatchLoss 단위 검증
#   - TestCreatePatchMask (2 tests): create_patch_mask 검증
#   - TestModelCrossHead (3 tests): cross_head 출력/gradient 검증
```
