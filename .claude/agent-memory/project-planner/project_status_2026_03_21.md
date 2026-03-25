---
name: 프로젝트 진척도 스냅샷 (2026-03-21)
description: 각 Phase별 완료율 및 workstream별 핵심 미완료 항목 목록
type: project
---

2026-03-21 기준 진척도 스냅샷.

**Why:** 매 대화마다 plan 파일 전체를 재독하는 비용 절감. 단, plan 파일이 업데이트되면 이 메모리보다 plan 파일이 우선.

**How to apply:** 새 대화 시 빠른 컨텍스트 복원에 활용. 항목 변경 시 plan 파일 확인 후 이 메모리도 갱신.

---

## 전체 진척도 (2026-03-21)

| Workstream | 완료율 | 미완료 항목 수 |
|---|---|---|
| Data Engineer | ~75% | 파서 3개, 증강 5개, 프로파일링 4개 |
| Model Architect | ~65% | MoE 검증, ForecastingHead, ModelConfig, 스케일링 |
| Train & Eval | ~70% | AMP, Validation/EarlyStopping, 로깅, DDP |

## Phase별 요약

### Phase 0 (기반 정비): 95% 완료
- 미완: `main.py`를 실제 진입점으로 교체 (현재 빈 템플릿)

### Phase 1 (데이터 파이프라인): 40% 완료
- 완료: Dataset, CollateFFD, Sampler, DataLoader, Sleep-EDF 파서, spatial_map
- 미완: SHHS/MESA/PhysioNet 파서, 데이터 증강, 프로파일링

### Phase 2 (모델 아키텍처): 80% 완료
- 완료: 모든 핵심 모듈, BiosignalFoundationModel, CombinedLoss, Inference API
- 미완: MultiResolutionPatchEmbedding 검증, MoE 안정화, ForecastingHead, ModelConfig

### Phase 3 (학습 파이프라인): 70% 완료
- 완료: 2-Phase 스크립트, Checkpoint, gradient clipping, CosineAnnealingLR
- 미완: AMP, warmup, Validation 루프, EarlyStopping, DDP

### Phase 4 (Inference API): 90% 완료
- 완료: extract_features, forecast, generate, _append_patch_to_batch
- 미완: loss 수렴 확인, 시각화

### Phase 5 (다운스트림 평가): 85% 완료
- 완료: eval/__init__, forecasting.py, fewshot.py, imputation.py, _metrics.py
- 미완: module/metrics.py (사전학습 메트릭), 시각화 스크립트

## 중요 구조 변경 사항 (plan 파일 미반영 → 반영 완료)
- `train_utils.py`, `1_channel_independency.py`, `2_any_variate.py`가 루트에서 `train/` 서브패키지로 이동됨
- `eval/_metrics.py` 신규 추가 (공통 회귀 메트릭 헬퍼)
- `user_test.py` 루트에 존재 (실험용 임시 스크립트 — plan 미기재)
- 테스트: 200 tests passing (plan_model.md 기준)
