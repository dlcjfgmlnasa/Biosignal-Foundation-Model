---
name: Single patch_size design decision
description: Multi-resolution patch 미사용 결정 — 강제 resampling + 단일 patch_size 전략 (확정 2026-03-25)
type: project
---

모든 신호를 전처리 단계에서 `target_sampling_rate = 100Hz`로 강제 resampling한 뒤 단일 고정 `patch_size=128`을 사용한다.

**Why:** `MultiResolutionPatchEmbedding`은 신호 유형별 sampling rate 차이를 모델 내부에서 처리하는 방식이었으나, 전처리 단계에서 통일하는 것이 아키텍처를 단순하게 유지하면서 동일한 문제를 해결한다.

**How to apply:**
- 모든 파서(`sleep_edf.py`, `shhs.py` 등)는 저장 전에 100Hz resampling 필수
- `manifest.json`의 `sampling_rate` 필드는 항상 100.0
- `PackCollate`에서 `patch_sizes` / `target_patch_duration_ms` / `variate_patch_sizes` 파라미터 사용 금지
- `BiosignalFoundationModel`은 `PatchEmbedding`(단일)만 사용, `MultiResolutionPatchEmbedding` 분기 없음
- `ModelConfig`에 `target_sampling_rate: float = 100.0` 필드 추가 예정
- patch_size=128 timesteps = 1.28s at 100Hz
