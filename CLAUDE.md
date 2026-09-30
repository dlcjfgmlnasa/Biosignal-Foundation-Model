# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A biosignal foundation model built with PyTorch. Early-stage research project targeting deep learning on physiological signal data (ECG, EEG, etc.).

## Environment Setup

- **Python**: 3.13.3 via `.venv/` (virtualenv)
- **Activate venv**: `source .venv/Scripts/activate` (Windows/bash)
- **Key deps**: `torch` 2.10.0, `einops`, `mne`

No `requirements.txt` exists yet — installed packages are in `.venv/`.

## Storage Layout

데이터는 두 디렉토리로 분리 (자세한 내용: `docs/data_pipeline.md` 의 "Storage Strategy"):
- `processed/<dataset>/` — `manifest_full.jsonl` + per-recording `.pt` (옵션)
- `sharded/<dataset>/` — `shard_index.json` + `shard_*.pt` (학습 시 핵심)

신규 데이터셋 추가는 **`scripts/parse_to_shard.py`** 단일 명령 권장 (parse → shard → cleanup 1-pass).
기존 코드 (`data/parser/vitaldb.py` + `scripts/build_shards.py`)는 점진적 진화 결과로 2-step.

## Architecture

```
module/     # Reusable neural network building blocks
model/      # High-level model definitions (assembles modules)
data/       # Data loading and preprocessing pipeline
loss/       # Loss functions (MaskedPatchLoss, NextPredictionLoss, CombinedLoss)
train/      # Training scripts and utilities (Phase 1 CI, Phase 2 Any-Variate)
main.py     # Entry point / training orchestration
```

**Data flow**: `data/` → `model/` (composed of `module/` primitives) → training loop in `main.py`

### Implemented Components

**`module/norm.py`**: `RMSNorm` (Root Mean Square Layer Normalization) + `LSCNorm` (RMSNorm + AdaLN-style affine modulation conditioned on `cond` vector, zero-init for safe pretrain swap-in). Encoder의 모든 layer norm은 LSCNorm을 사용한다.

**`module/transformer.py`**: TransformerEncoder / TransformerEncoderLayer with GQA, GLU FFN, MoE, RoPE, LSCNorm modulation 지원. 모든 layer가 `cond` 벡터를 받아 per-channel scale·shift gating 적용.

**`module/attention.py`**: GroupedQueryAttention, MultiHeadAttention, MultiQueryAttention.

**`module/ffn.py`**: FeedForward, GatedLinearUnitFeedForward, MoEFeedForward.

**`module/packed_scaler.py`**: PackedStdScaler, PackedAbsMeanScaler for packed batch normalization.

**`data/dataset.py`**: BiosignalDataset (Channel-Independent, lazy-loading, sliding window).

**`data/collate.py`**: PackCollate (FFD bin-packing collate). `patch_size`/`stride` 정렬 패딩. `spatial_ids` 전달 (v2: spatial_id 폐지로 값이 signal_type과 동일, 모델 미사용 — plumbing 유지).

**`data/spatial_map.py`**: signal_type 매핑 테이블 (v2: 단일 modality embedding, 10종 — ECG(0)·ABP(1)·PPG(2)·CVP(3)·CO2(4)·AWP(5)·ICP(6)·RESP_Impedance(7)·RESP_Flow(8)·PAP(9). PAP는 2026-09-13 SNUH OR 재학습을 위해 9번 끝 슬롯으로 복원(구 disk 6 → v2 9 remap, 기존 0~8 불변; PAP 학습 시 config `num_signal_types: 10`). RESP를 Impedance(7)/Flow(8)로 분리, ECG lead 통합, PAP 완전 제거(2026-06-23, 구 6번 슬롯 삭제 후 뒤 번호 1칸씩 당김 — ICP 7→6, RESP_Imp 8→7, RESP_Flow 9→8), spatial_id 소분류 폐지). `SIGNAL_TYPE_NAMES`, `CHANNEL_NAME_TO_SIGNAL_TYPE`, `remap_record_v2()` (load-time remap: 디스크 manifest 는 구 disk spec 유지 → PAP 6→9 · RESP 7/8 분기 · spatial 평탄화 · ICP 7→6), `MECHANISM_GROUP`, `CROSS_PRED_ALLOWED_PAIRS` (강결합 γ쌍 (0,1)(0,2)(1,2)(5,8)). `get_global_spatial_id()`/`TOTAL_SPATIAL_IDS`는 하위호환용 유지(반환값 = signal_type).

**`module/patch.py`**: PatchEmbedding (고정/overlapping 패치 토큰화) — Residual MLP projection (TimesFM 스타일). 단일 해상도.

**`model/biosignal_model.py`**: BiosignalFoundationModel — Scaler → PatchEmbedding → ModalityEmbedding(signal_type) → TransformerEncoder(LSCNorm) → Head 파이프라인. signal_type 단일 modality embedding (v2: num_signal_types=9, spatial_id 이중 임베딩 폐지 — RESP Impedance/Flow는 별도 signal_type, PAP 제거 2026-06-23). (loc, scale)은 `cond_proj`(MLP 2→d_cond→d_cond)를 거쳐 모든 layer의 LSCNorm modulation으로 주입됨. ⚠ 실제 granularity는 **window 단위**다 — `PackedStdScaler._make_group_key`가 `(sample_id, variate_id)` 쌍으로만 그룹을 만들어, 한 window × 한 채널당 loc/scale 하나가 계산되어 전 timestep에 broadcast된다(patch 인덱스는 그룹 키에 없음). 따라서 조건 경로에는 **시간 해상도가 없고**, 창 내부 추세는 담기지 않는다(2026-09-19 실측 확인). `task="masked"` (양방향 attention → reconstruction head + cross_head) / `task="next_pred"` (causal attention → next-patch head) 단일 encoder 기반 멀티태스크. forward 출력에 `time_id`, `cross_pred` 포함. `d_cond` (default=16)는 hyperparameter로 조정 가능.

**`loss/masked_mse_loss.py`**: MaskedPatchLoss (마스킹된 패치 MSE) + create_patch_mask (랜덤/variate-level 마스킹). Phase 2에서 variate_mask_prob로 전체 variate 마스킹 지원.

**`loss/next_prediction_loss.py`**: NextPredictionLoss — same-variate next-patch prediction + cross-modal prediction (같은 time_id, 다른 variate 간 예측).

**`loss/criterion.py`**: CombinedLoss — `α*MPM + β*NextPred + γ*CrossModal` 복합 손실 (⚠ γ 는 β 에 곱해지지 않는 **독립 가중치**다). **contrastive(δ)는 2026-09-22 전면 제거** — modality embedding 으로 정답이 풀려 붕괴했다. 되살리지 말 것. `learnable_coupling` 이 켜지면 결합 그래프 희소화 항 `coupling_l1 * mean(W)` 가 추가된다. MaskedMSELoss (하위 호환).

**`train/train_utils.py`**: 학습 공유 유틸리티 — TrainConfig, train_one_epoch(), load_manifest_from_processed(), checkpoint 헬퍼.

**`train/1_channel_independency.py`**: Phase 1 CI 사전학습 스크립트. collate_mode="ci", MPM + Next-Pred, random horizon.

**`train/2_any_variate.py`**: Phase 2 Any-Variate 학습 스크립트. Phase 1 checkpoint 로드, collate_mode="any_variate", cross-modal loss(γ), variate-level 마스킹.

**`data/parser/vitaldb.py`**: .vital 파서 (VitalDB Open·K-MIMIC·SNUH OR 공용). `TRACK_MAP` 선언 순서가 곧 우선순위(같은 signal_type 중복 시 먼저 나온 트랙 채택 — Intellivue ECG_II 500Hz > ECG_II_WAV 250Hz, Intellivue 가스모듈 > 마취기). 2026-09-13: SNUH 마취기 트랙(Primus·MedibusX·Datex-Ohmeda·CS2·CS650) 추가, `TRACK_UNIT_SCALE`로 GE CO2 %→mmHg(×7.13)·Dräger AWP hPa→cmH2O 통일, raw-0 sentinel(offset<0 트랙에서 값==offset) NaN 처리. Dräger Atlan은 스케일 불량으로 제외.

**`data/parser/sleep_edf.py`**: Sleep-EDF raw EDF → processed .pt 변환 스크립트.

## Coding Conventions

- 텐서 타입은 `torch.Tensor`로 선언하고, 차원은 인라인 주석으로 명시: `x: torch.Tensor,  # (batch, seq_len, dim)`
- Follow the `RMSNorm` pattern: `nn.Module` subclass with explicit `__init__` typed params and a typed `forward()`.
- No tests or linting config yet — when adding, prefer `pytest` and `ruff`.

## Downstream Tasks

**SSOT 는 논문 초고의 Table 1** (`obsidian/90_Paper/draft/draft_npj_ver02_ch.md`). 아래는 그 사본이며 불일치 시 Table 1 이 이긴다. 2026-09-22 기준 **21개 task / 9개 코호트** — 내부(SNUH·VitalDB) 10 · 외부 11 · MIMIC-III 계열 8 (Ext-CA 는 심정지 라벨 출처이지 task 아님). tier 명칭은 영문 유지.

### 1. Detection (5) — 현재 구간에 사건이 있는가
Arrhythmia(VitalDB 413, 4-class) · AF Detection(Ext-PPG 6,189) · False VT Alarm(VTaC 2,117 rec) · Hypoxemia(SNUH Vent 167) · Respiratory Event(CinC 2018 994)

### 2. Prediction (3) — 관측 종료 5분 뒤
Intraoperative Hypotension(VitalDB · ABP 단독 · 5분 창) · Massive Transfusion(SNUH MT 5,850) · Intracranial Hypertension(MIMIC-III 121)
- **ICU Hypotension(MIMIC-III) 은 2026-09-22 논문에서 완전히 제외.** 임상 기준선(MAP 추세 다변수 회귀)에 4조건 전부 패배했고, 저혈압 과제는 수술중(IOH)·ABP 단독으로 단일화했다. 되살리지 말 것.

### 3. Outcome (5) — 입원 단위
- Postoperative AKI(VitalDB 135) · In-Unit Mortality(MIMIC-III 1,485) · Mechanical Ventilation(MIMIC-III 2,857)
- **Cardiac Arrest** (MIMIC-III · 양성 532 + 1:2 매칭 음성 = 1,596 · ICU 입실 직후 3시간 관측). 2026-09-20 Detection → Outcome 이동, 구 5/10/15분 horizon 폐기 — 양성 정의의 주 출처인 ICD 코드에 시각이 없다. 코호트도 SCOPE(양성 74)에서 교체. 음성 매칭에 **재실 기간 필수**(심정지군 중앙값 5.2일 vs 2.8일, 노출 시간 교란).
- **ICU Length of Stay** (독립 무작위 1,000 · LOS > 3일 · 유병률 33.6% · 2026-09-20 신설). **심정지와 코호트를 공유하지 않는다** — 심정지의 재실 기간 매칭이 본 과제의 정답 분포를 평탄화한다. **3일 내 사망자 제외**(짧은 이유가 회복이 아니라 사망). 생리 과제가 아니므로 인구학·공변량 기준선이 강할 것 — Δ 를 그대로 보고.

### 4. Estimation (6) — 임상 절단점 이진 분류로 평가
PPV(229) · SVV(163) · Arterial Pressure · Serum Albumin(602) · Respiratory Rate(1,013) · Ejection Fraction LVEF(875, ABP 보유자)

### 5. Phenotype (2)
Age(Aurora-BP 1,209) · Sex(VitalDB 657)

### 논문 본문에서 제외된 것
Physiological Generation(Cross-Modal Reconstruction · Waveform Forecasting) 은 초고 본문에서 빠졌다. head 자체는 남아 있으므로 `cross_head`/`next_head` 를 재사용·fine-tune 하는 경로는 유효하다. Sepsis Prediction 도 Table 1 에 없다.

**Head 전략**: tier 1~5 는 사전학습 encoder 위에 task-specific classification head 부착(`extract_features()` 활용). 생성 계열만 사전학습 head 재사용.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
