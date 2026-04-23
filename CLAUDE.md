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

**`module/norm.py` — `RMSNorm`**: Root Mean Square Layer Normalization as a `torch.nn.Module`. Supports optional learnable weight (`gamma`), configurable `normalized_shape`, and numerical stability via `eps`.

**`module/transformer.py`**: TransformerEncoder / TransformerEncoderLayer with GQA, GLU FFN, MoE, RoPE support.

**`module/attention.py`**: GroupedQueryAttention, MultiHeadAttention, MultiQueryAttention.

**`module/ffn.py`**: FeedForward, GatedLinearUnitFeedForward, MoEFeedForward.

**`module/packed_scaler.py`**: PackedStdScaler, PackedAbsMeanScaler for packed batch normalization.

**`data/dataset.py`**: BiosignalDataset (Channel-Independent, lazy-loading, sliding window).

**`data/collate.py`**: PackCollate (FFD bin-packing collate). `patch_size`/`stride` 정렬 패딩, `patch_sizes`/`target_patch_duration_ms` 다중 해상도 지원. `spatial_ids` 전역 spatial_id 전달.

**`data/spatial_map.py`**: signal_type(대분류) + spatial_id(소분류) 이중 인코딩 매핑 테이블. `get_global_spatial_id()`, `TOTAL_SPATIAL_IDS`, `CHANNEL_NAME_TO_SPATIAL`.

**`module/patch.py`**: PatchEmbedding (고정/overlapping 패치 토큰화), MultiResolutionPatchEmbedding (MOIRAI 스타일 다중 해상도).

**`model/biosignal_model.py`**: BiosignalFoundationModel — Scaler → PatchEmbedding → SpatialEmbedding → TransformerEncoder → Head 파이프라인. signal_type + spatial_id 이중 임베딩(Dual Additive Embedding) 지원. `task="masked"` (양방향 attention → reconstruction head + cross_head) / `task="next_pred"` (causal attention → next-patch head) 단일 encoder 기반 멀티태스크. forward 출력에 `time_id`, `cross_pred` 포함.

**`loss/masked_mse_loss.py`**: MaskedPatchLoss (마스킹된 패치 MSE) + create_patch_mask (랜덤/variate-level 마스킹). Phase 2에서 variate_mask_prob로 전체 variate 마스킹 지원.

**`loss/next_prediction_loss.py`**: NextPredictionLoss — same-variate next-patch prediction + cross-modal prediction (같은 time_id, 다른 variate 간 예측).

**`loss/contrastive_loss.py`**: CrossModalContrastiveLoss — InfoNCE 기반 cross-modal contrastive loss. same (sample_id, time_id) + different variate_id = positive pair. Learnable log-temperature (CLIP 방식).

**`loss/criterion.py`**: CombinedLoss — α*MPM + β*(NextPred + γ*CrossModal) + δ*Contrastive 복합 손실. MaskedMSELoss (하위 호환).

**`train/train_utils.py`**: 학습 공유 유틸리티 — TrainConfig, train_one_epoch(), load_manifest_from_processed(), checkpoint 헬퍼.

**`train/1_channel_independency.py`**: Phase 1 CI 사전학습 스크립트. collate_mode="ci", MPM + Next-Pred, random horizon.

**`train/2_any_variate.py`**: Phase 2 Any-Variate 학습 스크립트. Phase 1 checkpoint 로드, collate_mode="any_variate", cross-modal loss(γ), variate-level 마스킹.

**`data/parser/sleep_edf.py`**: Sleep-EDF raw EDF → processed .pt 변환 스크립트.

## Coding Conventions

- 텐서 타입은 `torch.Tensor`로 선언하고, 차원은 인라인 주석으로 명시: `x: torch.Tensor,  # (batch, seq_len, dim)`
- Follow the `RMSNorm` pattern: `nn.Module` subclass with explicit `__init__` typed params and a typed `forward()`.
- No tests or linting config yet — when adding, prefer `pytest` and `ruff`.
