# Data Engineer 서브 플랜

> **지침**: 이 파일은 **Data Engineer** 역할의 서브 에이전트 전용 계획표이다.
> 작업을 시작하기 전 이 파일을 읽고, 할당된 `[ ]` 작업을 수행한 뒤 성공하면 `[x]`로 상태를 업데이트하라.
> 작업 전 반드시 `.plans/master_plan.md`의 **데이터 규약** (섹션 2)을 확인하라.

---

## 담당 파일

| 파일 | 역할 |
|------|------|
| `data/dataset.py` | `BiosignalDataset`, `BiosignalSample`, `RecordingManifest` |
| `data/collate.py` | `PackCollate`, `PackedBatch`, FFD bin-packing |
| `data/dataloader.py` | `create_dataloader()` |
| `data/sampler.py` | `GroupedBatchSampler` |
| `data/parser/sleep_edf.py` | Sleep-EDF EDF → .pt 변환 |
| `data/parser/` | 추가 데이터셋 파서 |
| `tests/test_data.py` | 데이터 파이프라인 테스트 |

---

## 1. 기존 구현 현황 (완료)

- [x] `RecordingManifest` 데이터클래스 정의 (`path`, `n_channels`, `n_timesteps`, `sampling_rate`, `signal_type`, `session_id`)
- [x] `BiosignalSample` 데이터클래스 정의 (`values: (time,)`, `length`, `channel_idx`, `recording_idx`, `sampling_rate`, `signal_type`, `session_id`, `win_start`)
- [x] `BiosignalDataset` — Lazy-loading + LRU cache (`cache_size` 파라미터)
- [x] `BiosignalDataset` — Sliding window 지원 (`window_seconds`, `stride_seconds`)
- [x] `BiosignalDataset` — Channel-Independent (CI) 패러다임: 모든 채널을 독립 샘플로 전개
- [x] `BiosignalDataset.from_tensors()` — 테스트용 in-memory 생성 팩토리
- [x] `PackedBatch` 데이터클래스 — `values`, `sample_id`, `variate_id`, `lengths`, `sampling_rates`, `signal_types`, `padded_lengths`, `variate_patch_sizes`
- [x] `PackCollate` — FFD bin-packing (First-Fit Decreasing)
- [x] `PackCollate` — `collate_mode="ci"` (채널 독립) / `"any_variate"` (세션 기반 그루핑)
- [x] `PackCollate` — `patch_size` 정렬 패딩 (각 variate를 patch_size 배수로 올림)
- [x] `PackCollate` — `stride` 파라미터 지원 (overlapping patch 정렬)
- [x] `PackCollate` — `patch_sizes` + `target_patch_duration_ms` (다중 해상도 패치) — **[비활성]** 단일 patch_size 전략으로 전환, 사용하지 않음
- [x] `GroupedBatchSampler` — `(session_id, physical_time_ms)` 기반 그루핑
- [x] `create_dataloader()` — `patch_size`, `stride` 파라미터 전달
- [x] `data/parser/sleep_edf.py` — EDF → .pt 변환, 채널별 native sampling rate 추출

---

## 2. 이슈 수정

### 2.1 sample_id가 전부 1로 나오는 문제
- [x] 조사 완료 — **버그 아님**, 파라미터 의존적 동작
  - `max_length=5000`일 때 sample 길이 3072 (30s×100Hz, patch_size=128 정렬) → row당 1 unit만 패킹 → sample_id 모두 1
  - `max_length=50000` (master_plan 기본값)이면 ~16 units 패킹 → sample_id 다양해짐
  - CI/any_variate 모두 동일 현상
- [x] `test_main3.py`의 디버그 `exit()` 제거 완료

### 2.2 Sleep-EDF 파서 안정화
- [x] `_raw_extras[0]` 방어 코드 — `try-except` + `raw.info['sfreq']` fallback 적용
- [x] 채널별 native sampling rate 검증 — 상이 시 stderr 로그 출력
- [x] bandpass Nyquist 엣지 케이스 — `hi <= lo` 시 필터 스킵 + 경고 로그

---

## 2.3 Spatial Positional Encoding 데이터 파이프라인
- [x] `data/spatial_map.py` 신규 생성 — 매핑 테이블, `get_global_spatial_id()`, `TOTAL_SPATIAL_IDS`
- [x] `RecordingManifest`에 `spatial_ids: list[int] | None = None` 추가
- [x] `BiosignalSample`에 `spatial_id: int = 0` 추가
- [x] `PackedBatch`에 `spatial_ids: torch.Tensor  # (total_variates,) long` 추가
- [x] `PackCollate`에서 spatial_id 수집 (signal_types와 동일 패턴)
- [x] `sleep_edf.py`에 `CHANNEL_SPATIAL` 매핑 + manifest 출력에 `spatial_ids` 포함
- [x] 기존 23개 테스트 하위 호환성 유지 확인

---

## 3. 추가 데이터셋 파서

### 3.1 파서 공통 인터페이스

> **[설계 결정 — Resampling 규약]**
> 모든 파서는 저장 전에 `target_sampling_rate = 100 Hz`로 resampling을 수행해야 한다.
> `manifest.json`의 `sampling_rate` 필드는 반드시 `target_sampling_rate` 값(100.0)으로 기록한다.
> `mne.resample()` 또는 `scipy.signal.resample_poly()`를 사용한다.

각 파서는 다음 출력을 생성해야 한다:
```
datasets/processed/{subject_id}/
  manifest.json            # master_plan.md 2.1절 스키마 준수, sampling_rate=100.0
  {session_id}_{type}.pt   # torch.Tensor (n_channels, n_timesteps) float32 — 100Hz로 resampling 완료
```

- [x] **[Priority: High]** `data/parser/_common.py` — 파서 공통 유틸리티
  - `resample_to_target(signal: np.ndarray, orig_sr: float, target_sr: float = 100.0) -> np.ndarray`
    - `scipy.signal.resample_poly` 사용, `orig_sr == target_sr`이면 복사 없이 통과
  - `quality_gate(signal: np.ndarray, min_duration_s: float, sr: float) -> bool` — 최소 길이 검증
  - `save_recording(tensor: torch.Tensor, out_path: str)` — float32 강제 변환 후 저장
- [ ] **[Priority: High]** `data/parser/shhs.py` — SHHS (Sleep Heart Health Study) 파서
  - 입력: EDF (EEG, ECG, EMG, SaO2, Airflow, Thorax, Abdomen)
  - signal_type 매핑: EEG→2, ECG→0, EMG→4, Resp→5
  - 원본 sampling rate가 다를 수 있음 (EEG 125Hz, ECG 250Hz 등) → **모두 100Hz로 resampling**
  - 출력 manifest의 `sampling_rate`는 항상 100.0
- [ ] `data/parser/mesa.py` — MESA (Multi-Ethnic Study of Atherosclerosis) 파서
  - 원본 sampling rate → **100Hz로 resampling** 후 저장
- [ ] `data/parser/physionet_ecg.py` — PhysioNet ECG 데이터셋 파서
  - 원본 sampling rate → **100Hz로 resampling** 후 저장

### 3.2 파서 테스트
- [ ] `tests/test_parser.py` 생성 — 각 파서의 출력 포맷 검증
  - manifest.json 스키마 검증
  - .pt 파일 shape 검증: `(n_channels, n_timesteps)`
  - **sampling_rate == 100.0 강제 검증** (resampling 정합성)
  - resampling 전후 신호 길이 비율 검증 (`n_timesteps_out / n_timesteps_in ≈ 100 / orig_sr`)

### 3.3 sleep_edf.py resampling 적용
- [x] **[Priority: High]** `data/parser/sleep_edf.py` — resampling 로직 추가
  - 현재: native sampling rate 그대로 저장
  - 변경: 로드 후 `resample_to_target(signal, orig_sr, target_sr=100.0)` 적용 후 저장
  - manifest의 `sampling_rate` 항목을 100.0으로 통일
  - 의존성: `_common.py` 구현 완료 후 적용 가능

---

## 4. 데이터 증강 (Augmentation)

### 4.1 Random Crop (배치 단위 길이 변동)
`BiosignalDataset` 레벨에서 윈도우 내 랜덤 sub-segment를 crop하여 입력 길이 다양성을 확보한다.

**설계 방향**: `window_seconds`는 상한(최대 윈도우)으로 고정하고, 실제 입력 길이 변동은 Random Crop이 담당한다.
이를 통해 DataLoader 재생성 없이 다양한 길이의 입력을 생성할 수 있으며,
커리큘럼 학습 시 epoch마다 `crop_ratio_range`만 조절하여 "짧은 시퀀스 → 긴 시퀀스" 점진적 확장이 가능하다.

- [ ] `BiosignalDataset`에 `crop_ratio_range: tuple[float, float]` 파라미터 추가 (기본값 `(1.0, 1.0)` = crop 비활성)
  - `(0.3, 1.0)` → 윈도우 길이의 30~100%를 랜덤 crop (e.g., window=30s → 9~30s)
  - `__getitem__`에서 crop ratio를 uniform 샘플링, 랜덤 start offset 결정
  - crop 후 길이가 patch_size 배수가 아니어도 PackCollate가 패딩 처리하므로 문제없음
  - `crop_ratio_range`는 런타임에 변경 가능하도록 setter 또는 mutable 속성으로 노출
- [ ] 최소 길이 보장: crop 후 길이가 `patch_size * 2` 미만이면 crop 비적용 (패치 1~2개로는 학습 불안정)
- [ ] **any_variate 모드 주의**: 같은 session의 채널들은 동일 구간을 crop해야 시간 정렬 유지
  - CI 모드: 각 채널 독립 crop → 문제없음
  - any_variate 모드: `(session_id, win_start)` 기준으로 동일 crop offset/length 적용 필요
  - 구현 방안: `__getitem__`에서 `(recording_idx, win_start)`를 seed로 crop 파라미터 결정 → 같은 그룹은 같은 crop

### 4.2 기타 증강
- [ ] `TimeShift` — 랜덤 시간축 이동 (circular shift)
- [ ] `ScaleJitter` — 진폭 랜덤 스케일링 (0.8~1.2배)
- [ ] `GaussianNoise` — 가우시안 노이즈 주입 (SNR 기반)
- [ ] `ChannelDropout` — 랜덤 채널 마스킹 (any_variate 모드)

### 4.3 증강 통합
- [ ] `BiosignalDataset`에 `transforms: list[Callable]` 파라미터 추가
- [ ] `__getitem__` 반환 전 transforms 적용 (Random Crop 이후)
- [ ] 증강 단위 테스트

---

## 5. 대규모 데이터 대응

### 5.1 프로파일링
- [ ] 10,000+ 레코딩 로드 시 `BiosignalDataset.__init__()` 시간 측정
- [ ] `PackCollate` FFD 패킹의 배치당 시간 측정
- [ ] LRU cache hit rate 모니터링

### 5.2 최적화
- [ ] `BiosignalDataset` — `use_mmap=True` 성능 검증 (Windows에서 동작 확인)
- [ ] `PackCollate` — 대규모 배치(batch_size > 64)에서 FFD 성능 프로파일링
- [ ] `num_workers > 0` 멀티프로세스 로딩 안정성 검증
- [ ] `persistent_workers=True` + `prefetch_factor` 튜닝

---

## 6. 핵심 참조 코드

### BiosignalDataset 생성 패턴
```python
from data import BiosignalDataset, RecordingManifest

manifest = [
    RecordingManifest(
        path="datasets/processed/SC_00/SC4001E0_eeg.pt",
        n_channels=2, n_timesteps=2880000,
        sampling_rate=100.0, signal_type=2, session_id="SC4001E0",
    ),
]
dataset = BiosignalDataset(manifest, window_seconds=30.0, cache_size=16)
```

### DataLoader 생성 패턴
```python
from data import create_dataloader

# Phase 1: Channel-Independent (1_channel_independency.py)
dataloader = create_dataloader(
    dataset, max_length=50000, batch_size=16,
    shuffle=True, collate_mode="ci",
    patch_size=128,
)

# Phase 2: Any-Variate (2_any_variate.py)
dataloader = create_dataloader(
    dataset, max_length=50000, batch_size=4,
    shuffle=True, collate_mode="any_variate",
    patch_size=128,
)
```

### PackedBatch 필드 접근
```python
for batch in dataloader:
    batch.values        # (B, max_length)
    batch.sample_id     # (B, max_length) — 1-based, 0=padding
    batch.variate_id    # (B, max_length) — 1-based, 0=padding
    batch.lengths       # (total_variates,)
    batch.sampling_rates  # (total_variates,)
    batch.signal_types    # (total_variates,)
    batch.padded_lengths  # (total_variates,) or None
```
