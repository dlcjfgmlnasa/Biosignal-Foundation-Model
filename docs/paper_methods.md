# 3. Methods

## 3.1 Data: Intraoperative Biosignal Dataset

### 3.1.1 VitalDB Dataset

We utilize the VitalDB open dataset [CITATION NEEDED], a large-scale collection of intraoperative physiological recordings from Seoul National University Hospital. VitalDB contains continuous waveform data from approximately 6,000 surgical cases, encompassing seven types of biosignals simultaneously acquired during general anesthesia: electrocardiogram (ECG), arterial blood pressure (ABP), electroencephalogram (EEG), photoplethysmogram (PPG), central venous pressure (CVP), capnography (CO2), and airway pressure (AWP). Each signal type is assigned a unique integer identifier $s \in \{0, 1, 2, 3, 4, 5, 6\}$ corresponding to ECG, ABP, EEG, PPG, CVP, CO2, and AWP, respectively.

> 서울대학교병원의 수술중 생리학적 기록을 대규모로 수집한 VitalDB 공개 데이터셋 [CITATION NEEDED]을 활용한다. VitalDB는 약 6,000건의 수술 사례로부터 전신마취 중 동시 획득된 7종 생체신호의 연속 파형 데이터를 포함한다: 심전도(ECG), 동맥혈압(ABP), 뇌파(EEG), 광용적맥파(PPG), 중심정맥압(CVP), 호기말이산화탄소(CO2), 기도내압(AWP). 각 신호 타입에 고유 정수 식별자 $s \in \{0, 1, 2, 3, 4, 5, 6\}$을 부여하여 ECG, ABP, EEG, PPG, CVP, CO2, AWP에 각각 대응시킨다.

The raw waveforms are recorded at heterogeneous sampling rates: 500 Hz for electrical and hemodynamic signals (ECG, ABP, PPG, CVP via SNUADC), 128 Hz for EEG (BIS monitor), and 62.5 Hz for respiratory signals (CO2, AWP via Primus ventilator). To enable unified processing, all signals are resampled to a common rate of 100 Hz using polyphase rational resampling, yielding a Nyquist frequency of 50 Hz sufficient for all clinically relevant spectral content.

> 원시 파형은 이질적인 샘플링 레이트로 기록된다: 전기 및 혈역학 신호(ECG, ABP, PPG, CVP — SNUADC 경유)는 500 Hz, EEG(BIS 모니터)는 128 Hz, 호흡 신호(CO2, AWP — Primus 인공호흡기)는 62.5 Hz이다. 통합 처리를 위해 모든 신호를 다상(polyphase) 유리수 리샘플링으로 100 Hz 공통 레이트로 변환하며, 이로써 임상적으로 관련된 모든 주파수 성분을 포함하는 50 Hz 나이퀴스트 주파수를 확보한다.

### 3.1.2 Signal-Specific Preprocessing

We apply a tailored preprocessing pipeline that respects the distinct physical measurement principles and artifact profiles of each signal modality. Signals are categorized into four groups based on their transduction mechanism [TABLE 1]:

> 각 신호 모달리티의 고유한 물리적 측정 원리와 아티팩트 특성을 반영한 맞춤형 전처리 파이프라인을 적용한다. 신호는 변환(transduction) 메커니즘에 따라 4개 그룹으로 분류된다 [TABLE 1]:

**Electrical signals** (ECG, EEG): Surface biopotentials susceptible to power-line interference and electrosurgical (Bovie) artifacts. Processed with bandpass filtering (ECG: 0.5-40 Hz; EEG: 0.5-45 Hz), 60 Hz notch filtering, and MAD-based spike detection ($10\sigma$ threshold) to blank electrosurgical contamination windows.

> **전기 신호** (ECG, EEG): 전원 간섭 및 전기소작기(Bovie) 아티팩트에 취약한 체표면 생체전위. 대역통과 필터(ECG: 0.5-40 Hz; EEG: 0.5-45 Hz), 60 Hz 노치 필터, MAD 기반 스파이크 검출($10\sigma$ 임계치)을 적용하여 전기소작기 오염 구간을 블랭킹한다.

**Fluid-coupled pressure signals** (ABP, CVP): Invasive catheter-based measurements prone to mechanical artifacts (line flush, clotting, catheter resonance). Processed with lowpass filtering (ABP: 15 Hz; CVP: 10 Hz) preserving the DC component (absolute pressure values), median filtering (kernel size 5 for ABP/PPG) to remove impulse noise, and step-change detection for level-shift artifacts.

> **액체 매개 압력 신호** (ABP, CVP): 기계적 아티팩트(라인 세척, 혈전, 카테터 공명)에 취약한 침습적 카테터 기반 측정. DC 성분(절대 혈압값)을 보존하는 저역통과 필터(ABP: 15 Hz; CVP: 10 Hz), 임펄스 노이즈 제거를 위한 중앙값 필터(커널 크기 5), 레벨 시프트 아티팩트를 위한 계단 변화 검출을 적용한다.

**Optical signals** (PPG): Infrared photoplethysmography susceptible to motion artifacts. Processed with lowpass filtering (8 Hz), 60 Hz notch filtering, second-derivative-based motion artifact detection, and median filtering.

> **광학 신호** (PPG): 체동 아티팩트에 취약한 적외선 광용적맥파. 저역통과 필터(8 Hz), 60 Hz 노치 필터, 2차 미분 기반 체동 아티팩트 검출, 중앙값 필터를 적용한다.

**Gas/respiratory signals** (CO2, AWP): Ventilator-integrated measurements with minimal measurement artifacts -- apparent "noise" typically reflects genuine clinical events (patient bucking, secretions). Processed with conservative lowpass filtering only (CO2: 5 Hz; AWP: 20 Hz) to preserve clinically meaningful waveform features.

> **기체/호흡 신호** (CO2, AWP): 측정 아티팩트가 거의 없는 인공호흡기 내장 측정 — 겉보기 "노이즈"는 대개 환자의 실제 임상 이벤트(bucking, 가래)를 반영한다. 임상적으로 의미 있는 파형 특징을 보존하기 위해 보수적 저역통과 필터만 적용한다(CO2: 5 Hz; AWP: 20 Hz).

Each signal undergoes physiological range validation (e.g., ABP: 20-300 mmHg, ECG: $\pm$5 mV), with out-of-range values replaced by NaN. Contiguous NaN-free segments exceeding 2 seconds are extracted and independently quality-assessed.

> 각 신호에 생리학적 범위 검증(예: ABP: 20-300 mmHg, ECG: $\pm$5 mV)을 수행하여 범위 밖 값을 NaN으로 대체한다. 2초를 초과하는 연속 NaN-free 세그먼트를 추출하여 독립적으로 품질을 평가한다.

### 3.1.3 Quality Control

Quality control is performed in two stages. First, a **generic quality gate** evaluates each sliding window (5 seconds for cardiovascular/neural signals, 15 seconds for respiratory signals) using four metrics: flatline ratio (consecutive identical values), clipping ratio (saturation at extremes), high-frequency energy ratio (first-derivative energy relative to signal energy), and peak-to-peak amplitude. Signal-specific thresholds are calibrated from empirical distributions [TABLE 1].

> 품질 관리는 2단계로 수행된다. 첫째, **범용 품질 게이트**가 슬라이딩 윈도우(심혈관/신경 신호: 5초, 호흡 신호: 15초) 단위로 4개 지표를 평가한다: 플랫라인 비율(연속 동일 값), 클리핑 비율(극값 포화), 고주파 에너지 비율(1차 미분 에너지 대 신호 에너지), peak-to-peak 진폭. 신호별 임계치는 실측 분포로부터 교정된다 [TABLE 1].

Second, **domain-specific quality checks** apply physiological plausibility constraints:
- *ECG/ABP/PPG*: R-peak or pulse-peak detection, heart rate validation (30-200 bpm), inter-beat interval regularity (coefficient of variation), and autocorrelation peak strength at physiological lag ranges.
- *EEG*: Spectral band power ratio (1-30 Hz band $\geq$ 10% of total power) and normalized spectral entropy ($0.3 \leq H_{\text{norm}} \leq 0.95$) to reject both pure noise and flatline signals.
- *CO2/AWP*: Respiratory rate estimation (4-40 breaths/min) via peak detection on the ventilation waveform.

> 둘째, **도메인 특화 품질 검사**가 생리학적 타당성 제약을 적용한다:
> - *ECG/ABP/PPG*: R-peak 또는 맥파 피크 검출, 심박수 유효성(30-200 bpm), 박동 간격 규칙성(변동계수), 생리학적 지연 범위에서의 자기상관 피크 강도.
> - *EEG*: 주파수 대역 파워 비율(1-30 Hz 대역이 전체 파워의 $\geq$ 10%)과 정규화 스펙트럼 엔트로피($0.3 \leq H_{\text{norm}} \leq 0.95$)로 순수 노이즈와 플랫라인 신호를 동시에 배제.
> - *CO2/AWP*: 환기 파형에서 피크 검출을 통한 호흡수 추정(4-40 회/분).

Only windows passing both stages are retained. Consecutive passing windows from the same variate are concatenated into contiguous segments and stored in HDF5 format (float16, gzip compression) with per-subject manifest files recording metadata (signal type, spatial location, sampling rate, segment length).

> 양 단계를 모두 통과한 윈도우만 보존한다. 같은 variate의 연속 통과 윈도우를 이어붙여 연속 세그먼트를 구성하고, HDF5 포맷(float16, gzip 압축)으로 저장하며 피험자별 매니페스트 파일에 메타데이터(신호 타입, 공간 위치, 샘플링 레이트, 세그먼트 길이)를 기록한다.

### 3.1.4 Spatial Encoding

Each signal channel is encoded with a two-level spatial identification scheme. The **signal type** $s \in \{0, \ldots, 6\}$ identifies the broad modality, while a **local spatial ID** $l$ distinguishes anatomical placement within each type (e.g., ECG Lead II vs. Lead V5, ABP Radial vs. Femoral). A deterministic mapping function $g(s, l) \rightarrow z$ converts these pairs to globally unique spatial IDs $z \in \{0, \ldots, 11\}$, which are used for spatial embedding in the model [FIGURE 1].

> 각 신호 채널은 2단계 공간 식별 체계로 인코딩된다. **신호 타입** $s \in \{0, \ldots, 6\}$은 대분류 모달리티를 식별하고, **로컬 공간 ID** $l$은 각 타입 내에서 해부학적 측정 위치를 구분한다(예: ECG Lead II vs. Lead V5, ABP 요골동맥 vs. 대퇴동맥). 결정적 매핑 함수 $g(s, l) \rightarrow z$가 이 쌍을 전역 고유 공간 ID $z \in \{0, \ldots, 11\}$로 변환하며, 이는 모델의 공간 임베딩에 사용된다 [FIGURE 1].

## 3.2 Model Architecture

Our model follows a Transformer encoder architecture with several domain-specific adaptations for heterogeneous biosignal processing. The overall pipeline is: **Instance Normalization -> Patch Tokenization -> Spatial Embedding -> Transformer Encoder -> Task-Specific Heads** [FIGURE 2].

> 본 모델은 이질적 생체신호 처리를 위한 도메인 특화 적응을 갖춘 Transformer 인코더 아키텍처를 따른다. 전체 파이프라인은 **인스턴스 정규화 -> 패치 토큰화 -> 공간 임베딩 -> Transformer 인코더 -> 태스크별 헤드**로 구성된다 [FIGURE 2].

### 3.2.1 Instance Normalization

Given a packed input batch $\mathbf{x} \in \mathbb{R}^{B \times L}$ with per-timestep sample identifiers $\mathbf{s}_{\text{id}} \in \mathbb{Z}^{B \times L}$ and variate identifiers $\mathbf{v}_{\text{id}} \in \mathbb{Z}^{B \times L}$, we compute per-variate Z-score statistics using scatter-based aggregation:

$$\mu_{g} = \frac{\sum_{t: g(t)=g} x_t}{|g|}, \quad \sigma_{g} = \sqrt{\frac{\sum_{t: g(t)=g} (x_t - \mu_g)^2}{|g| - 1} + \epsilon}$$

where $g$ indexes the unique (sample_id, variate_id) group, and $\epsilon = 10^{-5}$. The normalized signal is $\hat{x}_t = (x_t - \mu_g) / \sigma_g$.

> 패킹된 입력 배치 $\mathbf{x} \in \mathbb{R}^{B \times L}$과 타임스텝별 샘플 식별자 $\mathbf{s}_{\text{id}} \in \mathbb{Z}^{B \times L}$ 및 variate 식별자 $\mathbf{v}_{\text{id}} \in \mathbb{Z}^{B \times L}$이 주어지면, scatter 기반 집계로 variate별 Z-score 통계량을 계산한다. 여기서 $g$는 고유한 (sample_id, variate_id) 그룹의 인덱스이고, $\epsilon = 10^{-5}$이다. 정규화된 신호는 $\hat{x}_t = (x_t - \mu_g) / \sigma_g$이다.

### 3.2.2 Patch Tokenization

The normalized sequence is segmented into non-overlapping patches of size $P = 100$ (corresponding to 1 second at 100 Hz). Each patch inherits the sample_id, variate_id, and signal_type, and is assigned a **time_id** representing its ordinal position within its variate. Patch-to-embedding projection is performed by a linear projection $\mathbf{e}_i = W_{\text{proj}} \mathbf{p}_i + b_{\text{proj}}$.

> 정규화된 시퀀스를 크기 $P = 100$의 비중첩 패치(100 Hz에서 1초에 해당)로 분할한다. 각 패치는 sample_id, variate_id, signal_type을 상속받고, variate 내 순서 위치를 나타내는 **time_id**가 부여된다. 선형 투영 $\mathbf{e}_i = W_{\text{proj}} \mathbf{p}_i + b_{\text{proj}}$으로 패치를 임베딩으로 변환한다.

### 3.2.3 Dual Spatial Embedding

$$\mathbf{e}_i \leftarrow \mathbf{e}_i + \left(\text{Emb}_{\text{signal}}(s_i) + \text{Emb}_{\text{spatial}}(z_i)\right) \cdot \mathbb{1}[v_i > 0]$$

> 모달리티 정체성과 해부학적 측정 위치를 동시에 인코딩하는 이중 가산 임베딩. 패딩 토큰($v_i = 0$)은 영벡터를 유지한다.

### 3.2.4 Loc/Scale Injection

$$\mathbf{e}_i \leftarrow \mathbf{e}_i + \left(W_{\text{loc}} \mu_{g(i)} + W_{\text{scale}} \sigma_{g(i)}\right) \cdot \mathbb{1}[v_i > 0]$$

> 인스턴스 정규화 과정에서 소실된 절대적 생리학적 크기 정보를 학습된 선형 투영을 통해 토큰 표현에 다시 주입한다. 이는 저혈압 ABP 파형(MAP < 65 mmHg)과 정상 혈압 파형을 정규화 후에도 구분할 수 있게 한다.

### 3.2.5 Transformer Encoder

The token sequence is processed by $L$ Transformer encoder layers with pre-norm (RMSNorm), featuring:

1. **Grouped Query Attention (GQA)** with $H$ heads in $G$ groups
2. **Rotary Position Embedding (RoPE)** using time_id
3. **Binary Attention Bias**: learned same-variate vs. cross-variate bias
4. **GLU Feed-Forward Network** with GELU activation

> 토큰 시퀀스는 pre-norm(RMSNorm)의 $L$개 Transformer 인코더 레이어로 처리된다: GQA, RoPE, 이진 어텐션 바이어스, GLU FFN.

Attention masking enforces: (1) same-sample only attention, (2) padding exclusion, (3) causal mask for next-patch prediction.

> 어텐션 마스킹: (1) 동일 샘플 내만 어텐드, (2) 패딩 배제, (3) 다음 패치 예측 시 인과적 마스크.

### 3.2.6 Base Configuration

$d = 256$, $L = 12$, $H = 8$, GLU FFN, RoPE, BinaryAttentionBias, dropout $p = 0.1$, ~10M parameters, FP16 mixed precision.

> 기본 구성: $d = 256$, $L = 12$, $H = 8$, GLU FFN, RoPE, 이진 어텐션 바이어스, 드롭아웃 $p = 0.1$, 약 1,000만 파라미터, FP16 혼합 정밀도.

## 3.3 EEG Spectral Reconstruction Target

For each EEG patch $\mathbf{p} \in \mathbb{R}^P$ (signal_type $= 2$), we compute the STFT with Hann window, $N_{\text{fft}} = 64$, hop $= 16$, yielding 33 frequency bins $\times$ 3 time frames = 99-dimensional target:

$$\mathbf{y} = \text{vec}\left(\log(1 + |S|)\right) \in \mathbb{R}^{99}$$

followed by per-patch z-score normalization. Frequency resolution: $\approx 1.56$ Hz, covering delta through gamma bands.

> 각 EEG 패치에 대해 Hann 윈도우, $N_{\text{fft}} = 64$, 홉 $= 16$으로 STFT를 계산하여 33 주파수 빈 $\times$ 3 시간 프레임 = 99차원 타겟을 생성한다. 패치별 z-score 정규화 적용. 주파수 해상도: $\approx 1.56$ Hz, delta~gamma 대역 커버.

Separate 2-layer MLP heads ($d \rightarrow d \rightarrow 99$) for masked reconstruction and next-patch prediction target the spectral representation. Non-EEG signals use raw-waveform MSE.

> 별도 2층 MLP 헤드($d \rightarrow d \rightarrow 99$)로 마스크 복원과 다음 패치 예측의 스펙트럼 표현을 타겟으로 한다. 비-EEG 신호는 원시 파형 MSE를 사용한다.

## 3.4 Pre-training Objectives

$$\mathcal{L} = \alpha \mathcal{L}_{\text{MPM}} + \beta \left(\mathcal{L}_{\text{next}} + \gamma \mathcal{L}_{\text{cross}}\right) + \delta \mathcal{L}_{\text{contrastive}}$$

> 네 가지 목적함수를 결합한 복합 손실 함수.

### 3.4.1 Masked Patch Modeling (MPM)

$\mathcal{L}_{\text{MPM}} = \mathcal{L}_{\text{MPM}}^{\text{raw}} + \mathcal{L}_{\text{MPM}}^{\text{EEG}}$ — random patch masking ($r = 0.3$), bidirectional context. Phase 2 adds variate-level masking ($p_v = 0.3$) for virtual sensing.

> 랜덤 패치 마스킹($r = 0.3$), 양방향 컨텍스트. Phase 2에서 가상 센싱을 위한 variate 수준 마스킹($p_v = 0.3$) 추가.

### 3.4.2 Next-Patch Prediction

$\hat{\mathbf{p}}_{t+h} = W_{\text{next}} (\mathbf{z}_t^{\text{causal}} + \mathbf{h}_h)$, horizon-weighted MSE with $h \sim \text{Uniform}\{1, \ldots, H_{\max}\}$.

> 인과적 어텐션 마스킹 하에서 호라이즌 임베딩 조건화된 다음 패치 예측. 호라이즌 가중 MSE.

### 3.4.3 Cross-Modal Prediction

Restricted to same **mechanism group** (Cardiovascular: ECG/ABP/PPG/CVP, Neural: EEG, Respiratory: CO2/AWP).

> 같은 **메커니즘 그룹** 내로 제한 (심혈관: ECG/ABP/PPG/CVP, 신경: EEG, 호흡: CO2/AWP).

### 3.4.4 Cross-Modal Contrastive Loss

InfoNCE with learnable temperature, operates across all mechanism groups for representation alignment.

> 학습 가능한 온도의 InfoNCE, 표현 정렬을 위해 모든 메커니즘 그룹에 걸쳐 작동.

## 3.5 Data Collation and Packing

FFD bin packing with two modes: **Channel-Independent (CI)** for Phase 1 unimodal learning, **Any-Variate** for Phase 2 cross-modal learning.

> FFD 빈 패킹, 두 모드: Phase 1 단일 모달 학습용 **채널 독립(CI)**, Phase 2 교차 모달 학습용 **Any-Variate**.

## 3.6 Two-Phase Curriculum Training

**Phase 1 (CI)**: $\alpha=1.0, \beta=0.3, \gamma=0, \delta=0$, $H_{\max}=1$, AdamW lr=$8\times10^{-4}$, 100 epochs, 30% masking.

> **Phase 1 (CI)**: 채널 독립 모드, MPM + 다음 패치 예측만, AdamW lr=$8\times10^{-4}$, 100 에폭.

**Phase 2 (AV)**: $\alpha=0.7, \beta=0.3, \gamma=1.0, \delta=0.1$, variate masking $p_v=0.3$, lr=$10^{-4}$, 30 epochs.

> **Phase 2 (AV)**: 전체 손실 활성, variate 마스킹, 교차 모달 학습, lr=$10^{-4}$, 30 에폭.

---

[TABLE 1]: Signal-specific preprocessing parameters and quality thresholds.
[TABLE 2]: Mechanism group assignments for cross-modal constraint.
[FIGURE 1]: Spatial encoding scheme with signal_type and spatial_id dual embedding.
[FIGURE 2]: Overall model architecture diagram.
