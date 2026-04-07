# Signal Preprocessing & Quality Control

7종 수술중 생체신호의 측정 원리, 임상적 의미, 전처리 파이프라인, 신호별 QC 전략을 정리한다.

---

## 0. 생체신호 측정 원리 및 임상적 역할

수술 중 환자 모니터링에 사용되는 7종 생체신호의 측정 원리와 각 신호가 임상에서 무엇을 보기 위해 사용되는지를 정리한다.

### 0.1 ECG (심전도, Electrocardiogram)

**측정 원리**: 심장 근육이 수축/이완할 때 발생하는 **미세 전류**(mV 단위)를 피부 표면의 전극으로 감지한다. 심장의 전기 전도 시스템(동방결절 → 방실결절 → His속 → Purkinje 섬유)을 따라 전파되는 탈분극/재분극 전위차를 기록한다.

**무엇을 보는가**:
- **심박수(HR)**: R-R 간격으로 계산. 정상 60-100 bpm.
- **리듬(Rhythm)**: 부정맥 감지 — 심방세동, 심실빈맥, 서맥, 심실세동 등
- **ST segment**: 심근 허혈/경색의 조기 지표. ST 상승/하강은 관상동맥 혈류 부족을 의미
- **P-QRS-T 형태**: P파(심방 수축), QRS(심실 수축), T파(심실 재분극)의 형태 변화로 전해질 이상, 심비대 등 감지
- **수술 중 핵심 역할**: 실시간 부정맥 감시. 마취제/수술 조작에 의한 심장 자극 반응 모니터링

**VitalDB 소스**: SNUADC/ECG_II (Lead II, 주모니터링), SNUADC/ECG_V5 (V5, 보조). 500Hz 샘플링.

### 0.2 ABP (동맥혈압, Arterial Blood Pressure)

**측정 원리**: 요골동맥(radial artery) 또는 대퇴동맥(femoral artery)에 가느다란 **카테터(동맥 라인)**를 삽입하고, 카테터 내부를 생리식염수로 채워 외부의 **압력 트랜스듀서**에 연결한다. 심장이 박동할 때마다 동맥 내 압력 변화가 액체를 통해 트랜스듀서에 전달되어 전기 신호로 변환된다.

**무엇을 보는가**:
- **수축기 혈압(SBP)**: 심실 수축 시 최고 압력. 정상 ~120 mmHg
- **이완기 혈압(DBP)**: 심실 이완 시 최저 압력. 정상 ~80 mmHg
- **평균 동맥압(MAP)**: 장기 관류압 지표. MAP < 65 mmHg = 저혈압 경보
- **맥압(Pulse Pressure)**: SBP - DBP. 심박출량과 혈관 탄성의 지표
- **Dicrotic notch**: 대동맥판 폐쇄 시점. 소실되면 overdamping 또는 혈관 순응도 변화 의미
- **beat-to-beat 변동**: 수액 반응성(fluid responsiveness), 자율신경 반응 평가
- **수술 중 핵심 역할**: 실시간 혈역학 모니터링. 출혈, 수액 투여, 혈관작용제 반응 즉시 확인

**VitalDB 소스**: SNUADC/ART (요골동맥), SNUADC/FEM (대퇴동맥). 500Hz 샘플링.

### 0.3 EEG (뇌파, Electroencephalogram)

**측정 원리**: 대뇌 피질 신경세포의 **시냅스 후 전위(postsynaptic potential)**를 두피(이마) 표면에 부착한 전극으로 감지한다. 수술 중에는 주로 BIS(Bispectral Index) 모니터를 사용하며, 이마에 일회용 센서 스트립을 부착하여 전두부 EEG를 측정한다.

**무엇을 보는가**:
- **마취 깊이(Depth of Anesthesia)**: BIS 값 0-100 (0=isoelectric, 40-60=적정 마취, 60-100=각성). EEG 파형의 주파수 대역 비율로 산출
- **주파수 대역 변화**:
  - Delta (0.5-4Hz): 깊은 마취, 서파 수면. 진폭 크고 느림
  - Theta (4-8Hz): 중등도 마취
  - Alpha (8-13Hz): 가벼운 마취. 프로포폴 유도 alpha 활동
  - Beta (13-30Hz): 각성 상태. 피질 활성화
- **Burst Suppression**: 과도한 마취 시 뇌 활동이 간헐적으로 멈추는 패턴 (burst ↔ suppression 교대). 진폭 ±5μV 이하 구간이 0.5초 이상 지속
- **술중 각성(Intraoperative Awareness)**: EEG 패턴 변화로 환자가 수술 중 깨어나는 것을 조기 감지
- **수술 중 핵심 역할**: 마취 깊이 적정성 모니터링. 과소/과다 마취 방지

**VitalDB 소스**: BIS/EEG1_WAV, BIS/EEG2_WAV. 128Hz 샘플링 (BIS 모니터 내장).

### 0.4 PPG (광용적맥파, Photoplethysmogram)

**측정 원리**: 손가락(또는 이마/귀)에 부착한 센서에서 **적외선 LED**를 발사하고, 조직을 통과한 빛의 양을 **광수신기(photodetector)**로 측정한다. 심장 박동에 따라 혈관 내 혈액량이 변하면 빛의 흡수량이 달라지는 원리를 이용한다. 산화헤모글로빈(적색)과 환원헤모글로빈(적외선)의 흡수 차이로 SpO2를 계산한다.

**무엇을 보는가**:
- **산소포화도(SpO2)**: 동맥혈 산소포화도. 정상 95-100%. PPG의 적색/적외선 파장 비율로 산출
- **맥박수(PR)**: PPG pulse peak 간격으로 심박수 추정
- **말초 관류 지수(PI)**: PPG 진폭 크기. 진폭 작으면 말초 혈류 감소 (쇼크, 저체온, 혈관수축)
- **파형 형태**: systolic peak → dicrotic notch → diastolic valley. ABP와 유사한 형태
- **호흡 변동(Respiratory Variation)**: PPG 진폭의 호흡 주기 변동으로 수액 반응성 평가
- **수술 중 핵심 역할**: 산소화 상태 연속 모니터링. 비침습적이라 모든 환자에 적용

**VitalDB 소스**: SNUADC/PLETH (손가락). 500Hz 샘플링.

### 0.5 CVP (중심정맥압, Central Venous Pressure)

**측정 원리**: 경정맥(internal jugular vein) 또는 쇄골하정맥(subclavian vein)을 통해 **중심정맥 카테터**를 삽입하여 상대정맥/우심방 접합부의 압력을 측정한다. ABP와 동일하게 카테터 내 식염수를 통해 외부 압력 트랜스듀서에 전달한다.

**무엇을 보는가**:
- **우심방 압력/전부하(Preload)**: 정상 CVP 2-8 mmHg. 체액 상태(volume status)의 지표
- **우심부전**: CVP 지속 상승 (>15 mmHg) = 우심실 기능 저하
- **삼첨판 역류**: CVP 파형에 거대 v파 출현
- **CVP 파형 구성**: a파(심방 수축) → c파(삼첨판 돌출) → x 하강(심방 이완) → v파(심방 충만) → y 하강(삼첨판 개방). 진폭이 매우 작다 (2-5 mmHg 변동)
- **호흡 변동**: 기계 환기 시 CVP는 흡기에 상승, 호기에 하강. 이 변동의 크기로 수액 반응성 평가
- **수술 중 핵심 역할**: 수액 요법 가이드. 대량 출혈, 수액 투여, 혈관작용제 반응 평가

**VitalDB 소스**: SNUADC/CVP. 500Hz 샘플링. 중심정맥 카테터가 삽입된 환자만 (~20-30%).

### 0.6 CO2 (호기말 이산화탄소, Capnography)

**측정 원리**: 인공호흡기 호기 라인에 설치된 **적외선 가스 분석기**가 CO2 분자의 적외선 흡수 특성을 이용하여 호기 가스 중 CO2 농도를 실시간 측정한다. CO2 분자는 특정 파장(4.26μm)의 적외선을 강하게 흡수하는 성질을 가진다.

**무엇을 보는가**:
- **EtCO2 (End-tidal CO2)**: 호기말 CO2 분압. 정상 35-45 mmHg. 폐포 환기 적절성의 가장 직접적인 지표
- **환기 적절성**: EtCO2 상승 = 과소 환기 또는 CO2 생성 증가. EtCO2 하강 = 과환기 또는 심박출량 감소
- **기관삽관 확인**: 삽관 직후 CO2 파형이 나타나면 기관 내 삽관 확인 (식도 삽관 시 CO2 없음)
- **폐색전(PE) 조기 감지**: 폐색전 발생 시 EtCO2 급격히 하강 (사강 증가)
- **Capnogram 파형 진단**:
  - **Shark fin (상어 지느러미)**: 기관지 경련/천식 → Phase II-III가 비스듬한 삼각형. 기도 폐색 지표
  - **Curare cleft (V자 홈)**: 근이완제 풀림 → Phase III plateau 중간에 V자 함몰. 자발 호흡 시도 = 마취 깊이 지표
  - **Cardiogenic oscillation**: 심장 박동이 폐에 전달 → plateau 위 자잘한 톱니. Cross-modal 학습 가능
  - **Rebreathing**: Phase I baseline > 0 → CO2 재호흡. 호흡 회로 문제
- **수술 중 핵심 역할**: 환기 상태 실시간 모니터링. 기관삽관 위치 확인. 심폐 기능의 통합 지표

**VitalDB 소스**: Primus/CO2. 62.5Hz 샘플링 (Primus 마취기 내장).

### 0.7 AWP (기도내압, Airway Pressure)

**측정 원리**: 인공호흡기(Primus) 내부의 **압력 트랜스듀서**가 호흡 회로의 기도 압력을 직접 측정한다. 기계가 환자의 폐로 공기를 밀어넣을 때의 압력과 환자가 내쉴 때의 압력을 실시간으로 기록한다. ABP/CVP와 달리 카테터가 아닌 **호흡 회로 내 기체**를 통해 압력이 전달된다.

**무엇을 보는가**:
- **PIP (Peak Inspiratory Pressure)**: 흡기 시 최고 압력. 정상 15-30 cmH2O. 기도 저항 + 폐 탄성의 합
- **Plateau Pressure**: 흡기 말 정체 시 압력. 폐 compliance만 반영. 30 cmH2O 이하 유지가 폐보호 환기의 핵심
- **PEEP (Positive End-Expiratory Pressure)**: 호기말 양압. 폐포 허탈(atelectasis) 방지. 보통 4-8 cmH2O
- **기도 저항 변화**: PIP 상승 + Plateau 정상 = 기도 저항 증가 (기관지 경련, 튜브 꺾임, 가래)
- **폐 Compliance 변화**: PIP 상승 + Plateau 상승 = 폐 compliance 감소 (기흉, 폐부종, 복강 내압 상승)
- **환기 모드별 파형**: VCV(용적 제어) = 사다리꼴, PCV(압력 제어) = 사각파 상승 + 지수감소
- **임상적으로 중요한 이벤트**:
  - **Bucking/Fighting**: 환자가 인공호흡기와 싸움 → 날카로운 스파이크. 마취 깊이 부족 지표
  - **Auto-PEEP**: 불완전 호기 → 기저선 상승. COPD/천식 환자
  - **Compliance drop**: 복강경 가스 주입, 체위 변경 → 파형 형태 변화
- **수술 중 핵심 역할**: 기계 환기 적절성 모니터링. 폐보호 전략(Plateau < 30) 준수. 기도 문제 조기 감지

**VitalDB 소스**: Primus/AWP. 62.5Hz 샘플링 (Primus 마취기 내장).

### 0.8 신호 간 관계 (Cross-modal)

수술 중 7종 신호는 독립적이 아니라 **생리학적으로 연결**되어 있다:

| 관계 | 연결 메커니즘 | 모델 학습 관점 |
|------|-------------|---------------|
| ECG ↔ ABP | 심장 수축 → 동맥 맥파 전달 (0.1-0.2초 지연) | 같은 time_id에서 cross-attention |
| ECG ↔ PPG | 심장 수축 → 말초 혈류 변화 (0.3-0.5초 지연) | 지연 시간이 혈관 탄성 정보 |
| ABP ↔ PPG | 동일 심박 기원, 파형 형태 유사 | cross-modal prediction |
| EEG ↔ AWP | 마취 깊이 ↔ bucking (마취 얕으면 환기 저항) | 마취 심도 추정 |
| CO2 ↔ AWP | 동일 호흡 주기에 동기화 | 호흡 역학 학습 |
| CVP ↔ ABP | 우심 전부하 → 좌심 박출량 → 동맥압 | 혈역학 루프 |
| AWP cardiogenic osc. ↔ ECG | 심장 박동이 기도압에 전달 | cross-modal 보너스 신호 |

이러한 cross-modal 관계가 Phase 2 Any-Variate 학습에서 cross-attention, cross-modal prediction, contrastive loss를 통해 학습된다.

---

## 1. 전처리 파이프라인 개요

모든 신호는 다음 순서로 처리된다 (`vitaldb.py` 기준):

```
Raw Waveform (VitalDB .vital 파일)
  │
  ├─ Step 1: Range Check — 물리적 불가능 값 → NaN
  ├─ Step 2: Spike Detection — MAD 기반 급격한 변동 → NaN (전기/혈역학 신호만)
  ├─ Step 2b: Step Change Detection — 계단형 아티팩트 → NaN (ABP/PPG만)
  ├─ Step 2c: Motion Artifact Detection — 2차 미분 기반 → NaN (PPG만)
  ├─ Step 3: NaN-free 세그먼트 추출
  ├─ Step 4: Median Filter → Notch Filter → Bandpass/Lowpass
  ├─ Step 5: Resample → 100Hz (공통)
  ├─ Step 6: Window 단위 품질 검사 (기본 + Domain-specific)
  └─ Step 7: 연속 통과 윈도우 그룹 → .zarr 저장
```

---

## 2. 신호 분류 체계

신호를 **측정 원리 + 전달 매질**에 따라 4그룹으로 분류하고 차등 QC를 적용한다.

| 그룹 | 신호 | 측정 원리 | 매질 | 주요 아티팩트 원인 | QC 강도 |
|------|------|----------|------|-------------------|---------|
| **전기** | ECG, EEG | 체표면 전위 | 전기 | 전원 간섭(60Hz), 전기소작기(Bovie), 근전도, 전극 접촉 | 강 |
| **액체 압력** | ABP, CVP | 압력 트랜스듀서 | 혈액/식염수 | Flush, 피떡(clot), 과감쇠/저감쇠, 카테터 공명 | 강 |
| **광학** | PPG | LED/광센서 | 광 | 체동(motion), 손가락 위치, 전기소작기 | 강 |
| **기체/호흡** | AWP, CO2 | 압력계/적외선 | 공기/호기가스 | 거의 없음 — "노이즈"가 임상 이벤트 | 약 |

### 왜 같은 "압력" 신호인 ABP와 AWP를 다르게 처리하는가?

ABP와 AWP는 모두 압력을 측정하지만, **전달 매질의 물리적 특성**이 근본적으로 다르다:

- **ABP (액체 매개)**: 가느다란 동맥 카테터에 비압축성 액체(혈액/식염수)를 채워 측정한다. 액체는 관성이 크고, 튜빙 내 공명(자연 주파수 10-25Hz), 기포/피떡에 의한 감쇠(damping), 간호사의 라인 세척(flush) 등 **측정 시스템 자체의 기계적 오류**가 빈번하다. 이 아티팩트들은 환자의 혈역학 상태와 무관한 **가짜 데이터**이므로 반드시 제거해야 한다.

- **AWP (기체 매개)**: 지름 2cm 이상의 인공호흡기 주름관에 압축성 기체(공기)를 통해 측정한다. 기체는 관성이 작아 공명 현상이 거의 없고, 채혈/flush 같은 외부 개입이 없다. AWP 파형이 흔들리면 그것은 **환자의 폐와 기도에서 발생한 실제 물리적 사건**(bucking, 가래, 외과의 압박)이므로 보존해야 한다.

---

## 3. 신호별 전처리 설정

### 3.1 ECG (심전도)

| 항목 | 설정 | 근거 |
|------|------|------|
| **signal_type** | 0 | |
| **원본 SR** | 500Hz (SNUADC) | |
| **단위** | mV | |
| **valid_range** | -5.0 ~ 5.0 mV | 정상 QRS 진폭 0.5-3.0mV. ±5mV 초과 = 센서 탈락 |
| **filter_type** | bandpass 0.5-40Hz | baseline wander(호흡, <0.5Hz) 제거 + 고주파 노이즈(>40Hz) 제거 |
| **notch** | 60Hz | 한국 전원 주파수 간섭 제거 |
| **spike_detection** | True (10σ) | Bovie(전기소작기) 아티팩트 구간 NaN 마킹 |
| **median_kernel** | 0 (미적용) | |
| **max_high_freq_ratio** | 1.0 | QRS spike의 고주파 에너지 허용. 정상 ~0.4, P99=1.97 |
| **min_high_freq_ratio** | 0.05 | QRS spike가 없으면(hf<0.05) 불량 — baseline만 있는 것 |
| **min_amplitude** | 0.3 mV | QRS 진폭 최소 기준 |
| **quality_window_s** | 5.0s | 심박 주기(~0.8s) 대비 충분한 윈도우 |
| **Domain QC** | `ecg_quality_check()` | R-peak 기반 HR 30-200bpm, regularity(CV<0.7), autocorrelation |

**임상적 의미**:
- ECG는 심장의 전기적 활동을 기록한다. QRS complex(심실 탈분극), P파(심방 탈분극), T파(재분극)로 구성된다.
- **수술 중 주요 관심사**: 부정맥 감지, ST segment 변화(심근 허혈), HR 추세
- **주요 아티팩트**: 전기소작기(Bovie) 사용 시 수 mV 이상의 폭발적 고주파 간섭 발생 → spike_detection으로 구간 NaN 처리. 60Hz 전원 간섭 → notch filter.
- **bandpass 0.5-40Hz 근거**: 0.5Hz 이하는 호흡에 의한 baseline wander. 40Hz 이상은 근전도(EMG) 간섭. QRS의 주요 에너지는 5-25Hz에 집중.

### 3.2 EEG (뇌파)

| 항목 | 설정 | 근거 |
|------|------|------|
| **signal_type** | 2 | |
| **원본 SR** | 128Hz (BIS 모니터) | |
| **단위** | μV | |
| **valid_range** | -500 ~ 500 μV | BIS 하드웨어 최대 입력 ~1000μV p2p. ±500μV 초과 = 센서 탈락 |
| **filter_type** | bandpass 0.5-45Hz | delta(0.5-4Hz) ~ gamma(30-45Hz) 대역 보존 |
| **notch** | 60Hz | 전원 간섭 제거 |
| **spike_detection** | True (10σ) | Bovie 아티팩트 제거 |
| **max_high_freq_ratio** | 2.0 | EEG는 beta/gamma 고주파 성분이 정상적으로 존재 |
| **max_amplitude** | 500 μV | 마취 중 EEG 50-300μV + 여유. 500μV 초과 = 센서 이상 |
| **min_amplitude** | 2 μV | burst suppression(±5μV) 포함. 2μV 미만 = 완전 isoelectric |
| **quality_window_s** | 5.0s | |
| **Domain QC** | `eeg_quality_check()` | FFT 기반 band_ratio(1-30Hz ≥ 10%), spectral entropy(0.3-0.95) |

**임상적 의미**:
- EEG는 대뇌 피질의 전기적 활동을 기록한다. 수술 중에는 BIS(Bispectral Index) 모니터의 전두부 채널을 사용한다.
- **수술 중 주요 관심사**: 마취 깊이 모니터링. 각성(awareness) 시 beta 파워 증가, 깊은 마취 시 delta 우세, burst suppression(과도한 마취)
- **파워 대역**: delta(0.5-4Hz, 깊은 마취), theta(4-8Hz), alpha(8-13Hz, 가벼운 마취), beta(13-30Hz, 각성)
- **의식 상태별 진폭**: 각성 10-50μV, 마취 50-300μV, burst suppression ±5μV 이하
- **주요 아티팩트**: 전기소작기(가장 큰 문제), 안구 운동, 안면 근전도(EMG), 60Hz 전원
- **bandpass 0.5-45Hz 근거**: gamma(30-45Hz)까지 보존하여 EMG contamination 분석 가능. 45Hz 이상은 기기 노이즈.
- **amplitude threshold 근거 (실데이터 검증, 200 VitalDB cases)**:
  - 기존 `min=5, max=200`에서 basic QC pass rate 85.4%, full pass rate 85.2%
  - 변경 `min=2, max=500`에서 basic QC pass rate 88.8%, full pass rate 88.6% (+3.4%)
  - 15,187개 윈도우가 true rescue (domain check 99.3% 확인)
  - false accept 107개 (0.7%) — domain check(band_ratio + entropy)가 2차 방어선으로 제거
  - 기존 `max=200`은 정상 마취 EEG(50-300μV)를 45.7% 오판, `min=5`는 burst suppression을 54.3% 오판

### 3.3 ABP (동맥혈압)

| 항목 | 설정 | 근거 |
|------|------|------|
| **signal_type** | 1 | |
| **원본 SR** | 500Hz (SNUADC) | |
| **단위** | mmHg | |
| **valid_range** | 20 ~ 300 mmHg | 극저혈압 SBP~40mmHg 허용. 300mmHg 초과 = 센서 오류 또는 flush |
| **filter_type** | lowpass 15Hz | DC(절대 혈압값) 보존 + 카테터 공명(15Hz 이상) 제거 |
| **notch** | 없음 | ABP는 전기 신호가 아니라 60Hz 간섭 없음 |
| **spike_detection** | True (6σ) | Flush artifact(300mmHg+ 스파이크) 제거. 임계치 낮음(6σ) = 민감 |
| **median_kernel** | 5 | 임펄스 노이즈(flush, 채혈 시 순간 변동) 제거 |
| **step_change** | 적용 | 트랜스듀서 리셋 시 계단형 level shift 감지 → NaN |
| **max_high_freq_ratio** | 0.5 | ABP는 매우 부드러운 파형. hf>0.5 = underdamping/공명 |
| **max_flatline_ratio** | 0.3 | 동맥라인 막힘 시 일정 값으로 고정 |
| **min_amplitude** | 10 mmHg | pulse pressure 최소 기준. <10 = overdamping 또는 라인 문제 |
| **quality_window_s** | 5.0s | |
| **Domain QC** | `abp_quality_check()` | Systolic peak 기반 HR 30-200bpm, regularity(CV<0.5), autocorrelation |

**임상적 의미**:
- ABP는 요골동맥(radial) 또는 대퇴동맥(femoral)에 삽입한 카테터로 연속 동맥혈압을 측정한다.
- **수술 중 주요 관심사**: beat-to-beat 혈압 변동, 저혈압/고혈압 에피소드, 맥압(pulse pressure) 변화
- **파형 구성**: systolic upstroke → systolic peak → dicrotic notch(대동맥판 폐쇄) → diastolic decay
- **주요 아티팩트**:
  - **Flush artifact**: 간호사가 라인을 세척할 때 300mmHg+ 사각파 → spike_detection + median filter로 제거
  - **Overdamping**: 피떡/기포 → 파형 진폭 감소, dicrotic notch 소실 → min_amplitude<10 검출
  - **Underdamping/Ringing**: 카테터 공명 → systolic 과장, 가짜 고주파 진동 → lowpass 15Hz 제거
  - **Step change**: 트랜스듀서 영점 리셋 시 급격한 baseline shift → step_change_detection
- **lowpass 15Hz 근거**: pulse wave 기본 주파수 ~1Hz(60bpm) + 유의미한 고조파 10Hz까지. 카테터 시스템 자연공명 주파수 10-25Hz이므로 15Hz 이상은 공명 artifact.

### 3.4 PPG (광용적맥파)

| 항목 | 설정 | 근거 |
|------|------|------|
| **signal_type** | 3 | |
| **원본 SR** | 500Hz (SNUADC) | |
| **단위** | unitless (IR absorption) | |
| **valid_range** | 0 ~ 2000 | 센서 물리적 범위. 0 = 빛 차단, 2000 = 센서 포화 |
| **filter_type** | lowpass 8Hz | DC(산소포화도 정보) 보존 + 고주파 노이즈 제거 |
| **notch** | 60Hz | 적외선 LED에 전원 간섭이 들어올 수 있음 |
| **spike_detection** | True (6σ) | 체동, 센서 탈락 시 급격한 변동 |
| **median_kernel** | 5 | 임펄스 노이즈 제거 |
| **step_change** | 적용 | 센서 재부착 시 level shift |
| **motion_artifact** | 적용 | 2차 미분 기반 체동 아티팩트 구간 NaN |
| **max_high_freq_ratio** | 0.05 | PPG는 매우 부드러운 파형. 0.05 이상 = 노이즈 |
| **max_flatline_ratio** | 0.3 | 센서 탈락 시 일정값 고정 |
| **min_amplitude** | 5 | 말초순환 불량 시 진폭 감소 |
| **quality_window_s** | 5.0s | |
| **Domain QC** | `ppg_quality_check()` | Pulse peak 기반 HR 30-200bpm, regularity(CV<0.5), autocorrelation |

**임상적 의미**:
- PPG는 손가락에 부착한 광센서로 혈류량 변화를 측정한다. SpO2(산소포화도) 산출의 원시 파형이다.
- **수술 중 주요 관심사**: SpO2 추세, 말초 관류 상태, 자율신경 반응(PPG 진폭 변동)
- **파형 구성**: systolic peak → dicrotic notch → diastolic valley. ABP와 형태적으로 유사하지만 광학 기반.
- **주요 아티팩트**:
  - **체동(Motion)**: 환자 움직임 시 파형 왜곡 → motion_artifact_detection
  - **전기소작기**: LED 회로에 전기적 간섭 → notch 60Hz + spike_detection
  - **센서 탈락**: 파형 소실 또는 고정값 → flatline_ratio 검사
  - **말초 저관류**: 쇼크/저체온 시 진폭 극소화 → min_amplitude 검사
- **lowpass 8Hz 근거**: PPG는 ABP보다 고주파 성분이 적다. pulse wave 기본 주파수 + 고조파가 5Hz 이내. 8Hz면 충분.
- **max_high_freq_ratio=0.05**: PPG는 본질적으로 부드러운 파형이므로 hf>0.05는 확실한 노이즈 지표.

### 3.5 CVP (중심정맥압)

| 항목 | 설정 | 근거 |
|------|------|------|
| **signal_type** | 4 | |
| **원본 SR** | 500Hz (SNUADC) | |
| **단위** | mmHg | |
| **valid_range** | -5 ~ 40 mmHg | 정상 CVP 2-12mmHg. 음압 = 저혈량, 40+ = 센서 오류 |
| **filter_type** | lowpass 10Hz | DC 보존 + 고주파 제거 |
| **notch** | 없음 | 압력 트랜스듀서 — 전기 간섭 없음 |
| **spike_detection** | True (8σ) | Flush/라인 조작 아티팩트. ABP보다 관대(8σ) = CVP 정상 변동이 작으므로 |
| **median_kernel** | 0 | CVP 진폭이 작아 median filter 불필요 |
| **max_high_freq_ratio** | 0.5 | 저주파 신호 |
| **quality_window_s** | 5.0s | |
| **Domain QC** | `cvp_quality_check()` | a/c/v wave peak 기반 HR 30-200bpm, regularity(CV<0.7), autocorrelation(≥0.15), flatline(<0.3), respiratory variation |

**임상적 의미**:
- CVP는 경정맥(internal jugular vein)에 삽입한 중심정맥 카테터로 우심방 압력을 측정한다.
- **수술 중 주요 관심사**: 체액 상태(volume status) 평가, 우심부전 감지, 호흡 변동(Δ CVP)
- **파형 구성**: a파(심방 수축), c파(삼첨판 돌출), x 하강, v파(심방 충만), y 하강. 진폭이 매우 작다(2-5mmHg 변동).
- **주요 아티팩트**: ABP와 동일한 액체 매개 아티팩트(flush, clot, damping). 다만 진폭이 작아(2-12mmHg) 아티팩트와 정상 파형 구별이 어려움.
- **lowpass 10Hz 근거**: CVP 파형의 유의미한 주파수 성분은 호흡(0.2Hz) + 심박(1Hz) + 고조파(5Hz). ABP보다 낮은 cutoff 가능하지만, 10Hz로 여유를 둠.

### 3.6 CO2 (호기말 이산화탄소)

| 항목 | 설정 | 근거 |
|------|------|------|
| **signal_type** | 5 | |
| **원본 SR** | 62.5Hz (Primus) | |
| **단위** | mmHg | |
| **valid_range** | 0 ~ 100 mmHg | 흡기 CO2 ≈ 0mmHg. 100 초과 = 센서 오류 (생리적 불가) |
| **filter_type** | lowpass 5Hz | 호흡 신호 보존 + 센서 노이즈 제거 |
| **notch** | 없음 | 적외선 흡수 방식 — 전기 간섭 없음 |
| **spike_detection** | False | CO2 변동은 모두 임상적으로 유의미 |
| **median_kernel** | 0 | |
| **max_flatline_ratio** | 0.3 | 정상 capnogram에도 plateau(Phase III)가 있어 flatline으로 오인 가능 → 0.3으로 엄격 |
| **min_amplitude** | 5 mmHg | EtCO2 최소 변동. 5mmHg 미만 = 기계 분리 또는 센서 오류 |
| **quality_window_s** | 15.0s | 호흡 주기(3-5초)를 충분히 포함하려면 15초 필요 |
| **Domain QC** | `co2_quality_check()` | Peak 기반 호흡수 4-40 breaths/min, IQR>0.5 |

**임상적 의미**:
- CO2(Capnography)는 인공호흡기 호기 라인의 적외선 가스 분석기로 호기말 CO2 분압을 측정한다.
- **수술 중 주요 관심사**: 환기 적절성 평가(EtCO2 35-45mmHg 정상), 기관삽관 확인, 폐색전 조기 감지(급격한 EtCO2 하강)
- **Capnogram 4 Phase**:
  - Phase I: 사강(dead space) 호기 — CO2 ≈ 0
  - Phase II: 급격한 상승 (폐포 호기 시작)
  - Phase III: Plateau (폐포 가스 — EtCO2 측정 지점)
  - Phase 0: 흡기 시작 — 급격한 하강
- **임상적으로 중요한 "노이즈" 패턴** (필터링 금지):
  - **Shark fin**: Phase II-III가 비스듬한 삼각형 → 기관지 경련(천식)/COPD. 기도 폐색의 핵심 지표
  - **Curare cleft**: Phase III plateau 중간의 V자 홈 → 근이완제 풀림, 자발 호흡 시도. 마취 깊이 지표
  - **Cardiogenic oscillation**: Plateau 위 자잘한 톱니 → 심장 박동이 폐에 전달. Cross-modal 학습 가능
  - **Rebreathing**: Phase I baseline이 0이 아닌 양의 값 → CO2 재호흡. 호흡 회로 문제
- **lowpass 5Hz 유지 근거**: 위의 모든 임상 패턴은 5Hz 이하에 에너지가 집중됨. Shark fin은 파형 형태 변화, curare cleft는 2-3Hz, cardiogenic oscillation은 1-2Hz.
- **유일한 기계적 노이즈**: Sampling line 수분 맺힘 → 불규칙 스파이크. 이것도 lowpass 5Hz로 자연 제거됨.

### 3.7 AWP (기도내압)

| 항목 | 설정 | 근거 |
|------|------|------|
| **signal_type** | 6 | |
| **원본 SR** | 62.5Hz (Primus) | |
| **단위** | hPa (≈ cmH2O) | |
| **valid_range** | -20 ~ 80 cmH2O | 음압(강한 bucking): -20까지 허용. 80 초과 = 기계 고장/튜브 완전 폐색 |
| **filter_type** | lowpass 20Hz | 임상 이벤트(5-20Hz) 보존 + ADC/센서 노이즈 제거 |
| **notch** | 없음 | 기체 압력 신호 — 전기 간섭 면역 |
| **spike_detection** | False | AWP 스파이크 = 임상 이벤트 (제거 금지) |
| **median_kernel** | 0 | |
| **max_high_freq_ratio** | 1.0 | 관대한 threshold — bucking 등 고주파 이벤트 허용 |
| **min_amplitude** | 2 cmH2O | 기계 환기 시 최소 압력 변동. <2 = 기계 분리/꺼짐 |
| **quality_window_s** | 15.0s | CO2와 동일 — 호흡 주기 충분 포함 |
| **Domain QC** | `awp_quality_check()` | Peak 기반 호흡수 4-40 breaths/min, IQR>0.5 |

**임상적 의미**:
- AWP는 인공호흡기(Primus)가 환자의 폐로 공기를 밀어넣을 때의 기도 압력을 측정한다.
- **수술 중 주요 관심사**: 폐보호 환기(PIP<30cmH2O), 기도 저항 변화, 폐 compliance 변화
- **정상 환기 파형**: PEEP(~0-5 cmH2O) → Ramp 상승 → PIP(~15-30 cmH2O) → Plateau(압력 유지) → 호기 하강 → PEEP 복귀. 사다리꼴(trapezoid) 형태가 특징이며, 환기 모드(VCV, PCV)에 따라 형태가 달라진다.
- **임상적으로 중요한 "노이즈" 패턴** (필터링 금지):
  - **Bucking/Fighting**: 환자가 인공호흡기와 싸움 → 날카로운 스파이크 (5-20Hz 성분 포함). **마취 깊이 부족** 또는 통증 반응의 핵심 지표. spike_detection=False로 보존
  - **Cardiogenic oscillation**: 심장 박동이 기도압에 전달 → 1-2Hz 미세 진동. ECG/ABP와 cross-modal 학습 가능
  - **Compliance drop**: 복강경 가스 주입, 체위 변경 시 폐 compliance 변화 → 파형 전체 형태 변화. 수술 진행 상황의 간접 지표
  - **Auto-PEEP**: 불완전 호기 시 잔류 양압 → 기저선 상승. COPD/천식 환자에서 중요
  - **Water condensation**: 호흡 회로 물방울 → 자잘한 진동. 유일한 비임상 노이즈이지만, 다른 임상 패턴과 주파수 대역이 겹쳐 필터로 분리 불가
- **lowpass 20Hz 근거 (기존 5Hz에서 변경)**: 정상 호흡 주파수는 0.1-0.5Hz이지만, bucking spike의 sharp edge가 5-20Hz 성분을 포함한다. 5Hz cutoff는 이 성분을 제거하여 모델이 bucking을 학습할 수 없게 만든다. 20Hz로 변경하여 임상 이벤트를 보존하면서 ADC quantization noise만 제거한다. 198개 VitalDB case 검증 결과 accept/reject 변화 없음.
- **valid_range -20 근거 (기존 -10에서 변경)**: 강한 bucking(환자의 흡기 노력)이 -15~-20 cmH2O까지 도달할 수 있다. -10 cutoff는 이런 임상적으로 유의미한 음압 이벤트를 제거한다.

---

## 4. 품질 검사 (Quality Check) 체계

### 4.1 기본 품질 검사 (`segment_quality_score`)

모든 신호에 공통 적용되는 윈도우 단위(5s 또는 15s) 검사:

| 지표 | 계산 방법 | 의미 |
|------|----------|------|
| **flatline_ratio** | `sum(|diff| < 1e-4) / len` | 연속 동일 값 비율 → 센서 탈락/기계 꺼짐 |
| **clip_ratio** | min/max 고정 비율 | ADC 포화(clipping) |
| **high_freq_ratio** | `mean(diff²) / mean(signal²)` | 1차 미분 에너지 / 신호 에너지 (시간 도메인, FFT 아님) |
| **amplitude** | peak-to-peak | 신호 유효 진폭 |

### 4.2 Domain-specific 품질 검사

| 신호 | 검사 함수 | 핵심 로직 |
|------|----------|----------|
| ECG | `ecg_quality_check()` | R-peak → HR 30-200bpm, regularity(CV<0.7), autocorrelation |
| ABP | `abp_quality_check()` | Systolic peak → HR 30-200bpm, regularity(CV<0.5), autocorrelation |
| PPG | `ppg_quality_check()` | Pulse peak → HR 30-200bpm, regularity(CV<0.5), autocorrelation |
| CO2 | `co2_quality_check()` | Ventilation peak → RR 4-40 breaths/min, IQR>0.5 |
| AWP | `awp_quality_check()` | Ventilation peak → RR 4-40 breaths/min, IQR>0.5 |
| EEG | `eeg_quality_check()` | FFT 기반 1-30Hz band_ratio ≥ 0.1, spectral entropy 0.3-0.95 |
| CVP | `cvp_quality_check()` | a/c/v wave peak → HR 30-200bpm, regularity(CV<0.7), autocorrelation, flatline, respiratory variation |

---

## 5. 전처리 파이프라인 요약 테이블

| 신호 | 그룹 | Range | Filter | Notch | Spike | Median | Step | Motion | QC Window | Domain QC |
|------|------|-------|--------|-------|-------|--------|------|--------|-----------|-----------|
| ECG | 전기 | -5~5 mV | BP 0.5-40Hz | 60Hz | 10σ | - | - | - | 5s | HR/Reg/AC |
| EEG | 전기 | -500~500 μV | BP 0.5-45Hz | 60Hz | 10σ | - | - | - | 5s | Band/Entropy |
| ABP | 액체 | 20~300 mmHg | LP 15Hz | - | 6σ | 5 | Yes | - | 5s | HR/Reg/AC |
| PPG | 광학 | 0~2000 | LP 8Hz | 60Hz | 6σ | 5 | Yes | Yes | 5s | HR/Reg/AC |
| CVP | 액체 | -5~40 mmHg | LP 10Hz | - | 8σ | - | - | - | 5s | HR/Reg/AC/Resp |
| CO2 | 기체 | 0~100 mmHg | LP 5Hz | - | - | - | - | - | 15s | RR/IQR |
| AWP | 기체 | -20~80 hPa | LP 20Hz | - | - | - | - | - | 15s | RR/IQR |

**범례**: BP=bandpass, LP=lowpass, σ=MAD 배수, HR=심박수, Reg=regularity(CV), AC=autocorrelation, RR=호흡수

---

## 6. 설계 원칙

1. **매질 기반 차등 QC**: 전기/액체/광학 신호는 측정 시스템 오류(artifact)를 적극 제거하고, 기체 신호는 최소 필터링으로 임상 이벤트를 보존한다.
2. **DC 보존**: ABP, CVP, PPG, CO2, AWP는 lowpass filter를 사용하여 절대값(DC 성분)을 보존한다. 절대 혈압, 절대 기도압 등이 임상적으로 중요하기 때문이다. ECG/EEG는 baseline wander 제거를 위해 bandpass를 사용한다.
3. **공통 리샘플링**: 모든 신호를 100Hz로 통일 리샘플링하여 패치 크기, 윈도우 크기를 일관되게 관리한다.
4. **연속성 보장**: NaN으로 마킹된 구간 전후를 분리하여 불연속 시계열 concat을 방지한다 (`_extract_nan_free_segments`).
5. **Foundation model 우선**: 전통적 신호처리에서는 노이즈로 취급되는 패턴(bucking, cardiogenic oscillation 등)도 모델이 학습해야 할 정보로 간주한다. 필터는 **물리적으로 불가능한 값**과 **측정 시스템 오류**만 제거하고, 나머지는 트랜스포머가 self-supervised learning으로 표현을 학습하도록 맡긴다.
6. **실데이터 검증 기반 threshold 설정**: QC threshold는 임상 문헌 + VitalDB 실측 데이터 분포(200 cases, 44만+ 윈도우)를 교차 검증하여 결정한다. basic QC에서 reject된 윈도우가 domain check에서 pass하면 false reject(좋은 데이터 손실)으로 간주하여 threshold를 완화한다.

---

## 7. 실데이터 검증 결과 (200 VitalDB Cases)

200개 VitalDB case에서 전체 파이프라인의 pass rate를 검증한 결과:

| 신호 | Cases | Windows | Basic Pass | Full Pass | Range Out | 주요 Fail 원인 |
|------|-------|---------|-----------|-----------|-----------|--------------|
| ECG II | 199 | 474K | 92.2% | 89.9% | 0.1% | AmpLo(0.3mV), HF — 정당한 reject |
| EEG | 165+ | 448K | 88.8% | 88.6% | 0.3% | AmpLo/Hi → threshold 완화 후 개선 |
| ABP | 120 | 300K | 98.7% | 97.6% | 10.8% | Flush artifact — 정상적 제거 |
| PPG | 190 | 427K | 95.4% | 94.1% | 2.9% | AmpLo — 정당한 reject (저관류) |
| CVP | 63 | 160K | 99.8% | 95.1% | 22.8% | Range out — 추후 검토 가능 |
| CO2 | 199 | 1.3M | 84.9% | 60-95%* | 0.0% | 정상 (domain check가 기계 분리 구간 제거) |
| AWP | 198 | 1.3M | 73.6% | 63-99%* | 0.0% | 정상 (domain check가 기계 분리 구간 제거) |

*CO2/AWP의 Full Pass는 실제 파서 기준. 파서는 NaN-free 세그먼트 추출 후 QC를 적용하므로 기계 꺼짐/분리 구간이 미리 제거되어 높은 pass rate를 달성한다.

**CO2/AWP 참고**: 실제 파서에서는 NaN-free 세그먼트 추출 후 윈도우 QC를 적용하므로 기계 꺼짐/분리 구간이 미리 제거되어 60-99% pass rate를 달성한다. AWP 파형은 사다리꼴(plateau) 형태로, 정상 환기 구간에서는 peak detection이 정상 동작하며, FAIL 구간은 기계 설정 변경/분리 등 실제 불량 데이터이다.
