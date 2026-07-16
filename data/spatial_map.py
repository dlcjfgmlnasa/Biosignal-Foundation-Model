"""Signal type 매핑 테이블 (단일 modality embedding 체계, v2).

CARMEN v2: spatial_id(소분류)를 폐지하고 modality(signal_type) 단위 단일
임베딩만 사용한다. 각 signal_type 내 로컬 ID는 항상 0(Unknown) 하나뿐이며,
전역 spatial ID 총 수(``TOTAL_SPATIAL_IDS``)는 signal_type 수와 같다.

지원 신호 (9종, 2026-06-23 갱신 — PAP 제거 후 연속 번호로 재배치):
  ECG(0), ABP(1), PPG(2), CVP(3), CO2(4), AWP(5),
  ICP(6), RESP_Impedance(7), RESP_Flow(8)

이전 체계(10종, PAP 슬롯 유지) 대비 변경:
  - PAP(구 6) 완전 제거 — 데이터가 없어 박제돼 있던 dead 슬롯을 폐지.
  - 뒤 번호를 1칸씩 당김: ICP 7→6, RESP_Impedance 8→7, RESP_Flow 9→8.
  - num_signal_types 10 → 9 (재학습 전제이므로 번호 재배치 허용).

이전 spec(9종, ECG~RESP) 대비 변경(유지):
  - RESP(구 8) → RESP_Impedance(7) / RESP_Flow(8) 로 분리 (호흡 물리량 구분)
  - ECG 12-lead spatial → 단일 ECG(0) 으로 통합
  - ABP Radial/Femoral spatial → 단일 ABP(1) 으로 통합
  - spatial_id 소분류 전면 폐지 → 모든 type 이 local 0(Unknown) 단일

데이터 소스:
  VitalDB Open: SNUADC/ (OR, 500Hz) — ECG, ABP, PPG, CVP, CO2, AWP
  K-MIMIC-MORTAL: SNUADCM/, Solar8000/, Intellivue/ (ICU/OR mixed, 500Hz)

디스크 manifest 는 구 disk spec(signal_type 0~8: ECG~RESP, PAP=6, ICP=7,
RESP=8 + spatial_ids) 그대로 보존한다. load-time 에 ``remap_record_v2`` 로
새 9종 연속 번호 체계로 변환한다 (데이터 재생성 없음).
"""

from __future__ import annotations

# signal_type → {로컬 이름: 로컬 ID}
#
# v2: 모든 signal_type 이 단일 local 0(Unknown). spatial_id 폐지.
# 구 spec 의 spatial 소분류(ECG 12-lead, ABP Radial/Femoral, RESP Imp/Flow)는
# 아래 주석으로 메타데이터만 보존한다 (참고용, 코드에서 사용 안 함).
#
# 구 ECG(0) 12-lead: Lead_I~III, aVR/aVL/aVF, V1~V6 → 모두 단일 ECG(0)
# 구 ABP(1): Radial, Femoral → 모두 단일 ABP(1)
# 구 RESP(disk 8): Impedance(1)/Flow(2) → RESP_Impedance(7)/RESP_Flow(8) 로 승격
SPATIAL_MAP: dict[int, dict[str, int]] = {
    0: {"Unknown": 0},  # ECG
    1: {"Unknown": 0},  # ABP
    2: {"Unknown": 0},  # PPG
    3: {"Unknown": 0},  # CVP
    4: {"Unknown": 0},  # CO2 / Capnography
    5: {"Unknown": 0},  # AWP / Airway Pressure
    6: {"Unknown": 0},  # ICP / Intracranial Pressure
    7: {"Unknown": 0},  # RESP_Impedance (가슴 임피던스 호흡)
    8: {"Unknown": 0},  # RESP_Flow (ventilator flow waveform)
}

# signal_type별 offset (누적 합). v2 에서는 각 type 이 local 1개라
# offset == signal_type 이 되지만, 일반성을 위해 누적 합 로직 유지.
_OFFSETS: dict[int, int] = {}
_cumulative = 0
for _st in sorted(SPATIAL_MAP.keys()):
    _OFFSETS[_st] = _cumulative
    _cumulative += len(SPATIAL_MAP[_st])

TOTAL_SPATIAL_IDS: int = _cumulative
"""전역 spatial_id 총 수. v2 에서는 signal_type 수(=9)와 동일."""


# signal_type 번호 → 사람이 읽는 이름 (validator/count 스크립트 공용)
SIGNAL_TYPE_NAMES: dict[int, str] = {
    0: "ECG",
    1: "ABP",
    2: "PPG",
    3: "CVP",
    4: "CO2",
    5: "AWP",
    6: "ICP",
    7: "RESP_Impedance",
    8: "RESP_Flow",
}


# 소문자 task-facing key → signal_type 번호 (downstream/eval 공용 SSOT, v2).
# master_plan §7.1 번호 체계의 유일 소스. downstream 각 스크립트는 자체 dict 를
# 두지 말고 이 표(또는 ``SIGNAL_TYPE_TO_KEY``)를 import 해 사용한다.
#   - 구 단일 "resp"(disk 8) 는 v2 에서 "resp_impedance"(7) / "resp_flow"(8) 로 분리.
#   - "pap" 는 v2 에서 완전 제거 (데이터 부재).
SIGNAL_KEY_TO_TYPE: dict[str, int] = {
    "ecg": 0,
    "abp": 1,
    "ppg": 2,
    "cvp": 3,
    "co2": 4,
    "awp": 5,
    "icp": 6,
    "resp_impedance": 7,
    "resp_flow": 8,
}

# signal_type 번호 → 소문자 key (역방향, intersection/forecast 등에서 사용)
SIGNAL_TYPE_TO_KEY: dict[int, str] = {v: k for k, v in SIGNAL_KEY_TO_TYPE.items()}


def get_global_spatial_id(signal_type: int, local_id: int) -> int:
    """(signal_type, local_spatial_id) → 전역 고유 spatial_id 변환.

    v2: spatial_id 폐지로 local_id 는 항상 0 이어야 한다. 따라서 반환값은
    사실상 signal_type 의 offset(=signal_type) 과 같다. 하위 호환을 위해
    시그니처/동작은 유지한다.

    Parameters
    ----------
    signal_type:
        신호 대분류 코드 (0~8).
    local_id:
        signal_type 내 로컬 spatial ID. v2 에서는 항상 0(Unknown).

    Returns
    -------
    전역 고유 spatial_id (int).
    signal_type 이 매핑에 없으면 0(ECG Unknown)을 반환한다.
    """
    offset = _OFFSETS.get(signal_type, 0)
    return offset + local_id


# ── Mechanism Group ────────────────────────────────────────────
# Cross-Modal MSE reconstruction은 같은 mechanism group 내에서만 허용.
# Contrastive (InfoNCE)는 전체 허용 (그룹 무관).
#
# Cardiovascular (0): ECG, ABP, PPG, CVP, ICP — 심혈관계, 심박 주기 동기화
# Respiratory (1): CO2, AWP, RESP_Impedance, RESP_Flow — 호흡계, 환기 동기화

MECHANISM_GROUP: dict[int, int] = {
    0: 0,  # ECG            → Cardiovascular
    1: 0,  # ABP            → Cardiovascular
    2: 0,  # PPG            → Cardiovascular
    3: 0,  # CVP            → Cardiovascular
    4: 1,  # CO2            → Respiratory
    5: 1,  # AWP            → Respiratory
    6: 0,  # ICP            → Cardiovascular
    7: 1,  # RESP_Impedance → Respiratory
    8: 1,  # RESP_Flow      → Respiratory
}

MECHANISM_GROUP_NAMES: dict[int, str] = {
    0: "Cardiovascular",
    1: "Respiratory",
}


# ── Cross-Pred Allowed Pairs ──────────────────────────────────
# Cross-Modal MSE reconstruction에서 허용되는 signal type 쌍.
# 생리학적으로 waveform 복원이 가능한(인과 관계가 있는) 쌍만 포함.
# Contrastive (InfoNCE)에는 적용되지 않음 (전체 쌍 허용).
#
# 선별 원칙:
#   1. 같은 물리 도메인 (amplitude 예측 가능)
#   2. Waveform morphology 인과성 (상관이 아닌 파형 전달 관계)
#   3. 같은 시간 스케일 (cardiac ~1Hz vs respiratory ~0.2Hz 혼합 불가)
#   4. 외부 변수 배제 (인공호흡기 설정 등 기계 제어 신호 제외)
#
# v2 변경 (RESP 분리 반영):
#   - 구 (4,disk8) CO2↔RESP → (4,7) CO2↔RESP_Impedance 로 명시화 (의미 보존).
#     capnography(호기 CO2) ↔ 흉부 임피던스(폐 용적): 같은 호흡 주기, 용적 기반.
#   - 구 (5,disk8) AWP↔RESP → (5,8) AWP↔RESP_Flow 로 교체.
#     AWP 의 직접 짝은 임피던스(용적)가 아니라 flow(유량). 동일 환기 회로의
#     P–Q 관계(R = ΔP/Q)가 가장 직접적 인과이므로 Flow 와 묶는 것이 옳다.
#   - 구 (4,5) CO2↔AWP 제거. AWP 는 ventilator 설정(외부 제어변수)에 의존하여
#     선별 원칙 4(외부 변수 배제)에 위배. RESP_Flow 도입으로 자연 폐기.
#
# 제외 후보 (v2 에서 미포함):
#   (7,8) RESP_Impedance↔RESP_Flow: 같은 호흡 주기지만 물리량(임피던스 vs 유량)·
#     위상 차이가 커 waveform 복원 인과 약함. Contrastive 에서만 다뤄짐.
#   (5,7) AWP↔RESP_Impedance: 약한 상관은 있으나 직접 인과 아님.
#
# 기존(구 spec) 기각 후보 (유지):
#   ABP↔CVP: Frank-Starling은 스칼라 관계, 동맥파 vs 정맥파 morphology 완전 다름
#   ECG↔CVP/ICP: 전기→유체 도메인 단절 (mV→mmHg), morphology 복원 불가
#   PPG↔CO2: cardiac(~1Hz) vs respiratory(~0.2Hz) 시간 스케일 다름

# ── γ (CMPM, directed cross-modal MSE 복원) 대상 쌍 — 강결합만 ──
# 손실 배정 (팀 4 생리학 교차검증 + 문헌, 2026-06-03 확정):
#   "파형이 실제로 전달되는 강결합"만 γ(directed MSE)에 사용한다. 절대압·용적·
#   compliance 가 끼어 형태가 전기신호에 안 실리는 약결합은 δ(contrastive)로만
#   정렬한다(δ 는 ungated 전체 쌍이라 약결합은 자동으로 CMCL 만 받음).
#   약결합에 복원을 걸면 "리듬만 베끼는" shortcut 으로 loss 를 낮추므로 contrastive 가 안전.
#   문헌: cross-modal waveform 변환은 PPG↔ECG(CardioGAN 등)에 집중, ECG↔CVP 등은
#   공백 → 약결합은 합성(CMPM)이 아니라 정렬(CMCL)로 다루는 것이 근거와 일치.
# NOTE: 2026-06-23 PAP 제거로 signal_type 번호 재배치(ICP=6, RESP_Imp=7, RESP_Flow=8).
#   쌍은 의미로 정의되며 새 번호로 갱신했다.
CROSS_PRED_ALLOWED_PAIRS: set[tuple[int, int]] = {
    # Tier 2 — 강·타이밍 (전기-기계 수축 트리거, ECG 허브)
    (0, 1),  # ECG ↔ ABP — R-peak→수축 트리거, 형태도 cardiac 지배
    (0, 2),  # ECG ↔ PPG — cardiac cycle, 말초 맥파
    # Tier 1 — 강·형태 (waveform 직결)
    (1, 2),  # ABP ↔ PPG — arterial pulse wave (거의 동형)
    (5, 8),  # AWP ↔ RESP_Flow — airway pressure ↔ flow, P–Q 운동방정식 직접 인과
}
# ── γ 에서 제외된 쌍 (δ contrastive 로만 cross-modal 정렬) — 약결합 ──
#   (0,3) ECG↔CVP      : a/c/v파가 P/QRS/T에 타이밍 lock 되나, 절대압·진폭은
#       용적상태·우심기능·삼첨판·흉강내압이 결정 → ECG 에 없음 (타이밍 강·형태 약).
#   (1,6) ABP↔ICP      : P1(박동)만 ABP 인과. P2(compliance)·B-wave·plateau 불가.
#   (4,7) CO2↔RESP_Imp : capnography 는 V/Q·관류·대사 산물, 임피던스(용적)에 선행자 없음.
#   (7,8) RESP_Imp↔Flow: volume ≈ ∫flow 적분관계 → 90° 위상지연, patch-wise MSE 부담.
# ablation 후보: (0,3)·(7,8) 의 γ on/off — 약결합 directed 효과 사후 검증.
# 결과: γ 참여 = ECG/ABP/PPG/AWP/RESP_Flow(5종), δ전용 = CVP/CO2/ICP/RESP_Imp(4종).
#   모든 신호는 MPM + same-variate next + δ 수신.


# ── γ (CMPM) cross-modal 예측 방향 (directed) + 가중 ─────────────────────────
# W[(source, target)] = source 로 target 을 생성 예측할 때의 CMPM loss 가중치.
# patch(2s) 안에서 보이는 빠른·같은대역·shape 결합만 CMPM 재구성 대상이며, 방향은
# 생리적 복원가능성으로 확정한다 (범위·변조·희소 쌍은 δ contrastive 로만 정렬):
#   ECG → ABP/PPG   : 전기→기계 트리거 (역방향 QRS 복원불가 → 단방향)
#   ABP ↔ PPG       : 같은 동맥맥파 (양방향; ppg→abp = cuffless BP)
#   AWP ↔ RESP_Flow : P–V̇ 운동방정식 (양방향)
# 균일 1.0. 경험적 best-lag corr 실측(2026-07-16)은 쌍별 null-floor 편향(호흡쌍
# 바닥 ~0.5)·소스편차(SNUH↔K-MIMIC 3×)로 정밀 가중에 부적합 → 폐기(팀 4-agent
# 검증: 구조는 robust, 정밀 가중은 측정불가). 방향만 생리로 확정.
CROSS_PRED_DIRECTED: set[tuple[int, int]] = {
    (0, 1),          # ECG → ABP
    (0, 2),          # ECG → PPG
    (1, 2), (2, 1),  # ABP ↔ PPG
    (5, 8), (8, 5),  # AWP ↔ RESP_Flow
}
CROSS_COUPLING_WEIGHTS: dict[tuple[int, int], float] = {
    (_s, _t): 1.0 for (_s, _t) in CROSS_PRED_DIRECTED
}


# 채널명 → signal_type (v2: spatial 폐지, 단일 int 반환)
#
# 구 ``CHANNEL_NAME_TO_SPATIAL`` (튜플 반환) 을 대체한다. 모든 ECG lead 표기는
# 0 으로, ABP 변형(Radial/Femoral/ART/FEM)은 1 로 수렴한다.
# RESP/Impedance → 7(RESP_Impedance), FLOW/FLOW_WAV → 8(RESP_Flow).
CHANNEL_NAME_TO_SIGNAL_TYPE: dict[str, int] = {
    # ECG (0) — 모든 lead 표기 단일 수렴
    "ECG Lead I": 0, "ECG I": 0, "I": 0,
    "ECG Lead II": 0, "ECG II": 0, "II": 0,
    "ECG Lead III": 0, "ECG III": 0, "III": 0,
    "ECG aVR": 0, "aVR": 0,
    "ECG aVL": 0, "aVL": 0,
    "ECG aVF": 0, "aVF": 0,
    "ECG V1": 0, "V1": 0,
    "ECG V2": 0, "V2": 0,
    "ECG V3": 0, "V3": 0,
    "ECG V4": 0, "V4": 0,
    "ECG V5": 0, "V5": 0, "ECG Lead V5": 0,
    "ECG V6": 0, "V6": 0,
    # ABP (1) — Radial/Femoral 통합
    "ABP Radial": 1, "ART": 1,
    "ABP Femoral": 1, "FEM": 1,
    # PPG (2)
    "PPG": 2, "PLETH": 2, "PPG Finger": 2,
    # CVP (3)
    "CVP": 3,
    # CO2 (4)
    "CO2": 4,
    # AWP (5)
    "AWP": 5,
    # ICP (6)
    "ICP": 6,
    # RESP_Impedance (7)
    "RESP": 7, "Impedance": 7,
    # RESP_Flow (8)
    "FLOW": 8, "FLOW_WAV": 8,
}


# ── Load-time remap (구 disk spec → v2 9종 연속) ────────────────
# 디스크 manifest 는 구 disk spec(signal_type 0~8: ECG~RESP, 단 PAP=6·ICP=7·
# RESP=8) 을 그대로 보존한다. manifest 파싱 시점에 이 함수로 v2 새 번호 체계로
# 변환한다 (데이터 재생성 없음).
#
# 구 disk spec spatial 의미:
#   구 disk 8(RESP): spatial 1=Impedance, 2=Flow, 0=Unknown(폴백 Impedance)
#
# 변환 규칙 (구 disk → v2):
#   0~5            ECG~AWP    → 그대로 0~5
#   6              PAP        → None (drop, 데이터 제외 → 완전 폐기)
#   7              ICP        → 6              (1칸 당김)
#   8, sp=1        RESP Imped → (7, [0]*n)     RESP_Impedance
#   8, sp=2        RESP Flow  → (8, [0]*n)     RESP_Flow
#   8, sp=0/없음   RESP ?     → (7, [0]*n)     Impedance 폴백 (보수적)

# 구 disk signal_type → v2 signal_type (PAP/RESP 제외 본래 보존군의 재배치 맵)
_OLD_DISK_TO_V2_ST: dict[int, int] = {
    0: 0,  # ECG
    1: 1,  # ABP
    2: 2,  # PPG
    3: 3,  # CVP
    4: 4,  # CO2
    5: 5,  # AWP
    7: 6,  # ICP (구 disk 7 → v2 6)
}

# 구 disk RESP local spatial → v2 signal_type (7=Impedance, 8=Flow)
_OLD_RESP_SPATIAL_TO_ST: dict[int, int] = {1: 7, 2: 8}

# 구 disk PAP signal_type (drop 대상)
_PAP_DISK_SIGNAL_TYPE = 6
# 구 disk RESP signal_type (분리 대상)
_OLD_RESP_DISK_SIGNAL_TYPE = 8


def remap_record_v2(
    old_signal_type: int,
    old_spatial_ids: list[int] | None,
) -> tuple[int, list[int]] | None:
    """구 disk spec record 를 v2 (signal_type, spatial_ids) 로 변환한다.

    데이터(.pt/shard)는 건드리지 않고 manifest 파싱 시점에 적용한다.
    spatial_id 는 폐지되므로 항상 ``[0] * n_channels`` 로 평탄화한다.

    Parameters
    ----------
    old_signal_type:
        구 disk spec signal_type (0~8: ECG~RESP, PAP=6·ICP=7·RESP=8).
    old_spatial_ids:
        구 disk spec per-channel 로컬 spatial_id 리스트. None/빈 리스트면
        단일 채널(n=1)로 간주한다. 구 disk 8(RESP)의 분기 판단에 사용한다.

    Returns
    -------
    ``(new_signal_type, [0] * n_channels)`` 튜플.
    PAP(구 disk 6)이면 ``None`` 을 반환한다 (drop — 데이터 제외, 완전 폐기).
    """
    # n_channels: spatial_ids 길이 기준 (없으면 1)
    n_channels = len(old_spatial_ids) if old_spatial_ids else 1

    # PAP drop (완전 폐기)
    if old_signal_type == _PAP_DISK_SIGNAL_TYPE:
        return None

    # RESP 분기 (Impedance=7 / Flow=8)
    if old_signal_type == _OLD_RESP_DISK_SIGNAL_TYPE:
        # 구 disk RESP record 는 단일 채널(n_channels:1)이므로 첫 spatial 로 판단.
        first_sp = old_spatial_ids[0] if old_spatial_ids else 0
        new_st = _OLD_RESP_SPATIAL_TO_ST.get(first_sp, 7)  # 0/미상 → Impedance 폴백
        return new_st, [0] * n_channels

    # 그 외: 구 disk → v2 재배치 (ICP 7→6 포함) + spatial 평탄화
    new_st = _OLD_DISK_TO_V2_ST.get(old_signal_type, old_signal_type)
    return new_st, [0] * n_channels
