"""Signal type 매핑 테이블 (단일 modality embedding 체계, v2).

CARMEN v2: spatial_id(소분류)를 폐지하고 modality(signal_type) 단위 단일
임베딩만 사용한다. 각 signal_type 내 로컬 ID는 항상 0(Unknown) 하나뿐이며,
전역 spatial ID 총 수(``TOTAL_SPATIAL_IDS``)는 signal_type 수와 같다.

지원 신호 (10종, 2026-05-20 갱신 — Option A, single modality embedding):
  ECG(0), ABP(1), PPG(2), CVP(3), CO2(4), AWP(5),
  PAP(6, 슬롯 유지·데이터 제외), ICP(7),
  RESP_Impedance(8), RESP_Flow(9)

이전 spec(9종, ECG~RESP) 대비 변경:
  - RESP(구 8) → RESP_Impedance(8) / RESP_Flow(9) 로 분리 (호흡 물리량 구분)
  - ECG 12-lead spatial → 단일 ECG(0) 으로 통합
  - ABP Radial/Femoral spatial → 단일 ABP(1) 으로 통합
  - PAP(6) 슬롯은 유지하되 데이터는 파이프라인에서 제외 (remap drop)
  - spatial_id 소분류 전면 폐지 → 모든 type 이 local 0(Unknown) 단일

데이터 소스:
  VitalDB Open: SNUADC/ (OR, 500Hz) — ECG, ABP, PPG, CVP, CO2, AWP
  K-MIMIC-MORTAL: SNUADCM/, Solar8000/, Intellivue/ (ICU/OR mixed, 500Hz)

디스크 manifest 는 구 spec(signal_type 0~8 + spatial_ids) 그대로 보존한다.
load-time 에 ``remap_record_v2`` 로 새 번호 체계로 변환한다 (데이터 재생성 없음).
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
# 구 RESP(8): Impedance(1)/Flow(2) → RESP_Impedance(8)/RESP_Flow(9) 로 승격
SPATIAL_MAP: dict[int, dict[str, int]] = {
    0: {"Unknown": 0},  # ECG
    1: {"Unknown": 0},  # ABP
    2: {"Unknown": 0},  # PPG
    3: {"Unknown": 0},  # CVP
    4: {"Unknown": 0},  # CO2 / Capnography
    5: {"Unknown": 0},  # AWP / Airway Pressure
    6: {"Unknown": 0},  # PAP / Pulmonary Arterial Pressure (슬롯 유지·데이터 제외)
    7: {"Unknown": 0},  # ICP / Intracranial Pressure
    8: {"Unknown": 0},  # RESP_Impedance (가슴 임피던스 호흡)
    9: {"Unknown": 0},  # RESP_Flow (ventilator flow waveform)
}

# signal_type별 offset (누적 합). v2 에서는 각 type 이 local 1개라
# offset == signal_type 이 되지만, 일반성을 위해 누적 합 로직 유지.
_OFFSETS: dict[int, int] = {}
_cumulative = 0
for _st in sorted(SPATIAL_MAP.keys()):
    _OFFSETS[_st] = _cumulative
    _cumulative += len(SPATIAL_MAP[_st])

TOTAL_SPATIAL_IDS: int = _cumulative
"""전역 spatial_id 총 수. v2 에서는 signal_type 수(=10)와 동일."""


# signal_type 번호 → 사람이 읽는 이름 (validator/count 스크립트 공용)
SIGNAL_TYPE_NAMES: dict[int, str] = {
    0: "ECG",
    1: "ABP",
    2: "PPG",
    3: "CVP",
    4: "CO2",
    5: "AWP",
    6: "PAP",
    7: "ICP",
    8: "RESP_Impedance",
    9: "RESP_Flow",
}


# 소문자 task-facing key → signal_type 번호 (downstream/eval 공용 SSOT, v2).
# master_plan §7.1 번호 체계의 유일 소스. downstream 각 스크립트는 자체 dict 를
# 두지 말고 이 표(또는 ``SIGNAL_TYPE_TO_KEY``)를 import 해 사용한다.
#   - 구 단일 "resp"(8) 는 v2 에서 "resp_impedance"(8) / "resp_flow"(9) 로 분리.
#   - "pap"(6) 은 pretrain 데이터 제외이나 슬롯/번호는 유지(renumber 회피).
SIGNAL_KEY_TO_TYPE: dict[str, int] = {
    "ecg": 0,
    "abp": 1,
    "ppg": 2,
    "cvp": 3,
    "co2": 4,
    "awp": 5,
    "pap": 6,
    "icp": 7,
    "resp_impedance": 8,
    "resp_flow": 9,
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
        신호 대분류 코드 (0~9).
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
# Cardiovascular (0): ECG, ABP, PPG, CVP, PAP, ICP — 심혈관계, 심박 주기 동기화
# Respiratory (1): CO2, AWP, RESP_Impedance, RESP_Flow — 호흡계, 환기 동기화

MECHANISM_GROUP: dict[int, int] = {
    0: 0,  # ECG            → Cardiovascular
    1: 0,  # ABP            → Cardiovascular
    2: 0,  # PPG            → Cardiovascular
    3: 0,  # CVP            → Cardiovascular
    4: 1,  # CO2            → Respiratory
    5: 1,  # AWP            → Respiratory
    6: 0,  # PAP            → Cardiovascular (슬롯 유지·데이터 제외)
    7: 0,  # ICP            → Cardiovascular
    8: 1,  # RESP_Impedance → Respiratory
    9: 1,  # RESP_Flow      → Respiratory
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
#   - 구 (4,8) CO2↔RESP → (4,8) CO2↔RESP_Impedance 로 명시화 (의미 보존).
#     capnography(호기 CO2) ↔ 흉부 임피던스(폐 용적): 같은 호흡 주기, 용적 기반.
#   - 구 (5,8) AWP↔RESP → (5,9) AWP↔RESP_Flow 로 교체.
#     AWP 의 직접 짝은 임피던스(용적)가 아니라 flow(유량). 동일 환기 회로의
#     P–Q 관계(R = ΔP/Q)가 가장 직접적 인과이므로 Flow 와 묶는 것이 옳다.
#   - 구 (4,5) CO2↔AWP 제거. AWP 는 ventilator 설정(외부 제어변수)에 의존하여
#     선별 원칙 4(외부 변수 배제)에 위배. RESP_Flow 도입으로 자연 폐기.
#
# 제외 후보 (v2 에서 미포함):
#   (8,9) RESP_Impedance↔RESP_Flow: 같은 호흡 주기지만 물리량(임피던스 vs 유량)·
#     위상 차이가 커 waveform 복원 인과 약함. Contrastive 에서만 다뤄짐.
#   (5,8) AWP↔RESP_Impedance: 약한 상관은 있으나 직접 인과 아님.
#
# 기존(구 spec) 기각 후보 (유지):
#   ABP↔CVP: Frank-Starling은 스칼라 관계, 동맥파 vs 정맥파 morphology 완전 다름
#   ECG↔CVP/PAP/ICP: 전기→유체 도메인 단절 (mV→mmHg), morphology 복원 불가
#   PPG↔CO2: cardiac(~1Hz) vs respiratory(~0.2Hz) 시간 스케일 다름

CROSS_PRED_ALLOWED_PAIRS: set[tuple[int, int]] = {
    # Arterial-Cardiac (심박 주기 → 압력파 직접 인과)
    (0, 1),  # ECG ↔ ABP — cardiac cycle, pulse transit time
    (0, 2),  # ECG ↔ PPG — cardiac cycle, peripheral pulse
    (1, 2),  # ABP ↔ PPG — arterial pulse wave (거의 동형)
    # Systemic-Pulmonary (체순환 ↔ 폐순환)
    (1, 6),  # ABP ↔ PAP — 유사 동맥 morphology, 다른 amplitude scale (슬롯 유지)
    # Right Heart (우심계 혈역학)
    (3, 6),  # CVP ↔ PAP — 우심 전후부하, 같은 pressure 도메인 (슬롯 유지)
    # Cerebral Perfusion (뇌관류)
    (1, 7),  # ABP ↔ ICP — CPP = MAP - ICP, 뇌자동조절
    # Respiratory cycle (호흡 주기)
    (4, 8),  # CO2 ↔ RESP_Impedance — capnography ↔ 흉부 임피던스, 같은 호흡 cycle
    (5, 9),  # AWP ↔ RESP_Flow — airway pressure ↔ ventilator flow, P–Q 인과
}


# 채널명 → signal_type (v2: spatial 폐지, 단일 int 반환)
#
# 구 ``CHANNEL_NAME_TO_SPATIAL`` (튜플 반환) 을 대체한다. 모든 ECG lead 표기는
# 0 으로, ABP 변형(Radial/Femoral/ART/FEM)은 1 로 수렴한다.
# RESP/Impedance → 8(RESP_Impedance), FLOW/FLOW_WAV → 9(RESP_Flow).
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
    # PAP (6) — 슬롯 유지 (remap 단계에서 drop)
    "PAP": 6,
    # RESP_Impedance (8)
    "RESP": 8, "Impedance": 8,
    # RESP_Flow (9)
    "FLOW": 9, "FLOW_WAV": 9,
}


# ── Load-time remap (구 spec → v2) ─────────────────────────────
# 디스크 manifest 는 구 9종 spec(signal_type 0~8 + spatial_ids) 그대로 보존.
# manifest 파싱 시점에 이 함수로 v2 번호 체계로 변환한다 (데이터 재생성 없음).
#
# 구 spec spatial 의미:
#   구 8(RESP): spatial 1=Impedance, 2=Flow, 0=Unknown(폴백 Impedance)
#
# 변환 규칙:
#   (6, *)         PAP        → None (drop, 데이터 제외)
#   (8, sp=1)      RESP Imped → (8, [0]*n)   RESP_Impedance
#   (8, sp=2)      RESP Flow  → (9, [0]*n)   RESP_Flow
#   (8, sp=0/없음) RESP ?     → (8, [0]*n)   Impedance 폴백 (보수적)
#   (st, *)        그 외       → (st, [0]*n)  번호 보존, spatial 평탄화

# 구 RESP local spatial → 새 signal_type (8=Impedance, 9=Flow)
_OLD_RESP_SPATIAL_TO_ST: dict[int, int] = {1: 8, 2: 9}

# PAP signal_type (drop 대상)
_PAP_SIGNAL_TYPE = 6
# 구 RESP signal_type (분리 대상)
_OLD_RESP_SIGNAL_TYPE = 8


def remap_record_v2(
    old_signal_type: int,
    old_spatial_ids: list[int] | None,
) -> tuple[int, list[int]] | None:
    """구 spec record 를 v2 (signal_type, spatial_ids) 로 변환한다.

    데이터(.pt/shard)는 건드리지 않고 manifest 파싱 시점에 적용한다.
    spatial_id 는 폐지되므로 항상 ``[0] * n_channels`` 로 평탄화한다.

    Parameters
    ----------
    old_signal_type:
        구 spec signal_type (0~8). ECG~RESP 9종.
    old_spatial_ids:
        구 spec per-channel 로컬 spatial_id 리스트. None/빈 리스트면
        단일 채널(n=1)로 간주한다. 구 8(RESP)의 분기 판단에 사용한다.

    Returns
    -------
    ``(new_signal_type, [0] * n_channels)`` 튜플.
    PAP(구 6)이면 ``None`` 을 반환한다 (drop — 데이터 제외, 슬롯만 유지).
    """
    # n_channels: spatial_ids 길이 기준 (없으면 1)
    n_channels = len(old_spatial_ids) if old_spatial_ids else 1

    # PAP drop
    if old_signal_type == _PAP_SIGNAL_TYPE:
        return None

    # RESP 분기 (Impedance=8 / Flow=9)
    if old_signal_type == _OLD_RESP_SIGNAL_TYPE:
        # 구 RESP record 는 단일 채널(n_channels:1)이므로 첫 spatial 로 판단.
        first_sp = old_spatial_ids[0] if old_spatial_ids else 0
        new_st = _OLD_RESP_SPATIAL_TO_ST.get(first_sp, 8)  # 0/미상 → Impedance 폴백
        return new_st, [0] * n_channels

    # 그 외: 번호 보존 + spatial 평탄화
    return old_signal_type, [0] * n_channels
