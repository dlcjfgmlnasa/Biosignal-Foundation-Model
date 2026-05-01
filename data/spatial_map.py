"""Spatial ID 매핑 테이블.

signal_type(대분류) + spatial_id(소분류) 이중 인코딩 체계.
각 signal_type 내에서 로컬 ID를 정의하고,
전역 고유 ID(global spatial ID)로 변환하는 유틸리티를 제공한다.

로컬 ID 0은 항상 Unknown(위치 정보 없음)이다.

지원 신호 (9종, 2026-05-01 갱신):
  ECG(0), ABP(1), PPG(2), CVP(3), CO2(4), AWP(5), PAP(6), ICP(7), RESP(8)

데이터 소스:
  VitalDB Open: SNUADC/ (OR, 500Hz) — ECG, ABP, PPG, CVP, CO2, AWP
  K-MIMIC-MORTAL: SNUADCM/, Solar8000/, Intellivue/ (ICU/OR mixed, 500Hz)
"""

from __future__ import annotations

# signal_type → {로컬 이름: 로컬 ID}
SPATIAL_MAP: dict[int, dict[str, int]] = {
    # ECG (signal_type=0) — 12-lead 표준 + Unknown
    0: {
        "Unknown": 0,
        # Limb leads
        "Lead_I":   1,
        "Lead_II":  2,
        "Lead_III": 3,
        "aVR":      4,
        "aVL":      5,
        "aVF":      6,
        # Precordial leads (chest)
        "V1": 7,
        "V2": 8,
        "V3": 9,
        "V4": 10,
        "V5": 11,
        "V6": 12,
    },
    # ABP (signal_type=1)
    1: {
        "Unknown": 0,
        "Radial": 1,
        "Femoral": 2,
    },
    # PPG (signal_type=2)
    2: {
        "Unknown": 0,
        "Finger": 1,
    },
    # CVP (signal_type=3)
    3: {
        "Unknown": 0,
    },
    # CO2/Capnography (signal_type=4)
    4: {
        "Unknown": 0,
    },
    # AWP/Airway Pressure (signal_type=5)
    5: {
        "Unknown": 0,
    },
    # PAP/Pulmonary Arterial Pressure (signal_type=6)
    6: {
        "Unknown": 0,
    },
    # ICP/Intracranial Pressure (signal_type=7)
    7: {
        "Unknown": 0,
    },
    # RESP/Respiration (signal_type=8) — 호흡 wave 통합
    # Impedance: 가슴 임피던스 호흡 (ECG 전극 부수 측정)
    # Flow: ventilator flow waveform (intubation 환자만)
    8: {
        "Unknown": 0,
        "Impedance": 1,
        "Flow": 2,
    },
}

# signal_type별 offset (누적 합)
_OFFSETS: dict[int, int] = {}
_cumulative = 0
for _st in sorted(SPATIAL_MAP.keys()):
    _OFFSETS[_st] = _cumulative
    _cumulative += len(SPATIAL_MAP[_st])

TOTAL_SPATIAL_IDS: int = _cumulative
"""전역 spatial_id 총 수."""


def get_global_spatial_id(signal_type: int, local_id: int) -> int:
    """(signal_type, local_spatial_id) → 전역 고유 spatial_id 변환.

    Parameters
    ----------
    signal_type:
        신호 대분류 코드 (0~6).
    local_id:
        signal_type 내 로컬 spatial ID. 0 = Unknown.

    Returns
    -------
    전역 고유 spatial_id (int).
    signal_type이 매핑에 없으면 0(ECG Unknown)을 반환한다.
    """
    offset = _OFFSETS.get(signal_type, 0)
    return offset + local_id


# ── Mechanism Group ────────────────────────────────────────────
# Cross-Modal MSE reconstruction은 같은 mechanism group 내에서만 허용.
# Contrastive (InfoNCE)는 전체 허용 (그룹 무관).
#
# Cardiovascular (0): ECG, ABP, PPG, CVP, PAP, ICP — 심혈관계, 심박 주기 동기화
# Respiratory (1): CO2, AWP — 호흡계, 환기 동기화

MECHANISM_GROUP: dict[int, int] = {
    0: 0,  # ECG → Cardiovascular
    1: 0,  # ABP → Cardiovascular
    2: 0,  # PPG → Cardiovascular
    3: 0,  # CVP → Cardiovascular
    4: 1,  # CO2 → Respiratory
    5: 1,  # AWP → Respiratory
    6: 0,  # PAP → Cardiovascular
    7: 0,  # ICP → Cardiovascular
    8: 1,  # RESP → Respiratory
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
# 기각된 후보:
#   CO2↔AWP: AWP는 인공호흡기 설정(외부 변수)에 의존, 복원 불가
#   ABP↔CVP: Frank-Starling은 스칼라 관계, 동맥파 vs 정맥파 morphology 완전 다름
#   ECG↔CVP/PAP/ICP: 전기→유체 도메인 단절 (mV→mmHg), morphology 복원 불가
#   PPG↔CO2: cardiac(~1Hz) vs respiratory(~0.2Hz) 시간 스케일 다름

CROSS_PRED_ALLOWED_PAIRS: set[tuple[int, int]] = {
    # Arterial-Cardiac (심박 주기 → 압력파 직접 인과)
    (0, 1),  # ECG ↔ ABP — cardiac cycle, pulse transit time
    (0, 2),  # ECG ↔ PPG — cardiac cycle, peripheral pulse
    (1, 2),  # ABP ↔ PPG — arterial pulse wave (거의 동형)
    # Systemic-Pulmonary (체순환 ↔ 폐순환)
    (1, 6),  # ABP ↔ PAP — 유사 동맥 morphology, 다른 amplitude scale
    # Right Heart (우심계 혈역학)
    (3, 6),  # CVP ↔ PAP — 우심 전후부하, 같은 pressure 도메인
    # Cerebral Perfusion (뇌관류)
    (1, 7),  # ABP ↔ ICP — CPP = MAP - ICP, 뇌자동조절
    # Respiratory cycle (호흡 주기)
    (4, 8),  # CO2 ↔ RESP — capnography ↔ impedance, 같은 호흡 cycle
    (5, 8),  # AWP ↔ RESP — airway pressure ↔ impedance
    (4, 5),  # CO2 ↔ AWP — 같은 호흡 cycle, ventilator 동기 (제한적)
}


# 채널명 → (signal_type, local_spatial_id) 역매핑
CHANNEL_NAME_TO_SPATIAL: dict[str, tuple[int, int]] = {
    # ECG (0) — 12-lead
    "ECG Lead I":   (0, 1), "ECG I":   (0, 1), "I":   (0, 1),
    "ECG Lead II":  (0, 2), "ECG II":  (0, 2), "II":  (0, 2),
    "ECG Lead III": (0, 3), "ECG III": (0, 3), "III": (0, 3),
    "ECG aVR": (0, 4), "aVR": (0, 4),
    "ECG aVL": (0, 5), "aVL": (0, 5),
    "ECG aVF": (0, 6), "aVF": (0, 6),
    "ECG V1": (0, 7),  "V1": (0, 7),
    "ECG V2": (0, 8),  "V2": (0, 8),
    "ECG V3": (0, 9),  "V3": (0, 9),
    "ECG V4": (0, 10), "V4": (0, 10),
    "ECG V5": (0, 11), "V5": (0, 11), "ECG Lead V5": (0, 11),
    "ECG V6": (0, 12), "V6": (0, 12),
    # ABP (1)
    "ABP Radial": (1, 1),
    "ART": (1, 1),
    "ABP Femoral": (1, 2),
    "FEM": (1, 2),
    # PPG (2)
    "PPG": (2, 0),
    "PLETH": (2, 1),
    "PPG Finger": (2, 1),
    # CVP (3)
    "CVP": (3, 0),
    # CO2 (4)
    "CO2": (4, 0),
    # AWP (5)
    "AWP": (5, 0),
    # PAP (6)
    "PAP": (6, 0),
    # ICP (7)
    "ICP": (7, 0),
    # RESP (8)
    "RESP": (8, 1),
    "Impedance": (8, 1),
    "FLOW": (8, 2),
    "FLOW_WAV": (8, 2),
}
