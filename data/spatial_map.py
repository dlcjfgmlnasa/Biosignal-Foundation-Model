"""Spatial ID 매핑 테이블.

signal_type(대분류) + spatial_id(소분류) 이중 인코딩 체계.
각 signal_type 내에서 로컬 ID를 정의하고,
전역 고유 ID(global spatial ID)로 변환하는 유틸리티를 제공한다.

로컬 ID 0은 항상 Unknown(위치 정보 없음)이다.

지원 신호 (8종):
  ECG(0), ABP(1), PPG(2), CVP(3), CO2(4), AWP(5), PAP(6), ICP(7)

데이터 소스:
  VitalDB Open: SNUADC/ (OR, 500Hz) — ECG, ABP, PPG, CVP, CO2, AWP
  K-MIMIC-MORTAL: SNUADCM/ (ICU, 500Hz) — ECG, ABP, PPG, CVP, PAP, ICP
"""

from __future__ import annotations

# signal_type → {로컬 이름: 로컬 ID}
SPATIAL_MAP: dict[int, dict[str, int]] = {
    # ECG (signal_type=0)
    0: {
        "Unknown": 0,
        "Lead_II": 1, "Lead_V5": 2,
    },
    # ABP (signal_type=1)
    1: {
        "Unknown": 0,
        "Radial": 1, "Femoral": 2,
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
}

MECHANISM_GROUP_NAMES: dict[int, str] = {
    0: "Cardiovascular",
    1: "Respiratory",
}


# 채널명 → (signal_type, local_spatial_id) 역매핑
CHANNEL_NAME_TO_SPATIAL: dict[str, tuple[int, int]] = {
    # ECG (0)
    "ECG Lead II": (0, 1), "ECG II": (0, 1), "II": (0, 1),
    "ECG Lead V5": (0, 2), "ECG V5": (0, 2), "V5": (0, 2),
    # ABP (1)
    "ABP Radial": (1, 1), "ART": (1, 1),
    "ABP Femoral": (1, 2), "FEM": (1, 2),
    # PPG (2)
    "PPG": (2, 0), "PLETH": (2, 1), "PPG Finger": (2, 1),
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
}
