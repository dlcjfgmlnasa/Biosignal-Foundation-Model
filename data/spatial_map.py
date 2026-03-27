"""Spatial ID 매핑 테이블.

signal_type(대분류) + spatial_id(소분류) 이중 인코딩 체계.
각 signal_type 내에서 로컬 ID를 정의하고,
전역 고유 ID(global spatial ID)로 변환하는 유틸리티를 제공한다.

로컬 ID 0은 항상 Unknown(위치 정보 없음)이다.

VitalDB Waveform 기준 (2026-03-27):
  SNUADC/ECG_II (500Hz, mV), SNUADC/ECG_V5 (500Hz, mV)
  SNUADC/ART (500Hz, mmHg), SNUADC/FEM (500Hz, mmHg)
  SNUADC/CVP (500Hz, mmHg)
  SNUADC/PLETH (500Hz, unitless)
  Primus/AWP (62.5Hz, hPa), Primus/CO2 (62.5Hz, mmHg)
  BIS/EEG1_WAV (128Hz, μV), BIS/EEG2_WAV (128Hz, μV)
"""

from __future__ import annotations

# signal_type → {로컬 이름: 로컬 ID}
SPATIAL_MAP: dict[int, dict[str, int]] = {
    # ECG (signal_type=0) — VitalDB: ECG_II(Lead II), ECG_V5(Lead V5)
    0: {
        "Unknown": 0,
        "Lead_II": 1, "Lead_V5": 2,
    },
    # ABP (signal_type=1) — VitalDB: ART(Radial), FEM(Femoral)
    1: {
        "Unknown": 0,
        "Radial": 1, "Femoral": 2,
    },
    # EEG (signal_type=2) — VitalDB: BIS/EEG1_WAV, BIS/EEG2_WAV (forehead, 위치 구분 없음)
    2: {
        "Unknown": 0,
    },
    # PPG (signal_type=3) — VitalDB: PLETH (Finger)
    3: {
        "Unknown": 0,
        "Finger": 1,
    },
    # CVP (signal_type=4) — VitalDB: CVP
    4: {
        "Unknown": 0,
    },
    # CO2/Capnography (signal_type=5) — VitalDB: Primus/CO2 (Sidestream)
    5: {
        "Unknown": 0,
    },
    # AWP/Airway Pressure (signal_type=6) — VitalDB: Primus/AWP
    6: {
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


# 채널명 → (signal_type, local_spatial_id) 역매핑
CHANNEL_NAME_TO_SPATIAL: dict[str, tuple[int, int]] = {
    # ECG
    "ECG Lead II": (0, 1), "ECG II": (0, 1), "II": (0, 1),
    "ECG Lead V5": (0, 2), "ECG V5": (0, 2), "V5": (0, 2),
    # ABP
    "ABP Radial": (1, 1), "ART": (1, 1),
    "ABP Femoral": (1, 2), "FEM": (1, 2),
    # EEG
    "EEG": (2, 0), "BIS EEG": (2, 0),
    # PPG
    "PPG": (3, 0), "PLETH": (3, 1), "PPG Finger": (3, 1),
    # CVP
    "CVP": (4, 0),
    # CO2
    "CO2": (5, 0),
    # AWP
    "AWP": (6, 0),
}
