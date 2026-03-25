"""Spatial ID 매핑 테이블.

signal_type(대분류) + spatial_id(소분류) 이중 인코딩 체계.
각 signal_type 내에서 로컬 ID를 정의하고,
전역 고유 ID(global spatial ID)로 변환하는 유틸리티를 제공한다.

로컬 ID 0은 항상 Unknown(위치 정보 없음)이다.
"""

from __future__ import annotations

# signal_type → {로컬 이름: 로컬 ID}
SPATIAL_MAP: dict[int, dict[str, int]] = {
    # ECG (signal_type=0)
    0: {
        "Unknown": 0,
        "Lead_I": 1, "Lead_II": 2, "Lead_III": 3,
        "aVR": 4, "aVL": 5, "aVF": 6,
        "V1": 7, "V2": 8, "V3": 9, "V4": 10, "V5": 11, "V6": 12,
    },
    # ABP (signal_type=1)
    1: {
        "Unknown": 0,
        "Radial": 1, "Femoral": 2, "Brachial": 3,
    },
    # EEG (signal_type=2) — 10-20 system + bipolar montages
    2: {
        "Unknown": 0,
        "Fp1": 1, "Fp2": 2, "F3": 3, "F4": 4,
        "C3": 5, "C4": 6, "P3": 7, "P4": 8,
        "O1": 9, "O2": 10,
        "F7": 11, "F8": 12, "T3": 13, "T4": 14, "T5": 15, "T6": 16,
        "Fz": 17, "Cz": 18, "Pz": 19, "Oz": 20, "Fpz": 21,
        "Fpz-Cz": 22, "Pz-Oz": 23,
    },
    # PPG (signal_type=3)
    3: {
        "Unknown": 0,
        "Finger": 1, "Ear": 2, "Forehead": 3, "Wrist": 4,
    },
    # CVP (signal_type=4)
    4: {
        "Unknown": 0,
        "Internal_Jugular": 1, "Subclavian": 2,
    },
    # CO2/Capnography (signal_type=5)
    5: {
        "Unknown": 0,
        "Mainstream": 1, "Sidestream": 2,
    },
    # AWP/Airway Pressure (signal_type=6)
    6: {
        "Unknown": 0,
        "Proximal": 1, "Distal": 2,
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
    "ECG Lead I": (0, 1), "ECG Lead II": (0, 2), "ECG Lead III": (0, 3),
    "ECG I": (0, 1), "ECG II": (0, 2), "ECG III": (0, 3),
    "I": (0, 1), "II": (0, 2), "III": (0, 3),
    "aVR": (0, 4), "aVL": (0, 5), "aVF": (0, 6),
    "V1": (0, 7), "V2": (0, 8), "V3": (0, 9),
    "V4": (0, 10), "V5": (0, 11), "V6": (0, 12),
    # EEG — 10-20 system
    "EEG Fp1": (2, 1), "EEG Fp2": (2, 2),
    "EEG F3": (2, 3), "EEG F4": (2, 4),
    "EEG C3": (2, 5), "EEG C4": (2, 6),
    "EEG P3": (2, 7), "EEG P4": (2, 8),
    "EEG O1": (2, 9), "EEG O2": (2, 10),
    "EEG F7": (2, 11), "EEG F8": (2, 12),
    "EEG T3": (2, 13), "EEG T4": (2, 14),
    "EEG T5": (2, 15), "EEG T6": (2, 16),
    "EEG Fz": (2, 17), "EEG Cz": (2, 18),
    "EEG Pz": (2, 19), "EEG Oz": (2, 20),
    "EEG Fpz": (2, 21),
    # EEG — bipolar (Sleep-EDF)
    "EEG Fpz-Cz": (2, 22), "EEG Pz-Oz": (2, 23),
    # CVP
    "CVP": (4, 0),
    "CVP Internal Jugular": (4, 1), "CVP Subclavian": (4, 2),
    # CO2
    "CO2": (5, 0),
    "CO2 Mainstream": (5, 1), "CO2 Sidestream": (5, 2),
    # AWP
    "AWP": (6, 0),
    "AWP Proximal": (6, 1), "AWP Distal": (6, 2),
}