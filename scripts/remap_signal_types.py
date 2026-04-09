"""signal_type 재매핑 스크립트.

EEG 제거 후 signal_type 번호 체계 변경:
  이전: ECG=0, ABP=1, EEG=2, PPG=3, CVP=4, CO2=5, AWP=6
  새로: ECG=0, ABP=1, PPG=2, CVP=3, CO2=4, AWP=5

manifest.json 내 signal_type 정수를 새 번호로 변환한다.
EEG(이전 2)는 -1로 표시된다.

사용법:
    python scripts/remap_signal_types.py --data-dir ../updown/bio_fm/data
"""

import argparse
import json
from pathlib import Path

REMAP = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def remap(data_dir: str) -> None:
    root = Path(data_dir)

    for split_dir in sorted(root.rglob("manifest.json")):
        mf = split_dir
        data = json.load(open(mf))
        changed = False
        for s in data["sessions"]:
            for r in s["recordings"]:
                old = r["signal_type"]
                if old in REMAP:
                    if old != REMAP[old]:
                        changed = True
                    r["signal_type"] = REMAP[old]
                else:
                    r["signal_type"] = -1
                    changed = True
        if changed:
            json.dump(data, open(mf, "w"), indent=2, ensure_ascii=False)

    print("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="signal_type 재매핑")
    parser.add_argument("--data-dir", required=True, help="데이터 루트 디렉토리")
    args = parser.parse_args()
    remap(args.data_dir)
