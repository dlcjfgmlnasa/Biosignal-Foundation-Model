"""signal_type 재매핑 스크립트.

EEG 제거 후 signal_type 번호 체계 변경:
  이전: ECG=0, ABP=1, EEG=2, PPG=3, CVP=4, CO2=5, AWP=6
  새로: ECG=0, ABP=1, PPG=2, CVP=3, CO2=4, AWP=5

manifest.json 내 signal_type 정수를 새 번호로 변환한다.
EEG(이전 2) 레코딩은 manifest에서 완전히 제거된다.

사용법:
    python scripts/remap_signal_types.py --data-dir ../updown/bio_fm/data
"""

import argparse
import json
from pathlib import Path

REMAP = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4, 6: 5}


def remap(data_dir: str) -> None:
    root = Path(data_dir)
    n_files = 0
    n_removed = 0

    for mf in sorted(root.rglob("manifest.json")):
        data = json.load(open(mf))
        changed = False
        for s in data["sessions"]:
            new_recs = []
            for r in s["recordings"]:
                old = r["signal_type"]
                if old in REMAP:
                    if old != REMAP[old]:
                        changed = True
                    r["signal_type"] = REMAP[old]
                    new_recs.append(r)
                else:
                    changed = True
                    n_removed += 1
            s["recordings"] = new_recs
        if changed:
            json.dump(data, open(mf, "w"), indent=2, ensure_ascii=False)
        n_files += 1

    print(f"Done: {n_files} manifests processed, {n_removed} EEG recordings removed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="signal_type 재매핑")
    parser.add_argument("--data-dir", required=True, help="데이터 루트 디렉토리")
    args = parser.parse_args()
    remap(args.data_dir)
