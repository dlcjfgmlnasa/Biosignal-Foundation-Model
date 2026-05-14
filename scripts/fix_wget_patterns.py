# -*- coding: utf-8 -*-
"""wget_commands_*.txt 파일들의 -A 패턴을 broader 하게 fix.

MIMIC-III WFDB 의 환자 폴더는 master record (예: p000020-2183-04-28-17-47.hea) +
sub-records (예: 3544749_layout.hea, 3544749_0001.dat) 로 구성됨.

기존 wget 패턴 `-A "p000020-2183-04-28-17-47*"` 는 master 만 매칭 → .dat 0건.

Fix: 2단계 wget 파일 생성
  *_headers.txt — `-A "*.hea"` 헤더만 (작음, CO2 등 채널 확인용)
  *_signals.txt — `-A "*.hea,*.dat"` 헤더 + 신호 (full)

원본은 .orig 백업.

사용법:
  python -m scripts.fix_wget_patterns wget_commands_vent_need.txt \\
      wget_commands_sepsis.txt wget_commands_cardiac_arrest.txt
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


# 매칭 패턴: -A "p000020-2183-04-28-17-47*" 또는 유사한 record-name pattern
RECORD_PATTERN_RE = re.compile(r'-A\s+"[^"]+"')


def fix_line(line: str, hea_only: bool = False) -> str:
    """단일 wget 명령 fix."""
    if hea_only:
        new_filter = '-A "*.hea"'
    else:
        new_filter = '-A "*.hea,*.dat"'
    return RECORD_PATTERN_RE.sub(new_filter, line, count=1)


def fix_file(in_path: Path, hea_only: bool = False) -> Path:
    """파일 단위 fix. 결과는 별도 파일로 저장."""
    suffix = "_headers.txt" if hea_only else "_signals.txt"
    out_path = in_path.parent / (in_path.stem + suffix)

    n_total = 0
    n_changed = 0
    with open(in_path, encoding="utf-8") as f_in, \
         open(out_path, "w", encoding="utf-8", newline="\n") as f_out:
        for line in f_in:
            n_total += 1
            new_line = fix_line(line, hea_only=hea_only)
            if new_line != line:
                n_changed += 1
            f_out.write(new_line)
    print(f"  {in_path.name} → {out_path.name}: {n_changed}/{n_total} lines patched")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fix wget_commands_*.txt -A pattern for MIMIC-III WFDB sub-record downloads"
    )
    parser.add_argument(
        "files", nargs="+", type=str,
        help="원본 wget_commands_*.txt 파일 경로",
    )
    args = parser.parse_args()

    for fname in args.files:
        in_path = Path(fname)
        if not in_path.is_file():
            print(f"  SKIP {fname}: not found", file=sys.stderr)
            continue

        # 백업
        backup = in_path.with_suffix(in_path.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(in_path, backup)
            print(f"  Backed up {in_path.name} → {backup.name}")

        # Headers-only + full signals 두 버전 생성
        fix_file(in_path, hea_only=True)
        fix_file(in_path, hea_only=False)

    print("\nDone. Use *_headers.txt for header-only fetch (small, for CO2 pre-check),")
    print("or *_signals.txt for full .hea+.dat download.")


if __name__ == "__main__":
    main()
