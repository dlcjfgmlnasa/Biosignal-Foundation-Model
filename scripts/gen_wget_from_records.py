# -*- coding: utf-8 -*-
"""RECORDS-to-download → wget_commands_{task}.txt 생성기.

다른 task (cardiac_arrest/vent_need/mortality/extubation) 과 동일한 wget
명령 포맷으로, sepsis 등 신규 task 의 wget 파일을 생성한다.

MIMIC-III WFDB structure 에서 master record + sub-records 모두 받기 위해
-A "*.hea,*.dat" 패턴 사용 (이전 fix 와 일관).

per-patient unique 디렉토리만 wget — 한 환자의 여러 stay 가 같은 디렉토리 안에
섞여 있을 수 있어 sub-record 까지 모두 매칭.

사용법:
  python -m scripts.gen_wget_from_records \\
      --records downstream/outcome/sepsis/RECORDS-to-download \\
      --task-name sepsis \\
      --user dlcjfgmlnasa28 --password 'mypassword'

  → wget_commands_sepsis_headers.txt + wget_commands_sepsis_signals.txt 생성
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PHYSIONET_BASE = "https://physionet.org/files/mimic3wdb-matched/1.0"


def patient_dir_from_record(rec_path: str) -> str | None:
    """`p00/p000020/p000020-2183-04-28-17-47` → `p00/p000020/`"""
    parts = rec_path.strip().split("/")
    if len(parts) < 3:
        return None
    return f"{parts[0]}/{parts[1]}/"


def gen_wget(
    patient_dirs: list[str],
    task_name: str,
    user: str,
    password: str,
    accept_pattern: str,
    out_path: Path,
) -> None:
    """단일 wget 파일 생성."""
    out_dir = f"datasets/raw/mimic3-waveform-{task_name}"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for pdir in patient_dirs:
            url = f"{PHYSIONET_BASE}/{pdir}"
            cmd = (
                f"wget -q -c -r -np -nH --cut-dirs=3 -e robots=off "
                f"--user={user} --password={password} "
                f'-A "{accept_pattern}" '
                f'-P "{out_dir}" '
                f'"{url}"\n'
            )
            f.write(cmd)
    print(f"  → {out_path} ({len(patient_dirs)} commands)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate wget commands from RECORDS file")
    parser.add_argument(
        "--records", type=str, required=True,
        help="RECORDS-to-download 파일 (each line: p00/p000020/p000020-...)",
    )
    parser.add_argument(
        "--task-name", type=str, required=True,
        help="task 이름 (e.g., sepsis) — 출력 디렉토리는 mimic3-waveform-{task_name}",
    )
    parser.add_argument(
        "--user", type=str, default="dlcjfgmlnasa28",
        help="PhysioNet username",
    )
    parser.add_argument(
        "--password", type=str, required=True,
        help="PhysioNet password",
    )
    parser.add_argument(
        "--out-dir", type=str, default=".",
        help="wget txt 파일 출력 디렉토리 (default: repo root)",
    )
    args = parser.parse_args()

    rec_path = Path(args.records)
    if not rec_path.is_file():
        print(f"ERROR: records file not found: {rec_path}", file=sys.stderr)
        sys.exit(1)

    # 환자 dir unique 추출
    seen: set[str] = set()
    patient_dirs: list[str] = []
    n_lines = 0
    with open(rec_path, encoding="utf-8") as f:
        for line in f:
            n_lines += 1
            pdir = patient_dir_from_record(line)
            if pdir and pdir not in seen:
                seen.add(pdir)
                patient_dirs.append(pdir)

    print(f"Records read: {n_lines}, unique patient dirs: {len(patient_dirs)}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Headers only + full signals — 2 파일
    gen_wget(
        patient_dirs, args.task_name, args.user, args.password,
        accept_pattern="*.hea",
        out_path=out_root / f"wget_commands_{args.task_name}_headers.txt",
    )
    gen_wget(
        patient_dirs, args.task_name, args.user, args.password,
        accept_pattern="*.hea,*.dat",
        out_path=out_root / f"wget_commands_{args.task_name}_signals.txt",
    )

    print("\nDone. Run *_headers.txt first to fetch all .hea (sub-records included),")
    print("then scan_mimic_channels for modality pre-check, then *_signals.txt for full data.")


if __name__ == "__main__":
    main()
