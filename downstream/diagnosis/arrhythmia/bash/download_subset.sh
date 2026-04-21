#!/usr/bin/env bash
# MIMIC-III-Ext-PPG 선별 다운로드 스크립트.
#
# `RECORDS-arrhythmia-subset`의 각 환자 폴더(pXX/pXXXXXX)만 wget으로
# 재귀 다운로드한다. 전체 200GB → 선별 5-class × 200명 ≈ 20-40GB 수준.
#
# Prerequisites:
#   - build_subset.py 로 RECORDS-arrhythmia-subset 생성 완료
#   - ~/.netrc 에 PhysioNet 계정 설정 (machine physionet.org login <user> password <pw>)
#     또는 --user/--password 인자 사용
#
# 사용법:
#   bash downstream/diagnosis/arrhythmia/bash/download_subset.sh \
#       [OUT_DIR] [RECORDS_FILE] [PARALLEL]
#
# 예시:
#   bash downstream/diagnosis/arrhythmia/bash/download_subset.sh \
#       datasets/raw/mimic3-ext-ppg-arrhythmia \
#       downstream/diagnosis/arrhythmia/RECORDS-arrhythmia-subset \
#       4

set -euo pipefail

# Windows Git Bash에서 winget으로 설치한 wget.exe 자동 PATH 추가
WGET_WINGET="/c/Users/${USER:-$USERNAME}/AppData/Local/Microsoft/WinGet/Packages/JernejSimoncic.Wget_Microsoft.Winget.Source_8wekyb3d8bbwe"
if [[ -f "$WGET_WINGET/wget.exe" ]]; then
  export PATH="$WGET_WINGET:$PATH"
fi

OUT_DIR="${1:-datasets/raw/mimic3-ext-ppg-arrhythmia}"
RECORDS_FILE="${2:-downstream/diagnosis/arrhythmia/RECORDS-arrhythmia-subset}"
PARALLEL="${3:-4}"

# wget 존재 확인
if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: wget not found. Install via 'winget install JernejSimoncic.Wget' (Windows) or apt/brew." >&2
  exit 1
fi

BASE_URL="https://physionet.org/files/mimic-iii-ext-ppg/1.1.0"

if [[ ! -f "$RECORDS_FILE" ]]; then
  echo "ERROR: RECORDS file not found: $RECORDS_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
N=$(wc -l < "$RECORDS_FILE")
echo "Downloading $N patient folders to $OUT_DIR with $PARALLEL workers..."

download_one() {
  local folder="$1"
  local out="$2"
  # 환경변수 PHYSIONET_USER / PHYSIONET_PASS 있으면 사용, 없으면 .netrc 폴백
  local auth_args=()
  if [[ -n "${PHYSIONET_USER:-}" && -n "${PHYSIONET_PASS:-}" ]]; then
    auth_args=(--user="$PHYSIONET_USER" --password="$PHYSIONET_PASS")
  fi
  # wget 실행 (일부 404 무시). -c -N로 기존 파일 skip, 부분 파일 resume
  wget -q -c -N -r -np -nH --cut-dirs=4 \
    "${auth_args[@]}" \
    -P "$out" \
    "$BASE_URL/$folder/" 2>/dev/null || true
  # 실제 .dat 파일 존재로 성공 판단
  local dat_count=$(find "$out/$folder" -name "*.dat" 2>/dev/null | wc -l)
  if [[ "$dat_count" -gt 0 ]]; then
    echo "  OK   $folder ($dat_count .dat files)"
  else
    echo "  FAIL $folder (0 .dat files)"
  fi
}

export -f download_one
export BASE_URL

# GNU parallel 있으면 병렬, 없으면 순차
if command -v parallel >/dev/null 2>&1; then
  cat "$RECORDS_FILE" | parallel -j "$PARALLEL" download_one {} "$OUT_DIR"
else
  echo "WARN: GNU parallel not found, using sequential download." >&2
  while IFS= read -r folder; do
    [[ -z "$folder" ]] && continue
    download_one "$folder" "$OUT_DIR"
  done < "$RECORDS_FILE"
fi

echo
echo "Download complete. Summary:"
echo "  Target folders: $N"
echo "  Actual folders: $(find "$OUT_DIR" -mindepth 2 -maxdepth 2 -type d | wc -l)"
echo "  Total size:     $(du -sh "$OUT_DIR" | cut -f1)"
