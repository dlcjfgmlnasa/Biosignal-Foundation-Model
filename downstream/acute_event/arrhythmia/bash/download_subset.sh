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
#   bash downstream/acute_event/arrhythmia/bash/download_subset.sh \
#       [OUT_DIR] [RECORDS_FILE] [PARALLEL]
#
# 예시:
#   bash downstream/acute_event/arrhythmia/bash/download_subset.sh \
#       datasets/raw/mimic3-ext-ppg-arrhythmia \
#       downstream/acute_event/arrhythmia/RECORDS-arrhythmia-subset \
#       4

set -euo pipefail

# Windows Git Bash에서 winget으로 설치한 wget.exe 자동 PATH 추가
WGET_WINGET="/c/Users/${USER:-$USERNAME}/AppData/Local/Microsoft/WinGet/Packages/JernejSimoncic.Wget_Microsoft.Winget.Source_8wekyb3d8bbwe"
if [[ -f "$WGET_WINGET/wget.exe" ]]; then
  export PATH="$WGET_WINGET:$PATH"
fi

OUT_DIR="${1:-datasets/raw/mimic3-ext-ppg-arrhythmia}"
RECORDS_FILE="${2:-downstream/acute_event/arrhythmia/RECORDS-arrhythmia-subset}"
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
  # 이미 .dat 파일 많은 환자는 전체 skip (HEAD 체크 비용 회피)
  local existing=$(find "$out/$folder" -name "*.dat" 2>/dev/null | wc -l)
  if [[ "$existing" -gt 0 ]]; then
    echo "  SKIP $folder ($existing .dat files already present)"
    return 0
  fi
  # 환경변수 PHYSIONET_USER / PHYSIONET_PASS 있으면 사용, 없으면 .netrc 폴백
  local auth_args=()
  if [[ -n "${PHYSIONET_USER:-}" && -n "${PHYSIONET_PASS:-}" ]]; then
    auth_args=(--user="$PHYSIONET_USER" --password="$PHYSIONET_PASS")
  fi
  # 신규 환자만 wget
  # --cut-dirs=3: /files/mimic-iii-ext-ppg/1.1.0/ 3단계 제거 → p00/pXXXXXX/ 보존
  # -e robots=off: robots.txt 요청 skip (잡파일 방지)
  wget -q -c -r -np -nH --cut-dirs=3 -e robots=off \
    "${auth_args[@]}" \
    -P "$out" \
    "$BASE_URL/$folder/" 2>/dev/null || true
  local dat_count=$(find "$out/$folder" -name "*.dat" 2>/dev/null | wc -l)
  if [[ "$dat_count" -gt 0 ]]; then
    echo "  OK   $folder ($dat_count .dat files)"
  else
    echo "  FAIL $folder (0 .dat files)"
  fi
}

export -f download_one
export BASE_URL
export PHYSIONET_USER PHYSIONET_PASS

# 우선순위: GNU parallel → xargs -P (Git Bash 기본) → sequential
if command -v parallel >/dev/null 2>&1; then
  # CRLF 제거 후 parallel로 전달
  tr -d '\r' < "$RECORDS_FILE" | parallel -j "$PARALLEL" download_one {} "$OUT_DIR"
elif command -v xargs >/dev/null 2>&1; then
  echo "INFO: Using xargs -P $PARALLEL (GNU parallel not installed)." >&2
  tr -d '\r' < "$RECORDS_FILE" \
    | xargs -I {} -P "$PARALLEL" bash -c 'download_one "$1" "$2"' _ {} "$OUT_DIR"
else
  echo "WARN: Neither parallel nor xargs found, using sequential download." >&2
  while IFS= read -r folder; do
    folder="${folder%$'\r'}"  # CRLF 대응: trailing \r 제거
    [[ -z "$folder" ]] && continue
    download_one "$folder" "$OUT_DIR"
  done < "$RECORDS_FILE"
fi

echo
echo "Download complete. Summary:"
echo "  Target folders: $N"
echo "  Actual folders: $(find "$OUT_DIR" -mindepth 2 -maxdepth 2 -type d | wc -l)"
echo "  Total size:     $(du -sh "$OUT_DIR" | cut -f1)"
