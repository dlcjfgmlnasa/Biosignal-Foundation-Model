#!/usr/bin/env bash
# MIMIC-III Waveform Matched 선별 다운로드 스크립트 (ICP 레코드).
#
# `ICP-RECORDS`의 각 line은 특정 record path
# (예: `p00/p000907/p000907-2163-10-01-23-21`).
# 한 patient folder에 여러 record가 섞여있을 수 있으므로,
# wget의 -A 패턴으로 해당 record prefix와 일치하는 파일만 선별 다운로드한다.
#
# Multi-segment record 구성:
#   <rec>.hea, <rec>_layout.hea, <rec>_<seg_id>.hea, <rec>_<seg_id>.dat
#   → -A "<rec>*" 한 패턴으로 모두 매칭.
#
# Prerequisites:
#   - download_waveforms.py scan 으로 ICP-RECORDS 생성 완료
#   - ~/.netrc 에 PhysioNet 계정 설정
#     (machine physionet.org login <user> password <pw>)
#     또는 PHYSIONET_USER / PHYSIONET_PASS 환경변수
#
# 사용법:
#   bash downstream/acute_event/intracranial_hypertension/bash/download_subset.sh \
#       [OUT_DIR] [RECORDS_FILE] [PARALLEL]
#
# 예시:
#   bash downstream/acute_event/intracranial_hypertension/bash/download_subset.sh \
#       datasets/raw/mimic3-waveform-ich \
#       downstream/acute_event/intracranial_hypertension/ICP-RECORDS \
#       4

set -euo pipefail

# Windows Git Bash에서 winget으로 설치한 wget.exe 자동 PATH 추가
WGET_WINGET="/c/Users/${USER:-$USERNAME}/AppData/Local/Microsoft/WinGet/Packages/JernejSimoncic.Wget_Microsoft.Winget.Source_8wekyb3d8bbwe"
if [[ -f "$WGET_WINGET/wget.exe" ]]; then
  export PATH="$WGET_WINGET:$PATH"
fi

OUT_DIR="${1:-datasets/raw/mimic3-waveform-ich}"
RECORDS_FILE="${2:-downstream/acute_event/intracranial_hypertension/ICP-RECORDS}"
PARALLEL="${3:-4}"

# wget 존재 확인
if ! command -v wget >/dev/null 2>&1; then
  echo "ERROR: wget not found. Install via 'winget install JernejSimoncic.Wget' (Windows) or apt/brew." >&2
  exit 1
fi

BASE_URL="https://physionet.org/files/mimic3wdb-matched/1.0"

if [[ ! -f "$RECORDS_FILE" ]]; then
  echo "ERROR: RECORDS file not found: $RECORDS_FILE" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"
N=$(grep -cve '^\s*$' "$RECORDS_FILE")
echo "Downloading $N ICP records to $OUT_DIR with $PARALLEL workers..."

download_one() {
  local record_path="$1"        # e.g., p00/p000907/p000907-2163-10-01-23-21
  local out="$2"
  local patient_folder
  local rec_name
  patient_folder=$(dirname "$record_path")     # p00/p000907
  rec_name=$(basename "$record_path")          # p000907-2163-10-01-23-21
  local rec_dir="$out/$patient_folder"

  # Skip 조건: master .hea + layout .hea + 최소 1개 .dat 모두 존재.
  # wget -c가 누락분만 추가 다운로드하지만, 완전히 받은 record는 굳이 재시도 X.
  if [[ -f "$rec_dir/$rec_name.hea" && -f "$rec_dir/${rec_name}_layout.hea" ]]; then
    local dat_count
    dat_count=$(find "$rec_dir" -maxdepth 1 -name "${rec_name}_*.dat" 2>/dev/null | wc -l)
    if [[ "$dat_count" -gt 0 ]]; then
      echo "  SKIP $record_path ($dat_count .dat segments complete)"
      return 0
    fi
  fi

  # 환경변수 PHYSIONET_USER / PHYSIONET_PASS 있으면 사용, 없으면 .netrc 폴백
  local auth_args=()
  if [[ -n "${PHYSIONET_USER:-}" && -n "${PHYSIONET_PASS:-}" ]]; then
    auth_args=(--user="$PHYSIONET_USER" --password="$PHYSIONET_PASS")
  fi

  # -r/-np: patient folder 내 재귀, 상위 이동 차단
  # -nH --cut-dirs=3: /files/mimic3wdb-matched/1.0/ 3단계 제거 → p00/p000907/ 보존
  # -A "<rec>*": 해당 record prefix 매칭 파일만 보존
  #   (master .hea, _layout.hea, _<seg>.hea, _<seg>.dat 모두 포함)
  # -e robots=off: robots.txt 요청 skip
  wget -q -c -r -np -nH --cut-dirs=3 -e robots=off \
    -A "${rec_name}*" \
    "${auth_args[@]}" \
    -P "$out" \
    "$BASE_URL/$patient_folder/" 2>/dev/null || true

  local dat_count
  dat_count=$(find "$rec_dir" -maxdepth 1 -name "${rec_name}_*.dat" 2>/dev/null | wc -l)
  if [[ -f "$rec_dir/$rec_name.hea" && "$dat_count" -gt 0 ]]; then
    echo "  OK   $record_path ($dat_count .dat segments)"
  else
    local has_master="N"
    [[ -f "$rec_dir/$rec_name.hea" ]] && has_master="Y"
    echo "  FAIL $record_path (master_hea=$has_master, dats=$dat_count)"
  fi
}

export -f download_one
export BASE_URL
export PHYSIONET_USER PHYSIONET_PASS

# 우선순위: GNU parallel → xargs -P (Git Bash 기본) → sequential
if command -v parallel >/dev/null 2>&1; then
  tr -d '\r' < "$RECORDS_FILE" | parallel -j "$PARALLEL" download_one {} "$OUT_DIR"
elif command -v xargs >/dev/null 2>&1; then
  echo "INFO: Using xargs -P $PARALLEL (GNU parallel not installed)." >&2
  tr -d '\r' < "$RECORDS_FILE" \
    | xargs -I {} -P "$PARALLEL" bash -c 'download_one "$1" "$2"' _ {} "$OUT_DIR"
else
  echo "WARN: Neither parallel nor xargs found, using sequential download." >&2
  while IFS= read -r record_path; do
    record_path="${record_path%$'\r'}"  # CRLF 대응
    [[ -z "$record_path" ]] && continue
    download_one "$record_path" "$OUT_DIR"
  done < "$RECORDS_FILE"
fi

echo
echo "Download complete. Summary:"
echo "  Target records:    $N"
echo "  Master .hea found: $(find "$OUT_DIR" -mindepth 3 -maxdepth 3 -name "*.hea" ! -name "*_layout.hea" ! -name "*_[0-9]*.hea" 2>/dev/null | wc -l)"
echo "  Total .dat files:  $(find "$OUT_DIR" -name "*.dat" 2>/dev/null | wc -l)"
echo "  Total size:        $(du -sh "$OUT_DIR" | cut -f1)"