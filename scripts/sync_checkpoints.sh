#!/bin/bash
# 로컬 출력을 네트워크 마운트로 주기적으로 동기화 후 로컬 정리
# Usage: bash scripts/sync_checkpoints.sh <SRC> <DST> [INTERVAL]
#
# 예시:
#   bash scripts/sync_checkpoints.sh outputs/phase1/exp1 ../updown/outputs/phase1/exp1 30

SRC="${1:?Usage: $0 <SRC> <DST> [INTERVAL]}"
DST="${2:?Usage: $0 <SRC> <DST> [INTERVAL]}"
INTERVAL="${3:-60}"

mkdir -p "$DST/checkpoints" "$DST/figures/recon" "$DST/figures/next_pred"

echo "[sync] Started: $SRC -> $DST (every ${INTERVAL}s)"

while true; do
    # 1. 전체 복사
    cp -ru "$SRC/." "$DST/"

    # 2. 로컬 정리 (복사 완료 후)
    rm -f "$SRC"/checkpoints/checkpoint_*.pt 2>/dev/null
    rm -f "$SRC"/figures/recon/*.png 2>/dev/null
    rm -f "$SRC"/figures/next_pred/*.png 2>/dev/null

    echo "[sync] $(date '+%H:%M:%S') synced. Local disk: $(df -h . | tail -1 | awk '{print $4}') free"
    sleep $INTERVAL
done
