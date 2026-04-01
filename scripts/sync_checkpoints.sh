#!/bin/bash
# 로컬 디스크의 checkpoint를 네트워크 마운트로 주기적으로 동기화
# Usage: bash scripts/sync_checkpoints.sh <SRC> <DST> [INTERVAL]
#
# 주기적으로 best checkpoint + training_log.csv를 복사하고,
# 로컬의 오래된 non-best checkpoint를 삭제하여 디스크 공간 확보
#
# 예시:
#   bash scripts/sync_checkpoints.sh outputs/phase1/base_1min_d256_L12 ../updown/outputs/phase1/base_1min_d256_L12
#   bash scripts/sync_checkpoints.sh outputs/phase1/base_1min_d256_L12 ../updown/outputs/phase1/base_1min_d256_L12 120

SRC="${1:?Usage: $0 <SRC> <DST> [INTERVAL]}"
DST="${2:?Usage: $0 <SRC> <DST> [INTERVAL]}"
INTERVAL="${3:-60}"

mkdir -p "$DST/checkpoints"
mkdir -p "$DST/figures/recon"
mkdir -p "$DST/figures/next_pred"

echo "[sync] Started: $SRC -> $DST (every ${INTERVAL}s)"

while true; do
    # 1. best checkpoint 복사
    for f in "$SRC"/checkpoints/*best*; do
        [ -f "$f" ] && cp "$f" "$DST/checkpoints/"
    done

    # 2. training log 복사
    [ -f "$SRC/training_log.csv" ] && cp "$SRC/training_log.csv" "$DST/"

    # 3. 최신 figure 복사
    latest_recon=$(ls -t "$SRC"/figures/recon/*.png 2>/dev/null | head -1)
    latest_np=$(ls -t "$SRC"/figures/next_pred/*.png 2>/dev/null | head -1)
    [ -n "$latest_recon" ] && cp "$latest_recon" "$DST/figures/recon/"
    [ -n "$latest_np" ] && cp "$latest_np" "$DST/figures/next_pred/"

    # 4. 로컬 디스크 공간 확보: best가 아닌 오래된 checkpoint 삭제
    for f in "$SRC"/checkpoints/checkpoint_*.pt; do
        [ -f "$f" ] || continue
        echo "$f" | grep -q "best" && continue
        rm -f "$f"
        echo "[sync] Removed old checkpoint: $f"
    done

    echo "[sync] $(date '+%H:%M:%S') synced. Local disk: $(df -h . | tail -1 | awk '{print $4}') free"
    sleep $INTERVAL
done
