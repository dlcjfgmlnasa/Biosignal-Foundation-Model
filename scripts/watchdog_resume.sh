#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# watchdog_resume.sh — 학습 stall(NCCL hang)/crash 자동 복구 래퍼.
#
# 동작:
#   1) torchrun 으로 학습 시작 (마지막 ckpt 자동 --resume).
#   2) 로그 파일 mtime 을 60s 마다 확인 → STALL_SEC 동안 안 자라면
#      "hang" 으로 판정하고 프로세스를 강제 종료.
#   3) 종료(crash/kill/정상)되면 done 패턴 검사 → 미완이면 다시 resume.
#      Phase2 best ckpt 가 있으면 그걸로(이어서), 없으면 Phase1 transition.
#
# NCCL hang 을 무한대기 대신 에러로 떨구도록 async error handling 도 켠다.
#
# Usage (tmux 안에서):
#   bash scripts/watchdog_resume.sh
#
#   # 파라미터 override 예
#   STALL_SEC=600 NPROC=4 bash scripts/watchdog_resume.sh
#
#   # Phase1 감시로 쓰려면
#   MODULE=train.1_channel_independency \
#   CONFIG=configs/ablation/_phase1_base.yaml \
#   OUT=$HOME/workspace/k-mimic-/bio_fm/outputs/ablation/ablation_phase1_base \
#   LOG=logs/ablation_sweep/stage0_base_p1.log \
#   bash scripts/watchdog_resume.sh
# ─────────────────────────────────────────────────────────────────
set -uo pipefail

NPROC="${NPROC:-4}"
STALL_SEC="${STALL_SEC:-900}"          # 로그 15분 정체 = hang 판정 (보수적)
MAX_ATTEMPTS="${MAX_ATTEMPTS:-50}"     # 무한루프 backstop
RESTART_WAIT="${RESTART_WAIT:-30}"     # 재시작 전 대기(포트/메모리 정리)

MODULE="${MODULE:-train.2_any_variate}"
CONFIG="${CONFIG:-configs/ablation/_phase2_base.yaml}"
LOG="${LOG:-logs/ablation_sweep/stage1_base_p2.log}"
OUT="${OUT:-$HOME/workspace/k-mimic-/bio_fm/outputs/ablation/ablation_phase2_base}"
P1_DIR="${P1_DIR:-$HOME/workspace/k-mimic-/bio_fm/outputs/ablation/ablation_phase1_base/checkpoints}"
# 정상 완주 판정 패턴 (train 스크립트 종료 메시지)
DONE_RE="${DONE_RE:-complete\. Final|Phase 2 complete|Phase 1 complete|Final val loss}"

# NCCL hang → abort (무한대기 방지). 구/신버전 모두 set.
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$(dirname "$LOG")"

pick_resume() {
    # Phase2(자기) best 우선 → 없으면 Phase1 transition ckpt
    local c
    c=$(ls -t "$OUT"/checkpoints/*_best.pt 2>/dev/null | head -1)
    if [ -z "$c" ]; then
        c=$(ls -t "$P1_DIR"/*_best.pt 2>/dev/null | head -1)
    fi
    echo "$c"
}

ts() { date "+%Y-%m-%d %H:%M:%S"; }

attempt=0
while [ "$attempt" -lt "$MAX_ATTEMPTS" ]; do
    attempt=$((attempt + 1))
    RESUME="$(pick_resume)"
    if [ -z "$RESUME" ]; then
        echo "[watchdog] ❌ resume ckpt 없음 (OUT/P1_DIR 확인) — 중단" | tee -a "$LOG"
        exit 1
    fi
    echo "════════════════════════════════════════════════════════" | tee -a "$LOG"
    echo "[watchdog] attempt $attempt/$MAX_ATTEMPTS @ $(ts)" | tee -a "$LOG"
    echo "[watchdog] resume: $RESUME" | tee -a "$LOG"
    echo "════════════════════════════════════════════════════════" | tee -a "$LOG"

    torchrun --nproc_per_node="$NPROC" -m "$MODULE" \
        --config "$CONFIG" --resume "$RESUME" >> "$LOG" 2>&1 &
    PID=$!

    # ── stall 감시 ──
    while kill -0 "$PID" 2>/dev/null; do
        sleep 60
        if [ -f "$LOG" ]; then
            now=$(date +%s)
            mtime=$(stat -c %Y "$LOG" 2>/dev/null || echo "$now")
            age=$((now - mtime))
            if [ "$age" -gt "$STALL_SEC" ]; then
                echo "[watchdog] ⚠️ STALL ${age}s > ${STALL_SEC}s — hang 판정, kill @ $(ts)" | tee -a "$LOG"
                kill "$PID" 2>/dev/null
                sleep 10
                pkill -9 -f "$MODULE" 2>/dev/null
                sleep 5
                break
            fi
        fi
    done
    wait "$PID" 2>/dev/null
    rc=$?

    # ── 완주 판정 ──
    if grep -Eq "$DONE_RE" "$LOG"; then
        echo "[watchdog] ✅ DONE 패턴 감지 — 학습 완료, watchdog 종료 @ $(ts)" | tee -a "$LOG"
        exit 0
    fi

    echo "[watchdog] 프로세스 종료(rc=$rc), 미완 → ${RESTART_WAIT}s 후 재시작 @ $(ts)" | tee -a "$LOG"
    sleep "$RESTART_WAIT"
done

echo "[watchdog] ❌ MAX_ATTEMPTS($MAX_ATTEMPTS) 도달 — 수동 점검 필요 @ $(ts)" | tee -a "$LOG"
exit 1
