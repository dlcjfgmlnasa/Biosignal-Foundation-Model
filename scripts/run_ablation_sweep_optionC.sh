#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Ablation full sweep — 옵션 C (Phase 1 reuse 공유) 자동 실행 스크립트.
#
# Stage 0 : Base P2  (Base P1 ckpt 사용, base 자체 마무리)
# Stage 1 : reuse 그룹 — 01c / 01d / 03a / 03b / 04   (P2 만)
# Stage 2 : 작은 loss/arch — 01a / 01b / 02 / 06        (P1+P2)
# Stage 3 : Sampling — 05a / 05b                         (P1+P2)
# Stage 4 : Model scale — 07a → 07b → 07c (작은 → 큰)    (P1+P2)
#
# 전제:
#   * Base P1 (exp=ablation_phase1_base) ckpt 가 이미 존재
#   * configs/ablation/variants.yaml 의 00_full 이 base exp_name 가리킴
#   * 4-GPU L40S 환경
#
# Usage:
#   # foreground (보면서)
#   bash scripts/run_ablation_sweep_optionC.sh
#
#   # background (재접속 안전, 권장)
#   nohup bash scripts/run_ablation_sweep_optionC.sh \
#       > logs/ablation_sweep/_master.log 2>&1 &
#   tail -f logs/ablation_sweep/_master.log
#
#   # 특정 stage 부터 재개
#   START_STAGE=2 bash scripts/run_ablation_sweep_optionC.sh
# ─────────────────────────────────────────────────────────────────
set -uo pipefail
# set -e 는 일부러 끔 — 한 variant 실패해도 다음 진행

# ─── 환경 변수 (override 가능) ──────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$HOME/workspace/k-mimic-/bio_fm}"
NPROC="${NPROC:-4}"
START_STAGE="${START_STAGE:-0}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/ablation_sweep}"
BASE_P1_CKPT="$REPO_ROOT/outputs/ablation/ablation_phase1_base/checkpoints/best.pt"
BASE_P2_CKPT="$REPO_ROOT/outputs/ablation/ablation_phase2_base/checkpoints/best.pt"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

# ─── 유틸 ─────────────────────────────────────────────────────
ts() { date "+%Y-%m-%d %H:%M:%S"; }
say() { echo "[$(ts)] $*"; }
section() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "  $(ts)"
    echo "════════════════════════════════════════════════════════════════"
}
require_ckpt() {
    local ckpt="$1"
    local desc="$2"
    if [ ! -f "$ckpt" ]; then
        say "❌ $desc ckpt 없음: $ckpt"
        say "   해당 단계를 먼저 완주한 후 실행하세요."
        exit 1
    fi
    say "✅ $desc ckpt 존재 — $(du -h "$ckpt" | cut -f1)"
}
run_stage() {
    local stage_num="$1"
    local stage_name="$2"
    shift 2
    local log_file="$LOG_DIR/stage${stage_num}_${stage_name}.log"
    section "Stage ${stage_num} — ${stage_name}"
    say "log: $log_file"
    say "cmd: $*"
    if "$@" 2>&1 | tee -a "$log_file"; then
        say "✅ Stage ${stage_num} (${stage_name}) — DONE"
        echo "Stage ${stage_num} (${stage_name}): DONE @ $(ts)" >> "$LOG_DIR/_summary.txt"
    else
        rc=$?
        say "⚠️  Stage ${stage_num} (${stage_name}) — FAILED (rc=$rc)"
        echo "Stage ${stage_num} (${stage_name}): FAILED rc=$rc @ $(ts)" >> "$LOG_DIR/_summary.txt"
    fi
}

# ─── 환경 점검 ────────────────────────────────────────────────
section "Pre-check"
say "REPO_ROOT  = $REPO_ROOT"
say "NPROC      = $NPROC"
say "LOG_DIR    = $LOG_DIR"
say "START_STAGE= $START_STAGE"

if [ "$START_STAGE" -le 0 ]; then
    require_ckpt "$BASE_P1_CKPT" "Base P1"
fi

nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null \
    | head -n "$NPROC" \
    | while read line; do say "GPU $line"; done

# Reuse chain dry-run 검증 (Stage 1 진입 전)
if [ "$START_STAGE" -le 1 ]; then
    section "Dry-run — reuse chain 검증"
    python -m scripts.run_ablation --dry-run \
        --variant 01c_no_crossnext \
        2>&1 | grep -E "reuse|Phase 1|Phase 2" | head -10 || true
fi

# ─── Stage 0 — Base P2 ────────────────────────────────────────
if [ "$START_STAGE" -le 0 ]; then
    if [ -f "$BASE_P2_CKPT" ]; then
        say "✅ Stage 0 skip — Base P2 ckpt 이미 존재"
    else
        run_stage 0 "base_p2" \
            torchrun --nproc_per_node="$NPROC" \
                -m train.2_any_variate \
                --config configs/ablation/_phase2_base.yaml
        require_ckpt "$BASE_P2_CKPT" "Base P2"
    fi
fi

# ─── Stage 1 — reuse 그룹 (P2 만) ─────────────────────────────
if [ "$START_STAGE" -le 1 ]; then
    run_stage 1 "reuse" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 01c_no_crossnext \
            --variant 01d_no_contrastive \
            --variant 03a_samevar_only \
            --variant 03b_cross_only \
            --variant 04_phase1_only
fi

# ─── Stage 2 — 작은 loss/arch ablation (P1+P2) ────────────────
if [ "$START_STAGE" -le 2 ]; then
    run_stage 2 "loss_arch" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 01a_no_mpm \
            --variant 01b_no_next \
            --variant 02_no_lscnorm \
            --variant 06_no_attnbias
fi

# ─── Stage 3 — Modality-balanced sampling (P1+P2) ─────────────
if [ "$START_STAGE" -le 3 ]; then
    run_stage 3 "sampling" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --group sampling
fi

# ─── Stage 4 — Model scale (작은 → 큰) ────────────────────────
if [ "$START_STAGE" -le 4 ]; then
    # 4a: 3M (제일 빠름)
    run_stage 4a "scale_3m" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 07a_size_3m

    # 4b: 30M
    run_stage 4b "scale_30m" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 07b_size_30m

    # 4c: 100M (제일 무거움 — 약 6일)
    run_stage 4c "scale_100m" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 07c_size_100m
fi

# ─── 마무리 요약 ──────────────────────────────────────────────
section "ALL DONE"
say "전체 sweep 완료 — $(ts)"
say ""
say "Stage 결과 요약:"
if [ -f "$LOG_DIR/_summary.txt" ]; then
    cat "$LOG_DIR/_summary.txt"
fi
say ""
say "ckpt 점검:"
for exp in \
    ablation_phase1_base ablation_phase2_base \
    ablation_01a_no_mpm_p1 ablation_01a_no_mpm_p2 \
    ablation_01b_no_next_p1 ablation_01b_no_next_p2 \
    ablation_01c_no_crossnext_p2 ablation_01d_no_contrastive_p2 \
    ablation_02_no_lscnorm_p1 ablation_02_no_lscnorm_p2 \
    ablation_03a_samevar_p2 ablation_03b_cross_p2 \
    ablation_05a_samp_a0_p1 ablation_05a_samp_a0_p2 \
    ablation_05b_samp_a1_p1 ablation_05b_samp_a1_p2 \
    ablation_06_no_attnbias_p1 ablation_06_no_attnbias_p2 \
    ablation_07a_3m_p1 ablation_07a_3m_p2 \
    ablation_07b_30m_p1 ablation_07b_30m_p2 \
    ablation_07c_100m_p1 ablation_07c_100m_p2 \
; do
    ckpt="$REPO_ROOT/outputs/ablation/$exp/checkpoints/best.pt"
    if [ -f "$ckpt" ]; then
        echo "  ✅ $exp"
    else
        echo "  ❌ $exp (missing)"
    fi
done

say ""
say "다음 단계 — Downstream eval:"
say "  python -m scripts.run_ablation_downstream --nproc $NPROC"
