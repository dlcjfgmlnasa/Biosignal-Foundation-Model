#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────
# Ablation full sweep — 옵션 C (Phase 1 reuse 공유) 자동 실행 스크립트.
#
# Stage 0 : Base P1                                      (~42h)
# Stage 1 : Base P2  (Base P1 ckpt 사용)                  (~17h)
# Stage 2 : reuse 그룹 — 01c / 01d / 03a / 03b / 04       (~70h, P2 만)
# Stage 3 : 작은 loss/arch — 01a / 01b / 02 / 06         (~190h, P1+P2)
# Stage 4 : Sampling — 05a / 05b                         (~95h, P1+P2)
# Stage 5 : Model scale — 07a → 07b → 07c (작은 → 큰)    (~240h, P1+P2)
#                                                ──────────────
#                                                TOTAL ~654h ≈ 27일
#
# 전제:
#   * configs/ablation/variants.yaml 의 00_full 이 base exp_name 가리킴
#     (ablation_phase1_base / ablation_phase2_base)
#   * 4-GPU L40S 환경, NPROC=4 권장
#   * Base P1 학습부터 처음부터 진행
#
# Usage:
#   # foreground (보면서)
#   bash scripts/run_ablation_sweep_optionC.sh
#
#   # background (재접속 안전, 권장)
#   mkdir -p logs/ablation_sweep
#   nohup bash scripts/run_ablation_sweep_optionC.sh \
#       > logs/ablation_sweep/_master.log 2>&1 &
#   tail -f logs/ablation_sweep/_master.log
#
#   # 특정 stage 부터 재개 (예: Base P1 끝났으면 Stage 1 부터)
#   START_STAGE=1 bash scripts/run_ablation_sweep_optionC.sh
#
#   # 마지막 stage 까지 (예: Stage 4 까지만, model scale 제외)
#   STOP_STAGE=4 bash scripts/run_ablation_sweep_optionC.sh
# ─────────────────────────────────────────────────────────────────
set -uo pipefail
# set -e 는 일부러 끔 — 한 variant 실패해도 다음 진행

# ─── 환경 변수 (override 가능) ──────────────────────────────────
REPO_ROOT="${REPO_ROOT:-$HOME/workspace/k-mimic-/bio_fm}"
NPROC="${NPROC:-4}"
START_STAGE="${START_STAGE:-0}"
STOP_STAGE="${STOP_STAGE:-5}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/ablation_sweep}"
# 실제 ckpt 파일은 checkpoint_{phase}_epoch{NNN}_best.pt 형식이므로 glob 사용
BASE_P1_CKPT_DIR="$REPO_ROOT/outputs/ablation/ablation_phase1_base/checkpoints"
BASE_P2_CKPT_DIR="$REPO_ROOT/outputs/ablation/ablation_phase2_base/checkpoints"

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
    # glob 패턴으로 *_best.pt 최신 파일 검증
    local ckpt_dir="$1"
    local desc="$2"
    if [ ! -d "$ckpt_dir" ]; then
        say "❌ $desc ckpt 디렉토리 없음: $ckpt_dir"
        say "   해당 단계를 먼저 완주한 후 실행하세요."
        return 1
    fi
    # ls -t 로 최신 *_best.pt 찾기
    local latest
    latest=$(ls -t "$ckpt_dir"/*_best.pt 2>/dev/null | head -1)
    if [ -z "$latest" ] || [ ! -f "$latest" ]; then
        say "❌ $desc *_best.pt 파일 없음 in: $ckpt_dir"
        say "   디렉토리 내용:"
        ls -la "$ckpt_dir" 2>/dev/null | sed 's/^/     /' | head -20
        return 1
    fi
    say "✅ $desc ckpt: $(basename "$latest") ($(du -h "$latest" | cut -f1))"
    return 0
}
ckpt_exists() {
    # require_ckpt 의 silent 버전 — boolean only
    local ckpt_dir="$1"
    [ -d "$ckpt_dir" ] || return 1
    local latest
    latest=$(ls -t "$ckpt_dir"/*_best.pt 2>/dev/null | head -1)
    [ -n "$latest" ] && [ -f "$latest" ]
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
in_range() {
    # $1: stage_num, return 0 if START_STAGE <= $1 <= STOP_STAGE
    [ "$1" -ge "$START_STAGE" ] && [ "$1" -le "$STOP_STAGE" ]
}

# ─── 환경 점검 ────────────────────────────────────────────────
section "Pre-check"
say "REPO_ROOT  = $REPO_ROOT"
say "NPROC      = $NPROC"
say "START_STAGE= $START_STAGE"
say "STOP_STAGE = $STOP_STAGE"
say "LOG_DIR    = $LOG_DIR"

nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader 2>/dev/null \
    | head -n "$NPROC" \
    | while read line; do say "GPU $line"; done

# Stage 1+ 진입 시 Base P1 ckpt 사전 점검 (Stage 0 skip 또는 외부 학습 케이스)
if [ "$START_STAGE" -ge 1 ]; then
    say ""
    say "START_STAGE=$START_STAGE → Stage 0 (Base P1) skip, ckpt 존재 확인:"
    require_ckpt "$BASE_P1_CKPT_DIR" "Base P1" || exit 1
fi

# Reuse chain dry-run 검증 (Stage 2 진입 전 1회)
if in_range 2; then
    section "Dry-run — reuse chain 검증"
    python -m scripts.run_ablation --dry-run \
        --variant 01c_no_crossnext \
        2>&1 | grep -E "reuse|Phase 1|Phase 2" | head -10 || true
fi

# ─── Stage 0 — Base P1 ────────────────────────────────────────
if in_range 0; then
    if ckpt_exists "$BASE_P1_CKPT_DIR"; then
        say "✅ Stage 0 skip — Base P1 ckpt 이미 존재"
        require_ckpt "$BASE_P1_CKPT_DIR" "Base P1" || true
    else
        run_stage 0 "base_p1" \
            torchrun --nproc_per_node="$NPROC" \
                -m train.1_channel_independency \
                --config configs/ablation/_phase1_base.yaml
        require_ckpt "$BASE_P1_CKPT_DIR" "Base P1" || {
            say "❌ Base P1 학습 후에도 ckpt 미생성 — sweep 중단"
            exit 1
        }
    fi
fi

# ─── Stage 1 — Base P2 ────────────────────────────────────────
if in_range 1; then
    if ckpt_exists "$BASE_P2_CKPT_DIR"; then
        say "✅ Stage 1 skip — Base P2 ckpt 이미 존재"
        require_ckpt "$BASE_P2_CKPT_DIR" "Base P2" || true
    else
        run_stage 1 "base_p2" \
            torchrun --nproc_per_node="$NPROC" \
                -m train.2_any_variate \
                --config configs/ablation/_phase2_base.yaml
        require_ckpt "$BASE_P2_CKPT_DIR" "Base P2" || {
            say "⚠️  Base P2 미완 — reuse 그룹 Stage 2 는 Base P1 만 사용하므로 진행 가능"
        }
    fi
fi

# ─── Stage 2 — reuse 그룹 (P2 만, base P1 ckpt 재사용) ────────
if in_range 2; then
    run_stage 2 "reuse" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 01c_no_crossnext \
            --variant 01d_no_contrastive \
            --variant 03a_samevar_only \
            --variant 03b_cross_only \
            --variant 04_phase1_only
fi

# ─── Stage 3 — 작은 loss/arch ablation (P1+P2) ────────────────
if in_range 3; then
    run_stage 3 "loss_arch" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 01a_no_mpm \
            --variant 01b_no_next \
            --variant 02_no_lscnorm \
            --variant 06_no_attnbias
fi

# ─── Stage 4 — Modality-balanced sampling (P1+P2) ─────────────
if in_range 4; then
    run_stage 4 "sampling" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --group sampling
fi

# ─── Stage 5 — Model scale (작은 → 큰) ────────────────────────
if in_range 5; then
    # 5a: 3M (제일 빠름, ~25h)
    run_stage 5a "scale_3m" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 07a_size_3m

    # 5b: 30M (~75h)
    run_stage 5b "scale_30m" \
        python -m scripts.run_ablation \
            --nproc "$NPROC" \
            --skip-existing \
            --variant 07b_size_30m

    # 5c: 100M (제일 무거움, ~140h ≈ 6일)
    run_stage 5c "scale_100m" \
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
    ckpt_dir="$REPO_ROOT/outputs/ablation/$exp/checkpoints"
    latest=$(ls -t "$ckpt_dir"/*_best.pt 2>/dev/null | head -1)
    if [ -n "$latest" ] && [ -f "$latest" ]; then
        echo "  ✅ $exp — $(basename "$latest")"
    else
        echo "  ❌ $exp (missing)"
    fi
done

say ""
say "다음 단계 — Downstream eval:"
say "  python -m scripts.run_ablation_downstream --nproc $NPROC"
