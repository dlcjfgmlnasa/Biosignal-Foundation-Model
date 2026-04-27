#!/usr/bin/env bash
# AdaLN ablation runner (Biosignal-FM2 server, 2xL40S)
# Runs baseline → AdaLN sequentially, then prints comparison table.
#
# Usage:
#   bash scripts/run_adaln_ablation.sh
#
# Estimated wall-clock: ~2-3h per run on 2xL40S (6 epochs × 2000 batches).
# Logs: outputs/ablation_adaln/{baseline,adaln_dc16}/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

NPROC="${NPROC:-2}"
LOG_DIR="$REPO_ROOT/outputs/ablation_adaln"
mkdir -p "$LOG_DIR"

run() {
    local tag="$1"
    local cfg="$2"
    echo "============================================================"
    echo "[$(date '+%F %T')] START: $tag  ($cfg)"
    echo "============================================================"
    torchrun --nproc_per_node="$NPROC" -m train.1_channel_independency \
        --config "$cfg" \
        2>&1 | tee "$LOG_DIR/${tag}.stdout.log"
    echo "[$(date '+%F %T')] DONE:  $tag"
}

run baseline   configs/phase1_vital_ablation_baseline.yaml
run adaln_dc16 configs/phase1_vital_ablation_adaln.yaml

echo
echo "============================================================"
echo "Comparison summary"
echo "============================================================"
python scripts/compare_adaln_ablation.py \
    --baseline outputs/ablation_adaln/baseline/training_log.csv \
    --adaln    outputs/ablation_adaln/adaln_dc16/training_log.csv