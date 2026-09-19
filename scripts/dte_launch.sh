#!/usr/bin/env bash
# dte_launch.sh — launch G0 then G1 as background daemons
#
# Usage:
#   chmod +x scripts/dte_launch.sh
#   BENCH=/path/to/DTE-aerial-bench bash scripts/dte_launch.sh
#
# Both processes run independently and log to dte_runs/logs/.
# Check status: tail -f dte_runs/logs/g0_parity.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BENCH="${BENCH:-$ROOT/datasets/DTE-aerial-bench}"
META="$BENCH/DTE-aerial-bench-meta.csv"
CKPT="$ROOT/DTE-aerial-model/DTE_aerial_model.safetensors"
CFG="$ROOT/DTE-aerial-official/config/evaluation.yml"

G0_OUT="$ROOT/dte_runs/g0_parity"
G1_OUT="$ROOT/dte_runs/g1_diagnosis"
LOG_DIR="$ROOT/dte_runs/logs"

mkdir -p "$LOG_DIR" "$G0_OUT" "$G1_OUT"

echo "=== DTE pipeline launcher ==="
echo "  BENCH : $BENCH"
echo "  CKPT  : $CKPT"
echo "  G0_OUT: $G0_OUT"
echo "  G1_OUT: $G1_OUT"
echo ""

# ──────────────────────────────────────────────
# Guard: bench must exist
# ──────────────────────────────────────────────
if [ ! -d "$BENCH" ]; then
    echo "ERROR: DTE-aerial-bench not found at $BENCH"
    echo "Set BENCH=/path/to/bench or download first."
    exit 1
fi

if [ ! -f "$META" ]; then
    echo "ERROR: metadata CSV not found at $META"
    exit 1
fi

# ──────────────────────────────────────────────
# Launch G0 (QC + inference + parity)
# ──────────────────────────────────────────────
echo "Launching G0 parity …"
nohup python "$ROOT/scripts/dte_g0_parity.py" \
    --bench  "$BENCH" \
    --meta   "$META" \
    --ckpt   "$CKPT" \
    --cfg    "$CFG" \
    --out    "$G0_OUT" \
    > "$LOG_DIR/g0_parity.log" 2>&1 &
G0_PID=$!
echo "$G0_PID" > "$LOG_DIR/g0_parity.pid"
echo "  G0 PID: $G0_PID  |  log: $LOG_DIR/g0_parity.log"

# ──────────────────────────────────────────────
# Wait for G0, then chain G1
# ──────────────────────────────────────────────
(
    wait "$G0_PID"
    G0_EXIT=$?
    if [ "$G0_EXIT" -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] G0 PASS — launching G1 diagnosis …" >> "$LOG_DIR/g0_parity.log"
        nohup python "$ROOT/scripts/dte_g1_diagnose.py" \
            --bench "$BENCH" \
            --meta  "$META" \
            --preds "$G0_OUT/predictions" \
            --out   "$G1_OUT" \
            > "$LOG_DIR/g1_diagnose.log" 2>&1 &
        G1_PID=$!
        echo "$G1_PID" > "$LOG_DIR/g1_diagnose.pid"
        echo "  G1 PID: $G1_PID  |  log: $LOG_DIR/g1_diagnose.log"
    else
        echo "G0 FAILED (exit $G0_EXIT). G1 not started. Fix issues in $LOG_DIR/g0_parity.log"
    fi
) &

echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/g0_parity.log"
echo "  tail -f $LOG_DIR/g1_diagnose.log"
echo ""
echo "After G1 completes:"
echo "  cat $G1_OUT/minimal_diagnosis.md"
echo "  ls  $G1_OUT/figures/"
echo ""
echo "Then proceed to G2 (method design) — NO additional baseline runs required."
