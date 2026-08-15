#!/bin/bash
# run_tokenmoe_eval.sh — Wait for Token-MoE training to finish, then eval on SECOND-OC.
# Usage: bash run_tokenmoe_eval.sh

set -e
LOG=/tmp/tokenmoe_eval.log
exec > >(tee -a "$LOG") 2>&1

TRAIN_PID=771090
CKPT=SECOND/stage5_6_scd/best_model.pt
PRED_DIR=output/tokenmoe_preds
OUT_PRED=SECOND-OC/predictions/predictions_tokenmoe.json
OUT_RESULT=SECOND-OC/baseline_results/tokenmoe_results.json

echo "=== Token-MoE Eval Pipeline  $(date) ==="

# ── 1. Wait for training to finish ───────────────────────────────────────────
echo "Waiting for training PID $TRAIN_PID to finish..."
while kill -0 $TRAIN_PID 2>/dev/null; do
    EPOCH=$(tail -1 SECOND/stage5_6_scd/training_log.csv | cut -d',' -f1)
    F1=$(tail -1 SECOND/stage5_6_scd/training_log.csv | cut -d',' -f8)
    echo "  Epoch $EPOCH  val_f1=$F1  $(date +%H:%M:%S)"
    sleep 60
done
echo "Training complete."

# ── 2. Run inference → pixel maps ────────────────────────────────────────────
mkdir -p "$PRED_DIR/im1" "$PRED_DIR/im2"
echo "Running Token-MoE inference..."
python3.11 eval_test_set.py \
    --checkpoint "$CKPT" \
    --tokens_T1  SECOND/tokens_T1_test \
    --tokens_T2  SECOND/tokens_T2_test \
    --matches    SECOND/matches_test \
    --batch_size 8 \
    --device cuda \
    --save-preds "$PRED_DIR"

# ── 3. Convert pixel maps → object predictions ───────────────────────────────
echo "Converting to object predictions..."
python3.11 SECOND-OC/baselines/pixel_to_object_predictions.py \
    --pred-dir   "$PRED_DIR" \
    --model-name TokenMoE \
    --gt-json    SECOND-OC/annotations/change_annotations.json \
    --out        "$OUT_PRED"

# ── 4. Evaluate on SECOND-OC ─────────────────────────────────────────────────
mkdir -p SECOND-OC/baseline_results
echo "Evaluating on SECOND-OC benchmark..."
python3.11 SECOND-OC/eval/object_eval.py \
    --gt   SECOND-OC/annotations/change_annotations.json \
    --pred "$OUT_PRED" \
    --iou-threshold 0.5 \
    --out  "$OUT_RESULT"

echo ""
echo "=== Done. Results saved to $OUT_RESULT ==="
echo "ChangeStar2 baseline for comparison:"
cat SECOND-OC/baseline_results/changestar2_results.json
