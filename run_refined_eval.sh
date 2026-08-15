#!/bin/bash
# run_refined_eval.sh — Run evaluation on SECOND-OC benchmark for the refined model (Stage 10).
set -e

CKPT="SECOND/stage10_refined_v2/best_model.pt"
PRED_DIR="output/stage10_refined_preds"
OUT_PRED="SECOND-OC/predictions/predictions_stage10_refined.json"
OUT_RESULT="SECOND-OC/baseline_results/stage10_refined_results.json"
OUT_PRED_TOKEN="SECOND-OC/predictions/predictions_stage10_refined_token.json"
OUT_RESULT_TOKEN="SECOND-OC/baseline_results/stage10_refined_results_token.json"

echo "=== Refined Seg Stage 10 Evaluation Pipeline ==="

# 1. Run inference
echo "Running Token-MoE inference with refined masks..."
mkdir -p "$PRED_DIR/im1" "$PRED_DIR/im2"
cs2_venv/bin/python eval_test_set.py \
    --checkpoint "$CKPT" \
    --tokens_T1  SECOND/tokens_T1_test_v3 \
    --tokens_T2  SECOND/tokens_T2_test_v3 \
    --masks_T1   SECOND/sam2_masks_T1_test \
    --masks_T2   SECOND/sam2_masks_T2_test \
    --matches    SECOND/matches_test \
    --batch_size 8 \
    --device cuda \
    --save-preds "$PRED_DIR" \
    --use_spectral

# 2. Pixel-level object evaluation
echo "=== 1. PIXEL-RECONSTRUCTION EVALUATION ==="
echo "Converting pixel maps to object predictions..."
cs2_venv/bin/python SECOND-OC/baselines/pixel_to_object_predictions.py \
    --pred-dir   "$PRED_DIR" \
    --model-name TokenMoE \
    --gt-json    SECOND-OC/annotations/change_annotations.json \
    --out        "$OUT_PRED"

echo "Evaluating on SECOND-OC benchmark (Pixel-level)..."
cs2_venv/bin/python SECOND-OC/eval/object_eval.py \
    --gt   SECOND-OC/annotations/change_annotations.json \
    --pred "$OUT_PRED" \
    --iou-threshold 0.5 \
    --out  "$OUT_RESULT"

# 3. Token-level native evaluation
echo "=== 2. TOKEN-NATIVE EVALUATION ==="
echo "Converting token predictions to object predictions..."
cs2_venv/bin/python SECOND-OC/baselines/token_to_object_predictions.py \
    --token-dir  "$PRED_DIR/tokens" \
    --gt-json    SECOND-OC/annotations/change_annotations.json \
    --out        "$OUT_PRED_TOKEN"

echo "Evaluating on SECOND-OC benchmark (Token-level)..."
cs2_venv/bin/python SECOND-OC/eval/object_eval.py \
    --gt   SECOND-OC/annotations/change_annotations.json \
    --pred "$OUT_PRED_TOKEN" \
    --iou-threshold 0.5 \
    --out  "$OUT_RESULT_TOKEN"

echo ""
echo "=== Evaluation Done ==="
echo "--- Pixel-level Results ---"
cat "$OUT_RESULT"
echo "--- Token-level Results ---"
cat "$OUT_RESULT_TOKEN"
