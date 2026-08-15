#!/bin/bash
set -e
cd "/home/chung/RS/Image Segmentation"

echo "[1/4] Running eval_test_set to generate token-level predictions..."
cs2_venv/bin/python eval_test_set.py \
    --checkpoint SECOND/stage_rc_combined/best_model.pt \
    --tokens_T1  SECOND/tokens_T1_test_rc \
    --tokens_T2  SECOND/tokens_T2_test_rc \
    --matches    SECOND/matches_test \
    --use_spectral \
    --device cuda \
    --save-preds SECOND/predictions_rc

echo "[2/4] Converting token preds to object-level (with NN fallback, threshold sweep)..."
mkdir -p SECOND/predictions_rc_oc

for THRESH in 0.0 0.5 1.0 1.5 1.8 2.0 2.5; do
    cs2_venv/bin/python SECOND-OC/baselines/token_to_object_predictions.py \
        --token-dir SECOND/predictions_rc/tokens \
        --gt-json   SECOND-OC/annotations/change_annotations.json \
        --out       SECOND/predictions_rc_oc/thresh_${THRESH}.json \
        --threshold ${THRESH} \
        --tokens-pt-dir SECOND/tokens_T1_test_rc
    echo "  Threshold ${THRESH} done"
done

echo "[3/4] Evaluating each threshold..."
for THRESH in 0.0 0.5 1.0 1.5 1.8 2.0 2.5; do
    echo "=== Threshold ${THRESH} ==="
    cs2_venv/bin/python SECOND-OC/eval/object_eval.py \
        --pred SECOND/predictions_rc_oc/thresh_${THRESH}.json \
        --gt   SECOND-OC/annotations/change_annotations.json
done

echo "[4/4] Done!"
