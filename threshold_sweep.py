# threshold_sweep.py
"""
Tìm threshold tối ưu cho change_logits.
Hiện tại dùng threshold=0.0 (tương đương prob=0.5) -> Recall thấp.
Quét qua các ngưỡng logit khác nhau để tìm điểm cân bằng F1 tối ưu.
"""
import sys
import os
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

# Thêm path để import từ SECOND-OC/eval/
sys.path.append(str(Path(__file__).resolve().parent / "SECOND-OC" / "eval"))
try:
    from object_eval import match_predictions, match_semantic, prf1
except ImportError:
    # Fallback copy of the functions in case of path issues
    def bbox_iou(a: list, b: list) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)

    def match_predictions(gt_list: list[dict], pred_list: list[dict], iou_thresh: float) -> tuple[int, int, int]:
        gt_changed   = [g for g in gt_list  if g["change_type"] != "unchanged"]
        pred_changed = [p for p in pred_list if p.get("change_type", "") != "unchanged"]
        matched_gt = set()
        tp = 0
        for pred in pred_changed:
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gt_changed):
                if i in matched_gt:
                    continue
                iou = bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_iou >= iou_thresh and best_idx not in matched_gt:
                tp += 1
                matched_gt.add(best_idx)
        fp = len(pred_changed) - tp
        fn = len(gt_changed)   - tp
        return tp, fp, fn

    def match_semantic(gt_list: list[dict], pred_list: list[dict], iou_thresh: float) -> tuple[int, int, int]:
        gt_changed   = [g for g in gt_list  if g["change_type"] != "unchanged"]
        pred_changed = [p for p in pred_list if p.get("change_type", "") != "unchanged"]
        matched_gt = set()
        tp = 0
        for pred in pred_changed:
            best_iou, best_idx = 0.0, -1
            for i, gt in enumerate(gt_changed):
                if i in matched_gt:
                    continue
                iou = bbox_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if (best_iou >= iou_thresh
                    and best_idx not in matched_gt
                    and pred.get("class_T2") == gt_changed[best_idx]["class_name_T2"]):
                tp += 1
                matched_gt.add(best_idx)
        fp = len(pred_changed) - tp
        fn = len(gt_changed)   - tp
        return tp, fp, fn

    def prf1(tp: int, fp: int, fn: int) -> dict:
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4), "TP": tp, "FP": fp, "FN": fn}

GT_PATH = Path("SECOND-OC/annotations/change_annotations.json")
TOKEN_DIR = Path("output/spectral_transition_preds/tokens")

CLASS_NAMES = {
    0: "background",
    1: "tree",
    2: "buildings",
    3: "water",
    4: "non_veg_ground",
    5: "playground",
    6: "low_vegetation",
}
BG_CLASSES = {0, 4, 6}
NUM_CLASSES = 7

def infer_change_type(c1: int, c2: int, change_pred: bool) -> str:
    if not change_pred:
        return "unchanged"
    if c1 in BG_CLASSES and c2 not in BG_CLASSES:
        return "appeared"
    if c1 not in BG_CLASSES and c2 in BG_CLASSES:
        return "disappeared"
    if c1 == c2:
        return "unchanged"
    return "semantic_change"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--token-dir", default="output/spectral_transition_preds/tokens")
    parser.add_argument("--gt-json", default="SECOND-OC/annotations/change_annotations.json")
    args = parser.parse_args()

    token_dir = Path(args.token_dir)
    gt_path = Path(args.gt_json)

    if not token_dir.exists():
        print(f"[ERROR] Token predictions directory not found at: {token_dir}")
        return

    # Load GT data
    print("Loading GT change annotations...")
    with open(gt_path) as f:
        gt_data = json.load(f)

    # Pre-load token predictions to memory for speed
    print("Loading token predictions into memory...")
    token_preds_by_stem = {}
    for token_file in tqdm(list(token_dir.glob("*.json"))):
        stem = token_file.stem
        token_preds = json.loads(token_file.read_text())
        token_preds_by_stem[stem] = {t["token_idx"]: t for t in token_preds}

    # Range of logit thresholds to sweep
    # Since logit = log(p/(1-p)), 0.0 corresponds to prob=0.5.
    # We sweep from -6.0 to +2.0
    thresholds = [-6.0, -5.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]

    print(f"\n{'Thresh':>7} | {'Bin-P':>7} {'Bin-R':>7} {'Bin-F1':>7} | {'Sem-P':>7} {'Sem-R':>7} {'Sem-F1':>7} | {'SemAcc-TP':>10}")
    print("-" * 88)

    best_bin_f1 = 0
    best_bin_thresh = 0.0
    best_sem_f1 = 0
    best_sem_thresh = 0.0

    for thresh in thresholds:
        # Generate predictions for this threshold
        preds_by_stem = defaultdict(list)
        
        for stem, gt_changes in gt_data["changes"].items():
            by_idx = token_preds_by_stem.get(stem)
            if by_idx is None:
                continue

            for ch in gt_changes:
                parts = ch["change_id"].split("_")
                try:
                    mask_idx = int(parts[2])
                except (IndexError, ValueError):
                    continue

                tp = by_idx.get(mask_idx)
                if tp is None:
                    continue

                if tp.get("transition_pred") is not None:
                    trans = tp["transition_pred"]
                    c1 = trans // NUM_CLASSES
                    c2 = trans % NUM_CLASSES
                else:
                    c1 = tp["class_T1"]
                    c2 = tp["class_T2"]

                # Apply threshold on logit
                change_logit = tp["change_logit"]
                change_pred = change_logit > thresh

                change_type = infer_change_type(c1, c2, change_pred)
                class_T2_name = CLASS_NAMES.get(c2, f"class_{c2}")

                preds_by_stem[stem].append({
                    "stem":        stem,
                    "change_id":   ch["change_id"],
                    "change_type": change_type,
                    "class_T2":    class_T2_name,
                    "bbox":        ch["bbox"],
                })

        # Evaluate
        bin_tp = bin_fp = bin_fn = 0
        sem_tp = sem_fp = sem_fn = 0

        for stem, gt_changes in gt_data["changes"].items():
            preds = preds_by_stem.get(stem, [])
            # We evaluate on covered stems (native matching)
            pred_ids_this_stem = {p.get("change_id") for p in preds if "change_id" in p}
            if not pred_ids_this_stem:
                continue
            
            gt_covered = [g for g in gt_changes if g["change_id"] in pred_ids_this_stem]
            if not gt_covered:
                continue

            a, b, c = match_predictions(gt_covered, preds, 0.5)
            bin_tp += a; bin_fp += b; bin_fn += c

            a, b, c = match_semantic(gt_covered, preds, 0.5)
            sem_tp += a; sem_fp += b; sem_fn += c

        bin_res = prf1(bin_tp, bin_fp, bin_fn)
        sem_res = prf1(sem_tp, sem_fp, sem_fn)
        sem_acc = sem_tp / max(bin_tp, 1)

        print(f"{thresh:>7.2f} | {bin_res['P']:>7.4f} {bin_res['R']:>7.4f} {bin_res['F1']:>7.4f} | "
              f"{sem_res['P']:>7.4f} {sem_res['R']:>7.4f} {sem_res['F1']:>7.4f} | {sem_acc:>10.2%}")

        if bin_res['F1'] > best_bin_f1:
            best_bin_f1 = bin_res['F1']
            best_bin_thresh = thresh
        
        if sem_res['F1'] > best_sem_f1:
            best_sem_f1 = sem_res['F1']
            best_sem_thresh = thresh

    print(f"\n→ Best Binary-F1 threshold: {best_bin_thresh:.2f} (F1={best_bin_f1:.4f})")
    print(f"→ Best Semantic-F1 threshold: {best_sem_thresh:.2f} (F1={best_sem_f1:.4f})")

if __name__ == "__main__":
    main()
