"""
object_eval.py — Evaluate model predictions on the SECOND-OC benchmark.

Metrics:
    Binary-Object-F1 : changed vs unchanged, matched by bbox IoU >= iou_thresh
    Semantic-Object-F1 : additionally requires correct class_T2 prediction

Usage:
    python SECOND-OC/eval/object_eval.py \
        --gt  SECOND-OC/annotations/change_annotations.json \
        --pred predictions.json \
        --iou-threshold 0.5

Prediction format (JSON list):
    [
      {
        "stem":        "00004",
        "change_type": "semantic_change",   # or "unchanged"/"appeared"/"disappeared"
        "class_T2":    "buildings",         # required for semantic_change
        "bbox":        [x1, y1, x2, y2]    # pixel coords, same as GT bbox
      },
      ...
    ]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def bbox_iou(a: list, b: list) -> float:
    """Compute IoU of two [x1, y1, x2, y2] bboxes."""
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


def match_predictions(gt_list: list[dict],
                      pred_list: list[dict],
                      iou_thresh: float) -> tuple[int, int, int]:
    """
    Greedy matching of predictions to GT instances by bbox IoU.
    Returns (tp, fp, fn) for binary change detection.
    """
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


def match_semantic(gt_list: list[dict],
                   pred_list: list[dict],
                   iou_thresh: float) -> tuple[int, int, int]:
    """
    Like match_predictions but TP additionally requires correct class_T2.
    Only considers non-unchanged instances.
    """
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
    return {"P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
            "TP": tp, "FP": fp, "FN": fn}


def evaluate(gt_path: str, pred_path: str, iou_thresh: float = 0.5) -> dict:
    with open(gt_path) as f:
        gt_data = json.load(f)
    with open(pred_path) as f:
        pred_data = json.load(f)

    # Group predictions by stem
    preds_by_stem: dict = defaultdict(list)
    for p in pred_data:
        preds_by_stem[p["stem"]].append(p)

    bin_tp = bin_fp = bin_fn = 0
    sem_tp = sem_fp = sem_fn = 0

    for stem, gt_changes in gt_data["changes"].items():
        preds = preds_by_stem.get(stem, [])

        a, b, c = match_predictions(gt_changes, preds, iou_thresh)
        bin_tp += a; bin_fp += b; bin_fn += c

        a, b, c = match_semantic(gt_changes, preds, iou_thresh)
        sem_tp += a; sem_fp += b; sem_fn += c

    results = {
        "iou_threshold":     iou_thresh,
        "Binary-Object-F1":  prf1(bin_tp, bin_fp, bin_fn),
        "Semantic-Object-F1": prf1(sem_tp, sem_fp, sem_fn),
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="SECOND-OC object-level evaluation")
    parser.add_argument("--gt",            required=True, help="GT change_annotations.json")
    parser.add_argument("--pred",          required=True, help="Predictions JSON")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--out",           default=None, help="Save results JSON")
    args = parser.parse_args()

    results = evaluate(args.gt, args.pred, args.iou_threshold)

    print(f"\n── SECOND-OC Evaluation (IoU ≥ {args.iou_threshold}) ─────────────")
    for metric, vals in results.items():
        if isinstance(vals, dict):
            print(f"   {metric:<25} P={vals['P']:.4f}  R={vals['R']:.4f}  "
                  f"F1={vals['F1']:.4f}  "
                  f"(TP={vals['TP']} FP={vals['FP']} FN={vals['FN']})")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n   Saved: {args.out}")

    return results


if __name__ == "__main__":
    main()
