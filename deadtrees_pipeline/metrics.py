"""Reusable object, boundary, and structural segmentation metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import find_boundaries


@dataclass(frozen=True)
class MatchResult:
    pairs: list[tuple[int, int, float]]
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float


def mask_bboxes(masks: np.ndarray) -> list[tuple[int, int, int, int] | None]:
    boxes: list[tuple[int, int, int, int] | None] = []
    for mask in masks:
        ys, xs = np.where(mask)
        if len(xs) == 0:
            boxes.append(None)
        else:
            boxes.append((int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())))
    return boxes


def overlap_matrices(
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return IoU, intersections, intersection/GT, intersection/pred."""
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    intersections = np.zeros((n_gt, n_pred), dtype=np.int64)
    gt_areas = gt_masks.reshape(n_gt, -1).sum(axis=1) if n_gt else np.zeros(0)
    pred_areas = pred_masks.reshape(n_pred, -1).sum(axis=1) if n_pred else np.zeros(0)
    gt_boxes = mask_bboxes(gt_masks)
    pred_boxes = mask_bboxes(pred_masks)

    for i, gt_box in enumerate(gt_boxes):
        if gt_box is None:
            continue
        gx1, gy1, gx2, gy2 = gt_box
        for j, pred_box in enumerate(pred_boxes):
            if pred_box is None:
                continue
            px1, py1, px2, py2 = pred_box
            x1, y1 = max(gx1, px1), max(gy1, py1)
            x2, y2 = min(gx2, px2), min(gy2, py2)
            if x2 < x1 or y2 < y1:
                continue
            intersections[i, j] = np.logical_and(
                gt_masks[i, y1:y2 + 1, x1:x2 + 1],
                pred_masks[j, y1:y2 + 1, x1:x2 + 1],
            ).sum()

    unions = gt_areas[:, None] + pred_areas[None, :] - intersections
    iou = np.divide(
        intersections,
        unions,
        out=np.zeros_like(intersections, dtype=np.float64),
        where=unions > 0,
    )
    overlap_gt = np.divide(
        intersections,
        gt_areas[:, None],
        out=np.zeros_like(intersections, dtype=np.float64),
        where=gt_areas[:, None] > 0,
    )
    overlap_pred = np.divide(
        intersections,
        pred_areas[None, :],
        out=np.zeros_like(intersections, dtype=np.float64),
        where=pred_areas[None, :] > 0,
    )
    return iou, intersections, overlap_gt, overlap_pred


def hungarian_match(iou: np.ndarray, threshold: float) -> MatchResult:
    n_gt, n_pred = iou.shape
    if n_gt == 0 or n_pred == 0:
        tp = 0
        fp = n_pred
        fn = n_gt
        return MatchResult([], tp, fp, fn, 0.0, 0.0, 0.0)

    # A large valid-edge bonus maximizes TP cardinality first, total IoU second.
    valid = iou >= threshold
    score = valid.astype(np.float64) * (min(n_gt, n_pred) + 1.0) + iou
    rows, cols = linear_sum_assignment(-score)
    pairs = [
        (int(i), int(j), float(iou[i, j]))
        for i, j in zip(rows, cols)
        if iou[i, j] >= threshold
    ]
    tp = len(pairs)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MatchResult(pairs, tp, fp, fn, precision, recall, f1)


def structural_errors(
    intersections: np.ndarray,
    gt_areas: np.ndarray,
    significant_fraction: float = 0.10,
) -> dict:
    """Count split/merge relations before one-to-one matching.

    A relation is significant when a prediction covers at least the requested
    fraction of a GT instance.  Using GT-normalized overlap detects both a GT
    split into several proposals and one large proposal merging several GTs.
    """
    if len(gt_areas) == 0:
        relations = np.zeros_like(intersections, dtype=bool)
    else:
        relations = intersections >= (gt_areas[:, None] * significant_fraction)
    preds_per_gt = relations.sum(axis=1) if relations.size else np.zeros(len(gt_areas), dtype=int)
    gts_per_pred = relations.sum(axis=0) if relations.size else np.zeros(intersections.shape[1], dtype=int)
    split_gt = int((preds_per_gt >= 2).sum())
    merge_pred = int((gts_per_pred >= 2).sum())
    return {
        "split_gt_count": split_gt,
        "split_rate": split_gt / len(gt_areas) if len(gt_areas) else 0.0,
        "merge_pred_count": merge_pred,
        "merge_rate": merge_pred / intersections.shape[1] if intersections.shape[1] else 0.0,
        "uncovered_gt_count": int((preds_per_gt == 0).sum()),
        "unrelated_pred_count": int((gts_per_pred == 0).sum()),
    }


def boundary_metrics(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    tolerance_px: float = 3.0,
) -> dict:
    gt_boundary = find_boundaries(gt_mask, mode="inner")
    pred_boundary = find_boundaries(pred_mask, mode="inner")
    if not gt_boundary.any() or not pred_boundary.any():
        return {"boundary_f1": 0.0, "assd_px": float("inf"), "hd95_px": float("inf")}

    dist_to_gt = ndimage.distance_transform_edt(~gt_boundary)
    dist_to_pred = ndimage.distance_transform_edt(~pred_boundary)
    pred_distances = dist_to_gt[pred_boundary]
    gt_distances = dist_to_pred[gt_boundary]

    precision = float((pred_distances <= tolerance_px).mean())
    recall = float((gt_distances <= tolerance_px).mean())
    bf1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    all_distances = np.concatenate([pred_distances, gt_distances])
    return {
        "boundary_f1": bf1,
        "assd_px": float((pred_distances.mean() + gt_distances.mean()) / 2),
        "hd95_px": float(np.percentile(all_distances, 95)),
    }


def average_precision(
    predictions: list[tuple[str, float, np.ndarray]],
    gt_counts: dict[str, int],
    iou_threshold: float,
) -> float:
    """Class-agnostic AP for precomputed per-image IoU vectors.

    Each prediction tuple contains (stem, confidence, IoU-to-all-GTs-in-stem).
    """
    total_gt = sum(gt_counts.values())
    if total_gt == 0:
        return 0.0
    matched = {stem: set() for stem in gt_counts}
    tp, fp = [], []
    for stem, score, ious in sorted(predictions, key=lambda item: item[1], reverse=True):
        del score
        candidates = [
            (float(iou), index) for index, iou in enumerate(ious)
            if iou >= iou_threshold and index not in matched.setdefault(stem, set())
        ]
        if candidates:
            _, best_index = max(candidates)
            matched[stem].add(best_index)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / total_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return float(np.mean([
        precisions[recalls >= recall_level].max() if np.any(recalls >= recall_level) else 0.0
        for recall_level in np.linspace(0.0, 1.0, 101)
    ]))

