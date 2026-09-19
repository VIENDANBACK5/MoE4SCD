"""True interpolated COCO average precision for instance masks.

The repo's `evaluate_metrics` reports the F1 score at a fixed IoU threshold but names
it `ap50`. That is a legitimate quantity and every comparison made with it is
internally consistent, but it is not average precision: it ignores confidence ranking
entirely, so it cannot reward a model for putting its mistakes at the bottom of the
list, and it is not comparable to any published COCO number.

This adds the real thing: detections are pooled across all crops, sorted by score, and
matched greedily to ground truth, then AP is the 101-point interpolated area under the
precision-recall curve (COCO's definition).
"""
from __future__ import annotations

import numpy as np

IOU_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def iou_matrix(pred_map, pred_instances, gt_map):
    """(num_pred, num_gt) IoU matrix, computed once per crop by label co-occurrence."""
    gt_ids = np.unique(gt_map)
    gt_ids = gt_ids[gt_ids >= 2]
    n_gt, n_pred = len(gt_ids), len(pred_instances)
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_pred, n_gt), np.float32), n_gt

    gt_lut = np.zeros(int(gt_map.max()) + 1, np.int64) - 1
    for i, g in enumerate(gt_ids):
        gt_lut[g] = i
    max_pred = max(p["id"] for p in pred_instances)
    p_lut = np.zeros(max_pred + 1, np.int64) - 1
    for i, p in enumerate(pred_instances):
        p_lut[p["id"]] = i

    fp_, fg_ = pred_map.ravel(), gt_map.ravel()
    m = (fp_ >= 2) & (fg_ >= 2) & (fp_ <= max_pred)
    pi, gi = p_lut[fp_[m]], gt_lut[fg_[m]]
    ok = (pi >= 0) & (gi >= 0)
    inter = np.bincount(pi[ok] * n_gt + gi[ok], minlength=n_pred * n_gt) \
              .reshape(n_pred, n_gt).astype(np.float32)

    gt_area = np.bincount(gt_lut[fg_[fg_ >= 2]], minlength=n_gt).astype(np.float32)[None, :]
    pred_area = np.array([p["area"] for p in pred_instances], np.float32)[:, None]
    union = pred_area + gt_area - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0), n_gt


class APAccumulator:
    """Pools detections across crops; COCO accumulates globally, not per image."""

    def __init__(self, thresholds=IOU_THRESHOLDS):
        self.thresholds = np.asarray(thresholds)
        self.scores = []
        self.matches = []   # (n_thresh, n_det) boolean
        self.n_gt = 0

    def add(self, pred_map, pred_instances, gt_map):
        ious, n_gt = iou_matrix(pred_map, pred_instances, gt_map)
        self.n_gt += n_gt
        if not pred_instances:
            return
        scores = np.array([p["score"] for p in pred_instances], np.float64)
        order = np.argsort(-scores)
        self.scores.append(scores[order])
        hits = np.zeros((len(self.thresholds), len(order)), bool)
        for t_i, t in enumerate(self.thresholds):
            taken = np.zeros(n_gt, bool)
            for d_i, p_i in enumerate(order):
                if n_gt == 0:
                    break
                cand = np.where((ious[p_i] >= t) & ~taken)[0]
                if len(cand):
                    g = cand[np.argmax(ious[p_i, cand])]
                    taken[g] = True
                    hits[t_i, d_i] = True
        self.matches.append(hits)

    def ap(self):
        """Returns {'AP': AP@[.5:.95], 'AP50':…, 'AP75':…} in percent."""
        if not self.scores or self.n_gt == 0:
            return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
        scores = np.concatenate(self.scores)
        matches = np.concatenate(self.matches, axis=1)
        order = np.argsort(-scores, kind="stable")
        matches = matches[:, order]

        rec_grid = np.linspace(0, 1, 101)
        aps = []
        for t_i in range(len(self.thresholds)):
            tp = np.cumsum(matches[t_i])
            fp = np.cumsum(~matches[t_i])
            rec = tp / self.n_gt
            prec = tp / np.maximum(tp + fp, 1e-9)
            # monotone envelope, then sample at the 101 recall points
            prec = np.maximum.accumulate(prec[::-1])[::-1]
            idx = np.searchsorted(rec, rec_grid, side="left")
            q = np.where(idx < len(prec), prec[np.clip(idx, 0, len(prec) - 1)], 0.0)
            aps.append(q.mean())
        aps = np.array(aps)
        i50 = int(np.argmin(np.abs(self.thresholds - 0.5)))
        i75 = int(np.argmin(np.abs(self.thresholds - 0.75)))
        return {"AP": float(aps.mean() * 100), "AP50": float(aps[i50] * 100),
                "AP75": float(aps[i75] * 100)}


def instance_scores_from_surface(pred_map, instances, surface):
    """Confidence = peak surface response inside the instance.

    Merged instances carry score 1.0 from the merge step, which makes the ranking
    degenerate and AP meaningless. This restores a usable ordering from the network's
    own output without introducing a new learned component.
    """
    for p in instances:
        m = pred_map == p["id"]
        p["score"] = float(surface[m].max()) if m.any() else 0.0
    return instances
