"""Fast scale and post-mask calibration experiment for Canopy Watershed Net.
Tests different distance thresholds, topological persistence, and potential surface post-masking.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
import cv2
import numpy as np
import torch

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.decode import extract_persistent_apices
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet


def decode_with_postmask(
    surface: np.ndarray,
    boundary: np.ndarray,
    canopy: np.ndarray,
    kernel_size: int = 21,
    min_apex_val: float = 0.35,
    min_canopy_val: float = 0.40,
    pers_thresh: float = 0.15,
    min_distance: float = 50.0,
    bound_weight: float = 1.5,
    surf_edge_thresh: float = 0.15,
    min_area: int = 300,
) -> tuple[np.ndarray, list[dict]]:
    H, W = surface.shape
    peaks = extract_persistent_apices(
        surface=surface,
        canopy=canopy,
        kernel_size=kernel_size,
        min_apex_val=min_apex_val,
        min_canopy_val=min_canopy_val,
        pers_thresh=pers_thresh,
        min_distance=min_distance,
    )
    if not peaks:
        return np.zeros((H, W), dtype=np.int32), []

    markers = np.zeros((H, W), dtype=np.int32)
    markers[canopy < min_canopy_val] = 1

    for i, (y, x, _) in enumerate(peaks):
        inst_id = i + 2
        y_min, y_max = max(0, y - 2), min(H, y + 3)
        x_min, x_max = max(0, x - 2), min(W, x + 3)
        markers[y_min:y_max, x_min:x_max] = inst_id

    W_surf = (1.0 - surface) + bound_weight * boundary
    max_val = 1.0 + bound_weight
    W_u8 = np.clip((W_surf / max_val) * 255.0, 0, 255).astype(np.uint8)
    W_3c = cv2.merge([W_u8, W_u8, W_u8])

    cv2.watershed(W_3c, markers)

    # Apply potential surface threshold post-masking to trim sprawl
    if surf_edge_thresh > 0:
        markers[surface < surf_edge_thresh] = 0

    valid_ids = markers[markers >= 2]
    if len(valid_ids) == 0:
        return markers, []

    max_id = len(peaks) + 2
    areas = np.bincount(valid_ids, minlength=max_id)

    instances: list[dict] = []
    for i, (y, x, score) in enumerate(peaks):
        inst_id = i + 2
        area = int(areas[inst_id]) if inst_id < len(areas) else 0
        if area < min_area:
            markers[markers == inst_id] = 0
            continue
        instances.append({
            "id": inst_id,
            "apex": (y, x),
            "score": score,
            "area": area,
        })

    return markers, instances


def evaluate_sample(pred_markers: np.ndarray, pred_instances: list[dict], gt_map: np.ndarray) -> dict[str, float]:
    gt_ids = np.unique(gt_map)
    gt_ids = gt_ids[gt_ids >= 1]
    num_gt = len(gt_ids)
    num_pred = len(pred_instances)

    if num_gt == 0 and num_pred == 0:
        return {"ap50": 1.0, "ap75": 1.0, "pq": 1.0, "sq": 1.0, "rq": 1.0, "tp": 0, "gt": 0, "pred": 0}
    if num_gt == 0 or num_pred == 0:
        return {"ap50": 0.0, "ap75": 0.0, "pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "gt": num_gt, "pred": num_pred}

    flat_pred = pred_markers.ravel()
    flat_gt = gt_map.ravel()

    max_pred = int(flat_pred.max())
    if max_pred < 2:
        return {"ap50": 0.0, "ap75": 0.0, "pq": 0.0, "sq": 0.0, "rq": 0.0, "tp": 0, "gt": num_gt, "pred": num_pred}

    pred_lookup = np.full(max_pred + 1, -1, dtype=np.int64)
    for idx, p in enumerate(pred_instances):
        if p["id"] <= max_pred:
            pred_lookup[p["id"]] = idx

    max_gt = int(flat_gt.max())
    gt_lookup = np.full(max_gt + 1, -1, dtype=np.int64)
    for g_idx, gid in enumerate(gt_ids):
        gt_lookup[gid] = g_idx

    gt_valid = flat_gt[flat_gt >= 1]
    g_idx_arr_all = gt_lookup[gt_valid]
    gt_areas = np.bincount(g_idx_arr_all, minlength=num_gt)[None, :].astype(np.float32)

    valid_mask = (flat_pred >= 2) & (flat_gt >= 1)
    iou_mat = np.zeros((num_pred, num_gt), dtype=np.float32)

    if np.any(valid_mask):
        p_sub = flat_pred[valid_mask]
        g_sub = flat_gt[valid_mask]

        p_idx_arr = pred_lookup[p_sub]
        g_idx_arr = gt_lookup[g_sub]

        both_valid = (p_idx_arr >= 0) & (g_idx_arr >= 0)
        if np.any(both_valid):
            p_final = p_idx_arr[both_valid]
            g_final = g_idx_arr[both_valid]

            encoded = p_final * num_gt + g_final
            intersections = np.bincount(encoded, minlength=num_pred * num_gt).reshape(num_pred, num_gt)
            pred_areas = np.array([p["area"] for p in pred_instances], dtype=np.float32)[:, None]

            unions = pred_areas + gt_areas - intersections
            unions[unions <= 0] = 1.0
            iou_mat = intersections.astype(np.float32) / unions

    results: dict[str, float] = {}
    for thresh in [0.5, 0.75]:
        matched_gt = set()
        tp = 0
        order = np.argsort([-p["score"] for p in pred_instances])
        for p_idx in order:
            best_g_idx = -1
            best_iou = thresh
            for g_idx in range(num_gt):
                if g_idx not in matched_gt and iou_mat[p_idx, g_idx] >= best_iou:
                    best_iou = iou_mat[p_idx, g_idx]
                    best_g_idx = g_idx
            if best_g_idx >= 0:
                tp += 1
                matched_gt.add(best_g_idx)

        rec = tp / num_gt if num_gt > 0 else 0.0
        prec = tp / num_pred if num_pred > 0 else 0.0
        ap = 2.0 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        key = "ap50" if abs(thresh - 0.5) < 1e-4 else "ap75"
        results[key] = ap
        if thresh == 0.5:
            results["tp50"] = tp

    matched_ious = []
    used_preds = set()
    used_gts = set()
    for p_idx in range(num_pred):
        for g_idx in range(num_gt):
            if iou_mat[p_idx, g_idx] > 0.5:
                if p_idx not in used_preds and g_idx not in used_gts:
                    used_preds.add(p_idx)
                    used_gts.add(g_idx)
                    matched_ious.append(iou_mat[p_idx, g_idx])

    tp = len(matched_ious)
    fp = num_pred - tp
    fn = num_gt - tp
    sq = float(np.mean(matched_ious)) if tp > 0 else 0.0
    rq = tp / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) > 0 else 0.0
    pq = sq * rq
    results["pq"] = pq
    results["sq"] = sq
    results["rq"] = rq
    results["gt"] = num_gt
    results["pred"] = num_pred
    return results


def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading CanopyWatershedNet on {device}...")
    model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt = torch.load("DeadTrees/experiments/canopy_watershed/best_canopy_watershed.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=1024, augment=False, compute_targets=False)
    num_val = min(15, len(val_dataset))

    cached = []
    print(f"Caching {num_val} validation samples...")
    with torch.no_grad():
        for i in range(num_val):
            item = val_dataset[i]
            img = item["image"].unsqueeze(0).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                preds = model(img)
            surf_p = preds["surface"].squeeze().float().cpu().numpy()
            bound_p = preds["boundary"].squeeze().float().cpu().numpy()
            canopy_p = preds["canopy"].squeeze().float().cpu().numpy()
            gt_map = item["instance_label"].squeeze().numpy().astype(np.int32)
            cached.append({
                "surface": surf_p,
                "boundary": bound_p,
                "canopy": canopy_p,
                "gt_map": gt_map,
            })

    print(f"Done caching {num_val} items.")

    # Test sweep over calibrated parameter space
    test_configs = [
        # (min_dist, kern, pers, edge_thresh, bound_w, min_area)
        (30.0, 15, 0.10, 0.00, 1.0, 200),
        (30.0, 15, 0.15, 0.15, 1.5, 300),
        (45.0, 21, 0.10, 0.00, 1.0, 300),
        (45.0, 21, 0.15, 0.15, 1.5, 500),
        (60.0, 25, 0.10, 0.00, 1.0, 500),
        (60.0, 25, 0.15, 0.15, 1.5, 500),
        (60.0, 25, 0.20, 0.20, 2.0, 800),
        (75.0, 31, 0.10, 0.00, 1.0, 500),
        (75.0, 31, 0.15, 0.15, 1.5, 800),
        (75.0, 31, 0.20, 0.20, 2.0, 1000),
        (90.0, 35, 0.15, 0.15, 1.5, 800),
        (90.0, 35, 0.20, 0.20, 2.0, 1200),
        (110.0, 45, 0.15, 0.15, 1.5, 1200),
        (110.0, 45, 0.20, 0.20, 2.0, 1500),
    ]

    print(f"\n{'Dist':>5} | {'Kern':>4} | {'Pers':>5} | {'Edge':>5} | {'BndW':>5} | {'MinA':>5} | {'AvgGT':>6} | {'AvgPred':>7} | {'mAP50':>7} | {'mAP75':>7} | {'PQ':>6}")
    print("-" * 85)

    for min_dist, kern, pers, edge_t, bnd_w, min_a in test_configs:
        tot_ap50 = 0.0
        tot_ap75 = 0.0
        tot_pq = 0.0
        tot_gt = 0
        tot_pred = 0

        for item in cached:
            p_markers, p_insts = decode_with_postmask(
                surface=item["surface"],
                boundary=item["boundary"],
                canopy=item["canopy"],
                kernel_size=kern,
                min_apex_val=0.35,
                min_canopy_val=0.40,
                pers_thresh=pers,
                min_distance=min_dist,
                bound_weight=bnd_w,
                surf_edge_thresh=edge_t,
                min_area=min_a,
            )
            m = evaluate_sample(p_markers, p_insts, item["gt_map"])
            tot_ap50 += m["ap50"]
            tot_ap75 += m["ap75"]
            tot_pq += m["pq"]
            tot_gt += m["gt"]
            tot_pred += m["pred"]

        mean_ap50 = (tot_ap50 / num_val) * 100.0
        mean_ap75 = (tot_ap75 / num_val) * 100.0
        mean_pq = (tot_pq / num_val) * 100.0
        avg_gt = tot_gt / num_val
        avg_pred = tot_pred / num_val

        print(f"{min_dist:5.1f} | {kern:4d} | {pers:5.2f} | {edge_t:5.2f} | {bnd_w:5.1f} | {min_a:5d} | {avg_gt:6.1f} | {avg_pred:7.1f} | {mean_ap50:6.2f}% | {mean_ap75:6.2f}% | {mean_pq:5.2f}%")


if __name__ == "__main__":
    run()
