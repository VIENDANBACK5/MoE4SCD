"""Stage 3: Hamiltonian TopoFlow-Watershed Synthesis.

Unifies centripetal vector flow divergence (TreeFlowNet) with continuous
topological potential watershed flooding (CanopyWatershedNet).

Mathematical Formulation:
1. Flow Field Divergence:
   div(V) = d(v_x)/dx + d(v_y)/dy
   - Tree apex sinks: div(V) << 0 (strong convergence sink)
   - Inter-crown contact seams: div(V) >> 0 (sharp outward divergence / source of flux)
2. Hamiltonian TopoFlow Energy Relief:
   E(y, x) = (1.0 - U(y, x)) + alpha * ReLU(div(V)) + beta * B(y, x)
3. Topological Persistence Watershed Flooding:
   Floods crisp, non-overlapping crown basins separated by infinite divergence chasms.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)

for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.decode import extract_persistent_apices
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet


def compute_vector_divergence(flow_field: np.ndarray) -> np.ndarray:
    """Computes spatial divergence div(V) = d(vx)/dx + d(vy)/dy.

    Args:
        flow_field: (2, H, W) float32 array where channel 0 is vy and channel 1 is vx.

    Returns:
        div_v: (H, W) float32 divergence field.
    """
    vy = flow_field[0]
    vx = flow_field[1]

    # Compute spatial gradients via Sobel operators
    dvx_dx = cv2.Sobel(vx, cv2.CV_32F, 1, 0, ksize=3)
    dvy_dy = cv2.Sobel(vy, cv2.CV_32F, 0, 1, ksize=3)

    div_v = dvx_dx + dvy_dy
    return div_v


def decode_hamiltonian_topoflow(
    surface: np.ndarray,      # (H, W) potential in [0, 1]
    boundary: np.ndarray,     # (H, W) boundary barrier in [0, 1]
    canopy: np.ndarray,       # (H, W) canopy gate in [0, 1]
    flow: np.ndarray,         # (2, H, W) unit vector field
    kernel_size: int = 5,
    min_apex_val: float = 0.20,
    min_canopy_val: float = 0.30,
    pers_thresh: float = 0.03,
    min_distance: float = 5.0,
    bound_weight: float = 1.0,
    div_weight: float = 2.0,
    min_area: int = 25,
) -> tuple[np.ndarray, list[dict]]:
    """Decodes instances using unified Hamiltonian TopoFlow relief and persistence watershed."""
    H, W = surface.shape

    # 1. Compute flow divergence field
    div_v = compute_vector_divergence(flow)

    # Positive divergence marks inter-crown contact seams (outward flux sources)
    seam_chasm = np.maximum(0.0, div_v)
    max_c = seam_chasm.max()
    if max_c > 0:
        seam_chasm = seam_chasm / max_c

    # 2. Extract persistent tree apices
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

    # 3. Initialize watershed marker grid
    markers = np.zeros((H, W), dtype=np.int32)
    markers[canopy < min_canopy_val] = 1

    for i, (y, x, _) in enumerate(peaks):
        inst_id = i + 2
        y_min, y_max = max(0, y - 1), min(H, y + 2)
        x_min, x_max = max(0, x - 1), min(W, x + 2)
        markers[y_min:y_max, x_min:x_max] = inst_id

    # 4. Construct Hamiltonian TopoFlow topographic energy relief
    relief = (1.0 - surface) + bound_weight * boundary + div_weight * seam_chasm
    max_val = 1.0 + bound_weight + div_weight
    relief_u8 = np.clip((relief / max_val) * 255.0, 0, 255).astype(np.uint8)
    relief_3c = cv2.merge([relief_u8, relief_u8, relief_u8])

    # 5. Fast C++ Watershed Flooding
    cv2.watershed(relief_3c, markers)

    # 6. Extract vector instances
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
            continue
        instances.append({
            "id": inst_id,
            "apex": (y, x),
            "score": score,
            "area": area,
        })

    return markers, instances


def evaluate_metrics(
    pred_markers: np.ndarray,
    pred_instances: list[dict],
    gt_map: np.ndarray,
) -> dict[str, float]:
    """Evaluates mAP50, mAP75, and Panoptic Quality against Ground Truth map."""
    gt_ids = np.unique(gt_map)
    gt_ids = gt_ids[gt_ids >= 2]
    num_gt = len(gt_ids)
    num_pred = len(pred_instances)

    if num_gt == 0 and num_pred == 0:
        return {"ap50": 100.0, "ap75": 100.0, "pq": 100.0, "sq": 100.0, "rq": 100.0, "trees": 0}
    if num_gt == 0 or num_pred == 0:
        return {"ap50": 0.0, "ap75": 0.0, "pq": 0.0, "sq": 0.0, "rq": 0.0, "trees": num_pred}

    max_gt = int(gt_map.max())
    gt_lookup = np.full(max_gt + 1, -1, dtype=np.int64)
    for g_i, gid in enumerate(gt_ids):
        gt_lookup[gid] = g_i

    gt_valid = gt_map[gt_map >= 2]
    g_idx_arr = gt_lookup[gt_valid]
    gt_areas = np.bincount(g_idx_arr, minlength=num_gt)[None, :].astype(np.float32)

    flat_pred = pred_markers.ravel()
    flat_gt = gt_map.ravel()

    max_pred = int(flat_pred.max())
    if max_pred < 2:
        return {"ap50": 0.0, "ap75": 0.0, "pq": 0.0, "sq": 0.0, "rq": 0.0, "trees": num_pred}

    pred_lookup = np.full(max_pred + 1, -1, dtype=np.int64)
    for idx, p in enumerate(pred_instances):
        pred_lookup[p["id"]] = idx

    valid_mask = (flat_pred >= 2) & (flat_gt >= 2)
    p_vals = flat_pred[valid_mask]
    g_vals = flat_gt[valid_mask]

    p_indices = pred_lookup[p_vals]
    g_indices = gt_lookup[g_vals]

    valid_pair = (p_indices >= 0) & (g_indices >= 0)
    p_indices = p_indices[valid_pair]
    g_indices = g_indices[valid_pair]

    flat_pair_indices = p_indices * num_gt + g_indices
    intersection = np.bincount(flat_pair_indices, minlength=num_pred * num_gt).reshape(num_pred, num_gt).astype(np.float32)

    pred_areas = np.array([p["area"] for p in pred_instances], dtype=np.float32)[:, None]
    union = pred_areas + gt_areas - intersection
    iou_mat = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)

    # AP50
    iou_50 = (iou_mat >= 0.5)
    matched_gt50 = set()
    tp50 = 0
    scores = np.array([p["score"] for p in pred_instances])
    p_order = np.argsort(-scores)

    for p_i in p_order:
        gt_cands = np.where(iou_50[p_i])[0]
        for g_i in gt_cands:
            if g_i not in matched_gt50:
                matched_gt50.add(g_i)
                tp50 += 1
                break

    prec50 = tp50 / num_pred if num_pred > 0 else 0.0
    rec50 = tp50 / num_gt if num_gt > 0 else 0.0
    ap50 = 2 * (prec50 * rec50) / (prec50 + rec50) if (prec50 + rec50) > 0 else 0.0

    # AP75
    iou_75 = (iou_mat >= 0.75)
    matched_gt75 = set()
    tp75 = 0
    for p_i in p_order:
        gt_cands = np.where(iou_75[p_i])[0]
        for g_i in gt_cands:
            if g_i not in matched_gt75:
                matched_gt75.add(g_i)
                tp75 += 1
                break

    prec75 = tp75 / num_pred if num_pred > 0 else 0.0
    rec75 = tp75 / num_gt if num_gt > 0 else 0.0
    ap75 = 2 * (prec75 * rec75) / (prec75 + rec75) if (prec75 + rec75) > 0 else 0.0

    # PQ
    tp_mask = (iou_mat > 0.5)
    tp_pairs = np.argwhere(tp_mask)
    num_tp = len(tp_pairs)
    fp = num_pred - num_tp
    fn = num_gt - num_tp

    if num_tp > 0:
        iou_sum = sum(iou_mat[p, g] for p, g in tp_pairs)
        sq = iou_sum / num_tp
        rq = num_tp / (num_tp + 0.5 * fp + 0.5 * fn)
        pq = sq * rq
    else:
        sq, rq, pq = 0.0, 0.0, 0.0

    return {
        "ap50": float(ap50 * 100.0),
        "ap75": float(ap75 * 100.0),
        "pq": float(pq * 100.0),
        "sq": float(sq * 100.0),
        "rq": float(rq * 100.0),
        "trees": num_pred,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watershed_model", type=str, default="DeadTrees/experiments/canopy_watershed/best_canopy_watershed.pth")
    parser.add_argument("--flow_model", type=str, default="DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    parser.add_argument("--num_val", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Stage 3 TopoFlow-Watershed] Initializing models on {device}...", flush=True)

    # 1. Load Watershed Model
    ws_model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt_ws = torch.load(args.watershed_model, map_location=device, weights_only=False)
    ws_model.load_state_dict(ckpt_ws["model_state_dict"])
    ws_model.eval()

    # 2. Load TreeFlowNet
    flow_model = TreeFlowNet(pretrained_backbone=False).to(device)
    ckpt_flow = torch.load(args.flow_model, map_location=device, weights_only=False)
    flow_sd = ckpt_flow["model_state_dict"] if isinstance(ckpt_flow, dict) and "model_state_dict" in ckpt_flow else ckpt_flow
    flow_model.load_state_dict(flow_sd)
    flow_model.eval()

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=1024, augment=False, compute_targets=False)
    N = min(args.num_val, len(val_dataset))
    print(f"[Stage 3 TopoFlow-Watershed] Benchmarking on {N} validation crops...", flush=True)

    metrics_list = []
    t_start = time.perf_counter()

    with torch.no_grad():
        for idx in range(N):
            item = val_dataset[idx]
            img = item["image"].unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ws_preds = ws_model(img)
                flow_preds = flow_model(img)

            surf_p = ws_preds["surface"].squeeze().float().cpu().numpy()
            bound_p = ws_preds["boundary"].squeeze().float().cpu().numpy()
            canopy_p = ws_preds["canopy"].squeeze().float().cpu().numpy()
            flow_p = flow_preds["flow"].squeeze().float().cpu().numpy()  # (2, H, W)

            raw_inst = item["instance_label"].squeeze().numpy().astype(np.int32)
            gt_map = np.zeros_like(raw_inst)
            gt_map[raw_inst > 0] = raw_inst[raw_inst > 0] + 1

            pred_markers, pred_instances = decode_hamiltonian_topoflow(
                surface=surf_p,
                boundary=bound_p,
                canopy=canopy_p,
                flow=flow_p,
                kernel_size=5,
                min_apex_val=0.20,
                min_canopy_val=0.30,
                pers_thresh=0.03,
                min_distance=5.0,
                bound_weight=1.0,
                div_weight=2.0,
                min_area=25,
            )

            m = evaluate_metrics(pred_markers, pred_instances, gt_map)
            metrics_list.append(m)

            if (idx + 1) % 10 == 0 or (idx + 1) == N:
                cur_ap50 = np.mean([x["ap50"] for x in metrics_list])
                cur_pq = np.mean([x["pq"] for x in metrics_list])
                cur_trees = np.mean([x["trees"] for x in metrics_list])
                print(f"  [{idx+1:02d}/{N:02d}] mAP@50: {cur_ap50:.2f}% | PQ: {cur_pq:.2f}% | Avg Trees: {cur_trees:.1f}", flush=True)

    elapsed = time.perf_counter() - t_start
    final_ap50 = float(np.mean([x["ap50"] for x in metrics_list]))
    final_ap75 = float(np.mean([x["ap75"] for x in metrics_list]))
    final_pq = float(np.mean([x["pq"] for x in metrics_list]))
    final_sq = float(np.mean([x["sq"] for x in metrics_list]))
    final_rq = float(np.mean([x["rq"] for x in metrics_list]))
    final_trees = float(np.mean([x["trees"] for x in metrics_list]))

    print("\n" + "=" * 80)
    print("        STAGE 3: HAMILTONIAN TOPOFLOW-WATERSHED BENCHMARK RESULTS     ")
    print("=" * 80)
    print(f"  - Validation Samples : {N} crops (1024x1024)")
    print(f"  - Execution Time     : {elapsed:.2f}s ({N/elapsed:.1f} FPS)")
    print(f"  - Avg Detected Trees : {final_trees:.1f} trees/crop")
    print(f"  - mAP@50             : {final_ap50:.2f}%")
    print(f"  - mAP@75             : {final_ap75:.2f}%")
    print(f"  - Panoptic Quality   : {final_pq:.2f}% (SQ: {final_sq:.2f}%, RQ: {final_rq:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
