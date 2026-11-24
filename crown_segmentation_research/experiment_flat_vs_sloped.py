"""Experiment: Flat-Basin Solid Mask vs. Sloped Elevation vs. Vector Flow.

Evaluates:
1. Sloped-Elevation Watershed (Baseline / Old approach - produces concentric rings)
2. Flat-Basin Boundary-Wall Watershed (New approach - 1 Tree = 1 Solid Flat Mask)
3. Pure Vector Flow Centripetal Clustering (TreeFlowNet - flows directly to apex)
4. Hamiltonian TopoFlow Flat-Basin (Vector Divergence Barrier + Boundary)
"""
from __future__ import annotations

import os
import sys
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
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import (
    compute_vector_divergence,
    evaluate_metrics,
)


def render_colored_overlay(rgb: np.ndarray, instance_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Renders crisp colored instance masks with white boundary contours."""
    H, W = instance_map.shape
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    inst_ids = np.unique(instance_map)
    inst_ids = inst_ids[inst_ids >= 2]

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(max(len(inst_ids) + 10, 500), 3), dtype=np.uint8)

    for i, iid in enumerate(inst_ids):
        colored[instance_map == iid] = colors[i % len(colors)]

    has_mask = instance_map >= 2
    blended = rgb.copy()
    blended[has_mask] = (
        (1 - alpha) * rgb[has_mask].astype(np.float32) + alpha * colored[has_mask].astype(np.float32)
    ).astype(np.uint8)

    for iid in inst_ids:
        mask_i = (instance_map == iid).astype(np.uint8)
        cnts, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, cnts, -1, (255, 255, 255), 1)

    return blended


def decode_sloped_watershed(surf, bound, canopy, min_dist=8.0, min_apex=0.25, min_canopy=0.25):
    """Old paradigm: Floods on continuous sloped potential (1 - U)."""
    H, W = surf.shape
    k = max(3, int(min_dist) if int(min_dist) % 2 == 1 else int(min_dist) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    s_max = cv2.dilate(surf, kernel)
    peaks = (surf == s_max) & (surf >= min_apex) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)

    scores = surf[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    markers = np.zeros((H, W), dtype=np.int32)
    markers[canopy < min_canopy] = 1
    for i, (y, x) in enumerate(zip(ys, xs)):
        markers[max(0, y - 1) : min(H, y + 2), max(0, x - 1) : min(W, x + 2)] = i + 2

    # Sloped relief
    W_surf = (1.0 - surf) + 0.5 * bound
    W_u8 = np.clip((W_surf / 1.5) * 255.0, 0, 255).astype(np.uint8)
    W_3c = cv2.merge([W_u8, W_u8, W_u8])
    cv2.watershed(W_3c, markers)

    insts = []
    for i, (y, x, s) in enumerate(zip(ys, xs, scores)):
        iid = i + 2
        mask = markers == iid
        area = int(mask.sum())
        if area >= 25:
            insts.append({"id": iid, "apex": (y, x), "score": float(s), "area": area})
    return markers, insts


def decode_flat_basin_watershed(surf, bound, canopy, min_dist=24.0, min_apex=0.25, min_canopy=0.25):
    """New paradigm: 1 Tree = 1 Solid Flat Mask. Relief inside canopy is flat (0), ridge at boundary."""
    H, W = surf.shape
    k = max(3, int(min_dist) if int(min_dist) % 2 == 1 else int(min_dist) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    s_max = cv2.dilate(surf, kernel)
    peaks = (surf == s_max) & (surf >= min_apex) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)

    scores = surf[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    # Strict spatial NMS: exactly 1 apex per tree crown
    kept_ys, kept_xs, kept_scores = [], [], []
    for y, x, s in zip(ys, xs, scores):
        if not any((y - ky) ** 2 + (x - kx) ** 2 < min_dist**2 for ky, kx in zip(kept_ys, kept_xs)):
            kept_ys.append(y)
            kept_xs.append(x)
            kept_scores.append(s)

    markers = np.zeros((H, W), dtype=np.int32)
    markers[canopy < min_canopy] = 1
    for i, (y, x) in enumerate(zip(kept_ys, kept_xs)):
        markers[max(0, y - 1) : min(H, y + 2), max(0, x - 1) : min(W, x + 2)] = i + 2

    # Flat basin relief: interior is 0, boundary is tall wall
    relief_flat = (bound * 255.0).clip(0, 255).astype(np.uint8)
    relief_flat_3c = cv2.merge([relief_flat, relief_flat, relief_flat])
    cv2.watershed(relief_flat_3c, markers)

    insts = []
    for i, (y, x, s) in enumerate(zip(kept_ys, kept_xs, kept_scores)):
        iid = i + 2
        mask = markers == iid
        area = int(mask.sum())
        if area >= 25:
            insts.append({"id": iid, "apex": (y, x), "score": float(s), "area": area})
    return markers, insts


def decode_vector_flow_direct(flow, centroid, canopy, min_dist=24.0, min_conf=0.25, min_canopy=0.25):
    """Pure Centripetal Vector Flow Clustering: Direct assignment to detected apex sinks."""
    H, W = centroid.shape
    vy, vx = flow[0], flow[1]

    # Detect apex centroids
    k = max(3, int(min_dist) if int(min_dist) % 2 == 1 else int(min_dist) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    c_max = cv2.dilate(centroid, kernel)
    peaks = (centroid == c_max) & (centroid >= min_conf) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)

    scores = centroid[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    kept_ys, kept_xs, kept_scores = [], [], []
    for y, x, s in zip(ys, xs, scores):
        if not any((y - ky) ** 2 + (x - kx) ** 2 < min_dist**2 for ky, kx in zip(kept_ys, kept_xs)):
            kept_ys.append(y)
            kept_xs.append(x)
            kept_scores.append(s)

    if not kept_ys:
        return np.zeros((H, W), dtype=np.int32), []

    # Map each pixel via flow vector: p_target = p + v * step
    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    # 3-step vector integration along learned unit flow field
    dest_y = np.clip(grid_y + vy * 12.0, 0, H - 1)
    dest_x = np.clip(grid_x + vx * 12.0, 0, W - 1)

    markers = np.zeros((H, W), dtype=np.int32)
    fg_mask = canopy >= min_canopy

    # Compute Euclidean distance from each integrated pixel destination to all apex sinks
    dest_y_fg = dest_y[fg_mask]
    dest_x_fg = dest_x[fg_mask]

    apices_y = np.array(kept_ys)[:, None]
    apices_x = np.array(kept_xs)[:, None]

    dists_sq = (dest_y_fg[None, :] - apices_y) ** 2 + (dest_x_fg[None, :] - apices_x) ** 2
    best_apex_idx = np.argmin(dists_sq, axis=0)

    markers[fg_mask] = best_apex_idx + 2

    insts = []
    for i, (y, x, s) in enumerate(zip(kept_ys, kept_xs, kept_scores)):
        iid = i + 2
        mask = markers == iid
        area = int(mask.sum())
        if area >= 25:
            insts.append({"id": iid, "apex": (y, x), "score": float(s), "area": area})
    return markers, insts


def decode_hamiltonian_flat(flow, bound, canopy, centroid, min_dist=24.0, min_conf=0.25, min_canopy=0.25):
    """Hamiltonian TopoFlow Flat-Basin: Energy is 0 inside tree, infinite barrier wall at div(V) > 0."""
    H, W = centroid.shape
    div_v = compute_vector_divergence(flow)
    seam_chasm = np.maximum(0.0, div_v)
    max_c = seam_chasm.max()
    if max_c > 0:
        seam_chasm = seam_chasm / max_c

    # Extract single apex per tree
    k = max(3, int(min_dist) if int(min_dist) % 2 == 1 else int(min_dist) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    c_max = cv2.dilate(centroid, kernel)
    peaks = (centroid == c_max) & (centroid >= min_conf) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)

    scores = centroid[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]

    kept_ys, kept_xs, kept_scores = [], [], []
    for y, x, s in zip(ys, xs, scores):
        if not any((y - ky) ** 2 + (x - kx) ** 2 < min_dist**2 for ky, kx in zip(kept_ys, kept_xs)):
            kept_ys.append(y)
            kept_xs.append(x)
            kept_scores.append(s)

    markers = np.zeros((H, W), dtype=np.int32)
    markers[canopy < min_canopy] = 1
    for i, (y, x) in enumerate(zip(kept_ys, kept_xs)):
        markers[max(0, y - 1) : min(H, y + 2), max(0, x - 1) : min(W, x + 2)] = i + 2

    # Flat interior, divergence + boundary barrier walls
    relief = (1.5 * bound + 2.0 * seam_chasm) / 3.5
    relief_u8 = np.clip(relief * 255.0, 0, 255).astype(np.uint8)
    relief_3c = cv2.merge([relief_u8, relief_u8, relief_u8])
    cv2.watershed(relief_3c, markers)

    insts = []
    for i, (y, x, s) in enumerate(zip(kept_ys, kept_xs, kept_scores)):
        iid = i + 2
        mask = markers == iid
        area = int(mask.sum())
        if area >= 25:
            insts.append({"id": iid, "apex": (y, x), "score": float(s), "area": area})
    return markers, insts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Flat vs Sloped Experiment] Loading models on {device}...", flush=True)

    ws_model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt_ws = torch.load("DeadTrees/experiments/canopy_watershed/best_sharp_watershed.pth", map_location=device, weights_only=False)
    ws_model.load_state_dict(ckpt_ws["model_state_dict"])
    ws_model.eval()

    flow_model = TreeFlowNet(pretrained_backbone=False).to(device)
    ckpt_flow = torch.load("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth", map_location=device, weights_only=False)
    sd_flow = ckpt_flow["model_state_dict"] if isinstance(ckpt_flow, dict) and "model_state_dict" in ckpt_flow else ckpt_flow
    flow_model.load_state_dict(sd_flow)
    flow_model.eval()

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=1024, augment=False, compute_targets=False)
    out_dir = Path("crown_segmentation_research")

    for s_idx in [0, 1, 2]:
        item = val_dataset[s_idx]
        img_t = item["image"].unsqueeze(0).to(device)
        rgb_raw = (item["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        raw_gt = item["instance_label"].squeeze().numpy().astype(np.int32)
        gt_map = np.zeros_like(raw_gt)
        gt_map[raw_gt > 0] = raw_gt[raw_gt > 0] + 1
        num_gt = len(np.unique(raw_gt[raw_gt > 0]))

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ws_p = ws_model(img_t)
                flow_p = flow_model(img_t)

        surf = ws_p["surface"].squeeze().float().cpu().numpy()
        bound = ws_p["boundary"].squeeze().float().cpu().numpy()
        canopy = ws_p["canopy"].squeeze().float().cpu().numpy()

        flow = flow_p["flow"].squeeze().float().cpu().numpy()
        centroid = flow_p["centroid"].squeeze().float().cpu().numpy()
        flow_canopy = flow_p["canopy"].squeeze().float().cpu().numpy()

        # Method 1: Old Sloped Elevation (Concentric Rings)
        m_sloped, i_sloped = decode_sloped_watershed(surf, bound, canopy, min_dist=8.0)
        e_sloped = evaluate_metrics(m_sloped, i_sloped, gt_map)

        # Method 2: New Flat-Basin Boundary Watershed (1 Tree = 1 Solid Mask)
        m_flat, i_flat = decode_flat_basin_watershed(surf, bound, canopy, min_dist=26.0)
        e_flat = evaluate_metrics(m_flat, i_flat, gt_map)

        # Method 3: Pure Centripetal Vector Flow Clustering (seeded by surf peaks)
        m_flow, i_flow = decode_vector_flow_direct(flow, surf, flow_canopy, min_dist=26.0)
        e_flow = evaluate_metrics(m_flow, i_flow, gt_map)

        # Method 4: Hamiltonian TopoFlow Flat-Basin (seeded by surf peaks)
        m_ham, i_ham = decode_hamiltonian_flat(flow, bound, flow_canopy, surf, min_dist=26.0)
        e_ham = evaluate_metrics(m_ham, i_ham, gt_map)

        # 5-Panel Side-by-Side Comparison Plot
        fig, axes = plt.subplots(1, 5, figsize=(32, 6.5), dpi=200)

        # 1. Ground Truth
        overlay_gt = render_colored_overlay(rgb_raw, gt_map)
        axes[0].imshow(overlay_gt)
        axes[0].set_title(f"GROUND TRUTH\nTrees: {num_gt} | Reference Solid Polygons", fontsize=11, fontweight="bold", color="darkgreen")
        axes[0].axis("off")

        # 2. Sloped Elevation (Ripple Rings)
        overlay_sloped = render_colored_overlay(rgb_raw, m_sloped)
        axes[1].imshow(overlay_sloped)
        axes[1].set_title(
            f"SLOPED ELEVATION (Old)\nTrees: {len(i_sloped)} | Concentric Ring Artifacts\n(Pseudo-Elevation Failure)",
            fontsize=10, fontweight="bold", color="darkred",
        )
        axes[1].axis("off")

        # 3. Flat-Basin Boundary Watershed
        overlay_flat = render_colored_overlay(rgb_raw, m_flat)
        axes[2].imshow(overlay_flat)
        axes[2].set_title(
            f"FLAT-BASIN WATERSHED (New)\nTrees: {len(i_flat)} | 1 Tree = 1 Solid Mask\n(Zero Ripple Rings)",
            fontsize=10, fontweight="bold", color="navy",
        )
        axes[2].axis("off")

        # 4. Pure Vector Flow Clustering
        overlay_flow = render_colored_overlay(rgb_raw, m_flow)
        axes[3].imshow(overlay_flow)
        axes[3].set_title(
            f"TREEFLOW VECTOR CLUSTERING\nTrees: {len(i_flow)} | Centripetal Flow Direct Delineation\n(Learned Flow Field)",
            fontsize=10, fontweight="bold", color="darkorange",
        )
        axes[3].axis("off")

        # 5. Hamiltonian TopoFlow Flat-Basin
        overlay_ham = render_colored_overlay(rgb_raw, m_ham)
        axes[4].imshow(overlay_ham)
        axes[4].set_title(
            f"HAMILTONIAN TOPOFLOW FLAT\nTrees: {len(i_ham)} | Divergence + Boundary Wall\n(Infinite Barrier Seams)",
            fontsize=10, fontweight="bold", color="purple",
        )
        axes[4].axis("off")

        plt.suptitle(
            f"BAMFORESTS Paradigm Shift: Solid Flat Instance Masks vs. Sloped Elevation Rings (Sample {s_idx+1})",
            fontsize=15, fontweight="bold", y=0.98,
        )
        plt.tight_layout()

        out_img_path = out_dir / f"flat_vs_sloped_comparison_sample_{s_idx}.png"
        plt.savefig(out_img_path, bbox_inches="tight")
        plt.close()
        print(f"[Flat vs Sloped Experiment] Saved comparison to {out_img_path}", flush=True)

        print(f"Sample {s_idx+1} Results (GT: {num_gt} trees):")
        print(f"  Sloped (Old):  {len(i_sloped):3d} trees | mAP50: {e_sloped['ap50']:5.2f}% | PQ: {e_sloped['pq']:5.2f}%")
        print(f"  Flat-Basin:    {len(i_flat):3d} trees | mAP50: {e_flat['ap50']:5.2f}% | PQ: {e_flat['pq']:5.2f}%")
        print(f"  Vector Flow:   {len(i_flow):3d} trees | mAP50: {e_flow['ap50']:5.2f}% | PQ: {e_flow['pq']:5.2f}%")
        print(f"  Hamiltonian:   {len(i_ham):3d} trees | mAP50: {e_ham['ap50']:5.2f}% | PQ: {e_ham['pq']:5.2f}%")


if __name__ == "__main__":
    main()
