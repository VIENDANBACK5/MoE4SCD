"""Comprehensive Multi-Paradigm & Boundary Refinement Benchmark.

Evaluates 6 Paradigms across BAMFORESTS Validation:
1. Baseline CanopyWatershed (Sloped Relief)
2. Flat-Basin Watershed (Unrefined)
3. Mode A: Benchmark-Optimized Smoothed (Adaptive Douglas-Peucker)
4. Mode B: Organic MorphSnake Snapped (Geodesic Active Contour)
5. TreeFlow Vector Clustering
6. Hamiltonian TopoFlow-Watershed (Flow Divergence + Boundary Ridge)

Computes:
- Instance Metrics: mAP@50, mAP@75, Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ)
- Boundary Fidelity: Boundary-IoU (B-IoU), Dao et al. Over-Segmentation (OS), Under-Segmentation (US), Euclidean Distance (ED)
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
import json
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from crown_segmentation_research.methods.boundary_refinement.metrics_dao import (
    compute_boundary_iou,
    compute_dao_segmentation_indices,
)
from crown_segmentation_research.methods.boundary_refinement.morph_snakes import (
    refine_instances_morph_snakes,
)
from crown_segmentation_research.methods.boundary_refinement.adaptive_smoother import (
    apply_benchmark_smoothing,
)
from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode_flow_to_instances
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import (
    compute_vector_divergence,
    evaluate_metrics,
)


def run_sloped_watershed(surf: np.ndarray, bound: np.ndarray, canopy: np.ndarray, min_dist: float = 24.0, min_apex: float = 0.25, min_canopy: float = 0.25):
    """Old sloped watershed that suffers from ripple-ring artifacts."""
    H, W = surf.shape
    k = max(3, int(min_dist) if int(min_dist) % 2 == 1 else int(min_dist) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    s_max = cv2.dilate(surf, kernel)
    peaks = (surf == s_max) & (surf >= min_apex) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)

    scores = surf[ys, xs]
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

    # Old continuous sloped potential (1 - U) + B
    relief = ((1.0 - surf) * 0.7 + bound * 0.3) * 255.0
    relief_3c = cv2.merge([relief.clip(0, 255).astype(np.uint8)] * 3)
    cv2.watershed(relief_3c, markers)

    insts = []
    for i, (y, x, s) in enumerate(zip(kept_ys, kept_xs, kept_scores)):
        iid = i + 2
        mask = markers == iid
        area = int(mask.sum())
        if area >= 25:
            insts.append({"id": iid, "apex": (y, x), "score": float(s), "area": area})
    return markers, insts


def run_flat_basin_watershed(surf: np.ndarray, bound: np.ndarray, canopy: np.ndarray, min_dist: float = 24.0, min_apex: float = 0.25, min_canopy: float = 0.25):
    """Solid flat-basin watershed without internal gradient ripple artifacts."""
    H, W = surf.shape
    k = max(3, int(min_dist) if int(min_dist) % 2 == 1 else int(min_dist) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    s_max = cv2.dilate(surf, kernel)
    peaks = (surf == s_max) & (surf >= min_apex) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)

    scores = surf[ys, xs]
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

    # Flat basin relief = only pure boundary ridge (bound * 255)
    relief = (bound * 255.0).clip(0, 255).astype(np.uint8)
    relief_3c = cv2.merge([relief, relief, relief])
    cv2.watershed(relief_3c, markers)

    insts = []
    for i, (y, x, s) in enumerate(zip(kept_ys, kept_xs, kept_scores)):
        iid = i + 2
        mask = markers == iid
        area = int(mask.sum())
        if area >= 25:
            insts.append({"id": iid, "apex": (y, x), "score": float(s), "area": area})
    return markers, insts


def run_hamiltonian_topoflow(surf: np.ndarray, bound: np.ndarray, canopy: np.ndarray, flow: np.ndarray, min_dist: float = 24.0, min_apex: float = 0.25, min_canopy: float = 0.25):
    """Hamiltonian TopoFlow-Watershed combining vector flow divergence chasm with boundary ridge."""
    H, W = surf.shape
    div_v = compute_vector_divergence(flow)

    k = max(3, int(min_dist) if int(min_dist) % 2 == 1 else int(min_dist) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    s_max = cv2.dilate(surf, kernel)
    peaks = (surf == s_max) & (surf >= min_apex) & (canopy >= min_canopy)
    ys, xs = np.where(peaks)

    scores = surf[ys, xs]
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

    # TopoFlow energy: Positive divergence acts as vertical chasm barrier
    pos_div = np.maximum(0, div_v)
    pos_div_norm = (pos_div / (pos_div.max() + 1e-6))
    hamiltonian_energy = (0.6 * bound + 0.4 * pos_div_norm) * 255.0
    relief_3c = cv2.merge([hamiltonian_energy.clip(0, 255).astype(np.uint8)] * 3)
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
    parser = argparse.ArgumentParser(description="Multi-Paradigm Boundary Refinement Benchmark")
    parser.add_argument("--num_val", type=int, default=30, help="Number of validation crops to evaluate")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Benchmark] Initializing on {device} across {args.num_val} validation crops...", flush=True)

    ws_model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt_ws = torch.load("DeadTrees/experiments/canopy_watershed/best_sharp_watershed.pth", map_location=device, weights_only=False)
    ws_model.load_state_dict(ckpt_ws["model_state_dict"])
    ws_model.eval()

    flow_model = TreeFlowNet(pretrained_backbone=False).to(device)
    ckpt_flow = torch.load("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth", map_location=device, weights_only=False)
    sd_flow = ckpt_flow["model_state_dict"] if isinstance(ckpt_flow, dict) and "model_state_dict" in ckpt_flow else ckpt_flow
    flow_model.load_state_dict(sd_flow)
    flow_model.eval()

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=args.crop_size, augment=False, compute_targets=False)
    num_samples = min(args.num_val, len(val_dataset))

    methods = [
        "1. Baseline Sloped Watershed",
        "2. Flat-Basin Watershed (Unrefined)",
        "3. Mode A: Benchmark-Smoothed (Dao Optimal)",
        "4. Mode B: Organic MorphSnake Snapped",
        "5. TreeFlow Vector Clustering",
        "6. Hamiltonian TopoFlow-Watershed",
    ]

    records = {m: {"ap50": [], "ap75": [], "pq": [], "sq": [], "rq": [], "biou": [], "os": [], "us": [], "ed": []} for m in methods}

    t0 = time.time()
    for idx in range(num_samples):
        item = val_dataset[idx]
        img_t = item["image"].unsqueeze(0).to(device)
        rgb_raw = (item["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        raw_gt = item["instance_label"].squeeze().numpy().astype(np.int32)
        gt_map = np.zeros_like(raw_gt)
        gt_map[raw_gt > 0] = raw_gt[raw_gt > 0] + 1
        num_gt = len(np.unique(raw_gt[raw_gt > 0]))

        if num_gt == 0:
            continue

        with torch.no_grad():
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                ws_p = ws_model(img_t)
                flow_p = flow_model(img_t)

        surf = ws_p["surface"].squeeze().float().cpu().numpy()
        bound = ws_p["boundary"].squeeze().float().cpu().numpy()
        canopy = ws_p["canopy"].squeeze().float().cpu().numpy()
        flow = flow_p["flow"].squeeze().float().cpu().numpy()
        flow_canopy = flow_p["canopy"].squeeze().float().cpu().numpy()
        sdt = flow_p["sdt"].squeeze().float().cpu().numpy()
        centroid = flow_p["centroid"].squeeze().float().cpu().numpy()

        # Method 1: Baseline Sloped Watershed
        m1, insts1 = run_sloped_watershed(surf, bound, canopy)
        e1 = evaluate_metrics(m1, insts1, gt_map)
        d1 = compute_dao_segmentation_indices(m1, gt_map)
        b1 = compute_boundary_iou(m1, gt_map)

        # Method 2: Flat-Basin Watershed
        m2, insts2 = run_flat_basin_watershed(surf, bound, canopy)
        e2 = evaluate_metrics(m2, insts2, gt_map)
        d2 = compute_dao_segmentation_indices(m2, gt_map)
        b2 = compute_boundary_iou(m2, gt_map)

        # Method 3: Mode A - Benchmark Smoothed
        m3, insts3 = apply_benchmark_smoothing(m2, epsilon_ratio=0.008)
        e3 = evaluate_metrics(m3, insts3, gt_map)
        d3 = compute_dao_segmentation_indices(m3, gt_map)
        b3 = compute_boundary_iou(m3, gt_map)

        # Method 4: Mode B - Organic MorphSnake
        m4, insts4 = refine_instances_morph_snakes(m2, rgb_raw, canopy_mask=(canopy >= 0.25), iterations=6, alpha=60.0)
        e4 = evaluate_metrics(m4, insts4, gt_map)
        d4 = compute_dao_segmentation_indices(m4, gt_map)
        b4 = compute_boundary_iou(m4, gt_map)

        # Method 5: TreeFlow Vector Flow Decoding
        m5_raw, _ = decode_flow_to_instances(flow, flow_canopy, sdt, centroid, canopy_threshold=0.35, min_peak_distance=5)
        # Re-index to >= 2
        m5 = np.zeros_like(m5_raw)
        insts5 = []
        for new_id, old_id in enumerate(np.unique(m5_raw[m5_raw > 0])):
            mask_5 = m5_raw == old_id
            if mask_5.sum() >= 25:
                iid = new_id + 2
                m5[mask_5] = iid
                ys, xs = np.where(mask_5)
                insts5.append({"id": iid, "apex": (int(np.mean(ys)), int(np.mean(xs))), "score": 1.0, "area": int(mask_5.sum())})
        e5 = evaluate_metrics(m5, insts5, gt_map)
        d5 = compute_dao_segmentation_indices(m5, gt_map)
        b5 = compute_boundary_iou(m5, gt_map)

        # Method 6: Hamiltonian TopoFlow-Watershed
        m6, insts6 = run_hamiltonian_topoflow(surf, bound, canopy, flow, min_dist=24.0)
        e6 = evaluate_metrics(m6, insts6, gt_map)
        d6 = compute_dao_segmentation_indices(m6, gt_map)
        b6 = compute_boundary_iou(m6, gt_map)

        eval_tuples = [
            (methods[0], e1, d1, b1),
            (methods[1], e2, d2, b2),
            (methods[2], e3, d3, b3),
            (methods[3], e4, d4, b4),
            (methods[4], e5, d5, b5),
            (methods[5], e6, d6, b6),
        ]

        for m_name, ev, da, bi in eval_tuples:
            records[m_name]["ap50"].append(ev["ap50"])
            records[m_name]["ap75"].append(ev["ap75"])
            records[m_name]["pq"].append(ev["pq"])
            records[m_name]["sq"].append(ev["sq"])
            records[m_name]["rq"].append(ev["rq"])
            records[m_name]["biou"].append(bi * 100.0)
            records[m_name]["os"].append(da["mean_os"])
            records[m_name]["us"].append(da["mean_us"])
            records[m_name]["ed"].append(da["mean_ed"])

        if (idx + 1) % 5 == 0 or idx == num_samples - 1:
            print(f"[{idx+1:02d}/{num_samples:02d}] Evaluated. Mode A mAP50: {np.mean(records[methods[2]]['ap50']):.2f}% | Mode B B-IoU: {np.mean(records[methods[3]]['biou']):.2f}%", flush=True)

    elapsed = time.time() - t0
    print(f"\n[Benchmark Finished] Evaluated {num_samples} crops in {elapsed:.1f}s ({elapsed/num_samples:.2f}s/crop)\n")

    summary_list = []
    for m in methods:
        summary_list.append({
            "Method": m,
            "mAP@50 (%)": float(np.mean(records[m]["ap50"])),
            "mAP@75 (%)": float(np.mean(records[m]["ap75"])),
            "PQ (%)": float(np.mean(records[m]["pq"])),
            "SQ (%)": float(np.mean(records[m]["sq"])),
            "RQ (%)": float(np.mean(records[m]["rq"])),
            "Boundary-IoU (%)": float(np.mean(records[m]["biou"])),
            "Dao Over-Seg (OS)": float(np.mean(records[m]["os"])),
            "Dao Under-Seg (US)": float(np.mean(records[m]["us"])),
            "Dao Euclidean Dist (ED)": float(np.mean(records[m]["ed"])),
        })

    df = pd.DataFrame(summary_list)
    print("=" * 115)
    print(f"{'PARADIGM & BOUNDARY REFINEMENT COMPREHENSIVE BENCHMARK':^115}")
    print("=" * 115)
    print(df.to_string(index=False))
    print("=" * 115)

    # Save to disk
    out_csv = Path("crown_segmentation_research/boundary_refinement_benchmark.csv")
    out_json = Path("crown_segmentation_research/boundary_refinement_benchmark.json")
    df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(summary_list, f, indent=2)

    print(f"Saved benchmark results to {out_csv} and {out_json}")


if __name__ == "__main__":
    from crown_segmentation_research.research_integrity.a06_hardening_v2.runtime import reject_legacy_entrypoint
    reject_legacy_entrypoint(__file__)
    main()
