"""Unified High-Precision Evaluation & Multi-Biome Preview for Neuro-Flow-Graph (NFG).

Replaces naive convex polytope stamping with:
  1. Continuous Centripetal Flow Field Dynamics v(x, y)
  2. Analytical Vector Divergence Sinks div(v) < 0 (Topological Sink Centroids)
  3. GSD-Adaptive Physical Scaling (invariant across 5cm, 10cm, 20cm)
  4. Lagrangian Streamline Integration (pixel-accurate organic branch contours)
  5. Spectral Modularity Partitioning Q (disentangles crossed fallen logs)
  6. Medial Axis Bellman-Ford Shortest Path (bridges shadow gaps)

Outputs high-definition 5-panel diagnostic figures to:
  crown_segmentation_research/images/previews_neuro_deadwood_v1/
"""
from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment
from shapely.geometry import MultiPolygon, Polygon

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.graph_decode import (
    compute_divergence_field,
    compute_orientation_field,
    decode_neuro_flow_graph,
    get_adaptive_gsd_parameters,
)

BENCH_DIR = Path("DTE-Aerial-Data-public")
VAL_DIR = Path("DeadTrees/star_convex_targets_v1/val")
MODEL_PATH = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
OUT_DIR = Path("crown_segmentation_research/images/previews_neuro_deadwood_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_iou(poly1: Polygon, poly2: Polygon) -> float:
    if not poly1.is_valid or not poly2.is_valid:
        poly1 = poly1.buffer(0)
        poly2 = poly2.buffer(0)
    if not poly1.intersects(poly2):
        return 0.0
    try:
        inter = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def evaluate_instance_matching(
    pred_polys: list[Polygon],
    gt_polys: list[Polygon],
    iou_thresh: float = 0.25,
) -> dict[str, float]:
    n_pred = len(pred_polys)
    n_gt = len(gt_polys)
    if n_gt == 0:
        return {"tp": 0, "fp": n_pred, "fn": 0, "precision": 0.0 if n_pred > 0 else 1.0, "recall": 1.0, "f1": 0.0 if n_pred > 0 else 1.0}
    if n_pred == 0:
        return {"tp": 0, "fp": 0, "fn": n_gt, "precision": 1.0, "recall": 0.0, "f1": 0.0}

    cost_matrix = np.zeros((n_pred, n_gt), dtype=np.float32)
    for i, p in enumerate(pred_polys):
        for j, g in enumerate(gt_polys):
            cost_matrix[i, j] = compute_iou(p, g)

    row_ind, col_ind = linear_sum_assignment(-cost_matrix)
    tp = sum(1 for r, c in zip(row_ind, col_ind) if cost_matrix[r, c] >= iou_thresh)
    fp = n_pred - tp
    fn = n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1}


def draw_polygons_on_image(
    image: np.ndarray,
    polygons: list[Polygon],
    color: tuple[int, int, int],
    thickness: int = 2,
    alpha: float = 0.35,
) -> np.ndarray:
    canvas = image.copy()
    overlay = image.copy()
    
    for poly in polygons:
        subs = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for sub in subs:
            pts = np.array(sub.exterior.coords).round().astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color=color)
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
                
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = TreeFlowNet(pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded high-precision TreeFlowNet weights from {MODEL_PATH}")

    # 1. Evaluate on Validation Set (Site 5737)
    val_npz_files = sorted(VAL_DIR.glob("*.npz"))
    print(f"\nEvaluating Adaptive NFG on {len(val_npz_files)} validation patches in {VAL_DIR}...")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    canopy_ious = []

    for path in val_npz_files:
        data = np.load(path)
        img = data["image"]
        prob_gt = data["probability"]
        
        gt_mask = (prob_gt > 0.05).astype(np.uint8)
        cnts, _ = cv2.findContours(gt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_polys = [Polygon(c.squeeze(1)) for c in cnts if len(c) >= 3 and cv2.contourArea(c) >= 10]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        with torch.no_grad():
            out = model(img_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        _, pred_polys = decode_neuro_flow_graph(
            flow=flow,
            canopy=canopy,
            sdt=sdt,
            centroid=centroid,
            resolution="5cm",
            canopy_threshold=0.35,
            sdt_threshold=-0.25,
            centroid_threshold=0.15,
            use_divergence_seeds=True,
            use_bellman_ford=True,
            use_spectral_cut=True,
        )

        match_res = evaluate_instance_matching(pred_polys, gt_polys, iou_thresh=0.25)
        total_tp += match_res["tp"]
        total_fp += match_res["fp"]
        total_fn += match_res["fn"]

        gt_canopy_mask = gt_mask > 0
        pred_canopy_mask = canopy > 0.40
        c_inter = np.logical_and(gt_canopy_mask, pred_canopy_mask).sum()
        c_union = np.logical_or(gt_canopy_mask, pred_canopy_mask).sum()
        if c_union > 0:
            canopy_ious.append(c_inter / c_union)

    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    mean_canopy_iou = float(np.mean(canopy_ious)) if canopy_ious else 0.0

    print("=" * 65)
    print("VAL SET EVALUATION RESULTS (Adaptive Neuro-Flow-Graph NFG):")
    print(f"  Total TP: {total_tp:03d}, FP: {total_fp:03d}, FN: {total_fn:03d}")
    print(f"  Instance Precision: {overall_p:.4f} ({overall_p*100:.2f}%)")
    print(f"  Instance Recall:    {overall_r:.4f} ({overall_r*100:.2f}%)")
    print(f"  Instance F1-Score:  {overall_f1:.4f}")
    print(f"  Mean Canopy IoU:    {mean_canopy_iou:.4f}")
    print("=" * 65)

    # 2. Multi-Biome Diagnostic Previews on DTE Benchmark
    meta = pd.read_csv(BENCH_DIR / "DTE-aerial-bench-meta-public-assets.csv")
    samples = [
        ("Boreal Forests/Taiga", "5cm", "01_boreal_5cm"),
        ("Boreal Forests/Taiga", "10cm", "02_boreal_10cm"),
        ("Mediterranean Forests, Woodlands, and Scrub", "5cm", "03_mediterranean_5cm"),
        ("Mediterranean Forests, Woodlands, and Scrub", "10cm", "04_mediterranean_10cm"),
        ("Temperate Broadleaf and Mixed Forests", "5cm", "05_temperate_broadleaf_5cm"),
        ("Temperate Broadleaf and Mixed Forests", "20cm", "06_temperate_broadleaf_20cm"),
        ("Temperate Coniferous Forests", "5cm", "07_temperate_coniferous_5cm"),
        ("Tropical and Subtropical Moist Broadleaf Forests", "5cm", "08_tropical_5cm"),
    ]

    print(f"\nGenerating 8 publication-grade multi-biome diagnostic preview panels in {OUT_DIR}...")
    for biome, res, prefix in samples:
        sub = meta[(meta["biome"] == biome) & (meta["resolution"] == res)]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]

        img_orig = np.array(Image.open(BENCH_DIR / row["tile_path"]).convert("RGB"))
        mask_orig = np.array(Image.open(BENCH_DIR / row["mask_path"]))
        gt_mortality = mask_orig == 2

        gt_cnts, _ = cv2.findContours(gt_mortality.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_polys = [Polygon(c.squeeze(1)) for c in gt_cnts if len(c) >= 3 and cv2.contourArea(c) >= 10]

        img_t = torch.from_numpy(img_orig.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        # 1. Flow Magnitude masked by canopy and Divergence Sinks
        flow_mag = np.linalg.norm(flow, axis=0) * canopy
        neg_div = compute_divergence_field(flow)

        # 2. Adaptive NFG Decode
        _, pred_polys = decode_neuro_flow_graph(
            flow=flow,
            canopy=canopy,
            sdt=sdt,
            centroid=centroid,
            resolution=res,
            canopy_threshold=0.40,
            sdt_threshold=-0.25,
            centroid_threshold=0.15,
            use_divergence_seeds=True,
            use_bellman_ford=True,
            use_spectral_cut=True,
        )

        # Create 5-Panel High Definition Diagnostic Figure
        fig, axes = plt.subplots(1, 5, figsize=(24, 5.2), dpi=180)

        # Panel 1: RGB
        axes[0].imshow(img_orig)
        axes[0].set_title(f"RGB ({res})\n{biome[:22]}...", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Ground Truth
        vis_gt = draw_polygons_on_image(img_orig, gt_polys, color=(0, 255, 0), thickness=2, alpha=0.3)
        axes[1].imshow(vis_gt)
        axes[1].set_title(f"Ground Truth ({len(gt_polys)} instances)\nGreen: Target Crowns", fontsize=11, color="green", fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Centripetal Flow Magnitude & Direction Field
        im3 = axes[2].imshow(flow_mag, cmap="viridis", vmin=0, vmax=1.0)
        axes[2].set_title(r"Canopy Gated Flow $|\vec{v}| \cdot c$" + "\nCentripetal Vector Dynamics", fontsize=11, fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: Analytical Vector Divergence Sinks
        im4 = axes[3].imshow(neg_div, cmap="magma", vmin=0, vmax=1.0)
        axes[3].set_title(r"Topological Sinks ($\nabla \cdot \vec{v} < 0$)" + "\nSingularity Centroids", fontsize=11, color="purple", fontweight="bold")
        axes[3].axis("off")
        plt.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)

        # Panel 5: Adaptive NFG Decoded Instance Contours
        vis_pred = draw_polygons_on_image(img_orig, pred_polys, color=(255, 120, 0), thickness=2, alpha=0.35)
        axes[4].imshow(vis_pred)
        axes[4].set_title(f"Adaptive NFG Pred ({len(pred_polys)} instances)\nOrange: Streamline Contours", fontsize=11, color="darkorange", fontweight="bold")
        axes[4].axis("off")

        plt.tight_layout()
        out_path = OUT_DIR / f"{prefix}_neuro_deadwood.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [+] Saved publication preview: {out_path.name} (GT: {len(gt_polys)}, Pred: {len(pred_polys)})", flush=True)

    results = {
        "val_evaluation": {
            "model": str(MODEL_PATH),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": overall_p,
            "recall": overall_r,
            "f1": overall_f1,
            "mean_canopy_iou": mean_canopy_iou,
        },
        "preview_directory": str(OUT_DIR),
    }
    (OUT_DIR / "evaluation_summary.json").write_text(json.dumps(results, indent=2))
    print(f"\nAll publication-grade previews successfully regenerated in: {OUT_DIR}!")


if __name__ == "__main__":
    main()
