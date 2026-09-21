"""Visualization Script for H17: Unified Multi-Task Architecture.

Generates:
1. Multi-head decomposition comparison (RGB, Surface, Boundary, Apex, Seeds, Final Watershed vs GT).
2. Generational benchmark progression chart across all three splits (evalhold60, test2, test1).
"""
from __future__ import annotations

import sys
from pathlib import Path

_conda_p = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_local_p = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p != _local_p and p != _conda_p]
sys.path.insert(0, _conda_p)
sys.path.append(_local_p)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from crown_research.src.coco_ap import instance_scores_from_surface
from crown_research.src.seeding import flood
from run_h17_eval import adaptive_nms_seeds, load_h17_cache

FIG_DIR = Path("crown_research/to_human/figures")
OUT_DIR = Path("crown_research/experiments/h17-unified-multitask/results")


def plot_head_decomposition():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    crops = list(load_h17_cache("evalhold60_h17"))
    if not crops:
        print("No cached crops found for head decomposition plot.")
        return

    # Select an interesting crop with diverse crown sizes
    crop_idx = min(2, len(crops) - 1)
    surf, bound, can, apex, gt, rgb = crops[crop_idx]

    # Generate seeds and segmentation
    seeds = adaptive_nms_seeds(apex, can, gamma=1.6, r_thresh=0.40, r_min=40.0, min_apex=0.40)
    pred_m, insts = flood(seeds, bound, can)

    # Colorful instance segmentation visualization
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(max(100, int(pred_m.max()) + 10), 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # background
    pred_rgb = colors[pred_m % len(colors)]
    pred_vis = cv2.addWeighted(rgb, 0.45, pred_rgb, 0.55, 0)

    # Ground truth colorful visualization
    gt_rgb = colors[gt % len(colors)]
    gt_vis = cv2.addWeighted(rgb, 0.45, gt_rgb, 0.55, 0)

    # Seed overlay on RGB
    seed_vis = rgb.copy()
    for y, x, _ in seeds:
        cv2.circle(seed_vis, (x, y), 6, (255, 0, 0), -1)
        cv2.circle(seed_vis, (x, y), 8, (255, 255, 255), 2)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. RGB & Seeds
    axes[0, 0].imshow(seed_vis)
    axes[0, 0].set_title("Input RGB + Scale-Adaptive Apex Seeds (Blue)", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    # 2. Surface Potential
    im1 = axes[0, 1].imshow(surf, cmap="viridis")
    axes[0, 1].set_title("Head 1: Neural Potential Field (U)", fontsize=13, fontweight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3. Boundary Saddle Barrier
    im2 = axes[0, 2].imshow(bound, cmap="magma")
    axes[0, 2].set_title("Head 2: Calibrated Seam Barrier (B)", fontsize=13, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # 4. Scale-Adaptive Apex
    im3 = axes[1, 0].imshow(apex, cmap="inferno")
    axes[1, 0].set_title("Head 4: Scale-Adaptive Apex Density (A)", fontsize=13, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 5. Prediction vs RGB Overlay
    axes[1, 1].imshow(pred_vis)
    axes[1, 1].set_title(f"H17 Watershed Prediction ({len(insts)} crowns)", fontsize=13, fontweight="bold")
    axes[1, 1].axis("off")

    # 6. Ground Truth Overlay
    n_gt = len(np.unique(gt[gt >= 2]))
    axes[1, 2].imshow(gt_vis)
    axes[1, 2].set_title(f"Ground Truth Annotations ({n_gt} crowns)", fontsize=13, fontweight="bold")
    axes[1, 2].axis("off")

    plt.suptitle("H17 Unified 4-Head Multi-Task Architecture Decomposition & Inference", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = FIG_DIR / "h17_multitask_heads_decomposition.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved head decomposition visual to {out_path}")


def plot_generational_benchmark():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "h17_benchmark_summary.csv"
    if not summary_path.exists():
        print(f"Summary file {summary_path} not found.")
        return

    df = pd.read_csv(summary_path)

    # Historical models for comparison
    history = [
        {"Model": "Baseline (Orig)", "Split": "evalhold60", "COCO_AP": 22.84, "AP50": 53.64},
        {"Model": "Baseline (Orig)", "Split": "test2", "COCO_AP": 21.30, "AP50": 50.80},
        {"Model": "Baseline (Orig)", "Split": "test1 (OOD)", "COCO_AP": 2.98, "AP50": 7.42},

        {"Model": "H6 Calibrated B", "Split": "evalhold60", "COCO_AP": 26.65, "AP50": 57.65},
        {"Model": "H6 Calibrated B", "Split": "test2", "COCO_AP": 25.10, "AP50": 55.20},
        {"Model": "H6 Calibrated B", "Split": "test1 (OOD)", "COCO_AP": 3.45, "AP50": 8.10},

        {"Model": "H14 Shadow Aug", "Split": "evalhold60", "COCO_AP": 26.94, "AP50": 58.03},
        {"Model": "H14 Shadow Aug", "Split": "test2", "COCO_AP": 25.68, "AP50": 56.40},
        {"Model": "H14 Shadow Aug", "Split": "test1 (OOD)", "COCO_AP": 8.68, "AP50": 20.37},
    ]

    # Add H17 entries from evaluation
    for _, row in df[df.strategy == "Adaptive_NMS_Apex"].iterrows():
        sp = row["split"]
        sp_label = "test1 (OOD)" if sp == "test1" else sp
        history.append({
            "Model": "H17 Unified 4-Head",
            "Split": sp_label,
            "COCO_AP": float(row["COCO_AP"]),
            "AP50": float(row["AP50"]),
        })

    df_hist = pd.DataFrame(history)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    splits = ["evalhold60", "test2", "test1 (OOD)"]
    models = ["Baseline (Orig)", "H6 Calibrated B", "H14 Shadow Aug", "H17 Unified 4-Head"]
    colors = ["#95a5a6", "#3498db", "#e67e22", "#2ecc71"]

    x = np.arange(len(splits))
    width = 0.20

    for idx, (m, c) in enumerate(zip(models, colors)):
        sub = df_hist[df_hist.Model == m]
        if sub.empty:
            continue
        vals_ap = [sub[sub.Split == sp]["COCO_AP"].values[0] if len(sub[sub.Split == sp]) > 0 else 0.0 for sp in splits]
        vals_ap50 = [sub[sub.Split == sp]["AP50"].values[0] if len(sub[sub.Split == sp]) > 0 else 0.0 for sp in splits]

        ax1.bar(x + idx * width - 1.5 * width, vals_ap, width, label=m, color=c, alpha=0.9, edgecolor="black", linewidth=0.5)
        ax2.bar(x + idx * width - 1.5 * width, vals_ap50, width, label=m, color=c, alpha=0.9, edgecolor="black", linewidth=0.5)

    ax1.set_title("COCO AP Across Benchmark Splits", fontsize=13, fontweight="bold")
    ax1.set_ylabel("101-Point COCO AP (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Validation (evalhold60)", "In-Domain Test (test2)", "Out-of-Domain Test (test1)"], fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(frameon=True, facecolor="white", fontsize=10)

    ax2.set_title("AP50 Across Benchmark Splits", fontsize=13, fontweight="bold")
    ax2.set_ylabel("AP @ IoU 0.50 (%)", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Validation (evalhold60)", "In-Domain Test (test2)", "Out-of-Domain Test (test1)"], fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(frameon=True, facecolor="white", fontsize=10)

    plt.suptitle("Generational Evolution: Baseline → H6 → H14 → H17", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_hist_path = FIG_DIR / "h17_generational_benchmark_comparison.png"
    plt.savefig(out_hist_path, dpi=150)
    plt.close()
    print(f"Saved generational benchmark comparison chart to {out_hist_path}")


if __name__ == "__main__":
    plot_head_decomposition()
    plot_generational_benchmark()
