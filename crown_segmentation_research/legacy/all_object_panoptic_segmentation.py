"""Omni-Forest Panoptic & All-Object Instance Segmentation.

Combines:
  1. Meta SAM 2.1 Hiera-Large (Foundation All-Object Zero-Shot Mask Generator)
  2. TreeFlowNet / NFG Vector Dynamics & Mortality Discriminator
  3. Full-Scene Panoptic Tessellation (Every single live tree, dead tree, fallen log, and gap)

Saves publication-ready multi-panel panoptic previews into:
  crown_segmentation_research/images/previews_all_objects_panoptic/
"""
from __future__ import annotations

import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

import sys
sys.path.insert(0, "sam2")

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.graph_decode import compute_divergence_field

BENCH_DIR = Path("DTE-Aerial-Data-public")
FLOW_MODEL_PATH = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CKPT = "sam2/checkpoints/sam2.1_hiera_large.pt"
OUT_DIR = Path("crown_segmentation_research/images/previews_all_objects_panoptic")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_panoptic_masks(
    image: np.ndarray,
    masks: list[dict],
    alpha: float = 0.50,
) -> np.ndarray:
    """Renders every single segmented mask in a unique vibrant color with clean boundary."""
    canvas = image.copy()
    overlay = image.copy()
    np.random.seed(42)

    # Sort masks by area descending so smaller objects appear on top
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    colors = [
        tuple(int(c) for c in np.random.randint(30, 255, size=3))
        for _ in range(max(len(sorted_masks), 200))
    ]

    for i, m_dict in enumerate(sorted_masks):
        seg = m_dict["segmentation"].astype(bool)
        color = colors[i % len(colors)]
        overlay[seg] = color
        
        # Boundary outline
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.polylines(canvas, [cnt], isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    return canvas


def draw_semantic_breakdown(
    image: np.ndarray,
    masks: list[dict],
    deadwood_map: np.ndarray,
    canopy_map: np.ndarray,
    alpha: float = 0.55,
) -> np.ndarray:
    """Classifies every object into Live Crown (Green), Deadwood (Red/Orange), or Gap (Blue)."""
    canvas = image.copy()
    overlay = image.copy()

    for m_dict in masks:
        seg = m_dict["segmentation"].astype(bool)
        area = m_dict["area"]
        if area < 15:
            continue
            
        mean_dead = float(deadwood_map[seg].mean()) if seg.sum() > 0 else 0.0
        mean_canopy = float(canopy_map[seg].mean()) if seg.sum() > 0 else 0.0

        if mean_dead > 0.20:
            # Dead Tree / Deadwood (Bright Orange / Red)
            color = (255, 50, 30)
            boundary_color = (255, 220, 0)
            thickness = 2
        elif mean_canopy > 0.30 or mean_dead <= 0.05:
            # Live Tree Crown (Forest Green / Emerald)
            color = (30, 180, 60)
            boundary_color = (200, 255, 200)
            thickness = 1
        else:
            # Canopy Gap / Ground (Cyan / Blue)
            color = (40, 120, 220)
            boundary_color = (180, 220, 255)
            thickness = 1

        overlay[seg] = color
        contours, _ = cv2.findContours(seg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            cv2.polylines(canvas, [cnt], isClosed=True, color=boundary_color, thickness=thickness)

    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize SAM 2.1 Hiera-Large
    print("Loading SAM 2.1 Hiera-Large...")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.75,
        stability_score_thresh=0.82,
        min_mask_region_area=20,
    )
    print("SAM 2.1 Mask Generator ready!")

    # 2. Initialize TreeFlowNet
    print("Loading TreeFlowNet...")
    flow_model = TreeFlowNet(pretrained_backbone=False).to(device)
    flow_model.load_state_dict(torch.load(FLOW_MODEL_PATH, map_location=device, weights_only=True))
    flow_model.eval()

    meta = pd.read_csv(BENCH_DIR / "DTE-aerial-bench-meta-public-assets.csv")
    samples = [
        ("Tropical and Subtropical Moist Broadleaf Forests", "5cm", "01_tropical_all_objects"),
        ("Temperate Coniferous Forests", "5cm", "02_temperate_conifer_all_objects"),
        ("Temperate Broadleaf and Mixed Forests", "5cm", "03_temperate_broadleaf_all_objects"),
        ("Boreal Forests/Taiga", "5cm", "04_boreal_taiga_all_objects"),
        ("Mediterranean Forests, Woodlands, and Scrub", "5cm", "05_mediterranean_all_objects"),
        ("Temperate Broadleaf and Mixed Forests", "20cm", "06_temperate_20cm_all_objects"),
        ("Boreal Forests/Taiga", "10cm", "07_boreal_10cm_all_objects"),
        ("Mediterranean Forests, Woodlands, and Scrub", "10cm", "08_mediterranean_10cm_all_objects"),
    ]

    print(f"\nRunning All-Object Panoptic Segmentation across {len(samples)} diverse scenes...")

    for biome, res, prefix in samples:
        sub = meta[(meta["biome"] == biome) & (meta["resolution"] == res)]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]

        img_path = BENCH_DIR / row["tile_path"]
        img_orig = np.array(Image.open(img_path).convert("RGB"))

        # 1. SAM2 Automatic All-Object Segmentation
        print(f"\nProcessing {prefix} ({biome}, {res})...")
        masks = mask_generator.generate(img_orig)
        print(f"  [+] SAM2 detected {len(masks)} individual objects across the scene!")

        # 2. TreeFlowNet inference for semantic guidance
        img_t = torch.from_numpy(img_orig.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            out = flow_model(img_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        neg_div = compute_divergence_field(flow)

        # 3. Render Visualizations
        vis_all_objects = draw_panoptic_masks(img_orig, masks, alpha=0.45)
        vis_semantic = draw_semantic_breakdown(img_orig, masks, deadwood_map=centroid + neg_div, canopy_map=canopy, alpha=0.50)

        # 4-Panel Master Panoptic Diagnostic Figure
        fig, axes = plt.subplots(1, 4, figsize=(24, 5.8), dpi=180)

        # Panel 1: RGB
        axes[0].imshow(img_orig)
        axes[0].set_title(f"RGB Forest Scene ({res})\n{biome[:24]}...", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Continuous Flow Field & Topological Sinks
        flow_mag = np.linalg.norm(flow, axis=0)
        im2 = axes[1].imshow(flow_mag, cmap="viridis", vmin=0, vmax=1.0)
        axes[1].set_title(r"Topological Sinks ($\nabla \cdot \vec{v} < 0$)" + "\nTree Apex Flow Dynamics", fontsize=11, color="purple", fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: SAM2 All-Object Panoptic Tessellation (Every single tree/object)
        axes[2].imshow(vis_all_objects)
        axes[2].set_title(f"All-Object Panoptic Segmentation\n{len(masks)} Individual Objects Segmented", fontsize=11, color="navy", fontweight="bold")
        axes[2].axis("off")

        # Panel 4: Semantic Forest Breakdown (Live Trees vs Deadwood vs Gaps)
        axes[3].imshow(vis_semantic)
        axes[3].set_title("Semantic Forest Classification\nGreen: Live Trees | Red: Deadwood | Blue: Gaps", fontsize=11, color="darkgreen", fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        out_path = OUT_DIR / f"{prefix}_all_objects_panoptic.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  --> Saved Master Panoptic Figure: {out_path.name}", flush=True)

    print(f"\nAll All-Object Panoptic Segmentation figures successfully saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
