"""Full-Scene Panoptic All-Tree-Crown Instance Segmentation (OmniCrown / TreeFlowNet).

Segments EVERY individual tree in the image (all live trees + all dead trees)
using continuous centripetal flow dynamics and Lagrangian streamline watershed.

Outputs color-coded panoptic crown instances to:
  crown_segmentation_research/images/previews_all_canopy_crowns/
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

from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from shapely.geometry import MultiPolygon, Polygon

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.graph_decode import (
    compute_divergence_field,
    decode_neuro_flow_graph,
)

BENCH_DIR = Path("DTE-Aerial-Data-public")
MODEL_PATH = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
OUT_DIR = Path("crown_segmentation_research/images/previews_all_canopy_crowns")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_panoptic_crowns_colored(
    image: np.ndarray,
    polygons: list[Polygon],
    alpha: float = 0.45,
) -> np.ndarray:
    """Renders every individual tree crown in a distinct vibrant color with clean boundary."""
    canvas = image.copy()
    overlay = image.copy()
    np.random.seed(42) # deterministic vibrant colors

    # Generate distinct bright colors for every tree instance
    colors = [
        tuple(int(c) for c in np.random.randint(40, 255, size=3))
        for _ in range(max(len(polygons), 100))
    ]

    for i, poly in enumerate(polygons):
        color = colors[i % len(colors)]
        subs = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for sub in subs:
            pts = np.array(sub.exterior.coords).round().astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color=color)
                cv2.polylines(canvas, [pts], isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Model
    model = TreeFlowNet(pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print(f"Loaded high-precision TreeFlowNet from {MODEL_PATH}")

    meta = pd.read_csv(BENCH_DIR / "DTE-aerial-bench-meta-public-assets.csv")

    # Select 8 diverse scenes
    samples = [
        ("Tropical and Subtropical Moist Broadleaf Forests", "5cm", "01_tropical_dense_canopy"),
        ("Temperate Coniferous Forests", "5cm", "02_temperate_conifer_forest"),
        ("Temperate Broadleaf and Mixed Forests", "5cm", "03_temperate_broadleaf_forest"),
        ("Boreal Forests/Taiga", "5cm", "04_boreal_taiga_dense"),
        ("Mediterranean Forests, Woodlands, and Scrub", "5cm", "05_mediterranean_woodland"),
        ("Temperate Broadleaf and Mixed Forests", "20cm", "06_temperate_20cm_aerial"),
        ("Boreal Forests/Taiga", "10cm", "07_boreal_10cm_forest"),
        ("Mediterranean Forests, Woodlands, and Scrub", "10cm", "08_mediterranean_10cm_forest"),
    ]

    print(f"\nGenerating Full-Canopy Panoptic Tree Crown Segmentation for {len(samples)} scenes...")

    for biome, res, prefix in samples:
        sub = meta[(meta["biome"] == biome) & (meta["resolution"] == res)]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]

        img_path = BENCH_DIR / row["tile_path"]
        img_orig = np.array(Image.open(img_path).convert("RGB"))

        img_t = torch.from_numpy(img_orig.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        flow_mag = np.linalg.norm(flow, axis=0) * canopy
        neg_div = compute_divergence_field(flow)

        # Full Canopy All-Tree-Crown Decode (covers all live trees + dead trees)
        # Using canopy threshold = 0.20 to capture the whole forest canopy
        _, all_crown_polys = decode_neuro_flow_graph(
            flow=flow,
            canopy=canopy,
            sdt=sdt,
            centroid=centroid,
            resolution=res,
            canopy_threshold=0.20,
            sdt_threshold=-0.40,
            centroid_threshold=0.08,
            use_divergence_seeds=True,
            use_bellman_ford=True,
            use_spectral_cut=True,
        )

        vis_panoptic = draw_panoptic_crowns_colored(img_orig, all_crown_polys, alpha=0.45)

        # 4-Panel High-Quality Panoptic Diagnostic Figure
        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=180)

        # Panel 1: Original RGB Image
        axes[0].imshow(img_orig)
        axes[0].set_title(f"RGB Forest Aerial ({res})\n{biome[:24]}...", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Continuous Centripetal Flow Field
        im2 = axes[1].imshow(flow_mag, cmap="viridis", vmin=0, vmax=1.0)
        axes[1].set_title(r"Centripetal Flow Field $|\vec{v}| \cdot c$" + "\nEvery Tree Apex Sinks Inward", fontsize=11, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        # Panel 3: Divergence Sinks (Every individual tree center detected)
        im3 = axes[2].imshow(neg_div * canopy, cmap="magma", vmin=0, vmax=0.8)
        axes[2].set_title(r"Tree Apex Sinks ($\nabla \cdot \vec{v} < 0$)" + "\nTopological Tree Centers", fontsize=11, color="purple", fontweight="bold")
        axes[2].axis("off")
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

        # Panel 4: Full-Scene Panoptic Tree Crown Segmentation (Every single tree)
        axes[3].imshow(vis_panoptic)
        axes[3].set_title(f"OmniCrown Panoptic Segmentation\n{len(all_crown_polys)} Individual Trees Segmented", fontsize=11, color="darkgreen", fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        out_path = OUT_DIR / f"{prefix}_panoptic_all_trees.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  [+] Saved Panoptic Preview: {out_path.name} -> {len(all_crown_polys)} individual tree crowns segmented!", flush=True)

    print(f"\nAll Full-Canopy Panoptic Tree Crown previews saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
