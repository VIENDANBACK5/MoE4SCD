"""Diverse Preview Generator for Neuro-Flow-Graph (NFG) with GSD-Adaptive Scaling & Divergence.

Groups all previews into a dedicated subfolder:
  crown_segmentation_research/images/previews_adaptive_nfg/

Tests across diverse scene types from DeadTrees / DTE Benchmark:
  1. Boreal Forests / Taiga standing dead snags
  2. Mediterranean Woodlands sparse deadwood
  3. Temperate Broadleaf faint skeletal dead crowns
  4. Temperate Coniferous dense mortality
  5. Tropical Moist Broadleaf canopy gaps
  6. 5cm, 10cm, and 20cm ground sampling distances
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

from crown_segmentation_research.methods.tree_flow.graph_decode import (
    compute_divergence_field,
    decode_neuro_flow_graph,
)
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet

BENCH_DIR = Path("DTE-Aerial-Data-public")
OUT_DIR = Path("crown_segmentation_research/images/previews_adaptive_nfg")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw_polygons_on_image(
    image: np.ndarray,
    polygons: list[Polygon],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    canvas = image.copy()
    for poly in polygons:
        subs = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
        for sub in subs:
            pts = np.array(sub.exterior.coords).round().astype(np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)
    return canvas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(BENCH_DIR / "DTE-aerial-bench-meta-public-assets.csv")

    ckpt_path = Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    model = TreeFlowNet(pretrained_backbone=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # Select 8 diverse scenes representing every major biome and resolution
    samples = [
        # (biome, resolution, index_hint)
        ("Boreal Forests/Taiga", "5cm"),
        ("Boreal Forests/Taiga", "10cm"),
        ("Mediterranean Forests, Woodlands, and Scrub", "5cm"),
        ("Mediterranean Forests, Woodlands, and Scrub", "10cm"),
        ("Temperate Broadleaf and Mixed Forests", "5cm"),
        ("Temperate Broadleaf and Mixed Forests", "20cm"),
        ("Temperate Coniferous Forests", "5cm"),
        ("Tropical and Subtropical Moist Broadleaf Forests", "5cm"),
    ]

    selected_rows = []
    for biome, res in samples:
        sub = meta[(meta["biome"] == biome) & (meta["resolution"] == res)]
        if len(sub) > 0:
            selected_rows.append(sub.iloc[0])

    print(f"Generating {len(selected_rows)} diverse diagnostic previews in: {OUT_DIR}", flush=True)

    for i, row in enumerate(selected_rows, 1):
        image_orig = np.array(Image.open(BENCH_DIR / row["tile_path"]).convert("RGB"))
        mask_orig = np.array(Image.open(BENCH_DIR / row["mask_path"]))
        gt_mortality = mask_orig == 2

        gt_contours, _ = cv2.findContours(gt_mortality.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_polys = [Polygon(cnt.squeeze(1)) for cnt in gt_contours if len(cnt) >= 3 and cv2.contourArea(cnt) >= 10]

        image_f = image_orig.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_f).permute(2, 0, 1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(image_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        # 1. Divergence Map
        neg_div = compute_divergence_field(flow)

        # 2. Adaptive NFG Decode
        _, nfg_polys = decode_neuro_flow_graph(
            flow, canopy, sdt, centroid,
            resolution=row["resolution"],
            canopy_threshold=0.40,
            sdt_threshold=-0.25,
            centroid_threshold=0.15,
            use_divergence_seeds=True,
            use_bellman_ford=True,
            use_spectral_cut=True,
        )

        vis_gt = draw_polygons_on_image(image_orig, gt_polys, color=(0, 255, 0), thickness=2)
        vis_nfg = draw_polygons_on_image(image_orig, nfg_polys, color=(255, 140, 0), thickness=2)

        fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), dpi=150)

        axes[0].imshow(image_orig)
        axes[0].set_title(f"RGB ({row['biome'][:20]}.., {row['resolution']})", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(vis_gt)
        axes[1].set_title(f"Ground Truth ({len(gt_polys)} instances)", fontsize=11, color="green", fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(neg_div, cmap="magma")
        axes[2].set_title(r"Divergence Sinks $\nabla \cdot \vec{v} < 0$", fontsize=11, color="purple", fontweight="bold")
        axes[2].axis("off")

        axes[3].imshow(vis_nfg)
        axes[3].set_title(f"Adaptive NFG Pred ({len(nfg_polys)} instances)", fontsize=11, color="darkorange", fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        clean_biome = row["biome"].split()[0].replace(",", "")
        save_path = OUT_DIR / f"{i:02d}_{clean_biome}_{row['resolution']}_{row['patch_stem']}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"  [{i}/{len(selected_rows)}] Saved: {save_path.name}", flush=True)

    print(f"\nAll diverse preview images successfully saved in {OUT_DIR}/")


if __name__ == "__main__":
    main()
