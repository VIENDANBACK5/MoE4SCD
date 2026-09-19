"""Visual preview generator for Neuro-Flow-Graph (NFG) vs Vanilla TreeFlowNet vs StarConvex.

Renders side-by-side diagnostic overlays on key test patches illustrating:
  1. Bellman-Ford Bridging across shadow-interrupted fallen tree trunks.
  2. Spectral Modularity Separation on tangled crisscrossing deadwood piles.
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

from crown_segmentation_research.methods.tree_flow.graph_decode import decode_neuro_flow_graph
from crown_segmentation_research.methods.tree_flow.decode import decode_flow_to_instances
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet

BENCH_DIR = Path("DTE-Aerial-Data-public")
OUT_DIR = Path("crown_segmentation_research/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)
MIN_PRED_AREA = 30


def draw_polygons_on_image(image: np.ndarray, polygons: list[Polygon], color: tuple[int, int, int], thickness: int = 2) -> np.ndarray:
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

    # Pick 4 diverse representative patches
    sample_indices = [12, 16, 40, 65]

    for idx in sample_indices:
        if idx >= len(meta):
            continue
        row = meta.iloc[idx]
        image_orig = np.array(Image.open(BENCH_DIR / row["tile_path"]).convert("RGB"))
        mask_orig = np.array(Image.open(BENCH_DIR / row["mask_path"]))
        gt_mortality = mask_orig == 2

        # Extract GT contours as polygons
        gt_contours, _ = cv2.findContours(gt_mortality.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_polys = [Polygon(cnt.squeeze(1)) for cnt in gt_contours if len(cnt) >= 3 and cv2.contourArea(cnt) >= 15]

        image_f = image_orig.astype(np.float32) / 255.0
        image_t = torch.from_numpy(image_f).permute(2, 0, 1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(image_t)
            flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
            sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
            canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        # 1. Vanilla Euler Flow
        _, vanilla_polys = decode_flow_to_instances(
            flow, canopy, sdt, centroid,
            canopy_threshold=0.5, sdt_threshold=-0.2, centroid_threshold=0.25,
            min_instance_area=MIN_PRED_AREA,
        )

        # 2. Neuro-Flow-Graph (NFG) with Bridging & Spectral Cut
        _, nfg_polys = decode_neuro_flow_graph(
            flow, canopy, sdt, centroid,
            canopy_threshold=0.5, sdt_threshold=-0.2, centroid_threshold=0.25,
            min_instance_area=MIN_PRED_AREA,
            use_bellman_ford=True, use_spectral_cut=True,
        )

        # Draw visualizations
        vis_gt = draw_polygons_on_image(image_orig, gt_polys, color=(0, 255, 0), thickness=2)
        vis_vanilla = draw_polygons_on_image(image_orig, vanilla_polys, color=(0, 220, 255), thickness=2)
        vis_nfg = draw_polygons_on_image(image_orig, nfg_polys, color=(255, 140, 0), thickness=2)

        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
        axes[0].imshow(image_orig)
        axes[0].set_title(f"Input RGB ({row['biome']}, {row['resolution']})", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(vis_gt)
        axes[1].set_title(f"Ground Truth ({len(gt_polys)} instances)", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(vis_vanilla)
        axes[2].set_title(f"Vanilla TreeFlowNet ({len(vanilla_polys)} instances)", fontsize=11)
        axes[2].axis("off")

        axes[3].imshow(vis_nfg)
        axes[3].set_title(f"Neuro-Flow-Graph NFG ({len(nfg_polys)} instances)", fontsize=11)
        axes[3].axis("off")

        plt.tight_layout()
        save_path = OUT_DIR / f"nfg_ablation_{row['patch_stem']}.png"
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Saved visual diagnostic preview: {save_path}", flush=True)


if __name__ == "__main__":
    main()
