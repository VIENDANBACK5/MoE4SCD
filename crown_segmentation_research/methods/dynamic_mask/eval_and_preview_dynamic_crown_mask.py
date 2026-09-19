"""Inference and High-Resolution Diagnostic Previews for DynamicCrownNet (Deep Learning Panoptic/Instance Segmentation).

Evaluates the trained Dynamic Convolution Mask Network on diverse biomes and saves 4-panel previews to:
  crown_segmentation_research/images/previews_dynamic_crown_mask/
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
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from shapely.geometry import Polygon

from crown_segmentation_research.methods.dynamic_mask.train_dynamic_crown_mask import DynamicCrownNet

BENCH_DIR = Path("DTE-Aerial-Data-public")
MODEL_PATH = Path("DeadTrees/experiments/dynamic_crown_mask_v1/best_dynamic_crown_model.pth")
OUT_DIR = Path("crown_segmentation_research/images/previews_dynamic_crown_mask")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def decode_dynamic_crown_instances(
    model_output: dict[str, torch.Tensor],
    img_shape: tuple[int, int],
    centroid_threshold: float = 0.20,
    min_peak_distance: int = 5,
    min_area: int = 20,
) -> tuple[np.ndarray, list[Polygon]]:
    """Decodes Deep Learning Dynamic Kernels on P2 features into individual tree crown instances."""
    H, W = img_shape
    H4, W4 = H // 4, W // 4

    mask_feats = model_output["mask_feats"]  # (1, 32, H/4, W/4)
    kernels = model_output["kernels"]        # (1, 32, H/8, W/8)
    centroid = model_output["centroid"].squeeze().cpu().numpy()  # (H, W)
    canopy = model_output["canopy"].squeeze().cpu().numpy()      # (H, W)

    # Normalized Apex Energy Surface
    c_min, c_max = centroid.min(), centroid.max()
    centroid_norm = (centroid - c_min) / (c_max - c_min + 1e-6)
    can_min, can_max = canopy.min(), canopy.max()
    canopy_norm = (canopy - can_min) / (can_max - can_min + 1e-6)

    apex_energy = 0.60 * centroid_norm + 0.40 * canopy_norm
    fg_mask = apex_energy >= 0.15

    # 1. Multi-scale peak picking for tree apex discovery
    peak_filter = ndimage.maximum_filter(apex_energy, size=min_peak_distance * 2 + 1)
    peaks = (apex_energy == peak_filter) & (apex_energy >= centroid_threshold) & fg_mask
    seed_ys, seed_xs = np.nonzero(peaks)

    if len(seed_ys) == 0:
        return np.zeros((H, W), dtype=np.int32), []

    # 2. Dynamic Kernel Mask Projection
    instance_map = np.zeros((H, W), dtype=np.int32)
    polygons = []
    next_id = 1

    # Coordinate grid for spatial localization
    ys, xs = torch.meshgrid(torch.linspace(-1, 1, H4), torch.linspace(-1, 1, W4), indexing="ij")
    grid_y = ys.unsqueeze(0).to(mask_feats.device)  # (1, H4, W4)
    grid_x = xs.unsqueeze(0).to(mask_feats.device)  # (1, H4, W4)

    # Sort seeds by apex energy descending so prominent trees are processed first
    seed_probs = [apex_energy[y, x] for y, x in zip(seed_ys, seed_xs)]
    sorted_indices = np.argsort(-np.array(seed_probs))

    # Master claimed pixel tracker to enforce panoptic non-overlap
    claimed = np.zeros((H, W), dtype=bool)

    for idx in sorted_indices:
        sy, sx = seed_ys[idx], seed_xs[idx]
        ky = int(np.clip(round(sy / 8), 0, kernels.shape[2] - 1))
        kx = int(np.clip(round(sx / 8), 0, kernels.shape[3] - 1))
        w = kernels[0, :, ky, kx]  # (32,) dynamic conv weights

        # Dynamic dot-product: (1, 32, H4, W4) * (32, 1, 1) -> (H4, W4)
        dot_prod = (mask_feats[0] * w.view(-1, 1, 1)).sum(dim=0)

        # Spatial Gaussian prior centered at seed
        yn = (sy / H) * 2 - 1
        xn = (sx / W) * 2 - 1
        r2 = (grid_y - yn)**2 + (grid_x - xn)**2
        spatial_prior = torch.exp(-r2 / 0.03)

        raw_logit = dot_prod + 3.0 * spatial_prior.squeeze(0)
        prob_h4 = torch.sigmoid(raw_logit)

        # Upsample mask to full resolution (512x512)
        prob_full = F.interpolate(prob_h4.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze().cpu().numpy()

        inst_mask = (prob_full >= 0.40) & fg_mask & (~claimed)
        if inst_mask.sum() < min_area:
            continue

        # Extract contour polygon
        cnts, _ = cv2.findContours(inst_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            if len(cnt) >= 3:
                pts = cnt.squeeze(1)
                poly = Polygon(pts)
                if poly.is_valid and poly.area >= min_area:
                    polygons.append(poly)
                    instance_map[inst_mask] = next_id
                    claimed |= inst_mask
                    next_id += 1

    return instance_map, polygons


def render_4panel_deep_learning_preview(
    image: np.ndarray,
    centroid_map: np.ndarray,
    canopy_map: np.ndarray,
    instance_map: np.ndarray,
    polygons: list[Polygon],
    biome_title: str,
    out_path: Path,
):
    """Renders 4-panel publication diagnostics for DynamicCrownNet."""
    H, W = image.shape[:2]
    np.random.seed(42)
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids != 0]

    colors = [tuple(int(c) for c in np.random.randint(40, 255, size=3)) for _ in range(max(len(unique_ids) + 10, 400))]

    overlay = image.copy()
    canvas = image.copy()

    for i, uid in enumerate(unique_ids):
        m = (instance_map == uid)
        color = colors[i % len(colors)]
        overlay[m] = color
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            cv2.polylines(canvas, [cnt], isClosed=True, color=(255, 255, 255), thickness=1)

    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

    fig, axes = plt.subplots(1, 4, figsize=(24, 5.8), dpi=180)

    # Panel 1: RGB Image
    axes[0].imshow(image)
    axes[0].set_title(f"RGB Aerial Forest Scene\n{biome_title}", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Deep Learning Apex Centroid Heatmap
    im2 = axes[1].imshow(centroid_map, cmap="magma", vmin=0, vmax=1.0)
    axes[1].set_title(r"Deep Learning Apex Centroid Map" + f"\nDetected {len(polygons)} Individual Tree Apices", fontsize=11, color="darkred", fontweight="bold")
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Deep Learning Canopy Probability
    im3 = axes[2].imshow(canopy_map, cmap="viridis", vmin=0, vmax=1.0)
    axes[2].set_title("Deep Learning Canopy Cover Probability\nFull-Canopy Extent Gating", fontsize=11, color="navy", fontweight="bold")
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    # Panel 4: Deep Learning Dynamic Kernel Panoptic Crown Tessellation
    axes[3].imshow(canvas)
    axes[3].set_title(f"Deep Learning DynamicCrownNet (Zero SAM)\n{len(polygons)} Individual Tree Crowns Segmented", fontsize=11, color="darkgreen", fontweight="bold")
    axes[3].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading DynamicCrownNet on {device}...")
    model = DynamicCrownNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    print("DynamicCrownNet successfully loaded!")

    meta = pd.read_csv(BENCH_DIR / "DTE-aerial-bench-meta-public-assets.csv")
    samples = [
        ("Tropical and Subtropical Moist Broadleaf Forests", "5cm", "01_tropical_dynamic_dl"),
        ("Temperate Coniferous Forests", "5cm", "02_temperate_conifer_dynamic_dl"),
        ("Temperate Broadleaf and Mixed Forests", "5cm", "03_temperate_broadleaf_dynamic_dl"),
        ("Boreal Forests/Taiga", "5cm", "04_boreal_taiga_dynamic_dl"),
        ("Mediterranean Forests, Woodlands, and Scrub", "5cm", "05_mediterranean_dynamic_dl"),
        ("Temperate Broadleaf and Mixed Forests", "20cm", "06_temperate_20cm_dynamic_dl"),
        ("Boreal Forests/Taiga", "10cm", "07_boreal_10cm_dynamic_dl"),
        ("Mediterranean Forests, Woodlands, and Scrub", "10cm", "08_mediterranean_10cm_dynamic_dl"),
    ]

    print(f"\nRunning Deep Learning Dynamic Mask Inference across {len(samples)} diverse biomes...")

    for biome, res, prefix in samples:
        sub = meta[(meta["biome"] == biome) & (meta["resolution"] == res)]
        if len(sub) == 0:
            continue
        row = sub.iloc[0]
        img_path = BENCH_DIR / row["tile_path"]
        img_orig = np.array(Image.open(img_path).convert("RGB"))
        H, W = img_orig.shape[:2]

        img_t = torch.from_numpy(img_orig.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_t)

        inst_map, polys = decode_dynamic_crown_instances(preds, img_shape=(H, W))

        centroid_np = preds["centroid"].squeeze().cpu().numpy()
        canopy_np = preds["canopy"].squeeze().cpu().numpy()

        out_path = OUT_DIR / f"{prefix}_preview.png"
        render_4panel_deep_learning_preview(
            image=img_orig, centroid_map=centroid_np, canopy_map=canopy_np,
            instance_map=inst_map, polygons=polys,
            biome_title=f"{biome} ({res})", out_path=out_path
        )
        print(f"  [+] {prefix} ({biome[:22]}, {res}) -> Segmented {len(polys)} individual crowns! (Saved: {out_path.name})")

    print(f"\nAll Deep Learning Dynamic Crown Previews saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()
