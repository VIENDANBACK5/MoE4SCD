"""Generate visual preview comparisons: Ground Truth vs Predictions.

Compares:
1. Ground Truth (Lime Green)
2. StarConvex baseline (Yellow)
3. TreeFlowNet predictions (Cyan)
across held-out DeadTrees validation tiles and DTE benchmark patches.
Saves overlays to crown_segmentation_research/images/ per project memory rule.
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

import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPolygon

from crown_segmentation_research.methods.star_convex.decode import decode as star_convex_decode
from crown_segmentation_research.methods.star_convex.model import StarConvexNet
from crown_segmentation_research.methods.tree_flow.decode import decode_flow_to_instances
from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet

OUT_DIR = Path("crown_segmentation_research/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)
VAL_DIR = Path("DeadTrees/star_convex_targets_v1/val")

V8_CONFIG = dict(
    n_rays=16,
    prob_threshold=0.5,
    min_peak_distance=5,
    nms_iou_threshold=0.2,
    canopy_threshold=0.7,
    embedding_delta_d=3.0,
)


def get_gt_polygons_from_labelmap(label_map: np.ndarray) -> list[np.ndarray]:
    polys = []
    for label in np.unique(label_map):
        if label == 0:
            continue
        mask = (label_map == label).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if len(c) >= 3:
                polys.append(c.squeeze(1))
    return polys


def generate_previews(
    star_convex_ckpt: Path,
    tree_flow_ckpt: Path,
    device: torch.device,
    stems: list[str] | None = None,
) -> list[Path]:
    if stems is None:
        stems = [
            "dataset_5737_r00016_c00032",
            "dataset_5737_r00012_c00029",
            "dataset_5737_r00040_c00008",
        ]

    # Load StarConvex baseline
    sc_model = None
    if star_convex_ckpt.exists():
        sc_model = StarConvexNet(
            n_rays=16,
            pretrained_backbone=False,
            use_canopy_head=True,
            use_embedding_head=True,
            embedding_dim=8,
        ).to(device)
        sc_model.load_state_dict(torch.load(star_convex_ckpt, map_location=device, weights_only=True))
        sc_model.eval()

    # Load TreeFlowNet model
    tf_model = None
    if tree_flow_ckpt.exists():
        tf_model = TreeFlowNet(pretrained_backbone=False).to(device)
        tf_model.load_state_dict(torch.load(tree_flow_ckpt, map_location=device, weights_only=True))
        tf_model.eval()

    saved_paths = []

    for stem in stems:
        npz_path = VAL_DIR / f"{stem}.npz"
        if not npz_path.exists():
            continue

        data = np.load(npz_path)
        image_u8 = data["image"]
        image_f = image_u8.astype(np.float32) / 255.0
        label_map = data["instance_label"]
        gt_polys = get_gt_polygons_from_labelmap(label_map)

        image_t = torch.from_numpy(image_f).permute(2, 0, 1).float().unsqueeze(0).to(device)

        # StarConvex prediction
        sc_polygons = []
        if sc_model is not None:
            with torch.no_grad():
                out = sc_model(image_t)
                prob = out["probability"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                rays = out["rays"].squeeze(0).cpu().numpy().astype(np.float32)
                canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                embedding = out["embedding"].squeeze(0).cpu().numpy().astype(np.float32)
                sc_polygons = star_convex_decode(prob, rays, canopy=canopy, embedding=embedding, **V8_CONFIG)
                sc_polygons = [p for p in sc_polygons if p.is_valid and not p.is_empty]

        # TreeFlowNet prediction
        tf_polygons = []
        if tf_model is not None:
            with torch.no_grad():
                out = tf_model(image_t)
                flow = out["flow"].squeeze(0).cpu().numpy().astype(np.float32)
                sdt = out["sdt"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                centroid = out["centroid"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                canopy = out["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
                _, tf_polygons = decode_flow_to_instances(
                    flow, canopy, sdt, centroid,
                    canopy_threshold=0.5,
                    sdt_threshold=-0.2,
                    centroid_threshold=0.25,
                    min_instance_area=30,
                )

        # Plot 3-panel figure: RGB+GT | StarConvex | TreeFlowNet
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        # Panel 1: RGB + GT
        axes[0].imshow(image_u8)
        for poly in gt_polys:
            axes[0].add_patch(MplPolygon(poly, fill=False, edgecolor="lime", linewidth=1.6))
        axes[0].set_title(f"Ground Truth ({len(gt_polys)} crowns)", fontsize=13, color="green")
        axes[0].axis("off")

        # Panel 2: StarConvex Baseline
        axes[1].imshow(image_u8)
        for poly in gt_polys:
            axes[1].add_patch(MplPolygon(poly, fill=False, edgecolor="lime", linewidth=0.8, alpha=0.5))
        for poly in sc_polygons:
            subs = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
            for s in subs:
                coords = np.array(s.exterior.coords)
                axes[1].add_patch(MplPolygon(coords, fill=False, edgecolor="yellow", linewidth=1.4))
        axes[1].set_title(f"StarConvex Baseline ({len(sc_polygons)} pred)", fontsize=13)
        axes[1].axis("off")

        # Panel 3: TreeFlowNet (OmniCrown)
        axes[2].imshow(image_u8)
        for poly in gt_polys:
            axes[2].add_patch(MplPolygon(poly, fill=False, edgecolor="lime", linewidth=0.8, alpha=0.5))
        for poly in tf_polygons:
            subs = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
            for s in subs:
                coords = np.array(s.exterior.coords)
                axes[2].add_patch(MplPolygon(coords, fill=False, edgecolor="cyan", linewidth=1.4))
        axes[2].set_title(f"TreeFlowNet / OmniCrown ({len(tf_polygons)} pred)", fontsize=13, color="darkcyan")
        axes[2].axis("off")

        fig.suptitle(f"General Crown Segmentation: {stem}\nGreen=GT, Yellow=StarConvex, Cyan=TreeFlowNet", fontsize=15)
        out_path = OUT_DIR / f"treeflow_vs_starconvex_{stem.split('_r')[-1]}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visual preview to {out_path}", flush=True)
        saved_paths.append(out_path)

    return saved_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--star-convex-ckpt", type=Path, default=Path("DeadTrees/experiments/star_convex_scaleup_181sites/epoch_checkpoints/epoch_0065.pth"))
    parser.add_argument("--tree-flow-ckpt", type=Path, default=Path("DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0019.pth"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generate_previews(args.star_convex_ckpt, args.tree_flow_ckpt, dev)
