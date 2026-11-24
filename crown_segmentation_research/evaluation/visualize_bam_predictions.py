"""
Comprehensive BAM & Exhaustive Tree Segmentation Visualization Suite.

Generates high-resolution multi-panel visual comparisons:
1. Raw RGB Orthomosaic
2. Ground Truth Polygons (Lime Green)
3. Baseline Model (DeadTrees ep39 - failure case)
4. Newly Trained Exhaustive Model (Cyan)
5. Multi-task Intermediate Predictions:
   - Centroid Gaussian Heatmap
   - Angular Flow Field / Quiver Plot
   - Signed Distance Transform (SDT)
   - Canopy Probability Gate

Usage:
    python crown_segmentation_research/evaluation/visualize_bam_predictions.py \
        --ckpt crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth \
        --baseline-ckpt DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth \
        --num-samples 5 \
        --split eval
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import zipfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from pycocotools import mask as coco_mask_util
from pycocotools.coco import COCO

# ── Setup sys.path ────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.decode import decode, DECODE_DEFAULTS

BAM_COCO_DIR  = REPO / "data/itc_benchmarks/extracted/bam_coco2048"
BAM_ARCHIVE   = REPO / "data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"
MANIFEST_PATH = BAM_COCO_DIR / "zip_manifest.json"
OUT_DIR       = REPO / "crown_segmentation_research/images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


def _load_manifest() -> dict[str, str]:
    if MANIFEST_PATH.exists():
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    return {}


def _load_image_from_zip(zf: zipfile.ZipFile, file_name: str, manifest: dict[str, str]) -> np.ndarray:
    if file_name in manifest:
        try:
            data = zf.read(manifest[file_name])
            return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except KeyError:
            pass
    for prefix in ("", "images/", "coco2048/images/", "coco2048/val2023/", "coco2048/test2023/Test-Set-1/", "coco2048/test2023/Test-Set-2/", "coco2048/train2023/"):
        try:
            data = zf.read(prefix + file_name)
            return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        except KeyError:
            continue
    raise FileNotFoundError(f"{file_name} not found in archive")


def load_model(ckpt_path: str | Path) -> TreeFlowNet | None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"Warning: Checkpoint not found: {ckpt_path}")
        return None
    model = TreeFlowNet().to(DEVICE).eval()
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd.get("model_state_dict", sd.get("state_dict", sd))
    model.load_state_dict(state, strict=True)
    return model


def predict_on_crop(model: TreeFlowNet, img_rgb: np.ndarray, decode_cfg: dict | None = None) -> tuple[list[dict], dict[str, np.ndarray]]:
    if decode_cfg is None:
        decode_cfg = DECODE_DEFAULTS

    H, W = img_rgb.shape[:2]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = ((img_rgb.astype(np.float32) / 255.0) - mean) / std
    t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    # Pad if needed to multiple of 32
    pad_h = (32 - H % 32) % 32
    pad_w = (32 - W % 32) % 32
    if pad_h > 0 or pad_w > 0:
        t_img = F.pad(t_img, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
        preds = model(t_img)

    flow_t = preds["flow"][:, :, :H, :W]
    sdt_t  = preds["sdt"][:, :, :H, :W]
    cent_t = preds["centroid"][:, :, :H, :W]
    can_t  = preds["canopy"][:, :, :H, :W]

    instances = decode({"flow": flow_t, "sdt": sdt_t, "centroid": cent_t, "canopy": can_t}, device=DEVICE, **decode_cfg)

    raw_maps = {
        "flow": flow_t[0].cpu().numpy(),           # (2, H, W)
        "sdt": sdt_t[0, 0].cpu().numpy(),          # (H, W)
        "centroid": cent_t[0, 0].cpu().numpy(),    # (H, W)
        "canopy": can_t[0, 0].cpu().numpy(),       # (H, W)
    }
    return instances, raw_maps


def get_gt_polygons(coco: COCO, img_id: int, H: int, W: int) -> list[np.ndarray]:
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(ann_ids)
    polys = []
    for ann in anns:
        seg = ann.get("segmentation", [])
        if isinstance(seg, list):
            for poly_coords in seg:
                coords = np.array(poly_coords).reshape(-1, 2)
                if len(coords) >= 3:
                    polys.append(coords)
        elif isinstance(seg, dict):
            # RLE
            m = coco_mask_util.decode(seg)
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                if len(c) >= 3:
                    polys.append(c.squeeze(1))
    return polys


def render_comparison_figure(
    img_rgb: np.ndarray,
    gt_polys: list[np.ndarray],
    baseline_instances: list[dict] | None,
    new_instances: list[dict],
    raw_maps: dict[str, np.ndarray],
    title_suffix: str = "",
    save_path: Path | None = None,
) -> None:
    """Render a comprehensive 2x3 panel comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 13), facecolor="#111118")
    for ax in axes.flat:
        ax.set_facecolor("#0a0a0f")
        ax.axis("off")

    H, W = img_rgb.shape[:2]

    # ── Panel 1: Raw Image ───────────────────────────────────────────────────
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title("Input RGB Orthomosaic", color="white", fontsize=13, fontweight="bold", pad=8)

    # ── Panel 2: Ground Truth Annotations ────────────────────────────────────
    axes[0, 1].imshow(img_rgb)
    for p in gt_polys:
        poly_patch = MplPolygon(p, closed=True, edgecolor="#39FF14", facecolor="#39FF14", alpha=0.25, linewidth=1.5)
        axes[0, 1].add_patch(poly_patch)
    axes[0, 1].set_title(f"Ground Truth ({len(gt_polys)} live crowns)", color="#39FF14", fontsize=13, fontweight="bold", pad=8)

    # ── Panel 3: Baseline Predictions (DeadTrees ep39) ──────────────────────
    axes[0, 2].imshow(img_rgb)
    n_base = 0
    if baseline_instances is not None:
        n_base = len(baseline_instances)
        for inst in baseline_instances:
            m = inst["mask"]
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                if len(c) >= 3:
                    poly_patch = MplPolygon(c.squeeze(1), closed=True, edgecolor="#FF3366", facecolor="#FF3366", alpha=0.35, linewidth=1.5)
                    axes[0, 2].add_patch(poly_patch)
    axes[0, 2].set_title(f"Baseline: DeadTrees ep39 ({n_base} predicted - Failure)", color="#FF3366", fontsize=13, fontweight="bold", pad=8)

    # ── Panel 4: Exhaustive Model Predictions ───────────────────────────────
    axes[1, 0].imshow(img_rgb)
    for inst in new_instances:
        m = inst["mask"]
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if len(c) >= 3:
                poly_patch = MplPolygon(c.squeeze(1), closed=True, edgecolor="#00E5FF", facecolor="#00E5FF", alpha=0.30, linewidth=1.5)
                axes[1, 0].add_patch(poly_patch)
    axes[1, 0].set_title(f"TreeFlowNet Exhaustive ({len(new_instances)} crowns detected)", color="#00E5FF", fontsize=13, fontweight="bold", pad=8)

    # ── Panel 5: Centroid Heatmap + Flow Field ──────────────────────────────
    cent_map = raw_maps["centroid"]
    flow_map = raw_maps["flow"]  # (2, H, W)
    axes[1, 1].imshow(cent_map, cmap="inferno", vmin=0, vmax=1.0)
    # Subsample quiver for clean visualization
    step = max(H // 32, 8)
    y, x = np.mgrid[step // 2:H:step, step // 2:W:step]
    u = flow_map[0, y, x]
    v = flow_map[1, y, x]
    axes[1, 1].quiver(x, y, u, v, color="cyan", scale=25, width=0.003, headwidth=4, alpha=0.8)
    axes[1, 1].set_title("Centroid Heatmap & 2D Flow Field", color="#FFAA00", fontsize=13, fontweight="bold", pad=8)

    # ── Panel 6: SDT + Canopy Probability Gate ──────────────────────────────
    sdt_map = raw_maps["sdt"]
    can_map = raw_maps["canopy"]
    # Blend SDT with canopy mask
    axes[1, 2].imshow(sdt_map, cmap="viridis", vmin=-1.0, vmax=1.0)
    axes[1, 2].contour(can_map, levels=[0.5], colors=["#FF0055"], linewidths=1.5)
    axes[1, 2].set_title("SDT Distance Map + Canopy Gate Boundary", color="#A855F7", fontsize=13, fontweight="bold", pad=8)

    fig.suptitle(f"Exhaustive Crown Segmentation Verification {title_suffix}", color="white", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved figure: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth")
    parser.add_argument("--baseline-ckpt", type=str, default="DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth")
    parser.add_argument("--split", type=str, default="eval", choices=["eval", "test1", "test2", "train"])
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="crown_segmentation_research/images")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_file = f"instances_tree_{args.split}2023.json" if args.split != "test1" and args.split != "test2" else (
        "instances_tree_TestSet12023.json" if args.split == "test1" else "instances_tree_TestSet22023.json"
    )
    json_path = BAM_COCO_DIR / json_file
    print(f"Loading COCO annotations: {json_path}")
    coco = COCO(str(json_path))

    manifest = _load_manifest()
    zf = zipfile.ZipFile(BAM_ARCHIVE, "r")

    print(f"Loading newly trained model: {args.ckpt}")
    new_model = load_model(args.ckpt)
    print(f"Loading baseline model: {args.baseline_ckpt}")
    base_model = load_model(args.baseline_ckpt)

    img_ids = list(coco.imgs.keys())
    # Pick diverse sample images with varying crown counts
    np.random.seed(42)
    sample_ids = np.random.choice(img_ids, size=min(args.num_samples, len(img_ids)), replace=False)

    for i, img_id in enumerate(sample_ids):
        img_info = coco.imgs[img_id]
        fn = img_info["file_name"]
        print(f"\nProcessing sample {i+1}/{len(sample_ids)}: {fn} (ID {img_id})")

        img_rgb = _load_image_from_zip(zf, fn, manifest)
        H, W = img_rgb.shape[:2]
        # Crop central 1024x1024 or full image
        if H > 1024 or W > 1024:
            ch, cw = min(H, 1024), min(W, 1024)
            sy = (H - ch) // 2
            sx = (W - cw) // 2
            crop_rgb = img_rgb[sy:sy+ch, sx:sx+cw]
            # Offset GT polygons
            all_gt = get_gt_polygons(coco, img_id, H, W)
            gt_polys = []
            for p in all_gt:
                shifted = p - np.array([sx, sy])
                # Keep if polygon inside crop
                if np.all(shifted >= 0) and np.all(shifted < [cw, ch]):
                    gt_polys.append(shifted)
        else:
            crop_rgb = img_rgb
            gt_polys = get_gt_polygons(coco, img_id, H, W)

        # Baseline predictions
        base_instances = None
        if base_model is not None:
            base_instances, _ = predict_on_crop(base_model, crop_rgb)

        # Exhaustive model predictions
        new_instances, raw_maps = [], {"flow": np.zeros((2, 10, 10)), "sdt": np.zeros((10, 10)), "centroid": np.zeros((10, 10)), "canopy": np.zeros((10, 10))}
        if new_model is not None:
            new_instances, raw_maps = predict_on_crop(new_model, crop_rgb)

        save_name = out_dir / f"bam_eval_sample_{i+1:02d}_{Path(fn).stem}.png"
        render_comparison_figure(
            crop_rgb,
            gt_polys,
            base_instances,
            new_instances,
            raw_maps,
            title_suffix=f"[{args.split.upper()} Scene: {Path(fn).stem}]",
            save_path=save_name,
        )

    zf.close()
    print("\nVisualization generation complete!")


if __name__ == "__main__":
    main()
