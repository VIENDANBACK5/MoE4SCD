# run_sam2_deadtrees.py
"""
Chạy SAM2 automatic mask generation trên aerial images.
Dùng cấu hình chính xác từ generate_sam2_masks_test.py.
"""
import os, sys, glob
import numpy as np
import torch
import rasterio
from pathlib import Path
from tqdm import tqdm

# Setup SAM2 import path
SAM2_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

IMG_DIR  = "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
OUT_DIR  = "DeadTrees/sam2_masks"
os.makedirs(OUT_DIR, exist_ok=True)

SAM2_CKPT   = os.path.join(SAM2_REPO, "checkpoints", "sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

POINTS_PER_SIDE  = 32
PRED_IOU_THRESH  = 0.86
STABILITY_THRESH = 0.92
MIN_MASK_AREA    = 100

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM2 checkpoint from {SAM2_CKPT} onto {device}...")
    sam2_model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        min_mask_region_area=MIN_MASK_AREA,
        output_mode="binary_mask"
    )
    print("✅ SAM2 loaded successfully")

    img_files = glob.glob(f"{IMG_DIR}/**/*.tif", recursive=True)
    print(f"Processing {len(img_files)} images...")

    for img_path in tqdm(img_files):
        stem = Path(img_path).stem
        out_path = os.path.join(OUT_DIR, f"{stem}.npz")

        if os.path.exists(out_path):
            continue

        with rasterio.open(img_path) as src:
            image = src.read([1, 2, 3])  # (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, 3)

            if image.dtype != np.uint8:
                image = ((image - image.min()) /
                        (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

        try:
            with torch.inference_mode():
                masks_data = mask_generator.generate(image)
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            continue

        if not masks_data:
            # Fallback: one all-zero mask
            masks  = np.zeros((1, 1024, 1024), dtype=bool)
            scores = np.array([0.0], dtype=np.float32)
            areas  = np.array([0], dtype=np.int32)
        else:
            masks  = np.stack([m["segmentation"] for m in masks_data])
            scores = np.array([m.get("stability_score", 0.0) for m in masks_data])
            areas  = np.array([m.get("area", 0) for m in masks_data])

        np.savez_compressed(out_path, masks=masks, scores=scores, areas=areas)

    print(f"\n✅ SAM2 mask generation done.")
    n_generated = len(glob.glob(f"{OUT_DIR}/*.npz"))
    print(f"   Generated masks for {n_generated}/{len(img_files)} images")
    assert n_generated > len(img_files) * 0.8, "[FAIL] Quá nhiều ảnh không generate được mask"
    print(f"   ✅ Validation passed")

if __name__ == "__main__":
    main()
