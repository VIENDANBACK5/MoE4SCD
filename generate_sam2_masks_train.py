"""
generate_sam2_masks_train.py
============================
Generate and save SAM2 binary masks for the SECOND training set.

Output:
    SECOND/sam2_masks_T1/{stem}.npz  → {"masks": (N,512,512) bool,
                                        "scores": (N,) float32,
                                        "bboxes": (N,4) int32}
    SECOND/sam2_masks_T2/{stem}.npz

Usage:
    python generate_sam2_masks_train.py --resume
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

# ── Add SAM2 repo path ────────────────────────────────────────────────────────
SAM2_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.WARNING)
log.setLevel(logging.INFO)

# ── Constants & Paths ─────────────────────────────────────────────────────────
DATA_ROOT  = Path("SECOND")
IMG_T1_DIR = DATA_ROOT / "im1"
IMG_T2_DIR = DATA_ROOT / "im2"
OUT_T1_DIR = DATA_ROOT / "sam2_masks_T1"
OUT_T2_DIR = DATA_ROOT / "sam2_masks_T2"

SAM2_CKPT   = os.path.join(SAM2_REPO, "checkpoints", "sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"

POINTS_PER_SIDE  = 32
PRED_IOU_THRESH  = 0.75
STABILITY_THRESH = 0.85
MIN_MASK_AREA    = 256


def build_amg(device: str) -> SAM2AutomaticMaskGenerator:
    log.info(f"Loading SAM2 from {SAM2_CKPT}")
    model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)
    return SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=POINTS_PER_SIDE,
        pred_iou_thresh=PRED_IOU_THRESH,
        stability_score_thresh=STABILITY_THRESH,
        min_mask_region_area=MIN_MASK_AREA,
        output_mode="binary_mask",
    )


def process_masks(raw_masks: list) -> dict:
    """Convert SAM2 output list → numpy arrays for storage."""
    if not raw_masks:
        masks  = np.ones((1, 512, 512), dtype=bool)
        scores = np.array([0.0], dtype=np.float32)
        bboxes = np.array([[0, 0, 512, 512]], dtype=np.int32)
    else:
        masks  = np.stack([m["segmentation"] for m in raw_masks], axis=0)  # (N,512,512)
        scores = np.array([m.get("predicted_iou", 0.0) for m in raw_masks], dtype=np.float32)
        bboxes = np.array([m.get("bbox", [0, 0, 512, 512]) for m in raw_masks], dtype=np.int32)
    return {"masks": masks, "scores": scores, "bboxes": bboxes}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs where both output .npz already exist")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    log.info(f"Device: {device}")

    OUT_T1_DIR.mkdir(parents=True, exist_ok=True)
    OUT_T2_DIR.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in IMG_T1_DIR.glob("*.png"))
    log.info(f"Found {len(stems)} train stems")
    assert len(stems) == 2968, f"Expected 2968 stems, got {len(stems)}"

    amg = build_amg(device)

    skipped = errors = 0

    for stem in tqdm(stems, desc="Generating SAM2 train masks"):
        out_t1 = OUT_T1_DIR / f"{stem}.npz"
        out_t2 = OUT_T2_DIR / f"{stem}.npz"

        if args.resume and out_t1.exists() and out_t2.exists():
            skipped += 1
            continue

        try:
            img_t1 = np.array(Image.open(IMG_T1_DIR / f"{stem}.png").convert("RGB"))
            img_t2 = np.array(Image.open(IMG_T2_DIR / f"{stem}.png").convert("RGB"))

            masks_t1 = amg.generate(img_t1)
            masks_t2 = amg.generate(img_t2)

            np.savez_compressed(out_t1, **process_masks(masks_t1))
            np.savez_compressed(out_t2, **process_masks(masks_t2))

        except Exception as e:
            log.warning(f"Error on {stem}: {e}")
            errors += 1

    total = len(stems) - skipped - errors
    log.info(f"Done. Generated: {total} | Skipped: {skipped} | Errors: {errors}")

    # Quick verify
    sample = np.load(OUT_T1_DIR / f"{stems[0]}.npz")
    log.info(f"Sample {stems[0]}: masks={sample['masks'].shape}, "
             f"scores={sample['scores'].shape}")
    assert sample["masks"].ndim == 3
    assert sample["masks"].shape[1:] == (512, 512)
    log.info("✅ SAM2 train masks saved.")


if __name__ == "__main__":
    main()
