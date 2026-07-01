"""
config.py — Shared configuration for SECOND-OC benchmark pipeline.
Read by all phase scripts. Edit paths here if your layout differs.
"""
import json
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("SECOND")
IMG_T1_DIR  = DATA_ROOT / "test" / "im1"
IMG_T2_DIR  = DATA_ROOT / "test" / "im2"
LAB_T1_DIR  = DATA_ROOT / "test" / "label1"
LAB_T2_DIR  = DATA_ROOT / "test" / "label2"
MASK_T1_DIR = DATA_ROOT / "sam2_masks_T1_test"
MASK_T2_DIR = DATA_ROOT / "sam2_masks_T2_test"

OUT_ROOT = Path("SECOND-OC")
ANN_DIR  = OUT_ROOT / "annotations"
MASK_INST_DIR = OUT_ROOT / "masks"  # per-instance masks saved here
TIER1_DIR = OUT_ROOT / "tier1"
EVAL_DIR  = OUT_ROOT / "eval"

# ── GT Label encoding — SECOND RGB palette ────────────────────────────────────
# Each RGB tuple maps to a class ID.
# Source: SECOND dataset paper + verified from test/label1/*.png
RGB_TO_CLASS = {
    (0,   0,   0):   0,  # background / unlabeled
    (0,   128, 0):   1,  # tree
    (128, 0,   0):   2,  # buildings
    (0,   0,   255): 3,  # water
    (128, 128, 128): 4,  # non_veg_ground (impervious surface)
    (255, 255, 255): 5,  # playground
    (0,   255, 0):   6,  # low_vegetation
    (255, 0,   0):   7,  # other / farmland (rare, maps to nearest)
}

CLASS_NAMES = {
    0: "background",
    1: "tree",
    2: "buildings",
    3: "water",
    4: "non_veg_ground",
    5: "playground",
    6: "low_vegetation",
    7: "other",
}

# Pre-built numpy palette for fast nearest-neighbor RGB matching
_PALETTE_RGB = np.array(list(RGB_TO_CLASS.keys()), dtype=np.float32)
_PALETTE_CLS = np.array(list(RGB_TO_CLASS.values()), dtype=np.int32)


def rgb_to_class(rgb_img: np.ndarray) -> np.ndarray:
    """
    Convert (H, W, 3) uint8 RGB label image → (H, W) int class IDs.
    Uses exact lookup first, then nearest-color fallback for unknowns.
    """
    H, W, _ = rgb_img.shape
    flat = rgb_img.reshape(-1, 3)
    out  = np.full(len(flat), -1, dtype=np.int32)

    # Exact lookup (fast path for known colors)
    for rgb_tuple, cls_id in RGB_TO_CLASS.items():
        match = np.all(flat == rgb_tuple, axis=1)
        out[match] = cls_id

    # Nearest-color fallback for any remaining -1 pixels
    unknown_mask = out == -1
    if unknown_mask.any():
        unk = flat[unknown_mask].astype(np.float32)
        dists = ((unk[:, None, :] - _PALETTE_RGB[None, :, :]) ** 2).sum(-1)
        out[unknown_mask] = _PALETTE_CLS[dists.argmin(-1)]

    return out.reshape(H, W)


# ── Hyperparameters ───────────────────────────────────────────────────────────
MIN_AREA_PX          = 100    # drop SAM2 masks smaller than this (pixels)
DOMINANT_CLASS_THRESH = 0.5   # mask needs >= 50% single class to be "clean"
T2_CONF_MIN          = 0.4   # minimum class confidence for T2 lookup
DISAPPEARED_THRESH   = 0.8   # if >80% of T2 pixels are background → disappeared
