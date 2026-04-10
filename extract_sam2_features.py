"""
Stage 1: Offline Feature Extraction using SAM2 Encoder
=====================================================
Extracts image embeddings from SAM2 (Hiera-Large) for every image pair in
the SECOND dataset and saves them to disk as .pt files.

Output structure:
    dataset_root/
        embeddings_T1/   ← image_embed (1,256,64,64) from im1
        embeddings_T2/   ← image_embed (1,256,64,64) from im2
        highres_T1/      ← high_res_feats [(1,32,256,256), (1,64,128,128)]
        highres_T2/      ← high_res_feats

Usage:
    python extract_sam2_features.py --dataset_root /path/to/SECOND
    python extract_sam2_features.py --dataset_root /path/to/SECOND --split test
    python extract_sam2_features.py --dataset_root /path/to/SECOND --no-highres
"""

import argparse
import os
import sys
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# SAM2 imports — need to be run from the sam2 repo directory or have it on path
# ---------------------------------------------------------------------------
SAM2_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Suppress noisy SAM2 / Hydra / PIL logs
for name in ("sam2", "hydra", "PIL", "fvcore"):
    logging.getLogger(name).setLevel(logging.WARNING)
# SAM2 predictor logs to root logger at INFO level — suppress
logging.getLogger().setLevel(logging.WARNING)
# Ensure our own logger still shows INFO
log.setLevel(logging.INFO)


# ========================== CONFIG DEFAULTS =================================

DEFAULT_SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CKPT = "checkpoints/sam2.1_hiera_large.pt"

# ========================== DATASET =========================================


def build_image_pairs(dataset_root: str, split: str = "train"):
    """
    Build list of (filename, path_T1, path_T2) from SECOND dataset.

    SECOND layout:
        dataset_root/im1/xxx.png   (train T1)
        dataset_root/im2/xxx.png   (train T2)
        dataset_root/test/im1/xxx.png  (test T1)
        dataset_root/test/im2/xxx.png  (test T2)
    """
    root = Path(dataset_root)
    if split == "test":
        dir_t1 = root / "test" / "im1"
        dir_t2 = root / "test" / "im2"
    else:
        dir_t1 = root / "im1"
        dir_t2 = root / "im2"

    if not dir_t1.is_dir():
        raise FileNotFoundError(f"T1 image directory not found: {dir_t1}")
    if not dir_t2.is_dir():
        raise FileNotFoundError(f"T2 image directory not found: {dir_t2}")

    # Gather filenames present in BOTH T1 and T2
    t1_files = {f.name for f in dir_t1.iterdir() if f.suffix.lower() in (".png", ".jpg", ".tif")}
    t2_files = {f.name for f in dir_t2.iterdir() if f.suffix.lower() in (".png", ".jpg", ".tif")}
    common = sorted(t1_files & t2_files)

    if not common:
        raise RuntimeError(f"No matching image pairs found in {dir_t1} and {dir_t2}")

    pairs = [(fname, str(dir_t1 / fname), str(dir_t2 / fname)) for fname in common]
    log.info(f"Found {len(pairs)} image pairs ({split} split)")
    return pairs


# ========================== FEATURE EXTRACTION ==============================


def load_sam2_predictor(
    sam2_config: str,
    sam2_checkpoint: str,
    device: str = "cuda",
) -> SAM2ImagePredictor:
    """Load SAM2 model and return predictor."""
    log.info(f"Loading SAM2 from {sam2_checkpoint} ...")
    model = build_sam2(sam2_config, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    log.info(f"SAM2 ready on {predictor.device}")
    return predictor


@torch.no_grad()
def extract_features(
    predictor: SAM2ImagePredictor,
    image_path: str,
) -> dict:
    """
    Run SAM2 encoder on a single image and return features.

    Returns dict with:
        "image_embed": Tensor (1, 256, 64, 64)
        "high_res_feats": list of Tensors [(1,32,256,256), (1,64,128,128)]
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    predictor.set_image(img)

    features = {
        "image_embed": predictor._features["image_embed"].cpu(),
        "high_res_feats": [f.cpu() for f in predictor._features["high_res_feats"]],
    }

    # Ensure 4D: (1, C, H, W)
    if features["image_embed"].dim() == 3:
        features["image_embed"] = features["image_embed"].unsqueeze(0)

    predictor.reset_predictor()
    return features


# ========================== MAIN PIPELINE ===================================


def run_extraction(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    log.info(f"Device: {device}")

    # SAM2's build_sam2 uses Hydra compose() which needs config_name as a
    # relative path inside the sam2 package (pkg://sam2). The checkpoint
    # path must be absolute or relative to cwd.
    sam2_config = args.sam2_config  # e.g. "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Resolve checkpoint to absolute path (relative to sam2 repo)
    sam2_ckpt = args.sam2_ckpt
    if not os.path.isabs(sam2_ckpt):
        sam2_ckpt = os.path.join(SAM2_REPO, sam2_ckpt)

    predictor = load_sam2_predictor(sam2_config, sam2_ckpt, device)

    # Build image pairs
    pairs = build_image_pairs(args.dataset_root, args.split)

    # Output directories
    root = Path(args.dataset_root)
    suffix = f"_{args.split}" if args.split == "test" else ""

    emb_t1_dir = root / f"embeddings_T1{suffix}"
    emb_t2_dir = root / f"embeddings_T2{suffix}"
    emb_t1_dir.mkdir(parents=True, exist_ok=True)
    emb_t2_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_highres:
        hr_t1_dir = root / f"highres_T1{suffix}"
        hr_t2_dir = root / f"highres_T2{suffix}"
        hr_t1_dir.mkdir(parents=True, exist_ok=True)
        hr_t2_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving embeddings to {emb_t1_dir.parent}/")

    skipped = 0
    errors = 0

    for fname, path_t1, path_t2 in tqdm(pairs, desc=f"Extracting ({args.split})", unit="pair"):
        stem = Path(fname).stem  # e.g. "00001"

        # Output paths
        emb_t1_path = emb_t1_dir / f"{stem}.pt"
        emb_t2_path = emb_t2_dir / f"{stem}.pt"

        # Skip if both already exist (checkpoint mechanism)
        if emb_t1_path.exists() and emb_t2_path.exists():
            if not args.no_highres:
                hr_t1_path = hr_t1_dir / f"{stem}.pt"
                hr_t2_path = hr_t2_dir / f"{stem}.pt"
                if hr_t1_path.exists() and hr_t2_path.exists():
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

        # --- Extract T1 ---
        try:
            feats_t1 = extract_features(predictor, path_t1)
        except Exception as e:
            log.warning(f"Error processing T1 {fname}: {e}")
            errors += 1
            continue

        # --- Extract T2 ---
        try:
            feats_t2 = extract_features(predictor, path_t2)
        except Exception as e:
            log.warning(f"Error processing T2 {fname}: {e}")
            errors += 1
            continue

        # --- Save embeddings ---
        torch.save(feats_t1["image_embed"], emb_t1_path)
        torch.save(feats_t2["image_embed"], emb_t2_path)

        # --- Save high-res features (optional) ---
        if not args.no_highres:
            torch.save(feats_t1["high_res_feats"], hr_t1_dir / f"{stem}.pt")
            torch.save(feats_t2["high_res_feats"], hr_t2_dir / f"{stem}.pt")

    # Summary
    total = len(pairs)
    processed = total - skipped - errors
    log.info(f"Done! Processed: {processed} | Skipped (existing): {skipped} | Errors: {errors} | Total: {total}")

    # Verify one output
    sample = emb_t1_dir / f"{Path(pairs[0][0]).stem}.pt"
    if sample.exists():
        t = torch.load(sample, weights_only=True)
        log.info(f"Sample embedding shape: {t.shape} dtype: {t.dtype}")


# ========================== CLI =============================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: Extract SAM2 encoder features from SECOND dataset"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/chung/RS/phase1/SECOND",
        help="Path to SECOND dataset root",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--sam2_config",
        type=str,
        default=DEFAULT_SAM2_CONFIG,
        help="SAM2 config yaml (relative to sam2 repo)",
    )
    parser.add_argument(
        "--sam2_ckpt",
        type=str,
        default=DEFAULT_SAM2_CKPT,
        help="SAM2 checkpoint path (relative to sam2 repo)",
    )
    parser.add_argument(
        "--no-highres",
        action="store_true",
        help="Skip saving high-resolution features (saves disk space)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (not recommended)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(args)
