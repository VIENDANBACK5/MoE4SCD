"""
Stage 2: Region Tokenization using SAM2 Automatic Mask Generator
================================================================
For each image in the SECOND dataset:
  1. Generate region masks using SAM2AutomaticMaskGenerator
  2. Load pre-computed SAM2 encoder embeddings from Stage 1
  3. Downsample each mask (512x512) to embedding resolution (64x64)
  4. Masked average pool embedding → per-region token (256-dim)
  5. Compute centroid (normalized) and area for each token
  6. Save to disk as {tokens, centroids, areas}

Output structure:
    SECOND/
        tokens_T1/  xxx.pt  → {"tokens": (N,256), "centroids": (N,2), "areas": (N,)}
        tokens_T2/  xxx.pt

Usage:
    python tokenize_regions.py --dataset_root /path/to/SECOND
    python tokenize_regions.py --dataset_root /path/to/SECOND --split test
    python tokenize_regions.py --dataset_root /path/to/SECOND --points_per_side 16
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------------------------------------------------------------
# SAM2 repo path (raw SAM2 — no samgeo geo deps needed)
# ---------------------------------------------------------------------------
SAM2_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# ---------------------------------------------------------------------------
# Logging — suppress SAM2/Hydra noise
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger().setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 512      # SECOND image resolution
EMB_SIZE = 64       # SAM2 image_embed spatial resolution (512/8)
SCALE = IMG_SIZE // EMB_SIZE  # = 8


# ========================== DATASET =========================================

def build_image_pairs(dataset_root: str, split: str = "train"):
    """Return sorted list of (stem, path_T1, path_T2, emb_T1_path, emb_T2_path)."""
    root = Path(dataset_root)
    if split == "test":
        dir_t1 = root / "test" / "im1"
        dir_t2 = root / "test" / "im2"
        emb_t1_dir = root / "embeddings_T1_test"
        emb_t2_dir = root / "embeddings_T2_test"
    else:
        dir_t1 = root / "im1"
        dir_t2 = root / "im2"
        emb_t1_dir = root / "embeddings_T1"
        emb_t2_dir = root / "embeddings_T2"

    for d in [dir_t1, dir_t2, emb_t1_dir, emb_t2_dir]:
        if not d.is_dir():
            raise FileNotFoundError(f"Directory not found: {d}\n"
                                    f"Run Stage 1 (extract_sam2_features.py) first.")

    img_files = sorted(
        f.name for f in dir_t1.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".tif")
    )
    if not img_files:
        raise RuntimeError(f"No images found in {dir_t1}")

    pairs = [
        (Path(fname).stem, dir_t1 / fname, dir_t2 / fname,
         emb_t1_dir / f"{Path(fname).stem}.pt",
         emb_t2_dir / f"{Path(fname).stem}.pt")
        for fname in img_files
    ]
    log.info(f"Found {len(pairs)} image pairs ({split} split)")
    return pairs


# ========================== MASK GENERATOR ==================================

def load_mask_generator(
    config: str,
    checkpoint: str,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_thresh: float,
    min_mask_area: int,
) -> SAM2AutomaticMaskGenerator:
    """Build SAM2 model and wrap with automatic mask generator."""
    log.info(f"Loading SAM2 from {checkpoint} ...")
    model = build_sam2(config, checkpoint, device=device)
    amg = SAM2AutomaticMaskGenerator(
        model=model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_thresh,
        min_mask_region_area=min_mask_area,
        output_mode="binary_mask",  # returns bool ndarray per mask
    )
    log.info(f"SAM2 AutoMaskGenerator ready (pts_per_side={points_per_side})")
    return amg


# ========================== TOKENIZATION =====================================

def downsample_mask(mask: np.ndarray, out_size: int = EMB_SIZE) -> torch.Tensor:
    """
    Downsample a boolean mask from (512,512) to (64,64) using max-pooling.
    A pixel in the small mask is True if ANY original pixel in its 8x8 patch is True.

    Returns: bool Tensor (out_size, out_size)
    """
    t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)  # (1,1,512,512)
    small = F.max_pool2d(t, kernel_size=SCALE, stride=SCALE)       # (1,1,64,64)
    return small.squeeze() > 0.0                                    # (64,64) bool


def tokenize_image(
    masks: list,
    embedding: torch.Tensor,
) -> dict:
    """
    Convert SAM2 masks + SAM2 embedding into region tokens.

    Args:
        masks: list of dicts from SAM2AutomaticMaskGenerator.generate()
                each dict has 'segmentation' (512,512 bool), 'area' (int)
        embedding: Tensor (1, 256, 64, 64) or (256, 64, 64)

    Returns:
        dict with:
            "tokens":    Tensor (N, 256)  — masked average pooled features
            "centroids": Tensor (N, 2)    — (cx, cy) normalized to [0,1]
            "areas":     Tensor (N,)      — fraction of image covered
    """
    # Ensure embedding is (256, 64, 64)
    emb = embedding.squeeze(0)  # (256, 64, 64)

    tokens_list = []
    centroids_list = []
    areas_list = []
    cv_list = []

    for m in masks:
        seg = m["segmentation"]  # (512,512) bool ndarray

        # --- Downsample mask to embedding resolution ---
        mask_small = downsample_mask(seg)  # (64,64) bool tensor

        # Skip degenerate masks (too small after downsampling)
        if mask_small.sum() == 0:
            continue

        # --- Masked average pooling → token ---
        # emb: (256, 64, 64), mask_small: (64, 64)
        region_feats = emb[:, mask_small]  # (256, K) where K = #true pixels
        token = region_feats.mean(dim=1)   # (256,)
        tokens_list.append(token)
        
        # --- Compute Coefficient of Variation (CV) ---
        # Evaluate how uniform the features are inside the mask.
        if region_feats.shape[1] > 1:
            std = region_feats.std(dim=1)
            cv_val = (std / (torch.abs(token) + 1e-6)).mean()
        else:
            cv_val = torch.tensor(0.0, dtype=torch.float32).to(region_feats.device)
        cv_list.append(cv_val)

        # --- Centroid (in original 512x512 space, normalized) ---
        ys, xs = np.where(seg)
        cx = float(xs.mean()) / (IMG_SIZE - 1)  # [0, 1]
        cy = float(ys.mean()) / (IMG_SIZE - 1)  # [0, 1]
        centroids_list.append(torch.tensor([cx, cy], dtype=torch.float32))

        # --- Area (fraction of image) ---
        area = float(seg.sum()) / (IMG_SIZE * IMG_SIZE)
        areas_list.append(area)

    if not tokens_list:
        # Fallback: global average pooling if no valid masks
        token = emb.mean(dim=(1, 2))  # (256,)
        tokens_list = [token]
        centroids_list = [torch.tensor([0.5, 0.5], dtype=torch.float32)]
        areas_list = [1.0]
        cv_list = [torch.tensor(0.0, dtype=torch.float32)]

    res = {
        "tokens":    torch.stack(tokens_list),                          # (N, 256)
        "centroids": torch.stack(centroids_list),                       # (N, 2)
        "areas":     torch.tensor(areas_list, dtype=torch.float32),     # (N,)
        "cvs":       torch.stack(cv_list),                              # (N,)
    }
    
    # Optionally include raw boolean masks
    if getattr(tokenize_image, "save_masks", False):
        res["masks"] = [m["segmentation"] for m in masks] if masks else [np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)]

    return res


# ========================= AUTO-TUNING & OUTLIER REMOVAL =========================

def compute_nn_nroc(img: np.ndarray) -> float:
    """
    Computes Nearest Neighbor normalized Rate of Change (NN-nRoC)
    as a proxy for image complexity/texture density.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    diff_y = np.abs(gray[1:, :] - gray[:-1, :])
    diff_x = np.abs(gray[:, 1:] - gray[:, :-1])
    return float((np.mean(diff_y) + np.mean(diff_x)) / 2.0)

def dynamic_points_per_side(img: np.ndarray, base_points: int) -> int:
    """
    Auto-tunes points_per_side dynamically based on image complexity.
    """
    nroc = compute_nn_nroc(img)
    if nroc < 0.03:
        return max(16, base_points // 2)
    elif nroc > 0.08:
        return min(64, base_points * 2)
    return base_points

def remove_outliers_mad(tokens_dict: dict, k: float = 3.0) -> dict:
    """
    Parameter-free Outlier Removal using Median Absolute Deviation (MAD).
    Removes tokens that are extreme outliers in either Area or CV.
    """
    areas = tokens_dict["areas"]
    cvs = tokens_dict["cvs"]
    N = areas.shape[0]
    
    if N < 5:
        return tokens_dict
        
    def get_mad_mask(x: torch.Tensor) -> torch.Tensor:
        med = torch.median(x)
        mad = torch.median(torch.abs(x - med))
        if mad < 1e-6:
            return torch.ones_like(x, dtype=torch.bool)
        modified_z = 0.6745 * torch.abs(x - med) / mad
        return modified_z <= k

    # Valid if neither area nor CV is an extreme outlier
    valid_mask = get_mad_mask(areas) & get_mad_mask(cvs)
    
    # Check if we didn't filter everything
    if valid_mask.sum() == 0 or valid_mask.sum() == N:
        return tokens_dict
        
    filtered = {}
    for key, val in tokens_dict.items():
        if isinstance(val, torch.Tensor) and val.shape[0] == N:
            filtered[key] = val[valid_mask]
        elif isinstance(val, list) and len(val) == N:
            filtered[key] = [val[i] for i in range(N) if valid_mask[i]]
        else:
            filtered[key] = val
    return filtered


# ========================== MAIN PIPELINE ===================================

@torch.no_grad()
def run_tokenization(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    log.info(f"Device: {device}")

    # Resolve SAM2 config/checkpoint paths
    sam2_config = args.sam2_config
    sam2_ckpt = args.sam2_ckpt
    if not os.path.isabs(sam2_ckpt):
        sam2_ckpt = os.path.join(SAM2_REPO, sam2_ckpt)
    if not os.path.isfile(sam2_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {sam2_ckpt}")

    pps_options = set([
        args.points_per_side, 
        max(16, args.points_per_side // 2), 
        min(64, args.points_per_side * 2)
    ])
    
    amgs = {}
    for pps in pps_options:
        amgs[pps] = load_mask_generator(
            config=sam2_config,
            checkpoint=sam2_ckpt,
            device=device,
            points_per_side=pps,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_thresh=args.stability_thresh,
            min_mask_area=args.min_mask_area,
        )

    pairs = build_image_pairs(args.dataset_root, args.split)

    # Output directories
    root = Path(args.dataset_root)
    suffix = "_test" if args.split == "test" else ""
    tok_t1_dir = root / f"tokens_T1{suffix}"
    tok_t2_dir = root / f"tokens_T2{suffix}"
    tok_t1_dir.mkdir(parents=True, exist_ok=True)
    tok_t2_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving tokens to {tok_t1_dir.parent}/tokens_T1{suffix}/ and tokens_T2{suffix}/")

    skipped = 0
    errors = 0
    total_tokens = 0

    for stem, path_t1, path_t2, emb_t1_path, emb_t2_path in tqdm(
        pairs, desc=f"Tokenizing ({args.split})", unit="pair"
    ):
        out_t1 = tok_t1_dir / f"{stem}.pt"
        out_t2 = tok_t2_dir / f"{stem}.pt"

        # Checkpoint: skip if both output files already exist
        if out_t1.exists() and out_t2.exists():
            skipped += 1
            continue

# ========================= VISUALIZATION =========================

def save_tokenization_viz(img: np.ndarray, masks: list, cvs: torch.Tensor, output_path: str):
    """
    Saves a visualization of the tokenized regions.
    Each region is colored by its Coefficient of Variation (CV).
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    
    # Create colormap for CV (normalized 0.0 to 1.0)
    cv_np = cvs.cpu().numpy()
    norm = plt.Normalize(vmin=0, vmax=1.0)
    cmap = cm.get_cmap("jet")
    
    combined_mask = np.zeros((img.shape[0], img.shape[1], 4))
    
    for i, mask in enumerate(masks):
        m = mask["segmentation"] if isinstance(mask, dict) else mask
            
        color = cmap(norm(cv_np[i]))
        combined_mask[m] = color
        combined_mask[m, 3] = 0.5  # Alpha
        
        # Add index text at centroid
        coords = np.argwhere(m)
        if len(coords) > 0:
            y_c, x_c = coords.mean(axis=0)
            plt.text(x_c, y_c, str(i), color="white", fontsize=6, ha="center", va="center")

    plt.imshow(combined_mask)
    plt.title(f"Tokenization Preview (Color by CV: Blue=Clean, Red=Noisy)\nOutput: {Path(output_path).name}")
    plt.axis("off")
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()


def run_tokenization(args):
    """Main execution logic for Stage 2."""
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    log.info(f"Using device: {device}")

    # Build pairs
    pairs = build_image_pairs(args.dataset_root, args.split)

    # Output directory
    root = Path(args.dataset_root)
    split_str = f"_{args.split}" if args.split != "train" else ""
    tok_t1_dir = root / f"tokens_T1{split_str}"
    tok_t2_dir = root / f"tokens_T2{split_str}"
    tok_t1_dir.mkdir(parents=True, exist_ok=True)
    tok_t2_dir.mkdir(parents=True, exist_ok=True)
    
    viz_dir = root / "visualizations" / "stage2"
    if args.visualize:
        viz_dir.mkdir(parents=True, exist_ok=True)

    # Init SAM2 AMGs for different densities
    # (points_per_side, stability_thresh, etc.)
    # We maintain a dict of AMGs to avoid re-initializing if density changes
    amgs = {}
    pps_options = [16, 32, 64] # Core options for auto-tuning
    
    sam2_checkpoint = os.path.join(SAM2_REPO, args.sam2_ckpt)
    model_cfg = args.sam2_config
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    for pps in pps_options:
        amgs[pps] = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=pps,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_thresh,
            min_mask_region_area=args.min_mask_area,
        )

    # Default AMG if auto-tuning results in something else
    if args.points_per_side not in amgs:
        amgs[args.points_per_side] = SAM2AutomaticMaskGenerator(
            model=sam2_model,
            points_per_side=args.points_per_side,
            pred_iou_thresh=args.pred_iou_thresh,
            stability_score_thresh=args.stability_thresh,
            min_mask_region_area=args.min_mask_area,
        )

    skipped = 0
    errors = 0
    total_tokens = 0

    for i, (stem, path_t1, path_t2, emb_t1_path, emb_t2_path) in enumerate(tqdm(pairs, desc=f"Stage 2 ({args.split})")):
        out_t1 = tok_t1_dir / f"{stem}.pt"
        out_t2 = tok_t2_dir / f"{stem}.pt"

        if out_t1.exists() and out_t2.exists():
            skipped += 1
            continue

        try:
            # --- Load images ---
            img_t1 = np.array(Image.open(path_t1).convert("RGB"))
            img_t2 = np.array(Image.open(path_t2).convert("RGB"))

            # --- Load pre-computed embeddings from Stage 1 ---
            emb_t1 = torch.load(emb_t1_path, weights_only=True).to(device)
            emb_t2 = torch.load(emb_t2_path, weights_only=True).to(device)

            # Auto-tune AMG based on NN-nRoC
            pps_t1 = dynamic_points_per_side(img_t1, args.points_per_side)
            pps_t2 = dynamic_points_per_side(img_t2, args.points_per_side)
            amg_t1 = amgs[pps_t1]
            amg_t2 = amgs[pps_t2]

            # --- Generate masks for both time steps ---
            masks_t1 = amg_t1.generate(img_t1)
            masks_t2 = amg_t2.generate(img_t2)

            # --- Tokenize ---
            tokenize_image.save_masks = args.save_masks or args.visualize
            result_t1 = tokenize_image(masks_t1, emb_t1)
            result_t2 = tokenize_image(masks_t2, emb_t2)
            
            # --- Apply MAD Outlier Removal ---
            result_t1 = remove_outliers_mad(result_t1)
            result_t2 = remove_outliers_mad(result_t2)

            # Move to CPU before saving
            result_t1_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in result_t1.items()}
            result_t2_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in result_t2.items()}

            # --- Save ---
            # Remove masks from save dict if not explicitly asked (to save space)
            if not args.save_masks:
                if "masks" in result_t1_cpu: del result_t1_cpu["masks"]
                if "masks" in result_t2_cpu: del result_t2_cpu["masks"]

            torch.save(result_t1_cpu, out_t1)
            torch.save(result_t2_cpu, out_t2)

            # --- Visualization ---
            if args.visualize and i < args.num_vis:
                if "masks" in result_t1:
                    save_tokenization_viz(img_t1, result_t1["masks"], result_t1["cvs"], str(viz_dir / f"{stem}_T1_viz.png"))
                    save_tokenization_viz(img_t2, result_t2["masks"], result_t2["cvs"], str(viz_dir / f"{stem}_T2_viz.png"))

            total_tokens += result_t1["tokens"].shape[0] + result_t2["tokens"].shape[0]

        except Exception as e:
            log.warning(f"Error on {stem}: {e}")
            errors += 1
            continue

    processed = len(pairs) - skipped - errors
    avg_tokens = total_tokens / max(processed * 2, 1)
    log.info(
        f"Done! Processed: {processed} | Skipped: {skipped} | Errors: {errors} | "
        f"Avg tokens/image: {avg_tokens:.1f}"
    )

    # Verify a sample output
    sample_files = sorted(tok_t1_dir.glob("*.pt"))
    if sample_files:
        s = torch.load(sample_files[0], weights_only=True)
        log.info(
            f"Sample [{sample_files[0].name}]: "
            f"tokens={s['tokens'].shape}  "
            f"centroids={s['centroids'].shape}  "
            f"areas={s['areas'].shape}"
        )


# ========================== CLI =============================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2: Region tokenization via SAM2 masks + embedding pooling"
    )
    parser.add_argument(
        "--dataset_root", type=str,
        default="SECOND",
        help="Path to SECOND dataset root",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"],
        help="Dataset split",
    )
    parser.add_argument(
        "--sam2_config", type=str,
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
        help="SAM2 config (relative to sam2 repo)",
    )
    parser.add_argument(
        "--sam2_ckpt", type=str,
        default="checkpoints/sam2.1_hiera_large.pt",
        help="SAM2 checkpoint (relative to sam2 repo)",
    )
    # Mask generation params — tuned for RS imagery
    parser.add_argument("--points_per_side", type=int, default=32,
                        help="Grid density for automatic mask generation (default=32 → ~42 masks/image)")
    parser.add_argument("--pred_iou_thresh", type=float, default=0.75,
                        help="Minimum predicted IoU quality threshold")
    parser.add_argument("--stability_thresh", type=float, default=0.85,
                        help="Minimum mask stability score")
    parser.add_argument("--min_mask_area", type=int, default=256,
                        help="Minimum mask area in pixels (default=256 = 0.1% of 512x512)")
    parser.add_argument("--save_masks", action="store_true",
                        help="Save full binary masks in tokens for diagnostics (takes more space)")
    parser.add_argument("--visualize", action="store_true", help="Save sample PNG visualizations")
    parser.add_argument("--num_vis", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_tokenization(args)
