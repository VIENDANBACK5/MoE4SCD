# generate_sam2_spectral.py
"""
Re-generate SAM2 masks với spectral-guided prompts.
Thay thế uniform grid prompts bằng spectral homogeneous region centers.

Output: SECOND/sam2_masks_T1_test_spectral/*.npz
"""
import os, sys, glob, argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Add SAM2 repo to path
SAM2_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
if os.path.isdir(SAM2_REPO):
    sys.path.insert(0, SAM2_REPO)

from spectral_edge_prompt import generate_spectral_prompts

MERGE_IOU_THRESH = 0.7
MIN_AREA_PX      = 50


def compute_iou(m1, m2):
    m1_ds = m1[::8, ::8]
    m2_ds = m2[::8, ::8]
    inter = (m1_ds & m2_ds).sum()
    union = (m1_ds | m2_ds).sum()
    return float(inter) / float(union + 1e-8)


def merge_duplicate_masks(masks_list, iou_thresh=MERGE_IOU_THRESH):
    """Loại bỏ duplicate masks."""
    masks_list = sorted(masks_list, key=lambda x: x["area"], reverse=True)
    kept = []
    for cand in masks_list:
        is_dup = any(compute_iou(cand["mask"], k["mask"]) > iou_thresh
                     for k in kept)
        if not is_dup:
            kept.append(cand)
    return kept


def main(args):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    print(f"✅ SAM2 loaded on {device}")

    stems = sorted([Path(f).stem for f in glob.glob(f"{args.img_dir}/*.png")])
    if getattr(args, "n_samples", None) is not None:
        stems = stems[:args.n_samples]
        
    os.makedirs(args.out_dir, exist_ok=True)

    coverage_list = []
    n_prompt_list = []

    for stem in tqdm(stems):
        out_path = os.path.join(args.out_dir, stem + ".npz")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        image_path = os.path.join(args.img_dir, stem + ".png")
        if not os.path.exists(image_path):
            continue

        image = np.array(Image.open(image_path).convert("RGB"))

        # Spectral-guided prompts
        prompt_points, grad_map = generate_spectral_prompts(
            image,
            grid_size=args.grid_size,
            grad_thresh=args.grad_thresh
        )
        n_prompt_list.append(len(prompt_points))

        predictor.set_image(image)
        all_masks = []

        if len(prompt_points) > 0:
            coords = np.array(prompt_points, dtype=np.float32)[:, None, :]  # (B, 1, 2)
            labels = np.ones((len(prompt_points), 1), dtype=np.int32)        # (B, 1)
            
            try:
                # Batched prediction on GPU
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labels,
                    multimask_output=True,
                )  # masks: (B, 3, H, W), scores: (B, 3)
                
                for i in range(len(prompt_points)):
                    best_idx = scores[i].argmax()
                    m = masks[i, best_idx]
                    if m.sum() >= MIN_AREA_PX:
                        all_masks.append({
                            "mask":  m.astype(bool),
                            "score": float(scores[i, best_idx]),
                            "area":  int(m.sum()),
                        })
            except Exception as e:
                print(f"Error predicting batch for {stem}: {e}")

        merged = merge_duplicate_masks(all_masks)

        if not merged:
            # Fallback so downstream doesn't break
            masks_arr = np.zeros((1, 512, 512), dtype=bool)
            scores_arr = np.zeros(1, dtype=np.float32)
            np.savez_compressed(out_path, masks=masks_arr, scores=scores_arr)
            coverage_list.append(0.0)
            continue

        # Coverage
        H, W  = image.shape[:2]
        union = np.zeros((H, W), dtype=bool)
        for m in merged:
            union |= m["mask"]
        coverage_list.append(float(union.sum()) / (H * W))

        masks_arr  = np.stack([m["mask"]  for m in merged]).astype(bool)
        scores_arr = np.array([m["score"] for m in merged])
        np.savez_compressed(out_path, masks=masks_arr, scores=scores_arr)

    print(f"\n✅ Spectral-guided SAM2 masks generated.")
    if coverage_list:
        print(f"   Coverage:     {np.mean(coverage_list)*100:.1f}%")
        print(f"   Avg prompts:  {np.mean(n_prompt_list):.0f} per image")
        print(f"   (vs uniform grid 8×8 = 64 prompts)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir",         required=True)
    p.add_argument("--out-dir",         required=True)
    p.add_argument("--sam2-config",     default="configs/sam2.1/sam2.1_hiera_l.yaml")
    p.add_argument("--sam2-checkpoint", default=os.path.join(SAM2_REPO, "checkpoints", "sam2.1_hiera_large.pt"))
    p.add_argument("--grid-size",       type=int,   default=16)
    p.add_argument("--grad-thresh",     type=float, default=0.15)
    p.add_argument("--n-samples",       type=int,   default=None, help="Limit the number of images processed")
    p.add_argument("--overwrite",       action="store_true")
    
    args = p.parse_args()
    main(args)
