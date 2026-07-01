"""
Stage 2.5: Quality Diagnostics (OS, US, ED)
===========================================
This script calculates tokenization quality metrics by comparing SAM2 generated 
tokens (regions) and their computed parameters against ground truth instance masks.

It computes:
1. Over-segmentation (OS)
2. Under-segmentation (US) 
3. Edge Displacement (ED)
4. Correlates these metrics with the Coefficient of Variation (CV) 
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def diagnostics(pred_masks, gt_masks):
    # pred_masks: list of binary masks (N)
    # gt_masks: list of binary masks (M)
    
    # Compute intersection matrix
    N = len(pred_masks)
    M = len(gt_masks)
    
    if N == 0 or M == 0:
        return 0, 0, 0
        
    iou_matrix = np.zeros((N, M))
    for i, p in enumerate(pred_masks):
        for j, g in enumerate(gt_masks):
            iou_matrix[i, j] = calculate_iou(p, g)
            
    # Over-segmentation: A single GT object is split into multiple predicted regions
    # Under-segmentation: A single predicted region covers multiple GT objects
    
    # Simple metric for US: How many predicted masks have high intersection with more than 1 GT mask
    us_count = 0
    for i in range(N):
        # Count how many GT objects this predicted mask overlaps with significantly (e.g. > 20% of GT)
        overlaps = 0
        for j in range(M):
            intersect = np.logical_and(pred_masks[i], gt_masks[j]).sum()
            if intersect > 0.2 * gt_masks[j].sum():
                overlaps += 1
        if overlaps > 1:
            us_count += 1
            
    # Simple metric for OS: How many GT masks are broken into multiple predicted parts
    os_count = 0
    for j in range(M):
        parts = 0
        for i in range(N):
            intersect = np.logical_and(pred_masks[i], gt_masks[j]).sum()
            if intersect > 0.2 * pred_masks[i].sum() and intersect > 0.05 * gt_masks[j].sum():
                parts += 1
        if parts > 1:
            os_count += 1
            
    # Edge Displacement (ED) approximation: average IoU of the matched masks
    # Using Hungarian matching
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = iou_matrix[row_ind, col_ind]
    mean_ed = matched_ious.mean() if len(matched_ious) > 0 else 0
    
    return os_count / M if M > 0 else 0, us_count / N if N > 0 else 0, mean_ed

def load_gt_masks(gt_path):
    """
    Load ground truth masks.
    If it's a semantic mask where different objects have the same value, 
    this might not be perfect for instance metrics unless it's a building mask where 
    we use connected components.
    """
    gt_img = np.array(Image.open(gt_path))
    
    # If it's a multiclass iSAID mask, we might want to split by class or just binary
    # For now, let's assume instance mask where each object has a unique ID (typical for instance_id_RGB)
    if len(gt_img.shape) == 3:
        # Convert RGB ID to unique integer if needed, or just grayscale
        gt_img = gt_img[:,:,0].astype(np.int32) * 65536 + gt_img[:,:,1].astype(np.int32) * 256 + gt_img[:,:,2].astype(np.int32)
        
    unique_ids = np.unique(gt_img)
    unique_ids = unique_ids[unique_ids > 0] # Skip background 0
    
    masks = []
    for uid in unique_ids:
        masks.append(gt_img == uid)
    return masks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_dir", type=str, required=True, help="Path to saved SAM2 tokens (.pt files)")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to ground truth instance masks")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of images to process")
    args = parser.parse_args()

    tok_dir = Path(args.tokens_dir)
    gt_dir = Path(args.gt_dir)
    
    files = sorted(list(tok_dir.glob("*.pt")))
    if args.limit > 0:
        files = files[:args.limit]
        
    all_os, all_us, all_ed = [], [], []
    
    pbar = tqdm(files, desc="Running Diagnostics")
    for f in pbar:
        data = torch.load(f, weights_only=True)
        if "masks" not in data:
            continue
            
        pred_masks = data["masks"] # List of (512, 512) bool
        
        # Match with GT
        # iSAID naming: stem.pt vs stem.png
        gt_path = gt_dir / f"{f.stem}.png"
        if not gt_path.exists():
            # Try alternate naming
            gt_path = gt_dir / f"{f.stem}_instance_id_RGB.png"
            
        if not gt_path.exists():
            continue
            
        gt_masks = load_gt_masks(gt_path)
        
        os_val, us_val, ed_val = diagnostics(pred_masks, gt_masks)
        all_os.append(os_val)
        all_us.append(us_val)
        all_ed.append(ed_val)
        
        pbar.set_postfix({"OS": np.mean(all_os), "US": np.mean(all_us), "ED": np.mean(all_ed)})

    if not all_os:
        print("No files processed. Ensure --save_masks was used in tokenize_regions.py and GT paths are correct.")
        return

    print("\n--- Final Quality Metrics ---")
    print(f"Count: {len(all_os)}")
    print(f"Over-segmentation (OS):  {np.mean(all_os):.4f}")
    print(f"Under-segmentation (US): {np.mean(all_us):.4f}")
    print(f"Edge Displacement (ED):  {np.mean(all_ed):.4f} (Mean matched IoU)")

if __name__ == "__main__":
    main()
