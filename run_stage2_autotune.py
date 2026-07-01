"""
run_stage2_autotune.py
======================
Performs a sweep over different SAM2 tokenization configurations (points_per_side)
and evaluates them using OS, US, and ED metrics against GT masks.
Helps find the 'Pareto Front' of token density vs. quality.
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm

import tokenize_regions
from stage2_quality_diagnostics import load_gt_masks, diagnostics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--pps_sweep", type=int, nargs="+", default=[16, 24, 32, 48, 64])
    parser.add_argument("--limit", type=int, default=20, help="Number of samples to test for autotune")
    parser.add_argument("--output", type=str, default="autotune_results.json")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    gt_dir = Path(args.gt_dir)
    
    # 1. Gather sample pairs
    pairs = tokenize_regions.build_image_pairs(args.dataset_root, args.split)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    
    results = {}

    # 2. Sweep PPS
    for pps in args.pps_sweep:
        log.info(f"Evaluating points_per_side = {pps} ...")
        
        # We'll run a mini tokenization in memory (or temp dir)
        # For simplicity, we just run the tokenization for these samples
        
        # Configure Stage 2 args
        s2_args = argparse.Namespace(
            dataset_root=args.dataset_root,
            split=args.split,
            sam2_config="configs/sam2.1/sam2.1_hiera_l.yaml",
            sam2_ckpt="checkpoints/sam2.1_hiera_large.pt",
            points_per_side=pps,
            pred_iou_thresh=0.7,
            stability_thresh=0.8,
            min_mask_area=256,
            save_masks=True,  # Crucial for diagnostics
            visualize=False,
            num_vis=0,
            cpu=False
        )
        
        # Load generator once for this PPS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        amg = tokenize_regions.load_mask_generator(
            s2_args.sam2_config, 
            os.path.join(tokenize_regions.SAM2_REPO, s2_args.sam2_ckpt),
            str(device), pps, 
            s2_args.pred_iou_thresh, 
            s2_args.stability_thresh, 
            s2_args.min_mask_area
        )
        
        all_os, all_us, all_ed = [], [], []
        total_tokens = 0
        
        for stem, path_t1, path_t2, emb_t1_path, emb_t2_path in tqdm(pairs, desc=f"PPS={pps}"):
            try:
                # Process T1 only for diagnostic simplicity
                import numpy as np
                from PIL import Image
                img = np.array(Image.open(path_t1).convert("RGB"))
                emb = torch.load(emb_t1_path, weights_only=True).to(device)
                
                masks = amg.generate(img)
                res = tokenize_regions.tokenize_image(masks, emb)
                
                # Diagnostics
                gt_path = gt_dir / f"{stem}.png"
                if not gt_path.exists():
                    gt_path = gt_dir / f"{stem}_instance_id_RGB.png"
                
                if gt_path.exists():
                    gt_masks = load_gt_masks(gt_path)
                    pred_masks = [m["segmentation"] for m in masks]
                    
                    os_v, us_v, ed_v = diagnostics(pred_masks, gt_masks)
                    all_os.append(os_v)
                    all_us.append(us_v)
                    all_ed.append(ed_v)
                    total_tokens += len(masks)
            except Exception as e:
                log.warning(f"Error on {stem}: {e}")

        if all_os:
            results[pps] = {
                "avg_os": float(np.mean(all_os)),
                "avg_us": float(np.mean(all_us)),
                "avg_ed": float(np.mean(all_ed)),
                "avg_tokens": float(total_tokens / len(all_os))
            }
            log.info(f"Results for PPS={pps}: OS={results[pps]['avg_os']:.4f}, US={results[pps]['avg_us']:.4f}, ED={results[pps]['avg_ed']:.4f}, Tokens={results[pps]['avg_tokens']:.1f}")

    # 3. Save and summary
    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    
    print("\n--- Autotune Summary ---")
    print("PPS\tOS\tUS\tED\tTokens")
    for pps in sorted(results.keys()):
        r = results[pps]
        print(f"{pps}\t{r['avg_os']:.3f}\t{r['avg_us']:.3f}\t{r['avg_ed']:.3f}\t{r['avg_tokens']:.1f}")
        
    # Pick "best" (heuristic: minimize US + OS, maximize ED)
    best_pps = min(results.keys(), key=lambda k: results[k]["avg_os"] + results[k]["avg_us"] - results[k]["avg_ed"])
    print(f"\nRecommended points_per_side: {best_pps}")

if __name__ == "__main__":
    main()
