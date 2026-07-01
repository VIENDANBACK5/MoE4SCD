"""
preprocess_whu.py
=================
Tiling and preprocessing script for the WHU Building Change Detection dataset.
Handles large TIFF mosaics and converts them to a standardized 512x512 tile structure.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def tile_image(img_path, output_dir, tile_size=512, stride=512, prefix=""):
    img = np.array(Image.open(img_path))
    h, w = img.shape[:2]
    
    stem = Path(img_path).stem
    count = 0
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile = img[y:y+tile_size, x:x+tile_size]
            
            # Skip empty tiles (mostly background in RS)
            if np.mean(tile) < 2 and prefix != "label": # heuristic for non-labels
                continue
                
            out_name = f"{prefix}_{stem}_{y}_{x}.png"
            Image.fromarray(tile).save(output_dir / out_name)
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whu_root", type=str, required=True, help="Path to WHU dataset root (contains train/val/test)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tile_size", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.whu_root)
    out_root = Path(args.output_dir)
    
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists(): continue
        
        log_dir = out_root / split
        (log_dir / "im1").mkdir(parents=True, exist_ok=True)
        (log_dir / "im2").mkdir(parents=True, exist_ok=True)
        (log_dir / "label").mkdir(parents=True, exist_ok=True)
        
        # WHU usually has folders like 'before', 'after', 'change'
        t1_dir = split_dir / "before"
        t2_dir = split_dir / "after"
        gt_dir = split_dir / "change"
        
        files = sorted(list(t1_dir.glob("*.tif"))) + sorted(list(t1_dir.glob("*.png")))
        
        for f in tqdm(files, desc=f"Processing WHU {split}"):
            stem = f.stem
            t2_f = t2_dir / f.name
            gt_f = gt_dir / f.name
            
            if not t2_f.exists() or not gt_f.exists():
                continue
                
            # Tile T1
            tile_image(f, log_dir / "im1", args.tile_size, prefix="im1")
            # Tile T2
            tile_image(t2_f, log_dir / "im2", args.tile_size, prefix="im2")
            # Tile GT
            tile_image(gt_f, log_dir / "label", args.tile_size, prefix="label")

    print(f"Preprocessing complete. Output in {args.output_dir}")

if __name__ == "__main__":
    main()
