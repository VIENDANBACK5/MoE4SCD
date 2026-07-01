"""
preprocess_isaid.py
===================
Tiling script for iSAID dataset.
Large mosaic images -> 512x512 tiles.
Filters tiles with very low information content.
Supports visualization of the tiling grid.
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

def tile_image(img_path, mask_path, output_img_dir, output_mask_dir, tile_size=512, overlap=64, min_info=0.05):
    """Tiles an image and its corresponding mask."""
    img = np.array(Image.open(img_path))
    mask = np.array(Image.open(mask_path)) if mask_path else None
    
    h, w = img.shape[:2]
    stem = Path(img_path).stem
    
    stride = tile_size - overlap
    count = 0
    
    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            tile_img = img[y:y+tile_size, x:x+tile_size]
            
            # Check info content (non-black pixels)
            if np.mean(tile_img > 0) < min_info:
                continue
                
            out_img_path = output_img_dir / f"{stem}_tile_{y}_{x}.png"
            Image.fromarray(tile_img).save(out_img_path)
            
            if mask is not None:
                tile_mask = mask[y:y+tile_size, x:x+tile_size]
                out_mask_path = output_mask_dir / f"{stem}_tile_{y}_{x}.png"
                Image.fromarray(tile_mask).save(out_mask_path)
            
            count += 1
    return count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Dir containing raw images")
    parser.add_argument("--mask_dir", type=str, default=None, help="Dir containing raw masks")
    parser.add_argument("--output_dir", type=str, required=True, help="Target dir for processed tiles")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_img_dir = output_path / "images"
    output_mask_dir = output_path / "masks"
    
    output_img_dir.mkdir(parents=True, exist_ok=True)
    if args.mask_dir:
        output_mask_dir.mkdir(parents=True, exist_ok=True)
        
    img_files = sorted(list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")))
    
    log_file = output_path / "tiling_log.txt"
    with open(log_file, "w") as f:
        for img_p in tqdm(img_files, desc="Tiling"):
            mask_p = None
            if args.mask_dir:
                mask_p = Path(args.mask_dir) / img_p.name
                if not mask_p.exists():
                    # Check for instance color naming convention
                    mask_p = Path(args.mask_dir) / img_p.name.replace(".png", "_instance_color_RGB.png")
            
            num_tiles = tile_image(img_p, mask_p, output_img_dir, output_mask_dir, args.tile_size, args.overlap)
            f.write(f"{img_p.name}: {num_tiles} tiles\n")
            
            if args.visualize and img_p == img_files[0]:
                # Save one visualization of the grid
                img = np.array(Image.open(img_p))
                plt.figure(figsize=(12, 12))
                plt.imshow(img)
                h, w = img.shape[:2]
                stride = args.tile_size - args.overlap
                for y in range(0, h - args.tile_size + 1, stride):
                    for x in range(0, w - args.tile_size + 1, stride):
                        rect = plt.Rectangle((x, y), args.tile_size, args.tile_size, fill=False, color="red", linewidth=0.5)
                        plt.gca().add_patch(rect)
                plt.title(f"Tiling Grid for {img_p.name}")
                plt.savefig(output_path / "tiling_viz_sample.png")
                plt.close()

    print(f"Preprocessing complete. Tiles saved to {output_path}")

if __name__ == "__main__":
    main()
