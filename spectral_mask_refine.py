# spectral_mask_refine.py
"""
Post-processing refinement của SAM2 masks dựa trên spectral information.

Hai operations:
1. SPLIT: mask có CV > threshold → dùng k-means (k=2) trên spectral
           để split thành 2 sub-masks có spectral đồng nhất hơn

2. MERGE: 2 adjacent masks có spectral distance < threshold
           → merge thành 1 mask

Output: SECOND/sam2_masks_T1_test_refined/*.npz (cùng format)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob, argparse, multiprocessing
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
from concurrent.futures import ProcessPoolExecutor

def spectral_mean(image, mask):
    """Mean RGB của vùng mask, float32."""
    pixels = image[mask.astype(bool)].astype(np.float32)
    return pixels.mean(axis=0) if len(pixels) > 0 else np.zeros(3)


def compute_cv(image, mask):
    """Coefficient of Variation của mask."""
    pixels = image[mask.astype(bool)].astype(np.float32)
    if len(pixels) < 10:
        return 0.0
    mean = pixels.mean(axis=0) + 1e-8
    std  = pixels.std(axis=0)
    return float((std / mean).mean())


def split_mask_spectral(image, mask, n_clusters=2, min_area_px=50):
    """
    Split mask bằng k-means trên spectral.
    Downsamples pixels to at most 5,000 to speed up KMeans fitting.
    """
    ys, xs = np.where(mask)
    if len(ys) < n_clusters * min_area_px:
        return [mask]   # quá nhỏ để split

    pixels_rgb = image[ys, xs].astype(np.float32)
    n_pixels = len(pixels_rgb)

    try:
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
        if n_pixels > 5000:
            # Sample 5000 random pixels to speed up clustering
            rng = np.random.default_rng(42)
            idx_sample = rng.choice(n_pixels, size=5000, replace=False)
            pixels_sample = pixels_rgb[idx_sample]
            km.fit(pixels_sample)
            labels = km.predict(pixels_rgb)
        else:
            labels = km.fit_predict(pixels_rgb)
    except Exception:
        return [mask]

    # Tạo sub-masks
    sub_masks = []
    H, W = mask.shape
    for k in range(n_clusters):
        sub = np.zeros((H, W), dtype=bool)
        idx = labels == k
        sub[ys[idx], xs[idx]] = True
        if sub.sum() >= min_area_px:
            sub_masks.append(sub)

    return sub_masks if len(sub_masks) > 1 else [mask]


def are_adjacent(mask1, mask2, bbox1, bbox2, dilation=3):
    """Kiểm tra 2 masks có kề nhau không sử dụng precomputed bboxes + crop + dilation."""
    if bbox1 is None or bbox2 is None:
        return False
    min_y1, max_y1, min_x1, max_x1 = bbox1
    min_y2, max_y2, min_x2, max_x2 = bbox2
    
    # Check if bounding boxes are outside adjacent range
    if (min_y1 > max_y2 + dilation or min_y2 > max_y1 + dilation or
        min_x1 > max_x2 + dilation or min_x2 > max_x1 + dilation):
        return False
        
    # Crop to the union of bounding boxes plus padding
    y1_crop = max(0, min(min_y1, min_y2) - dilation)
    y2_crop = min(mask1.shape[0], max(max_y1, max_y2) + dilation + 1)
    x1_crop = max(0, min(min_x1, min_x2) - dilation)
    x2_crop = min(mask1.shape[1], max(max_x1, max_x2) + dilation + 1)
    
    m1_crop = mask1[y1_crop:y2_crop, x1_crop:x2_crop]
    m2_crop = mask2[y1_crop:y2_crop, x1_crop:x2_crop]
    
    dilated1 = binary_dilation(m1_crop, iterations=dilation)
    return bool((dilated1 & m2_crop).any())


def refine_masks(image, masks, cv_split_thresh=0.15, merge_dist_thresh=20.0, min_area_px=50):
    """
    Áp dụng split và merge trên tất cả masks của 1 ảnh.
    Sử dụng Union-Find + Precomputed Bounding Boxes + Downsampled KMeans để tối đa hóa tốc độ.
    """
    # --- STEP 1: SPLIT masks có CV cao ---
    refined = []
    for mask in masks:
        cv = compute_cv(image, mask)
        if cv > cv_split_thresh:
            sub_masks = split_mask_spectral(image, mask, n_clusters=2, min_area_px=min_area_px)
            refined.extend(sub_masks)
        else:
            refined.append(mask)

    # --- STEP 2: MERGE adjacent masks có spectral giống nhau ---
    n = len(refined)
    if n == 0:
        return []

    # Precompute spectral means and bounding boxes
    means = [spectral_mean(image, m) for m in refined]
    bboxes = []
    for m in refined:
        ys, xs = np.where(m)
        if len(ys) == 0:
            bboxes.append(None)
        else:
            bboxes.append((ys.min(), ys.max(), xs.min(), xs.max()))

    # DSU initialization
    parent = list(range(n))
    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    # So sánh từng cặp
    for i in range(n):
        for j in range(i+1, n):
            # 1. So sánh spectral distance trước (rất rẻ)
            dist = float(np.linalg.norm(means[i] - means[j]))
            if dist >= merge_dist_thresh:
                continue

            # 2. So sánh adjacency (chỉ check nếu dist nhỏ)
            if are_adjacent(refined[i], refined[j], bboxes[i], bboxes[j]):
                union(i, j)

    # Gom nhóm theo root parent
    groups = {}
    for i in range(n):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    merged_masks = []
    for idxs in groups.values():
        merged = refined[idxs[0]]
        for idx in idxs[1:]:
            merged = merged | refined[idx]
        if merged.sum() >= min_area_px:
            merged_masks.append(merged)

    return merged_masks


def process_single_image(args):
    stem, img_path, mask_path, out_path, cv_split_thresh, merge_dist_thresh, min_area_px = args
    if not os.path.exists(img_path) or not os.path.exists(mask_path):
        return 0, 0
    try:
        image = np.array(Image.open(img_path).convert("RGB"))
        data  = np.load(mask_path)
        masks = [data["masks"][i].astype(bool) for i in range(len(data["masks"]))]

        refined_masks = refine_masks(
            image, masks,
            cv_split_thresh=cv_split_thresh,
            merge_dist_thresh=merge_dist_thresh,
            min_area_px=min_area_px
        )

        # Save
        if len(refined_masks) > 0:
            masks_arr = np.stack(refined_masks).astype(bool)
            scores    = np.ones(len(refined_masks), dtype=np.float32)
        else:
            masks_arr = np.zeros((1, 512, 512), dtype=bool)
            scores    = np.zeros(1, dtype=np.float32)

        np.savez_compressed(out_path, masks=masks_arr, scores=scores)
        return len(masks), len(refined_masks)
    except Exception as e:
        print(f"Error processing {stem}: {e}")
        return 0, 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split",             default="test", choices=["train", "test"])
    p.add_argument("--orig-mask-dir",     default=None)
    p.add_argument("--out-dir",           default=None)
    p.add_argument("--img-dir",           default=None)
    p.add_argument("--cv-split-thresh",   type=float, default=0.15)
    p.add_argument("--merge-dist-thresh", type=float, default=20.0)
    p.add_argument("--min-area-px",       type=int,   default=50)
    args = p.parse_args()

    # Determine default paths based on split if not provided
    if args.split == "test":
        orig_mask_dir = args.orig_mask_dir or "SECOND/sam2_masks_T1_test"
        out_dir = args.out_dir or "SECOND/sam2_masks_T1_test_refined"
        img_dir = args.img_dir or "SECOND/test/im1"
    else:
        orig_mask_dir = args.orig_mask_dir or "SECOND/sam2_masks_T1"
        out_dir = args.out_dir or "SECOND/sam2_masks_T1_refined"
        img_dir = args.img_dir or "SECOND/im1"

    os.makedirs(out_dir, exist_ok=True)
    stems = sorted([Path(f).stem for f in glob.glob(f"{orig_mask_dir}/*.npz")])
    print(f"Refining masks for split {args.split}...")
    print(f"  Source masks: {orig_mask_dir}")
    print(f"  Output directory: {out_dir}")
    print(f"  Image directory: {img_dir}")
    print(f"  CV Split Threshold: {args.cv_split_thresh}")
    print(f"  Merge Distance Threshold: {args.merge_dist_thresh}")

    tasks = []
    for stem in stems:
        img_path  = f"{img_dir}/{stem}.png"
        mask_path = f"{orig_mask_dir}/{stem}.npz"
        out_path  = f"{out_dir}/{stem}.npz"
        tasks.append((
            stem, img_path, mask_path, out_path,
            args.cv_split_thresh, args.merge_dist_thresh, args.min_area_px
        ))

    num_workers = min(16, multiprocessing.cpu_count())
    print(f"Using {num_workers} parallel workers...")

    total_orig = 0
    total_refined = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(stems), desc="Refining masks"))

    for orig, refined in results:
        total_orig += orig
        total_refined += refined

    print(f"\n✅ Refinement done.")
    print(f"   Original masks:  {total_orig} ({total_orig/max(len(stems), 1):.1f}/image)")
    print(f"   Refined masks:   {total_refined} ({total_refined/max(len(stems), 1):.1f}/image)")
    print(f"   Change:          {total_refined - total_orig:+d}")


if __name__ == "__main__":
    main()
