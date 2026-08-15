# eval_seg_deadtrees.py
"""
Đánh giá SAM2 masks vs GT deadwood polygons bằng OS/US/ED
(Dao et al. 2021 metrics).

OS (Over-segmentation): 1 GT polygon bị chia thành nhiều SAM2 masks
US (Under-segmentation): nhiều GT polygons gộp vào 1 SAM2 mask
ED = sqrt((OS² + US²) / 2)
"""
import os, glob, json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage

IMG_DIR   = "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
GT_DIR    = "DeadTrees/masks_gt"
SAM2_DIR  = "DeadTrees/sam2_masks"
OUT_PATH  = "DeadTrees/results/seg_quality_deadtrees.json"
os.makedirs("DeadTrees/results", exist_ok=True)


def compute_os_us_ed(gt_mask: np.ndarray, sam2_masks: np.ndarray):
    """
    gt_mask:    (H, W) binary — 1 GT object (connected component)
    sam2_masks: (N, H, W) binary — tất cả SAM2 predicted masks

    Tính OS, US, ED cho GT object này với SAM2 mask overlap nhiều nhất.
    """
    best_iou = 0
    best_overlap = None
    best_mask_area = 0

    gt_area = gt_mask.sum()
    if gt_area == 0:
        return None

    for sam_mask in sam2_masks:
        overlap = (gt_mask & sam_mask).sum()
        if overlap == 0:
            continue
        iou = overlap / (gt_mask | sam_mask).sum()
        if iou > best_iou:
            best_iou = iou
            best_overlap = overlap
            best_mask_area = sam_mask.sum()

    if best_overlap is None:
        return {"OS": 1.0, "US": 1.0, "ED": 1.0}  # hoàn toàn miss

    OS = 1 - (best_overlap / gt_area)
    US = 1 - (best_overlap / max(best_mask_area, 1))
    ED = np.sqrt((OS**2 + US**2) / 2)

    return {"OS": float(OS), "US": float(US), "ED": float(ED)}


# Map stems to absolute image paths dynamically
img_files = glob.glob(f"{IMG_DIR}/**/*.tif", recursive=True)
stem_to_img_path = {Path(f).stem: f for f in img_files}

gt_files = glob.glob(f"{GT_DIR}/*_deadwood.png")
all_os, all_us, all_ed = [], [], []
per_image_results = []

for gt_path in tqdm(gt_files, desc="Evaluating segmentation"):
    stem = Path(gt_path).stem.replace("_deadwood", "")

    img_path = stem_to_img_path.get(stem)
    if img_path is None:
        continue

    sam2_path = f"{SAM2_DIR}/{stem}.npz"
    if not os.path.exists(sam2_path):
        continue

    gt_mask = np.array(Image.open(gt_path)) > 127  # binary
    if gt_mask.sum() == 0:
        continue  # ảnh này không có deadwood

    sam2_data  = np.load(sam2_path)
    sam2_masks = sam2_data["masks"]

    # Tách GT thành các connected components (individual deadwood trees)
    labeled_gt, n_components = ndimage.label(gt_mask)

    for comp_id in range(1, n_components + 1):
        comp_mask = labeled_gt == comp_id
        if comp_mask.sum() < 20:  # Skip very small trees
            continue

        result = compute_os_us_ed(comp_mask, sam2_masks)
        if result is None:
            continue

        all_os.append(result["OS"])
        all_us.append(result["US"])
        all_ed.append(result["ED"])

    per_image_results.append({
        "stem": stem,
        "n_gt_deadwood_trees": int(n_components),
        "n_sam2_masks": len(sam2_masks),
    })

# --- Summary ---
summary = {
    "n_images_evaluated": len(per_image_results),
    "n_gt_objects_total":  len(all_os),
    "OS_mean": round(float(np.mean(all_os)), 4) if all_os else None,
    "US_mean": round(float(np.mean(all_us)), 4) if all_us else None,
    "ED_mean": round(float(np.mean(all_ed)), 4) if all_ed else None,
}

with open(OUT_PATH, "w") as f:
    json.dump({"summary": summary, "per_image": per_image_results}, f, indent=2)

print(json.dumps(summary, indent=2))

# Validation
assert summary["n_gt_objects_total"] > 10, \
    f"[FAIL] Quá ít GT objects được evaluate ({summary['n_gt_objects_total']}) — kiểm tra lại D2"
print(f"\n✅ Validation passed. Compare với SECOND OS/US/ED để reference.")
