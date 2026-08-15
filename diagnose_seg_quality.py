# diagnose_seg_quality.py
"""
Đo chất lượng segmentation hiện tại bằng 3 metrics:

1. Intra-CV (Coefficient of Variation within mask):
   CV = std / mean cho mỗi channel RGB, trung bình 3 channels
   CV thấp → mask đồng nhất → TỐT
   CV cao  → mask có nhiều objects khác nhau → XẤU (over-merge)

2. Inter-spectral-distance (giữa adjacent masks):
   Khoảng cách Euclidean giữa mean RGB của 2 masks kề nhau
   Cao → masks phân biệt tốt → TỐT
   Thấp → 2 masks nên được merge → XẤU (over-segment)

3. Boundary accuracy (so với GT label):
   Tỷ lệ pixels trong mask thuộc cùng 1 GT class
   Cao → mask align với semantic boundaries → TỐT

Output: seg_quality_report.json
"""
import os, glob, json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

DATA_ROOT    = "SECOND"
SPLIT        = "test"
N_SAMPLE     = 200    # số ảnh để sample, tăng lên 500 nếu cần
NO_CHANGE_VAL = 0
EPS          = 1e-8

def compute_intra_cv(image_rgb, mask):
    """CV của pixels trong mask — đo độ đồng nhất spectral."""
    pixels = image_rgb[mask.astype(bool)]  # (N_px, 3)
    if len(pixels) < 5:
        return None
    mean = pixels.mean(axis=0) + EPS    # (3,)
    std  = pixels.std(axis=0)            # (3,)
    cv   = (std / mean).mean()           # scalar
    return float(cv)


def compute_boundary_purity(label, mask):
    """Tỷ lệ pixels trong mask thuộc dominant GT class."""
    pixels = label[mask.astype(bool)]
    semantic = pixels[pixels != NO_CHANGE_VAL]
    if len(semantic) == 0:
        return None
    values, counts = np.unique(semantic, return_counts=True)
    dominant_ratio = counts.max() / len(semantic)
    return float(dominant_ratio)


def compute_inter_distance(mask1_mean, mask2_mean):
    """Khoảng cách Euclidean giữa mean RGB của 2 masks."""
    return float(np.linalg.norm(mask1_mean - mask2_mean))


if __name__ == "__main__":
    stems = sorted([Path(f).stem for f in
        glob.glob(f"{DATA_ROOT}/{SPLIT}/im1/*.png")])[:N_SAMPLE]

    all_intra_cvs    = []
    all_purity       = []
    all_inter_dists  = []
    bad_masks        = []   # masks có CV cao (cần split)
    good_merge_pairs = []   # pairs với inter_dist thấp (cần merge)

    CV_HIGH_THRESH   = 0.15   # CV > 0.15 → mask không đồng nhất
    MERGE_DIST_THRESH = 20.0  # inter_dist < 20 → nên merge (RGB 0-255 scale)
    PURITY_LOW_THRESH = 0.7   # boundary purity < 70% → mask không align với GT

    for stem in tqdm(stems, desc="Diagnosing seg quality"):
        img_path  = f"{DATA_ROOT}/{SPLIT}/im1/{stem}.png"
        mask_path = f"{DATA_ROOT}/sam2_masks_T1_test/{stem}.npz"
        lab_path  = f"{DATA_ROOT}/{SPLIT}/label1/{stem}.png"

        if not all(os.path.exists(p) for p in [img_path, mask_path, lab_path]):
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        label = np.array(Image.open(lab_path))
        masks = np.load(mask_path)["masks"]  # (N, H, W)

        # --- Intra-CV và Boundary Purity cho từng mask ---
        mask_means = []
        for i, mask in enumerate(masks):
            mask_bool = mask.astype(bool)
            if mask_bool.sum() < 20:
                continue

            # Intra-CV
            cv = compute_intra_cv(image.astype(np.float32), mask_bool)
            if cv is not None:
                all_intra_cvs.append(cv)
                if cv > CV_HIGH_THRESH:
                    bad_masks.append({"stem": stem, "mask_id": i, "cv": cv})

            # Boundary purity
            purity = compute_boundary_purity(label, mask_bool)
            if purity is not None:
                all_purity.append(purity)

            # Mean RGB cho inter-distance
            pixels = image[mask_bool].astype(np.float32)
            mask_means.append(pixels.mean(axis=0))

        # --- Inter-distance giữa adjacent masks ---
        if len(mask_means) < 2:
            continue

        mask_means = np.array(mask_means)
        for i in range(len(mask_means)):
            for j in range(i+1, min(i+4, len(mask_means))):  # chỉ check nearby
                dist = compute_inter_distance(mask_means[i], mask_means[j])
                all_inter_dists.append(dist)
                if dist < MERGE_DIST_THRESH:
                    good_merge_pairs.append({
                        "stem": stem,
                        "mask_i": i, "mask_j": j,
                        "spectral_distance": dist
                    })

    # --- Report ---
    report = {
        "n_images_sampled": len(stems),
        "intra_cv": {
            "mean":   float(np.mean(all_intra_cvs)) if all_intra_cvs else 0,
            "median": float(np.median(all_intra_cvs)) if all_intra_cvs else 0,
            "pct_high_cv": len(bad_masks) / max(len(all_intra_cvs), 1),
            "interpretation": (
                "GOOD: masks are spectrally homogeneous"
                if np.mean(all_intra_cvs) < 0.10 else
                "BAD: many masks contain multiple spectral classes"
            )
        },
        "boundary_purity": {
            "mean":   float(np.mean(all_purity)) if all_purity else 0,
            "pct_low_purity": sum(1 for p in all_purity if p < PURITY_LOW_THRESH) / max(len(all_purity), 1),
            "interpretation": (
                "GOOD: mask boundaries align with GT labels"
                if np.mean(all_purity) > 0.75 else
                "BAD: masks cross GT label boundaries"
            )
        },
        "inter_spectral_distance": {
            "mean":   float(np.mean(all_inter_dists)) if all_inter_dists else 0,
            "pct_low_dist": len(good_merge_pairs) / max(len(all_inter_dists), 1),
            "interpretation": (
                "GOOD: adjacent masks are spectrally distinct"
                if np.mean(all_inter_dists) > 30 else
                "BAD: many adjacent masks should be merged"
            )
        },
        "n_bad_masks":       len(bad_masks),
        "n_merge_pairs":     len(good_merge_pairs),
    }

    with open("seg_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\n[DECISION]")
    print(f"  Intra-CV mean: {report['intra_cv']['mean']:.4f}")
    print(f"  Purity mean:   {report['boundary_purity']['mean']:.4f}")
    print(f"  → {report['intra_cv']['interpretation']}")
    print(f"  → {report['boundary_purity']['interpretation']}")
