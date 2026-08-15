# eval_seg_quality.py
"""
So sánh chất lượng segmentation giữa 3 versions:
  1. Original SAM2 (automatic)
  2. Refined (post-processing: split+merge)
  3. Spectral-guided SAM2

3 metrics (giống T2.1):
  - Intra-CV (thấp = tốt)
  - Boundary purity (cao = tốt)
  - Coverage (cao = tốt)

Output: bảng so sánh
"""
import os, glob, json
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# Import compute functions từ diagnose_seg_quality.py
from diagnose_seg_quality import compute_intra_cv, compute_boundary_purity

DATA_ROOT = "SECOND"
SPLIT     = "test"
N_SAMPLE  = 200

VERSIONS = {
    "original":        f"{DATA_ROOT}/sam2_masks_T1_test",
    "refined":         f"{DATA_ROOT}/sam2_masks_T1_test_refined",
    "spectral_guided": f"{DATA_ROOT}/sam2_masks_T1_test_spectral",
}


def main():
    stems = sorted([Path(f).stem for f in
        glob.glob(f"{DATA_ROOT}/{SPLIT}/im1/*.png")])[:N_SAMPLE]

    results = {}

    for version_name, mask_dir in VERSIONS.items():
        if not os.path.exists(mask_dir):
            print(f"[SKIP] {version_name}: {mask_dir} not found")
            continue

        all_cvs      = []
        all_purity   = []
        all_coverage = []

        for stem in tqdm(stems, desc=version_name):
            mask_path = os.path.join(mask_dir, stem + ".npz")
            img_path  = f"{DATA_ROOT}/{SPLIT}/im1/{stem}.png"
            lab_path  = f"{DATA_ROOT}/{SPLIT}/label1/{stem}.png"

            if not all(os.path.exists(p) for p in [mask_path, img_path, lab_path]):
                continue

            image = np.array(Image.open(img_path).convert("RGB"))
            label = np.array(Image.open(lab_path))
            masks = np.load(mask_path)["masks"]

            H, W = image.shape[:2]
            union = np.zeros((H, W), dtype=bool)

            for mask in masks:
                mask_bool = mask.astype(bool)
                # Skip background fallback mask if it's all zeros/empty
                if mask_bool.sum() == 0:
                    continue
                union |= mask_bool

                cv = compute_intra_cv(image.astype(np.float32), mask_bool)
                if cv is not None:
                    all_cvs.append(cv)

                purity = compute_boundary_purity(label, mask_bool)
                if purity is not None:
                    all_purity.append(purity)

            all_coverage.append(float(union.sum()) / (H * W))

        results[version_name] = {
            "intra_cv_mean":      round(float(np.mean(all_cvs)), 4) if all_cvs else 0.0,
            "boundary_purity":    round(float(np.mean(all_purity)), 4) if all_purity else 0.0,
            "coverage":           round(float(np.mean(all_coverage)), 4) if all_coverage else 0.0,
            "n_masks_per_image":  len(all_cvs) / max(len(stems), 1),
        }

    # Print comparison table
    print(f"\n{'Version':<20} {'IntraCV↓':>10} {'Purity↑':>10} {'Coverage↑':>10} {'N_masks':>10}")
    print("-" * 65)
    for name, r in results.items():
        print(f"{name:<20} {r['intra_cv_mean']:>10.4f} "
              f"{r['boundary_purity']:>10.4f} "
              f"{r['coverage']:>10.4f} "
              f"{r['n_masks_per_image']:>10.1f}")

    with open("seg_quality_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[DECISION]")
    valid_results = {k: v for k, v in results.items() if v['intra_cv_mean'] > 0}
    if valid_results:
        best = min(valid_results.items(), key=lambda x: x[1]["intra_cv_mean"])
        print(f"  Best IntraCV:     {best[0]} ({best[1]['intra_cv_mean']:.4f})")
        best_pur = max(valid_results.items(), key=lambda x: x[1]["boundary_purity"])
        print(f"  Best Purity:      {best_pur[0]} ({best_pur[1]['boundary_purity']:.4f})")
        best_cov = max(valid_results.items(), key=lambda x: x[1]["coverage"])
        print(f"  Best Coverage:    {best_cov[0]} ({best_cov[1]['coverage']:.4f})")
        print(f"\n  → Use best version for re-tokenization")
    else:
        print("  No valid results generated.")


if __name__ == "__main__":
    main()
