"""
phase0_verify.py — Verify SECOND-OC data layout before running Phase 1.

Checks:
  - 1694 test pairs exist
  - GT labels are RGB, correct unique values
  - SAM2 mask .npz files exist (from generate_sam2_masks_test.py)
  - Output directories exist
  - Sample end-to-end sanity

Output: phase0_report.json
"""
import json
import os
import numpy as np
from pathlib import Path
from PIL import Image

# Use config from parent directory
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    IMG_T1_DIR, IMG_T2_DIR, LAB_T1_DIR, LAB_T2_DIR,
    MASK_T1_DIR, MASK_T2_DIR, OUT_ROOT, ANN_DIR,
    MASK_INST_DIR, TIER1_DIR, EVAL_DIR, rgb_to_class, CLASS_NAMES
)

REPORT_PATH = Path("SECOND-OC") / "phase0_report.json"


def main():
    report = {}

    # 1. Check required directories
    dirs = {
        "im1": IMG_T1_DIR, "im2": IMG_T2_DIR,
        "label1": LAB_T1_DIR, "label2": LAB_T2_DIR,
        "masks_T1": MASK_T1_DIR, "masks_T2": MASK_T2_DIR,
    }
    report["dirs_exist"] = {k: str(v.exists()) for k, v in dirs.items()}

    missing = [k for k, v in dirs.items() if not v.exists()]
    if missing:
        print(f"[ERROR] Missing dirs: {missing}")
        if "masks_T1" in missing or "masks_T2" in missing:
            print("  → Run: python generate_sam2_masks_test.py")
    else:
        print("[1] All required directories exist ✅")

    # 2. Count test pairs
    stems_im  = sorted(p.stem for p in IMG_T1_DIR.glob("*.png"))
    stems_lab = sorted(p.stem for p in LAB_T1_DIR.glob("*.png"))
    stems_msk = sorted(p.stem for p in MASK_T1_DIR.glob("*.npz")) if MASK_T1_DIR.exists() else []

    common = sorted(set(stems_im) & set(stems_lab))
    report["n_test_pairs"] = len(common)
    report["split_ok"]     = len(common) == 1694
    report["n_masks_T1"]   = len(stems_msk)
    report["sample_stems"] = common[:5]

    print(f"[2] Test pairs: {len(common)}  ({'✅ 1694' if len(common) == 1694 else '⚠️ MISMATCH'})")
    print(f"    SAM2 masks T1: {len(stems_msk)} ({'✅' if len(stems_msk) == 1694 else '⚠️ need generate_sam2_masks_test.py'})")

    # 3. Verify GT label format
    label_rgbs = set()
    for stem in common[:30]:
        arr = np.array(Image.open(LAB_T1_DIR / f"{stem}.png").convert("RGB"))
        for row in np.unique(arr.reshape(-1, 3), axis=0):
            label_rgbs.add(tuple(row.tolist()))

    report["gt_label_unique_rgb"] = sorted(str(r) for r in label_rgbs)
    print(f"[3] GT label unique RGB colors (30 samples): {sorted(label_rgbs)}")

    # 4. Test rgb_to_class on a sample
    arr = np.array(Image.open(LAB_T1_DIR / f"{common[0]}.png").convert("RGB"))
    cls_map = rgb_to_class(arr)
    cls_vals = np.unique(cls_map).tolist()
    report["sample_class_ids"] = cls_vals
    print(f"[4] Sample {common[0]} class IDs: {cls_vals}")
    print(f"    Class names: {[CLASS_NAMES[c] for c in cls_vals]}")

    # 5. Check SAM2 mask sample
    if MASK_T1_DIR.exists() and stems_msk:
        data = np.load(MASK_T1_DIR / f"{stems_msk[0]}.npz")
        n_masks = data["masks"].shape[0]
        report["sample_mask"] = {
            "stem": stems_msk[0],
            "n_masks": n_masks,
            "mask_shape": list(data["masks"].shape),
        }
        print(f"[5] SAM2 mask sample {stems_msk[0]}: {n_masks} masks, shape={data['masks'].shape}")
    else:
        report["sample_mask"] = None
        print("[5] ⚠️  SAM2 masks not found — run generate_sam2_masks_test.py first")

    # 6. Create output dirs
    for d in [OUT_ROOT, ANN_DIR, MASK_INST_DIR / "T1", MASK_INST_DIR / "T2",
              TIER1_DIR, EVAL_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("[6] Output directories ready ✅")

    # 7. Save report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Phase 0 complete. Report: {REPORT_PATH}")

    # 8. Summary decision
    blockers = []
    if not report["split_ok"]:
        blockers.append(f"Test pairs = {len(common)} (expected 1694)")
    if report["n_masks_T1"] != 1694:
        blockers.append(f"SAM2 masks = {report['n_masks_T1']} (expected 1694) — run generate_sam2_masks_test.py")

    if blockers:
        print("\n⛔ BLOCKERS before Phase 1:")
        for b in blockers:
            print(f"   - {b}")
    else:
        print("\n✅ All checks passed — ready to run phase1_extract_instances.py")


if __name__ == "__main__":
    main()
