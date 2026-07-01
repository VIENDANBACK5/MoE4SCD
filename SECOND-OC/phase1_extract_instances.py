"""
phase1_extract_instances.py — Extract object instances from SAM2 masks ∩ GT labels.

For each test image (T1 and T2):
  1. Load SAM2 masks (from generate_sam2_masks_test.py output)
  2. Load GT semantic label (RGB → class_id via config.rgb_to_class)
  3. Per mask: assign dominant class, filter small / mixed-class masks
  4. Save per-instance mask as SECOND-OC/masks/{T1|T2}/{instance_id}.npy
  5. Collect metadata into instances_T1.json / instances_T2.json

Output:
    SECOND-OC/annotations/instances_T1.json
    SECOND-OC/annotations/instances_T2.json
    SECOND-OC/masks/T1/{instance_id}.npy
    SECOND-OC/masks/T2/{instance_id}.npy
"""
import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    IMG_T1_DIR, LAB_T1_DIR, LAB_T2_DIR,
    ANN_DIR, MASK_INST_DIR,
    MIN_AREA_PX, DOMINANT_CLASS_THRESH,
    CLASS_NAMES, rgb_to_class,
)
from sam2_loader import load_sam2_masks


def extract_instances(stem: str, time: str) -> list[dict]:
    """
    Extract object instances for one image (T1 or T2).
    Returns list of instance metadata dicts (no numpy arrays).
    Masks are saved to disk per-instance.
    """
    lab_dir  = LAB_T1_DIR if time == "T1" else LAB_T2_DIR
    lab_path = lab_dir / f"{stem}.png"
    if not lab_path.exists():
        return []

    gt_rgb   = np.array(Image.open(lab_path).convert("RGB"))
    gt_class = rgb_to_class(gt_rgb)  # (512, 512) int

    sam2_masks = load_sam2_masks(stem, time)
    if not sam2_masks:
        return []

    instances = []
    mask_out_dir = MASK_INST_DIR / time
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    for item in sam2_masks:
        mask     = item["mask"]   # (512, 512) bool
        mask_id  = item["mask_id"]
        score    = item["score"]

        area_px = int(mask.sum())
        if area_px < MIN_AREA_PX:
            continue

        # Assign class from GT label pixels inside the mask
        pixels = gt_class[mask]
        # Exclude background (class 0) from class assignment
        semantic_pixels = pixels[pixels != 0]
        if len(semantic_pixels) == 0:
            continue

        values, counts = np.unique(semantic_pixels, return_counts=True)
        dom_idx        = np.argmax(counts)
        dom_class_id   = int(values[dom_idx])
        dom_ratio      = float(counts[dom_idx]) / len(semantic_pixels)

        # Drop masks where no single class dominates
        if dom_ratio < DOMINANT_CLASS_THRESH:
            continue

        class_name  = CLASS_NAMES.get(dom_class_id, f"class_{dom_class_id}")
        ys, xs      = np.where(mask)
        centroid    = [float(xs.mean()) / 512, float(ys.mean()) / 512]
        bbox        = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        instance_id = f"{stem}_{time}_{mask_id:03d}"
        mask_file   = f"masks/{time}/{instance_id}.npy"

        # Save binary mask
        np.save(MASK_INST_DIR / time / f"{instance_id}.npy", mask)

        instances.append({
            "instance_id":    instance_id,
            "stem":           stem,
            "time":           time,
            "sam2_mask_id":   mask_id,
            "sam2_score":     round(score, 4),
            "class_id":       dom_class_id,
            "class_name":     class_name,
            "dominant_ratio": round(dom_ratio, 3),
            "area_px":        area_px,
            "centroid":       centroid,
            "bbox":           bbox,
            "mask_file":      mask_file,
        })

    return instances


def main():
    stems = sorted(p.stem for p in IMG_T1_DIR.glob("*.png"))
    print(f"Processing {len(stems)} stems...")

    all_T1: dict = {}
    all_T2: dict = {}
    skip_T1 = skip_T2 = 0

    for stem in tqdm(stems, desc="Phase 1"):
        insts_T1 = extract_instances(stem, "T1")
        insts_T2 = extract_instances(stem, "T2")
        all_T1[stem] = insts_T1
        all_T2[stem] = insts_T2
        if not insts_T1:
            skip_T1 += 1
        if not insts_T2:
            skip_T2 += 1

    # Save JSON
    out_T1 = ANN_DIR / "instances_T1.json"
    out_T2 = ANN_DIR / "instances_T2.json"
    with open(out_T1, "w") as f:
        json.dump({"n_images": len(all_T1), "instances": all_T1}, f, indent=2)
    with open(out_T2, "w") as f:
        json.dump({"n_images": len(all_T2), "instances": all_T2}, f, indent=2)

    # Stats
    total_T1 = sum(len(v) for v in all_T1.values())
    total_T2 = sum(len(v) for v in all_T2.values())
    print(f"\n✅ Phase 1 done.")
    print(f"   T1 instances: {total_T1}  ({skip_T1} images with 0)")
    print(f"   T2 instances: {total_T2}  ({skip_T2} images with 0)")
    print(f"   Saved: {out_T1}")
    print(f"   Saved: {out_T2}")

    # Validation
    assert total_T1 > 1000, f"[FAIL] T1 count {total_T1} too low — check SAM2 masks"
    assert total_T2 > 1000, f"[FAIL] T2 count {total_T2} too low — check SAM2 masks"
    assert skip_T1 < len(stems) * 0.2, \
        f"[FAIL] {skip_T1}/{len(stems)} T1 images have 0 instances"
    print("   ✅ Validation passed")


if __name__ == "__main__":
    main()
