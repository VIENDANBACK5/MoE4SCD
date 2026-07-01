"""
phase2_classify_changes.py — Classify per-object changes between T1 and T2.

For each T1 instance:
  Apply its mask to T2 GT label → dominant T2 class → change_type:
    unchanged       : T1_class == T2_class  (confidence >= T2_CONF_MIN)
    disappeared     : >DISAPPEARED_THRESH pixels are background in T2
    semantic_change : T1_class != T2_class  (confidence >= T2_CONF_MIN)
    ambiguous       : T2 has no dominant class (skip in eval)

For each T2 instance not covered by T1 foreground:
    appeared        : T1 pixels under T2 mask are mostly background

Output:
    SECOND-OC/annotations/change_annotations.json
"""
import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LAB_T1_DIR, LAB_T2_DIR,
    OUT_ROOT, ANN_DIR, MASK_INST_DIR,
    T2_CONF_MIN, DISAPPEARED_THRESH,
    CLASS_NAMES, rgb_to_class,
)


def _t2_lookup(mask: np.ndarray, gt_class_T2: np.ndarray) -> tuple[str, int, float]:
    """
    Apply a T1 instance mask to the T2 GT label.
    Returns (change_type, t2_class_id, t2_confidence).
    """
    t2_pixels = gt_class_T2[mask]
    total = len(t2_pixels)
    if total == 0:
        return "ambiguous", 0, 0.0

    bg_ratio = float((t2_pixels == 0).sum()) / total
    if bg_ratio > DISAPPEARED_THRESH:
        return "disappeared", 0, bg_ratio

    fg = t2_pixels[t2_pixels != 0]
    if len(fg) == 0:
        return "disappeared", 0, bg_ratio

    values, counts = np.unique(fg, return_counts=True)
    idx = counts.argmax()
    t2_cls = int(values[idx])
    t2_conf = float(counts[idx]) / len(fg)
    return "pending", t2_cls, t2_conf


def process_stem(stem: str,
                 insts_T1: list[dict],
                 insts_T2: list[dict]) -> list[dict]:
    changes = []

    # Load GT labels once per stem
    lab_t1 = LAB_T1_DIR / f"{stem}.png"
    lab_t2 = LAB_T2_DIR / f"{stem}.png"
    if not lab_t1.exists() or not lab_t2.exists():
        return changes

    gt_T1 = rgb_to_class(np.array(Image.open(lab_t1).convert("RGB")))
    gt_T2 = rgb_to_class(np.array(Image.open(lab_t2).convert("RGB")))

    # ── T1 instances → T2 lookup ───────────────────────────────────────────────
    for inst in insts_T1:
        mask_path = OUT_ROOT / inst["mask_file"]
        if not mask_path.exists():
            continue
        mask = np.load(mask_path).astype(bool)

        status, t2_cls, t2_conf = _t2_lookup(mask, gt_T2)

        if status == "pending":
            if t2_conf < T2_CONF_MIN:
                change_type = "ambiguous"
            elif t2_cls == inst["class_id"]:
                change_type = "unchanged"
            else:
                change_type = "semantic_change"
        else:
            change_type = status

        if change_type == "ambiguous":
            continue  # exclude noisy instances from benchmark

        changes.append({
            "change_id":      f"{stem}_T1_{inst['sam2_mask_id']:03d}",
            "stem":           stem,
            "source":         "T1",
            "instance_id":    inst["instance_id"],
            "class_id_T1":    inst["class_id"],
            "class_name_T1":  inst["class_name"],
            "class_id_T2":    t2_cls if change_type != "disappeared" else 0,
            "class_name_T2":  CLASS_NAMES.get(t2_cls, "background")
                              if change_type != "disappeared" else "background",
            "change_type":    change_type,
            "t2_conf":        round(t2_conf, 3),
            "area_px":        inst["area_px"],
            "centroid":       inst["centroid"],
            "bbox":           inst["bbox"],
            "sam2_score":     inst["sam2_score"],
        })

    # ── T2 instances → appeared detection ─────────────────────────────────────
    for inst in insts_T2:
        mask_path = OUT_ROOT / inst["mask_file"]
        if not mask_path.exists():
            continue
        mask = np.load(mask_path).astype(bool)

        t1_pixels = gt_T1[mask]
        t1_bg_ratio = float((t1_pixels == 0).sum()) / max(len(t1_pixels), 1)

        if t1_bg_ratio > DISAPPEARED_THRESH:
            changes.append({
                "change_id":      f"{stem}_T2_{inst['sam2_mask_id']:03d}_appeared",
                "stem":           stem,
                "source":         "T2",
                "instance_id":    inst["instance_id"],
                "class_id_T1":    0,
                "class_name_T1":  "background",
                "class_id_T2":    inst["class_id"],
                "class_name_T2":  inst["class_name"],
                "change_type":    "appeared",
                "t2_conf":        inst["dominant_ratio"],
                "area_px":        inst["area_px"],
                "centroid":       inst["centroid"],
                "bbox":           inst["bbox"],
                "sam2_score":     inst["sam2_score"],
            })

    return changes


def main():
    ann_T1 = ANN_DIR / "instances_T1.json"
    ann_T2 = ANN_DIR / "instances_T2.json"
    assert ann_T1.exists(), f"Run phase1 first: {ann_T1}"
    assert ann_T2.exists(), f"Run phase1 first: {ann_T2}"

    with open(ann_T1) as f:
        data_T1 = json.load(f)
    with open(ann_T2) as f:
        data_T2 = json.load(f)

    stems = sorted(data_T1["instances"].keys())
    print(f"Processing {len(stems)} stems...")

    all_changes: dict = {}
    type_counter: Counter = Counter()

    for stem in tqdm(stems, desc="Phase 2"):
        changes = process_stem(
            stem,
            data_T1["instances"].get(stem, []),
            data_T2["instances"].get(stem, []),
        )
        all_changes[stem] = changes
        for c in changes:
            type_counter[c["change_type"]] += 1

    out_path = ANN_DIR / "change_annotations.json"
    with open(out_path, "w") as f:
        json.dump({"n_images": len(all_changes), "changes": all_changes}, f, indent=2)

    total = sum(type_counter.values())
    print(f"\n✅ Phase 2 done. Total change events: {total}")
    for ct, n in sorted(type_counter.items()):
        print(f"   {ct:<20}: {n:>6}")
    print(f"   Saved: {out_path}")

    assert type_counter["semantic_change"] > 50, \
        "[FAIL] Too few semantic_change events — check GT label alignment"
    assert type_counter.get("appeared", 0) + type_counter.get("disappeared", 0) > 50, \
        "[FAIL] Too few appeared/disappeared — check DISAPPEARED_THRESH"
    print("   ✅ Validation passed")


if __name__ == "__main__":
    main()
