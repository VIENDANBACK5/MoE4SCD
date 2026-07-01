"""
phase3a_template_descriptions.py — Generate template captions for each change event.

Reads change_annotations.json → writes captions.jsonl (one JSON per line).
Skips "unchanged" events (no description needed).

Templates are varied by hashing instance_id to pick one of 3 variants,
ensuring the benchmark doesn't look like a simple fill-in-the-blank dataset.

Output:
    SECOND-OC/annotations/captions.jsonl
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import ANN_DIR

# ── Template banks ─────────────────────────────────────────────────────────────
TEMPLATES = {
    "appeared": [
        "A new {cls_T2} region emerged in the {quad} of the scene.",
        "In T2, a {size} {cls_T2} area appeared in the {quad} that was not present in T1.",
        "{cls_T2} land cover was newly established in the {quad} between T1 and T2.",
    ],
    "disappeared": [
        "The {cls_T1} area in the {quad} of the scene disappeared by T2.",
        "A {size} {cls_T1} region visible in T1 was no longer present in T2.",
        "Land cover in the {quad} changed from {cls_T1} to bare ground between T1 and T2.",
    ],
    "semantic_change": [
        "Land cover in the {quad} changed from {cls_T1} to {cls_T2} between T1 and T2.",
        "A {size} {cls_T1} area in the {quad} was converted to {cls_T2} in T2.",
        "In the {quad}, the {cls_T1} visible in T1 was replaced by {cls_T2} in T2.",
    ],
}

LABEL_PRETTY = {
    "background":     "bare ground",
    "tree":           "tree canopy",
    "buildings":      "buildings",
    "water":          "water body",
    "non_veg_ground": "non-vegetated ground",
    "playground":     "playground",
    "low_vegetation": "low vegetation",
    "other":          "other land cover",
}


def _quadrant(cx: float, cy: float) -> str:
    """Convert normalized centroid → human-readable quadrant."""
    v = "upper" if cy < 0.5 else "lower"
    h = "left"  if cx < 0.5 else "right"
    if 0.33 < cx < 0.67 and 0.33 < cy < 0.67:
        return "center"
    return f"{v}-{h}"


def _size_label(area_px: int) -> str:
    if area_px < 500:
        return "small"
    if area_px < 3000:
        return "medium-sized"
    return "large"


def _pretty(class_name: str) -> str:
    return LABEL_PRETTY.get(class_name, class_name.replace("_", " "))


def make_caption(change: dict) -> str:
    ct = change["change_type"]
    if ct not in TEMPLATES:
        return ""

    # pick template variant deterministically via hash of instance_id
    variants = TEMPLATES[ct]
    idx = hash(change["change_id"]) % len(variants)
    template = variants[idx]

    cx, cy = change["centroid"]
    return template.format(
        cls_T1=_pretty(change["class_name_T1"]),
        cls_T2=_pretty(change["class_name_T2"]),
        quad=_quadrant(cx, cy),
        size=_size_label(change["area_px"]),
    )


def main():
    ann_path = ANN_DIR / "change_annotations.json"
    assert ann_path.exists(), f"Run phase2 first: {ann_path}"

    with open(ann_path) as f:
        data = json.load(f)

    out_path = ANN_DIR / "captions.jsonl"
    n_written = n_skipped = 0

    with open(out_path, "w") as out:
        for stem, changes in data["changes"].items():
            for ch in changes:
                if ch["change_type"] == "unchanged":
                    n_skipped += 1
                    continue

                caption = make_caption(ch)
                if not caption:
                    n_skipped += 1
                    continue

                record = {
                    "change_id":   ch["change_id"],
                    "stem":        stem,
                    "change_type": ch["change_type"],
                    "caption":     caption,
                    "class_T1":    ch["class_name_T1"],
                    "class_T2":    ch["class_name_T2"],
                    "centroid":    ch["centroid"],
                    "bbox":        ch["bbox"],
                    "area_px":     ch["area_px"],
                }
                out.write(json.dumps(record) + "\n")
                n_written += 1

    print(f"✅ Phase 3A done.")
    print(f"   Written: {n_written} captions → {out_path}")
    print(f"   Skipped: {n_skipped} (unchanged / ambiguous)")

    # Spot-check: print 3 random captions
    print("\nSample captions:")
    import random
    with open(out_path) as f:
        lines = f.readlines()
    for line in random.sample(lines, min(3, len(lines))):
        rec = json.loads(line)
        print(f"  [{rec['change_type']}] {rec['caption']}")

    assert n_written > 500, f"[FAIL] Only {n_written} captions — check change_annotations.json"
    print("\n   ✅ Validation passed")


if __name__ == "__main__":
    main()
