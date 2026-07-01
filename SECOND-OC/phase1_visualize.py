"""
phase1_visualize.py — Spot-check Phase 1 output on 5 random samples.

Draws instance mask overlays + class labels on T1 RGB images.
Saves to phase1_viz_{stem}.png in current directory.

Run: python SECOND-OC/phase1_visualize.py
"""
import json
import random
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from config import IMG_T1_DIR, OUT_ROOT, MASK_INST_DIR

ANN_PATH = OUT_ROOT / "annotations" / "instances_T1.json"

CLASS_COLORS = {
    "background":    (40,  40,  40),
    "tree":          (34,  139, 34),
    "buildings":     (220, 50,  50),
    "water":         (30,  100, 255),
    "non_veg_ground":(180, 170, 140),
    "playground":    (255, 200, 0),
    "low_vegetation":(100, 220, 100),
    "other":         (150, 100, 200),
}


def main():
    with open(ANN_PATH) as f:
        data = json.load(f)

    stems = [s for s, v in data["instances"].items() if v]
    chosen = random.sample(stems, min(5, len(stems)))

    for stem in chosen:
        img = Image.open(IMG_T1_DIR / f"{stem}.png").convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        insts = data["instances"][stem]
        for inst in insts:
            mask_path = OUT_ROOT / inst["mask_file"]
            if not mask_path.exists():
                continue
            mask = np.load(mask_path).astype(bool)
            color = CLASS_COLORS.get(inst["class_name"], (128, 128, 128))

            ys, xs = np.where(mask)
            for y, x in zip(ys[::4], xs[::4]):  # downsample for speed
                draw.point((x, y), fill=color + (70,))

            x1, y1, x2, y2 = inst["bbox"]
            draw.rectangle([x1, y1, x2, y2], outline=color + (220,), width=2)
            label = f"{inst['class_name'][:4]} {inst['area_px']}"
            draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 255, 255))

        result = Image.alpha_composite(img, overlay).convert("RGB")
        out_path = f"phase1_viz_{stem}.png"
        result.save(out_path)
        print(f"Saved {out_path}  ({len(insts)} instances)")

    print("\nCheck these files for visual validation:")
    print("  Each building should be a SEPARATE colored region (not merged blob).")


if __name__ == "__main__":
    main()
