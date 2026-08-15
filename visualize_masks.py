# visualize_masks.py
"""
So sánh mask quality giữa:
  - SAM2 gốc (frozen)
  - SAM2 sau LoRA fine-tuning

Kiểm tra 2 vấn đề chính:
  1. Boundary sharpness: ranh giới có sắc nét không?
  2. Fragmentation: 1 object có bị split thành nhiều mask không?

Output: 10 ảnh so sánh side-by-side
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw

DATA_ROOT    = "SECOND/test"
MASKS_ORIG   = "SECOND/sam2_masks_T1_test"   # masks từ SAM2 gốc
MASKS_LORA   = "SECOND/sam2_masks_T1_test_lora_r4"  # masks từ SAM2 + LoRA (cần tạo nếu LoRA thay đổi masks)
OUT_DIR      = "mask_quality_comparison"
N_SAMPLES    = 10

os.makedirs(OUT_DIR, exist_ok=True)

# Load stems ngẫu nhiên
all_stems = sorted([f.replace(".npz", "") for f in os.listdir(MASKS_ORIG)
                    if f.endswith(".npz")])
random.seed(42)
stems = random.sample(all_stems, N_SAMPLES)


def draw_masks_on_image(img: Image.Image, masks: np.ndarray,
                         title: str) -> Image.Image:
    """Vẽ mask boundaries lên ảnh với màu ngẫu nhiên."""
    img_rgba = img.convert("RGBA")
    overlay  = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    colors = [(np.random.randint(100,255), np.random.randint(100,255),
               np.random.randint(100,255), 80) for _ in range(len(masks))]

    for i, (mask, color) in enumerate(zip(masks, colors)):
        # Vẽ filled region
        ys, xs = np.where(mask)
        for y, x in zip(ys[::4], xs[::4]):  # downsample
            draw.point((x, y), fill=color)

        # Vẽ boundary bằng erosion đơn giản trong numpy
        # erosion thủ công: nếu pixel có lân cận là 0
        from scipy.ndimage import binary_erosion
        boundary = mask ^ binary_erosion(mask)
        ys_b, xs_b = np.where(boundary)
        for y, x in zip(ys_b, xs_b):
            draw.point((x, y), fill=(255, 255, 0, 200))  # vàng = boundary

    result = Image.alpha_composite(img_rgba, overlay).convert("RGB")

    # Thêm title
    draw_r = ImageDraw.Draw(result)
    draw_r.rectangle([0, 0, result.width, 25], fill=(0, 0, 0))
    draw_r.text((5, 5), title, fill=(255, 255, 255))

    # Thêm metrics
    n_masks = len(masks)
    avg_area = masks.sum(axis=(1, 2)).mean() if len(masks) > 0 else 0
    draw_r.text((5, img.height - 30),
                f"N masks={n_masks}, avg area={avg_area:.0f}px",
                fill=(255, 255, 0))
    return result


for stem in stems:
    # Load ảnh gốc
    img_path = os.path.join(DATA_ROOT, "im1", stem + ".png")
    if not os.path.exists(img_path):
        continue
    img = Image.open(img_path)

    # Load masks gốc (SAM2 frozen)
    data_orig = np.load(os.path.join(MASKS_ORIG, stem + ".npz"))
    masks_orig = data_orig["masks"].astype(bool)

    # Load masks LoRA (nếu có)
    lora_path = os.path.join(MASKS_LORA, stem + ".npz")
    has_lora = os.path.exists(lora_path)

    if has_lora:
        data_lora = np.load(lora_path)
        masks_lora = data_lora["masks"].astype(bool)
    else:
        masks_lora = masks_orig  # fallback

    # Vẽ so sánh
    img_orig = draw_masks_on_image(img, masks_orig, "SAM2 Frozen (Baseline)")
    img_lora = draw_masks_on_image(img, masks_lora,
                                    "SAM2 + LoRA r=4" if has_lora else "LoRA N/A")

    # Side-by-side
    combined = Image.new("RGB", (img.width * 2, img.height + 50))
    combined.paste(img_orig, (0, 50))
    combined.paste(img_lora, (img.width, 50))

    # Header
    draw_c = ImageDraw.Draw(combined)
    draw_c.rectangle([0, 0, combined.width, 50], fill=(30, 30, 30))
    draw_c.text((10, 15), f"Mask Quality Comparison — {stem}", fill=(255,255,255))

    # Metrics comparison
    n_orig = len(masks_orig)
    n_lora = len(masks_lora) if has_lora else 0
    draw_c.text((combined.width//2 + 10, 15),
                f"N masks: {n_orig} → {n_lora} "
                f"({'same' if n_orig==n_lora else 'CHANGED'})",
                fill=(255, 200, 0))

    out_path = os.path.join(OUT_DIR, f"comparison_{stem}.png")
    combined.save(out_path)
    print(f"Saved: {out_path}")

print(f"\n✅ Done. Check {OUT_DIR}/ for {N_SAMPLES} comparison images.")
