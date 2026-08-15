
# AGENT PLAN: Spectral-Guided Segmentation Quality Improvement (Tầng 2)
> Mục tiêu: Dùng spectral information để cải thiện chất lượng SAM2 mask boundaries
> Approach: Chẩn đoán → Post-processing refinement → Spectral-guided prompting → Eval
> Chạy theo thứ tự: T2.1 → T2.2 → T2.3 → T2.4 → eval

---

## BỨC TRANH TỔNG THỂ

```
Hiện tại (Tầng 1 only):
  RGB image
      ↓ SAM2 (visual only)
  Masks (boundaries dựa trên hình dạng)
      ↓ spectral features
  Classification

Sau khi fix Tầng 2:
  RGB image
      ↓ spectral analysis → tìm vùng đồng nhất vs biên giới
  Spectral-informed prompts
      ↓ SAM2
  Better masks (boundaries align với spectral boundaries)
      ↓ spectral features
  Better classification
```

---

## FILE MAP

```
Image Segmentation/
├── diagnose_seg_quality.py      [NEW] T2.1: đo chất lượng masks hiện tại
├── spectral_mask_refine.py      [NEW] T2.2: post-processing split/merge masks
├── spectral_edge_prompt.py      [NEW] T2.3: spectral edge detection → SAM2 prompts
├── generate_sam2_spectral.py    [NEW] T2.3: re-generate masks với spectral prompts
├── eval_seg_quality.py          [NEW] T2.4: đo improvement sau refinement
└── retrain_refined.py           [NEW] T2.4: retrain Token-MoE với refined masks
```

---

## ════════════════════════════════════
## T2.1 — CHẨN ĐOÁN: Đo chất lượng masks hiện tại
## ════════════════════════════════════
> Không cần GPU. Chạy trước, lấy số làm baseline.

### Mục đích
Thầy nói: *"Cái bước segmentation làm chưa tốt ảnh hưởng classification về sau"*
→ Cần số liệu cụ thể: masks hiện tại TỐT hay XẤU về mặt spectral?

**2 tiêu chí từ Dao et al. 2021:**
- **Intra-object homogeneity**: pixels trong cùng mask có spectral giống nhau không?
  (CV thấp = đồng nhất = mask tốt)
- **Inter-object heterogeneity**: masks khác nhau có spectral khác nhau không?
  (khoảng cách spectral cao = phân biệt tốt = mask tốt)

---

### T2.1.0 — Prerequisite

```bash
python -c "
import glob
checks = {
    'sam2_masks_T1_test': 'SECOND/sam2_masks_T1_test/*.npz',
    'images_test_T1':     'SECOND/test/im1/*.png',
    'label_test_T1':      'SECOND/test/label1/*.png',
}
for name, pattern in checks.items():
    n = len(glob.glob(pattern))
    print(f'{'✅' if n>0 else '❌'} {name}: {n} files')
"
```

---

### T2.1.1 — Tạo `diagnose_seg_quality.py`

```python
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
from scipy.spatial import KDTree

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
```

**Chạy:** `python diagnose_seg_quality.py`

**Đọc kết quả — 3 trường hợp:**

```
Case A: CV mean < 0.10 AND Purity > 0.80
  → SAM2 masks đã tốt về spectral
  → Vấn đề là thiếu spectral features trong token (đã fix với B1)
  → Không cần T2.2, T2.3

Case B: CV mean > 0.15 (nhiều masks không đồng nhất)
  → Masks bị over-merge: 1 mask chứa 2 objects khác nhau
  → Cần T2.2: SPLIT masks dựa trên spectral

Case C: Purity < 0.70 (boundaries không align với GT)
  → SAM2 boundaries cắt ngang semantic objects
  → Cần T2.3: Spectral-guided prompting để tạo better boundaries
```

---

## ════════════════════════════════════
## T2.2 — POST-PROCESSING: Split/Merge Masks theo Spectral
## ════════════════════════════════════
> Chạy nếu Case B hoặc C từ T2.1.

### Mục đích
- **Split**: Mask có CV cao → chia thành 2+ masks theo spectral boundary
- **Merge**: 2 adjacent masks có spectral giống nhau → gộp lại

### T2.2.1 — Tạo `spectral_mask_refine.py`

```python
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
import os, glob
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans

DATA_ROOT     = "SECOND"
ORIG_MASK_DIR = "SECOND/sam2_masks_T1_test"
OUT_MASK_DIR  = "SECOND/sam2_masks_T1_test_refined"
IMG_DIR       = "SECOND/test/im1"
os.makedirs(OUT_MASK_DIR, exist_ok=True)

# Hyperparameters (từ seg_quality_report.json để calibrate)
CV_SPLIT_THRESH   = 0.15   # CV > này → split
MERGE_DIST_THRESH = 20.0   # spectral distance < này → merge (RGB 0-255)
MIN_AREA_PX       = 50     # bỏ mask quá nhỏ sau split
EPS = 1e-8


def spectral_mean(image, mask):
    """Mean RGB của vùng mask, float32."""
    pixels = image[mask.astype(bool)].astype(np.float32)
    return pixels.mean(axis=0) if len(pixels) > 0 else np.zeros(3)


def compute_cv(image, mask):
    """Coefficient of Variation của mask."""
    pixels = image[mask.astype(bool)].astype(np.float32)
    if len(pixels) < 10:
        return 0.0
    mean = pixels.mean(axis=0) + EPS
    std  = pixels.std(axis=0)
    return float((std / mean).mean())


def split_mask_spectral(image, mask, n_clusters=2):
    """
    Split mask bằng k-means trên spectral.
    Returns list of sub-masks (có thể 1 nếu không split được).
    """
    ys, xs = np.where(mask)
    if len(ys) < n_clusters * MIN_AREA_PX:
        return [mask]   # quá nhỏ để split

    # Lấy RGB của pixels trong mask
    pixels_rgb = image[ys, xs].astype(np.float32)

    # K-means trên RGB space
    try:
        km = KMeans(n_clusters=n_clusters, n_init=3, random_state=42)
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
        if sub.sum() >= MIN_AREA_PX:
            sub_masks.append(sub)

    return sub_masks if len(sub_masks) > 1 else [mask]


def are_adjacent(mask1, mask2, dilation=3):
    """Kiểm tra 2 masks có kề nhau không (dùng dilation)."""
    from scipy.ndimage import binary_dilation
    dilated1 = binary_dilation(mask1, iterations=dilation)
    return bool((dilated1 & mask2).any())


def refine_masks(image, masks):
    """
    Áp dụng split và merge trên tất cả masks của 1 ảnh.
    Returns: list of refined masks
    """
    refined = []

    # --- STEP 1: SPLIT masks có CV cao ---
    for mask in masks:
        cv = compute_cv(image, mask)
        if cv > CV_SPLIT_THRESH:
            sub_masks = split_mask_spectral(image, mask, n_clusters=2)
            refined.extend(sub_masks)
        else:
            refined.append(mask)

    # --- STEP 2: MERGE adjacent masks có spectral giống nhau ---
    merged = True
    while merged:
        merged = False
        new_refined = []
        used = [False] * len(refined)

        for i in range(len(refined)):
            if used[i]:
                continue
            current = refined[i]
            current_mean = spectral_mean(image, current)

            for j in range(i+1, len(refined)):
                if used[j]:
                    continue

                # Check spectral distance
                other_mean = spectral_mean(image, refined[j])
                dist = float(np.linalg.norm(current_mean - other_mean))

                # Check adjacency
                if dist < MERGE_DIST_THRESH and are_adjacent(current, refined[j]):
                    # Merge
                    current = current | refined[j]
                    current_mean = spectral_mean(image, current)
                    used[j] = True
                    merged = True

            new_refined.append(current)
            used[i] = True

        refined = new_refined

    # Filter nhỏ quá
    refined = [m for m in refined if m.sum() >= MIN_AREA_PX]
    return refined


# --- Main loop ---
stems = sorted([Path(f).stem for f in glob.glob(f"{ORIG_MASK_DIR}/*.npz")])
total_orig = 0
total_refined = 0

for stem in tqdm(stems, desc="Refining masks"):
    img_path  = f"{IMG_DIR}/{stem}.png"
    mask_path = f"{ORIG_MASK_DIR}/{stem}.npz"

    if not os.path.exists(img_path):
        continue

    image = np.array(Image.open(img_path).convert("RGB"))
    data  = np.load(mask_path)
    masks = [data["masks"][i].astype(bool) for i in range(len(data["masks"]))]

    refined_masks = refine_masks(image, masks)

    total_orig    += len(masks)
    total_refined += len(refined_masks)

    # Save
    masks_arr = np.stack(refined_masks).astype(bool)
    scores    = np.ones(len(refined_masks), dtype=np.float32)
    np.savez_compressed(
        f"{OUT_MASK_DIR}/{stem}.npz",
        masks=masks_arr, scores=scores
    )

print(f"\n✅ Refinement done.")
print(f"   Original masks:  {total_orig} ({total_orig/len(stems):.1f}/image)")
print(f"   Refined masks:   {total_refined} ({total_refined/len(stems):.1f}/image)")
print(f"   Change:          {total_refined - total_orig:+d}")
```

**Chạy:** `python spectral_mask_refine.py`

---

## ════════════════════════════════════
## T2.3 — SPECTRAL-GUIDED SAM2 PROMPTING
## ════════════════════════════════════
> Cải thiện boundaries ngay từ đầu, không phải post-process.
> Cần GPU. Chạy sau T2.2 để so sánh.

### Ý tưởng
```
Thay vì để SAM2 tự chọn prompt points (automatic mode)
→ Dùng spectral edge detection để tìm vùng:
    1. Spectral gradient cao → ranh giới giữa 2 objects
    2. Spectral homogeneous center → tâm của 1 object
→ Dùng 2 loại điểm này làm SAM2 prompts
→ SAM2 sẽ tạo mask align với spectral boundaries
```

### T2.3.1 — Tạo `spectral_edge_prompt.py`

```python
# spectral_edge_prompt.py
"""
Tính spectral edge map từ ảnh RGB.
Output: prompt points cho SAM2 (center points của homogeneous regions)

Sử dụng Sobel gradient trên từng channel RGB
→ Vùng gradient thấp = interior of objects → SAM2 foreground prompts
→ Vùng gradient cao = boundaries → SAM2 không prompt tại đây
"""
import numpy as np
from PIL import Image
from scipy import ndimage


def compute_spectral_gradient(image_rgb):
    """
    Tính magnitude của spectral gradient.
    Gradient cao = ranh giới spectral giữa objects.

    Returns: (H, W) float32, normalized [0, 1]
    """
    image_f = image_rgb.astype(np.float32) / 255.0
    grad_mag = np.zeros(image_f.shape[:2], dtype=np.float32)

    for c in range(3):  # R, G, B
        channel = image_f[:, :, c]
        # Sobel gradient
        gx = ndimage.sobel(channel, axis=1)
        gy = ndimage.sobel(channel, axis=0)
        grad_mag += np.sqrt(gx**2 + gy**2)

    # Normalize
    grad_mag /= 3.0
    if grad_mag.max() > 0:
        grad_mag /= grad_mag.max()

    return grad_mag


def find_homogeneous_centers(grad_map, grid_size=16, low_grad_thresh=0.15):
    """
    Tìm các điểm trung tâm của vùng spectral đồng nhất.
    Đây là nơi tốt nhất để prompt SAM2.

    Args:
        grad_map: (H, W) spectral gradient magnitude
        grid_size: khoảng cách giữa các candidate points
        low_grad_thresh: gradient < này → interior of object

    Returns: list of (x, y) prompt points
    """
    H, W = grad_map.shape
    points = []

    for y in range(grid_size // 2, H, grid_size):
        for x in range(grid_size // 2, W, grid_size):
            # Lấy local region
            y1, y2 = max(0, y-grid_size//4), min(H, y+grid_size//4)
            x1, x2 = max(0, x-grid_size//4), min(W, x+grid_size//4)
            local_grad = grad_map[y1:y2, x1:x2].mean()

            # Chỉ prompt tại vùng có gradient thấp (interior)
            if local_grad < low_grad_thresh:
                points.append((x, y))

    return points


def generate_spectral_prompts(image_rgb, grid_size=16, grad_thresh=0.15):
    """
    Main function: từ ảnh RGB → prompt points cho SAM2.
    """
    grad_map = compute_spectral_gradient(image_rgb)
    points   = find_homogeneous_centers(grad_map, grid_size, grad_thresh)
    return points, grad_map


# --- Test ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = np.array(Image.open("SECOND/test/im1/00004.png").convert("RGB"))
    points, grad_map = generate_spectral_prompts(img, grid_size=16)

    print(f"Image size: {img.shape}")
    print(f"Spectral gradient range: [{grad_map.min():.3f}, {grad_map.max():.3f}]")
    print(f"Number of prompt points: {len(points)}")
    print(f"  (vs grid prompting 8×8 = {8*8} points)")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")

    axes[1].imshow(grad_map, cmap='hot')
    axes[1].set_title("Spectral Gradient\n(bright = boundary)")

    axes[2].imshow(img)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    axes[2].scatter(xs, ys, c='lime', s=20, alpha=0.7)
    axes[2].set_title(f"SAM2 Prompts\n({len(points)} points, spectral-guided)")

    plt.tight_layout()
    plt.savefig("spectral_prompts_00004.png", dpi=100)
    print("Saved: spectral_prompts_00004.png")
    print("✅ Test passed")
```

### T2.3.2 — Tạo `generate_sam2_spectral.py`

```python
# generate_sam2_spectral.py
"""
Re-generate SAM2 masks với spectral-guided prompts.
Thay thế uniform grid prompts bằng spectral homogeneous region centers.

Output: SECOND/sam2_masks_T1_test_spectral/*.npz
"""
import os, glob, argparse
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from spectral_edge_prompt import generate_spectral_prompts

MERGE_IOU_THRESH = 0.7
MIN_AREA_PX      = 50


def compute_iou(m1, m2):
    inter = (m1 & m2).sum()
    union = (m1 | m2).sum()
    return float(inter) / float(union + 1e-8)


def merge_duplicate_masks(masks_list, iou_thresh=MERGE_IOU_THRESH):
    """Loại bỏ duplicate masks."""
    masks_list = sorted(masks_list, key=lambda x: x["area"], reverse=True)
    kept = []
    for cand in masks_list:
        is_dup = any(compute_iou(cand["mask"], k["mask"]) > iou_thresh
                     for k in kept)
        if not is_dup:
            kept.append(cand)
    return kept


def main(args):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    print(f"✅ SAM2 loaded on {device}")

    stems = sorted([Path(f).stem for f in glob.glob(f"{args.img_dir}/*.png")])
    os.makedirs(args.out_dir, exist_ok=True)

    coverage_list = []
    n_prompt_list = []

    for stem in tqdm(stems):
        out_path = os.path.join(args.out_dir, stem + ".npz")
        if os.path.exists(out_path) and not args.overwrite:
            continue

        image = np.array(Image.open(
            os.path.join(args.img_dir, stem + ".png")
        ).convert("RGB"))

        # Spectral-guided prompts
        prompt_points, grad_map = generate_spectral_prompts(
            image,
            grid_size=args.grid_size,
            grad_thresh=args.grad_thresh
        )
        n_prompt_list.append(len(prompt_points))

        predictor.set_image(image)
        all_masks = []

        for (x, y) in prompt_points:
            input_point = np.array([[x, y]], dtype=np.float32)
            input_label = np.array([1], dtype=np.int32)

            try:
                masks, scores, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                best_idx = scores.argmax()
                m = masks[best_idx].astype(bool)
                if m.sum() >= MIN_AREA_PX:
                    all_masks.append({
                        "mask":  m,
                        "score": float(scores[best_idx]),
                        "area":  int(m.sum()),
                    })
            except Exception:
                continue

        merged = merge_duplicate_masks(all_masks)

        if not merged:
            continue

        # Coverage
        H, W  = image.shape[:2]
        union = np.zeros((H, W), dtype=bool)
        for m in merged:
            union |= m["mask"]
        coverage_list.append(float(union.sum()) / (H * W))

        masks_arr  = np.stack([m["mask"]  for m in merged]).astype(bool)
        scores_arr = np.array([m["score"] for m in merged])
        np.savez_compressed(out_path, masks=masks_arr, scores=scores_arr)

    print(f"\n✅ Spectral-guided SAM2 masks generated.")
    if coverage_list:
        print(f"   Coverage:     {np.mean(coverage_list)*100:.1f}%")
        print(f"   Avg prompts:  {np.mean(n_prompt_list):.0f} per image")
        print(f"   (vs uniform grid 8×8 = 64 prompts)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir",         required=True)
    p.add_argument("--out-dir",         required=True)
    p.add_argument("--sam2-config",     default="sam2/configs/sam2_hiera_l.yaml")
    p.add_argument("--sam2-checkpoint", default="sam2/checkpoints/sam2_hiera_large.pt")
    p.add_argument("--grid-size",       type=int,   default=16)
    p.add_argument("--grad-thresh",     type=float, default=0.15)
    p.add_argument("--overwrite",       action="store_true")
    main(p.parse_args())
```

**Chạy:**
```bash
# Visualize prompts trước để kiểm tra
python spectral_edge_prompt.py
# Mở spectral_prompts_00004.png — prompt points có nằm trong objects không?

# Nếu ổn thì generate masks
python generate_sam2_spectral.py \
    --img-dir SECOND/test/im1 \
    --out-dir SECOND/sam2_masks_T1_test_spectral
```

---

## ════════════════════════════════════
## T2.4 — EVAL: So sánh mask quality trước và sau
## ════════════════════════════════════

### T2.4.1 — Tạo `eval_seg_quality.py`

```python
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

# Import compute functions từ T2.1
from diagnose_seg_quality import compute_intra_cv, compute_boundary_purity

DATA_ROOT = "SECOND"
SPLIT     = "test"
N_SAMPLE  = 200

VERSIONS = {
    "original":        f"{DATA_ROOT}/sam2_masks_T1_test",
    "refined":         f"{DATA_ROOT}/sam2_masks_T1_test_refined",
    "spectral_guided": f"{DATA_ROOT}/sam2_masks_T1_test_spectral",
}

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
            union |= mask_bool

            cv = compute_intra_cv(image.astype(np.float32), mask_bool)
            if cv is not None:
                all_cvs.append(cv)

            purity = compute_boundary_purity(label, mask_bool)
            if purity is not None:
                all_purity.append(purity)

        all_coverage.append(float(union.sum()) / (H * W))

    results[version_name] = {
        "intra_cv_mean":      round(float(np.mean(all_cvs)), 4),
        "boundary_purity":    round(float(np.mean(all_purity)), 4),
        "coverage":           round(float(np.mean(all_coverage)), 4),
        "n_masks_per_image":  len(all_cvs) / max(N_SAMPLE, 1),
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
best = min(results.items(), key=lambda x: x[1]["intra_cv_mean"])
print(f"  Best IntraCV:     {best[0]} ({best[1]['intra_cv_mean']:.4f})")
best_pur = max(results.items(), key=lambda x: x[1]["boundary_purity"])
print(f"  Best Purity:      {best_pur[0]} ({best_pur[1]['boundary_purity']:.4f})")
best_cov = max(results.items(), key=lambda x: x[1]["coverage"])
print(f"  Best Coverage:    {best_cov[0]} ({best_cov[1]['coverage']:.4f})")
print(f"\n  → Use best version for re-tokenization")
```

**Chạy:** `python eval_seg_quality.py`

---

### T2.4.2 — Re-tokenize với best masks và retrain

```bash
# Sau khi biết version nào tốt nhất (từ eval_seg_quality.py):
BEST_MASKS="SECOND/sam2_masks_T1_test_refined"  # hoặc _spectral

# Re-tokenize test set với refined masks
python tokenize_regions_v2.py \
    --masks-T1 ${BEST_MASKS} \
    --out-dir SECOND/tokens_T1_test_v3

# Retrain với refined masks (train set cũng cần refine)
python spectral_mask_refine.py \
    --split train \
    --orig-mask-dir SECOND/sam2_masks_T1 \
    --out-dir SECOND/sam2_masks_T1_refined

python tokenize_regions_v2.py \
    --masks-T1 SECOND/sam2_masks_T1_refined \
    --out-dir SECOND/tokens_T1_v3

# Retrain
mkdir -p SECOND/stage9_refined_seg

nohup python train_reasoner_spectral.py \
    --tokens_T1 SECOND/tokens_T1_v3 \
    --tokens_T2 SECOND/tokens_T2_v3 \
    --output    SECOND/stage9_refined_seg \
    --epochs 60 --batch_size 8 \
    --use_spectral --gt_change_labels \
    --lambda_semantic 0.3 --lambda_transition 0.1 \
    --pretrain SECOND/stage6_combined/best_model.pt \
    --device cuda \
    > /tmp/train_refined_seg.log 2>&1 &
```

---

## ════════════════════════════════════
## BẢNG QUYẾT ĐỊNH CUỐI
## ════════════════════════════════════

```bash
python -c "
import json

files = {
    'stage6 (current best)':  'SECOND-OC/baseline_results/tokenmoe_desc_results.json',
    'T2.2 refined masks':     'SECOND-OC/baseline_results/refined_seg_results.json',
    'T2.3 spectral-guided':   'SECOND-OC/baseline_results/spectral_guided_results.json',
    'SCanNet (ref)':          'SECOND-OC/baseline_results/scannet_results.json',
}
print(f'{'Model':<25} {'Bin-F1':>8} {'Sem-F1':>8} {'Sem-Acc-TP':>12}')
print('-' * 58)
for name, path in files.items():
    try:
        with open(path) as f: d = json.load(f)
        print(f'{name:<25} {d.get(\"Binary-Object-F1\",0):>8.4f} '
              f'{d.get(\"Semantic-Object-F1\",0):>8.4f} '
              f'{d.get(\"Semantic-Acc-on-TP\",0):>12.1%}')
    except: print(f'{name:<25} {\"(chưa có)\":>30}')
"
```

---

## KHAI BÁO DỪNG

```
T2.1: Nếu CV mean < 0.10 AND Purity > 0.80
      → SAM2 masks đã tốt, SKIP T2.2 và T2.3
      → Vấn đề không phải segmentation quality

T2.2: Nếu sau refine, N_masks thay đổi > 50% (quá nhiều split/merge)
      → Điều chỉnh CV_SPLIT_THRESH và MERGE_DIST_THRESH
      → Thử lại với threshold conservative hơn

T2.3: Nếu spectral prompts < 20 points/image
      → grad_thresh quá thấp → tăng lên 0.20 hoặc 0.25

T2.4: Nếu Sem-F1 sau refined seg KHÔNG tăng so với stage6
      → Segmentation quality không phải bottleneck chính
      → Báo cáo: "thầy chỉ ra đúng hướng nhưng SAM2 quality
        đã đủ tốt, vấn đề nằm ở spectral features và supervision"
```

---

## Ý NGHĨA CHO PAPER

```
Nếu T2 cải thiện Sem-F1:
  → Confirm lời thầy: "segmentation quality ảnh hưởng classification"
  → Contribution mới: spectral-guided SAM2 prompting cho SCD

Nếu T2 KHÔNG cải thiện:
  → Insight quan trọng: "SAM2 boundary quality đã đủ tốt,
    bottleneck nằm ở feature completeness và training supervision"
  → Finding này cũng có giá trị cho paper
```