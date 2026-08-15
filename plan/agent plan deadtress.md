# AGENT PLAN: Deadtrees.earth Dataset Processing
> Package 1 (Aerial images) + Package 2 (Standing deadwood polygons)
> Mục đích: Validate spectral-guided segmentation trên dataset thứ 2, cân bằng hơn SECOND
> Chạy theo thứ tự: D1 → D2 → D3 → D4 → D5 → eval

---

## BỨC TRANH TỔNG THỂ

```
deadtrees.earth data:
  Aerial images (RGB, 1024×1024)     ← Package 1
  Deadwood polygons (GeoPackage)     ← Package 2
       ↓
  D1: Download + explore format
       ↓
  D2: Rasterize polygons → binary GT mask
       ↓
  D3: Chạy SAM2 → instance masks
       ↓
  D4: Đánh giá segmentation (OS/US/ED) + spectral features
       ↓
  D5: Classification (deadwood vs other) — shape-only vs shape+spectral
```

---

## FILE MAP

```
Image Segmentation/
├── download_deadtrees.py         [NEW] D1: download packages
├── explore_deadtrees_format.py   [NEW] D1: hiểu structure
├── rasterize_polygons.py         [NEW] D2: polygon → mask
├── run_sam2_deadtrees.py         [NEW] D3: SAM2 trên aerial images
├── eval_seg_deadtrees.py         [NEW] D4: OS/US/ED + spectral
├── classify_deadwood.py          [NEW] D5: classification ablation
└── DeadTrees/                    [NEW] thư mục data
    ├── raw/
    │   ├── aerial_images/
    │   ├── tree_cover.gpkg
    │   └── deadwood.gpkg
    ├── masks_gt/
    ├── sam2_masks/
    └── results/
```

---

## ════════════════════════════════════
## D1 — Download và Khám phá Format
## ════════════════════════════════════

### D1.0 — Prerequisite

```bash
pip install geopandas rasterio shapely fiona --break-system-packages
python -c "
import geopandas, rasterio
print('✅ geopandas:', geopandas.__version__)
print('✅ rasterio:', rasterio.__version__)
"
```

---

### D1.1 — Tạo `download_deadtrees.py`

```python
# download_deadtrees.py
"""
Download 3 packages tu deadtrees.earth (presigned S3 URLs, TTL 7 ngay
tu 2026-07-14 -- het han khoang 2026-07-21, download NGAY khi chay).

Ca 3 package deu la .zip, KHONG phai .gpkg/.tif truc tiep.
Sau khi download can giai nen roi moi tim file .gpkg/.tif ben trong
(cau truc noi bo chua biet -- se explore o buoc D1.2).
"""
import os
import zipfile
import requests
from tqdm import tqdm

OUT_DIR = "DeadTrees/raw"
os.makedirs(OUT_DIR, exist_ok=True)

# URL that, presigned S3, het han ~2026-07-21 (7 ngay tu luc tao)
DOWNLOADS = {
    "standing-deadwood-aerial-global-conservative.zip":
        "https://s3.bwsfs.uni-freiburg.de/frct-deadtrees-products/prepackaged/v2026-06-17/standing-deadwood-aerial-global-conservative_2026.06.17.zip?response-content-disposition=attachment%3B%20filename%3D%22standing-deadwood-aerial-global-conservative_2026.06.17.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NR872ZPSDS295Q3E5XH1%2F20260714%2Ffr1-ec82%2Fs3%2Faws4_request&X-Amz-Date=20260714T145518Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=0cfc0ccf0628e7f8eb33729a90eae757eeac149f57ecce7583dd43109f816f26",

    "image-tiles-1024-global-aerial-sampled-20-random.zip":
        "https://s3.bwsfs.uni-freiburg.de/frct-deadtrees-products/prepackaged/v2026-06-17/image-tiles-1024-global-aerial-sampled-20-random_2026.06.17.zip?response-content-disposition=attachment%3B%20filename%3D%22image-tiles-1024-global-aerial-sampled-20-random_2026.06.17.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NR872ZPSDS295Q3E5XH1%2F20260714%2Ffr1-ec82%2Fs3%2Faws4_request&X-Amz-Date=20260714T145545Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=327412e71c409484c48662ce609a50349bc8a3d071ca38c181bdd999ede89890",

    "tree-cover-aerial-global.zip":
        "https://s3.bwsfs.uni-freiburg.de/frct-deadtrees-products/prepackaged/v2026-06-17/tree-cover-aerial-global_2026.06.17.zip?response-content-disposition=attachment%3B%20filename%3D%22tree-cover-aerial-global_2026.06.17.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NR872ZPSDS295Q3E5XH1%2F20260714%2Ffr1-ec82%2Fs3%2Faws4_request&X-Amz-Date=20260714T145555Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=6951db7cb6f4dac073b61ac53dcd2e41c7b8af61263c37067ecb35b624e8749a",
}


def download_file(url, out_path):
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return
    response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))

    with open(out_path, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True, desc=os.path.basename(out_path)
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(extract_to)
    print(f"  Extracted -> {extract_to}")


def main():
    for filename, url in DOWNLOADS.items():
        out_path = os.path.join(OUT_DIR, filename)
        print(f"Downloading {filename}...")
        download_file(url, out_path)

        extract_dir = os.path.join(OUT_DIR, filename.replace(".zip", ""))
        os.makedirs(extract_dir, exist_ok=True)
        extract_zip(out_path, extract_dir)

    print("\nDone. Chay explore_deadtrees_format.py de xem cau truc thuc te ben trong.")


if __name__ == "__main__":
    main()
```

**Chay:** `python download_deadtrees.py`

**Neu gap loi 403/Access Denied:** URL da het han (qua 7 ngay ke tu 2026-07-14) -- can lay link moi bang cach lap lai buoc chuot phai Copy Link Address tren trang deadtrees.earth.

**Sau khi download xong, giải nén:**
```bash
cd DeadTrees/raw
# Da tu dong giai nen trong download_deadtrees.py, kiem tra lai:
ls DeadTrees/raw/*/  | head -20
ls -la *.gpkg
```

---

### D1.2 — Tạo `explore_deadtrees_format.py`

```python
# explore_deadtrees_format.py
"""
Kham pha cau truc data sau khi giai nen 3 packages.
KHONG doan format -- luon verify truoc khi viet code xu ly tiep.
"""
import glob, os

print("=" * 60)
print("LIET KE TOAN BO FILE TRONG 3 THU MUC DA GIAI NEN")
print("=" * 60)

folders = {
    "deadwood":   "DeadTrees/raw/standing-deadwood-aerial-global-conservative",
    "treecover":  "DeadTrees/raw/tree-cover-aerial-global",
    "aerial":     "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random",
}

found_files = {}
for name, folder in folders.items():
    all_files = glob.glob(f"{folder}/**/*", recursive=True)
    all_files = [f for f in all_files if os.path.isfile(f)]
    found_files[name] = all_files
    print(f"\n[{name}] {len(all_files)} files")
    # In ra các đuôi file khác nhau để biết định dạng thật
    exts = set(os.path.splitext(f)[1] for f in all_files)
    print(f"  Extensions: {exts}")
    for f in all_files[:5]:
        print(f"  - {f}")

print("\n" + "=" * 60)
print("2. KIEM TRA FILE VECTOR (GEOPACKAGE/SHAPEFILE/GEOJSON)")
print("=" * 60)
import geopandas as gpd

for name in ["deadwood", "treecover"]:
    vector_files = [f for f in found_files[name]
                    if f.endswith((".gpkg", ".shp", ".geojson"))]
    for vf in vector_files:
        print(f"\n[{name}] {vf}")
        gdf = gpd.read_file(vf)
        print(f"  Rows: {len(gdf)}")
        print(f"  CRS: {gdf.crs}")
        print(f"  Columns: {list(gdf.columns)}")
        id_cols = [c for c in gdf.columns if any(
            k in c.lower() for k in ['id', 'image', 'file', 'ortho', 'source'])]
        print(f"  Cot co the lien ket voi anh: {id_cols}")
        if len(gdf) > 0:
            print(f"  Sample row:\n{gdf.iloc[0]}")

print("\n" + "=" * 60)
print("3. KIEM TRA FILE ANH (TIF/PNG)")
print("=" * 60)
import rasterio

img_files = [f for f in found_files["aerial"]
             if f.endswith((".tif", ".tiff", ".png", ".jpg"))]
if img_files:
    sample = img_files[0]
    print(f"Sample: {sample}")
    if sample.endswith((".tif", ".tiff")):
        with rasterio.open(sample) as src:
            print(f"  CRS: {src.crs}")
            print(f"  Size: {src.width} x {src.height}")
            print(f"  Bands: {src.count}, Dtype: {src.dtypes}")
            print(f"  Bounds: {src.bounds}")
    else:
        from PIL import Image
        img = Image.open(sample)
        print(f"  Size: {img.size}, Mode: {img.mode}")
        print(f"  [WARNING] .png/.jpg khong co geospatial metadata")
        print(f"  Can tim file world file (.tfw/.wld) hoac metadata rieng")

print("\n[AGENT DECISION]")
print("Xac dinh: moi polygon lien ket voi anh nao qua cot gi?")
print("CRS cua anh va polygon co khop nhau khong?")
print("Anh co geospatial metadata (CRS) hay chi la RGB thuan?")
```

**Chay:** `python explore_deadtrees_format.py`

**Agent doc output va xac dinh 3 dieu truoc khi sang D2:**
1. File vector là `.gpkg` hay format khác (`.shp`, `.geojson`)?
2. Có cột nào trong polygon link tới ảnh cụ thể không (ví dụ `image_id`, `dataset_id`)?
3. Ảnh trong package "aerial" là `.tif` (có CRS) hay `.png`/`.jpg` (không có geospatial metadata)?
   - Nếu là `.png`/`.jpg` không CRS → cần tìm cách khác để link polygon với ảnh
     (có thể qua tên file, hoặc file metadata `.json`/`.csv` đi kèm)

---

## ════════════════════════════════════
## D2 — Rasterize Polygons thành GT Masks
## ════════════════════════════════════

### D2.1 — Tạo `rasterize_polygons.py`

```python
# rasterize_polygons.py
"""
Convert deadwood polygons (vector) → binary raster masks
để so sánh trực tiếp với SAM2 masks (cũng là raster).

Input:  DeadTrees/raw/standing-deadwood-aerial-global-conservative/*.gpkg
        + DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random/*.tif
Output: DeadTrees/masks_gt/{image_stem}_deadwood.png (binary mask)
"""
import os, glob
import geopandas as gpd
import rasterio
import rasterio.features
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

IMG_DIR      = "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
DEADWOOD_GPKG = "DeadTrees/raw/standing-deadwood-aerial-global-conservative"  # agent: tim file .gpkg cu the trong nay sau D1.2
OUT_DIR      = "DeadTrees/masks_gt"
os.makedirs(OUT_DIR, exist_ok=True)

# Load polygons 1 lần
import glob as _glob
_gpkg_candidates = _glob.glob(f"{DEADWOOD_GPKG}/**/*.gpkg", recursive=True)
assert len(_gpkg_candidates) > 0, f"Khong tim thay file .gpkg trong {DEADWOOD_GPKG} -- chay explore_deadtrees_format.py truoc"
deadwood_gdf = gpd.read_file(_gpkg_candidates[0])
print(f"Loaded gpkg: {_gpkg_candidates[0]}")
print(f"Loaded {len(deadwood_gdf)} deadwood polygons, CRS={deadwood_gdf.crs}")

img_files = glob.glob(f"{IMG_DIR}/**/*.tif", recursive=True)
print(f"Processing {len(img_files)} images...")

n_with_deadwood = 0
n_without = 0

for img_path in tqdm(img_files):
    stem = Path(img_path).stem

    with rasterio.open(img_path) as src:
        img_crs = src.crs
        transform = src.transform
        H, W = src.height, src.width
        img_bounds = src.bounds

    # Reproject polygons về CRS của ảnh nếu cần
    if deadwood_gdf.crs != img_crs:
        gdf_reproj = deadwood_gdf.to_crs(img_crs)
    else:
        gdf_reproj = deadwood_gdf

    # Lọc polygons nằm trong bounds của ảnh này (spatial filter)
    from shapely.geometry import box
    img_box = box(*img_bounds)
    relevant_polys = gdf_reproj[gdf_reproj.intersects(img_box)]

    if len(relevant_polys) == 0:
        # Ảnh này không có deadwood → tạo mask toàn 0
        mask = np.zeros((H, W), dtype=np.uint8)
        n_without += 1
    else:
        # Rasterize polygons thành binary mask
        shapes = [(geom, 1) for geom in relevant_polys.geometry]
        mask = rasterio.features.rasterize(
            shapes,
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        n_with_deadwood += 1

    # Save
    out_path = os.path.join(OUT_DIR, f"{stem}_deadwood.png")
    Image.fromarray(mask * 255).save(out_path)  # 0/255 for visualization

print(f"\n✅ Rasterization done.")
print(f"   Images with deadwood:    {n_with_deadwood}")
print(f"   Images without deadwood: {n_without}")

# Validation
assert n_with_deadwood > 0, "[FAIL] Không có ảnh nào chứa deadwood — kiểm tra CRS/bounds"
print(f"   ✅ Validation passed")
```

**Chạy:** `python rasterize_polygons.py`

**Nếu `n_with_deadwood = 0`:** Có vấn đề CRS mismatch hoặc spatial join sai. Agent kiểm tra:
```bash
python -c "
import geopandas as gpd
import glob as _glob
gpkg_path = _glob.glob('DeadTrees/raw/standing-deadwood-aerial-global-conservative/**/*.gpkg', recursive=True)[0]
gdf = gpd.read_file(gpkg_path)
print('Deadwood CRS:', gdf.crs)
print('Deadwood bounds:', gdf.total_bounds)

import rasterio
import glob
sample = glob.glob('DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random/**/*.tif', recursive=True)[0]
with rasterio.open(sample) as src:
    print('Image CRS:', src.crs)
    print('Image bounds:', src.bounds)
"
```
So sánh bounds — nếu hoàn toàn không overlap, có thể ảnh và polygon từ 2 khu vực khác nhau (cần lọc lại images chỉ trong vùng có deadwood annotation).

---

## ════════════════════════════════════
## D3 — Chạy SAM2 trên Aerial Images
## ════════════════════════════════════

### D3.1 — Tạo `run_sam2_deadtrees.py`

```python
# run_sam2_deadtrees.py
"""
Chạy SAM2 automatic mask generation trên aerial images.
Dùng lại logic từ generate_sam2_masks_test.py (đã có trong project).
"""
import os, glob
import numpy as np
import torch
import rasterio
from pathlib import Path
from tqdm import tqdm

IMG_DIR  = "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
OUT_DIR  = "DeadTrees/sam2_masks"
os.makedirs(OUT_DIR, exist_ok=True)

MIN_AREA_PX = 100

def main():
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_model = build_sam2(
        "sam2/configs/sam2_hiera_l.yaml",
        "sam2/checkpoints/sam2_hiera_large.pt",
        device=device
    )
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        min_mask_region_area=MIN_AREA_PX,
    )
    print(f"✅ SAM2 loaded on {device}")

    img_files = glob.glob(f"{IMG_DIR}/**/*.tif", recursive=True)
    print(f"Processing {len(img_files)} images...")

    for img_path in tqdm(img_files):
        stem = Path(img_path).stem
        out_path = os.path.join(OUT_DIR, f"{stem}.npz")

        if os.path.exists(out_path):
            continue

        with rasterio.open(img_path) as src:
            # Đọc 3 bands đầu (RGB), một số ảnh drone có thể có 4 bands (RGBA/NIR)
            image = src.read([1, 2, 3])  # (3, H, W)
            image = np.transpose(image, (1, 2, 0))  # (H, W, 3)

            if image.dtype != np.uint8:
                # Normalize về uint8 nếu cần
                image = ((image - image.min()) /
                        (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

        try:
            masks_data = mask_generator.generate(image)
        except Exception as e:
            print(f"[ERROR] {stem}: {e}")
            continue

        if not masks_data:
            continue

        masks  = np.stack([m["segmentation"] for m in masks_data])
        scores = np.array([m["stability_score"] for m in masks_data])
        areas  = np.array([m["area"] for m in masks_data])

        np.savez_compressed(out_path, masks=masks, scores=scores, areas=areas)

    print(f"\n✅ SAM2 mask generation done.")

    # Validation
    n_generated = len(glob.glob(f"{OUT_DIR}/*.npz"))
    print(f"   Generated masks for {n_generated}/{len(img_files)} images")
    assert n_generated > len(img_files) * 0.8, \
        "[FAIL] Quá nhiều ảnh không generate được mask"
    print(f"   ✅ Validation passed")


if __name__ == "__main__":
    main()
```

**Chạy:** `python run_sam2_deadtrees.py`

---

## ════════════════════════════════════
## D4 — Đánh giá Segmentation Quality + Spectral
## ════════════════════════════════════

### D4.1 — Tạo `eval_seg_deadtrees.py`

```python
# eval_seg_deadtrees.py
"""
Đánh giá SAM2 masks vs GT deadwood polygons bằng OS/US/ED
(Dao et al. 2021 metrics — ĐÚNG metric cho segmentation task,
KHÔNG dùng Binary-Object-F1 vì đó là detection metric).

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


def compute_spectral_features(image, mask):
    """24 spectral features — dùng lại logic từ spectral_extractor.py"""
    pixels = image[mask.astype(bool)].astype(np.float32)
    if len(pixels) == 0:
        return np.zeros(6)
    mean = pixels.mean(axis=0)
    std  = pixels.std(axis=0)
    return np.concatenate([mean, std])  # 6 dims (simplified for single-time)


# --- Main loop ---
gt_files = glob.glob(f"{GT_DIR}/*_deadwood.png")
all_os, all_us, all_ed = [], [], []
per_image_results = []

for gt_path in tqdm(gt_files, desc="Evaluating segmentation"):
    stem = Path(gt_path).stem.replace("_deadwood", "")

    img_path  = None
    for ext in [".tif", ".png"]:
        candidate = f"{IMG_DIR}/{stem}{ext}"
        if os.path.exists(candidate):
            img_path = candidate
            break
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
        if comp_mask.sum() < 20:
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
    "[FAIL] Quá ít GT objects được evaluate — kiểm tra lại D2"
print(f"\n✅ Validation passed. Compare với SECOND OS/US/ED để reference.")
```

**Chạy:** `python eval_seg_deadtrees.py`

**Expected:**
```
{
  "n_images_evaluated": ...,
  "n_gt_objects_total": ...,
  "OS_mean": 0.3-0.6,   # tham khảo Dao et al. 2021: 0.5-0.8 tùy scale
  "US_mean": 0.1-0.3,
  "ED_mean": 0.3-0.5
}
```

---

## ════════════════════════════════════
## D5 — Classification Ablation: Shape-only vs Shape+Spectral
## ════════════════════════════════════

### D5.1 — Tạo `classify_deadwood.py`

```python
# classify_deadwood.py
"""
Bài toán đơn giản: deadwood vs other (binary classification)
So sánh: shape-only features vs shape+spectral features

Không cần MoE/Transformer phức tạp — dùng Random Forest đơn giản
để validate finding CORE: spectral có giúp classification không.
"""
import os, glob
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

IMG_DIR  = "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
GT_DIR   = "DeadTrees/masks_gt"
SAM2_DIR = "DeadTrees/sam2_masks"


def extract_shape_features(mask):
    """Shape-only: area, perimeter, compactness, bbox ratio."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros(4)
    area = mask.sum()
    bbox_h = ys.max() - ys.min() + 1
    bbox_w = xs.max() - xs.min() + 1
    bbox_ratio = bbox_h / max(bbox_w, 1)

    # Perimeter approx qua boundary
    boundary = mask ^ ndimage.binary_erosion(mask)
    perimeter = boundary.sum()
    compactness = (4 * np.pi * area) / max(perimeter**2, 1)

    return np.array([np.log(area + 1), bbox_ratio, compactness, perimeter])


def extract_spectral_features(image, mask):
    """Spectral: mean/std RGB."""
    pixels = image[mask.astype(bool)].astype(np.float32)
    if len(pixels) == 0:
        return np.zeros(6)
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0)])


# --- Build dataset ---
X_shape, X_spectral, y = [], [], []

gt_files = glob.glob(f"{GT_DIR}/*_deadwood.png")
for gt_path in tqdm(gt_files, desc="Building classification dataset"):
    stem = Path(gt_path).stem.replace("_deadwood", "")

    img_path = None
    for ext in [".tif", ".png"]:
        c = f"{IMG_DIR}/{stem}{ext}"
        if os.path.exists(c):
            img_path = c
            break
    if img_path is None:
        continue

    sam2_path = f"{SAM2_DIR}/{stem}.npz"
    if not os.path.exists(sam2_path):
        continue

    import rasterio
    with rasterio.open(img_path) as src:
        image = np.transpose(src.read([1,2,3]), (1,2,0))

    gt_mask    = np.array(Image.open(gt_path)) > 127
    sam2_masks = np.load(sam2_path)["masks"]

    for sam_mask in sam2_masks:
        sam_mask = sam_mask.astype(bool)
        if sam_mask.sum() < 50:
            continue

        # Label: overlap > 50% với GT deadwood → positive
        overlap = (sam_mask & gt_mask).sum()
        label = 1 if overlap / sam_mask.sum() > 0.5 else 0

        X_shape.append(extract_shape_features(sam_mask))
        X_spectral.append(extract_spectral_features(image, sam_mask))
        y.append(label)

X_shape    = np.array(X_shape)
X_spectral = np.array(X_spectral)
y          = np.array(y)

print(f"\nDataset: {len(y)} samples, {y.sum()} positive (deadwood), "
      f"{len(y)-y.sum()} negative")

# --- Train/test split ---
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
)

def evaluate_features(X, name):
    clf = RandomForestClassifier(n_estimators=100, random_state=42,
                                 class_weight="balanced")
    clf.fit(X[idx_train], y[idx_train])
    pred = clf.predict(X[idx_test])

    acc = accuracy_score(y[idx_test], pred)
    f1  = f1_score(y[idx_test], pred)

    print(f"\n{'='*50}")
    print(f"Features: {name}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1:       {f1:.4f}")
    print(classification_report(y[idx_test], pred,
                                target_names=["other", "deadwood"]))
    return {"accuracy": acc, "f1": f1}


# --- Ablation: shape-only vs shape+spectral ---
result_shape = evaluate_features(X_shape, "Shape-only (baseline)")

X_combined = np.concatenate([X_shape, X_spectral], axis=1)
result_combined = evaluate_features(X_combined, "Shape + Spectral")

# --- Summary ---
print(f"\n{'='*50}")
print("ABLATION SUMMARY")
print(f"{'='*50}")
print(f"Shape-only:       Acc={result_shape['accuracy']:.4f}, "
      f"F1={result_shape['f1']:.4f}")
print(f"Shape + Spectral: Acc={result_combined['accuracy']:.4f}, "
      f"F1={result_combined['f1']:.4f}")

delta_acc = result_combined['accuracy'] - result_shape['accuracy']
delta_f1  = result_combined['f1'] - result_shape['f1']
print(f"\nImprovement: +{delta_acc:.4f} accuracy, +{delta_f1:.4f} F1")

import json
with open("DeadTrees/results/classification_ablation.json", "w") as f:
    json.dump({
        "shape_only": result_shape,
        "shape_spectral": result_combined,
        "delta_accuracy": delta_acc,
        "delta_f1": delta_f1,
    }, f, indent=2)

# Validation
assert delta_f1 > 0, \
    "[UNEXPECTED] Spectral features không cải thiện — cần điều tra thêm"
print(f"\n✅ Validation: spectral features improve classification "
      f"(+{delta_f1:.1%} F1)")
```

**Chạy:** `python classify_deadwood.py`

**Expected output:**
```
Shape-only:       Acc=0.75-0.85, F1=0.60-0.75
Shape + Spectral: Acc=0.85-0.92, F1=0.75-0.88

Improvement: +0.05-0.15 accuracy, +0.10-0.20 F1
```

---

## BẢNG KẾT QUẢ CUỐI CÙNG

```bash
python -c "
import json

print('=== SEGMENTATION QUALITY (Deadtrees) ===')
with open('DeadTrees/results/seg_quality_deadtrees.json') as f:
    d = json.load(f)['summary']
print(f'OS: {d[\"OS_mean\"]:.4f}, US: {d[\"US_mean\"]:.4f}, ED: {d[\"ED_mean\"]:.4f}')

print()
print('=== CLASSIFICATION ABLATION (Deadtrees) ===')
with open('DeadTrees/results/classification_ablation.json') as f:
    d = json.load(f)
print(f'Shape-only F1:       {d[\"shape_only\"][\"f1\"]:.4f}')
print(f'Shape+Spectral F1:   {d[\"shape_spectral\"][\"f1\"]:.4f}')
print(f'Improvement:         +{d[\"delta_f1\"]:.4f}')
"
```

---

## Ý NGHĨA CHO PAPER

```
Nếu delta_f1 > 0 (kỳ vọng):
  → Finding từ SECOND (spectral giúp Sem-F1 +87.7%)
    ĐƯỢC XÁC NHẬN trên dataset thứ 2, khác domain
    (satellite vs drone), khác task (SCD vs single-time classification)
  → Tăng tính tổng quát hóa của claim chính

Nếu delta_f1 ≈ 0 hoặc âm:
  → Cần điều tra: có thể do class imbalance trong sample,
    hoặc deadwood có shape đặc trưng đủ để phân biệt
    (khác với low_veg vs barren trong SECOND)
  → Vẫn là finding có giá trị, cần phân tích thêm
```

---

## KHAI BÁO DỪNG

```
D1: Nếu không tìm được URL download thật → dừng, hỏi user link cụ thể
D2: Nếu n_with_deadwood = 0 → kiểm tra CRS mismatch
D3: Nếu SAM2 fail > 20% ảnh → kiểm tra format ảnh (RGBA vs RGB, bit depth)
D4: Nếu n_gt_objects_total < 10 → data quá ít, cần thêm ảnh hoặc check rasterize
D5: Nếu delta_f1 âm đáng kể → không phải lỗi, là finding cần phân tích thêm
```