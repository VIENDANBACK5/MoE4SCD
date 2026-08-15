# rasterize_polygons.py
"""
Convert deadwood polygons (vector) -> binary raster masks
to compare directly with SAM2 masks (which are also rasters).

Input:  DeadTrees/raw/standing-deadwood-aerial-global-conservative/*.gpkg
        + DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random/tiles/**/*.tif
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
DEADWOOD_GPKG = "DeadTrees/raw/standing-deadwood-aerial-global-conservative"
OUT_DIR      = "DeadTrees/masks_gt"
os.makedirs(OUT_DIR, exist_ok=True)

# Find .gpkg file
_gpkg_candidates = glob.glob(f"{DEADWOOD_GPKG}/**/*.gpkg", recursive=True)
assert len(_gpkg_candidates) > 0, f"Khong tim thay file .gpkg trong {DEADWOOD_GPKG}"
gpkg_path = _gpkg_candidates[0]

print(f"Loading standing_deadwood layer from: {gpkg_path}")
# Filter to target datasets to optimize speed and memory
deadwood_gdf = gpd.read_file(
    gpkg_path,
    layer="standing_deadwood",
    where="dataset_id IN (3889, 3968, 5737, 5653, 5650)"
)
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

    # Reproject polygons to image CRS if needed
    if deadwood_gdf.crs != img_crs:
        gdf_reproj = deadwood_gdf.to_crs(img_crs)
    else:
        gdf_reproj = deadwood_gdf

    # Filter polygons within image bounds (spatial join / filter)
    from shapely.geometry import box
    img_box = box(*img_bounds)
    relevant_polys = gdf_reproj[gdf_reproj.intersects(img_box)]

    if len(relevant_polys) == 0:
        mask = np.zeros((H, W), dtype=np.uint8)
        n_without += 1
    else:
        # Rasterize polygons to binary mask
        shapes = [(geom, 1) for geom in relevant_polys.geometry]
        mask = rasterio.features.rasterize(
            shapes,
            out_shape=(H, W),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        n_with_deadwood += 1

    # Save as 0/255 binary mask
    out_path = os.path.join(OUT_DIR, f"{stem}_deadwood.png")
    Image.fromarray(mask * 255).save(out_path)

print(f"\n✅ Rasterization done.")
print(f"   Images with deadwood:    {n_with_deadwood}")
print(f"   Images without deadwood: {n_without}")

# Validation
assert n_with_deadwood > 0, "[FAIL] Không có ảnh nào chứa deadwood — kiểm tra CRS/bounds"
print(f"   ✅ Validation passed")
