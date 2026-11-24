"""
Pre-cache Quebec GeoTIFF scenes into fast local .npz chips.

Extracts random 512x512 patches with instance labels from Z1, Z2, Z3.
Each .npz file contains:
  - 'image': uint8 (512, 512, 3)
  - 'instance_label': int32 (512, 512)
"""

from __future__ import annotations

import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import box
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
QUEBEC_DIR = REPO / "data/itc_benchmarks/extracted/quebec_2021_09_02"
CACHE_DIR  = REPO / "data/itc_benchmarks/extracted/quebec_cached_npz"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

PATCH_SIZE = 512
PATCHES_PER_ZONE = 200
ZONES = ["Z1", "Z2", "Z3"]


def cache_zone(zone: str, count: int = PATCHES_PER_ZONE):
    tif_path  = QUEBEC_DIR / f"2021-09-02-sbl-{zone.lower()}-rgb-cog.tif"
    gpkg_path = QUEBEC_DIR / f"{zone}_polygons.gpkg"

    print(f"\nProcessing Zone {zone}:")
    print(f"  Raster: {tif_path.name}")
    print(f"  Vector: {gpkg_path.name}")

    gdf = gpd.read_file(str(gpkg_path))
    with rasterio.open(tif_path) as src:
        H, W = src.height, src.width
        crs  = src.crs
        transform = src.transform

    if gdf.crs != crs:
        print(f"  Reprojecting vector to {crs}...")
        gdf = gdf.to_crs(crs)

    # Build spatial index
    sindex = gdf.sindex
    print(f"  Total crowns in zone: {len(gdf):,}")

    saved = 0
    attempts = 0
    max_attempts = count * 5

    with rasterio.open(tif_path) as src:
        while saved < count and attempts < max_attempts:
            attempts += 1
            r0 = random.randint(0, max(0, H - PATCH_SIZE))
            c0 = random.randint(0, max(0, W - PATCH_SIZE))

            win = Window(c0, r0, PATCH_SIZE, PATCH_SIZE)
            img = src.read([1, 2, 3], window=win)  # [3, H, W]
            img = img.transpose(1, 2, 0)            # [H, W, 3]

            # Skip pure black/nodata patches
            if (img.sum(axis=-1) > 20).mean() < 0.5:
                continue

            if img.dtype != np.uint8:
                img = np.clip(img / (img.max() + 1e-6) * 255.0, 0, 255).astype(np.uint8)

            w_transform = src.window_transform(win)
            ul = w_transform * (0, 0)
            lr = w_transform * (PATCH_SIZE, PATCH_SIZE)
            patch_geom = box(min(ul[0], lr[0]), min(ul[1], lr[1]),
                             max(ul[0], lr[0]), max(ul[1], lr[1]))

            # Fast spatial index query
            possible_matches_idx = list(sindex.intersection(patch_geom.bounds))
            if not possible_matches_idx:
                continue

            possible_matches = gdf.iloc[possible_matches_idx]
            exact_matches = possible_matches[possible_matches.intersects(patch_geom)]

            if len(exact_matches) < 2:
                continue

            # Rasterize each crown with unique integer ID
            label_map = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.int32)
            cur_id = 1
            for _, row in exact_matches.iterrows():
                geom = row.geometry.intersection(patch_geom)
                if geom.is_empty:
                    continue
                m = rasterize([(geom, 1)], out_shape=(PATCH_SIZE, PATCH_SIZE),
                              transform=w_transform, fill=0, dtype=np.uint8)
                if m.sum() >= 10:
                    label_map[m == 1] = cur_id
                    cur_id += 1

            if cur_id < 3:
                continue

            out_file = CACHE_DIR / f"quebec_{zone.lower()}_patch_{saved:04d}.npz"
            np.savez_compressed(out_file, image=img, instance_label=label_map)
            saved += 1

    print(f"  Successfully cached {saved} patches for {zone}.")


def main():
    print(f"Pre-caching Quebec patches to {CACHE_DIR}...")
    random.seed(42)
    np.random.seed(42)
    for z in ZONES:
        cache_zone(z, count=PATCHES_PER_ZONE)
    total_cached = len(list(CACHE_DIR.glob("*.npz")))
    print(f"\nCompleted! Total cached Quebec patches: {total_cached}")


if __name__ == "__main__":
    main()
