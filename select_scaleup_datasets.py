# select_scaleup_datasets.py
"""Select additional deadtrees.earth sites to scale up star-convex training
data beyond the current 5 sites (3889/3968/5650/5653/5737 -- all Temperate
Coniferous Forests / Temperate Grasslands, per METADATA.csv), which is why
research.md Addendum 4 found the current split has almost no biome
diversity. Extends the selection logic of `find_optimal_datasets.py`
(polygon-count ranking + "has all 20 tiles in the image zip" filter) with a
biome-stratified quota so the enlarged training set actually covers the
biomes DTE-aerial-bench evaluates against.

Quota rationale: proportional-ish to each biome's available site count,
with a floor for rare biomes (Boreal/Mangroves/Tundra/Desert/Montane/
Flooded) so they are not zero-represented, and an extra allowance for
Temperate Broadleaf and Mixed Forests since it is both the largest bench
biome (189/525 DTE-aerial-bench patches) and the one where the current
scratch checkpoint scored worst (F1=0.035, see research.md Addendum 4).
"""
import json
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from download_deadtrees import HTTPRangeFile

AERIAL_URL = "https://s3.bwsfs.uni-freiburg.de/frct-deadtrees-products/prepackaged/v2026-06-17/image-tiles-1024-global-aerial-sampled-20-random_2026.06.17.zip?response-content-disposition=attachment%3B%20filename%3D%22image-tiles-1024-global-aerial-sampled-20-random_2026.06.17.zip%22&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=NR872ZPSDS295Q3E5XH1%2F20260916%2Ffr1-ec82%2Fs3%2Faws4_request&X-Amz-Date=20260916T062515Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=808c0091d3299b1c1a639acc1a8c04e28e0168d88093bf4be04dd6631f65ede0"

EXISTING_SITES = [3889, 3968, 5650, 5653, 5737]
MIN_POLYGONS = 10
SEED = 42

BIOME_QUOTAS = {
    "Temperate Broadleaf and Mixed Forests": 25,
    "Tropical and Subtropical Moist Broadleaf Forests": 20,
    "Temperate Coniferous Forests": 15,
    "Tropical and Subtropical Grasslands, Savannas, and Shrublands": 20,
    "Mediterranean Forests, Woodlands, and Scrub": 20,
    "Temperate Grasslands, Savannas, and Shrublands": 15,
    "Tropical and Subtropical Dry Broadleaf Forests": 15,
    "Tropical and Subtropical Coniferous Forests": 15,
    "Boreal Forests/Taiga": 20,
    "Mangroves": 5,
    "Tundra": 5,
    "Deserts and Xeric Shrublands": 4,
    "Montane Grasslands and Shrublands": 3,
    "Flooded Grasslands and Savannas": 2,
}

CACHE_PATH = Path("DeadTrees/raw/image_tile_zip_counts_2026.06.17.json")


def get_zip_tile_counts() -> dict[int, int]:
    if CACHE_PATH.exists():
        print(f"Using cached zip tile counts: {CACHE_PATH}")
        return {int(k): v for k, v in json.loads(CACHE_PATH.read_text()).items()}

    print("Opening remote 287GB zip to scan central directory (listing only, no bulk download)...")
    f = HTTPRangeFile(AERIAL_URL)
    with zipfile.ZipFile(f) as z:
        namelist = z.namelist()
    zip_counts: dict[int, int] = {}
    for name in namelist:
        if not name.endswith(".tif"):
            continue
        parts = name.split("/")
        if len(parts) >= 3 and parts[0] == "tiles":
            try:
                dataset_id = int(parts[1])
            except ValueError:
                continue
            zip_counts[dataset_id] = zip_counts.get(dataset_id, 0) + 1
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(zip_counts))
    print(f"Cached zip tile counts for {len(zip_counts)} datasets -> {CACHE_PATH}")
    return zip_counts


def main():
    meta = pd.read_csv("DeadTrees/raw/standing-deadwood-aerial-global-conservative/METADATA.csv")
    gdf = gpd.read_file(
        "DeadTrees/raw/standing-deadwood-aerial-global-conservative/standing-deadwood-aerial-global-conservative_2026.06.17.gpkg",
        layer="standing_deadwood",
    )
    poly_counts = gdf.groupby("dataset_id").size().rename("n_polygons")

    zip_counts = get_zip_tile_counts()

    candidates = meta.set_index("dataset_id").join(poly_counts, how="inner")
    candidates["n_tiles_available"] = candidates.index.map(lambda d: zip_counts.get(d, 0))
    candidates = candidates[
        (candidates["n_tiles_available"] == 20)
        & (candidates["n_polygons"] >= MIN_POLYGONS)
        & (~candidates.index.isin(EXISTING_SITES))
    ]
    print(f"Eligible candidates (20/20 tiles, >={MIN_POLYGONS} polygons, not already used): {len(candidates)}")

    selected: list[int] = []
    selection_report = {}
    rng = pd.Series(range(len(candidates))).sample(frac=1, random_state=SEED).index  # deterministic shuffle order
    for biome, quota in BIOME_QUOTAS.items():
        pool = candidates[candidates["biome_name"] == biome]
        pool = pool.sample(n=min(quota, len(pool)), random_state=SEED)
        selection_report[biome] = {"quota": quota, "available": len(candidates[candidates["biome_name"] == biome]), "selected": len(pool)}
        selected.extend(int(d) for d in pool.index)

    print("\nSelection by biome:")
    for biome, info in selection_report.items():
        print(f"  {biome}: selected {info['selected']}/{info['quota']} (available {info['available']})")
    print(f"\nTotal new sites selected: {len(selected)}")
    print(f"Total sites after adding to existing 5: {len(selected) + len(EXISTING_SITES)}")
    print(f"Total new tiles to download: {len(selected) * 20}")

    output = {
        "seed": SEED,
        "min_polygons": MIN_POLYGONS,
        "existing_sites": EXISTING_SITES,
        "biome_quotas": BIOME_QUOTAS,
        "selection_report": selection_report,
        "selected_dataset_ids": sorted(selected),
    }
    out_path = Path("DeadTrees/raw/scaleup_selection_v1.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved selection -> {out_path}")


if __name__ == "__main__":
    main()
