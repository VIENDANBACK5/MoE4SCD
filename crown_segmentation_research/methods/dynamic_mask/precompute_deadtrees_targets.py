"""Precompute (image, probability, rays, instance_label) star-convex targets
for DeadTrees tiles, matching the exact npz schema already used by
DeadTrees/star_convex_targets_v1 (see research.md Addendum 3/4). Unlike BAM's
precompute_star_targets.py (pre-clipped native-pixel WKB, Affine.identity()),
DeadTrees polygons come from instances_gt/instances.gpkg in EPSG:4326 and must
be reprojected to each tile's own CRS before rasterizing with that tile's
native affine transform (same approach gt_instances.py documents evaluators
should use).

Parallelized across images with multiprocessing, same rationale as
precompute_star_targets.py: target generation is CPU-bound and images are
independent.
"""
from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import argparse
import re
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from pyproj import Transformer
from shapely.ops import transform as transform_geometry

from crown_segmentation_research.methods.star_convex.targets import build_targets

DATASET_PATTERN = re.compile(r"dataset_(\d+)_")
_INSTANCES: gpd.GeoDataFrame | None = None
_N_RAYS: int = 16
_OUT_DIR: Path | None = None


def _init_worker(instances_path: str, n_rays: int, out_dir: str) -> None:
    global _INSTANCES, _N_RAYS, _OUT_DIR
    _INSTANCES = gpd.read_file(instances_path, layer="instances")
    _N_RAYS = n_rays
    _OUT_DIR = Path(out_dir)


def _process_one(image_path_str: str) -> tuple[str, int, float]:
    image_path = Path(image_path_str)
    image_id = image_path.stem
    out_path = _OUT_DIR / f"{image_id}.npz"
    if out_path.exists():
        return image_id, -1, 0.0

    t0 = time.time()
    with rasterio.open(image_path) as src:
        image_rgb = src.read([1, 2, 3]).transpose(1, 2, 0).astype(np.uint8)
        height, width = src.height, src.width
        tile_instances = _INSTANCES[_INSTANCES["stem"] == image_id]
        masks = []
        shapes = []
        if len(tile_instances) and src.crs is not None:
            to_native = Transformer.from_crs(_INSTANCES.crs, src.crs, always_xy=True)
            for label, row in enumerate(tile_instances.itertuples(), start=1):
                geometry_native = transform_geometry(to_native.transform, row.geometry)
                if geometry_native.is_empty:
                    continue
                mask = rasterio.features.rasterize(
                    [(geometry_native, 1)], out_shape=(height, width), transform=src.transform,
                    fill=0, dtype="uint8",
                ).astype(bool)
                if mask.any():
                    masks.append(mask)
                    shapes.append((geometry_native, label))

    if masks:
        probability, rays = build_targets(masks, n_rays=_N_RAYS)
        instance_label = rasterio.features.rasterize(
            shapes, out_shape=(height, width), transform=rasterio.open(image_path).transform,
            fill=0, dtype="int32",
        )
    else:
        probability = np.zeros((height, width), dtype=np.float32)
        rays = np.zeros((_N_RAYS, height, width), dtype=np.float32)
        instance_label = np.zeros((height, width), dtype=np.int32)
    elapsed = time.time() - t0

    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        image=image_rgb,
        probability=probability.astype(np.float16),
        rays=rays.astype(np.float16),
        instance_label=instance_label.astype(np.int16),
    )
    tmp_path.replace(out_path)
    return image_id, len(masks), elapsed


def run(args: argparse.Namespace) -> None:
    image_paths = sorted(args.image_root.glob("**/*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No GeoTIFF images found under {args.image_root}")

    def dataset_id_of(path: Path) -> int:
        match = DATASET_PATTERN.match(path.stem)
        if not match:
            raise ValueError(f"Cannot parse dataset_id from image stem: {path.stem}")
        return int(match.group(1))

    excluded = set(args.exclude_sites)
    image_paths = [p for p in image_paths if dataset_id_of(p) not in excluded]
    print(f"{len(image_paths)} images to process (excluded sites: {sorted(excluded)})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_items = [str(p) for p in image_paths]

    import multiprocessing

    with multiprocessing.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=(str(args.instances_gpkg), args.n_rays, str(args.out_dir)),
    ) as pool:
        completed = 0
        for image_id, n_instances, elapsed in pool.imap_unordered(_process_one, work_items):
            completed += 1
            if n_instances == -1:
                print(f"[{completed}/{len(work_items)}] {image_id}: already cached, skipped", flush=True)
            else:
                print(f"[{completed}/{len(work_items)}] {image_id}: {n_instances} instances, {elapsed:.2f}s", flush=True)

    new_rows = pd.DataFrame([{"image_id": Path(p).stem, "archive_member": Path(p).stem} for p in work_items])
    manifest_path = args.out_dir / "manifest.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path)
        combined = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates(subset="image_id")
    else:
        combined = new_rows
    combined.to_csv(manifest_path, index=False)
    print(f"done: {len(new_rows)} images processed this run, {len(combined)} total in manifest at {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path,
                         default=Path("DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"))
    parser.add_argument("--instances-gpkg", type=Path, default=Path("DeadTrees/instances_gt/instances.gpkg"))
    parser.add_argument("--out-dir", type=Path, default=Path("DeadTrees/star_convex_targets_v1/train"))
    parser.add_argument("--exclude-sites", type=int, nargs="*", default=[5737],
                         help="dataset_ids to skip (default: held-out val site 5737)")
    parser.add_argument("--n-rays", type=int, default=16)
    parser.add_argument("--n-workers", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
