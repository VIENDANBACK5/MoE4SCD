"""Backfill a `boundary_weight` channel into already-precomputed star-convex
target caches (code/precompute_star_targets.py), without recomputing the
expensive (probability, rays) targets.

boundary_weight_map only needs per-instance masks + a cheap per-instance
distance transform (see star_convex_targets.py docstring), unlike
object_probability_map/ray_distance_maps which needed the slow per-instance
ray-marching -- so this is a separate, much cheaper pass over the same
cached image set, run once to add the new channel to every existing .npz.
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
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.features
import shapely.wkb
from affine import Affine

from crown_segmentation_research.methods.star_convex.targets import boundary_weight_map

MASK_SIZE = 2048


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.target_dir / "manifest.csv")
    instances_df = pd.read_parquet("benchmark/manifests/bam_instances.parquet")

    for i, image_id in enumerate(manifest["image_id"].astype(str)):
        npz_path = args.target_dir / f"{image_id.replace(':', '__')}.npz"
        if not npz_path.exists():
            continue
        existing = dict(np.load(npz_path))
        if "boundary_weight" in existing and not args.overwrite:
            print(f"[{i + 1}/{len(manifest)}] {image_id}: already has boundary_weight, skipped")
            continue

        image_instances = instances_df[instances_df["image_id"].astype(str) == image_id]
        masks = []
        for row in image_instances.itertuples():
            geometry = shapely.wkb.loads(row.geometry_wkb)
            mask = rasterio.features.rasterize(
                [(geometry, 1)], out_shape=(MASK_SIZE, MASK_SIZE), transform=Affine.identity(), fill=0, dtype="uint8"
            ).astype(bool)
            masks.append(mask)

        t0 = time.time()
        weight = boundary_weight_map(masks, w0=args.w0, sigma=args.sigma) if masks else np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
        elapsed = time.time() - t0

        existing["boundary_weight"] = weight.astype(np.float16)
        # Write to a temp file + atomic rename, not directly to npz_path:
        # this script overwrites already-cached targets in place, and a
        # kill mid-write (this environment's sandbox has been observed to
        # kill background processes without warning) would otherwise
        # truncate the file and destroy the original probability/rays data
        # along with it, not just fail to add the new channel. Confirmed
        # this happened once (bam:train:719) before this fix.
        # Name must itself end in ".npz" -- np.savez_compressed silently
        # appends ".npz" to any path that doesn't already end with exactly
        # that suffix, so a ".npz.tmp" name would actually be written as
        # ".npz.tmp.npz" and the rename below would fail (confirmed
        # directly while regenerating bam:train:719).
        tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
        np.savez_compressed(tmp_path, **existing)
        tmp_path.replace(npz_path)
        print(f"[{i + 1}/{len(manifest)}] {image_id}: {len(masks)} instances, {elapsed:.2f}s", flush=True)

    print(f"done, boundary_weight added under {args.target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--w0", type=float, default=10.0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
