"""Backfill an `instance_label` channel into already-precomputed star-convex
target caches: a (2048,2048) int16 array, 0=background, 1..N=instance id,
needed by the discriminative embedding loss (code/discriminative_loss.py)
to know which pixels belong to the same instance. Not derivable from the
existing (probability, rays) cache the way canopy was -- those aggregate
all instances into one generic map with no per-instance identity left --
so this re-reads the raw GT polygons, same as add_boundary_weights.py, and
is similarly cheap (a single rasterize call per image, no ray-marching).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio.features
import shapely.wkb
from affine import Affine

MASK_SIZE = 2048


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.target_dir / "manifest.csv")
    instances_df = pd.read_parquet("benchmark/manifests/bam_instances.parquet")

    for i, image_id in enumerate(manifest["image_id"].astype(str)):
        npz_path = args.target_dir / f"{image_id.replace(':', '__')}.npz"
        if not npz_path.exists():
            continue
        existing = dict(np.load(npz_path))
        if "instance_label" in existing and not args.overwrite:
            print(f"[{i + 1}/{len(manifest)}] {image_id}: already has instance_label, skipped")
            continue

        image_instances = instances_df[instances_df["image_id"].astype(str) == image_id]
        shapes = []
        for label, row in enumerate(image_instances.itertuples(), start=1):
            geometry = shapely.wkb.loads(row.geometry_wkb)
            shapes.append((geometry, label))

        t0 = time.time()
        if shapes:
            label_map = rasterio.features.rasterize(
                shapes, out_shape=(MASK_SIZE, MASK_SIZE), transform=Affine.identity(), fill=0, dtype="int32"
            )
        else:
            label_map = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.int32)
        elapsed = time.time() - t0

        existing["instance_label"] = label_map.astype(np.int16)
        tmp_path = npz_path.with_name(npz_path.stem + ".tmp.npz")
        np.savez_compressed(tmp_path, **existing)
        tmp_path.replace(npz_path)
        print(f"[{i + 1}/{len(manifest)}] {image_id}: {len(shapes)} instances, {elapsed:.2f}s", flush=True)

    print(f"done, instance_label added under {args.target_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
