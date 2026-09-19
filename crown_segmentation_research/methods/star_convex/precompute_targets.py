"""Precompute (probability, rays) targets to disk for a BAM subset, in parallel.

target generation is slow (~1-2s/instance) and CPU-bound (no GPU used), so
this parallelizes across images with multiprocessing -- each worker opens
its own handle to the archive/manifests (zipfile handles are not safely
shared across a fork). Measured ~50s/image serially; with N workers this
should scale close to linearly since images are independent.
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
import io
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
import rasterio.features
import shapely.wkb
from affine import Affine

from crown_segmentation_research.methods.star_convex.targets import build_targets

_INSTANCES_DF: pd.DataFrame | None = None
_ARCHIVE_PATH: Path | None = None
_N_RAYS: int = 16
_OUT_DIR: Path | None = None


def _init_worker(instances_path: str, archive_path: str, n_rays: int, out_dir: str) -> None:
    global _INSTANCES_DF, _ARCHIVE_PATH, _N_RAYS, _OUT_DIR
    _INSTANCES_DF = pd.read_parquet(instances_path)
    _ARCHIVE_PATH = Path(archive_path)
    _N_RAYS = n_rays
    _OUT_DIR = Path(out_dir)


def _process_one(args: tuple[str, str]) -> tuple[str, int, float]:
    image_id, archive_member = args
    out_path = _OUT_DIR / f"{image_id.replace(':', '__')}.npz"
    if out_path.exists():
        return image_id, -1, 0.0

    with zipfile.ZipFile(_ARCHIVE_PATH, "r") as archive:
        buffer = archive.read(archive_member)
    image_rgb = np.array(PIL.Image.open(io.BytesIO(buffer)))[:, :, :3]
    image_instances = _INSTANCES_DF[_INSTANCES_DF["image_id"].astype(str) == image_id]
    masks = []
    for instance_row in image_instances.itertuples():
        geometry = shapely.wkb.loads(instance_row.geometry_wkb)
        mask = rasterio.features.rasterize(
            [(geometry, 1)], out_shape=(2048, 2048), transform=Affine.identity(), fill=0, dtype="uint8"
        ).astype(bool)
        masks.append(mask)

    t0 = time.time()
    if masks:
        probability, rays = build_targets(masks, n_rays=_N_RAYS)
    else:
        # build_targets([]) returns (0, 0)-shaped arrays (no image to infer
        # shape from), which crashes the training DataLoader when batched
        # against 2048x2048 images from other samples. An image with zero
        # GT instances (e.g. bam:train:1419) is a legitimate all-background
        # training example, not an error -- give it full-size zero targets.
        probability = np.zeros(image_rgb.shape[:2], dtype=np.float32)
        rays = np.zeros((_N_RAYS, *image_rgb.shape[:2]), dtype=np.float32)
    elapsed = time.time() - t0
    # Write to a temp path + atomic rename: a kill mid-write would otherwise
    # leave a truncated file at out_path that the exists()-check above would
    # treat as "already cached" on the next run, silently hiding the
    # corruption instead of regenerating it (see add_boundary_weights.py's
    # docstring for a concrete instance of this failure mode elsewhere).
    # Name must itself end in ".npz" -- np.savez_compressed silently
    # appends ".npz" to any path that doesn't already end with exactly
    # that suffix, so a ".npz.tmp" name actually gets written as
    # ".npz.tmp.npz" and the rename below would fail (confirmed directly).
    tmp_path = out_path.with_name(out_path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        image=image_rgb,
        probability=probability.astype(np.float16),
        rays=rays.astype(np.float16),
    )
    tmp_path.replace(out_path)
    return image_id, len(masks), elapsed


def run(args: argparse.Namespace) -> None:
    images_df = pd.read_csv("benchmark/manifests/bam_images.csv")
    images_df["image_id"] = images_df["image_id"].astype(str)
    split_df = images_df[images_df["split"] == args.split].reset_index(drop=True)
    split_df = split_df.sample(n=min(args.n_images, len(split_df)), random_state=args.seed).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work_items = [(str(row.image_id), row.archive_member) for row in split_df.itertuples()]

    import multiprocessing

    with multiprocessing.Pool(
        processes=args.n_workers,
        initializer=_init_worker,
        initargs=("benchmark/manifests/bam_instances.parquet", str(args.archive), args.n_rays, str(args.out_dir)),
    ) as pool:
        completed = 0
        for image_id, n_instances, elapsed in pool.imap_unordered(_process_one, work_items):
            completed += 1
            if n_instances == -1:
                print(f"[{completed}/{len(work_items)}] {image_id}: already cached, skipped", flush=True)
            else:
                print(f"[{completed}/{len(work_items)}] {image_id}: {n_instances} instances, {elapsed:.1f}s", flush=True)

    new_rows = pd.DataFrame([{"image_id": image_id, "archive_member": member} for image_id, member in work_items])
    manifest_path = args.out_dir / "manifest.csv"
    if manifest_path.exists():
        # Accumulate across multiple precompute runs (e.g. different seeds
        # adding more images to the same split) instead of overwriting the
        # manifest and losing images an earlier run already cached to disk.
        existing = pd.read_csv(manifest_path)
        combined = pd.concat([existing, new_rows], ignore_index=True).drop_duplicates(subset="image_id")
    else:
        combined = new_rows
    combined.to_csv(manifest_path, index=False)
    print(f"done: {len(new_rows)} images processed this run, {len(combined)} total in manifest at {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-images", type=int, default=10)
    parser.add_argument("--n-rays", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=20)
    parser.add_argument("--archive", type=Path, default=Path("data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"))
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
