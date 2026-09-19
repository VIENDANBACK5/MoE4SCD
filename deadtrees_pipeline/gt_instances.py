"""Extract tile-level GT instances while preserving source polygon identity.

The source DeadTrees GeoPackage is EPSG:4326, while image tiles use several
projected CRSs.  This module clips each source polygon to the exact tile
footprint and stores all clipped geometries in one lightweight EPSG:4326
GeoPackage.  Raster masks are deliberately not persisted: evaluators rasterize
the geometry with each image's native transform, so polygon holes and separate
touching instances are retained.
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import geopandas as gpd
import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as transform_geometry
from tqdm import tqdm


DEFAULT_IMAGE_ROOT = Path(
    "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
)
DEFAULT_VECTOR_ROOT = Path(
    "DeadTrees/raw/standing-deadwood-aerial-global-conservative"
)
DEFAULT_OUTPUT = Path("DeadTrees/instances_gt/instances.gpkg")
DATASET_PATTERN = re.compile(r"dataset_(\d+)_")


def dataset_id_from_stem(stem: str) -> int:
    match = DATASET_PATTERN.match(stem)
    if not match:
        raise ValueError(f"Cannot parse dataset_id from image stem: {stem}")
    return int(match.group(1))


def _polygonal_or_none(geometry):
    if geometry.is_empty:
        return None
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    polygon_parts = [
        geom for geom in getattr(geometry, "geoms", [])
        if geom.geom_type in {"Polygon", "MultiPolygon"} and not geom.is_empty
    ]
    if not polygon_parts:
        return None
    from shapely.ops import unary_union

    return unary_union(polygon_parts)


def extract_instances(
    image_root: Path = DEFAULT_IMAGE_ROOT,
    vector_root: Path = DEFAULT_VECTOR_ROOT,
    output_path: Path = DEFAULT_OUTPUT,
    layer: str = "standing_deadwood",
    overwrite: bool = False,
) -> dict:
    """Clip one DeadTrees vector layer to each tile footprint.

    `layer` defaults to `standing_deadwood` (the original deadwood-instance
    extraction). Pass `layer="tree_cover"` to run the identical procedure
    against the tree-cover polygons instead, producing a parallel instance
    GeoPackage that `classify_objects.py` uses as a coarse "alive canopy"
    proxy. Both layers carry the same unresolved crown-vs-group ambiguity
    documented in `audit/annotation_schema.md`; this function does not
    resolve that, it only clips whichever layer is requested.
    """
    image_paths = sorted(image_root.glob("**/*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No GeoTIFF images found under {image_root}")

    gpkg_candidates = sorted(vector_root.glob("**/*.gpkg"))
    if not gpkg_candidates:
        raise FileNotFoundError(f"No source GeoPackage found under {vector_root}")
    source_gpkg = gpkg_candidates[0]

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Pass --overwrite to replace it."
        )

    dataset_ids = sorted({dataset_id_from_stem(path.stem) for path in image_paths})
    where = "dataset_id IN ({})".format(",".join(map(str, dataset_ids)))
    source = gpd.read_file(
        source_gpkg,
        layer=layer,
        where=where,
        fid_as_index=True,
    )
    if source.crs is None:
        raise ValueError("Source standing_deadwood layer has no CRS")
    source = source[source.geometry.notna() & ~source.geometry.is_empty].copy()
    if not source.geometry.is_valid.all():
        source.geometry = source.geometry.make_valid()

    by_dataset = {
        int(dataset_id): group.copy()
        for dataset_id, group in source.groupby("dataset_id", sort=True)
    }

    records: list[dict] = []
    geometries = []
    per_image_counts: dict[str, int] = {}

    for image_path in tqdm(image_paths, desc="Extracting GT polygon instances"):
        stem = image_path.stem
        dataset_id = dataset_id_from_stem(stem)
        dataset_polygons = by_dataset.get(dataset_id)
        if dataset_polygons is None or dataset_polygons.empty:
            per_image_counts[stem] = 0
            continue

        with rasterio.open(image_path) as src:
            if src.crs is None:
                raise ValueError(f"Image has no CRS: {image_path}")
            to_wgs84 = Transformer.from_crs(src.crs, source.crs, always_xy=True)
            tile_native = box(*src.bounds)
            tile_wgs84 = transform_geometry(to_wgs84.transform, tile_native)
            pixel_area_native = abs(src.transform.a * src.transform.e)
            to_native = Transformer.from_crs(source.crs, src.crs, always_xy=True)

        candidate_idx = dataset_polygons.sindex.query(
            tile_wgs84, predicate="intersects"
        )
        candidates = dataset_polygons.iloc[candidate_idx]
        n_instances = 0

        for source_fid, row in candidates.iterrows():
            clipped = _polygonal_or_none(row.geometry.intersection(tile_wgs84))
            if clipped is None:
                continue

            clipped_native = transform_geometry(to_native.transform, clipped)
            approx_pixel_area = clipped_native.area / max(pixel_area_native, 1e-12)
            if approx_pixel_area <= 0:
                continue

            instance_id = f"{stem}_fid{int(source_fid)}"
            records.append({
                "instance_id": instance_id,
                "source_fid": int(source_fid),
                "stem": stem,
                "dataset_id": dataset_id,
                "image_path": str(image_path),
                "approx_pixel_area": float(approx_pixel_area),
                "was_clipped": not clipped.equals(row.geometry),
            })
            geometries.append(clipped)
            n_instances += 1

        per_image_counts[stem] = n_instances

    instances = gpd.GeoDataFrame(records, geometry=geometries, crs=source.crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    instances.to_file(output_path, layer="instances", driver="GPKG")

    summary = {
        "source_gpkg": str(source_gpkg),
        "output_gpkg": str(output_path),
        "crs": str(instances.crs),
        "n_images": len(image_paths),
        "n_instances": len(instances),
        "n_images_with_instances": sum(v > 0 for v in per_image_counts.values()),
        "instances_by_dataset": {
            str(k): int(v)
            for k, v in sorted(Counter(instances["dataset_id"]).items())
        },
        "instances_per_image": per_image_counts,
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract per-tile DeadTrees polygon instances"
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--vector-root", type=Path, default=DEFAULT_VECTOR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--layer",
        type=str,
        default="standing_deadwood",
        help="GeoPackage layer to clip (e.g. standing_deadwood or tree_cover)",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = extract_instances(
        image_root=args.image_root,
        vector_root=args.vector_root,
        output_path=args.output,
        layer=args.layer,
        overwrite=args.overwrite,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "instances_per_image"}, indent=2))


if __name__ == "__main__":
    main()

