#!/usr/bin/env python3
"""Reproducible G0B acquisition-output audit for the frozen ITC datasets.

This script intentionally stops at manifests and annotation/geometry QC.  It
does not harmonize imagery, define a final evaluator, tile data, or train a
model.  Geometry is stored as source WKB; repaired geometry is used only for
diagnostic calculations when a source feature is invalid.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import shapely
from matplotlib.patches import Polygon as MplPolygon
from pycocotools import mask as coco_mask
from rasterio.features import shapes as raster_shapes
from rasterio.windows import Window, from_bounds
from shapely import Geometry, box, make_valid
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import transform as geom_transform


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "itc_benchmarks"
ARCHIVES = DATA / "raw_archives"
EXTRACTED = DATA / "extracted"
MANIFESTS = ROOT / "benchmark" / "manifests"
QC = ROOT / "benchmark" / "qc"

BAM_ZIP = ARCHIVES / "Bamberg_coco2048.zip"
QUEBEC_DIR = EXTRACTED / "quebec_2021_09_02"
BCI_ZIP = ARCHIVES / "BCI_50ha_2020_08_01_crownmap_raw.zip"
BCI_DIR = EXTRACTED / "bci_2021_raw" / "BCI_50ha_2020_08_01_crownmap_raw"

BCI_MEMBER_ROOT = "BCI_50ha_2020_08_01_crownmap_raw"
BCI_GLOBAL_MEMBER = f"{BCI_MEMBER_ROOT}/BCI_50ha_2020_08_01_global.tif"
BCI_SHP = BCI_DIR / "BCI_50ha_2020_08_01_crownmap_formatted.shp"

QUEBEC_GSD_M = 0.0164
BAM_GSD_CM = {
    "Hain": 1.82,
    "Stadtwald": 1.70,
    "Tretzendorf-1": 1.61,
    "Tretzendorf-2": 1.79,
    "Tretzendorf": None,  # Two acquisitions: 1.61 and 1.79 cm; filename is ambiguous.
}

IMAGE_COLUMNS = [
    "image_id", "dataset", "site", "split", "filepath", "archive_path",
    "archive_member", "width", "height", "bands", "dtype", "crs",
    "gsd_cm", "gsd_note", "source_role", "acquisition_date",
    "annotation_exhaustive", "background_evaluation", "bounds",
]

INSTANCE_COLUMNS = [
    "image_id", "dataset", "site", "split", "instance_id",
    "canonical_tree_id", "physical_tree_id_known", "geometry_wkb",
    "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax", "area_px",
    "area_m2", "health_status", "edge_flag", "outside_image_flag",
    "clipped_flag", "ignore_flag", "source_annotation_id", "source_label",
    "source_geometry_type", "is_valid", "was_make_valid", "has_holes",
    "multipart_parts", "duplicate_geometry_group", "overlap_degree",
    "inside_publication_inference_zone",
]


def relpath(path: Path) -> str:
    """Return a stable workspace-relative path for manifests."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def vsi_zip(archive: Path, member: str) -> str:
    return f"/vsizip/{archive.resolve().as_posix()}/{member}"


def json_dump(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def polygon_parts(geom: Geometry) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in polygon_parts(g)]
    return []


def holes_count(geom: Geometry) -> int:
    return sum(len(p.interiors) for p in polygon_parts(geom))


def safe_geometry(geom: Geometry) -> tuple[Geometry, bool]:
    if geom is None or geom.is_empty or geom.is_valid:
        return geom, False
    return make_valid(geom), True


def quantiles(values: Sequence[float]) -> dict[str, float | None]:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(a):
        return {k: None for k in ("min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max", "mean")}
    qs = np.quantile(a, [0, .01, .05, .25, .5, .75, .95, .99, 1])
    return {
        "min": float(qs[0]), "p01": float(qs[1]), "p05": float(qs[2]),
        "p25": float(qs[3]), "median": float(qs[4]), "p75": float(qs[5]),
        "p95": float(qs[6]), "p99": float(qs[7]), "max": float(qs[8]),
        "mean": float(a.mean()),
    }


def normalized_wkb_key(geom: Geometry) -> str | None:
    if geom is None or geom.is_empty:
        return None
    try:
        return shapely.to_wkb(shapely.normalize(geom), hex=True)
    except Exception:
        return shapely.to_wkb(geom, hex=True)


def duplicate_groups(
    geoms: Sequence[Geometry], group_ids: Sequence[str] | None = None,
) -> tuple[list[str | None], dict]:
    keys = [normalized_wkb_key(g) for g in geoms]
    if group_ids is not None:
        keys = [f"{group_ids[i]}::{k}" if k is not None else None for i, k in enumerate(keys)]
    counts = Counter(k for k in keys if k is not None)
    repeated = sorted(k for k, n in counts.items() if n > 1)
    mapping = {k: f"dupgeom-{i + 1:05d}" for i, k in enumerate(repeated)}
    groups = [mapping.get(k) for k in keys]
    return groups, {
        "groups": len(repeated),
        "instances_in_groups": int(sum(counts[k] for k in repeated)),
        "max_group_size": int(max((counts[k] for k in repeated), default=1)),
    }


def overlap_audit(
    geoms: Sequence[Geometry], group_ids: Sequence[str] | None = None,
    area_epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Count positive-area intersections, restricted to equal group_ids if supplied."""
    degree = np.zeros(len(geoms), dtype=np.int32)
    positive_areas: list[float] = []
    pairs = 0
    groups: dict[str, list[int]] = defaultdict(list)
    if group_ids is None:
        groups["all"] = list(range(len(geoms)))
    else:
        for i, gid in enumerate(group_ids):
            groups[str(gid)].append(i)
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        arr = np.asarray([geoms[i] for i in idxs], dtype=object)
        valid_local = [j for j, g in enumerate(arr) if g is not None and not g.is_empty]
        if len(valid_local) < 2:
            continue
        clean = arr[valid_local]
        tree = shapely.STRtree(clean)
        ij = tree.query(clean, predicate="intersects")
        keep = ij[0] < ij[1]
        left = ij[0][keep]
        right = ij[1][keep]
        if not len(left):
            continue
        intersections = shapely.intersection(clean[left], clean[right])
        areas = shapely.area(intersections)
        for l, r, area in zip(left, right, areas):
            # GEOS can emit machine-precision slivers when repaired rings share
            # a boundary.  Do not report those as biological crown overlap.
            if float(area) <= area_epsilon:
                continue
            gi = idxs[valid_local[int(l)]]
            gj = idxs[valid_local[int(r)]]
            degree[gi] += 1
            degree[gj] += 1
            pairs += 1
            positive_areas.append(float(area))
    return degree, {
        "positive_area_pairs": pairs,
        "instances_in_positive_overlap": int(np.count_nonzero(degree)),
        "overlap_area": quantiles(positive_areas),
        "total_overlap_area": float(sum(positive_areas)),
    }


def common_geometry_summary(
    raw_geoms: Sequence[Geometry], analysis_geoms: Sequence[Geometry],
    edges: Sequence[bool], outside: Sequence[bool], clipped: Sequence[bool],
    areas_px: Sequence[float], areas_m2: Sequence[float | None],
    duplicate_info: dict, overlap_info: dict,
) -> dict:
    types = Counter(g.geom_type if g is not None else "NULL" for g in raw_geoms)
    invalid_reasons = Counter(
        str(shapely.is_valid_reason(g)).split("[")[0]
        for g in raw_geoms if g is not None and not g.is_empty and not g.is_valid
    )
    return {
        "n_polygons": len(raw_geoms),
        "n_invalid": int(sum(g is not None and not g.is_empty and not g.is_valid for g in raw_geoms)),
        "n_empty": int(sum(g is not None and g.is_empty for g in raw_geoms)),
        "n_null": int(sum(g is None for g in raw_geoms)),
        "n_zero_area": int(sum(g is not None and float(g.area) <= 0 for g in raw_geoms)),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "geometry_types": dict(sorted(types.items())),
        "n_multipolygon": int(types.get("MultiPolygon", 0)),
        "n_with_holes": int(sum(holes_count(g) > 0 for g in raw_geoms if g is not None)),
        "n_holes": int(sum(holes_count(g) for g in raw_geoms if g is not None)),
        "n_repaired_for_analysis": int(sum(r is not a for r, a in zip(raw_geoms, analysis_geoms))),
        "n_outside_image": int(sum(outside)),
        "n_touching_edge": int(sum(edges)),
        "n_clipped_or_partly_outside": int(sum(clipped)),
        "area_px": quantiles(areas_px),
        "area_m2": quantiles(areas_m2),
        "exact_duplicate_geometry": duplicate_info,
        "gt_overlap": overlap_info,
        "qc_software": {
            "python": os.sys.version.split()[0],
            "geopandas": gpd.__version__,
            "pandas": pd.__version__,
            "rasterio": rasterio.__version__,
            "shapely": shapely.__version__,
        },
    }


def raster_meta(filepath: str) -> dict:
    with rasterio.open(filepath) as src:
        return {
            "width": src.width, "height": src.height, "bands": src.count,
            "dtype": src.dtypes[0], "crs": src.crs.to_string() if src.crs else None,
            "gsd_x": abs(float(src.transform.a)), "gsd_y": abs(float(src.transform.e)),
            "bounds_obj": box(*src.bounds), "bounds": [float(x) for x in src.bounds],
            "transform": src.transform,
        }


def infer_bam_site(name: str) -> str:
    low = name.lower()
    if "stadt" in low:
        return "Stadtwald"
    if "tretz" in low:
        return "Tretzendorf"
    if "hain" in low:
        return "Hain"
    return name.split("_")[0].split("-")[0]


def coco_polygon(segmentation, height: int, width: int) -> Geometry:
    if isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = coco_mask.frPyObjects(rle, height, width)
        mask = coco_mask.decode(rle)
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        parts = [shape(g) for g, value in raster_shapes(mask.astype(np.uint8), mask=mask.astype(bool)) if int(value) == 1]
        if not parts:
            return Polygon()
        merged = shapely.union_all(parts)
        polys = polygon_parts(merged)
        return polys[0] if len(polys) == 1 else MultiPolygon(polys)
    if not isinstance(segmentation, list):
        raise ValueError(f"Unsupported COCO segmentation type: {type(segmentation).__name__}")
    if segmentation and isinstance(segmentation[0], (int, float)):
        segmentation = [segmentation]
    parts = []
    for flat in segmentation:
        if len(flat) < 6 or len(flat) % 2:
            continue
        coords = np.asarray(flat, dtype=float).reshape(-1, 2)
        p = Polygon(coords)
        if not p.is_empty:
            parts.append(p)
    if not parts:
        return Polygon()
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def find_bam_jsons() -> list[Path]:
    candidates = sorted((EXTRACTED / "bam_coco2048").glob("*.json"))
    if len(candidates) != 4:
        raise FileNotFoundError(f"Expected 4 extracted BAM COCO JSONs, found {len(candidates)}")
    return candidates


def audit_bam(make_montage: bool = True) -> dict:
    jsons = find_bam_jsons()
    with zipfile.ZipFile(BAM_ZIP) as zf:
        members = set(zf.namelist())
    image_rows: list[dict] = []
    raw_rows: list[dict] = []
    all_raw: list[Geometry] = []
    all_analysis: list[Geometry] = []
    image_bounds: dict[str, Geometry] = {}
    image_dims: dict[str, tuple[int, int]] = {}
    image_members: dict[str, str] = {}
    image_site: dict[str, str] = {}
    image_gsd_cm: dict[str, float | None] = {}
    image_annotations: dict[str, list[tuple[str, Geometry]]] = defaultdict(list)
    categories: dict[int, str] = {}
    missing_members: list[str] = []
    metadata_mismatch: list[dict] = []
    iscrowd_counts: Counter = Counter()
    segmentation_formats: Counter = Counter()

    for jp in jsons:
        obj = json.loads(jp.read_text())
        split = re.sub(r"[^A-Za-z0-9]+", "_", jp.stem).strip("_").lower()
        # Prefer a recognizable split token over a verbose annotation filename.
        compact_split = split.replace("_", "")
        split_aliases = (("train", "train"), ("eval", "val"), ("val", "val"),
                         ("testset1", "test1"), ("test1", "test1"),
                         ("testset2", "test2"), ("test2", "test2"))
        for source_token, canonical in split_aliases:
            if source_token in compact_split:
                split = canonical
                break
        categories.update({int(c["id"]): str(c.get("name", c["id"])) for c in obj.get("categories", [])})
        by_id = {int(i["id"]): i for i in obj["images"]}
        id_map: dict[int, str] = {}
        for source_id, im in by_id.items():
            filename = str(im["file_name"]).replace("\\", "/")
            matches = [m for m in members if m == filename or m.endswith("/" + filename)]
            if len(matches) != 1:
                missing_members.append(filename)
                continue
            member = matches[0]
            iid = f"bam:{split}:{source_id}"
            id_map[source_id] = iid
            filepath = vsi_zip(BAM_ZIP, member)
            meta = raster_meta(filepath)
            if meta["width"] != int(im["width"]) or meta["height"] != int(im["height"]):
                metadata_mismatch.append({"image": filename, "coco": [im["width"], im["height"]], "tiff": [meta["width"], meta["height"]]})
            site = infer_bam_site(filename)
            gsd = BAM_GSD_CM.get(site)
            # Release polygons use half-pixel boundary coordinates.
            image_bounds[iid] = box(-0.5, -0.5, int(im["width"]) - 0.5, int(im["height"]) - 0.5)
            image_dims[iid] = (int(im["width"]), int(im["height"]))
            image_members[iid] = member
            image_site[iid] = site
            image_gsd_cm[iid] = gsd
            image_rows.append({
                "image_id": iid, "dataset": "BAMFORESTS_coco2048", "site": site,
                "split": split, "filepath": filepath, "archive_path": relpath(BAM_ZIP),
                "archive_member": member, "width": meta["width"], "height": meta["height"],
                "bands": meta["bands"], "dtype": meta["dtype"], "crs": meta["crs"],
                "gsd_cm": gsd, "gsd_note": "native AOI GSD" if gsd else "Tretzendorf acquisition ambiguity: native 1.61 or 1.79 cm",
                "source_role": "official_overlapping_2048_crop", "acquisition_date": None,
                "annotation_exhaustive": "documented_complete_visible_crowns_with_ambiguity_in_dense_deciduous_canopy",
                "background_evaluation": "official_split_only; boundary-ignore policy required in G0C",
                "bounds": json.dumps([0, 0, meta["width"], meta["height"]]),
            })
        for ann in obj["annotations"]:
            src_image_id = int(ann["image_id"])
            if src_image_id not in id_map:
                continue
            iid = id_map[src_image_id]
            segmentation_formats["rle" if isinstance(ann["segmentation"], dict) else "polygon"] += 1
            height, width = image_dims[iid][1], image_dims[iid][0]
            geom = coco_polygon(ann["segmentation"], height, width)
            analysis, repaired = safe_geometry(geom)
            row_i = len(all_raw)
            all_raw.append(geom)
            all_analysis.append(analysis)
            category = categories.get(int(ann.get("category_id", -1)), str(ann.get("category_id")))
            instance_id = f"bam:{split}:ann:{ann['id']}"
            iscrowd_counts[int(ann.get("iscrowd", 0))] += 1
            raw_rows.append({
                "_index": row_i, "image_id": iid, "dataset": "BAMFORESTS_coco2048",
                "site": image_site[iid], "split": split,
                "instance_id": instance_id, "canonical_tree_id": None,
                "physical_tree_id_known": False, "geometry_wkb": shapely.to_wkb(geom),
                "health_status": "UNKNOWN", "ignore_flag": bool(ann.get("iscrowd", 0)),
                "source_annotation_id": str(ann["id"]), "source_label": category,
                "source_geometry_type": geom.geom_type, "is_valid": bool(geom.is_valid),
                "was_make_valid": repaired, "has_holes": holes_count(geom) > 0,
                "multipart_parts": len(polygon_parts(geom)), "inside_publication_inference_zone": None,
                "_source_area": float(ann.get("area", np.nan)),
                "_source_bbox": [float(x) for x in ann.get("bbox", [np.nan] * 4)],
            })
            image_annotations[iid].append((instance_id, analysis))

    edge, outside, clipped, area_px, area_m2 = [], [], [], [], []
    for row, geom in zip(raw_rows, all_analysis):
        bounds = image_bounds[row["image_id"]]
        is_outside = not bounds.covers(geom) if geom is not None and not geom.is_empty else False
        sx, sy, sw, sh = row["_source_bbox"]
        width, height = image_dims[row["image_id"]]
        bbox_edge = bool(np.isfinite([sx, sy, sw, sh]).all() and (
            sx <= 0 or sy <= 0 or sx + sw >= width or sy + sh >= height
        ))
        touches_geom = geom.intersects(bounds.boundary) if geom is not None and not geom.is_empty else False
        touches = bool(bbox_edge or touches_geom)
        edge.append(bool(touches)); outside.append(bool(is_outside)); clipped.append(bool(touches or is_outside))
        a = float(geom.area) if geom is not None else 0.0
        area_px.append(a)
        gsd_cm = image_gsd_cm[row["image_id"]]
        area_m2.append(a * (float(gsd_cm) / 100) ** 2 if gsd_cm is not None else None)
    source_area_delta: list[float] = []
    source_area_relative_delta: list[float] = []
    source_bbox_delta: list[float] = []
    dups, dup_info = duplicate_groups(all_analysis, [r["image_id"] for r in raw_rows])
    degrees, overlap_info = overlap_audit(all_analysis, [r["image_id"] for r in raw_rows])
    for i, row in enumerate(raw_rows):
        b = all_analysis[i].bounds if all_analysis[i] is not None and not all_analysis[i].is_empty else (np.nan,) * 4
        row.update({
            "bbox_xmin": b[0], "bbox_ymin": b[1], "bbox_xmax": b[2], "bbox_ymax": b[3],
            "area_px": area_px[i], "area_m2": area_m2[i], "edge_flag": edge[i],
            "outside_image_flag": outside[i], "clipped_flag": clipped[i],
            "ignore_flag": bool(row["ignore_flag"] or clipped[i] or not row["is_valid"] or area_px[i] <= 0),
            "duplicate_geometry_group": dups[i], "overlap_degree": int(degrees[i]),
        })
        if np.isfinite(row["_source_area"]):
            source_area_delta.append(abs(float(row["_source_area"]) - area_px[i]))
            if float(row["_source_area"]) > 0:
                source_area_relative_delta.append(source_area_delta[-1] / float(row["_source_area"]))
        sx, sy, sw, sh = row["_source_bbox"]
        source_xyxy = np.asarray([sx, sy, sx + sw, sy + sh], dtype=float)
        if np.isfinite(source_xyxy).all():
            source_bbox_delta.append(float(np.max(np.abs(source_xyxy - np.asarray(b)))))
        row.pop("_index", None); row.pop("_source_area", None); row.pop("_source_bbox", None)

    pd.DataFrame(image_rows, columns=IMAGE_COLUMNS).sort_values(["split", "image_id"]).to_csv(MANIFESTS / "bam_images.csv", index=False)
    pd.DataFrame(raw_rows, columns=INSTANCE_COLUMNS).to_parquet(MANIFESTS / "bam_instances.parquet", index=False)
    summary = common_geometry_summary(all_raw, all_analysis, edge, outside, clipped, area_px, area_m2, dup_info, overlap_info)
    summary.update({
        "dataset": "BAMFORESTS coco2048", "images_loaded": len(image_rows),
        "images_by_split": dict(Counter(r["split"] for r in image_rows)),
        "crowns_by_split": dict(Counter(r["split"] for r in raw_rows)),
        "categories": categories, "iscrowd": {str(k): v for k, v in sorted(iscrowd_counts.items())},
        "segmentation_formats": dict(sorted(segmentation_formats.items())),
        "source_coco_area_absolute_delta_px2": quantiles(source_area_delta),
        "source_coco_area_relative_delta": quantiles(source_area_relative_delta),
        "source_coco_bbox_max_coordinate_delta_px": quantiles(source_bbox_delta),
        "source_coco_area_delta_gt_1px2": int(sum(x > 1 for x in source_area_delta)),
        "source_coco_bbox_delta_gt_1px": int(sum(x > 1 for x in source_bbox_delta)),
        "primary_ignore_instances": int(sum(bool(r["ignore_flag"]) for r in raw_rows)),
        "missing_tiff_members": missing_members, "tiff_coco_dimension_mismatch": metadata_mismatch,
        "native_gsd_cm_by_site": BAM_GSD_CM,
        "biological_id_audit": "No biological tree identifier is supplied; annotation IDs remain crop-instance IDs and canonical_tree_id is null.",
        "crop_duplicate_audit": "Overlapping 2048 crops can repeat physical crowns. The paper constructs train/validation/Test-2 from complete hectare plots so 50%-overlap crops remain within a split; Test-1 is the independent Hain AOI. The COCO release provides no biological ID, so cross-crop identity is not recoverable from annotation_id; official splits must be preserved.",
        "degenerate_geometry_policy": "Invalid or zero-area crop annotations are preserved in raw WKB but set ignore_flag=true. They are not silently repaired into primary GT.",
    })
    expected_images = {"train": 1439, "val": 382, "test1": 313, "test2": 322}
    expected_crowns = {"train": 58228, "val": 15177, "test1": 6720, "test2": 12320}
    if summary["images_by_split"] != expected_images:
        raise RuntimeError(f"BAM image inventory differs from frozen archive preflight: {summary['images_by_split']}")
    if summary["crowns_by_split"] != expected_crowns:
        raise RuntimeError(f"BAM crown inventory differs from frozen archive preflight: {summary['crowns_by_split']}")
    if missing_members or metadata_mismatch:
        raise RuntimeError(f"BAM TIFF integrity failure: missing={len(missing_members)}, dimension_mismatch={len(metadata_mismatch)}")
    json_dump(QC / "bam_geometry.json", summary)
    if make_montage:
        make_bam_montage(image_rows, raw_rows, image_annotations, image_members)
    return summary


def iter_exteriors(geom: Geometry) -> Iterable[np.ndarray]:
    for p in polygon_parts(geom):
        yield np.asarray(p.exterior.coords)


def stretch_rgb(arr: np.ndarray) -> np.ndarray:
    arr = np.moveaxis(arr[:3], 0, -1).astype(float)
    out = np.zeros_like(arr, dtype=float)
    for c in range(min(3, arr.shape[-1])):
        band = arr[..., c]
        valid = np.isfinite(band)
        if not valid.any():
            continue
        lo, hi = np.percentile(band[valid], [2, 98])
        out[..., c] = np.clip((band - lo) / max(hi - lo, 1e-9), 0, 1)
    return out


def choose_stratified(rows: list[dict], n: int = 30) -> list[tuple[int, str]]:
    areas = np.asarray([float(r["area_px"]) for r in rows])
    degree = np.asarray([int(r["overlap_degree"]) for r in rows])
    edge = np.asarray([bool(r["edge_flag"]) for r in rows])
    usable = areas > 0
    positive_areas = areas[usable]
    pools = [
        ("small", np.where(usable & (areas <= np.quantile(positive_areas, .05)))[0]),
        ("large", np.where(usable & (areas >= np.quantile(positive_areas, .95)))[0]),
        ("dense", np.where(usable & (degree >= max(1, np.quantile(degree, .90))))[0]),
        ("isolated", np.where(usable & (degree == 0))[0]),
        ("edge", np.where(usable & edge)[0]),
    ]
    dead = np.asarray([str(r.get("health_status", "")).upper() == "DEAD" for r in rows])
    if dead.any():
        pools.append(("dead", np.where(dead)[0]))
    shadow = np.asarray([bool(r.get("_shadow_hint", False)) for r in rows])
    if shadow.any():
        pools.append(("shadow/low-light", np.where(shadow)[0]))
    chosen: list[tuple[int, str]] = []
    seen: set[int] = set()
    rng = np.random.default_rng(20210824)
    quota = max(1, n // len(pools))
    for label, pool in pools:
        for idx in rng.permutation(pool)[:quota]:
            if int(idx) not in seen:
                chosen.append((int(idx), label)); seen.add(int(idx))
    for idx in rng.permutation(len(rows)):
        if len(chosen) >= n:
            break
        if int(idx) not in seen:
            chosen.append((int(idx), "random")); seen.add(int(idx))
    return chosen[:n]


def make_bam_montage(image_rows, instance_rows, image_annotations, image_members) -> None:
    selected = choose_stratified(instance_rows, 30)
    fig, axes = plt.subplots(5, 6, figsize=(24, 20), constrained_layout=True)
    image_lookup = {r["image_id"]: r for r in image_rows}
    for ax, (idx, stratum) in zip(axes.flat, selected):
        target = instance_rows[idx]
        iid = target["image_id"]
        with rasterio.open(vsi_zip(BAM_ZIP, image_members[iid])) as src:
            arr = src.read([1, 2, 3], out_shape=(3, 512, 512))
        ax.imshow(stretch_rgb(arr), origin="upper")
        sx = 512 / image_lookup[iid]["width"]
        sy = 512 / image_lookup[iid]["height"]
        for ann_id, geom in image_annotations[iid]:
            color = "yellow" if ann_id == target["instance_id"] else "#00ffff"
            lw = 1.6 if ann_id == target["instance_id"] else .35
            for xy in iter_exteriors(geom):
                # COCO coordinates use a top-left pixel origin.
                ax.add_patch(MplPolygon(np.c_[xy[:, 0] * sx, xy[:, 1] * sy], fill=False, edgecolor=color, linewidth=lw))
        ax.set_title(f"{stratum} | {target['site']} | {target['split']}\n{target['instance_id']}", fontsize=7)
        ax.axis("off")
    fig.suptitle("BAMFORESTS: native RGB + GT (cyan), sampled crown (yellow)", fontsize=15)
    fig.savefig(QC / "bam_examples.png", dpi=150)
    plt.close(fig)


def find_quebec_cog(zone: int) -> Path:
    matches = sorted(QUEBEC_DIR.glob(f"*z{zone}-rgb-cog.tif"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one Quebec Z{zone} Sep-02 COG, found {matches}")
    return matches[0]


def audit_quebec(make_montage: bool = True) -> dict:
    inference = gpd.read_file(QUEBEC_DIR / "inference_zone.gpkg")
    inference_union = shapely.union_all(inference.geometry.to_numpy())
    image_rows: list[dict] = []
    rows: list[dict] = []
    raw_geoms: list[Geometry] = []
    analysis_geoms: list[Geometry] = []
    zone_bounds: dict[str, Geometry] = {}
    gdfs: dict[int, gpd.GeoDataFrame] = {}
    label_counts: Counter = Counter()
    health_counts: Counter = Counter()
    edge: list[bool] = []
    outside: list[bool] = []
    clipped: list[bool] = []
    areas_px: list[float] = []
    areas_m2: list[float] = []
    quebec_archive = ARCHIVES / "quebec_trees_dataset_2021-09-02.zip"
    with zipfile.ZipFile(quebec_archive) as zf:
        archive_members = {Path(m).name: m for m in zf.namelist()}

    for zone in (1, 2, 3):
        cog = find_quebec_cog(zone)
        meta = raster_meta(str(cog))
        iid = f"quebec:2021-09-02:z{zone}"
        zone_bounds[iid] = meta["bounds_obj"]
        image_rows.append({
            "image_id": iid, "dataset": "Quebec_Trees_2021-09-02", "site": f"Z{zone}",
            "split": "external_health", "filepath": relpath(cog), "archive_path": relpath(quebec_archive),
            "archive_member": archive_members.get(cog.name, cog.name), "width": meta["width"], "height": meta["height"],
            "bands": meta["bands"], "dtype": meta["dtype"], "crs": meta["crs"],
            "gsd_cm": 100 * meta["gsd_x"], "gsd_note": "native COG pixel size; STAC nominal GSD is 1.64 cm",
            "source_role": "dated_zone_orthomosaic", "acquisition_date": "2021-09-02",
            "annotation_exhaustive": "no_for_object_level_evaluation; paper documents image regions not annotated",
            "background_evaluation": "ignore/unscored for object-level false positives; matched-GT metrics only",
            "bounds": json.dumps(meta["bounds"]),
        })
        gpkg = QUEBEC_DIR / f"Z{zone}_polygons.gpkg"
        gdf = gpd.read_file(gpkg)
        if gdf.crs != rasterio.crs.CRS.from_epsg(32618):
            raise ValueError(f"Unexpected Quebec polygon CRS: {gdf.crs}")
        gdfs[zone] = gdf
        for source_row, record in gdf.iterrows():
            geom = record.geometry
            analysis, repaired = safe_geometry(geom)
            raw_geoms.append(geom); analysis_geoms.append(analysis)
            label = str(record["Label"]) if pd.notna(record["Label"]) else "UNKNOWN"
            health = "DEAD" if label.strip().lower() == "mort" else ("UNKNOWN" if label == "UNKNOWN" else "NON_DEAD")
            label_counts[label] += 1; health_counts[health] += 1
            bounds = zone_bounds[iid]
            out = not bounds.covers(analysis)
            touch = analysis.intersects(bounds.boundary)
            edge.append(bool(touch)); outside.append(bool(out)); clipped.append(bool(touch or out))
            area_m2 = float(analysis.area)
            areas_m2.append(area_m2); areas_px.append(area_m2 / (meta["gsd_x"] * meta["gsd_y"]))
            b = analysis.bounds
            rows.append({
                "image_id": iid, "dataset": "Quebec_Trees_2021-09-02", "site": f"Z{zone}",
                "split": "external_health", "instance_id": f"quebec:z{zone}:row:{source_row}",
                "canonical_tree_id": None, "physical_tree_id_known": False,
                "geometry_wkb": shapely.to_wkb(geom), "bbox_xmin": b[0], "bbox_ymin": b[1],
                "bbox_xmax": b[2], "bbox_ymax": b[3], "area_px": areas_px[-1], "area_m2": area_m2,
                "health_status": health, "edge_flag": edge[-1], "outside_image_flag": outside[-1],
                "clipped_flag": clipped[-1], "ignore_flag": clipped[-1], "source_annotation_id": str(source_row),
                "source_label": label, "source_geometry_type": geom.geom_type, "is_valid": bool(geom.is_valid),
                "was_make_valid": repaired, "has_holes": holes_count(geom) > 0,
                "multipart_parts": len(polygon_parts(geom)), "duplicate_geometry_group": None,
                "overlap_degree": 0, "inside_publication_inference_zone": bool(analysis.centroid.within(inference_union)),
            })

    dups, dup_info = duplicate_groups(analysis_geoms)
    degrees, overlap_info = overlap_audit(analysis_geoms, [r["image_id"] for r in rows])
    for i, row in enumerate(rows):
        row["duplicate_geometry_group"] = dups[i]
        row["overlap_degree"] = int(degrees[i])
    pd.DataFrame(image_rows, columns=IMAGE_COLUMNS).to_csv(MANIFESTS / "quebec_images.csv", index=False)
    pd.DataFrame(rows, columns=INSTANCE_COLUMNS).to_parquet(MANIFESTS / "quebec_instances.parquet", index=False)
    summary = common_geometry_summary(raw_geoms, analysis_geoms, edge, outside, clipped, areas_px, areas_m2, dup_info, overlap_info)
    summary.update({
        "dataset": "Quebec Trees 2021-09-02", "images_loaded": len(image_rows),
        "images": [{k: r[k] for k in ("image_id", "width", "height", "gsd_cm", "bounds")} for r in image_rows],
        "crowns_by_zone": dict(Counter(r["site"] for r in rows)),
        "labels": dict(sorted(label_counts.items())), "health_status": dict(sorted(health_counts.items())),
        "native_gsd_cm": [r["gsd_cm"] for r in image_rows],
        "inference_zone": {
            "features": len(inference), "valid": int(inference.geometry.is_valid.sum()),
            "crowns_centroid_inside": int(sum(r["inside_publication_inference_zone"] for r in rows)),
        },
        "tree_id_audit": "Published GPKGs contain only Label and geometry. Field/biological tree identifiers were stripped, so polygon-to-field-ID mapping, duplicate biological IDs, and repeated-tree identity cannot be audited. canonical_tree_id is null; instance_id is zone+source row only.",
        "registration_audit": "Polygon and COG CRS/bounds are checked quantitatively; visual overlay montage is the qualitative registration check.",
    })
    if len(image_rows) != 3 or len(rows) != 22933:
        raise RuntimeError(f"Quebec inventory differs from frozen release: images={len(image_rows)}, crowns={len(rows)}")
    if summary["n_invalid"] or summary["n_empty"] or summary["n_zero_area"]:
        raise RuntimeError("Quebec contains an unexpected invalid/empty/zero-area source geometry")
    json_dump(QC / "quebec_geometry.json", summary)
    if make_montage:
        make_geo_montage("Quebec", rows, analysis_geoms, image_rows, QC / "quebec_examples.png")
    return summary


def audit_bci(make_montage: bool = True) -> dict:
    global_path = vsi_zip(BCI_ZIP, BCI_GLOBAL_MEMBER)
    global_meta = raster_meta(global_path)
    gdf = gpd.read_file(BCI_SHP)
    raw_geoms = list(gdf.geometry)
    analysis_pairs = [safe_geometry(g) for g in raw_geoms]
    analysis_geoms = [x[0] for x in analysis_pairs]
    repaired = [x[1] for x in analysis_pairs]
    global_bounds = global_meta["bounds_obj"]

    image_rows: list[dict] = []
    iid = "bci:global"
    bci_gsd_cm = 100 * global_meta["gsd_x"]
    image_rows.append({
        "image_id": iid, "dataset": "BCI_2021_raw_manual", "site": "BCI_50ha",
        "split": "external_ood", "filepath": global_path, "archive_path": relpath(BCI_ZIP),
        "archive_member": BCI_GLOBAL_MEMBER, "width": global_meta["width"], "height": global_meta["height"],
        "bands": global_meta["bands"], "dtype": global_meta["dtype"], "crs": global_meta["crs"],
        "gsd_cm": bci_gsd_cm, "gsd_note": "exact native transform (source description rounds to 4 cm)",
        "source_role": "annotation_reference_orthomosaic", "acquisition_date": "2020-08-01",
        "annotation_exhaustive": "selected_manually_delineated_canopy_crowns; not exhaustive",
        "background_evaluation": "ignore/unscored for false positives; matched-GT metrics only",
        "bounds": json.dumps(global_meta["bounds"]),
    })
    tile_bounds: list[Geometry] = []
    tile_ids: list[str] = []
    for tile in range(50):
        member = f"{BCI_MEMBER_ROOT}/tiles/output_raster_{tile}.tif"
        filepath = vsi_zip(BCI_ZIP, member)
        meta = raster_meta(filepath)
        tid = f"bci:tile:{tile:02d}"
        tile_ids.append(tid); tile_bounds.append(meta["bounds_obj"])
        image_rows.append({
            "image_id": tid, "dataset": "BCI_2021_raw_manual", "site": "BCI_50ha",
            "split": "external_ood_derived_view", "filepath": filepath, "archive_path": relpath(BCI_ZIP),
            "archive_member": member, "width": meta["width"], "height": meta["height"],
            "bands": meta["bands"], "dtype": meta["dtype"], "crs": meta["crs"],
            "gsd_cm": 100 * meta["gsd_x"], "gsd_note": "exact native transform",
            "source_role": "derived_overlapping_crop_of_global_orthomosaic", "acquisition_date": "2020-08-01",
            "annotation_exhaustive": "inherits selected crown subset; instances anchored to global image to avoid duplication",
            "background_evaluation": "never random split; not a separate independent scene",
            "bounds": json.dumps(meta["bounds"]),
        })

    edge = [bool(g.intersects(global_bounds.boundary)) for g in analysis_geoms]
    outside = [bool(not global_bounds.covers(g)) for g in analysis_geoms]
    clipped = [a or b for a, b in zip(edge, outside)]
    areas_m2 = [float(g.area) for g in analysis_geoms]
    areas_px = [a / (global_meta["gsd_x"] * global_meta["gsd_y"]) for a in areas_m2]
    dups, dup_info = duplicate_groups(analysis_geoms)
    degrees, overlap_info = overlap_audit(analysis_geoms)
    tag_counts = gdf["tag"].value_counts(dropna=False)
    rows: list[dict] = []
    for i, (source_row, record) in enumerate(gdf.iterrows()):
        geom = raw_geoms[i]; a = analysis_geoms[i]; b = a.bounds
        tag = int(record["tag"])
        tag_is_unique = tag != -9999 and int(tag_counts.loc[tag]) == 1
        mnemonic = record.get("Mnemonic")
        source_label = str(mnemonic) if pd.notna(mnemonic) else "UNKNOWN"
        seen = record.get("SeenInFiel")
        source_uncertain = pd.isna(seen) or str(seen).strip().lower() == "no" or float(record.get("Flag", 0) or 0) == 1.0
        illumination = pd.to_numeric(pd.Series([record.get("Illuminati")]), errors="coerce").iloc[0]
        rows.append({
            "image_id": iid, "dataset": "BCI_2021_raw_manual", "site": "BCI_50ha", "split": "external_ood",
            "instance_id": f"bci:globalid:{record['GlobalID']}",
            "canonical_tree_id": f"bci:tag:{tag}" if tag_is_unique else None,
            "physical_tree_id_known": tag_is_unique, "geometry_wkb": shapely.to_wkb(geom),
            "bbox_xmin": b[0], "bbox_ymin": b[1], "bbox_xmax": b[2], "bbox_ymax": b[3],
            "area_px": areas_px[i], "area_m2": areas_m2[i], "health_status": "UNKNOWN",
            "edge_flag": edge[i], "outside_image_flag": outside[i], "clipped_flag": clipped[i],
            "ignore_flag": bool(source_uncertain or clipped[i]), "source_annotation_id": str(record["GlobalID"]),
            "source_label": source_label, "source_geometry_type": geom.geom_type,
            "is_valid": bool(geom.is_valid), "was_make_valid": repaired[i], "has_holes": holes_count(geom) > 0,
            "multipart_parts": len(polygon_parts(geom)), "duplicate_geometry_group": dups[i],
            "overlap_degree": int(degrees[i]), "inside_publication_inference_zone": None,
            "_shadow_hint": bool(pd.notna(illumination) and float(illumination) <= 2),
        })

    # Raster tile overlap is spatial leakage evidence, independent of crown labels.
    tile_degree, tile_overlap = overlap_audit(tile_bounds)
    tile_union = shapely.union_all(tile_bounds)
    # overlap_audit records area in square metres because tile bounds are projected.
    pd.DataFrame(image_rows, columns=IMAGE_COLUMNS).to_csv(MANIFESTS / "bci_images.csv", index=False)
    pd.DataFrame(rows, columns=INSTANCE_COLUMNS).to_parquet(MANIFESTS / "bci_instances.parquet", index=False)
    summary = common_geometry_summary(raw_geoms, analysis_geoms, edge, outside, clipped, areas_px, areas_m2, dup_info, overlap_info)
    crown_area_delta = np.abs(np.asarray(areas_m2) - pd.to_numeric(gdf["crownArea"], errors="coerce").to_numpy())
    summary.update({
        "dataset": "BCI 2021 raw/manual crowns", "images_loaded": len(image_rows),
        "image_inventory": {"global_orthomosaic": 1, "derived_tiles": 50},
        "native_gsd_cm": bci_gsd_cm, "crs": global_meta["crs"], "global_bounds": global_meta["bounds"],
        "global_id_unique": bool(gdf["GlobalID"].is_unique),
        "canonical_tree_id": {
            "source": "field tag when non-sentinel and unique",
            "known_unique_rows": int(sum(bool(r["physical_tree_id_known"]) for r in rows)),
            "unknown_or_ambiguous_rows": int(sum(not bool(r["physical_tree_id_known"]) for r in rows)),
        },
        "tag_audit": {
            "unique": int(gdf["tag"].nunique(dropna=False)),
            "duplicate_groups": int((tag_counts > 1).sum()),
            "rows_in_duplicate_groups": int(tag_counts[tag_counts > 1].sum()),
            "max_multiplicity": int(tag_counts.max()),
            "largest_groups": {str(k): int(v) for k, v in tag_counts.head(10).items()},
        },
        "seen_in_field": {str(k): int(v) for k, v in gdf["SeenInFiel"].value_counts(dropna=False).items()},
        "source_uncertain_or_revision_ignore_rows": int(sum(r["ignore_flag"] for r in rows)),
        "crown_condition": {str(k): int(v) for k, v in gdf["CrownCondi"].value_counts(dropna=False).items()},
        "source_crown_area_absolute_delta_m2": quantiles(crown_area_delta),
        "tile_relationship": {
            "all_tiles_within_global_bounds": bool(all(global_bounds.covers(b) for b in tile_bounds)),
            "tile_union_covers_global_bounds": bool(tile_union.covers(global_bounds)),
            "global_area_not_covered_by_tiles_m2": float(global_bounds.difference(tile_union).area),
            "tile_area_outside_global_m2": float(tile_union.difference(global_bounds).area),
            "positive_overlap_tile_pairs": tile_overlap["positive_area_pairs"],
            "tiles_in_overlap": tile_overlap["instances_in_positive_overlap"],
            "overlap_area_m2": tile_overlap["overlap_area"],
            "policy": "The 50 rasters are overlapping derived views of one orthomosaic; never random-split or count them as independent scenes.",
        },
        "geometry_repair_policy": "Raw WKB is preserved. Invalid source polygons are repaired with Shapely make_valid only for QC measurements; G0C must make this deterministic and versioned.",
        "registration_audit": "All crowns fall within the projected global raster bounds; visual overlay montage is the qualitative registration check.",
    })
    if len(image_rows) != 51 or len(rows) != 2454:
        raise RuntimeError(f"BCI inventory differs from frozen release: images={len(image_rows)}, crowns={len(rows)}")
    json_dump(QC / "bci_geometry.json", summary)
    if make_montage:
        make_geo_montage("BCI", rows, analysis_geoms, image_rows[:1], QC / "bci_examples.png")
    return summary


def make_geo_montage(dataset: str, rows: list[dict], geoms: list[Geometry], image_rows: list[dict], output: Path) -> None:
    selected = choose_stratified(rows, 30)
    lookup = {r["image_id"]: r for r in image_rows}
    by_image: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_image[row["image_id"]].append(i)
    fig, axes = plt.subplots(5, 6, figsize=(24, 20), constrained_layout=True)
    # Opening a deflated GeoTIFF inside a ZIP repeatedly can force GDAL to scan
    # the large member each time.  Keep one reader open per source image.
    with contextlib.ExitStack() as stack:
        sources = {
            iid: stack.enter_context(rasterio.open(
                im["filepath"] if str(im["filepath"]).startswith("/vsizip/") else ROOT / im["filepath"]
            )) for iid, im in lookup.items()
        }
        for ax, (idx, stratum) in zip(axes.flat, selected):
            row = rows[idx]; geom = geoms[idx]
            minx, miny, maxx, maxy = geom.bounds
            span = max(maxx - minx, maxy - miny, 5.0)
            pad = max(4.0, span * 1.2)
            req = (minx - pad, miny - pad, maxx + pad, maxy + pad)
            src = sources[row["image_id"]]
            win = from_bounds(*req, transform=src.transform).round_offsets().round_lengths()
            win = win.intersection(Window(0, 0, src.width, src.height))
            arr = src.read([1, 2, 3], window=win, out_shape=(3, 384, 384))
            wt = src.window_transform(win)
            ax.imshow(stretch_rgb(arr), origin="upper")
            sx = 384 / win.width; sy = 384 / win.height
            # Plot every GT intersecting the requested map window.
            map_window = box(*rasterio.windows.bounds(Window(0, 0, win.width, win.height), wt))
            for j in by_image[row["image_id"]]:
                if not geoms[j].intersects(map_window):
                    continue
                color = "yellow" if j == idx else "#00ffff"
                lw = 1.8 if j == idx else .55
                for xy in iter_exteriors(geoms[j]):
                    px = np.asarray([~wt * (x, y) for x, y in xy])
                    ax.add_patch(MplPolygon(np.c_[px[:, 0] * sx, px[:, 1] * sy], fill=False, edgecolor=color, linewidth=lw))
            label = row["health_status"] if dataset == "Quebec" else row["source_label"]
            ax.set_title(f"{stratum} | {row['site']} | {label}\n{row['instance_id']}", fontsize=7)
            ax.axis("off")
    fig.suptitle(f"{dataset}: native RGB + GT (cyan), sampled crown (yellow)", fontsize=15)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", "bam", "quebec", "bci"], default="all")
    parser.add_argument("--no-montage", action="store_true")
    args = parser.parse_args()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    QC.mkdir(parents=True, exist_ok=True)
    make_montage = not args.no_montage
    results = {}
    if args.dataset in ("all", "bam"):
        results["bam"] = audit_bam(make_montage)
    if args.dataset in ("all", "quebec"):
        results["quebec"] = audit_quebec(make_montage)
    if args.dataset in ("all", "bci"):
        results["bci"] = audit_bci(make_montage)
    print(json.dumps({k: {"images": v["images_loaded"], "crowns": v["n_polygons"]} for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
