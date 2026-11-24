"""
tests/test_large_scale_v2.py
============================
Unit tests for the pieces of the v2 large-scale pipeline that the reported numbers depend on:
tile-grid coverage, one-owner stitching, and the evaluator's split of false positives into
intra-crown fragments vs outside-label candidates.

Run with:
    python -m pytest tests/test_large_scale_v2.py -v
or, without pytest installed:
    python tests/test_large_scale_v2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from crown_segmentation_research.experiments.large_scale_v2.eval_polygons import evaluate_polygons
from crown_segmentation_research.experiments.large_scale_v2.run_large_scale_v2 import stitch, tile_starts


def test_tile_starts_cover_aoi():
    for size, tile, overlap in [(10880, 1024, 128), (10880, 1024, 512), (5632, 1024, 512), (1024, 1024, 512)]:
        s = tile_starts(size, tile, overlap)
        assert s[0] == 0
        assert s[-1] + tile == size, "last tile must end exactly at the AOI edge"
        assert all(b - a <= tile - overlap for a, b in zip(s, s[1:])), "no gaps between tiles"


def test_owner_stitching_keeps_each_crown_exactly_once():
    rng = np.random.default_rng(0)
    size, tile, overlap = 5632, 1024, 512
    starts = tile_starts(size, tile, overlap)
    centres = rng.uniform(0, size, size=(300, 2))
    tiles = {(r, c): [] for r in starts for c in starts}
    for i, (gy, gx) in enumerate(centres):
        # a crown appears in every tile whose extent contains its centroid, as it would after decoding
        for r in starts:
            for c in starts:
                if r <= gy < r + tile and c <= gx < c + tile:
                    poly = box(gx - 5, gy - 5, gx + 5, gy + 5)
                    tiles[(r, c)].append({"wkb": poly.wkb, "score": 1.0 - i * 1e-4, "cy": gy - r, "cx": gx - c})
    kept = stitch(tiles, starts, starts, tile, "owner")
    assert len(kept) == len(centres), f"expected {len(centres)} crowns, kept {len(kept)}"


def test_eval_splits_false_positives_by_location():
    gt = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10)])
    pred = gpd.GeoDataFrame({"score": [0.9, 0.5, 0.5]},
                            geometry=[box(0, 0, 10, 10),     # matches the GT crown
                                      box(2, 2, 4, 4),       # fragment inside the labelled crown
                                      box(20, 20, 24, 24)])  # outside every label
    m = evaluate_polygons(pred, gt)
    assert m["R50"] == 1.0
    assert abs(m["P50"] - 1 / 3) < 1e-9
    assert m["FP_intra_crown_fragment"] == 1
    assert m["FP_outside_label"] == 1
    assert m["gt_with_ge2_preds_share"] == 1.0  # the GT crown holds two prediction centroids


def test_tree_outcomes_categories():
    from crown_segmentation_research.experiments.large_scale_v2.eval_polygons import tree_outcomes
    # IoU exactly 0.5 counts as a match (COCO rule), so pieces/merges are sized to stay clearly below it.
    gt = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10),     # delineated exactly
                                    box(20, 0, 30, 10),    # shredded into three pieces (IoU 0.3/0.3/0.4)
                                    box(40, 0, 50, 10),    # swallowed together with the next tree
                                    box(50, 0, 60, 10),
                                    box(80, 0, 90, 10),    # nothing predicted
                                    box(100, 0, 110, 10)])  # matched (IoU 0.8) but with a fragment inside
    pred = gpd.GeoDataFrame({"score": [0.9, 0.8, 0.8, 0.8, 0.7, 0.9, 0.5]},
                            geometry=[box(0, 0, 10, 10), box(20, 0, 23, 10), box(23, 0, 26, 10), box(26, 0, 30, 10),
                                      box(40, 0, 62, 10), box(100, 0, 108, 10), box(108, 0, 110, 10)])
    o = tree_outcomes(pred, gt)
    got = tuple(o[f"n_{k}"] for k in ("one_to_one", "matched_plus_fragments", "split", "merged", "missed"))
    assert got == (1, 1, 1, 2, 1), got


def test_pinched_contour_is_repaired_not_dropped():
    # Two blocks touching at one diagonal pixel: OpenCV returns a single outer contour that self-touches,
    # which shapely marks invalid. Before 2026-09-28 such crowns were silently dropped.
    from crown_segmentation_research.experiments.large_scale_v2.run_large_scale_v2 import _mask_to_polys
    mask = np.zeros((60, 60), np.uint8)
    mask[5:25, 5:25] = 1
    mask[25:45, 25:45] = 1
    polys = _mask_to_polys(mask, 0, 0)
    assert polys, "pinched crown was dropped"
    assert sum(p.area for p in polys) > 600  # two 20x20 blocks -> ~2 x 19^2 contour area


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
