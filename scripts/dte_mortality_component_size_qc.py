"""QC check: how much of the DTE-aerial-bench mortality label is noise-floor-sized.

Motivated by a visual spot-check on 6 random tiles (seed=42) that showed dense
scattered mortality speckles on visually uniform canopy in 3/6 tiles. This
script re-checks that hypothesis across the *entire* 525-tile bench instead of
a handful of examples, using 8-connected-component size as the proxy for
"resolvable object" vs "noise floor".

Usage:
    python3 scripts/dte_mortality_component_size_qc.py

Output: prints the same table as reports/dte_aerial_mortality_component_size_qc.md.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

META_CSV = Path("DTE-Aerial-Data-public/DTE-aerial-bench-meta-public-assets.csv")
DATA_ROOT = Path("DTE-Aerial-Data-public")
SIZE_THRESHOLDS_M2 = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def run() -> dict:
    rows = list(csv.DictReader(open(META_CSV)))
    all_sizes_m2: list[float] = []
    by_res_sizes: dict[str, list[float]] = defaultdict(list)
    n_tiles_with_mortality = 0
    total_components = 0

    for row in rows:
        mask = np.array(Image.open(DATA_ROOT / row["mask_path"]))
        mortality = mask == 2
        if not mortality.any():
            continue
        n_tiles_with_mortality += 1
        labeled, n = ndimage.label(mortality)
        sizes_px = ndimage.sum(mortality, labeled, range(1, n + 1))
        gsd = float(row["gsd_m"])
        sizes_m2 = sizes_px * (gsd**2)
        all_sizes_m2.extend(sizes_m2.tolist())
        by_res_sizes[row["resolution_cm"]].extend(sizes_m2.tolist())
        total_components += n

    sizes = np.array(all_sizes_m2)
    total_area = sizes.sum()

    result = {
        "n_tiles_total": len(rows),
        "n_tiles_with_mortality": n_tiles_with_mortality,
        "total_components": total_components,
        "percentiles_m2": {
            p: float(np.percentile(sizes, p)) for p in (1, 5, 10, 25, 50, 75, 90, 95, 99)
        },
        "min_m2": float(sizes.min()),
        "max_m2": float(sizes.max()),
        "fraction_of_components_below_threshold": {
            t: float((sizes < t).mean()) for t in SIZE_THRESHOLDS_M2
        },
        "fraction_of_total_area_below_threshold": {
            t: float(sizes[sizes < t].sum() / total_area) for t in SIZE_THRESHOLDS_M2
        },
        "by_resolution_cm": {
            res: {
                "n_components": len(arr),
                "median_m2": float(np.median(arr)),
                "fraction_below_0.1m2": float((np.array(arr) < 0.1).mean()),
            }
            for res, arr in sorted(by_res_sizes.items(), key=lambda kv: int(kv[0]))
        },
    }
    return result


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
