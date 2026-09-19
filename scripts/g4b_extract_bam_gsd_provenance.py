"""Extract per-image BAMFORESTS GSD provenance from embedded GeoTIFF tags.

The released COCO filenames collapse both Tretzendorf AOIs to ``Tretzendorf``.
However, every crop retains ``ModelPixelScaleTag`` and ``ModelTiepointTag``.
This script reads those source tags without decoding the raster payload and
creates a fail-closed image-level mapping for scale-aware experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


AOI_BY_GSD_CM = {
    1.818260: "Hain",
    1.699800: "Stadtwald",
    1.611970: "Tretzendorf-1",
    1.793110: "Tretzendorf-2",
}

SOURCE_ARCHIVE_SHA256 = (
    "34ec28ca887c89d31d3a1191bbfde108c737788d34acb6656b64910793479a33"
)
AOI_LABEL_SOURCE = "DLR thesis elib.dlr.de/214261; Troles et al. 2024"


def nearest_aoi(gsd_cm: float, tolerance_cm: float = 1e-4) -> str:
    expected = min(AOI_BY_GSD_CM, key=lambda value: abs(value - gsd_cm))
    if abs(expected - gsd_cm) > tolerance_cm:
        raise ValueError(f"Unexpected BAM GSD {gsd_cm:.9f} cm/pixel")
    return AOI_BY_GSD_CM[expected]


def atomic_write_csv(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    temporary.replace(output)


def extract_mapping(manifest_path: Path, archive_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    required = {"image_id", "split", "archive_member"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Manifest image_id values must be unique")

    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        for row in manifest.itertuples(index=False):
            member = str(row.archive_member)
            if member not in members:
                raise FileNotFoundError(f"Archive member missing: {member}")
            with archive.open(member) as source:
                with Image.open(source) as image:
                    pixel_scale = image.tag_v2.get(33550)
                    tiepoint = image.tag_v2.get(33922)
            if pixel_scale is None or len(pixel_scale) < 2:
                raise ValueError(f"ModelPixelScaleTag missing: {member}")
            if tiepoint is None or len(tiepoint) < 6:
                raise ValueError(f"ModelTiepointTag missing: {member}")

            scale_x_m = float(pixel_scale[0])
            scale_y_m = float(pixel_scale[1])
            if not np.isclose(scale_x_m, scale_y_m, rtol=0, atol=1e-9):
                raise ValueError(f"Non-square source pixels: {member}")
            gsd_cm = scale_x_m * 100.0
            aoi = nearest_aoi(gsd_cm)
            rows.append(
                {
                    "image_id": str(row.image_id),
                    "split": str(row.split),
                    "aoi": aoi,
                    "gsd_cm": round(gsd_cm, 6),
                    "source": "GeoTIFF ModelPixelScaleTag",
                    "provenance": (
                        f"archive_sha256={SOURCE_ARCHIVE_SHA256}; "
                        f"member={member}; tag=33550; aoi_label={AOI_LABEL_SOURCE}"
                    ),
                    "archive_member": member,
                    "pixel_size_x_m": scale_x_m,
                    "pixel_size_y_m": scale_y_m,
                    "tiepoint_x_m": float(tiepoint[3]),
                    "tiepoint_y_m": float(tiepoint[4]),
                }
            )

    result = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if len(result) != len(manifest):
        raise RuntimeError("Not every manifest image received a GSD record")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmark/manifests/bam_images.csv"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark/manifests/bam_gsd_provenance.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("benchmark/qc/bam_gsd_provenance_summary.json"),
    )
    args = parser.parse_args()

    if args.output.exists() or args.summary.exists():
        raise FileExistsError("GSD provenance outputs already exist; refusing overwrite")
    mapping = extract_mapping(args.manifest, args.archive)
    atomic_write_csv(mapping, args.output)

    summary = {
        "schema_version": 1,
        "status": "PASS",
        "manifest": str(args.manifest),
        "archive": str(args.archive),
        "archive_sha256": SOURCE_ARCHIVE_SHA256,
        "n_images": int(len(mapping)),
        "n_missing": int(mapping["gsd_cm"].isna().sum()),
        "groups": [
            {
                "split": split,
                "aoi": aoi,
                "gsd_cm": float(gsd),
                "n_images": int(len(group)),
            }
            for (split, aoi, gsd), group in mapping.groupby(
                ["split", "aoi", "gsd_cm"], sort=True
            )
        ],
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.summary.with_suffix(args.summary.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
