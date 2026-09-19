#!/usr/bin/env python3
"""Download the public DTE-aerial-bench gallery assets and build eval masks.

The official deadtrees.earth release page currently marks the ZIP download as
"Coming soon", but publishes all 25 x 21 benchmark RGB patches and their two
binary reference-mask layers as immutable PNG assets. This script follows the
URL construction used by the official, content-hashed frontend bundle, records
provenance, and combines the layers into the three-class semantic mask expected
by the official evaluator:

    0 = background, 1 = tree cover, 2 = mortality

Mortality takes precedence where the two public binary layers overlap, matching
the official release-page renderer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


BUNDLE_URL = "https://deadtrees.earth/assets/index-BTPWok27.js"
BUNDLE_SHA256 = "5eab7ad857959d0ac224df676dab159503338c03f1357c26a4385df90c192027"
ASSET_BASE = (
    "https://data2.deadtrees.earth/reference/"
    "69e57f93-d003-4b64-9108-fe3dfa654918"
)
EXPECTED_SITES = 25
EXPECTED_PATCHES_PER_SITE = 21
EXPECTED_PATCHES = EXPECTED_SITES * EXPECTED_PATCHES_PER_SITE
EXPECTED_RESOLUTION_COUNTS = {5: 400, 10: 100, 20: 25}


@dataclass(frozen=True)
class Site:
    id: int
    file_name: str
    biome: str
    license: str
    citation_url: str | None
    export_seed: str
    asset_version: str | None
    longitude: float
    latitude: float
    patch_count: int


@dataclass(frozen=True)
class Patch:
    site: Site
    resolution_cm: int
    patch_index: int
    stem: str
    version: str
    rgb_url: str
    forest_url: str
    mortality_url: str


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_bytes(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DTE-aerial-reproducibility/1.0"},
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
            time.sleep(2 ** (attempt - 1))
    raise RuntimeError("unreachable")


def parse_sites(bundle: bytes) -> list[Site]:
    text = bundle.decode("utf-8")
    start_marker = "_S={slug:`dte-aerial-bench`"
    end_marker = "]}},vS=[gS,_S]"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise RuntimeError("Could not isolate DTE-aerial-bench release metadata")
    release = text[start:end]

    pattern = re.compile(
        r"\{id:(?P<id>\d+),"
        r"fileName:`(?P<file_name>[^`]*)`,"
        r"biome:`(?P<biome>[^`]*)`,"
        r"license:`(?P<license>[^`]*)`,"
        r"citationUrl:(?P<citation>null|`[^`]*`),"
        r"thumbnailPath:`[^`]*`,"
        r"exportSeed:`(?P<export_seed>\d+)`"
        r"(?:,assetVersion:`(?P<asset_version>[^`]*)`)?[,]"
        r"center:\{lon:(?P<lon>-?[0-9.]+),lat:(?P<lat>-?[0-9.]+)\},"
        r"patchCount:(?P<patch_count>\d+)\}"
    )

    sites: list[Site] = []
    for match in pattern.finditer(release):
        citation = match.group("citation")
        sites.append(
            Site(
                id=int(match.group("id")),
                file_name=match.group("file_name"),
                biome=match.group("biome"),
                license=match.group("license"),
                citation_url=None if citation == "null" else citation.strip("`"),
                export_seed=match.group("export_seed"),
                asset_version=match.group("asset_version"),
                longitude=float(match.group("lon")),
                latitude=float(match.group("lat")),
                patch_count=int(match.group("patch_count")),
            )
        )

    if len(sites) != EXPECTED_SITES:
        raise RuntimeError(f"Expected {EXPECTED_SITES} sites, parsed {len(sites)}")
    if len({site.id for site in sites}) != EXPECTED_SITES:
        raise RuntimeError("Duplicate benchmark site IDs in release metadata")
    if any(site.patch_count != EXPECTED_PATCHES_PER_SITE for site in sites):
        raise RuntimeError("Unexpected per-site benchmark patch count")
    return sites


def patch_stem(site: Site, resolution_cm: int, index: int) -> str:
    if resolution_cm == 20:
        return f"{site.id}_20_{site.export_seed}_20cm"
    if resolution_cm == 10:
        return f"{site.id}_{site.export_seed}_{index}_10cm"
    if resolution_cm == 5:
        return f"{site.id}_{index // 4}_{index % 4}_5cm"
    raise ValueError(f"Unsupported resolution: {resolution_cm}")


def iter_patches(sites: Iterable[Site]) -> Iterable[Patch]:
    counts = {5: 16, 10: 4, 20: 1}
    for site in sites:
        version = site.asset_version or site.export_seed
        for resolution_cm, count in counts.items():
            for index in range(count):
                stem = patch_stem(site, resolution_cm, index)
                base = f"{ASSET_BASE}/{site.id}/png/{stem}"
                suffix = f"?v={version}"
                yield Patch(
                    site=site,
                    resolution_cm=resolution_cm,
                    patch_index=index,
                    stem=stem,
                    version=version,
                    rgb_url=f"{base}.png{suffix}",
                    forest_url=f"{base}_forestcover_ref.png{suffix}",
                    mortality_url=f"{base}_deadwood_ref.png{suffix}",
                )


def is_valid_png(path: Path, mode: str) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.load()
            return image.size == (1024, 1024) and image.mode == mode
    except Exception:
        return False


def download_one(url: str, path: Path, expected_mode: str) -> tuple[Path, bool]:
    if is_valid_png(path, expected_mode):
        return path, False
    data = fetch_bytes(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_bytes(data)
    if not is_valid_png(part, expected_mode):
        part.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded asset is not a 1024x1024 {expected_mode} PNG: {url}")
    os.replace(part, path)
    return path, True


def build_semantic_mask(forest_path: Path, mortality_path: Path, output_path: Path) -> dict:
    forest = np.asarray(Image.open(forest_path).convert("L")) > 127
    mortality = np.asarray(Image.open(mortality_path).convert("L")) > 127
    mask = np.zeros(forest.shape, dtype=np.uint8)
    mask[forest] = 1
    mask[mortality] = 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask, mode="L").save(output_path, optimize=True)
    return {
        "background_pixels": int((mask == 0).sum()),
        "tree_cover_pixels": int((mask == 1).sum()),
        "mortality_pixels": int((mask == 2).sum()),
        "source_layer_overlap_pixels": int(np.logical_and(forest, mortality).sum()),
    }


def relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("DTE-Aerial-Data-public"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--bundle-url", default=BUNDLE_URL)
    args = parser.parse_args()

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance_dir = output / "provenance"
    provenance_dir.mkdir(exist_ok=True)

    bundle = fetch_bytes(args.bundle_url)
    bundle_digest = sha256_bytes(bundle)
    if args.bundle_url == BUNDLE_URL and bundle_digest != BUNDLE_SHA256:
        raise RuntimeError(
            "Official frontend bundle changed; inspect its release metadata before downloading"
        )
    (provenance_dir / "official_release_bundle.js").write_bytes(bundle)
    sites = parse_sites(bundle)
    patches = list(iter_patches(sites))
    if len(patches) != EXPECTED_PATCHES:
        raise RuntimeError(f"Expected {EXPECTED_PATCHES} patches, built {len(patches)}")

    with (provenance_dir / "release_sites.json").open("w", encoding="utf-8") as handle:
        json.dump([asdict(site) for site in sites], handle, indent=2, ensure_ascii=False)

    jobs: list[tuple[str, Path, str]] = []
    for patch in patches:
        jobs.extend(
            [
                (patch.rgb_url, output / "tiles" / f"{patch.stem}.png", "RGB"),
                (
                    patch.forest_url,
                    output / "source_masks" / "tree_cover" / f"{patch.stem}.png",
                    "L",
                ),
                (
                    patch.mortality_url,
                    output / "source_masks" / "mortality" / f"{patch.stem}.png",
                    "L",
                ),
            ]
        )

    completed = 0
    downloaded = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(download_one, url, path, mode): (url, path)
            for url, path, mode in jobs
        }
        for future in as_completed(futures):
            url, _ = futures[future]
            try:
                _, was_downloaded = future.result()
            except Exception as exc:
                print(f"FAILED {url}: {exc}", file=sys.stderr)
                return 2
            completed += 1
            downloaded += int(was_downloaded)
            if completed % 100 == 0 or completed == len(jobs):
                print(f"assets {completed}/{len(jobs)} (new {downloaded})", flush=True)

    rows: list[dict] = []
    totals = {
        "background_pixels": 0,
        "tree_cover_pixels": 0,
        "mortality_pixels": 0,
        "source_layer_overlap_pixels": 0,
    }
    resolution_counts = {5: 0, 10: 0, 20: 0}
    for patch in patches:
        rgb = output / "tiles" / f"{patch.stem}.png"
        forest = output / "source_masks" / "tree_cover" / f"{patch.stem}.png"
        mortality = output / "source_masks" / "mortality" / f"{patch.stem}.png"
        semantic = output / "masks" / f"{patch.stem}.png"
        stats = build_semantic_mask(forest, mortality, semantic)
        for key, value in stats.items():
            totals[key] += value
        resolution_counts[patch.resolution_cm] += 1
        rows.append(
            {
                "id": patch.site.id,
                "site_id": patch.site.id,
                "tile_path": relative(rgb, output),
                "mask_path": relative(semantic, output),
                "tree_cover_mask_path": relative(forest, output),
                "mortality_mask_path": relative(mortality, output),
                "biome": patch.site.biome,
                "resolution": f"{patch.resolution_cm}cm",
                "resolution_cm": patch.resolution_cm,
                "gsd_m": patch.resolution_cm / 100.0,
                "patch_index": patch.patch_index,
                "patch_stem": patch.stem,
                "source_file_name": patch.site.file_name,
                "license": patch.site.license,
                "citation_url": patch.site.citation_url or "",
                "longitude": patch.site.longitude,
                "latitude": patch.site.latitude,
                "rgb_url": patch.rgb_url,
                "tree_cover_mask_url": patch.forest_url,
                "mortality_mask_url": patch.mortality_url,
                **stats,
            }
        )

    if resolution_counts != EXPECTED_RESOLUTION_COUNTS:
        raise RuntimeError(f"Unexpected resolution counts: {resolution_counts}")

    meta_path = output / "DTE-aerial-bench-meta-public-assets.csv"
    with meta_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checksum_path = output / "SHA256SUMS"
    checksum_targets = sorted(
        path
        for directory in [output / "tiles", output / "source_masks", output / "masks"]
        for path in directory.rglob("*.png")
    )
    checksum_targets.extend([meta_path, provenance_dir / "release_sites.json"])
    with checksum_path.open("w", encoding="utf-8") as handle:
        for path in checksum_targets:
            handle.write(f"{sha256_file(path)}  {relative(path, output)}\n")

    report = {
        "status": "PASS",
        "source_release_page": "https://deadtrees.earth/releases/dte-aerial-bench",
        "source_bundle_url": args.bundle_url,
        "source_bundle_sha256": bundle_digest,
        "asset_base": ASSET_BASE,
        "site_count": len(sites),
        "patch_count": len(patches),
        "resolution_counts": {str(k): v for k, v in resolution_counts.items()},
        "asset_file_count": len(jobs),
        "semantic_mask_count": len(patches),
        "class_and_overlap_pixel_totals": totals,
        "semantic_rule": "background=0; forest>127 => 1; mortality>127 => 2 (override)",
        "metadata_csv": relative(meta_path, output),
        "checksums": relative(checksum_path, output),
    }
    with (output / "qc_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print(f"output_bytes={sum(p.stat().st_size for p in output.rglob('*') if p.is_file())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
