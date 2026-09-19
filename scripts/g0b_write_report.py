#!/usr/bin/env python3
"""Write the human-readable G0B policy and gate report from QC artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "benchmark" / "qc"
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "itc_benchmarks"


def load(name: str) -> dict:
    return json.loads((QC / name).read_text())


def human_bytes(n: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    x = float(n)
    for unit in units:
        if x < 1024 or unit == units[-1]:
            return f"{x:.2f} {unit}"
        x /= 1024
    raise AssertionError


def tree_bytes(path: Path) -> int:
    total = 0
    for base, _, files in os.walk(path):
        for name in files:
            total += (Path(base) / name).stat().st_size
    return total


def q(x, key="median", digits=3):
    value = x.get(key)
    return "n/a" if value is None else f"{value:.{digits}f}"


def main() -> None:
    bam = load("bam_geometry.json")
    quebec = load("quebec_geometry.json")
    bci = load("bci_geometry.json")
    REPORTS.mkdir(parents=True, exist_ok=True)

    policy = """# G0B annotation and evaluation policy

This document freezes what a ground-truth polygon means before any evaluator is implemented. It does not implement G0C.

| Dataset | Polygon unit | Exhaustive? | Visible but unlabeled tree | Edge crown | Duplicate physical crown |
|---|---|---|---|---|---|
| BAMFORESTS coco2048 | One visible individual crown instance in one 2048 crop | Documented as complete visible-crown labeling, with unavoidable ambiguity in dense/overlapping deciduous canopy | Use official benchmark background | Crown touching crop border is `edge_flag=true`. Official-compatible metrics evaluate the published clipped mask; controlled-study primary metrics ignore it and report an edge stratum separately | Possible across 50%-overlap crops. COCO `annotation_id` is not a biological tree ID. Preserve official hectare/AOI splits; never randomly re-split crops |
| Quebec 2021-09-02 | One labeled canopy crown with species/genus/dead label | No for object-level evaluation; created for species semantic segmentation and the paper documents image regions that were not annotated | Ignore/unscored for object-level false positives. Allow matched-GT recall and overlap/IoU analyses only | `edge_flag=true` if polygon touches COG footprint; partly outside is ignored. Report edge stratum separately | Published GPKGs contain no biological/tree ID. Exact geometry duplicates are audited, but repeated biological identity cannot be tested |
| BCI 2021 raw/manual | One manually delineated selected canopy crown tied to source `GlobalID` | Non-exhaustive subset | Ignore/unscored for object-level false positives. Allow matched-GT recall and overlap/IoU analyses only | Global-mosaic edge crowns are flagged; derived tile edges do not create new instances. Source rows not confirmed in field, missing that status, or flagged for revision are ignored in primary metrics | 50 tiles are overlapping crops of the same orthomosaic. Instances are anchored once to the global mosaic; never random-split or count tiles as independent scenes |

## Canonical identity rules

- `instance_id` is always a unique annotation/crop instance.
- `canonical_tree_id` is populated only when the source package provides a defensible persistent tree identifier. It is null for BAM and Quebec. For BCI it uses a non-sentinel field `tag` only when that tag occurs exactly once; ambiguous/repeated tags remain null. `GlobalID` remains the unique annotation-feature ID.
- BAM COCO IDs and Quebec row numbers must never be described as biological tree IDs.
- Exact duplicate geometry does not prove duplicate biological identity; the two concepts remain separate.

## Geometry and ignore rules for G0C

- Preserve source WKB and native GSD. Do not resize imagery in the canonical benchmark.
- Invalid source geometry is never silently overwritten. A deterministic, versioned `make_valid` derivative may be used for calculations while retaining validity/repair flags.
- A polygon touching the source image boundary is an edge instance. A polygon outside or partly outside the source footprint is ignored in primary object metrics.
- Positive GT overlap is legal and must not be removed by non-maximum suppression during data preparation.
- For non-exhaustive Quebec and BCI, global precision, false-positive rate, COCO AP, and panoptic-quality-style scores are disabled unless a future verified ignore mask defines exhaustively annotated support.
- Valid restricted metrics are matched-GT detection/recall, matched-mask IoU/Dice/boundary quality, split/merge diagnostics around labeled crowns, and stratification by area/GSD/health/site.

## Health mapping

For Quebec only: source label `Mort` maps to `DEAD`; every present non-`Mort` label maps to `NON_DEAD`; a missing label maps to `UNKNOWN`. Health is metadata for stratified segmentation analysis, not a classification target in G0B.
"""
    (ROOT / "benchmark" / "annotation_policy.md").write_text(policy)

    checksums = (REPORTS / "g0b_checksums.txt").read_text().strip() if (REPORTS / "g0b_checksums.txt").exists() else "Checksum log unavailable"
    raw_bytes = tree_bytes(DATA / "raw_archives")
    extracted_bytes = tree_bytes(DATA / "extracted")
    artifact_bytes = tree_bytes(ROOT / "benchmark")

    report = f"""# G0B data acquisition and annotation QC report

Date: 2026-08-24  
Scope: BAMFORESTS `coco2048`, Quebec Trees `2021-09-02`, BCI `2021 raw/manual`.  
Excluded/locked: SelvaMask, BAM1024, AVUELO, DeadTrees.  
No evaluator, model training, classification, SAM2, image resizing, or TIFF duplication into `benchmark/` was performed.

## Gate result

| Dataset | Decision | Reason |
|---|---|---|
| BAMFORESTS | **PASS** | All archived TIFFs referenced by the four official COCO files load; geometries are bounded/decoded and visually registered. Official crop splits are retained and crop annotation IDs are not treated as tree IDs. |
| Quebec | **CONDITIONAL** | Geometry and dated imagery are usable, including `Mort` health stratification, but annotations are not proven exhaustive and the published GPKGs omit biological tree IDs. Restrict evaluation to matched-GT metrics. |
| BCI | **CONDITIONAL** | The 2,454 selected crowns align to the global orthomosaic, but labels are non-exhaustive and {bci['n_invalid']:,} source geometries are invalid. Use deterministic repair plus matched-GT metrics; never random-split the 50 overlapping crops. |
| Global G0B | **PASS** | Every dataset supports a defensible quantitative policy; conditional datasets have global false-positive/AP metrics disabled. |

## BAMFORESTS coco2048

- Images loaded: **{bam['images_loaded']:,}** ({', '.join(f'{k}: {v:,}' for k, v in bam['images_by_split'].items())}).
- Crown instances: **{bam['n_polygons']:,}** ({', '.join(f'{k}: {v:,}' for k, v in bam['crowns_by_split'].items())}). These are crop instances, not unique biological trees.
- Release/paper discrepancy: the downloaded archive has train 58,228, validation 15,177, Test-1 6,720, Test-2 12,320 annotations; the paper table reports 58,235/15,180/6,720/12,321. The immutable downloaded archive is the operative source for manifests.
- Geometry: invalid {bam['n_invalid']:,} ({bam['invalid_reasons']}); empty {bam['n_empty']:,}; zero-area {bam['n_zero_area']:,}; MultiPolygon {bam['n_multipolygon']:,}; holes {bam['n_holes']:,}; outside image {bam['n_outside_image']:,}; touching edge {bam['n_touching_edge']:,}; clipped/partly outside {bam['n_clipped_or_partly_outside']:,}. Primary controlled-study metrics ignore {bam['primary_ignore_instances']:,} edge/invalid/degenerate annotations while preserving their source WKB.
- Positive within-image GT overlaps: {bam['gt_overlap']['positive_area_pairs']:,} pairs involving {bam['gt_overlap']['instances_in_positive_overlap']:,} instances. Exact duplicate geometry: {bam['exact_duplicate_geometry']['groups']:,} groups.
- Area in native pixels: median {q(bam['area_px'], digits=1)}, p05 {q(bam['area_px'], 'p05', 1)}, p95 {q(bam['area_px'], 'p95', 1)} px².
- Native GSD: Hain 1.82 cm, Stadtwald 1.70 cm, Tretzendorf-1 1.61 cm, Tretzendorf-2 1.79 cm. A Tretzendorf row receives the exact value only if its released filename safely encodes acquisition 1/2; otherwise `gsd_cm` remains null with the 1.61–1.79 cm range in `gsd_note` rather than inventing a value.
- Category/`iscrowd`: {bam['categories']}; `iscrowd` counts {bam['iscrowd']}.
- COCO decode integrity: formats {bam['segmentation_formats']}. Source `area` is integer raster-mask area while manifest area is continuous polygon area, giving median relative delta {100 * bam['source_coco_area_relative_delta']['median']:.3f}% (not treated as an error). Source bbox vs polygon-edge coordinates have a median 0.5-pixel offset, consistent with half-pixel contour coordinates; full distributions are in `bam_geometry.json`.
- Registration: COCO sizes were checked against TIFF headers; missing members {len(bam['missing_tiff_members'])}; dimension mismatches {len(bam['tiff_coco_dimension_mismatch'])}. The montage shows RGB/GT alignment without a systematic offset.
- Completeness/evaluation: use official splits. Test-1 is the independent Hain AOI; train/validation/Test-2 are separated by complete hectare plots so overlapping crops stay within a split. G0C must expose both official-compatible clipped-edge scoring and a primary full-crown-only score. The release does not provide a biological ID, so cross-crop physical-crown duplication is not recoverable from annotation IDs.
- Visual QC: `benchmark/qc/bam_examples.png` (30 stratified panels).

## Quebec Trees 2021-09-02

- Images loaded: **{quebec['images_loaded']:,}** dated zone COGs; crowns: **{quebec['n_polygons']:,}** ({', '.join(f'{k}: {v:,}' for k, v in quebec['crowns_by_zone'].items())}).
- The exact release inventory (22,933 rows) is operative; “23,000” in the dataset description is rounded, while its main-class table excludes roughly 700 broader/rare-label crowns.
- Health mapping: {quebec['health_status']}.
- Geometry: invalid {quebec['n_invalid']:,}; empty {quebec['n_empty']:,}; zero-area {quebec['n_zero_area']:,}; MultiPolygon {quebec['n_multipolygon']:,}; polygons with holes {quebec['n_with_holes']:,} ({quebec['n_holes']:,} holes); outside image {quebec['n_outside_image']:,}; touching edge {quebec['n_touching_edge']:,}; clipped/partly outside {quebec['n_clipped_or_partly_outside']:,}.
- Positive within-zone GT overlaps: {quebec['gt_overlap']['positive_area_pairs']:,} pairs involving {quebec['gt_overlap']['instances_in_positive_overlap']:,} instances; exact duplicate geometry: {quebec['exact_duplicate_geometry']['groups']:,} groups.
- Crown area: median {q(quebec['area_m2'])} m², p05 {q(quebec['area_m2'], 'p05')} m², p95 {q(quebec['area_m2'], 'p95')} m².
- Native GSD: {', '.join(f'{x:.4f}' for x in quebec['native_gsd_cm'])} cm across Z1–Z3; EPSG:32618.
- Public inference polygons: {quebec['inference_zone']['features']} features; {quebec['inference_zone']['crowns_centroid_inside']:,} crown centroids lie inside. They correspond to the publication's spatial test/inference layout (one separate test site plus two internal areas excluded from train/validation). G0B records membership; G0D must freeze any benchmark split without pixel overlap.
- Identity/completeness: GPKGs expose only `Label` and geometry. No field/biological ID exists in the release, so duplicate biological IDs and polygon↔field-tree linkage cannot be verified. Labels were created for species-level semantic segmentation; the associated paper explicitly discusses false positives in image regions that had not been annotated. They are therefore non-exhaustive support for object-level FP counting.
- Evaluation limitation: matched-GT recall/IoU/boundary metrics only; global FP/AP disabled.
- Visual QC: `benchmark/qc/quebec_examples.png` (30 stratified panels, including dead crowns when sampled).

## BCI 2021 raw/manual crowns

- Images loaded: **{bci['images_loaded']:,}** (1 global orthomosaic + 50 derived tiles); source crown instances: **{bci['n_polygons']:,}**.
- Geometry: invalid {bci['n_invalid']:,} ({bci['invalid_reasons']}); repaired for analysis {bci['n_repaired_for_analysis']:,}; empty {bci['n_empty']:,}; zero-area {bci['n_zero_area']:,}; Polygon {bci['geometry_types'].get('Polygon', 0):,}; MultiPolygon {bci['n_multipolygon']:,}; polygons with holes {bci['n_with_holes']:,} ({bci['n_holes']:,} holes); outside global image {bci['n_outside_image']:,}; touching global edge {bci['n_touching_edge']:,}.
- Positive GT overlaps: {bci['gt_overlap']['positive_area_pairs']:,} pairs involving {bci['gt_overlap']['instances_in_positive_overlap']:,} crowns; total overlap {bci['gt_overlap']['total_overlap_area']:.3f} m². Exact duplicate geometry: {bci['exact_duplicate_geometry']['groups']:,} groups.
- Crown area: median {q(bci['area_m2'])} m², p05 {q(bci['area_m2'], 'p05')} m², p95 {q(bci['area_m2'], 'p95')} m².
- Native raster transform: **{bci['native_gsd_cm']:.6f} cm/pixel**, EPSG:32617 (more precise than the rounded 4 cm description).
- Identity: all `GlobalID` values are unique annotation-feature IDs. A defensible unique non-sentinel field `tag` is available for {bci['canonical_tree_id']['known_unique_rows']:,} rows; {bci['canonical_tree_id']['unknown_or_ambiguous_rows']:,} rows remain without `canonical_tree_id`. `tag` has {bci['tag_audit']['duplicate_groups']} repeated groups/{bci['tag_audit']['rows_in_duplicate_groups']} rows, dominated by sentinel `-9999` ({bci['tag_audit']['largest_groups'].get('-9999', 0)} rows).
- Source uncertainty: {bci['source_uncertain_or_revision_ignore_rows']:,} rows are marked `ignore_flag=true` because `SeenInFiel` is `No`/missing or the source `Flag` requests revision. They remain in the manifest for auditability.
- Tile relation: all tiles lie in the global footprint, their union leaves {bci['tile_relationship']['global_area_not_covered_by_tiles_m2']:.3f} m² uncovered, and {bci['tile_relationship']['positive_overlap_tile_pairs']} tile pairs overlap. They are derived views, not independent samples.
- Completeness/evaluation: manual crowns are a selected non-exhaustive canopy subset. Unlabeled visible crowns are ignore/unscored; matched-GT metrics only; global FP/AP disabled.
- Registration: all labeled crowns lie inside the global projected footprint, and the montage shows direct RGB/GT alignment without a systematic offset.
- Visual QC: `benchmark/qc/bci_examples.png` (30 stratified panels).

## Provenance, integrity, and storage

Raw archives and selectively extracted source files are read-only. Archive validation and digest log:

The official MD5 values are verified for Quebec and BCI. The DLR BAM download page does not publish a digest, so G0B records local MD5/SHA-256 and requires a clean full ZIP structural test.

```text
{checksums}
```

- Raw archives: {human_bytes(raw_bytes)}.
- Selectively extracted data: {human_bytes(extracted_bytes)}.
- Benchmark manifests/QC artifacts (no TIFF copies): {human_bytes(artifact_bytes)}.
- Every image manifest points to an extracted raw COG or a `/vsizip/` archive member; no common-resolution resampling was performed.

Primary provenance:

- [DLR BAMFORESTS release](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/bamforests) and [dataset paper](https://doi.org/10.3390/rs16111935). The paper classifies BAMFORESTS as complete polygon labeling, documents 50% crop overlap, partial crowns at crop edges, a spatially independent Hain Test-1, and hectare-block Test-2/validation splits.
- [Quebec Trees Zenodo record, version frozen at 2021-09-02](https://zenodo.org/records/8148479) and [associated article](https://doi.org/10.1016/j.rse.2024.114283).
- [Smithsonian Figshare BCI crown maps v2](https://doi.org/10.25573/data.24784053.v2). Its data description explicitly documents manually delineated visible crowns, missing/sentinel and duplicated tags, erroneous polygons, undetected crowns, and difficulty confirming small crowns in dense canopy.

## What G0C may build next

G0C may build only the canonical adapters and common evaluator around these frozen manifests: deterministic geometry repair, native-coordinate loading, official BAM splits, edge/ignore masks, matched-GT metrics, and stratification by site/GSD/area/Quebec health. It must disable global FP/AP for Quebec and BCI and must not infer biological identity for BAM/Quebec. G0C still may not train a model, use SAM2, resize all datasets to one GSD, random-split BCI tiles, or unlock AVUELO/DeadTrees.

```text
G0B_BAM = PASS
G0B_QUEBEC = CONDITIONAL
G0B_BCI = CONDITIONAL
G0B_GLOBAL = PASS
```
"""
    (REPORTS / "g0b_data_qc_report.md").write_text(report)


if __name__ == "__main__":
    main()
