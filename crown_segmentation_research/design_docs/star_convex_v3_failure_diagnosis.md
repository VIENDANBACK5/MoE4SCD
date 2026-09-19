# Star-convex v3 — qualitative failure diagnosis (misses + false positives)

## Material Passport

- Origin: `star_convex_v3_training_and_merge_result.md` concluded the merge
  module gives a null result and the real gap is miss_rate (24.3%) and
  precision (43.4%). Rather than guess at a new architecture, this
  diagnoses *which* GT crowns are missed and *where* false positives come
  from, per this project's own diagnose-before-design convention.
- Method: reused the canonical `benchmark.evaluator.matching.InstanceMatcher`
  directly (same matcher the evaluator uses internally) on all 50 BAM_val
  images with v3's adopted decode config (prob=0.5, peak-dist=8, nms=0.2),
  recording per-instance area/edge-distance/probability-signal for every
  miss, match, and false positive.
- Script: `code/diagnose_star_convex_failures.py`. Per-instance CSVs at
  `experiments/star_convex_screen_v3/failure_diagnosis/{misses,matches,false_positives}.csv`.
  Preview images: `images/star_convex_v3_failure_diagnosis_val203.png`
  (dense, high-miss image) and `_val23.png` (high-false-positive image).

## Result — quantitative

332 misses, 815 matches, 1075 false positives across 50 val images.

| | missed GT | matched GT |
|---|---:|---:|
| area_px (median) | 25,605 | 55,398 |
| edge_dist_px (median) | 360 | 431 |

| Missed-crown probability signal | value |
|---|---:|
| network probability AT the true centroid (median) | **0.000** |
| fraction of misses with centroid probability > 0.1 | 5% |
| max probability anywhere in the GT's own bbox (median) | 0.314 |
| fraction reaching the 0.5 decode threshold anywhere in bbox | 22% |

| False positives | value |
|---|---:|
| overlap some GT below the 0.5 IoU threshold (near-miss) | 30% |
| **no overlap with any GT at all (pure clutter)** | **70%** |
| FP area (median) | 45,231 px (matched-GT median: 55,398 px — similar scale, not noise-sized blobs) |

## Result — qualitative (preview images)

- **`val203` (dense, homogeneous pale-canopy stand, 65 GT / 44 missed)**:
  missed crowns are smaller sub-crowns tightly packed among visually
  near-identical neighbors — the true boundaries are faint even to the eye
  in places. The network's matched (green) detections skew toward the
  larger, more visually separated crowns; small touching crowns of the same
  apparent species/color are the ones it fails to individuate.
- **`val23` (mixed stand, 18 GT / 31 false positives)**: the yellow
  false-positive polygons are concentrated over the **darker, lower
  understory/matrix vegetation texture between real crowns**, not
  duplicated on top of real crown objects — visually consistent with the
  70%-pure-clutter statistic. The network appears to be firing its
  object-probability head on generic dark-green textured vegetation that is
  not an individual crown at all, rather than only on crown material.

## Interpretation

Two distinct, separable failure modes, not one:

1. **Instance-separation blind spot in dense/homogeneous stands.** Missed
   crowns are ~2.2x smaller by area and get essentially **zero** probability
   signal at their true center (not merely below-threshold — genuinely
   near-zero for 95% of misses). This is a real detection gap, not a decode
   hyperparameter problem — consistent with the earlier decode-sweep finding
   that no threshold in the 36-config grid recovered baseline recall
   (`star_convex_v2_decode_sweep.md`).
2. **Background/foreground confusion on non-crown texture.** 70% of false
   positives have zero overlap with any real crown; they are the network
   mistaking generic dark understory/matrix vegetation for crown material,
   not duplicate detections of crowns it already found (which would show up
   as `merge_rate`, currently only 0.037 — confirming FPs are a distinct
   phenomenon from the already-solved over-segmentation problem).

Edge-distance is a secondary factor at most (360 vs 431 px median — a real
but modest gap, not close to explaining 332 misses on its own).

## What this rules in / out for next steps

- **Rules against**: inventing a new geometric/shape representation. The
  crowns star-convex *does* detect are decoded as clean single-piece
  polygons (per the oracle test and the visual previews) — the failure is
  not in shape reconstruction, it's in *deciding where an instance center
  is at all* (misses) and *rejecting non-crown texture* (false positives).
  A different polygon representation would not obviously fix either.
- **Consistent with**: this being a training-signal problem the v2->v3
  data-scale-up was already chipping away at (recall +2pp, split_rate -37%
  relative just from 300->536 images). Two concrete, low-risk next levers
  that directly target the two diagnosed failure modes, rather than a
  ground-up redesign:
  - More training data/epochs (continues the trend already measured).
  - A loss term or sampling change that explicitly penalizes high
    probability on non-crown background (addresses the clutter-FP mode)
    and/or sharpens probability contrast between touching same-appearance
    instances (addresses the dense-stand miss mode) — e.g. hard-negative
    mining on background patches, or a boundary/edge-aware loss term
    (this is the specific gap the Omnipose/MultiStar/HSD literature review
    flagged as open and unaddressed by existing published work, per the
    session's earlier literature synthesis).
