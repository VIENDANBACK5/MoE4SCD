# Findings — ITC Crown Segmentation, Closing the Recognition-Quality Gap

*Last updated: 2026-09-19, after H0 and H1.*

## The question

The CanopyWatershed pipeline on BAMFORESTS was reported at mAP@50 = 31.3% with
SQ = 67.7% but RQ = 31.3%. A boundary-refinement effort (MorphSnakes, adaptive spline
smoothing, Dao ED objective) had moved it to 32.99%. Where is the error actually
located, and what is the highest-leverage intervention?

## Current understanding

**1. The error budget lives in marker selection, not in contours or in the boundary head.**
A 2x2 oracle ablation (H0) swapped seeds and energy independently. Oracle seeds with
the *predicted* boundary field: 31.3 -> **70.8** mAP@50. Oracle energy with predicted
seeds: 31.3 -> **29.9**. Marker selection owns ~39.5 of the ~40 recoverable points;
the learned boundary field owns approximately none of them.

**2. The mechanical cause was a decode constant that was ~6x too small.**
The pipeline emitted **1.90 seeds per true crown**. `min_distance = 24 px` was set
against crowns whose mean equivalent diameter in this dataset is **~146 px**. Raising
suppression to 160 px (and the apex threshold 0.25 -> 0.50) moves mAP@50 to 65.5 on
held-out data — 93% of the oracle ceiling — with no retraining and no new code.

**3. Global NMS is trapped on a precision/recall frontier that tuning cannot escape.**
Loose suppression finds **83.1%** of all crowns but plants 0.82 duplicate seeds inside
each one. Tight suppression drops duplicates to 0.04/crown but misses **37.5%** of
crowns. The oracle beats both because it is off the frontier entirely: 100% hit, zero
duplicates. Tuning slides along the frontier; it does not leave it.

**4. Boundary refinement was optimising a second-order term.**
MorphSnakes bought +1.66 mAP@50 against a 40-point marker problem. Its real value is
geometric, not recognitional: B-IoU 10.3 -> 12.9, Dao ED 0.613 -> 0.593. H0 cell C
confirms the boundary head *is* lossy — oracle seams raise SQ 67.7 -> 74.3 and B-IoU
10.3 -> 24.4 — but that loss is invisible while RQ is the binding constraint.

**5. The previously published evaluation subset is unrepresentatively hard.**
The same baseline scores 31.33 on the original 30 crops and 44.29 on 60 disjoint
crops. Every number previously reported on those 30 crops is pessimistic by roughly
13 points, so cross-method comparisons drawn from them need re-checking.

## Patterns and insights

- **`PQ = SQ x RQ` is the diagnostic that should have been read first.** SQ 68 / RQ 31
  said "masks fine, recognition broken" before a single experiment was run. Months of
  contour work went into the SQ term while RQ was the binding constraint.
- **Oracle ablations are cheap and decisive.** Four cells, one afternoon, and the
  entire research direction changed. They should precede any optimisation effort.
- **Per-instance target normalisation destroys scale.** `u_inst = mask_edt / max_dist`
  makes every crown peak at 1.0, so the surface head cannot signal crown size. Scale
  survives only in the *support* of the superlevel sets, and recovering it from there
  is too noisy to be useful (H1-b). If scale awareness is wanted, it has to be
  supervised — an explicit size/radius head, not a decode-time trick.
- **A grid whose optimum sits on its boundary has not found an optimum.** H1's locked
  grid reported 52.87 as "best"; the true plateau was at 66.15, far outside it.

## Lessons and constraints

- `min_distance` 144-160 px, `min_apex` 0.50, `min_canopy` 0.25, `min_area` 25 is the
  current best decode setting for BAMFORESTS at 1024 px crops. Do not use 24 / 0.25.
- `min_apex >= 0.60` collapses recall fast (ratio 0.62, mAP@50 -> 61.6). 0.50 is the knee.
- Superlevel-set radius estimation: `t = 0.7` shatters components (seed ratio 7.8).
  Unusable above `t = 0.5`. The whole approach is dropped.
- `evaluate_metrics` reports **F1 at a fixed IoU**, not interpolated average precision,
  while labelling it `ap50`. Internally consistent, but must be renamed or replaced
  with true COCO AP before publication.
- Tune on `eval30`, always re-score frozen settings on `evalhold60` (eval idx 100-159)
  before believing any delta.
- Cached head outputs (`crown_research/src/pred_cache.py`) turn a sweep from
  GPU-minutes into CPU-seconds. Use them for anything decode-side.

## Open questions

1. **Can over-seed + merge escape the frontier?** (H2, next.) Loose seeding's 83% hit
   rate with duplicates removed post-flood implies a ceiling well above the tuned 66.
2. What is the oracle ceiling on the *hold-out* split? H0 cell B (70.79) was measured
   only on the harder 30 crops, so the true headroom above 65.5 is not yet known.
3. Why does the pipeline miss 16.9% of crowns even under loose seeding? Suppressed
   peaks, or crowns the surface head never lights up at all?
4. Does the tuned setting hold across biomes, or is 160 px a Bamberg-specific scale?
   BAMFORESTS `test1`/`test2` and DTE-Aerial are available and untouched.
5. Now that RQ is no longer binding, is the boundary head's ~7-point SQ deficit (H0
   cell C) the next-largest term?
