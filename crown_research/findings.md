# Findings — ITC Crown Segmentation, Closing the Recognition-Quality Gap

*Last updated: 2026-09-19, after H0-H5. Report: crown_research/to_human/report.html*

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

**3b. But marker selection is now CLOSED, and it was not the whole story.**
Three structurally independent perfect-marker routes — oracle seeds (69.0), oracle
seeds plus oracle seams (68.6 on the tuning split), oracle merging (69.7) — all
converge at 69-70 on held-out data. No marker method can pass that. Meanwhile the same
unmodified decoder reaches **97.30 mAP@50 / 93.10 mAP@75 / PQ 89.69** when all three
heads are correct (H3). The decoder is not the limit; the heads are.

**3c. The heads are non-separable, and the boundary head is mis-calibrated.**
Correcting the canopy gate alone costs -24.3; correcting the boundary field alone costs
-32.5; correcting both gains +12.4. The boundary head scores **AUROC 0.866** against GT
seams but has median output **0.0011** and averages **0.105 on true seam pixels** — it
learned the ranking and lost its magnitude to the ~6% positive rate. Every use of it so
far (H2's absolute merge threshold, H4's `canopy - lambda*B`) was an absolute-magnitude
use, which is exactly what a collapsed head cannot support. A rank transform does *not*
fix it (-9.4): the ordering inside the near-zero bulk is noise.

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

**6. The crowns the pipeline misses are the small ones, and the surface head simply
never lights them.**
Under loose seeding 16.5% of GT crowns receive no seed; at the tuned operating point
that rises to 28.1%. Almost none of it is NMS suppression (0.1-2.2%). The dominant
cause is **`never_lit`**: the surface head never exceeds `min_apex` anywhere inside the
crown. Those crowns have a **median area of 1,899 px against 24,382 px for all crowns**
— below the 10th percentile. This is a second, independent consequence of the same
per-instance normalisation identified in H1: the target says every crown peaks at 1.0,
but after FPN smoothing a crown a tenth the typical size cannot produce a sharp
isolated peak, so it falls under the threshold.

**7. A scale-invariant apex target already exists in the codebase and is discarded.**
`targets_sharp.compute_sharp_canopy_targets` returns a fourth tensor: a Gaussian peak
at each crown centroid with a **fixed sigma regardless of crown size**. `dataset.py`
ships it to the training loop as `batch["apex"]` on every step. `train_sharp.py` never
reads it, and `CanopyWatershedNet` has only three output channels — so the one signal
in the project that is scale-invariant by construction is computed and thrown away.
That is exactly the signal finding 6 says is missing.

**8. The remaining headroom splits into two learnable deficits.**
From the H3 cube on the hold-out: predicted seeds with both other heads correct reaches
**77.96**; adding correct seeds reaches **97.30**.
- boundary + canopy quality: 65.5 -> 78.0, about **+12**
- seeding (i.e. the surface head's small-crown blindness): 78.0 -> 97.3, about **+19**

H6 addresses the first. The second needs the apex head.

**9. Under true COCO AP the conclusions hold, but the absolute numbers are much lower.**
The repo's `evaluate_metrics` reports F1 at a fixed IoU while calling it `ap50`.
Re-scored with real 101-point interpolated AP (`crown_research/src/coco_ap.py`),
detections pooled globally and confidence taken as the peak surface response inside
each instance:

| configuration (hold-out) | COCO AP | AP50 | AP75 | repo F1@50 |
|---|---:|---:|---:|---:|
| published (md=24, ma=0.25) | 15.81 | 40.34 | 9.48 | 44.29 |
| H1 tuned (md=160, ma=0.50) | 23.85 | 56.19 | 15.77 | 65.54 |
| H5 best (tuned + percentile merge) | **24.32** | **56.56** | 16.48 | 66.91 |
| H3 oracle heads | 84.48 | 96.00 | 91.85 | 97.30 |

The **ordering is unchanged**, so every conclusion in this document survives the
stricter metric. Two things change. First, the honest headline for the work so far is
**COCO AP 15.81 -> 24.32**, not 44 -> 67. Second, AP@[.5:.95] is far below AP50 (24.3
vs 56.6) while the oracle keeps 84.5 vs 96.0 — mask quality at strict IoU is a much
bigger deficit than F1@50 ever showed, which independently corroborates finding 8's
split of the headroom.

**10. The suppression rule transfers across splits; the model does not transfer across
crown scale.**
Scored on BAMFORESTS TestSet1 (39 crops) and TestSet2 (40), both untouched by any
tuning. The single `eval`-chosen setting lands within **1.35 F1@50** of each test
split's own swept optimum, while the published 24 px setting degrades catastrophically
off its tuning split — on TestSet1 it emits **3.18 seeds per crown** and collapses to
COCO AP 0.86.

| split | crown diameter D | published | frozen (160, 0.50) |
|---|---:|---:|---:|
| evalhold60 | 211 px | AP 15.81 | **AP 23.85** |
| test2 | 212 px | AP 13.18 | **AP 21.29** |
| test1 | **314 px** | AP 0.86 | AP 2.48 |

But TestSet1 is an order of magnitude worse *even at its own optimum* (AP 2.98), while
TestSet2 matches eval. The distinguishing statistic is crown scale: TestSet1's crowns
are **2.2x larger in area** (D 314 vs 211 px). The decoder conclusions still hold there
— the suppression rule works — but the heads produce almost nothing usable.

**This bounds the external validity of everything above.** The error budget (H0, H3),
the 69-70 marker wall and the 97.3 oracle ceiling were all measured at one crown scale,
and TestSet1 shows the model degrades severely under a 1.5x scale shift. H6 and H7 may
be optimising inside a regime the model cannot leave.

**11. A class-balanced boundary loss buys geometry, not recognition — and retires the
merge step.**
Retraining only the boundary term (per-batch balanced BCE + Tversky alpha=0.3/beta=0.7,
12 epochs, everything else held fixed) moved the hold-out from **COCO AP 24.32 to
26.84** and **AP75 16.48 to 20.40**, with F1@75 30.45 -> 37.06, SQ +1.8, B-IoU +2.4 and
Dao ED 0.638 -> 0.595. F1@50 is flat (+0.07): the gain is entirely in where masks land,
not in how many crowns are found — which is correct, since recognition is bounded by
seeding and H6 did not touch seeding.

Two things this settles:
- **Merging has expired.** Under the calibrated head, no-merge (AP 26.84) beats
  percentile merge (26.29) and absolute-threshold merge (25.66). Merging existed to
  repair over-separation caused by a weak ridge; a stronger ridge separates correctly,
  so the repair now deletes real boundaries.
- **The calibration is only half-fixed.** Mean B on true seams rose 0.105 -> 0.186
  (target was 0.40), but background rose 0.013 -> 0.041, so the seam-to-background
  ratio *fell* from 8.0x to 4.5x. Class-balanced weighting raises the whole field
  rather than sharpening contrast. Further boundary work should change the **target**
  (a 3 px band blurred at sigma 0.8 may be too thin to supervise at this resolution),
  not the loss weighting.

My 0.40 success criterion was invented rather than derived. The flood relief is
`boundary * 255`, so 0.105 -> 0.186 nearly doubles the barrier height between crowns
(27/255 -> 47/255), which is plainly enough to change flooding — and did. The criterion
failed while the mechanism it was meant to test held.

**12. Where this project actually stands (literature, checked late).**
The BAMFORESTS paper itself reports no baselines. Published results on the benchmark:
**TreeCoG AP 53.21 / AP@50 73.14**; **FG-TreeSeg mAP@50 67.31**. This project's best is
**AP 26.84 / AP@50 58.06** — roughly half TreeCoG's AP. Protocols are not verified to
match (I score 1024 px crops from the 2048 px tiles on 60 held-out validation crops),
so this is indicative of a large gap rather than a like-for-like ranking, but a 2x gap
is too wide to be protocol alone.

**The contribution here is diagnostic, not competitive.** It locates where a watershed
ITC pipeline loses accuracy and shows the decoder is not the limit (84.48 AP with
correct heads). It does not produce a state-of-the-art detector, and the write-up must
say so plainly.

**13. Two earlier conclusions were scoped too widely and are corrected.**
- **H2's "the merge framework tops out at ~70" was too broad.** TreeCoG uses exactly
  that paradigm — deliberate over-segmentation then merging — and is the published SOTA.
  What H2's oracle bounded was merging *of this pipeline's basins*, flooded against a
  boundary head with mean seam response 0.105. TreeCoG differs in both components H2
  held fixed: a dedicated **contour network** for over-segmentation, and a **learned
  GNN** merge criterion rather than a threshold on one scalar — which is precisely the
  failure mode H2 and H5 diagnosed. H6 corroborates the reading: a stronger ridge
  turned merging from mildly helpful to harmful, so merging's value tracks basin
  quality.
- **H8's "the model does not survive a 1.5x scale shift" under-described the cause.**
  TestSet1 is the benchmark's *designated* out-of-domain split: the complete Hain-AOI,
  a **city park** rather than managed forest, "of severely different structure",
  5-20 km distant, captured with **another UAV and sensor** (DJI Phantom 4 at 85 m AGL
  vs Trinity F90+ at 120 m), on a different date, flown after sunrise rather than at
  midday. Crown scale (D 314 vs 211 px) is one symptom of a multi-factor shift, not the
  cause. A scale-only remedy may not help.

**14. Crown scale explains only ~7% of the out-of-domain collapse.**
Test-time rescaling of TestSet1 (fully convolutional network, no retraining) peaked
**exactly** at the scale-matching factor `s = D_train/D_test1 = 0.672`, lifting COCO AP
5.46 -> 6.91, while the TestSet2 control was **perfectly flat** (s = 1.0 optimal,
rescaling strictly harmful: 25.24 -> 19.33 -> 12.00). Both controls behaving correctly
means the measured effect is the scale axis and nothing else — and it recovers

```
(6.91 - 5.46) / (25.24 - 5.46) = 7.3% of the domain gap
```

So **~93% of the transfer failure is appearance and structure**, not geometry: city-park
canopy rather than managed forest, a different sensor's colour response, and low-sun
illumination with long shadows. The crown-diameter framing is now **dropped** as an
explanation of the domain gap (it remains the correct account of the *decode constant*,
which is a separate claim). Multi-scale training targets the 7% and is deprioritised;
photometric augmentation or domain adaptation is the lever.

**15. The generalisation gap is larger than everything this project has fixed.**
In-domain AP 25.24 (TestSet2) vs out-of-domain AP 5.46 (TestSet1) is a **4.6x drop**.
Every improvement produced here — +8.0 AP from the decode constants, +2.5 from the
calibrated head — is smaller than that gap. It belongs in any results table as a
headline pair, per the project's standing constraint in `CLAUDE.md`.

## Lessons and constraints

- `min_distance` 144-160 px, `min_apex` 0.50, `min_canopy` 0.25, `min_area` 25 is the
  current best decode setting for BAMFORESTS at 1024 px crops. Do not use 24 / 0.25.
- `min_apex >= 0.60` collapses recall fast (ratio 0.62, mAP@50 -> 61.6). 0.50 is the knee.
- Superlevel-set radius estimation: `t = 0.7` shatters components (seed ratio 7.8).
  Unusable above `t = 0.5`. The whole approach is dropped.
- `train_sharp.py` warm-starts from `best_canopy_watershed.pth`; `train_calibrated.py`
  (H6) warm-starts from `best_sharp_watershed.pth` instead, so it is a fine-tune of the
  current best rather than a re-run from the older checkpoint. Keep that in mind when
  comparing training curves — the boundary loss values are on different scales and are
  not comparable across the two scripts.
- `evaluate_metrics` reports **F1 at a fixed IoU**, not interpolated average precision,
  while labelling it `ap50`. `crown_research/src/coco_ap.py` now provides the real
  metric; report both, and never quote the repo's `ap50` as AP in writing.
- COCO AP needs a meaningful confidence per instance. Merged instances leave `score`
  at 1.0, which makes the ranking degenerate; use
  `coco_ap.instance_scores_from_surface` to set score = peak surface response inside
  the instance before accumulating.
- Tune on `eval30`, always re-score frozen settings on `evalhold60` (eval idx 100-159)
  before believing any delta.
- Cached head outputs (`crown_research/src/pred_cache.py`) turn a sweep from
  GPU-minutes into CPU-seconds. Use them for anything decode-side.

## Current best configuration

`best_calibrated_watershed.pth` (H6) with `min_dist=160`, `min_apex=0.50`,
`min_canopy=0.25`, `min_area=25`, flat-basin relief, **no merge**. On 60 held-out crops:

| | published settings | **current best** |
|---|---:|---:|
| COCO AP | 15.81 | **26.84** |
| COCO AP50 | 40.34 | **58.06** |
| COCO AP75 | 9.48 | **20.40** |
| F1@50 | 44.29 | 66.98 |
| PQ | 31.01 | 49.55 |
| Dao ED | 0.598 | 0.594 |

Arc in COCO AP: published 15.81 -> H1 tuned 23.85 -> H5 merge 24.32 -> **H6 26.84**.

## Open questions

1. ~~Does a class-balanced boundary loss restore magnitude?~~ **ANSWERED (H6):**
   partially — 0.105 -> 0.186, not the 0.40 aimed at, and the seam-to-background ratio
   worsened. Still worth +2.5 COCO AP. Further boundary work should target the **seam
   target's width**, not the loss weighting.
2. ~~Do merge and gate coupling become viable with a better B?~~ **ANSWERED (H6):** no
   — merging now *hurts* (26.84 no-merge vs 26.29 merged). The workaround has expired.
3. ~~Why does the pipeline miss 16.9% of crowns under loose seeding?~~ **ANSWERED:**
   `never_lit` small crowns (median 1,899 px vs 24,382 px), not suppression. See
   finding 6.
4. ~~Does the tuned setting hold across splits?~~ **ANSWERED (H8):** yes — within 1.35
   F1@50 of each split's own optimum on TestSet1 and TestSet2. See finding 10.
6. ~~Which factor dominates the TestSet1 collapse?~~ **ANSWERED (H9):** not scale —
   it accounts for ~7%. See finding 14. The live version of this question is now
   **which appearance factor dominates** (colour/sensor response, illumination and
   shadow, or canopy structure), and whether photometric augmentation closes any of it.
8. **Does boundary calibration improve out-of-domain robustness disproportionately?**
   TestSet1 went 2.98 -> 5.46 AP between the old and calibrated checkpoints (+83%) vs
   ~+10% in-domain — but the two runs used different suppression radii, so this is
   confounded. Needs a paired re-run at identical decode settings before it is claimed.
7. **Does a learned merge criterion on H6's improved basins close any of the gap to
   TreeCoG?** H2's threshold merge failed because B was uninformative in magnitude, and
   TreeCoG shows a learned merger over a good over-segmentation reaches AP 53.21.
5. The canopy head reaches AUROC 0.961 and looks healthy, yet H3 shows the system
   depends on its *hard barriers* (32.1% of true seams) rather than its ranking. Is
   there a supervision change that makes that dependence explicit rather than accidental?
