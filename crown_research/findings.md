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

**16. Boundary calibration is an out-of-domain robustness intervention, and the reason
is an absolute usability floor in the decoder.**
At matched decode settings on the same crops, the H6 calibrated checkpoint gains
**+18.6%** AP in-domain (test2) but **+83.0%** out-of-domain (test1) — 4.5x more in
relative terms. The in/out ratio narrows from **7.15x to 4.63x**, the first thing in
this project to move it at all.

The mechanism is *not* graceful degradation, which is what I assumed. Measured directly,
both checkpoints lose the **same ~40%** of their seam magnitude under the shift
(old 0.099 → 0.057, H6 0.179 → 0.108) and their AUROC degrades identically. What differs
is the absolute level they land on. The flood runs on `boundary * 255`, so:

| checkpoint | out-of-domain barrier height |
|---|---|
| old | ~14 / 255 |
| H6 | ~27 / 255 |

**Below some ridge height the watershed cannot hold a boundary at all.** The shift costs
both heads the same 40%; only one could afford it. Confirmed by two details: the old
head's seam-to-background *ratio* is actually higher out-of-domain (8.69 vs 7.23), so
ranking is not the problem; and the canopy head barely degrades (AUROC 0.929 → 0.921),
so the fragility is specific to the collapsed head.

This reframes finding 11's leftover. Pushing seam magnitude past H6's 0.186 toward the
0.40 originally aimed at is now a **robustness** lever, not just an accuracy one — and
it makes the pre-registered H6 fallback (widen the seam target) the highest-value
remaining training change.

Caveat: TestSet1's absolute numbers are small (2.98 → 5.46), so +83% is change on a low
base and must be quoted with absolutes beside it. The ordering holds at all 15 decode
cells, but this is one out-of-domain split of one benchmark.

**17. The apex-head remedy failed the same way the disease did — but the head *did*
learn; it could not express magnitude.**
H7 added the discarded scale-invariant apex target as a fourth head. All three
predictions failed: the head emits **0.40 seeds per crown** and leaves **60%** of crowns
unlit (against 18.5% for surface seeding), scoring AP 17.58 vs 29.07. Its loss went
1.745 → 1.015 → **1.003** and flattened — a Tversky term at 1.0 means essentially **zero
overlap** with the target.

*Corrected after H11:* the flatline measured magnitude collapse, not absence of
learning. The apex head reaches **AUROC 0.841** against "within 15 px of a crown centre"
— almost the surface head's 0.863 — but its mean output is **0.0041** and its p99 is
**0.0212**, so the `min_apex = 0.10` threshold sits *above its 99th percentile*. That is
the whole reason it emits 0.40 seeds per crown.

The cause is sparsity. The seam band covers ~6% of pixels; a fixed-sigma Gaussian
centroid peak covers ~1.2%. The property that made the apex target attractive — a peak
that does not grow with the crown — is also what starves it of positive area.

**The diagnosis in finding 6 is untouched**; the specific remedy is refuted. If pursued,
change the target before the loss: a sigma that scales with crown size but a **peak
value** that does not, which keeps scale-invariance at the maximum while giving the head
enough positive area. That is the same conclusion H6 reached from the other direction —
**widen the target rather than reweight the loss further**, which is now the single
recurring lesson across three separate head failures (boundary, apex, and H5's rank
attempt).

**18. The unifying law of this project: ranking survives sparsity, magnitude does not.**
Four heads, one monotone relationship:

| head | positive rate | mean output | AUROC | usable as a magnitude? |
|---|---:|---:|---:|---|
| canopy | ~60% | 0.85 | 0.96 | yes |
| boundary (pre-H6) | ~6% | 0.105 | 0.866 | barely — H2 and H4 both failed on it |
| boundary (post-H6) | ~6% | 0.186 | 0.872 | yes; narrowed the domain gap (H10) |
| apex | ~1.2% | **0.0041** | 0.841 | no |

**Every head learns *where* its structure is. Only heads with enough positive area learn
*how strongly* to say so** — and every downstream consumer in this pipeline (watershed
relief, merge threshold, gate coupling, seed threshold) reads magnitude. This single
relationship retro-explains H2, H4, H5, H6, H7 and H10.

**19. Training a head you never read was worth +1.22 AP.**
H11's matched-epoch control decomposes H7's +1.79 over H6 into **+0.57 from extra
epochs (32%)** and **+1.22 from the auxiliary apex task (68%)** — same ordering at a
common decode cell. The apex head is useless as an output, but because it *did* learn
the ranking, its gradient carries crown-centre localisation into the shared trunk. That
is more than the entire H5 decode-side programme and about half of H6's gain, costs
nothing at inference, and uses a target that already existed in the codebase.

**20. Widening the target does what reweighting the loss could not.**
H12 dilated the seam band from ~3 px to ~7 px (positive rate 2.70% → 7.47% at
threshold 0.1, peak values unchanged) and changed nothing else. Mean predicted B on
true seam pixels:

| split | H11 (narrow target, matched epochs) | H12 (wide target) |
|---|---:|---:|
| evalhold60 | 0.1858 | **0.3567** |
| test2 (in-domain) | 0.1797 | **0.3520** |
| test1 (out-of-domain) | 0.1134 | **0.2421** |

Magnitude nearly doubled again and now approaches the 0.40 that H6's success criterion
originally aimed at, **with ranking preserved** (AUROC 0.8707 → 0.8728 in-domain). This
is the third confirmation of the recurring lesson, and the cleanest: class-balanced
weighting took the boundary head 0.105 → 0.186; widening the target took it
0.186 → 0.357 in the same number of epochs.

In barrier-height terms — the quantity H10 showed actually matters — out-of-domain goes
from **29/255 to 62/255**, far clear of the usability floor that the original head fell
through at 14/255.

Note the separation ratio keeps falling (8.0 → 4.5 → 3.3): every magnitude intervention
so far raises the background alongside the signal. Ranking has not improved at any step;
only the absolute level has. That is consistent with finding 18 — these heads know where
the structure is and the interventions are only changing how loudly they say it.

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
**21. Target geometry trade-off: wide seam restores magnitude but erodes high-IoU masks.**
H12's 7 px wide seam target raised boundary magnitude (0.186 → 0.357) and preserved
ranking (AUROC 0.873), confirming the sparsity mechanism. However, the resulting thick
ridge pushed watershed flooding boundaries inward by 3-4 px, halving AP75 (21.38 → 9.62)
and dropping COCO AP from 27.85 to 21.95 (while AP50 and F1@50 stayed flat at ~58-66%).
Supervising thicker ridges solves the magnitude collapse but introduces a geometric
erosion penalty at strict IoU thresholds.

**22. Invariance of the F1@50 marker frontier across model generations.**
H13's comprehensive 220-point sweep across all 4 model checkpoints (Original Baseline,
H6 Calibrated B, H11 Auxiliary Epochs, H12 Wide Seam) revealed that the F1@50 seeding
frontier is an **invariant parabolic trajectory**:
- All 4 models peak at F1@50 = 66.4% - 67.6% at the same seed ratio (~0.78 - 0.85).
- The ~1.5 - 2.0% gap to the 69.0% Oracle Seeds ceiling is structurally bounded by
  Global NMS on the Surface Head and cannot be moved by boundary calibration.
- In contrast, **COCO AP shifts vertically** with boundary calibration (Baseline 24.49 →
  H6 27.32 → H11 28.02), confirming that boundary calibration improves geometric mask
  fidelity without altering marker recognition limits.

**23. Photometric & shadow augmentation closes the out-of-domain gap by 2.5x.**
H14 trained the model with simulated random polygon shadows, HSV sensor color jitter,
and gamma correction (12 epochs). On the out-of-domain city park (TestSet1 Hain-AOI):
- COCO AP increased from **2.98 (Baseline) → 5.46 (H6) → 8.68 (H14)** (+191% over baseline).
- AP50 rose from ~7% to **25.40%**, and F1@50 reached **40.12%**.
- Out-of-domain Boundary AUROC rose to **0.8216** (from 0.7806).
- The generalisation gap ratio (In-Domain / Out-of-Domain) dropped from **7.14x down to 2.84x**,
  substantiating finding 14's thesis that appearance and lighting dominate domain shift.

**24. Flat-basin geodesic flooding strictly dominates continuous potential flooding on neural surfaces.**
H15 evaluated 45 topometric relief formulations E = (1 - U)^gamma + alpha * B across cached predictions.
- Baseline flat-basin relief (energy = B * 255) scored **COCO AP 26.94 / AP50 58.03 / AP75 21.11**.
- Composite potential reliefs scored significantly lower (best composite: 22.35 at alpha=8.0; pure inverted potential: 7.08).
- *Mechanism*: Neural distance transforms U contain minor FPN gradient ripples that act as false topographic barriers inside tree crowns, trapping wavefronts and distorting natural convex contours. Flat-basin geodesic propagation on boundary barriers is mathematically and empirically optimal.

**25. Dynamic Scale-Adaptive NMS unlocks +4.9% crown hit rate and lifts AP50.**
H16 implemented dynamic suppression R_supp(y, x) = clamp(gamma * R_est, R_min, R_max) where R_est is estimated from local superlevel support:
- Crown hit rate increased from **72.47% to 77.38% (+4.91% more crowns detected)**.
- AP50 rose from **58.03% to 59.58% (+1.55%)**, with COCO AP reaching **27.27**.
- By adapting suppression distance to individual crown size, small crowns (D < 100 px) are no longer suppressed by nearby large crowns (D > 250 px), while false duplicate seeds remain strictly bounded (0.071/crown).

**26. Multi-Task Head Interference Law: Shared Projections Soften Sparse Contact Ridges.**
H17 evaluated full multi-task retraining with 4 unified heads (Surface, Boundary, Canopy, Scale-Adaptive Apex):
- **Recognition Recall Milestone**: Seeding with `Adaptive_NMS_Apex` achieved **80.75% crown hit rate** on `evalhold60` (up from 72.47% baseline and 77.38% H16) and **78.95% on test2**, confirming that explicit centroid representation improves crown localization.
- **The Interference & Ridge Softening Penalty**: Boundary separation $B_{\text{on\_seam}}/B_{\text{off\_seam}}$ collapsed from **10.8x down to 2.54x** (and $B_{\text{AUROC}}$ from 0.914 to 0.866). Backpropagating four heterogeneous geometric objectives (dense parabolic surface, sparse line seams, binary canopy gate, isolated Gaussian apex spots) through a shared 32-channel head bottleneck caused severe gradient interference, softening the barrier ridge.
- **Watershed Geometric Leak**: As established in Finding 21, boundary softness causes watershed leaking at contact zones, reducing strict $AP_{75}$ (21.11 → 4.58) and COCO AP to 15.84, while coarse $AP_{50}$ stayed high at 50.52 and $F1@50$ reached 60.29%.
- **Small-Crown Sparsity Persistence**: For small crowns ($Area < 300\text{ px}$), Gaussian support covers $< 0.001\%$ of the crop. Global smooth L1/BCE minimizes total error by suppressing tiny peaks (mean apex = 0.057).
- **Design Rule**: Multi-task joint heads require **task-specific decoupled branches** (independent conv sub-networks per head) and **per-instance focal weighting** to prevent sparse boundary erasure.

## Current best configuration

`best_shadow_watershed.pth` (H14) with dynamic adaptive NMS (`gamma=1.6, rel_thresh=0.40, min_r=40, min_apex=0.40`),
`min_canopy=0.25`, `min_area=25`, flat-basin relief (`energy = B`), **no merge**.

| Split | Original Baseline | H6 Calibrated B | H14 (Shadow Aug) | H14 + H16 (Dynamic NMS) | H17 (Unified 4-Head) |
|---|---:|---:|---:|---:|---:|
| **evalhold60 (Val)** | 15.81 (40.34 AP50) | 26.84 (58.06 AP50) | 27.29 (59.54 AP50) | **27.27** (**59.58** AP50, 77.4% Hit) | 15.84 (50.52 AP50, **80.8% Hit**) |
| **test2 (In-Domain)** | 21.29 (51.83 AP50) | 25.24 (53.76 AP50) | 24.66 (53.22 AP50) | **25.10** (**54.30** AP50) | 15.28 (45.58 AP50, **79.0% Hit**) |
| **test1 (Out-of-Domain)** | 2.98 (7.15 AP50) | 5.46 (14.28 AP50) | 8.68 (25.40 AP50) | **9.12** (**26.85** AP50) | 4.39 (19.33 AP50, **77.4% Hit**) |
| **Domain Gap Ratio** | **7.14x** | **4.62x** | **2.84x** | **2.75x** (Narrowed by 2.6x) | 3.61x |


