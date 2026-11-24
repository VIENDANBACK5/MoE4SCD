# Phase 0 D3-query (G3): CellSAM on LIVECell val, zero-shot (2026-11-24)

## Setup
- **Code:** `code/d3_livecell_cellsam.py` (run) and `code/compare_livecell_query.py` (table and preview).
- **Log:** `logs/d3_livecell_cellsam.log`.
- **Results:** `results/livecell_d3_cellsam.json`, `results/livecell_query_comparison.json`,
  `results/cellsam_absorbing_mask_multiplicity.json`.
- **Model:** CellSAM general v1.2, owner's cache; md5 matches.
- **Data:** all 570 val images, inference only. The test split was not touched.
- **Fidelity check:** the reimplementation of `segment_cellular_image` (default settings) matches the package exactly on 6
  pilot images (same instance counts, foreground IoU ≥ 0.9999).
- **Code vs paper:** two code details differ from the paper, and the code was followed:
  - the mask threshold is 0.4, not 0.5;
  - the label map takes the max index (a later mask overwrites an earlier one), not an argmax.
- **Package bug found:** `segment_cellular_image` crashes (AttributeError) when every mask is rejected.
- **Absorption measurement:** same GT, same border exclusion, same instances as Cellpose3 and CenterMask2. The script
  asserts that the instance sets are identical.

## Main table (d = 2)

| Model (all 160,540 instances) | Recall | Absorption rate (absorbed / all) | Share of misses | Strong-stratum rate |
|---|---|---|---|---|
| Cellpose3 livecell_cp3 (in-domain) | 0.616 | 0.085 | 0.222 | 0.022 |
| CenterMask2 (in-domain) | 0.790 | 0.071 | 0.338 | 0.024 |
| **CellSAM label map (zero-shot)** | **0.303** | **0.124** | **0.178** | 0.009 |
| CellSAM per-instance masks | 0.303 | 0.124 | 0.178 | 0.009 |
| CellSAM with GT boxes as prompts (oracle hypotheses) | 0.904 | 0.047 | 0.495 | 0.016 |

After the sliver filter (thick ≥ 3, n = 155,626), the absorption rate is:
- CenterMask2: 0.055;
- CellSAM: 0.126;
- GT-box oracle: 0.033.

Size × density standardised asymmetry effect for CellSAM: −0.010 [−0.026, 0.006], i.e. none.

## Pre-declared reading (PHASE0_SYNTHESIS §6): **INCONCLUSIVE**
- The rule required both the share of misses and the absorption rate to be ≥ CenterMask2's.
- The rate is higher (0.124 vs 0.071), but the share is lower (0.178 vs 0.338).
- The share is lower because zero-shot CellSAM misses about 70% of cells outright (recall 0.30). Exactly this
  domain-shift effect was anticipated when the rule was written.
- The rule is applied as written; no re-reading after the fact.

## Post-hoc observations (not part of the rule)

**1. CellSAM's absorption is mostly colony-level.**
- Each absorbed cell sits in a CellSAM mask that covers, at the median, **6** GT cells. Distribution:
  - 2 cells: 20%;
  - 3–4 cells: 19%;
  - 5–9 cells: 25%;
  - ≥ 10 cells: 36%.
- By cell type: BT474 0.38 and MCF7 0.28, which are the raft-growing lines; A172 0.05 and SKOV3 0.02.
- The preview (8 random crops, eye judgement coarse) shows whole rafts outlined as one object.
- Reading: this is a granularity mismatch between the training domains and LIVECell. It is not comparable to
  CenterMask2's mostly pairwise absorption. The effect of set prediction on absorption therefore remains unknown.

**2. Decomposition (robust to the domain-shift caveat, because it describes where the loss happens).** Of 19,869
absorbed cells:

| Category | Share |
|---|---|
| No raw box with box-IoU ≥ 0.5, at any score | 46.0% |
| Such a box exists but scored below the threshold | 46.1% |
| Box kept, mask dropped for predicted IoU | < 0.1% |
| Box kept, mask wrong | 7.4% |
| Correct instance mask lost when compositing the label map | 0.6% |

Wording corrected after an external review. The categories describe **output stages**, not representation causes:
- **Candidate localisation, 46.0%.** No output box reaches box-IoU 0.5 with the cell. This does not show that the model
  has no internal representation of the cell; it shows only that no output box passes the chosen matching cut. See
  Step 1 for the sensitivity of the IoU cut.
- **Candidate selection / survival, 46.1%.** A well-localised box exists, but its confidence is below the threshold.
  This is NOT a query-formation failure.
- **Assignment, 7.4% + 0.6%.** A kept box gives a wrong mask, or a correct mask is lost when compositing.

**3. The GT-box oracle confirms this.**
- When every cell receives its own correct box prompt, SAM's decoder produces a mask with IoU ≥ 0.5 for 90% of cells.
- The absorption rate falls from 0.124 to 0.047 (0.033 after the sliver filter).
- 7.1% of prompted masks still cover ≥ 50% of a neighbour.
- The standardised asymmetry effect under oracle prompts is +0.20 [0.09, 0.30]. A small cell's box prompt tends to leak
  into its large neighbour. This is a mask-decoder effect, separate from formation.

## What this does and does not establish
- **Established [V], wording per the review:** in zero-shot CellSAM on LIVECell, most measured absorption errors are
  associated with inaccurate or rejected object-box candidates. Mask reconstruction and label-map compositing account
  for a smaller fraction. This identifies candidate generation and selection as important failure stages under the
  evaluated domain shift. It does not establish a general failure mechanism of in-domain set prediction.
- **Verified 2026-11-24** (`code/multiplicity_centermask.py`), checking an earlier unverified claim made in this file:
  - CenterMask2's absorbing masks cover a median of 2 GT cells. 83% cover exactly 2, 16% cover 3–4, 0.7% cover 5–9, and
    none cover ≥ 10. This holds for every cell type (median 2).
  - CellSAM: median 6, and 36% cover ≥ 10.
  - The two models' absorptions differ in kind (pairwise vs colony-level).
- **Not established:** whether an IN-DOMAIN set-prediction model absorbs less than NMS / flow models. Zero-shot
  granularity mismatch dominates the CellSAM numbers.
- **Not done:** QuebecTrees has no query model.

## Step 1 (external review): query-selection diagnostic, no training (`code/d3q_threshold_sweep.py`)

Only the box-score threshold changes. Every candidate with score ≥ 0.05 is decoded. The absorption metric is unchanged.
The dynamic-default row reproduces the main run exactly (recall 0.303, rate 0.124).

**Per-candidate masks:**

| Box threshold | Predictions | FP (unmatched) | Duplicates | Recall | Absorption rate | Default-absorbed cells rescued (of 19,869) | Mask AP | AP50 |
|---|---|---|---|---|---|---|---|---|
| dynamic (≈0.27–0.41) | 65,907 | 12,237 | 90 | 0.303 | 0.124 | 112 | 0.037 | 0.267 |
| 0.25 | 84,250 | 21,474 | 1,187 | 0.351 | 0.152 | 660 | 0.039 | 0.306 |
| 0.20 | 100,654 | 32,695 | 2,556 | 0.378 | 0.181 | 998 | 0.041 | 0.327 |
| 0.15 | 130,188 | 56,237 | 4,785 | 0.409 | 0.211 | 1,541 | 0.042 | 0.345 |
| 0.10 | 182,802 | 101,759 | 8,722 | 0.445 | 0.237 | 2,344 | 0.043 | 0.366 |
| 0.05 | 289,880 | 199,634 | 17,294 | 0.493 | 0.260 | 3,880 | 0.044 | 0.384 |

**Label map (package compositing):** recall peaks at 0.386 at threshold 0.10–0.15, then falls to 0.323 at 0.05 as
overlapping masks overwrite each other. The absorption rate rises from 0.124 to 0.244.

**Sensitivity of the "no_box" category** (best raw-box IoU at any score, 9,136 cells):
- median 0.34;
- 60% have a box with IoU ≥ 0.3;
- 35% have one with IoU ≥ 0.4;
- 21% have none above 0.2.

Most "no_box" cells therefore have a loosely localised candidate rather than none.

**Reading:**
- Lowering the threshold rescues at most 20% of the default absorptions (3,880 of 19,869), at a cost of 16× more false
  positives.
- AP is flat (0.037 → 0.044).
- The **absorption rate doubles** (0.124 → 0.260): the extra low-score candidates include multi-cell / colony masks as
  well as single cells.
- In this zero-shot model, the confidence score does not separate single-cell from group hypotheses, so score
  calibration alone does not fix absorption.
- Per the review's stop rule, this is recorded and Step 1 stops.
- Caveat: zero-shot. This is a statement about CellSAM's candidate pool under domain shift, not about in-domain set
  prediction.
