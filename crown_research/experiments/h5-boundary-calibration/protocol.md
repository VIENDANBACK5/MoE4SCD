# H5 — The boundary head is mis-calibrated, not uninformative

**Status:** LOCKED (pre-registration) · 2026-09-19
**Type:** CONFIRMATORY

## Motivation

The head diagnostic (run after H4's flat lambda sweep) separates two things that H2 and
H4 had conflated:

| head | target | AUROC | AP / chance | mean | median | p99 |
|---|---|---:|---:|---:|---:|---:|
| boundary B | GT seam band (r=3) | **0.866** | 4.6x | **0.019** | **0.0011** | 0.51 |
| canopy C | GT foreground | 0.961 | 1.7x | 0.550 | 0.688 | 0.996 |
| 1 - canopy C | GT seam band (r=3) | 0.567 | 1.0x | — | — | — |

**B ranks seams well (AUROC 0.866) but is collapsed toward zero in magnitude.** Even
*on true seam pixels* it averages 0.105. This is a calibration failure produced by the
~6% positive rate of the seam class, not a failure to learn.

Every previous use of B has been an **absolute-magnitude** use, and each failed exactly
as this predicts:
- H2's merge threshold `tau` was flat for all `tau >= 0.35` because no seam's mean B
  ever reaches 0.35. The selected best was 0.15, the smallest value swept.
- H4's `canopy - lambda*B` moved mAP@50 by < 0.2 across `lambda in [0,2]`, because
  subtracting 2 x 0.019 from the gate changes nothing.

If the information is in the *ranking*, then rank-based use should recover it with no
retraining.

**Correction carried from H3.** H3's analysis proposed that the predicted canopy gate
was acting as the system's seam detector. The `1 - canopy` row above (AUROC 0.567,
AP at chance) **does not support that claim**, and it is withdrawn. The H3 interaction
itself is a measured fact and stands; its mechanism is re-opened, and arm D below tests
the replacement explanation directly.

## Arms

- **A control** — H1-tuned seeds, raw B relief, no merge. (hold-out 65.54)
- **B rank relief** — replace relief with the per-crop rank transform of B, so the flood
  sees a full-dynamic-range ridge field. Sweep gamma in {0.5, 1.0, 2.0} on the rank.
- **C percentile merge** — H2's seam merge but thresholding each seam's mean B at a
  per-crop **percentile** `q in {50, 70, 80, 90, 95}` instead of an absolute `tau`.
- **D gap-barrier audit** (diagnostic, not scored) — fraction of GT crown-crown seam
  pixels that fall in predicted background (`canopy < 0.25`), versus the fraction of
  crown-interior pixels that do. Tests whether the canopy gate separates crowns through
  hard barriers even though it ranks seams poorly.
- **E best-of** — rank relief + percentile merge together.

Tuned on `eval30`, frozen, re-scored on `evalhold60`.

## Predictions (committed before running)

1. **Rank relief (B) beats raw-B control by >= +2 mAP@50 on the hold-out.**
2. **Percentile merge (C) beats H2's absolute-threshold merge (65.96) by >= +2.**
3. **Arm D shows the canopy gate covers >= 30% of GT seam pixels** — i.e. hard barriers
   in gaps are a real separation mechanism even at AUROC 0.57.

If 1 and 2 both fail, B's ranking information is not usable at decode time at all, and
the project proceeds to retraining the boundary head with a class-balanced objective
(H6), which the diagnostic already justifies independently.

## Outputs
- `results/h5_sweep.csv`, `results/h5_summary.csv`, `results/h5_gap_audit.csv`
