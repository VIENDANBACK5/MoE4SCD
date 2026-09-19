# H10 — Does boundary calibration help out-of-domain more than in-domain?

**Status:** LOCKED (pre-registration) · 2026-09-20
**Type:** CONFIRMATORY · resolves a confound logged in H9

## The confound being resolved

H9 noted that TestSet1 scored **AP 2.98** with the old checkpoint (measured in H8) and
**AP 5.46** with the H6 calibrated checkpoint — apparently +83%, against roughly +10%
in-domain. That would be a genuinely interesting robustness result, but the two runs
**used different suppression radii** (H8 swept to test1's AP-optimum at md=240; H9 fixed
md=280), so the comparison is invalid as it stands. H9 logged it as an open question
rather than reporting it.

This resolves it by running both checkpoints over the **same decode grid** on the
**same crops**, so the only thing that differs is the weights.

## Design

- Checkpoints: `best_sharp_watershed.pth` (old) vs `best_calibrated_watershed.pth` (H6).
- Splits: `test1` (out-of-domain by benchmark design) and `test2` (in-domain), 40 crops
  each, plus `evalhold60` as the validation reference.
- Decode grid swept identically for both: `min_dist in {160, 200, 240, 280, 320}` x
  `min_apex in {0.35, 0.50, 0.65}`, no merge (H6 retired it).
- Each checkpoint is scored at **its own best cell per split**, and additionally at a
  **common cell** so the comparison can be read both ways.
- Metric: COCO AP (primary), with AP50 and F1@50 reported.

## Predictions (committed before running)

1. **Calibration helps out-of-domain more in relative terms.**
   `ΔAP_rel(test1) > ΔAP_rel(test2)`, where `ΔAP_rel = (new - old) / old` at each
   split's own best cell.
2. **The out-of-domain gain is real, not noise.** `ΔAP(test1) >= +1.0` absolute.
3. **The apparent +83% shrinks once decode is controlled.** The controlled
   `ΔAP_rel(test1)` is **below 83%** — i.e. part of what H9 observed was the decode
   difference, not the weights.

Prediction 3 is deliberately a prediction against my own earlier observation. If it
fails and the gain really is ~83% at matched decode, that is a stronger robustness
result than expected and should be pursued.

If prediction 1 fails, boundary calibration is domain-neutral — a useful negative that
would close the question rather than leave it hanging.

## Why this matters beyond bookkeeping

H9 showed ~93% of the domain gap is appearance, not geometry. If a *training-side*
change aimed at calibration also buys out-of-domain robustness, that points at the kind
of intervention worth making next — and it would be the first evidence in this project
that anything narrows the 4.6x generalisation gap.

## Outputs
- `results/h10_paired_sweep.csv`, `results/h10_summary.csv`
