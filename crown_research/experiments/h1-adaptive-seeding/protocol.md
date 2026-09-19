# H1 — Scale-adaptive seed suppression

**Status:** LOCKED (pre-registration) · 2026-09-19
**Type:** CONFIRMATORY
**Target to approach:** H0 cell B = **70.79 mAP@50** (oracle seeds, predicted energy).
**Baseline to beat:** 31.33 mAP@50 (global 24px NMS).

## Mechanism

`targets_sharp.py:59-63` builds the surface target as

```python
mask_edt = distance_transform_edt(mask)
u_inst   = mask_edt / mask_edt.max()     # <-- per-instance normalisation
```

Every crown is normalised to peak at exactly 1.0 **regardless of its size**. The
surface head U is therefore *scale-blind at its maxima* by construction: a 6 m crown
and a 20 m crown produce identical peak values. With no scale signal at the peak, the
only suppression rule available to the decoder is a single global radius — the
`min_distance = 24 px` in `run_flat_basin_watershed`. That is the structural reason
the pipeline over-seeds at 1.90 seeds per crown (H0).

But normalisation does *not* destroy scale everywhere. The **superlevel set**
`{U >= t}` around a peak still has absolute pixel extent proportional to crown area,
because only the *values* were rescaled, not the support. So a per-peak radius is
recoverable from the network's existing output with no retraining:

```
A_i = area of the connected component of {U >= t} & {canopy >= c} containing peak i
r_i = sqrt(A_i / pi)
suppress peak i against accepted peak j iff  dist(i,j) < gamma * max(r_i, r_j)
```

## Arms

- **H1-a (scale-blind ceiling).** Grid over the two existing knobs:
  `min_dist in {16, 20, 24, 28, 32, 40, 48, 56}` x `min_apex in {0.20, 0.25, 0.35, 0.45}`.
  Establishes how far a *global* radius can possibly go.
- **H1-b (scale-adaptive).** Superlevel-set radius, grid over
  `t in {0.3, 0.5, 0.7}` x `gamma in {0.6, 0.8, 1.0, 1.2}` x `min_apex in {0.25, 0.35}`,
  with a floor of 6 px on r_i to keep degenerate peaks from self-suppressing.

Both arms share: cached `best_sharp_watershed.pth` outputs on the same 30 eval crops,
flat-basin flooding, `min_canopy = 0.25`, `min_area = 25`, identical metric code.

## Predictions (committed before running)

1. **H1-a plateaus well short of the oracle.** The best global-radius cell reaches
   **mAP@50 <= 48%**, i.e. it recovers less than half of H0's +39.5 gap.
2. **H1-b beats the best H1-a cell by >= 3 mAP@50 points.**
3. H1-b's seed ratio `n_pred/n_gt` lands in **[0.85, 1.20]** at its best setting
   (vs 1.904 for the baseline).

If prediction 2 fails, the superlevel-set radius is too noisy an estimator and the
project moves to H2 (region-adjacency merging on boundary evidence), which repairs
over-segmentation after flooding instead of preventing it.

## Selection rule (locked to prevent metric gaming)

Best cell per arm is chosen by **mean mAP@50 across all 30 crops**, ties broken by PQ.
Reported alongside: mAP@75, PQ/SQ/RQ, B-IoU, Dao OS/US/ED, seed ratio.

## Outputs
- `results/h1a_global_sweep.csv`, `results/h1b_adaptive_sweep.csv`
- `results/h1_summary.csv` — baseline vs best-a vs best-b vs H0 cell B
