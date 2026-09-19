# H6 — Restoring magnitude to the boundary head with a class-balanced objective

**Status:** LOCKED (pre-registration) · 2026-09-19
**Type:** CONFIRMATORY · first training hypothesis of the project

## Diagnosis being acted on

The boundary head scores **AUROC 0.866** against GT seams while its median output is
**0.0011** and it averages **0.105 on true seam pixels**. It learned the ranking and
lost its magnitude.

The cause is visible in `train_sharp.py:83-85`:

```python
l_bound_bce  = F.binary_cross_entropy(pred_bound, bound_tgt)   # unweighted
l_bound_dice = dice_loss(pred_bound, bound_tgt)
l_bound_total = l_bound_bce + l_bound_dice
```

With a ~6% positive rate, unweighted BCE is minimised by predicting near zero
everywhere, and the training log confirms the head never escapes: the boundary term
sits at 1.08 -> 1.05 across all of epoch 1 and stays there, i.e. it is essentially the
Dice term pinned near 1.0 because predicted-vs-target overlap never materialises.

Every downstream failure so far follows from this and not from anything else:
- H2's merge threshold: no seam's mean B ever reached 0.35, so the sweep was flat and
  the "best" tau was the smallest one offered.
- H4's `canopy - lambda*B`: subtracting up to 2 x 0.019 changes nothing.
- H5's rank transform: the near-zero bulk is noise, so restoring range amplified it.

## Intervention (single factor)

New script `train_calibrated.py`, leaving `train_sharp.py` untouched so the existing
baseline stays reproducible. **Only the boundary term changes**; architecture, warm
start, optimiser, schedule, augmentation, `seam_weight`, surface and canopy losses are
identical.

```
l_bound = weighted_BCE(pred, tgt, pos_weight = neg/pos per batch, clamped to 20)
        + tversky(pred, tgt, alpha=0.3, beta=0.7)
```

- **Per-batch `pos_weight`** rather than a tuned constant: the imbalance ratio is a
  measurable property of the data, not a hyperparameter to search. Clamped at 20 to
  bound gradient scale on crops that happen to be nearly seam-free.
- **Tversky with beta > alpha** (Salehi et al., 2017) penalises false negatives ~2.3x
  more than false positives, which is the asymmetry a collapsed head needs. Dice is the
  alpha = beta = 0.5 special case and has already been shown not to suffice here.

Warm-started from `best_sharp_watershed.pth`, 12 epochs, lr 1e-4 cosine to 1e-6,
batch 2, seeds fixed (torch/numpy/random = 0). Output:
`DeadTrees/experiments/canopy_watershed/best_calibrated_watershed.pth`.

## Predictions (committed before running)

1. **Calibration — the success criterion.** Mean predicted B on true seam pixels rises
   above **0.40** on the hold-out split (from 0.105).
2. **Ranking is not sacrificed.** AUROC against the GT seam band stays **>= 0.85**
   (from 0.866).
3. **The absolute-magnitude uses become viable.** With the new head, H2's
   absolute-threshold seam merge beats no-merge by **>= +2 mAP@50** on the hold-out,
   having been worth ~0 before.
4. **The system breaks the marker wall.** Best end-to-end configuration reaches
   **>= 70 mAP@50** on the hold-out, exceeding the 69-70 ceiling that every
   perfect-marker route hit with the old head (H0 69.03, H2 69.74).

Predictions 1-2 concern the head itself and 3-4 the system. **Prediction 1 failing
while 3-4 pass, or vice versa, is itself informative** and will be reported as such
rather than reframed.

If 1 fails, positive weighting is insufficient against this target and the next step is
the *target*, not the loss — the seam band is 3 px wide and blurred at sigma 0.8, which
may simply be too thin to supervise at this resolution.

## Controls

The old checkpoint is re-scored side by side on the identical pipeline, so every number
below is a paired comparison on the same 60 hold-out crops. Surface and canopy heads are
re-diagnosed too, to catch backbone damage from the changed boundary gradient.

## Outputs
- `results/h6_head_diagnostic.csv` — old vs new head calibration and AUROC
- `results/h6_system.csv` — end-to-end metrics, both checkpoints, both splits
- `results/h6_merge_sweep.csv` — absolute-tau merge sweep under the new head
