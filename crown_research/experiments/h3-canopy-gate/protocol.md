# H3 — Completing the error budget: the canopy gate as the third factor

**Status:** LOCKED (pre-registration) · 2026-09-19
**Type:** CONFIRMATORY

## Motivation

H0 ran a 2x2 over {seeds} x {energy} and concluded marker selection owned the error
budget. That conclusion was correct but the design was incomplete: **both factors were
varied while the canopy head was held fixed at its predicted value in every cell.**

The canopy gate is not a passive input. In `flood()` it sets `markers[canopy < 0.25] = 1`,
which makes every sub-threshold pixel a *background seed*. Watershed cannot expand an
instance into territory already claimed by marker 1, so the predicted canopy mask
**hard-caps the extent of every instance**, no matter how good the seeds or seams are.

H2 produced the evidence that this matters: three independent perfect-marker routes
(oracle seeds / oracle seeds+seams / oracle merge) all converge at 68.6-70.8 mAP@50
with SQ stuck near 73 and mAP@75 never above 37. A shared cap is acting on all of them,
and the canopy gate is the only component common to every one.

## Design — full 2^3 factorial

| factor | predicted | oracle |
|---|---|---|
| seeds | apices from U, `md=160, ma=0.50` (H1 best) | EDT-max, one per GT crown |
| energy | boundary head B | GT inter-instance seams, blurred sigma=1 |
| canopy | canopy head, threshold 0.25 | `gt_map >= 2` |

8 cells, on both `eval30` and `evalhold60`. All other factors fixed
(`min_area=25`, flat-basin flooding, identical metric code). Main effects and the
seeds x canopy interaction are reported as mean deltas across the cube.

Note: predicted seeds now use the H1-tuned constants, not the original 24/0.25, so the
baseline cell is the current best-known configuration rather than the historical one.

## Predictions (committed before running)

1. **The canopy gate is the new binding constraint.** Oracle canopy with predicted
   seeds and predicted energy gains **>= +8 mAP@50** over the all-predicted cell.
2. **Oracle seeds + oracle canopy (predicted energy) exceeds 85 mAP@50**, i.e. it
   breaks decisively through the ~70 wall that H0/H2 could not.
3. **Full oracle (all three) >= 92 mAP@50.**

If prediction 2 fails, the ~70 wall is *not* the canopy gate, and the limit is in the
flat-basin watershed formulation itself (flood geometry / the `min_area` and
single-pixel-seam discretisation) — which would mean the architecture, not any head,
is the thing to replace. That outcome would redirect the project to a different
decoder family (e.g. direct mask regression per marker) rather than more head work.

## Outputs
- `results/h3_factorial.csv` — all 8 cells x 2 splits
- `results/h3_main_effects.csv` — main effects and interactions
