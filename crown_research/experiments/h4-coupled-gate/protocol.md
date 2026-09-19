# H4 — Decode-time consistency coupling between the canopy gate and the boundary head

**Status:** LOCKED (pre-registration) · 2026-09-19
**Type:** CONFIRMATORY (cheap probe before committing GPU time to retraining)

## Motivation

H3 showed the canopy gate and the boundary field only help when **mutually consistent**:
correcting either alone costs 24-33 mAP@50, correcting both gains 12-32. The predicted
canopy mask's inter-crown dips are currently the system's real seam detector, while the
boundary head B is barely used.

If joint consistency is the active ingredient, it should be partly imposable *at decode
time*, with no retraining, by carving the boundary ridge directly into the gate:

```
gate = canopy - lambda * boundary
markers[gate < 0.25] = 1
```

Every pixel the boundary head believes is a seam becomes background, so the mask edge
and the ridge field agree by construction. This costs minutes; the training change it
stands in for costs hours.

## Design

Sweep `lambda in {0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0}` at the H1-tuned seeding
(`md=160, ma=0.50`), flat-basin relief unchanged, on `eval30`; freeze the best and
re-score on `evalhold60`. Report with and without the H2 seam merge
(`tau=0.15, L=30`), since arm E was the best known non-oracle configuration.

`lambda = 0` reproduces the current system exactly and is the control.

## Predictions (committed before running)

1. Best `lambda > 0` beats `lambda = 0` by **>= +2 mAP@50 on the hold-out**.
2. The gain comes with an SQ increase of **>= +1.5**, not from recognition alone —
   if consistency is really the mechanism, mask geometry should improve too.
3. The best `lambda` is **<= 1.0**. Larger values over-erode the canopy and should
   fragment crowns (seed ratio rising above 1.1).

If prediction 1 fails, decode-time coupling is insufficient and the consistency must be
learned, so the project proceeds directly to H5 (seam-excluded canopy target +
shared-definition boundary target, retraining `CanopyWatershedNet`).

## Outputs
- `results/h4_lambda_sweep.csv`, `results/h4_summary.csv`
