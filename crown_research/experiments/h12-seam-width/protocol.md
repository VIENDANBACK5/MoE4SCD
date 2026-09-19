# H12 — Widen the seam target: testing the usability-floor mechanism directly

**Status:** LOCKED (pre-registration) · 2026-09-20, **before H11's result is known**
**Type:** CONFIRMATORY · queued behind H11 (shared GPU)

## Why this is the right last experiment

Three independent head failures in this project have the same shape, and the same
lesson: **sparse targets collapse under any loss weighting, and the fix is target
geometry, not loss weighting.**

- the boundary head (~6% positive pixels) collapsed to median 0.001; class-balanced
  weighting rescued it only partway, 0.105 → 0.186 against the 0.40 aimed at (H6);
- the apex head (<1% positive pixels) never learned at all, Tversky flat at 1.00 (H7);
- H5's rank transform, which tried to recover magnitude post-hoc, amplified noise (−9.4).

H6's own pre-registered fallback named the remedy: *"the seam band is 3 px wide and
blurred at sigma 0.8, which may simply be too thin to supervise at this resolution."*

H10 then promoted that from an accuracy tweak to a **robustness** lever. Its mechanism
finding was that the domain shift costs both checkpoints the **same ~40%** of seam
magnitude, and what decides the outcome is the **absolute level** they land on relative
to a usability floor (~14/255 fails, ~27/255 works). If that is right, extra magnitude
must pay **disproportionately out-of-domain**. This experiment tests that prediction
directly rather than assuming it.

## Intervention (single factor)

`train_wide.py`, a copy of `train_calibrated.py` with **only the boundary target
widened**, on the fly, in the training loop:

```python
bound_wide = F.max_pool2d(bound_tgt, kernel_size=5, stride=1, padding=2)
```

A 5x5 morphological dilation takes the ~3 px seam band to ~7 px while preserving peak
values, roughly tripling positive area (~6% → ~15%). Done on GPU in the loop so
`dataset.py`, `targets_sharp.py`, `train_sharp.py` and `train_calibrated.py` all stay
untouched and every earlier baseline stays reproducible.

Warm-started from `best_calibrated_watershed.pth`, 12 epochs, same optimiser, schedule,
augmentation and seeds. Output `best_wide_watershed.pth`.

The matched-epoch control is **H11** (`best_calibrated_x2`), which trains 12 further
epochs from the same start with the target unchanged — so H12 vs H11 isolates the target
width, not the extra epochs.

## Predictions (committed before running, and before H11's result is known)

1. **Magnitude rises.** Mean predicted B on true seam pixels exceeds **0.30**
   on `evalhold60` (from 0.186).
2. **In-domain gain is modest.** `AP(H12) − AP(H11)` on `evalhold60` is in
   **[−1.0, +2.0]** — widening the target is not expected to be an in-domain win, and
   may cost a little precision by thickening predicted seams.
3. **The floor mechanism holds: out-of-domain gains more, relatively.**
   `ΔAP_rel(test1) > ΔAP_rel(test2)` where each Δ is H12 against H11 at matched decode.

Prediction 3 is the one that matters. It is a direct consequence of H10's mechanism and
would fail if H10's account were wrong.

A note on evaluation: predicted seams will be thicker, so `min_dist`/`min_apex` are
re-swept per checkpoint over the same grid, and every comparison is made at each
checkpoint's own best cell **and** at a common cell.

## Anticipated failure modes
- If prediction 1 fails, target width is not the lever either and the head's magnitude
  ceiling is architectural (the final 1x1 conv on a stride-1 feature map), which would
  be a different and larger change.
- If 1 passes and 3 fails, H10's floor account is wrong and the out-of-domain benefit of
  calibration came from something else — which would need explaining before any of H10
  is written up as mechanism rather than correlation.

## Outputs
- `results/h12_sweep.csv`, `results/h12_summary.csv`, `results/h12_head_diagnostic.csv`
