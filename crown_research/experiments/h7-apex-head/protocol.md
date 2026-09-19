# H7 — Seeding from a scale-invariant apex head

**Status:** LOCKED (pre-registration) · 2026-09-19 · **runs after H6 completes** (shared GPU)
**Type:** CONFIRMATORY

## Diagnosis being acted on

The missed-crown analysis attributes almost all unseeded crowns to a single cause:

| setting | crowns missed | `never_lit` | `suppressed` | median area of missed |
|---|---:|---:|---:|---:|
| loose (md=24, ma=0.25) | 17.5% | **16.5%** | 0.0% | 1,899 px |
| tuned (md=160, ma=0.50) | 31.1% | **28.1%** | 2.2% | 5,341 px |

(hold-out; all-crown median area is 24,382 px, p10 is 2,554 px)

Missed crowns are **small** crowns — an order of magnitude below typical — whose
surface response never reaches `min_apex`. NMS is not the culprit.

The mechanism is the same per-instance normalisation that defeated H1:
`u_inst = mask_edt / mask_edt.max()` asserts a peak of 1.0 for every crown regardless
of size, but a crown a tenth the typical area has a correspondingly tiny EDT support,
and after FPN upsampling its peak is smoothed below threshold. The surface head cannot
represent what the target demands at small scales.

## The unused asset

`targets_sharp.compute_sharp_canopy_targets` already returns a fourth tensor,
`apex_target`: a Gaussian centred on each crown centroid with **`apex_sigma` fixed
independent of crown size**. `dataset.py:196,214` ships it as `batch["apex"]` on every
training step. `train_sharp.py` never reads it and `CanopyWatershedNet` emits only three
channels, so it is computed and discarded.

A fixed-sigma centroid peak is scale-invariant by construction — precisely the property
the surface head lacks.

## Intervention

1. Widen `CanopyWatershedNet`'s shared head from 3 to 4 output channels, adding
   `apex = sigmoid(logits[:, 3:4])`. Existing weights load by name with the mismatched
   final conv re-initialised; the other three channels' behaviour is preserved by
   warm-starting from H6's checkpoint.
2. Supervise `apex` with `batch["apex"]` using the same class-balanced objective H6
   validated on the boundary head (the apex target is even sparser, so an unweighted
   loss would collapse it the same way).
3. **Seed from the apex head instead of the surface head**; keep the surface head for
   everything else it is used for. Re-sweep `min_dist` and `min_apex` for the new head,
   since its response scale differs.

`train_apex.py`, a new file; `train_sharp.py` and `train_calibrated.py` stay untouched
so both earlier baselines remain reproducible.

## Predictions (committed before running)

1. **`never_lit` at the tuned operating point falls below 15%** (from 28.1%).
2. **Apex-head seeding beats surface-head seeding by >= +3 mAP@50** on the hold-out,
   at each method's own best `(min_dist, min_apex)`.
3. **Combined with H6's calibrated boundary head, the best end-to-end configuration
   reaches >= 73 mAP@50** on the hold-out.

If prediction 1 succeeds but 2 fails, the extra seeds are being found but are not
landing usefully — which would point at the flooding of small basins (`min_area = 25`
and the single-pixel watershed seam) rather than at detection.

## Dependency

H7 needs H6's checkpoint as its warm start, so it is queued behind H6 rather than run
concurrently. Its predictions are recorded now, before H6's results are known, so they
cannot be adjusted to fit them.

## Outputs
- `results/h7_missed_crowns.csv`, `results/h7_seed_sweep.csv`, `results/h7_system.csv`
