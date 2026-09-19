# Oracle test result: dense H/V representation — KILL

## Material Passport

- Origin: cheap pre-training validation of Section 3 in
  `method_design_dense_crown_separation_v1.md`, requested directly ("làm
  phần này ngay") before committing to building/training a prediction
  network for the dense `P_fg, H, V` representation.
- Verification Status: VERIFIED — tested on real BAM GT polygons (not
  synthetic), oracle-mode (targets built directly from ground truth, no
  network prediction error involved at all).
- Code: `crown_segmentation_research/code/dense_hv_representation.py`
  (target generation + marker-controlled-watershed decode),
  `crown_segmentation_research/code/oracle_test_hv_decode.py` (evaluation
  harness, reuses `benchmark.evaluator` for numbers directly comparable to
  G1B baseline and the Section-4 merge module).

## Method

`compute_hv_targets()` follows HoVer-Net's per-instance normalization: for
each instance, every pixel's horizontal/vertical distance to the centroid is
normalized to [-1, 1] independently per side (left/right, top/bottom) by
that side's own extent. `decode_instances()` computes the Sobel-gradient
separation energy `S = max(|dH/dx|, |dV/dy|)`, extracts markers as the
lowest-percentile-S region within each connected foreground blob, and runs
marker-controlled watershed. Both were unit-tested on synthetic shapes first
(4/4 tests pass, including a case that caught and fixed a real numerical
tie-fragility bug in the naive fixed-threshold version — see the code
docstrings for that fix).

**This is the maximally favorable case for the representation**: real BAM GT
polygons, no touching neighbours in the single-instance test, and the target
fields are mathematically exact (not predicted by a network with any
error).

## Result

On BAM_val, oracle-decoding **single, isolated real GT instances one at a
time** (no neighbouring crowns at all — the simplest possible case):

- **Only 6/47 instances (12.8%) decode cleanly to exactly 1 piece.**
- Fragment count distribution across the 47: min 1, up to 44 pieces for one
  real instance; median around 10-12 pieces per real crown.
- A median-area instance (38,735 px) decoded into **20 pieces**. The
  largest instance in the same image (180,966 px) decoded into **28
  pieces**.

Full-image evaluation (30 val images, all instances together, same protocol
as the Section-4 merge sweep): matched_iou 0.696, split_rate 0.173,
merge_rate 0.007, miss_rate 0.022, **n_pred=11,119 against n_gt=733**
(15.2x over-prediction). Tightening the marker threshold made it *worse*
(n_pred rose to 13,243, matched_iou fell to 0.645) — ruling out "just needs
threshold tuning" as an explanation.

## Interpretation

The representation itself is the problem, not decode hyperparameters or
network prediction error (there is no network here). Real tree crowns are
not star-convex/well-behaved with respect to "signed distance to centroid,
normalized per-axis by extent" — natural canopy silhouettes have lobes and
concavities where this parametrization is not smoothly monotonic, so the
Sobel gradient spikes inside a single true crown wherever local shape
curvature changes rapidly, not only at true crown-to-crown boundaries. This
is the same mechanism the literature synthesis (`literature/synthesis/00_SYNTHESIS.md`)
already flagged as a known gap ("every distance/vector/star-convex family
assumes exactly one center per instance... tree crowns are not always
star-convex"), but the severity found here — 87% of instances, not only
unusually large ones — is much worse than the design doc assumed when it was
written from literature alone, before this test existed.

## Verdict

```text
SECTION_3_DENSE_HV_REPRESENTATION = KILL (oracle-level failure)
```

Do not build or train a prediction network for `P_fg, H, V` as specified.
The failure is at the representation/decode level and would not be fixed by
better training data, more epochs, or a stronger encoder — the oracle
already had perfect information and still failed.

## What remains standing

Section 4 (multi-seed merge module) is **unaffected by this result** — it
operates on Mask R-CNN's own mask output, not on H/V fields, and already has
a real, positive, measured result on the full official BAM_test2 split
(matched_iou unchanged, split_rate -24.4%, see
`design_docs/method_design_crown_confidence_v2.md`). This oracle failure is
a reason to *not* pursue Section 3 further, not a reason to doubt Section 4.

The classifier-under-confidence problem (81% of G4C's 668 measured failures)
also remains open and untouched by this result — Section 3 was never
targeted at that failure mode in the first place (see
`method_design_crown_confidence_v2.md` Section 6).

## Honest note on why this wasn't caught earlier

The original design (`method_design_dense_crown_separation_v1.md`) was
written from the 10-paper literature synthesis before any oracle test
existed for this specific dataset's real crown shapes. The literature
correctly flagged the star-convexity assumption as a risk in the abstract,
but nothing in that review quantified *how often* real BAM crowns violate
it. This oracle test is exactly the kind of check that should run before,
not after, committing engineering time to a training pipeline -- it did its
job here.
