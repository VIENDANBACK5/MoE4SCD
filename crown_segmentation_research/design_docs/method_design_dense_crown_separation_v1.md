# Dense Crown Separation — Method Design v1

## Material Passport

- Origin: synthesizes (1) the original user brainstorm formulation
  `(P_crown, P_center, D_sep) -> marker-controlled watershed`, (2) the
  representation-family literature review in
  `crown_segmentation_research/literature/synthesis/00_SYNTHESIS.md` (10 papers),
  (3) measured BAM failure modes in
  `reports/g4b_scale_sensitivity_paired_validation.md` and
  `reports/g4c_missed_crown_characterization.md`, (4) the DTE-aerial mortality
  component-size QC in `reports/dte_aerial_mortality_component_size_qc.md`.
- Verification Status: DESIGN ONLY — not implemented, not trained, not run.
- Explicitly does not replace the standing gate from `g4c`: RPN/ROI-stage
  instrumentation of the frozen Mask R-CNN checkpoint is still the correct
  *experimental* next step to localize where recall is actually lost. This
  document is a parallel design activity (hypothesis formation from literature
  + EDA), not a claim that method work should start training before that
  instrumentation runs.

## 1. Problem restated

Individual tree-crown instance segmentation on BAM/DeadTrees-style aerial RGB.
Requirement from this session: the method must not have a discrete "generate a
candidate, maybe generate none" step, because that is the diagnosed mechanism
behind SAM2's and Mask R-CNN's measured miss rates (13-48% in
`DeadTrees/experiments/segmentation_comparison_v1/summary.json`; strict-miss
population in G4C characterized by small size + low local contrast).

## 2. Design principle: proposal-free by construction

Two-stage detectors (Mask R-CNN, SAM2 with point/box prompts) commit to an
object list before segmentation happens; anything absent from that list is
lost with no downstream recovery mechanism. The method below instead predicts
a dense field over *every* pixel first, and only forms instances by grouping
after that dense prediction exists. An object can still be **merged or split
incorrectly**, but it cannot be **silently absent** the way a never-generated
proposal is, because there is no discrete candidate-generation step to fail.

This is the same reasoning HoVer-Net uses for touching nuclei, extended here
with an explicit multi-seed branch because BAM's own G1B baseline behaves
differently from cell segmentation: individual crowns are large enough
relative to the image, and irregular enough in shape, that a single
center-per-instance assumption (shared by HoVer-Net, DCME, Cellpose, StarDist)
is expected to fail specifically on large/multi-lobed crowns — the exact
failure Freudenberg et al. (2022) measured independently (over-segmentation
2.4 polygons/tree on large crowns) and that this project's own brainstorm
called "Hướng C — Adaptive Multi-Seed."

## 3. Representation

For input image \(I\), predict three dense heads from a shared encoder-decoder
(U-Net-style, no region proposal network):

\[
I \rightarrow \big(P_{fg}(x,y),\; H(x,y),\; V(x,y)\big)
\]

- \(P_{fg}\): per-pixel foreground (crown) probability. Standard dense
  binary segmentation head; every pixel is classified, so foreground
  detection cannot omit a region the way an RPN can.
- \(H, V\): horizontal/vertical signed distance of each foreground pixel to
  the centroid of *the crown it belongs to*, normalized to \([-1, 1]\) within
  each instance's bounding box — this is HoVer-Net's representation, chosen
  over Deep-Watershed-style unit-direction vectors because H/V gradients give
  a documented, reproducible instance-separation signal (Section II-B of the
  HoVer-Net paper) without requiring a hand-tuned energy discretization.

Markers and separation energy are *derived*, not predicted directly:

\[
S(x,y) = \max\big(|\partial_x H|,\; |\partial_y V|\big)
\]

\[
M(x,y) = \sigma\big(\tau(H, V) - \tau(S, k)\big) \cdot \tau(P_{fg}, h)
\]

(Sobel-gradient marker extraction, same construction as HoVer-Net Eq. 6, with
thresholds \(h, k\) tuned on validation — not hand-picked without a source, see
Section 6.)

## 4. Multi-seed extension for large/irregular crowns (the actual novel piece)

None of the 10 reviewed papers solve this: every distance/vector/star-convex
family assumes exactly one center per instance. Apply marker-controlled
watershed over \((P_{fg}, S)\) to get an initial partition
\(\{R_1, \dots, R_K\}\), which may **over-segment** a single large crown into
several regions sharing one contiguous canopy. Then merge adjacent regions
\(R_i, R_j\) (sharing a boundary of length \(\geq\) 3 px) if:

\[
\text{merge}(R_i, R_j) \iff \| E(R_i) - E(R_j) \|_2 < \delta_E \;\wedge\;
d_{\text{phys}}(c_i, c_j) < \delta_d \cdot \text{GSD}
\]

where \(E(R)\) is a learned per-region appearance embedding (mean-pooled from
the same decoder features, in the spirit of the discriminative-loss embedding
family — De Brabandere et al. 2017 — but applied post-hoc to watershed regions
rather than replacing watershed), and \(\delta_d\) is expressed in **physical**
units (meters via GSD) rather than pixels, so the same merge rule transfers
across 5/10/20cm without retuning — this reuses the physical-scale
normalization already validated in
`experiments/g4a_physbound/model.py` (`A_physical = A_pixels * GSD^2`
convention, Section 3.2 of that design).

\(\delta_E, \delta_d\) are hyperparameters with no data-derived value yet;
Section 6 fixes how they will be chosen (grid search on BAM_val only, never on
test, per the frozen protocol in `benchmark/experiment_protocol.yaml`).

## 4bis. Classifier under-confidence fix (the actually-indicated priority, added after G4D)

**Correction (same session, before any of this was implemented):** the
original version of this section assumed size/low-contrast — the same
covariates driving G4C's strict-miss population — would also explain this
81% classifier under-confidence, and proposed context-widening/FPN-level
fixes on that premise. A cheap check (join `roi_score_vs_mask_split.csv` to
the already-computed `area_m2`/`rgb_contrast_l2` columns in
`per_crown_transitions.csv`, no new model run) found that premise **not
supported**: `tree_class_score` correlates with area at r=0.137 and with
contrast at r=0.021 — both too weak to be the driver (see
`reports/g4d_rpn_roi_instrumentation.md`, "is under-confidence explained by
size or contrast?"). The FPN-level and context-widening interventions below
are therefore **not yet justified by evidence** — they are plausible generic
small-object-detection techniques, not a targeted fix, until the real driver
is identified.

**What is actually indicated next: qualitative inspection, not another
architecture change.** Neither covariate this project has instrumented so
far (size, contrast) explains why the classifier is broadly unconfident
across this 541-crown population. Before designing or implementing any fix,
look directly at a sample of the images/proposal crops for these crowns
(same rendering approach already used for the DTE-aerial visual QC earlier
this session) to form a new hypothesis grounded in what is actually visible
-- e.g. species/canopy-texture appearance atypical of the training
distribution, occlusion, or a systematic difference in the BAM sites this
population concentrates in. Implementing either candidate below without that
step risks fixing a cause that was never diagnosed:

1. Check FPN level assignment for the 541 crowns by proposal size (still
   worth doing as one candidate covariate, just not a confirmed one).
2. Widen RoI context for the box classification branch (standard small-object
   technique, not from the 10-paper survey — flagged as such, not novel).

Given 81% of the measured failure sits at this stage, resolving *what drives
it* is still the higher priority than implementing Sections 3-4, but the next
step is looking at images, not writing more code.

## 5. GSD conditioning (reuse, not reinvent)

`experiments/g4a_physbound/model.py` already implements an identity-initialized
FiLM conditioner on \(z = \log(g/g_0)\) that was unit-tested (6/6 passing,
nonzero gradient checks) but never scientifically run because exact GSD
provenance for Tretzendorf was unresolved. That module is architecture-agnostic
(it modulates FPN-level features) and can be reattached to this dense U-Net's
encoder features unchanged. Do not re-derive this component; the blocker is
data provenance, not code.

## 6. What this design does and does not claim to fix

**Update after `reports/g4d_rpn_roi_instrumentation.md` (instrumentation now
run, including the score-vs-mask follow-up, not hypothetical):** RPN proposal
recall for `lost_strict_miss` crowns is 94-99.98% at IoU>=0.50 — proposals
overwhelmingly survive. Of the 668 crowns that still fail downstream, **81%
fail because the box classifier's tree-class score is too low (median 0.18
against a 0.40 threshold), not because the mask is wrong (only 2.8%)**. The
CPP-Net-style intervention originally planned below (Section "large/irregular
crowns") targets *mask/boundary* quality via multi-point context sampling —
that is aimed at the wrong 2.8%, not the dominant 81%. The row below is
revised accordingly, and Section 4bis adds the actually-indicated
intervention.

| BAM failure mode (measured) | Does this design address it? | Why |
|---|---|---|
| Strict miss / partial overlap, classifier under-confidence (81% of 668 failing crowns) | **Not by the original Section 4/6 plan** — needs Section 4bis below | The failure is upstream of mask decoding, in the box head's confidence on an already-well-localized region. Multi-point context sampling around a *centroid* (CPP-Net's original use case) does not obviously transfer to *classification* confidence the way it does to boundary regression; treat Section 4bis as the higher-priority arm, not the mask-quality fix originally planned here. |
| Mask decode quality (2.8% of 668 failing crowns) | Yes, low priority now | CPP-Net-style multi-point context remains applicable here, but this is a small minority of the measured failure — do not lead the method-design effort with it. |
| Partial-overlap failure (weak localization, size-independent) | **Yes, directly targeted** | CPP-Net's finding — centroid-pixel-alone lacks context — applies equally to H/V regression here. Add CPP-Net-style multi-point sampling: for each region \(R\), sample \(N\) points along rays from its current centroid estimate toward the current boundary estimate, average their H/V predictions weighted by a learned confidence (Chen et al. 2023, Section 3), instead of trusting the single centroid pixel. |
| Large/multi-peak crown over-segmentation (Freudenberg, independent dataset) | **Yes, this is the primary novel contribution** | Section 4 merge rule, not present in any reviewed paper. |
| Boundary quality under GSD shift | Already shown **not broken** (G4B, CI includes 0) | No boundary-specific module added; do not spend budget "fixing" a failure that measurement already ruled out. |

## 7. Minimal experiment sequence (design only, not yet authorized to run)

1. Reimplement \(P_{fg}, H, V\) heads on the existing G1B encoder backbone
   (reuse trained weights as initialization, same principle as G4A's
   warm-start control).
2. Screen on the same deterministic 300-image BAM-train / 50-image BAM-val
   subset G4A already defined, same seed 42, same promotion thresholds
   pattern (boundary/error-rate deltas with a pre-declared minimum effect
   size) — do not invent a new evaluation protocol per model.
3. Multi-seed merge module (Section 4) is evaluated as a separate ablation arm
   against the same watershed decode without merging, isolating its
   contribution exactly like G4A isolated boundary vs scale conditioning.
4. `Bamberg_coco2048.zip` has been re-downloaded and verified
   (`raw_archives/Bamberg_coco2048.zip`, exact size + file-count match against
   `data_manifest/provenance.md`, spot-read via `/vsizip/` confirmed) — step 1
   is no longer blocked on data availability.

## 8. Open questions this design does not resolve

- ~~Whether the encoder bottleneck (strict-miss population) is worth
  attacking before or in parallel with Section 4~~ — **resolved**: G4D showed
  it is not an encoder/RPN bottleneck; no separate encoder fix is indicated.
- Whether the ROI-stage loss for `lost_strict_miss`/`lost_partial_overlap` is
  dominated by classification score or by mask/box quality (G4D Section
  "What this does not resolve") — still open, and matters for whether the
  CPP-Net-style multi-point-context head (Section 6) is the right first
  intervention or whether a classification-threshold/calibration issue should
  be checked first, since it is cheaper to check than to implement.
- \(\delta_E, \delta_d\) have no validated values; treat any number quoted for
  them elsewhere as a placeholder until the Section 6 grid search runs.
