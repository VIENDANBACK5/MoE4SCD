# Star-convex v6 — boundary-weight loss result

## Material Passport

- Origin: second Stage-1 fix from `research.md`'s staged recommendations,
  after canopy-gating (v5). Adds a U-Net-style (Ronneberger 2015) boundary
  weight map that up-weights the probability-head loss on the thin
  background ridge between two close/touching instances, targeting the
  dense-stand instance-separation failure mode that canopy-gating (a
  foreground/background fix) did not address.
- Implementation (additive/opt-in, `--use-boundary-weight`, default off):
  `star_convex_targets.py::boundary_weight_map` (new function, 2 new unit
  tests), backfilled into all 1439 train + 50 val cached targets via new
  `code/add_boundary_weights.py` (cheap: ~0.3-0.7s/image, distance
  transform only, no ray-marching) rather than recomputing the expensive
  (probability, rays) targets. w0=10 (original paper's constant), sigma=10px
  (matches the observed touching-crown gap scale in this project's own
  failure-diagnosis previews, not an arbitrary guess).
- Config: identical to v5 in every other respect (full 1439-image dataset,
  focal loss, canopy head, 60 epochs) so the delta from v5 is attributable
  to the boundary-weight term alone.
- Two operational incidents during this run, both fixed and documented in
  code comments (not just here): (1) `add_boundary_weights.py`'s in-place
  `np.savez_compressed` overwrite was killed mid-write once by a
  session/sandbox teardown, corrupting `bam:train:719`'s cached target
  (lost the original probability/rays, not just the new channel) --
  regenerated via `precompute_star_targets.py` and fixed both scripts to
  write-to-temp-then-atomic-rename. (2) `np.savez_compressed` silently
  appends `.npz` to any path not already ending in exactly that suffix, so
  a naive `.npz.tmp` temp name was actually written as `.npz.tmp.npz`,
  breaking the rename -- fixed to `.tmp.npz`. (3) The bare `python3` on
  PATH resolved to a broken torch/torchvision combination partway through
  this session (unrelated to this project's code); found and fixed by
  invoking `./cs2_venv/bin/python3` explicitly for all training/inference
  commands going forward.
- Decode threshold re-swept including canopy_threshold (108 configs,
  `experiments/star_convex_screen_v6/decode_sweep.csv`). Adopted:
  `prob_threshold=0.5, min_peak_distance=3, nms_iou_threshold=0.2,
  canopy_threshold=0.7`.

## Result 1 — aggregate metrics, BAM_val, 50 images (best-F1 decode config each)

| | v5 (canopy only) | **v6 (+ boundary-weight)** |
|---|---:|---:|
| matched_iou | 0.7401 | 0.7330 (-1.0%) |
| precision | 0.4587 | 0.4474 (-2.5%) |
| recall | 0.6610 | 0.6780 (+2.6%) |
| f1 | 0.5416 | 0.5390 (-0.5%, ~flat) |
| split_rate | 0.0502 | 0.0672 (+34%, worse) |
| miss_rate | 0.2031 | 0.1817 (-10.5%, better) |

At the best-F1 operating point, this is a **wash, not a win** -- f1 is
essentially unchanged, and split_rate got meaningfully worse while
miss_rate improved. Unlike v5's clear, unambiguous gain over v4, v6 is a
mixed result at the aggregate level.

## Result 2 — per-instance failure diagnosis: the mechanism worked, the outcome didn't (yet)

| | v5 | **v6** |
|---|---:|---:|
| n_misses (/50 val images) | 275 | **246** (-10.5%) |
| n_matches | 895 | **918** (+2.6%) |
| n_false_positives | 1056 | 1134 (+7.4%, worse) |
| pure-clutter FP fraction | 65% | 67% (~flat) |
| missed-crown prob_at_centroid (median) | 0.006 | **0.070** (~10x higher) |
| missed-crown prob_at_centroid > 0.1 (fraction) | 11% | **36%** |

**This is the most interesting finding of this run.** The boundary-weight
term did exactly what it was designed to do at the mechanism level: pixels
near touching-instance boundaries that the network used to predict ~0
probability for now carry real, measurable signal (median probability at
still-missed centroids up 10x, more than 3x as many misses now have
*some* signal above 0.1). But this extra signal mostly falls short of the
0.5 decode threshold, so it does not yet convert into more correct
detections at the adopted operating point -- and where it does push
predictions over threshold, it comes with a comparable increase in false
positives (visible in `images/star_convex_v6_failure_diagnosis_val203.png`:
same missed-crown count as v5 on this image, but visibly larger/more
numerous yellow false-positive polygons in the dense cluster).

## Interpretation

Boundary-weight is a **real, verified partial mechanism improvement that
has not yet translated into a net metric win** -- different from canopy-
gating (v5), where both the mechanism and the outcome moved together. Two
non-exclusive explanations, neither yet tested:

1. **Hyperparameters are untuned.** w0=10/sigma=10 are the original paper's
   defaults for microscopy, not re-tuned for crowns at BAM's ~1.7cm GSD --
   this Stage-1 screen was about testing the mechanism, not sweeping its
   hyperparameters. A different sigma may sharpen the localization of where
   the extra signal appears rather than merely raising it everywhere near a
   boundary.
2. **A pixel-loss weight alone may be the wrong lever to fully close this
   gap**, matching `research.md`'s own staging logic: a boundary-weighted
   loss is still a *region/probability*-based signal (Section A2 in that
   report), not the *explicit inter-instance repulsion* signal Section A3
   argues is the strongest structural fix (discriminative embedding loss,
   Neven/EmbedSeg). The v6 result -- more signal near boundaries, but not
   enough to reliably separate two specific neighboring instances -- is
   consistent with that report's prediction that pixel-reweighting
   mechanisms (focal loss, boundary weight) help but do not fully solve
   this specific failure mode, and that an embedding/flow head is the next
   lever with a qualitatively different mechanism.

## What this means for the next step

Both of the Stage-1 items tried so far (canopy-gate: clear win; boundary-
weight: proven mechanism, neutral outcome) have now been screened. Soft-
NMS/Adaptive-NMS (the remaining untried Stage-1 item) is a plausible next
cheap experiment, but the v2-v5 decode-sweep history already showed hard
NMS-threshold tuning alone stays in a narrow band regardless of parameters
-- soft-NMS is a different mechanism (continuous decay vs. hard cutoff) so
it is not strictly ruled out by that history, but it is a smaller
expected-value bet than what the evidence in this file points to: the
Stage-2 embedding/discriminative-loss head is now the best-supported next
experiment, since v6 directly demonstrates the ceiling of "add more signal
to the same region-based probability map" -- the network can be pushed to
notice a boundary exists, but reweighting/regularizing that same map
doesn't fully teach it whose boundary it is.
