# Star-convex v5 — canopy-gate result

## Material Passport

- Origin: after `research.md` (external literature synthesis) confirmed the
  focal-loss diagnosis and staged Stage-1 (cheap) fixes ahead of an
  embedding/flow head, user chose to try canopy-gating first: an auxiliary
  binary canopy/background head that gates the object-probability map
  before peak-finding, targeting the 70%-pure-clutter false-positive mode
  from `star_convex_v3_failure_diagnosis.md`.
- Implementation (all additive/opt-in, default off, v2-v4 exactly
  reproducible without the new flags):
  - `star_convex_model.py::StarConvexNet(use_canopy_head=True)` -- a third
    `DenseHead` (sigmoid, 1 channel) alongside probability/rays.
  - `train_star_convex.py::PrecomputedStarDataset` derives the canopy
    target **for free** from already-cached data: `object_probability_map`
    is strictly 0 outside every instance and >0 (distance_transform_edt
    gives boundary-adjacent foreground pixels distance>=1, not 0) at every
    foreground pixel, so `probability > 0` recovers the exact canopy
    silhouette -- no re-precompute of the 1439-image cache was needed.
  - `star_convex_decode.py::decode(canopy=..., canopy_threshold=...)` zeros
    the probability map wherever predicted canopy < threshold, before
    peak-finding.
- Config: identical to v4 in every other respect (full 1439-image dataset,
  focal loss gamma=2.0, 60 epochs, same warm-started backbone) so the delta
  from v4 is attributable to the canopy head alone.
- Training was interrupted once by a session/sandbox teardown at epoch 5
  (not a code bug -- the nohup+disown process itself was killed, unlike
  earlier v2-v4 runs where the same pattern survived); resumed cleanly from
  the epoch-5 checkpoint via the existing resume logic, using `setsid` this
  time for a more robust detach. Total wall time epoch 0->59 (including the
  gap): 2026-09-11 15:14 -> 2026-09-12 12:18.
- Decode threshold re-swept including the new `canopy_threshold` dimension
  (108 configs: `experiments/star_convex_screen_v5/decode_sweep.csv`).
  Adopted: `prob_threshold=0.5, min_peak_distance=8, nms_iou_threshold=0.2,
  canopy_threshold=0.7`.

## Result 1 — aggregate metrics, BAM_val, 50 images (best-F1 decode config each)

| | v4 (no canopy) | **v5 (+ canopy-gate)** | G1B baseline |
|---|---:|---:|---:|
| matched_iou | 0.7329 | **0.7401** (+1.0%) | 0.76-0.80 |
| precision | 0.4565 | 0.4587 (~flat) | — |
| recall | 0.6086 | **0.6610** (+8.6%) | ~90% |
| f1 | 0.5217 | **0.5416** (+3.8%) | ~0.60+ |
| split_rate | 0.0384 | 0.0502 (+31%, worse) | 0.168-0.198 |
| miss_rate | 0.2585 | **0.2031** (-21%, better) | 0.045-0.062 |

**Recall and miss_rate moved the most they have across the entire v2-v5
sequence** -- every prior change (more data, focal loss) left recall stuck
in the 55-61% band; canopy-gating is the first change to break out of it
(66.1%).

## Result 2 — per-instance failure diagnosis (same method as v3/v4)

| | v3 | v4 | **v5** |
|---|---:|---:|---:|
| n_misses (/50 val images) | 332 | 349 | **275** |
| n_matches | 815 | 824 | **895** |
| n_false_positives | 1075 | 981 | 1056 |
| FP with zero GT overlap (pure clutter) | 70% | 70% | **65%** |

Hand-picked preview images (same two images used for v3/v4, for direct
comparison): `images/star_convex_v5_failure_diagnosis_val203.png` (dense
stand: 44 missed [v3] -> 38 [v4] -> **32 [v5]**, a consistent
version-over-version improvement) and `_val23.png` (18 GT: missed
8->5->**2**, false positives 31->24->**22** -- both failure modes improve
together on this image).

## Interpretation — canopy-gate helped, but through a different mechanism than designed

The original hypothesis (canopy-gate fixes the *clutter-FP* failure mode
specifically) is only **partly** confirmed: pure-clutter fraction did drop
(70%->65%), a real but modest effect, consistent with v4's focal-loss
result already having taken a bite out of the same problem. It did **not**
make clutter disappear, and total FP count did not drop (981->1056,
slightly up) -- because n_pred rose overall (more peaks accepted).

The bigger, less-predicted effect is on **recall/miss_rate**, which no
prior change had moved. The likely mechanism: the canopy head is a
multi-task auxiliary loss on the shared backbone/FPN features, not purely a
decode-time filter -- training the backbone to also answer "is this crown
material at all" appears to sharpen the shared feature representation in a
way that helps the *primary* probability head separate true crown centers
from background more confidently too, independent of the explicit gating
step. This is a genuinely useful, if not fully anticipated, result: **the
side effect (auxiliary-task regularization) mattered more than the intended
mechanism (decode-time gating)** for this specific gain.

split_rate got worse (+31% relative), consistent with more peaks overall
being accepted at the adopted threshold -- some of the recall gain trades
against a bit more over-segmentation, though split_rate (0.050) is still
far below the G1B baseline's 0.168-0.198.

## What this means for the next step

Canopy-gating is a real, positive, cheap win (f1 +3.8%, matched_iou +1.0%,
miss_rate -21%) -- confirms Stage 1 was worth doing before an embedding
head. But the two originally-diagnosed failure modes are **still both
present, just smaller**: 65% of FPs are still pure clutter, and the dense
homogeneous-stand preview (`val203`) still shows a large connected cluster
of missed, visually-near-identical crowns on the right side of the image --
the same structural pattern as v3/v4, just with more of the easier
instances now caught. This is consistent with `research.md`'s own staging
logic: Stage 1 (cheap fixes) narrows the gap but does not close it, and the
remaining gap in that specific dense-cluster region still looks like the
structural problem an explicit inter-instance signal (Stage 2: embedding
head, or Stage 3: flow-field decoder) targets, since gating/reweighting
alone cannot tell the network where the boundary between two touching,
visually-identical crowns falls.
