# Star-convex v7 — discriminative embedding head result

## Material Passport

- Origin: direct follow-up to `star_convex_v6_boundary_weight_result.md`'s
  "What this means for the next step" conclusion -- v6 showed that adding
  more signal to the same region/probability map (boundary-weight loss)
  does not reliably teach the network *whose* boundary a pixel is near.
  v7 tries the qualitatively different lever that conclusion pointed to:
  an explicit inter-instance repulsion signal via a discriminative
  instance-embedding head (De Brabandere, Neven, Van Gool 2017,
  arXiv:1708.02551).
- Implementation (additive/opt-in, `--use-embedding-head`, default off):
  `star_convex_model.py`'s `DenseHead(fpn_out_channels, embedding_dim=8)`
  embedding head, trained with the new standalone, fully-vectorized
  `code/discriminative_loss.py` (`delta_v=0.5, delta_d=1.5` -- paper
  defaults, `var_weight=dist_weight=1.0, reg_weight=0.001`). At decode
  time, `star_convex_decode.py::decode` samples the embedding at each
  detected peak and passes it into `polygon_nms`, which keeps two
  high-IoU peaks apart (instead of collapsing them, as plain IoU-NMS
  would) when their embedding distance exceeds `embedding_delta_d`.
- **Correction (caught during the deeper failure analysis below):** the
  actual training command used for v7
  (`crown_segmentation_research/code/train_star_convex.py --use-focal-loss
  --use-canopy-head --use-embedding-head ...`, no `--use-boundary-weight`)
  did **not** carry v6's boundary-weight loss forward -- it is OFF in v7,
  while it was ON in v6. This is a confound: the v6->v7 delta below is
  attributable to *both* "boundary-weight removed" and "embedding head
  added", not to the embedding head alone as originally written here.
  Otherwise same as v6 (full 1439-image dataset, focal loss, canopy head).
  Backbone warm-started from `g1b_baselines/maskrcnn_seed42_best.pth`
  (strict=True, no missing/unexpected keys). 60 epochs, completed in
  under a day (not the ~40h worst case estimated beforehand).
- Training loss: total 8.758 (epoch 0) -> 1.387 (epoch 59); embedding_loss
  0.318 -> 0.0162, decreasing every epoch with no sign of divergence
  (`experiments/star_convex_screen_v7/training_history.json`).
- Decode threshold re-swept with a new `embedding_delta_d` axis added
  (162 configs: `prob_threshold` x3, `min_peak_distance` x3, `nms_iou` x3,
  `canopy_threshold` x2, `embedding_delta_d` x3 in
  {0.75, 1.5 (training default), 3.0}; `experiments/star_convex_screen_v7/decode_sweep.csv`).
  Adopted (best F1): `prob_threshold=0.5, min_peak_distance=5,
  nms_iou_threshold=0.2, canopy_threshold=0.7, embedding_delta_d=3.0`
  (`experiments/star_convex_screen_v7/adopted_decode_config.json`) --
  note the best decode-time delta_d (3.0) is 2x the training-time
  delta_d (1.5), i.e. NMS wants a wider embedding margin than the loss
  was trained to enforce before it trusts two peaks are different trees.

## Result — aggregate metrics, BAM_val, 50 images (best-F1 decode config each)

| | v6 (+boundary-weight) | **v7 (+embedding head)** |
|---|---:|---:|
| matched_iou | 0.7330 | 0.7265 (-0.9%) |
| precision | 0.4474 | 0.4569 (+2.1%) |
| recall | 0.6780 | 0.6499 (-4.1%) |
| f1 | 0.5390 | 0.5366 (-0.4%, ~flat) |
| **split_rate** | 0.0672 | **0.0332 (-51%)** |
| merge_rate | -- | 0.0379 |
| miss_rate | 0.1817 | 0.2097 (+15%, worse) |
| n_pred | -- | 1926 |

Same pattern as v6: **f1 is a wash at the aggregate level**, but unlike v6
(where the mechanism moved and the outcome didn't), v7's mechanism *and*
outcome both move -- just in different, partly-offsetting directions.
Embedding-aware NMS does exactly what it was designed to do: split_rate
(one tree wrongly cut into multiple predictions) is cut roughly in half.
That gain is paid for by a higher miss_rate -- the wider embedding margin
needed at decode time (`embedding_delta_d=3.0`) also raises the bar for
keeping some genuinely-separate low-confidence peaks, and precision goes
up only marginally (+2.1%) while recall drops (-4.1%), for a near-zero net
change in f1.

## Qualitative preview

`images/star_convex_v7_failure_diagnosis_203.png` and `_23.png` (green =
GT, yellow = predicted, adopted config, same two val images used in v6's
failure diagnosis for direct comparison):

- `val:203` (dense stand, n_gt=85, n_pred=48): predictions track GT well
  on large/separated crowns with little visible over-splitting, but a
  large fraction of GT crowns in the denser interior have no matching
  prediction at all -- the qualitative face of the +15% miss_rate.
- `val:23` (sparser stand, n_gt=38, n_pred=35): much closer prediction
  count to GT count, most yellow outlines track their green GT closely --
  embedding-aware NMS appears to help most where crowns are already
  reasonably separated, and least in dense/touching clusters, which is
  the opposite of where v6's boundary-weight signal was strongest.

## Deeper failure analysis: the miss population changed *kind*, not just size

Ran `code/diagnose_star_convex_failures.py` (now embedding-aware) on v7 at
the adopted config, and compared per-instance miss/match/FP CSVs directly
against v6's stored diagnosis
(`experiments/star_convex_screen_{v6,v7}/failure_diagnosis/`):

| | v6 | v7 |
|---|---:|---:|
| n_misses | 246 | 284 (+15%) |
| n_matches | 918 | 880 (-4%) |
| n_fp | 1134 | 1046 (-8%) |
| missed-crown `prob_at_centroid` **< 0.01** ("blind spot": network found nothing here at all) | **4%** | **70%** |
| missed-crown `prob_at_centroid` in [0.01, 0.1) | 59% | 18% |
| missed-crown `prob_at_centroid` **>= 0.1** ("borderline": some real signal) | 36% | 12% |
| miss with `prob_max_in_bbox >= prob_threshold` anyway (miss caused by canopy-gate/NMS/min-peak-distance suppression downstream of the probability map, *not* a probability-threshold problem) | 23% | 23% (unchanged) |

This is the key finding, and it reframes the whole v6->v7 comparison. In
v6, the extra misses were overwhelmingly **borderline** -- the network
found *some* signal (median prob_at_centroid 0.070) but not enough to
clear the decode threshold, i.e. a genuine threshold/calibration
trade-off, recoverable in principle by loosening thresholds. In v7, 70%
of misses are **blind spots** the network produces essentially zero signal
for (median prob_at_centroid 0.003) -- these crowns are architecturally
undetected, not merely under-thresholded, and no decode-time retuning
(prob_threshold, canopy_threshold, embedding_delta_d, min_peak_distance)
can recover them, since decode only acts on peaks the probability map
already contains.

Two things this analysis rules out as the cause:
- **Not a decode-pipeline suppression artifact.** The
  `prob_max_in_bbox >= threshold` fraction (misses where a real detectable
  peak existed but was discarded by canopy-gate / NMS / min-peak-distance)
  is identical in v6 and v7 (23%) -- so embedding-aware NMS, canopy
  gating, and the peak-finding step are not the source of the new blind
  spots.
- **Not an aggregate probability-head regression.** v6 and v7's
  `probability_loss` training trajectories are nearly identical epoch-for-
  epoch (both converge to ~0.0427 by epoch 59) -- so this is not the
  probability head globally getting worse at its job. It looks like a
  **redistribution of where the network places detection confidence**,
  plausibly from the shared-backbone gradient interaction between the new
  embedding loss and the probability loss, not a capacity or convergence
  problem visible in the aggregate numbers.

This also means Interpretation-point-2 from the original write-up above
(*"a lower prob_threshold should recover recall"*) is **only partially
right**: the sweep data confirms fixing `embedding_delta_d=3.0` and
lowering `prob_threshold` from 0.5 to 0.3 does recover some recall (miss_rate
0.210 -> 0.179, better than v6's 0.182) --

| prob_threshold | canopy_threshold | precision | recall | f1 | split_rate | miss_rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 (adopted) | 0.7 | 0.457 | 0.650 | **0.537** | 0.033 | 0.210 |
| 0.3 | 0.5 | 0.431 | 0.672 | 0.525 | 0.048 | **0.179** |

-- but this only recovers the *borderline* misses that still existed in
the swept grid; it cannot touch the 70%-blind-spot population, which is
a network-level phenomenon, not a threshold one.

**Bonus finding from the existing sweep** (no new compute, just re-reading
`decode_sweep.csv`): 3 of the 162 swept configs (all `prob_threshold=0.3,
canopy_threshold=0.5, embedding_delta_d=3.0`, varying `min_peak_distance`
in {3, 5, 8}) **dominate v6 on split_rate AND miss_rate simultaneously**
(e.g. `min_peak_distance=8`: split_rate=0.044 vs v6's 0.067, miss_rate=
0.179 vs v6's 0.182) at a small aggregate-F1 cost (0.527 vs v6's 0.539,
driven by precision 0.433 vs 0.447). Depending on the downstream use case
(if double-counting/over-splitting a tree is worse than a small precision
hit), this is arguably a better adoption candidate than the pure-best-F1
config currently in `adopted_decode_config.json`.

## Instance-level join: is it the same crowns, or a redistribution?

`diagnose_star_convex_failures.py` now records GT `instance_id` per row.
Reran v6 (its original adopted config) and v7 (now the split+miss-
dominant config above, not the old best-F1 pick) and joined on
`(image_id, instance_id)` across the same 50 val images:

| | count |
|---|---:|
| v6 n_miss / v7 n_miss | 246 / 242 (roughly flat with the dominant config -- confirms `prob_threshold=0.3` recovers most of what looked like a net miss_rate regression under the old best-F1 config) |
| Missed in **both** v6 and v7 (persistent-hard) | 148 |
| Matched in v6, newly **missed** in v7 (regressions) | 68 |
| Missed in v6, newly **matched** in v7 (improvements) | 66 |
| Matched in both | 791 |

This confirms the **redistribution hypothesis directly**: it is not simply
"v7 forgot 15% more trees" -- 66 trees v6 couldn't find are now correctly
detected in v7 (median IoU 0.671, not marginal), almost exactly offsetting
68 trees v6 had that v7 now misses. Net miss count is roughly a wash at
this config.

But the 68 regressions are not a random draw: their `prob_at_centroid` is
just as blind as the persistent-hard cases (median 0.004, 71% under 0.01 --
essentially indistinguishable from the persistent set's 0.002/81%), and
they are **larger, not smaller**, than the persistent-hard misses (median
area 34,902 px^2 vs 18,177 px^2) -- i.e. these are not intrinsically
"hard" crowns by size; v6 detected them fine, and v7's network genuinely
stopped producing signal for them specifically. Combined with the
identical decode-pipeline-suppression rate (23%, unchanged, from the
earlier analysis), this rules out decode config as the explanation for
the regressions and narrows the cause to the training-time interaction
between the embedding loss and the probability head's shared backbone.

## Training-level ablation: v8 (embedding-loss-weight=0.3)

Launched to test the gradient-competition hypothesis directly. Rather than
touching `discriminative_loss.py`'s internal `var_weight`/`dist_weight`
(which shape the pull/push loss *between instances*, not its overall
influence on the shared backbone), used the existing
`train_star_convex.py --embedding-loss-weight` knob (already exposed,
default 1.0, scales the embedding loss's contribution to `total`) --
the more direct lever for "is the embedding task competing with the
probability task for backbone capacity." v8 command is byte-for-byte
identical to v7's (`run_train_v7.sh`) except `--embedding-loss-weight 0.3`
(vs v7's 1.0), so the only variable under test is that weight; everything
else (data, epochs=60, focal loss, canopy head, warm-start checkpoint,
boundary-weight still off to match v7 exactly) is held fixed. If v8's
blind-spot fraction among misses drops back toward v6's 4% (from v7's
~70-81%), that confirms embedding-loss magnitude, not the embedding task
per se, is driving the regressions. Still running as of this analysis
(expect ~1 day based on v7's wall-clock time);
`experiments/star_convex_screen_v8/training_history.json` and a rerun of
the same diagnosis pipeline will confirm or refute this once it
completes.

## v8 result: embedding-loss-weight=0.3 ablation (confirms, partially)

v8 finished 60 epochs (`training_history.json`: total 9.797 -> 1.405,
embedding_loss 0.457 -> 0.0289 -- converges cleanly, same as v7).
`decode_sweep.csv` re-swept (same 162-config grid). At the **identical
decode config** used for v7's and v6's best-F1 comparison
(`prob_threshold=0.5, min_peak_distance=5, nms_iou=0.2, canopy_threshold=0.7,
embedding_delta_d=3.0`) for a clean apples-to-apples read:

| | v6 | v7 (embedding-loss-weight=1.0) | **v8 (embedding-loss-weight=0.3)** |
|---|---:|---:|---:|
| matched_iou | 0.7330 | 0.7265 | 0.7389 |
| precision | 0.4474 | 0.4569 | **0.4696** |
| recall | 0.6780 | 0.6499 | 0.6551 |
| **f1** | 0.5390 | 0.5366 | **0.5470** (best of all three) |
| split_rate | 0.0672 | 0.0332 | 0.0428 |
| miss_rate | 0.1817 | 0.2097 | 0.2053 |
| n_match / n_miss (of ~1164 GT) | 918 / 246 | 880 / 284 | 887 / 278 |
| n_fp | 1134 | 1046 | **1002** (lowest of all three) |
| **blind-spot fraction of misses** (`prob_at_centroid<0.01`) | **4%** | **70%** | **53%** |

Two things at once:

1. **The gradient-competition hypothesis is confirmed, partially.**
   Lowering `embedding_loss_weight` from 1.0 to 0.3 (nothing else changed)
   dropped the blind-spot fraction from 70% to 53% -- a real, substantial
   effect in the predicted direction. But it is still nowhere near v6's
   4%, so embedding-loss magnitude is **a** contributing cause, not the
   whole story. Either a lower weight than 0.3 would continue closing the
   gap (untested), or part of the blind-spot effect has a different cause
   entirely (e.g. the embedding head's architecture/backbone placement
   itself, independent of its loss weight, or the boundary-weight-loss
   confound noted at the top of this doc).
2. **v8 is a net win, not just a mitigated trade-off.** F1 (0.547) beats
   *both* v6 (0.539) and v7 (0.537) at this decode config -- precision is
   the highest of the three (0.470) and false-positive count the lowest
   (1002), while split_rate stays far below v6's (0.043 vs 0.067, most of
   v7's split-rate win preserved) and recall/miss_rate land between v6 and
   v7. Unlike v7, lowering the embedding loss weight did not trade away
   the split_rate gain to get here.

`experiments/star_convex_screen_v8/adopted_decode_config.json` records
this as v8's adopted config. Note: no v8 config in the swept grid
dominates v6 on both split_rate *and* miss_rate simultaneously (unlike
v7's 3 dominant configs), so best-F1 is the natural pick here rather than
a dominance argument.

`images/star_convex_v8_failure_diagnosis_203.png` and `_23.png` (same two
val images, same green=GT/yellow=pred convention): `val:203` n_pred rose
from v7's 48 to **50** (closer to n_gt=85) with visibly tighter tracking
of GT outlines in the dense interior; `val:23` stayed at n_pred=35/n_gt=38,
essentially unchanged from v7 -- consistent with the aggregate numbers
showing v8's gain concentrated in recovering some of v7's dense-stand
misses rather than changing the already-good sparse-stand behavior.

## Interpretation

The discriminative-embedding lever is a **real, working mechanism at the
decode level** (split_rate cut ~50-60%, exactly as designed), but it
appears to come bundled with a **training-level side effect**: a
meaningful chunk of crowns the network used to detect with real confidence
in v6 now get essentially zero probability signal in v7. That side effect,
not decode-threshold miscalibration, is the dominant driver of v7's higher
miss_rate.

## What this means for the next step

All three follow-ups above have been actioned:

1. **Done** -- instance-level join above confirms redistribution (66
   improvements vs 68 regressions), not a net loss, but the regressions
   are genuine training-level blind spots (not decode-recoverable, not
   smaller/harder crowns by size).
2. **In progress** -- v8 (`embedding-loss-weight=0.3`) launched to test
   the gradient-competition hypothesis; result pending (see above).
3. **Done** -- `adopted_decode_config.json` now points at the
   split_rate+miss_rate-dominant config (`prob_threshold=0.3,
   canopy_threshold=0.5, min_peak_distance=8, nms_iou_threshold=0.2,
   embedding_delta_d=3.0`), which strictly dominates v6 on split_rate
   (0.044 vs 0.067) and is roughly at parity on miss_rate (242 vs 246
   instances) once evaluated at the instance level, in exchange for a
   small aggregate-F1 cost.

**v8 update: done, see "v8 result" section above.** Blind-spot fraction
dropped 70% -> 53% (partial confirmation, not full), and f1 (0.547) beat
both v6 and v7 at the shared best-F1 decode config -- **v8 is now the
best checkpoint of the three and the recommended working checkpoint**
going forward.

Two directions worth trying next, in order of expected value:

1. **Push `embedding_loss_weight` lower still** (e.g. 0.1, or a small
   sweep {0.05, 0.1, 0.2} rather than guessing one value) to see if the
   blind-spot fraction keeps closing toward v6's 4% with a monotonic
   dose-response, or plateaus -- a plateau above v6's level would point
   at a structural cause (embedding head placement in the shared FPN)
   rather than pure loss-weight competition, which is a materially
   different (larger) fix.
2. **Re-add `--use-boundary-weight`** (the confound flagged at the top of
   this doc) to a v8-style run, isolating it as its own variable --
   v6 had it on, v7/v8 do not; it is not yet known how much of the
   remaining gap to v6's 4% blind-spot rate is attributable to its
   absence rather than the embedding task itself.
