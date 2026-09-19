# Star-convex v4 — full dataset + focal loss result

## Material Passport

- Origin: user chose "thử fix training trước" (try training-side fixes
  before designing a new method) after `star_convex_v3_failure_diagnosis.md`
  found two separable failure modes (dense-stand instance-separation blind
  spot; background/foreground confusion) that plain BCE + limited data are
  well-positioned to explain.
- Two changes from v3, both directly targeting the diagnosed failure modes:
  1. **Full BAM_train dataset** (1439/1439 images, vs v3's 536) — finally
     matches the G1B baseline's own training budget, closing the caveat
     flagged in `star_convex_v2_training_result.md`.
  2. **Soft-target focal loss** for the probability head (`--use-focal-loss
     --focal-gamma 2.0`, `code/train_star_convex.py::focal_bce_loss`)
     replacing plain BCE, behind an opt-in flag (plain BCE remains the
     default, preserving exact reproducibility of v2/v3). Down-weights
     already-correct (easy) pixels so gradient concentrates on hard
     pixels -- background the model currently over-predicts on, and
     touching-instance boundaries.
  3. Epochs reduced 150 -> 60 to hold the total gradient-update budget
     roughly comparable to v3 (150 epochs x 536 images = 80,400 updates;
     60 epochs x 1439 images = 86,340 updates), rather than arbitrarily
     multiplying total training time by dataset-size ratio.
- Training: 13:39 -> 04:21 (~14h42m), 60/60 epochs, no OOM/crash. Loss
  (focal + smooth-L1, NOT directly comparable in absolute value to v2/v3's
  plain-BCE loss numbers since focal loss is scaled down by construction):
  epoch 0 total=9.23 -> epoch 59 total=1.41, monotonically decreasing
  throughout, no plateau-then-divergence.
- Decode threshold re-swept (checkpoint-specific, as established):
  `experiments/star_convex_screen_v4/decode_sweep.csv`. Adopted:
  `prob_threshold=0.5, min_peak_distance=5, nms_iou_threshold=0.2`.

## Result 1 — aggregate metrics, BAM_val, 50 images (best-F1 decode config each)

| | v3 (536 img, plain BCE) | **v4 (1439 img, focal loss)** | G1B baseline (BAM_test2) |
|---|---:|---:|---:|
| matched_iou | 0.7155 | **0.7329** (+2.4%) | 0.7585-0.7951 |
| precision | 0.4312 | **0.4565** (+5.9%) | — |
| recall | 0.6019 | 0.6086 (+1.1%) | ~90% |
| f1 | 0.5025 | **0.5217** (+3.8%) | ~0.60+ |
| split_rate | 0.0332 | 0.0384 (+16%, worse) | 0.1676-0.1982 |
| miss_rate | 0.2452 | 0.2585 (+5%, worse) | 0.0446-0.0617 |

Real, positive net movement (f1, matched_iou, precision all up) but **not
uniformly better** -- split_rate and miss_rate both tick slightly worse.

## Result 2 — per-instance failure diagnosis (same method as v3's)

| | v3 | **v4** |
|---|---:|---:|
| n_misses | 332 | 349 (+5%, worse) |
| n_false_positives | 1075 | **981 (-8.7%, better)** |
| missed-crown prob_at_centroid (median) | 0.000 | 0.016 |
| missed-crown prob_at_centroid > 0.1 (fraction) | 5% | **13%** |
| FP with zero GT overlap (pure clutter) | 70% | 70% (unchanged) |

Preview images: `images/star_convex_v4_failure_diagnosis_val203.png` (same
dense-stand image as v3's, 44 missed -> **38 missed**), `_val23.png` (same
FP-heavy image, 8 missed -> 5 missed, 31 FP -> **24 FP**) -- both hand-picked
images improve visibly, though the *aggregate* miss count over all 50 images
went up slightly (332 -> 349), so this is genuine image-to-image variance,
not a uniform win.

## Interpretation — a real, partial, honest improvement, not a fix

The two changes together produced a real net gain (f1 +3.8%, matched_iou
+2.4%, false positives -8.7%) -- confirming the diagnose-before-design
choice was worth it: training-side fixes did move the needle without any
architecture change. **But they did not resolve either diagnosed failure
mode decisively**:

- **Background/foreground confusion improved but did not disappear.** FP
  count dropped 8.7%, consistent with focal loss's intended effect
  (down-weighting easy background reduces the "pull" toward predicting high
  probability on background texture). But the *proportion* that is pure
  clutter (zero GT overlap) is unchanged at exactly 70% -- the remaining
  false positives are the same *kind* of error, just fewer of them.
- **Dense-stand instance-separation did NOT clearly improve.** Missed-crown
  probability signal at the true centroid did rise (median 0.000 -> 0.016,
  fraction >0.1 rising 5% -> 13%), meaning the network is somewhat less
  "blind" on average -- but the aggregate miss *count* went up slightly, not
  down. The most likely explanation is that focal loss made the network's
  overall probability output sharper/more confident (fewer, more
  confident peaks pass the decode threshold, matching the tighter adopted
  min_peak_distance of 5 vs v3's 8 and the drop in total predictions), which
  trades a bit of recall for the precision gain -- a real mechanism, not
  free money, and not the same thing as actually separating two
  touching-and-similar-looking crowns.

## What this means for the next step

The gap to the G1B baseline (matched_iou 0.76-0.80 / recall ~90%) is still
large after both diagnosed-and-targeted training fixes. This is meaningful
new evidence: the two failure modes respond differently to "more data +
focal loss" --

- Background clutter is a *frequency* problem (focal loss's down-weighting
  of easy negatives has the right emitted mechanism for it) and responded
  as expected, even if not solved outright -- more training/hard-negative
  mining likely continues to help here.
- Dense-stand instance separation looks like it needs something focal loss
  does not provide: an explicit push to make probability *fall between* two
  touching, visually-similar instances, not just an overall gain in
  confidence. Focal loss reweights gradient by difficulty, but the
  *target* itself is unchanged -- it does not give the network any new
  information about where one crown ends and its neighbor begins beyond
  what the (already correct, per the oracle test) star-convex distance
  target already encodes. This points toward a genuinely different lever
  for that specific failure mode next: e.g. an explicit contour/edge term
  between adjacent instances, or contrastive supervision between
  neighboring instance centers -- not simply more of the same signal.
