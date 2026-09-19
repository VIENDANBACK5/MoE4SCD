# Star-convex v2 — decode hyperparameter sweep

## Material Passport

- Origin: `star_convex_v2_training_result.md` flagged prob_threshold=0.5,
  min_peak_distance=5, nms_iou=0.3 as untuned defaults ("magic numbers" per
  CLAUDE.md Rule 5). This sweeps them properly on BAM_val before trusting
  any number derived from them.
- Method: cached raw (probability, rays) network output once
  (`code/cache raw outputs` step), then swept 4x3x3=36 decode configs
  decode-only (no repeated forward pass) — same "cache once, sweep cheaply"
  pattern as the Section-4 merge module.
- Script: `code/sweep_star_convex_decode.py`. Full grid:
  `experiments/star_convex_screen_v2/decode_sweep.csv`.

## Result

| Config (prob / peak-dist / nms) | matched_iou | precision | recall | f1 | split_rate | miss_rate |
|---|---:|---:|---:|---:|---:|---:|
| Untuned default (0.5 / 5 / 0.3) | 0.7235 | 0.419 | 0.566 | 0.480 | 0.0657 | 0.262 |
| **Best F1 (0.4 / 3 / 0.2)** | 0.7224 | 0.422 | 0.582 | **0.489** | **0.0524** | 0.241 |

Best-F1 config: F1 +1.9% relative, split_rate -22% relative, miss_rate -8%
relative, matched_iou essentially unchanged (-0.0011). Every one of the 36
configs stayed in a narrow band: matched_iou 0.720-0.726, f1 0.459-0.489,
recall 0.558-0.600, precision 0.386-0.429 (full grid in the CSV).

## Interpretation

Decode tuning gives a **real but small** improvement — it recovers some of
the split-rate advantage this representation already had, but **cannot
close the recall/precision gap to the G1B baseline** (baseline recall
~90% vs this model's 55-60% across the entire grid). This confirms the
gap identified in `star_convex_v2_training_result.md` is a training-scale
problem (300/1,439 images, no augmentation, no multi-seed confirmation),
not a decode-threshold problem — no point in this grid recovers baseline-
competitive recall or precision, so further threshold search is low-value
until more training data is used.

## Adopted default

`prob_threshold=0.4, min_peak_distance=3, nms_iou_threshold=0.2` (best F1)
is the new default for evaluating this checkpoint going forward, replacing
the placeholder 0.5/5/0.3 used in `star_convex_v2_training_result.md`.
This value is specific to `star_convex_screen_v2/star_convex_screen.pth` —
it must be re-swept for any future checkpoint (more training data/epochs
will shift the network's output distribution).

## What would actually move the needle (highest priority next step)

Scale up training data (currently 300/1,439 available BAM_train images)
and/or epochs (loss plateaued by epoch ~260, so more epochs on the *same*
300 images has limited further room — more *images* is the more promising
lever, not more epochs on this subset).
