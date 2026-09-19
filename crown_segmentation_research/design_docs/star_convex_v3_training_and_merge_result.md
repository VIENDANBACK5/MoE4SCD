# Star-convex v3 — scaled-up training + merge-module integration result

## Material Passport

- Origin: user's explicit plan — "chờ train v3 xong, rồi tích hợp module gộp
  vùng vào output của nó, nếu sau fail thì theo hướng tự tìm và thiết kế
  mới" (wait for v3, integrate the Section-4 merge module into its output,
  evaluate, and only pivot to designing a new method if that combination
  still fails).
- v3 training: 536 BAM_train images (300 seed=42 + 236 new seed=43,
  deduplicated), 150 epochs, batch_size=1, same warm-started backbone and
  architecture as v2. Loss 10.68 -> 1.32 (epoch 0 -> 149), plateaued from
  ~epoch 100 onward. Checkpoint:
  `experiments/star_convex_screen_v3/star_convex_screen.pth`.
- Decode threshold re-swept for this checkpoint (v2's adopted values are
  checkpoint-specific per `star_convex_v2_decode_sweep.md`'s own caveat):
  full grid at `experiments/star_convex_screen_v3/decode_sweep.csv`.
  New adopted default: `prob_threshold=0.5, min_peak_distance=8,
  nms_iou_threshold=0.2` (best F1; several nearby configs tie within noise).
- Merge module applied with the same delta already adopted for the Mask
  R-CNN result (`delta_appearance=0.15, delta_distance_m=2.0`) — this
  parameter is about appearance/distance similarity between mask fragments,
  not specific to which segmenter produced them, so it was reused rather
  than re-swept.
- Scripts: `code/cache_star_convex_raw_outputs.py` (new, forward-pass cache
  for any checkpoint), `code/evaluate_star_convex_with_merge.py` (new,
  decode + merge + evaluate, all via `benchmark.evaluator`).

## Result 1 — v3 vs v2 (more training data), BAM_val, 50 images

| | v2 (300 img/300 ep) | **v3 (536 img/150 ep)** |
|---|---:|---:|
| matched_iou | 0.7224 | 0.7155 |
| precision | 0.422 | 0.431 |
| recall | 0.582 | 0.602 |
| f1 | 0.489 | 0.502 |
| split_rate | 0.0524 | **0.0332** |
| miss_rate | 0.241 | 0.245 |

More training images produced a real, if modest, improvement (recall +2.0pp,
f1 +1.3pp, split_rate -37% relative) except matched_iou, which dropped
slightly (-0.007, within the noise band already observed across the whole
v2 decode-sweep grid — see `star_convex_v2_decode_sweep.md`). Consistent
with the earlier diagnosis that data scale is the limiting factor.

## Result 2 — merge module integration on v3's decoded output, BAM_val

| | matched_iou | precision | recall | f1 | split_rate | merge_rate | miss_rate | n_pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| star_convex v3 (no merge) | 0.7155 | 0.4312 | 0.6019 | 0.5025 | 0.0332 | 0.0370 | 0.2452 | 1890 |
| **star_convex v3 + merge** | 0.7160 | 0.4340 | 0.6027 | 0.5046 | 0.0347 | 0.0372 | 0.2430 | 1880 |

Every metric moves by <0.3pp in either direction; split_rate actually gets
very slightly *worse* (+4.5% relative). Visual check on `bam:val:197` (86 GT
crowns, the densest val image) confirms this numerically-null result is
real, not a measurement artifact: the merge module produced **56 instances
before and 56 after** — it found no fragment pairs to merge on this image at
all. Preview saved to `images/star_convex_v3_merge_val197_preview.png`
(GT | star-convex decode | + merge, side by side).

## Interpretation — merge module does NOT help here (in contrast to Mask R-CNN)

This is the expected outcome in hindsight, not a bug: the merge module
(Section 4 of `method_design_crown_confidence_v2.md`) targets a specific
failure mode — a single large/multi-lobed crown getting fragmented into
several adjacent Mask R-CNN proposals. Star-convex's own decode was already
oracle-validated to avoid exactly this failure mode (per-instance ray
encoding reconstructs true crown shape as one convex-from-center polygon,
`oracle_test_stardist_encoding.md`), which is why v3's split_rate (0.033) is
already **6x lower** than the G1B Mask R-CNN baseline (0.1676-0.1982) even
before any merging. There is very little of that specific failure mode left
for the merge module to fix in this representation's output — confirmed
directly by the near-zero effect size and the visual check.

## Where the real gap is (unchanged from v2, now better isolated)

| | matched_iou | recall | precision |
|---|---:|---:|---:|
| G1B Mask R-CNN baseline (BAM_test2) | 0.7585-0.7951 | ~90% | — |
| star_convex v3 + merge (BAM_val) | 0.7160 | 60.3% | 43.4% |

The gap to the baseline is not over-segmentation (already solved by this
representation) — it is **miss_rate (24.3%: real crowns that never produce
a decode peak at all) and precision (43.4%: more than half of predicted
instances are false positives)**. Neither is addressed by the merge module,
which only operates on fragments of already-detected regions. The v3
preview image shows this directly: several small/lower-contrast crowns
(e.g. the cluster in the upper-right of `bam:val:197`) have no predicted
polygon at all, while the polygons that are produced are generally accurate
single-piece shapes.

## Verdict on the user's stated fallback condition

Per the user's own plan ("nếu sau fail thì theo hướng tự tìm và thiết kế
mới"): merge-module integration is a **null result** — it does not close
any of the gap to the G1B baseline. This satisfies the "fails" condition as
stated. The honest characterization is not "star-convex the representation
failed" (the representation itself remains well-validated by the oracle
test and its split_rate advantage is real) but "post-hoc fragment-merging is
the wrong lever for this representation's remaining gap" — the gap is in
detection recall/precision, which is a training-time (loss/data/sampling)
or decode-time (peak-finding sensitivity) problem, not a post-hoc geometry
problem.

## What is NOT yet known (should be resolved before designing anything new)

Following this project's own established methodology (diagnose before
designing — see `g4d_rpn_roi_instrumentation.md` and the H/V oracle-test
KILL), the miss_rate/precision gap has **not yet been qualitatively
characterized**: are misses concentrated in small crowns, low-contrast
crowns, crowns near tile edges, or crowns whose true center produces a low
probability score for some other reason? Are false positives concentrated
at double-peaks on the same crown (a decode/NMS problem), background
clutter, or partial/edge crowns? This is the same category of open question
flagged for Mask R-CNN's classifier under-confidence in
`method_design_crown_confidence_v2.md` Section 6 and never resolved there
either — it should not be skipped a second time before committing to a new
architecture.
