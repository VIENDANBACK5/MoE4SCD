# M1: executed conventional scale-exposure control

**ORDINARY_SCALE_EXPOSURE_REDUCES_ENCODING_GAP_WITH_NATIVE_TRADEOFF_ORIGINAL_METHOD_NOT_ESTABLISHED**

The original-method objective remains active and unachieved. This completed study is a necessary established-method control, not a new architecture, loss or scale principle. Historical branches and their decisions are unchanged.

## What actually ran

Both arms reload the completed original mask-only Mask DINO seed42, train all original trainable components, preserve 300 queries and the segmentation objective. S0 uses native 1024-pixel inputs. S1 alternates native inputs with exact 2× pixel replication. Same 796 TRAIN windows, augmentation draws, source supervision and 398 optimizer updates per arm. Three entire diagnostic tiles were excluded from these additional updates, but the parent carrier already trained on them: this is not independent validation.

Scientific training completed, both model/optimizer states saved, 24 RGB-only inference outputs frozen before new metrics. Verification recomputed 7,200 raw and 3,463 final masks. No SAME6 tuning, TEST or protected review. A failed initial strict bf16 numerical check and its bounded engineering correction are retained.

## Main evidence

All counts below refer to the same 298 exposed TRAIN reference occurrences, including 133 source-small annotation occurrences. These are crop-level results, not SAME6, biological trees or independent sites.

| Measure | S0 native continuation | S1 ordinary scale exposure |
|---|---:|---:|
| Native A raw qualifying50 | 279 | 277 |
| Replicated B raw qualifying50 | 132 | 241 |
| A–B raw50 gap | 147 | 36 |
| Native A final TP50 | 276 | 276 |
| Native A small TP50 | 118 | 116 |
| Native A FP50 | 163 | 222 |
| Native A AP | 62.601 | 61.641 |
| Native A AP75 | 77.169 | 74.496 |
| Replicated B final TP50 | 114 | 236 |
| Replicated B TP50 at equal 443 outputs | 114 | 233 |

The registered narrow screen passes: the raw50 gap shrinks 75.51%, while native final TP50 does not decrease. That screen did not certify AP/FP quality. Native TP50 hides 2 new / 2 lost matches; both losses are small and both gains are outside that stratum. Native small Recall50 falls 1.504 pp, AP falls 0.960, AP75 falls 2.673, FP rises 36.20%. Native AP declines on each of the three diagnostic sources. At equal native counts, S1 loses one net TP.

For B, there are 123 new / 1 lost final matches; 107 new matches had no qualifying raw mask in S0B. The effect therefore includes actual raw formation, not just more accepted predictions. Final TP gains occur on all three source windows, but this is one crop per source and one paired seed.

The treatment does not establish general scale invariance: the untrained C/D pair's raw50 discrepancy increases from 35 to 73; its final TP50 discrepancy increases from 46 to 79. S1B still trails S1A by 36 raw qualifications and 40 final matches.

## What this means for a new method

OBSERVED: changing conventional scale exposure materially changes the replicated-input failure. Input replication deletes no original RGB information, yet the trained predictor remains dependent on raster representation.

DERIVED: the frozen pilot alone cannot establish that an original observability architecture is necessary. For this deliberately constructed replication, the known left inverse also returns exactly the original input; ordinary canonicalization is a relevant established alternative.

UNVERIFIED: a particular architectural constraint, the isolated role of exposure versus extra raster computation, the cause of BAM losses, a deployable observability estimator, and a distinct mathematical mechanism with net small-crown benefit.

S1 processes 2.5× the input raster pixels overall; optimizer updates and source exposure are matched, computation is not. Training times were 295.5 / 571.3 seconds; peak allocated GPU memory 4,173 / 12,833 MiB. The improvement belongs to this augmentation/computation treatment; it is not a proof about a pure scale-learning variable.

Close this fixed one-pass control. Do not release S1 as a better practical ITC system, extend the schedule, rescue thresholds or open an observability loss/V2 from these outputs. Keep RR_MASK_INIT and established carriers as the practical references. M1 remains a possible broad question, but this experiment has not supplied original-method entry conditions. Independent geographic efficacy and reference stability remain NOT_ESTABLISHED. This decision does not claim the whole RGB ITC problem exhausted.

[Full executed report](experiments/scale_exposure_control_v1/REPORT.md), [protocol](experiments/scale_exposure_control_v1/protocol.md), [reproduction](experiments/scale_exposure_control_v1/REPRODUCIBILITY.md), [actual metrics](experiments/scale_exposure_control_v1/results/METRICS.csv), [per-reference gains/losses](experiments/scale_exposure_control_v1/results/RECOVERY_AND_LOSS.csv).
