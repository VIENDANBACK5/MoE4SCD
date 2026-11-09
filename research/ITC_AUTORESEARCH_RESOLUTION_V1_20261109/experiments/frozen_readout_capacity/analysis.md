# Executed result: sampled readout fitting is not a capacity ceiling

**Decision: SAMPLED_READOUT_FIT_NOT_A_VALID_NEGATIVE_CAPACITY_TEST.** Close this particular probe; no loss sweep or full-carrier training follows from it.

Protocol/source/cohort were committed as `96fc102` before execution. Seed42 completed mask-only Mask DINO, three fixed eligible TRAIN windows from three already-seen sources, 298 cropped annotation occurrences. All carrier parameters remained frozen. Exactly three carrier forwards, 15.714 seconds reported pipeline time, 1654.46 MiB peak allocated GPU memory on a shared RTX 5880 Ada. This is a TRAIN oracle experiment, not SAME6 performance or independent generalization.

| Endpoint | Native final-query raw pool | Fitted per-reference oracle masks |
|---|---:|---:|
| Qualifying IoU50 references | 278/298 | 260/298 |
| Qualifying IoU75 references | 247/298 | 139/298 |
| Source-small IoU50 (source annotation <9 m²) | 116/133 | 102/133 |

Native availability uses all 300 queries. Oracle outputs use GT to fit and choose one vector per known target; these are different hypothesis-generation rules and **not** matched deployable systems. The native-best-to-its-fitted-target trajectory yields one new IoU50 qualification and 19 lost qualifications (15 small). Only one of 20 native-raw-missing targets receives a qualifying fitted mask. These are crop/reference occurrences, not independent biological trees. Separate RLE-pool reference availability and maximum one-to-one diagnostics happen to agree numerically in this cohort; both are saved and neither is deployment recall.

## Measured loss mechanism

Recomputed complete native-pixel intersections/unions from saved RLEs agree with reported IoUs within 1e-6. **All 19 lost qualifications have increased nonmember foreground pixels; none has increased false-negative target pixels.** Thus their IoU failures coincide with foreground expansion, despite improved target coverage and very small sampled balanced logistic losses. This localizes the observed geometry change; it does not identify a universal feature or training bottleneck.

The free readout is supervised on 4096 positive and 4096 negative sampled pixels. Small mean sampled error does not tightly control the number of foreground false pixels over approximately a million valid pixels, especially for very small targets. Choosing by sampled TRAIN loss does not make the resulting masks IoU-optimal. The native initialization itself already supplies 278 feasible qualifying readouts; the worse optimizer output cannot erase that constructive evidence.

## Mathematical counterexample, not novelty

Partition valid pixels into target P, sampled negative locations N_s and unobserved negative region B. Let F=(1,0) on P and B, and F=(-1,0) on N_s. For q=(a,0), a tending to positive infinity, sampled balanced logistic loss tends to zero. Thresholded foreground is P union B, so IoU=|P|/(|P|+|B|), below 0.5 whenever |B|>|P|. Thus arbitrarily low sampled loss alone does not guarantee qualifying mask geometry even within a linear-readout family. This elementary counterexample does not assert that the actual frozen features have that form or that every readout fails. It demonstrates why a negative sampled fit cannot establish a representation ceiling.

## What is and is not learned

- OBSERVED: one previously missing TRAIN reference has an explicit qualifying free readout in the frozen pixel-feature family.
- OBSERVED: the tested fit is not an improvement recipe; its foreground expansion damages many initially qualifying masks.
- DERIVED: native feature-family representability is already constructively supported for native successes; low sampled loss is an inadequate certificate for complete-mask quality.
- UNVERIFIED: why RGB-only decoder queries fail to retrieve the remaining targets; maximum attainable full-mask IoU for those targets; generalization; biological identity.
- NOT_RUN: a novel method, carrier fine-tuning, SAME6 evaluation, independent imagery evaluation and new human review.

Do not relabel these outputs as an RGB-only model or propose another sampled auxiliary loss from this negative result. The outer loop must preserve the established practical solution and seek evidence that can discriminate explanations; no original-method entry has been established.

## Reproduction

From the repository root, the fitting command was:

`cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_readout_capacity/code/run.py`

The code has a completion guard. For an independent refit copy this isolated experiment with its protocol/cohort to a new output directory; do not delete or overwrite the recorded results. Source/checkpoint/data hashes are in `results/BINDINGS.json`. Actual native RGB-only masks, separate GT-conditioned masks, embeddings, optimizer states, frozen fields, per-reference table, pixel errors and pool IoU matrices are retained. No AP scores were fabricated for oracle outputs.
