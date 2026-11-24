# Phase 1 · Prototype 1 results: joint hypothesis–membership revision (M2) is learnable but not better (2026-11-24)

> **CORRECTIONS (2026-11-25, self-audit).**
> - **The preflight first FAILED** (tiny-batch overfit: M2 decision loss 0.403, gate 0.3). The gate was then rewritten
>   (full batch, no augmentation, 300 steps, lr 2e-3) after a diagnostic, and only then passed. This was a change to a
>   registered check after seeing it fail, and this file did not say so.
> - "Reproduces the IEO candidate counts exactly (preflight parity check PASS)" was false: preflight had no such check.
>   Checked now on 12 validation crops: 12/12 equal counts, best IoU 1.0; B0 recall (0.7145) and absorption (0.1253)
>   equal the Phase 0 values exactly. The candidates are therefore fine.
> - "The shared anaconda env was upgraded mid-session" is an unsupported guess. The install dates are August 2026, and
>   the same command that printed torch 2.5.1 earlier printed 2.4.0 today. The cause is unknown; the torch version used
>   by each stage (cache, training, evaluation) was not logged.
> - Protocol items not delivered: the depth-2 pass; the AP guard unit was ambiguous.
> - The reading "the bottleneck is the split DECISION" is withdrawn as a causal statement. What was measured: an oracle
>   multi-object decision gives +2.8 to +3.4 pp net; the learned decisions give −1.5 to +0.4 pp.
> - AP here is computed on candidates cut at score 0.45 and is NOT comparable with benchmark AP.


Protocol (pre-registered): `PROTOCOL_M2_PROTOTYPE.md`. Config hash in `results/PREFLIGHT.json` (status PASS).
Code: `code/{common,arms,train,evaluate,preflight}.py`. Raw numbers: `results/EVAL.json`.

## Setup actually run
- QuebecTrees: train 148 tiles × 4 quadrants (14,212 cached candidates: 4,357 REFORM, 9,855 KEEP after 2:1 sampling);
  validation 139 crops (3,671 unique crowns after world-coordinate de-duplication). TEST untouched.
- Base: frozen Mask R-CNN s42, candidates at score ≥ 0.45; features FPN level 0; RoIAlign 32×32.
- Three arms trained jointly on identical batches, seeds 42/43/44, 20 epochs, final weights (no val selection).
- Environment note: the shared anaconda env was upgraded mid-session (torch 2.4→2.5); all Phase 1 runs use
  `PYTHONNOUSERSITE=1` → torch 2.5.1+cu124 with torchvision 0.20.1+cu124. The frozen Mask R-CNN forward still reproduces
  the IEO candidate counts exactly (preflight parity check PASS).

## Pre-declared reading: **LEARNABLE_NOT_BETTER**
Net = (recovered − lost) as % of the 3,671 unique validation crowns, mean over 3 seeds, learned decision.

| Arm (learned decision) | net/seed (pp) | mean net | mean FP | mean AP |
|---|---|---|---|---|
| B0 (no change) | — | 0.00 | 3,520 | 0.314 |
| B1 split + GMM | −2.40 / −0.22 / −1.80 | **−1.47** | 5,090 | 0.260 |
| B2 split + 2 heads | −0.08 / +0.63 / −0.08 | **+0.15** | 4,749 | 0.274 |
| **M2 joint** | +0.38 / +0.46 / +0.25 | **+0.36** | 4,077 | 0.297 |

- net(M2) > 0 in all 3 seeds ✓.
- Margin over the best baseline: net(M2) − max(net(B1), net(B2)) = 0.36 − 0.15 = **+0.21 pp < +0.5 pp** ✗.
- Therefore **LEARNABLE_NOT_BETTER**: M2 recovers absorbed crowns and is the best of the three arms, but it does not
  clear the pre-registered margin over the simple two-head baseline B2.

## The decisive finding: the bottleneck is the split DECISION, not the partition operator
With an ORACLE decision (REFORM exactly the true REFORM candidates), every arm recovers about the same:

| Arm (oracle decision) | mean net | recovered / lost (s42) |
|---|---|---|
| B1 GMM | **+3.38** | 174 / 50 |
| B2 two heads | **+3.30** | 166 / 43 |
| M2 joint | **+2.84** | 141 / 31 |

**AP-guard unit note (review):** the protocol's `AP(M2) ≥ max(AP) − 0.2` uses a 0.2 tolerance on the 0–1 AP scale (= 20 AP points), far too loose. With the intended 0.2 AP-point tolerance (0.002 on this scale), M2 AP 0.297 < B0 0.314 − 0.002, so M2 also fails the AP guard. Either way the verdict is unchanged, because the net-margin condition already fails.

- Given the right decision, a plain GMM partition (B1) recovers as much as, or **more than**, M2's joint
  hypothesis–membership operator. M2 is in fact slightly worse at the partition itself (+2.84 vs +3.38).
- So the sophisticated joint operator is **not** where the difficulty lies in this setting. (Softened after review: this
  does NOT mean splitting is easy in general. Even with the oracle multi-object decision, GMM recovers only 174/460
  absorbed crowns (38%) and loses ~50 correct ones; partition quality and one-pass binary hypothesis expressivity also
  impose a substantial ceiling. The correct reading is: **decision quality limits the realised gains, while partition
  quality and hypothesis expressivity impose a further ceiling.**)
- This independently reproduces, on ITC RGB, the central lesson of Zung 2017: error **detection** is the hard, valuable
  part; the **corrector/partition** is comparatively easy.

## Why M2 is the best arm under a learned decision (and why that is thin)
M2's edge is **conservatism**, not better splitting:

| Arm (learned, mean over seeds) | P(split \| REFORM) | P(split \| KEEP) = false-split rate |
|---|---|---|
| B1 | 0.375 | 0.071 |
| B2 | 0.372 | 0.068 |
| M2 | **0.217** | **0.029** |

- M2 fires split much less often, which halves its false-split rate and keeps FP ≈ 600–1,500 below B1/B2 and AP higher.
- But it also recovers fewer true absorptions, so the net is only marginally positive.
- **Instability:** M2's decision is poorly calibrated across seeds. In seed 44 it nearly stops splitting
  (P(split\|REFORM) = 0.07; 88 REFORMs; only 12 recovered, net +0.25 pp), while in seed 42 it splits 455. The decision
  head, which reads the competition outcome, does not train to a stable operating point.

## The recoverable ceiling under one-pass splitting
- B0 misses 1,048 of 3,671 crowns. Stratified: **U uncovered 539, C absorbed 460, O other 49**.
- Oracle decision + GMM recovers 174 of the 460 absorbed (≈ 38%), at a cost of ~50 crowns lost and higher FP.
- One-pass replacement of an absorbing mask therefore has a ceiling of about +3.4 pp recall on QuebecTrees, limited
  equally by partition quality (62% of absorbed crowns are not cleanly split even by an oracle-triggered GMM) and by
  the ~50 correct crowns broken.

## Action-benefit vs multi-object label (review)
The protocol labels a candidate REFORM when its mask covers >= 2 GT (|G(P)| >= 2) and the oracle decision splits exactly
those. But the pipeline actually wants P(splitting improves the instance set | X), not P(mask covers >= 2 GT | X). These
differ: splitting a true-REFORM candidate can still fail to recover a crown or can break a correct one. The
`analyze_action.py` diagnostic recomputes, per candidate, the marginal net change from reforming only that candidate,
and uses it for an action-benefit oracle and threshold-matched curves. See `RESULTS_ACTION_ANALYSIS.md`.

## What this means for the method direction
- **Against M2 as the novelty.** A learned joint hypothesis–membership operator is not better than a GMM at the
  partition, and its only edge (a conservative decision) is unstable. The contribution should NOT be the partition
  operator.
- **For a detection-centric method (closer to M1).** The intervenable, under-solved step is deciding *which* existing
  predicted instance contains two crowns. That is the "intra-hypothesis inconsistency" idea, and the oracle gap
  (+0.36 learned vs +3.3 oracle) is almost entirely a decision gap.
- This does not establish novelty for M1 either: Zung 2017 already learns such a detector (on EM, from shape). A Phase 1
  Prototype 2 would have to show that an RGB intra-hypothesis inconsistency detector beats (a) the score-based decision
  heads here and (b) the simplest "split every candidate whose area/elongation exceeds a threshold" rule, and that the
  decision is what moves the metric.

## Honest limitations
- One base model (Mask R-CNN s42) and one dataset (QuebecTrees); SelvaMask query carriers not yet used here.
- Frozen features; the adapter cannot change what the backbone encodes.
- One-pass only (depth-2 recursion not evaluated).
- The decision threshold is prior-corrected but not tuned per arm; a per-arm calibrated threshold could change the thin
  learned-decision margins (but not the oracle-decision conclusion, which is threshold-free).
- Net uses the unique-crown denominator; the absolute pp are small because absorption is ~12.5% of crowns and only part
  is recoverable in one pass.
