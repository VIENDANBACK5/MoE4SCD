# Action-consequence diagnostic (2026-11-25): answering the review, no retraining

> **RETRACTION (2026-11-25, self-audit). Do not cite Findings 1 and 2 below.**
> - **Finding 1 is tautological.** The "action oracle" sums only the positive ΔU, so it is ≥ any other selection by
>   construction, including the multi-object oracle. The 1.35× ratio is not evidence that the label is wrong.
> - **The additivity approximation failed its own check.** Marginal sum vs the true joint evaluation:
>   multi-object oracle 12.67 vs 3.38 pp (B1), about 11.5 vs 3.3 (B2), about 10.4 vs 2.8 (M2); learned B2 s42 +1.47 vs
>   −0.08 and B2 s44 +1.25 vs −0.08. Signs flip, so the statement "tracks the headline sign" below is false.
>   The M2/B2 order also flips (marginal B2 > M2; true M2 > B2). Overlapping tiles double-count crowns, and several
>   candidates can be credited with the same crown.
> - **Finding 2 (GMM has the highest ceiling)** rests on the same marginal sums. Only the true joint oracle in EVAL.json
>   (B1 3.38, B2 3.30, M2 2.84 pp) may be used.
> - **Finding 3 (AUC)** uses the marginal ΔU as target, so it is weak. The question in review §5 (B2 at M2's false-split
>   rate) is NOT answered: the threshold sweep used a wrong denominator.
> - The difference between the action label and the multi-object label remains an untested idea.


Code: `code/analyze_action.py`. Raw: `results/ACTION.json`. Uses the 9 trained models (3 arms × 3 seeds) unchanged.

**What it computes.** For every validation candidate P_j, reform ONLY P_j (replace it by that arm's two children, all
other B0 predictions unchanged) and measure the exact marginal change on that crop's border-excluded crowns:
ΔU_j = (crowns absorbed in B0 that become hit) − (crowns hit in B0 that become not hit). This is the action-consequence
ΔU the review asked for. Units below are **raw per-crop crowns** (not world-de-duplicated), so they are a relative
operator diagnostic and are NOT on the same scale as the headline EVAL net; they are used only to compare arms,
objectives and operating points.

**Marginal-additivity caveat.** Summing ΔU_j over reformed candidates assumes the actions are independent. Reforms are
sparse and spatially local, so this is a good approximation; the learned-operating-point sum tracks the headline sign.

## Finding 1 (review §3/§6, confirmed and stable): the label is wrong; action-benefit beats multi-object coverage
For every arm and every seed, the action-benefit oracle (reform iff ΔU_j > 0) recovers far more than the multi-object
oracle (reform iff the mask covers ≥ 2 GT, which is what the main eval used):

| Operator | action-benefit oracle (raw crowns) | multi-object oracle | ratio |
|---|---|---|---|
| B1 GMM | 626 / 626 / 626 | 465 | **1.35×** |
| B2 two heads | 572 / 543 / 570 | 424 / 435 / 412 | **1.33×** |
| M2 joint | 513 / 517 / 543 | 389 / 362 / 396 | **1.37×** |

- Splitting a candidate that covers ≥ 2 GT does not always help, and some candidates that cover only one GT at ≥ 50%
  still help when split (they cover a second crown at 40–50%, or the split removes a false positive).
- So `P(mask covers ≥ 2 GT | X)` — the quantity the three arms are trained on — is the wrong target. The right target
  is `P(splitting improves the instance set | X)`, i.e. the sign of ΔU. The ceiling is ~35% higher, consistently.

## Finding 2 (confirms earlier): M2's joint partition is not a better partitioner
Under the action-benefit oracle (a perfect per-candidate decision), the partition-quality ceiling is:
- **B1 GMM 626 ≥ B2 562 ≥ M2 524** raw crowns (means over seeds).
- The plain GMM has the HIGHEST ceiling; M2's joint hypothesis–membership operator is the lowest. The sophisticated
  operator is not where value lies. Do not pursue the M2 operator as the contribution.

## Finding 3 (revises my earlier wording): M2's decision is comparable to B2's, not clearly better
Decision discrimination measured threshold-free by AUC of the decision logit predicting ΔU_j > 0:

| Arm | AUC / seed | mean |
|---|---|---|
| B1 | 0.771 / 0.787 / 0.770 | 0.776 |
| B2 | 0.831 / 0.826 / 0.808 | 0.822 |
| M2 | 0.844 / 0.819 / 0.844 | **0.836** |

- M2 and B2 are comparable (M2 wins 2 of 3 seeds by ~0.01–0.03; B2 wins one). Both clearly beat B1.
- This **revises** the earlier "M2's edge is conservatism": at matched discrimination, M2's thin headline advantage over
  B2 (+0.21 pp) is an operating-point effect, not clearly better decision quality. The review's §5 concern is correct:
  a conservatively-thresholded B2 would plausibly match M2. AUC, being threshold-free, is the fair comparison, and it
  does not separate M2 from B2.
- All three decision heads are trained on the multi-object label, so even their AUC is against a mislabelled target
  (Finding 1); an action-benefit-trained head is untested.

(The `sweep`/`false_split_rate` field in ACTION.json uses a non-standard denominator — reformed-that-are-KEEP over all
reformed, not over all KEEP — so its threshold-matched rows are not interpretable. AUC supersedes it and is what is
reported here. The sweep field is left in the JSON but should not be used.)

## Net reading
- **Keep:** the headline verdict LEARNABLE_NOT_BETTER, and "the partition operator is not the contribution" (now
  doubly supported: GMM has the highest action-benefit ceiling).
- **Correct:** "M2's edge is conservatism" → M2 and B2 have comparable decision quality; M2's small headline edge is an
  operating point.
- **Soften:** "once you know a mask holds two crowns, splitting is easy" → even with a perfect decision, one-pass
  binary GMM leaves most absorbed crowns unrecovered; partition and hypothesis expressivity are a real further ceiling.
- **New, review-driven and stable:** the training label (multi-object coverage) is suboptimal; an action-benefit
  objective has a ~35% higher ceiling across all operators and seeds. This is a change of LEARNING OBJECTIVE, testable
  with the partition held fixed — not a new architecture and not a claim that "the objective is the bottleneck".

## Proposed Prototype 2 (controlled objective test; NOT yet run, awaiting approval)
Hold the partition fixed (GMM) and the candidate set fixed. Compare three decision heads under identical budget:
- D0: trivial geometry rule (split iff area/elongation above a threshold);
- D1: the current multi-object classifier (this prototype's head);
- D2: an action-benefit head trained to predict sign/magnitude of ΔU_j (computable from GT at train time).

Pre-declared question: does D2 realise more net recovery than D1 and D0 at a matched false-split rate (now defined
correctly as reformed-KEEP / all-KEEP)? This isolates the objective change. It does not, by itself, establish novelty:
cost-sensitive / action-value learning and learned error correction (Zung 2017) are prior art, so a positive D2 would
still need to be distinguished from a learned merge-error detector.
