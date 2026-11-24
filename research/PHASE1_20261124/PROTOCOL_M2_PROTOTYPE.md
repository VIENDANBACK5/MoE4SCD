# Phase 1 · Prototype 1: learnability of joint hypothesis–membership revision (M2)

Written 2026-11-24, BEFORE any training. Owner decision: Phase 1 GO for a learnability prototype; it is not a final
method and not evidence of novelty.

## Question
Given an EXISTING predicted instance P and frozen image features, can a small module learn to revise P into two
hypotheses together with their pixel memberships, and recover absorbed instances without breaking correct ones?

The module chooses one action:
- **KEEP:** P is left unchanged;
- **REFORM:** P is replaced by {P1, P2}.

The question is whether joint formation (M2) does this better than:
- (B1) a split decision followed by a geometric partition;
- (B2) a split decision with two independent mask heads;

when all three use the same features, data and training budget.

## Fixed setting (defaults chosen by the agent; reasons in brackets)
**Dataset:** QuebecTrees, the locked primary ITC dataset.
- Train: all 148 train tiles × 4 quadrants (1,666 px).
- Evaluation: the 139 validation crops. TEST is not touched.

**Base model:** Mask R-CNN seed 42, the validation-selected checkpoint (`clean_benchmark/results/mrcnn_s42_val_selection.json`).
- Frozen.
- Candidates = its predictions with score ≥ 0.45 (the IEO validation F1 cut).
- [Why: the NMS family has the largest absorption share on QuebecTrees (0.31–0.44 of misses).]

**Features:** that model's own FPN level '0' (stride 4, 256 channels), frozen.
- Every candidate gets a RoI: its tight box enlarged 1.25× about the centre, with a minimum side of 24 px.
- The RoI is RoIAligned to 32 × 32.
- The candidate mask is resampled to the same grid as one extra channel.

**Labels:** for candidate P, G(P) = {GT g : |g ∩ P| / |g| ≥ 0.5}.

| |G(P)| | Label | Use |
|---|---|---|
| 0 | FP | not used for training |
| 1 | KEEP | — |
| ≥ 2 | REFORM | targets = the two g with the largest \|g ∩ P\|; pixels of any other covered g are ignored in the mask loss |

**Training set:**
- every real REFORM candidate on train;
- KEEP candidates sampled 2:1 against REFORM (seed 0, fixed);
- no synthetic unions.
- Train counts measured on 35 tiles: 1,237 REFORM / 9,456 KEEP (9.6% reform), against 8.6% on validation.

## Arms (identical adapter, data, order, steps and seeds; trained in the same loop on identical batches)
**Shared adapter.**
- A: [F, M] (257 ch) → conv3 → GN → ReLU → conv3 → GN → ReLU → conv3 → X ∈ R^{128×32×32}.
- Pooled descriptor p = [masked mean of X, masked max of X].

**B1, split classifier + geometric partition.**
- Decision r = σ(MLP(p)).
- If REFORM: a 2-component GaussianMixture on the pixel coordinates of P at full resolution (sklearn, random_state 0).
- This is Split-and-Expand-like, without centres, and the partition is not learned.

**B2, split classifier + two independent mask heads.**
- r = σ(MLP(p)).
- Masks m1, m2 = σ(head(X)), with head = conv3 → ReLU → conv1 → 2 channels.
- The two masks are independent sigmoids: no competition and no iterative update.

**M2, joint hypothesis–membership revision.** Parent-conditioned two-hypothesis competition:
- Parent statistics:
  - μ = Σ_x M(x)X(x) / Σ M;
  - C = Σ_x M(x)(X(x) − μ)(X(x) − μ)ᵀ / Σ M;
  - u = 6 steps of power iteration on C (from a learned start vector), with λ = uᵀCu.
- Initial hypotheses:
  - z1 = μ + α√λ·u and z2 = μ − α√λ·u (α learned, initialised at 1);
  - z0 = mean of X outside P within the RoI (the rest of the scene).
- Iterate T = 3 times:
  - ℓ_j(x) = ⟨W_q LN(z_j), W_k X(x)⟩ / √128;
  - a_j(x) = softmax_{j ∈ {0,1,2}} ℓ_j(x). **Pixels are allocated exclusively and competitively.**
  - z_j ← GRU(z_j, Σ_x ā_j(x) W_v X(x)) + MLP, where ā_j is a_j normalised over x.
- Membership: P_k = {x : argmax_j a_j(x) = k}, for k = 1, 2.
- Existence, decided from the competition outcome:
  - r = σ(MLP([p, z1, z2, |z1 − z2|, m1, m2, min(m1, m2), H̄]));
  - m_k = Σ a_k M / Σ M is the mass each hypothesis wins inside the parent;
  - H̄ = the mean membership entropy inside P.
- What differs from B2: hypotheses and memberships are updated together, under exclusive competition, and the split
  decision reads the outcome of that competition.
- Relation to prior art: this is slot attention (Locatello 2020) applied locally, initialised from the parent's
  principal split. The initialisation is PDDP / LBG-style. Novelty is NOT claimed.

**Parameter counts:** B2 gets an extra conv3 (128 → 128) block so that its trainable count is within 25% of M2's. The
exact counts are written to the run's PARAMS.json.

**Losses**
- Decision: BCE with pos_weight = n_KEEP / n_REFORM.
- REFORM samples only:
  - B2: Hungarian (Dice cost) between {m1, m2} and the two targets; BCE + Dice on the matched pairs.
  - M2: the same, on (a1, a2); a0 is supervised towards 1 − (t1 ∪ t2), with ignored pixels excluded.
  - B1: decision loss only.
- Optimiser: AdamW (lr 1e-3, wd 1e-4), batch 128, 20 epochs over the fixed training set.
- Augmentation: the 8 dihedral transforms of the RoI tensors.
- Seeds 42, 43, 44.
- No validation-based checkpoint selection: the final weights are evaluated.

## Inference and evaluation (validation)
**Inference**
- Each arm is applied to every candidate.
- REFORM replaces P by the two children: 32×32 logits upsampled to the RoI and pasted, ∩ valid. Each child keeps the
  parent's score.
- A child below 100 px is dropped. If fewer than two children survive, P is kept unchanged.
- One pass only (no recursion); a depth-2 pass is reported as secondary.

**Metrics** (d = 2; world-coordinate de-duplication and border exclusion as in `PHASE0_20261124/code/d2d3_quebectrees.py`)
- recall@0.5 (one-to-one);
- absorption rate;
- **recovered**: GT absorbed under B0 that become hits;
- **lost**: GT hit under B0 that are no longer hits;
- **net = recovered − lost**, as % of unique validation crowns;
- false splits = REFORM applied to a candidate whose label is KEEP;
- FP;
- COCO mask AP over the 139 crops.

**Decision vs partition, separated** (the review's two probabilities)
- P(REFORM | label REFORM) and P(REFORM | label KEEP);
- P(recovered | REFORM decided, absorbed);
- an ORACLE-DECISION variant of every arm (REFORM iff the label is REFORM), isolating partition quality.

**Auxiliary (no decision attached):** U / C / O stratification of the B0 misses.

## Pre-declared reading (mean over the 3 seeds, final weights)
| Outcome | Condition |
|---|---|
| **M2_CONTINUE** | net(M2) − max(net(B1), net(B2)) ≥ +0.5 pp of unique validation crowns, AND the difference is > 0 in all 3 seeds, AND AP(M2) ≥ max(AP(B0), AP(B1), AP(B2)) − 0.2 |
| **LEARNABLE_NOT_BETTER** | net(M2) > 0 in all 3 seeds, but the M2_CONTINUE margin fails |
| **NOT_LEARNABLE_HERE** | net(M2) ≤ 0 in ≥ 2 seeds |

The oracle-decision variants are descriptive. They show whether any failure sits in the decision or in the partition.

## Harness
Exploratory pre-flight in the spirit of a06:
- a frozen `CONFIG.json`, whose hash is recorded;
- `preflight.py` must PASS before training. Checks:
  - the train/validation tile sets are disjoint;
  - no validation data is touched in training;
  - label counts;
  - the shapes of the operator unit tests;
  - permutation invariance of M2's two hypotheses;
  - a tiny-batch overfit for every arm;
  - parameter counts.

The new module is not registered in the a06 ENTRYPOINT_REGISTRY; that registry covers core training entrypoints.
