# D1.1: Mechanism-level gap analysis (2026-11-24)

**Scope.** Eight works closest to the failure we measured, chosen along four axes:
- query formation;
- dynamic cardinality;
- instance ownership;
- correction beyond fixed candidates.

Every work was read in full. Sources:
- `LITERATURE_FULLTEXT_20261119/READING_NOTES_INSTANCE_FORMATION.md` (2026-11-20: Mask2Former, Mask DINO, AdaSlot,
  GASP, Lifted Multicut);
- `READING_NOTES_MASKDINO.md`;
- `READING_NOTES_INSTANCE_DECISION_LEARNING.md` (2026-11-24: CellSAM, MDQE, Zung 2017, Split and Expand).

This is not a new survey.

**Labels:**
- **[A]** stated or measured by the authors;
- **[I]** our inference from their equations or design. It is NOT an author claim, and "they do not report absorption"
  is never taken as evidence that they fail on it;
- **[O]** measured by us on our data.

**The failure being matched [O]:**
- On SelvaMask, in-domain Mask2Former, Mask R-CNN and TreeFlowNet all absorb 20–23% of reference crowns. Recall is comparable for
  Mask2Former and Mask R-CNN (0.62–0.66) but NOT for TreeFlowNet (0.507), so TreeFlowNet is not strength-matched.
- For 89–91% of Mask2Former's absorbed crowns, no decoded query has mask IoU ≥ 0.5.
- On LIVECell, zero-shot CellSAM with oracle GT boxes as prompts gets 90% of cells at IoU ≥ 0.5, and absorption drops
  0.124 → 0.047.
- The failing stage is therefore that **no adequate candidate exists for one of two touching objects**. When a correct
  candidate is supplied, a mask decoder mostly recovers the object.
- Caveat: "No candidate at IoU 0.5" does not prove that the representation lacks the object.

---

## 1. Mask2Former (Cheng et al., CVPR 2022): query formation
1. **Problem.** A universal mask-classification architecture; slow convergence of MaskFormer-style decoders.
2. **Prior assumption.**
   - N = 100–200 learnable queries [A].
   - The initial queries act as image-independent "region proposals" (Fig. 3: AR@100 of the layer-0 query masks) [A].
   - The decoder refines each query inside its previous mask (masked attention) [A].
   - The count = queries whose class confidence exceeds a threshold.
3. **Counterexample.**
   - Two touching objects whose evidence falls inside one query's previous mask. Masked attention keeps that query
     within the joint region, and no step obliges another query to claim the second object at inference [I].
   - The authors state that small objects remain weak (APS lags) [A].
4. **New principle.** Changes the decoder's attention, not the formation rule: masked cross-attention restricted to the
   foreground of the previous prediction, plus multi-scale features [A].
5. **Mechanism.**
   - X_l = softmax(M_{l−1} + Q_l K_lᵀ) V_l + X_{l−1}, with M_{l−1} = 0 inside the previous mask and −∞ outside.
   - Training uses Hungarian matching and BCE/Dice on point samples [A].
6. **Evidence.**
   - The masked-attention ablation and convergence curves isolate the decoder change [A].
   - Nothing isolates how hypotheses are formed for dense touching objects.
7. **Remaining gap.**
   - [O] In-domain on SelvaMask, absorption is 0.88–0.98 × Mask R-CNN's. The earlier carrier study
     (ITC_CARRIER_CAUSAL_MECHANISMS) showed that about 271 of 577 final matches first qualify after initialisation (mean over 3 seeds), so
     queries do relocate during decoding, but not for the absorbed crowns.
   - [I] Hungarian training gives every GT a query, yet at inference nothing enforces that each object found in the
     evidence gets one.

## 2. Mask DINO (Li et al., CVPR 2023): data-dependent query formation
1. **Problem.** Unify detection and segmentation in DINO; weak initial queries.
2. **Prior assumption.** Dense encoder tokens are scored, and the top-K tokens initialise content and anchor queries
   ("unified query selection") [A]. The set of hypotheses is the top-K by score, so it is formed from evidence but in a
   single shot.
3. **Counterexample.** [I] When two touching objects produce one high-scoring token region, or the second object's
   token scores lower than tokens elsewhere, top-K never selects it. No later step revisits that decision.
4. **New principle.** Hypotheses are FORMED from dense, image-dependent evidence rather than learned constants. Masks
   also initialise the anchor boxes.
5. **Mechanism.**
   - m = q_c ⊗ M(T(C_b) + F(C_e)).
   - Top-K token selection from the encoder classification scores.
   - Unified denoising; hybrid matching cost [A].
6. **Evidence.**
   - Query selection alone gives 39.6 mask AP at decoder layer 0, against 1.1 for Mask2Former (Tab. 8) [A]. Formation
     quality is measured directly.
   - Removing all components costs −2.7 AP (Tab. 15) [A].
   - APS 26.0 vs 23.4 (Tab. 3) [A].
7. **Remaining gap.**
   - [I] Formation is one-shot top-K over tokens. Cardinality is fixed by K and a score threshold.
   - Not measured by us; our in-domain query baseline is Mask2Former.
   - **Closest prior art for "form hypotheses from evidence".**

## 3. CellSAM (Marks, Israel et al., Nature Methods 2025): detector-formed prompts, SAM decoding
1. **Problem.** A generalist cell segmenter without manual prompts.
2. **Prior assumption.** CellFinder (Anchor DETR, 3,500 queries, NMS-free set prediction) supplies a box for every cell;
   SAM turns boxes into masks [A].
3. **Counterexample.** A missing, mislocalised or low-scored box [I]. The authors attribute weak LIVECell cell lines to a
   possible "limitation of bounding boxes as a prompting strategy" [A].
4. **New principle.** Separates WHERE (box set prediction) from WHAT (promptable mask decoding) [A].
5. **Mechanism.**
   - Anchor DETR with Hungarian set loss.
   - Box threshold 2/3·0.4 + 1/3·T_μ (k-means).
   - SAM neck fine-tuned on GT boxes [A].
6. **Evidence.**
   - GT-box prompting is an upper bound far above point or grid prompting (Ext. Data Fig. 1) [A].
   - The generalist matches specialists [A].
7. **Remaining gap.** [O] On LIVECell (zero-shot):
   - absorbed cells: 46% no box at IoU 0.5, 46% box below the threshold, 8% assignment errors;
   - lowering the threshold rescues only 20% of absorbed cells and doubles absorption;
   - GT boxes cut absorption to 0.047.
   - [I] The decoder is not the bottleneck; forming and selecting box hypotheses is.

## 4. MDQE (Li et al., CVPR 2023): instance ownership via repulsion
1. **Problem.** Query-based VIS confuses occluded, similar-looking objects in crowded scenes.
2. **Prior assumption.** Each GT has a matched query; the remaining errors are pixels shared with neighbours [A].
3. **Counterexample.** [I] An object that has no query whose mask is near it. The repulsion is defined for matched pairs
   only (K matched GT).
4. **New principle.** Ownership by contrast: each matched mask is pushed off the union of nearby non-target GT masks.
   Queries are also initialised from grid peaks of a semantic map [A].
5. **Mechanism.**
   - o_i = {j : box-IoU(B_i, B_j) > 0.1}.
   - Dice-inter = 1 − (2|M̂⊙M| + |(1−M̂)⊙M_oi|) / (|M̂| + |M| + |M_oi|).
   - BCE weight 2 on target and neighbour pixels [A].
6. **Evidence.**
   - Table 1c: 29.0 → 31.2 mAP on OVIS, with APho +1.7 [A].
   - Grid-guided initialisation: 15.4 → 19.8 → 24.2 (Tab. 1a) [A], a larger effect than the repulsion.
7. **Remaining gap.**
   - [I] Repulsion sharpens ownership between EXISTING hypotheses; it cannot form a missing one.
   - The larger gain came from **evidence-based query initialisation (formation)**, consistent with our [O] finding that
     formation, not ownership, dominates.

## 5. Adaptive Slot Attention (Fan et al., CVPR 2024): dynamic cardinality
1. **Problem.** Slot methods need a predefined number of slots, while scenes vary in object count [A].
2. **Prior assumption.** Start from K_max slots; the right count is reached by DROPPING slots [A].
3. **Counterexample.** [I] The regulariser is λ per kept slot against a reconstruction gain that scales with the object's
   area, so small or dense objects are dropped first. The authors state: "performance on small, dense objects is
   limited" [A].
4. **New principle.** Cardinality as a learned, per-image subset selection.
5. **Mechanism.**
   - min E_Z[L_recon] + λ·L_reg(π), with π = softmax(h_θ(S)).
   - Mean-field Gumbel keep/drop Z ∈ {0, 1}^K.
   - A masked slot decoder [A].
6. **Evidence.**
   - MOVi-C/E: matches the best fixed K without tuning it.
   - COCO: about 20 ARI points above the 33-slot model, but "the best results obtained with fixed slot numbers are
     marginally superior to our results" [A].
7. **Remaining gap.**
   - [I] Dynamic cardinality exists only as pruning from a fixed pool.
   - Unsupervised; no evidence on dense touching instances.

## 6. Zung, Tartavull, Lee, Seung (NeurIPS 2017): correction beyond the first segmentation
1. **Problem.** Remaining split and merge errors in neuron segmentation [A].
2. **Prior assumption.**
   - Errors are sparse; a detector can localise them.
   - The candidate mask is a SUPERSET of the true object.
   - The segmentation is assembled from supervoxels [A].
3. **Counterexample.** The authors state that errors "already present at the supervoxel level" cannot be fixed [A].
   [I] The method also needs a location (the central voxel) to say WHICH object to extract.
4. **New principle.**
   - Detection and correction are separated.
   - The corrector re-forms the object at a given location from a superset mask: "object mask pruning", the opposite of
     agglomeration.
   - Detector output serves as "advice" [A].
5. **Mechanism.**
   - Error map Err(Obj).
   - The corrector outputs a 6-D embedding v, with M = exp(−‖v − v(centre)‖²).
   - The graph is updated by cliques and cuts when M(S) ∉ [0.1, 0.9] [A].
6. **Evidence.**
   - Detector precision and recall > 90%, even from shape alone [A].
   - With advice: 707 of 944 errors fixed, 68 introduced; without advice: 547 fixed, 77 introduced (Tab. 2) [A].
7. **Remaining gap.**
   - **This IS prior art for "form a new instance inside an already-merged mask"**, given a location and supervoxel atoms.
   - [I] It is not learned jointly with the first-pass segmenter. It is 3D EM and sparse-error. We have not measured
     whether such a corrector is limited by its atoms on touching crowns.

## 7. Split and Expand (Foo et al., 2022): creation from centre evidence
1. **Problem.** Clumped cells merged into one instance, and missing small cells, under point supervision [A].
2. **Prior assumption.** A separately supervised centre head (trained on reliable point labels) detects cells more
   reliably than the segmentation map. Each centre implies one cell [A].
3. **Counterexample.** [I] Missing or merged centres. Membership that a weighted GMM cannot express (non-ellipsoidal
   cells, adjacent cells of very different size).
4. **New principle.** The count and location come from an independent existence signal. A geometric partition then
   SPLITS (k = number of centres) and LRP EXPANDS centres that have no segment into new instances [A].
5. **Mechanism.**
   - A k-component bivariate GMM, weighted by the centre map, inside a clump.
   - LRP heatmap thresholded at 0.1.
   - Overlap ≤ 0.5 against existing cells [A].
6. **Evidence.**
   - Split: AJI +0.012 to +0.052 (paired t-test significant) [A].
   - Expand: small-cell AJI +0.012 [A].
7. **Remaining gap.**
   - **This is prior art for creating instances that the segmentation missed**, provided a centre detector supplies them.
   - [O] EXP-029: injecting oracle points into the TFN flow decoder recovered 0/210, but that test did not partition
     geometrically.
   - The Split-style test with oracle centres on our absorbed masks is NOT yet run.

## 8. GASP / Lifted Multicut (Bailoni et al.; Keuper et al., ICCV 2015, arXiv 1505.06973): ownership by partition, free cardinality
1. **Problem.** Graph partition into an unknown number of clusters from signed (attractive/repulsive) evidence [A].
2. **Prior assumption.** Nodes (pixels or superpixels) are fixed atoms. Cardinality emerges from the optimum, and pairwise
   costs carry all the evidence [A].
3. **Counterexample.**
   - Small clusters have few inter-cluster edges, so noise is not averaged and they merge wrongly [A, GASP SSBM analysis].
   - Single linkage returns large under-segmented clusters [A].
4. **New principle.** Count-free partition. Lifted edges add long-range evidence without changing the feasible set [A].
5. **Mechanism.** The linkage criterion decides merging until attraction ≤ repulsion. Lifted multicut minimises a linear
   cut cost over local and lifted edges (NP-hard; KLj heuristic) [A].
6. **Evidence.** Average linkage is most robust under spatially correlated CNN noise [A].
7. **Remaining gap.**
   - [I] Cardinality is free, but it hinges on pairwise evidence across the shared boundary.
   - [O] In our crowns, mostly no such evidence exists: on QuebecTrees, 98% of absorbed crowns had their own basin, but
     only 18% had a direct pass to the absorber at or above τ (EXP-034).

---

## Cross-cutting answer to the review's question
**"Which existing mechanism can form an additional instance when no predicted mask is good enough for it?"**

| Mechanism that can create an instance | Needs | Work |
|---|---|---|
| Hungarian assignment of a query to each GT | Training only; nothing at inference | Mask2Former, CellFinder, Mask DINO [A]; [O] still absorbs in-domain |
| One-shot formation from dense evidence (top-K tokens) | The object's token must rank in the top-K | Mask DINO [A] |
| Evidence-based query initialisation | Peaks of a semantic map | MDQE [A] |
| Re-forming the object inside a superset mask | A location, plus supervoxel atoms | Zung 2017 [A] |
| Creating from an independent existence signal | A reliable centre/point detector | Split and Expand [A] |
| Free cardinality via partition | Pairwise boundary evidence; fixed atoms | GASP / Lifted multicut [A] |
| Pruning from a pool | A pool that already contains the object | AdaSlot [A] |

**Not read, but must be cited:** marked point processes for tree crowns (birth–death moves); the "Overlap Loss" paper
(ICIP 2023); Cellpose-SAM.

## Draft research-gap statement (D4 revision; PROPOSAL, pending owner review)
- **[O]** Across NMS detection, flow grouping and set prediction trained in-domain on ITC:
  - touching-instance absorption is 20–23% of reference crowns;
  - in the query model it arises mainly because no decoded candidate fits one of the two objects;
  - a promptable mask decoder recovers most objects when given correct candidates (zero-shot cell evidence).
- **[A]** Existing remedies either:
  - act after candidates exist (repulsion, NMS-free selection, slot pruning); or
  - form candidates once from dense scores (top-K query selection); or
  - create candidates only with an external existence signal (centres) or within fixed atoms (supervoxels, graph
    nodes).
- **[I] Open:** a learned formation step that, after a first set of hypotheses has claimed the evidence, decides from
  the image that a claimed region contains a further object and instantiates it, with no external centre detector and
  no fixed atoms.
- **Novelty risks, to resolve before committing:**
  - iterative and residual detection schemes (unread here);
  - marked point processes;
  - Mask DINO-style re-selection applied in several rounds.
