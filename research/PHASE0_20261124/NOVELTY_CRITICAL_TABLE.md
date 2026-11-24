# Novelty-critical reading: instance-birth mechanisms (2026-11-24)

Track A status: PROBLEM LOCKED · MECHANISM OPEN · NOVELTY NOT ESTABLISHED. **Phase 1 (mechanism invention): GO for a learnability prototype** (owner, 2026-11-24). See `../PHASE1_20261124/`.

**Corrections after owner review (2026-11-24):**
- Item 2 below over-claimed. The narrower conclusion is: hypothesis generation is an important, intervenable bottleneck. It is NOT shown to be the only one.
  - Oracle location plus direct supervision changes several factors at once.
  - A GT box supplies location, scale and extent, not just existence.
  - A decoder can still merge given a correct prompt.
- The U/C experiment is demoted to an auxiliary diagnostic. It is not a GO/NO-GO gate. If it is run, it will:
  - use three strata (U uncovered, C absorbed per D3, O other misses);
  - report P(trigger hit | stratum) separately from P(recovered | trigger hit, stratum);
  - drop the 2× rule. A coverage difference can arise from size, visibility or GT quality, so it cannot prove structural blindness.
- ECRQ's R(x) ≈ 0 on covered crowns follows from its definition. What remains to measure is only how often that case occurs.

**Read in full** (Read tool, every page; PDFs in `LITERATURE_FULLTEXT_20261119/novelty_critical_dl/`):
- Ren & Zemel, CVPR 2017 (arXiv 1605.09410, 12 pp);
- IterDet (Rukhovich et al., arXiv 2005.05708v2, 13 pp);
- Progressive End-to-End Detection (Zheng et al., CVPR 2022, 10 pp);
- Aval et al. 2018, a tree MPP with multiple birth and death (ISPRS J. 146, 16 pp; HAL author version).

**NOT READ:**
- Recurrent DETR (IEEE Access 2023): the IEEE download returns an empty file. Only the abstract is known (a reused
  decoder and a "Pondering Hungarian Loss" give a variable-size prediction set; +0.8 AP COCO, +0.4 AP CrowdHuman).
- The RSE 2018 MPP tree-crown paper named in the review: not located as open access.

**In-repo branches re-read** (decision documents only):
- ITC_QUERY_BIRTH_LEARNABILITY_V1;
- ITC_ECRQ_V1;
- ITC_HISTORY_CONDITIONED_FORMATION_V1;
- ITC_RESIDUAL_SET_CORRECTION_V1;
- ITC_COUNTERFACTUAL_INDIVIDUATION_V1;
- ITC_READOUT_EXPRESSIVITY_V1.

**Labels:** [A] stated or measured by the authors; [I] our inference; [O] measured in this repo.

## The four questions per mechanism

| Mechanism | What triggers a new object? | When is that trigger wrong? | Evidence |
|---|---|---|---|
| **Ren & Zemel 2017**: recurrent attention | A box LSTM reads the image plus an external-memory canvas c_t = max(c_{t−1}, y_{t−1}) of already-segmented pixels; a score net stops at s < 0.5. Trained with Hungarian soft-IoU matching and scheduled sampling [A] | [A] "omitting distant objects and under-segmentation", attributed to 4× downsampling. [I] The canvas marks a merged region as segmented, so a second object inside it looks already explained | CVPPP SBD 84.9, \|DiC\| 0.8; KITTI ablation: no box net MWCov 57.0 vs 75.1 [A]. Small scale (≤ 3k training images) |
| **IterDet 2020**: history map | A history image H_xy = number of already-detected boxes covering the pixel, added after the first conv. Trained by splitting GT at random into B_old (into history) and B_new (targets) [A] | [A] More iterations raise recall but lower precision; 2 iterations are best; cost ∝ iterations. [I] **During training the history holds only CORRECT GT boxes. A wrong, merged prediction covering two objects never appears in the training history**, yet at inference that is exactly what sits over an absorbed object | CrowdHuman full-body (Faster R-CNN): recall 90.24 → 95.80, AP 84.95 → 88.08 [A]. The target case is overlapping boxes (2.4 pairs/img at IoU > 0.5) |
| **Progressive 2022**: accepted vs noisy queries | Does not create. Queries with score ≥ 0.7 are "accepted"; the rest are refined using relation features from IoU-neighbouring accepted boxes. Label assignment matches accepted queries first, then noisy ones to the remaining GT [A] | [A] Fig. 6 (TIDE error analysis): **"Missing" errors are unchanged (0.995 vs 0.995)**. The gains come from duplicates (1.780 → 0.902) and localisation [A]. [I] It cannot recover an object for which no query exists | CrowdHuman AP 90.7 → 92.0, MR 44.7 → 41.4 (S-RCNN, 500 queries) [A]. Relation extractor alone: +0.8 AP [A] |
| **Recurrent DETR 2023** | NOT READ (abstract only): the decoder is re-applied a variable number of times | — | +0.4 AP on CrowdHuman (abstract) |
| **MPP multiple birth and death** (Aval 2018) | A birth map ∝ the data energy of a disk (fraction of high-vegetation pixels in the disk vs its annulus). Death is driven by the data energy plus an overlap penalty U_o plus contextual energy. Simulated annealing; disks 2–8 m [A] | [A] "significant overlaps between the tree crowns": crowns are hard to discern in the high-vegetation mask (Fig. 15); very small crowns are omitted. [I] A data term computed from a binary canopy mask cannot tell one crown from two inside continuous canopy, and the overlap prior penalises two hypotheses in the same place | F-score 91 / 75 / 85% vs 70 / 41 / 20% for a region-growing baseline [A]. Street trees with strong priors (alignment, height) |
| **[O] Query Birth** (repo) | 8 reallocated queries placed by an RGB witness (decoder layer 3) | The RGB witness uniquely hits only 8 of 218 missing FIT occurrences. Hungarian matching gives birth slots 2 positive assignments in 8,640 opportunities | Oracle location + direct supervision: 130/147 births reach IoU 50. RGB direct: small Recall50 +1.1 pp, FP +10.9% [O] |
| **[O] ECRQ** (repo) | Residual witness R(x) = 1 − max_q a_q(x) (evidence not covered by any query); low-utility queries are recycled towards it | [I] An absorbed crown IS covered by its absorber, so R ≈ 0 there by construction | All gates failed; tiny-crown raw coverage +0 [O] |
| **[O] History-conditioned formation** (repo; IterDet-style, using ACTUAL predicted history) | Conv(2 → 256) history features (support and score-weighted support) added to the mask field and queries | The training history comes from real parent predictions, so the IterDet train/inference mismatch is removed | MASK − native: AP +0.15, small Recall50 +1.65 pp, FP +17% (guard is +5%). 311 new / 150 lost; target not met [O] |
| **[O] Counterfactual individuation** (repo; 1-vs-2 on a fixed binary tree) | Relational RGB scoring of "one vs two" on fixed hierarchy nodes | Supervision/inference mismatch; fixed hierarchy | Arms collapsed (AP 0.01). Relational RGB AUROC 0.77 on the hard panel [O] |
| **[O] EoMT readout oracle** (repo) | (Diagnostic, not a method) a linear oracle embedding on the frozen mask field | The SHIFT control (the same mask shape moved off the crown) also reaches about 0.95, so the field is highly EXPRESSIVE. This does not show that the field carries object identity | Oracle 0.943 vs deployed 0.283 for 1–4 m² crowns [O] |

## What the table says

1. **Ways to create a new object already exist:**
   - from uncovered evidence (canvas, history, residual);
   - from score ranking (top-K, accepted vs noisy, birth maps);
   - from an external location (centres, error detector).

   Sequential or multi-pass birth is NOT a contribution.
2. **(CORRECTED, see top.) The data suggest that hypothesis generation is an important bottleneck:**
   - given a correct location, decoders form the mask (Query Birth oracle 130/147; CellSAM GT boxes 90%) [O];
   - inferring that location from RGB fails (8/218) [O].
3. **[I] Hypothesis: trigger signals are blind to absorption specifically.**
   - Residual/canvas signals treat a covered region as explained.
   - The IterDet training history never contains wrong merged objects.
   - Score-ranked formation has, by definition of the failure, no adequate candidate there.
   - An absorbed object sits under an *incorrect but covering* hypothesis, the one state that none of these triggers is
     trained to treat as "evidence for another object".
   - Partial counter-evidence: our History-conditioned branch used real predicted history (which removes IterDet's
     mismatch) and still gained only +1.65 pp small recall at +17% FP.

## Q3 and Q4: what the method would change, and the experiment that separates the mechanisms
The method is not chosen yet. The only operation that differs from every row above is a candidate [P], not a claim:
- **Trigger on intra-hypothesis inconsistency rather than on uncovered evidence or rank.** For each EXISTING predicted
  instance, compute whether the evidence it covers is better explained by one object or by two. If two, emit the location
  of the second as a birth.
- Closest prior art: Zung 2017 (shape-based error detector giving locations to a corrector) and our own Counterfactual
  individuation (failed on a fixed hierarchy).
- Novelty risk: HIGH unless the inconsistency test and its training signal differ from Zung's.

**Separating experiment, to run BEFORE any training (CPU, existing outputs, a pre-registered reading):**
- **Step 1.** Stratify every missed reference crown in the saved parent predictions (SelvaMask, Mask DINO/Mask2Former
  family) into:
  - (U) uncovered: no prediction covers ≥ 50% of it;
  - (C) covered/absorbed: the D3 absorption event.
- **Step 2.** For each stratum, measure how often each existing trigger fires on the crown:
  - (a) the ECRQ residual R(x);
  - (b) the History-conditioned branch's new hits (its saved predictions vs the parent);
  - (c) the Query Birth RGB witness, if its per-reference output is saved.
- **Pre-declared reading:**
  - If all existing triggers recover (U) at least 2× more often than (C), they are structurally blind to absorption, and
    an inconsistency-based trigger is distinct in what it targets.
  - If (C) is recovered as often as (U), absorption is not special for triggers, and the candidate operation has no
    specific basis.
- Feasibility: History-conditioned predictions are saved (`pilot64/predictions`, `parent_history`). ECRQ and Query
  Birth per-reference outputs need checking.

~~No full training until this experiment has been read.~~ Superseded by the owner's Phase 1 decision: the prototype goes ahead, and the U/C/O diagnostic is auxiliary.
