# Hypothesis Availability Is Not Instance Recovery: Frozen Cross-Carrier Evidence for RGB Tree-Crown Segmentation

**Empirical/mechanistic first draft, 2026-11-10.** No original segmentation method, independently validated biological identities, or geographically confirmed general solution is claimed. Authorship, venue, final submission format and contribution-level review remain open. This draft records completed work; it does not replace the unachieved method-development goal.

## Abstract

Recovering individual tree crowns from RGB requires both mask-formable hypotheses and their preservation through deployment inference. We study these quantities separately using frozen region-based and query-based carriers. A metadata-selected external TRAIN cohort contains 24 native-resolution BAMFORESTS tiles and 1,106 published reference occurrences. Nine frozen systems produce 1,944 view-level forwards, with predictions hashed before reference access. An established redundancy-removal component adapted to Mask DINO yields a mean paired Recall50 improvement of 0.452 percentage points over matched continuation, an AP change of −0.041 points, and 64.7 additional reference-relative false positives per seed. At matched output counts the recall change is 0.332 points. Across three seeds, 58 reference–seed occurrences are newly matched and 43 are lost; only eight gains lack a qualifying control final-query raw mask. Three representative carriers share 447 final omissions and 320 raw absences in their observed pools. A view-containment bound excludes hard geometric infeasibility at IoU50/75 for this cohort, without establishing visual identifiability or learnability. A separate TRAIN oracle demonstrates that low sampled readout loss can coexist with complete-mask degradation. These bounded findings motivate explicit separation of formation, retention and reference reliability. They establish neither a new method nor a shared architectural cause.

## 1. Introduction

Individual tree crown segmentation predicts independent masks from RGB imagery, including small crowns adjacent to larger ones, visually similar contacting crowns, shadows and nonconvex canopies. Published polygons specify an evaluation convention. They are not necessarily independently verified biological individuals, and a mismatch alone does not identify the computational mechanism responsible.

The project already contains successful Mask R-CNN, Mask2Former and Mask DINO carriers, as well as an established-component redundancy-removal adaptation, RR_MASK_INIT. Previous development repeatedly tested plausible explanations for missing instances. Those closed recipes cannot be interpreted as general impossibility results, and their failures do not justify another architectural module automatically. Here we instead ask what can be concluded from actual frozen predictions under an explicitly bounded transfer evaluation.

Our contributions are empirical and methodological in measurement: (i) a reproducible distinction between reference-wise raw availability, simultaneous one-to-one recovery and final score-ranked quality; (ii) a matched three-seed transfer comparison of RR and its continuation control with gross gains/losses and equal-output-count controls; and (iii) counterexamples to inferring feature insufficiency from an unsuccessful sampled readout fit. These are proposed report contributions, not an assertion of publication-level novelty. The unresolved requirement is a reliable independent-reference failure cohort from which a discriminating causal intervention could be justified.

## 2. Related mechanisms and annotation evidence

[Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html) forms masks conditioned on regions. [Mask2Former](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.html) uses query masks and masked cross-attention. [Mask DINO](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Mask_DINO_Towards_a_Unified_Transformer-Based_Framework_for_Object_Detection_CVPR_2023_paper.html) couples detection and segmentation hypotheses. Whole-system comparisons do not isolate any one of these mechanisms, because training, pretraining, hypothesis budgets and scoring also differ.

[Salience DETR](https://openaccess.thecvf.com/content/CVPR2024/html/Hou_Salience_DETR_Enhancing_Detection_Transformer_with_Hierarchical_Salience_Filtering_Refinement_CVPR_2024_paper.html) explicitly addresses scale bias and redundancy in two-stage selection. RR_MASK_INIT adapts its redundancy-removal component; it is neither a reproduction of the complete Salience DETR architecture nor our novel selector. The tested code uses top-4K class responses, level-wise NMS over ±1-cell grid supports at IoU 0.3, followed by exactly K=300 sites. These are grid supports, not predicted instance-box IoUs. Both RR and the matched control retain the recorded mask-enhanced initialization and their sealed training lineage.

Independent-reference uncertainty already has direct prior art. [Steier et al.](https://doi.org/10.3390/rs16152786) assess crown annotation quality. [RandCrowns](https://doi.org/10.1109/JSTARS.2021.3122345) studies evaluation with imprecise labels. Its inspected multi-annotator experiment uses rectangular delineations assisted by multiple remote-sensing products and removes some disagreement cases; it is not automatically a same-single-RGB independent mask reference. [Mei et al.](https://doi.org/10.5194/isprs-archives-XLVIII-2-2024-257-2024) report independent true-color annotation before multispectral-assisted fusion. A fusion method's use of NIR does not by itself disqualify its original RGB annotations, but the required label assets have not been obtained here. We do not claim annotation uncertainty, affinity learning or crop consistency as novel principles.

## 3. Measurement framework

Let image I with valid domain V produce a raw mask set P_raw, an accepted set P_acc, and a final set P_final after registered suppression. Masks are independent subsets of V; overlaps are allowed. For a reference Y_i and threshold τ, define availability

\[
A_\tau(Y_i;P)=\mathbf 1\{\max_{M\in P}\operatorname{IoU}(M,Y_i)\ge\tau\}.
\]

Reference-wise availability may count the same prediction for several references. Final TP is instead the maximum cardinality bipartite matching over qualifying mask–reference edges:

\[
T_\tau(P,Y)=\max_{\mathcal B\text{ one-to-one}}|\mathcal B|,
\quad (M,Y_i)\in\mathcal B\Rightarrow\operatorname{IoU}(M,Y_i)\ge\tau.
\]

Recall is T_τ/|Y|; FP is the number of final predictions not matched at that threshold. These reference-relative FPs are not verified non-tree objects. Score-ranked COCO AP is a separate measure with maxDets [1,10,100,400]. The reported all-accepted maximum-matching recall is not COCO's maxDets-truncated AR. Where membership could vary across maximum matchings, alternating-path ambiguity is retained rather than treating a chosen correspondence as unique identity. No such ambiguity affected the paired gain/loss tables here.

For matched RR/control outputs, NEW counts reference occurrences matched only by RR and LOST counts those matched only by control. The identity

\[
\Delta T_{50}=\mathrm{NEW}-\mathrm{LOST}
\]

prevents gross recoveries being reported as net benefit. We separately ask whether a new final match had any qualifying control final-query raw mask. This is an observed-pool comparison; it does not prove an inaccessible intermediate state never contained useful information.

### A geometric feasibility bound

If every output mask from a view lies in its support W, then for any nonempty reference Y,

\[
\operatorname{IoU}(M,Y)\le |Y\cap W|/|Y|.
\]

Proof: |M∩Y|≤|W∩Y| and |M∪Y|≥|Y|. Maximizing the right side over the registered views gives an upper bound U(Y). This elementary bound is not novel. U<τ certifies hard containment infeasibility for single-view masks; U≥τ supplies no constructive mask or RGB identity guarantee. Context outside the view may still matter. The support analysis was registered after inference began but before reference joins, and is treated as a secondary diagnostic.

## 4. Data and frozen execution

We use installed [BAMFORESTS](https://doi.org/10.3390/rs16111935) TRAIN imagery through the canonical treecrown_data inference role. No protected TEST/reviewer material is opened. All 1,439 TRAIN raster headers provide metric geometry; a fixed hash order and at least 25 m footprint separation select 12 tiles in each of two 2.5 km-neighbor coordinate components. These are spatial groups, not two independent countries, acquisitions or biological populations.

Delivered spacing is 1.61–1.79 cm/pixel, with 2048×2048 RGB/validity tiles. Each carrier receives the same nine unresized 1024×1024 views, with axis starts [0,753,1024]. R5's historical 768-pixel stride is explicitly changed to the common schedule. Its returned RoI limit is 256, Mask2Former has 192 queries, and Mask DINO has 300. Cross-architecture budgets are not equated; the RR/control contrast alone preserves exactly the same query budget and training exposure.

The common deployed mask materialization uses native coordinates, valid-pixel clipping, score≥0.05, at least 30 valid mask pixels and greedy mask-IoU suppression above 0.5. Query-system scores are class probability times mean foreground probability; R5 uses its recorded class score. No score calibration or threshold fitting is performed. Scoring differences remain a cross-carrier confound. R5 raw means its returned post-box-pipeline masks, not all RPN proposals; query raw means final decoder query masks.

RR and matched NATIVE_MASK_INIT continuation checkpoints are paired at seeds 42/43/44. Original Mask DINO mask-only, Mask2Former FP32-uniform and R5 context controls use seed 42. All nine systems remain frozen. Source policies, native RGB, valid-pixel semantics, independent RLE masks, inference outputs and hashes are saved. All 216 tile–system outputs were frozen before reference access. The nonempty valid reference denominator is 1,106 annotation occurrences. Full source-crown identities/areas are not available from this COCO inventory.

Task-training inventories contain the three SelvaMask sources rather than BAM. This does not exclude earlier project exposure or all pretrained-data overlap. The source paper's additional-date/field-assisted labeling process does not establish independent single-RGB interpretation. Thus the experiment is external TRAIN reference fidelity under changed conditions, not a pristine independent final test. Scale, forest, acquisition and annotation policies change jointly; this is not an isolated GSD experiment. The 30-pixel floor and physical view footprint have different meanings from the historical 4.5 cm cohort.

## 5. Final-mask results

| System | AP | AP75 | Recall50 % | Recall75 % | TP50 | FP50 | Raw50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| rr_seed42 | 6.342 | 3.641 | 44.665 | 15.552 | 494 | 4881 | 710 |
| control_seed42 | 6.222 | 3.582 | 44.846 | 15.190 | 496 | 4818 | 689 |
| rr_seed43 | 4.017 | 2.604 | 36.438 | 12.387 | 403 | 3973 | 622 |
| control_seed43 | 4.187 | 2.720 | 35.624 | 12.206 | 394 | 3893 | 625 |
| rr_seed44 | 5.299 | 3.609 | 37.794 | 12.929 | 418 | 5000 | 641 |
| control_seed44 | 5.373 | 3.658 | 37.071 | 13.110 | 410 | 4949 | 623 |
| dino_seed42 | 6.149 | 3.405 | 44.756 | 15.280 | 495 | 4768 | 697 |
| r5_seed42 | 4.012 | 2.438 | 45.389 | 13.653 | 502 | 6976 | 543 |
| m2f_seed42 | 3.822 | 2.217 | 33.454 | 10.488 | 370 | 3356 | 533 |

AP is in 0–100 units and Recall in percent. Higher R5 Recall50 coexists with many more unmatched predictions; it is not unconditional superiority. Historical SAME6 metrics are not put in this table or directly subtracted from it.

### Matched RR effect

| Seed | ΔAP | ΔRecall50 pp | New | Lost | Net TP50 | ΔFP50 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | +0.120 | -0.181 | 15 | 17 | -2 | +63 |
| 43 | -0.170 | +0.814 | 23 | 14 | +9 | +80 |
| 44 | -0.074 | +0.723 | 20 | 12 | +8 | +51 |

Mean paired changes are Recall50 +0.452 percentage points, AP −0.041, AP75 −0.035, TP50 +5.0 and FP50 +64.7 per seed. The prespecified equal-count comparison truncates each tile's final outputs to the smaller of its two GT-free counts, retaining original score order. Mean Recall50 change is then +0.332 points. One seed loses recall and two gain it. There is no statistical-significance claim or seed-based geographic confidence interval.

There are 58 new and 43 lost matched reference–seed occurrences. Of the new matches, eight lack a qualifying control raw mask, 30 previously fail acceptance and 20 fail suppression. Of the lost matches, four lack an RR qualifying raw mask, 19 fail acceptance and 20 fail suppression. These categories describe frozen policies and observed pools. They do not isolate a unique learning cause or show that all changes are score ranking: decoder context and geometry also differ between the trained checkpoints. Raw availability changes by +21, −3 and +18 references across seeds. Net final performance remains the relevant endpoint.

Original-source-small Recall50 and its predeclared +2-point gate are **NOT_ESTIMABLE**. The 488 tile-mask references below 9 m² are retained only as a crop-area proxy, with boundary and clipping information. A cropped large crown cannot silently become a small source crown. Missing estimability is not a negative small-generalization result.

### Shared omissions and permissive support

The R5/Mask2Former/original-DINO seed-42 final-success patterns are 000:447, 001:92, 010:26, 011:39, 100:91, 101:106, 110:47 and 111:258. Their observed raw pools share 320 qualifying-IoU50 absences. These results show complementary reference recovery and common published-reference omissions; their union is an oracle diagnostic, not deployable fused output.

Every reference has U(Y)≥0.75 and every actual raw IoU respects the bound. Therefore hard single-view containment exclusion does not explain these IoU50/75 absences. Neither support nor model disagreement establishes visually stable identity, RGB sufficiency or a common architectural bottleneck. The reliable multi-annotator denominator required for R_common and R_all is missing.

### Exploratory view-edge observations

After inspecting the fixed metadata-selected figures, an edge-contact diagnostic was defined without changing predictions. Original DINO has 3,073 of 4,768 unmatched predictions touching an internal view edge; 166 of 495 matched masks do too. A stricter reference-conditioned fragment proxy identifies 111 unmatched masks with another retained qualifying view prediction. The remaining edge masks are unclassified. Neither contact prevalence nor this conservative proxy measures the causal share of errors or the net effect of a filter. Generic crop association already has [CropFormer prior art](https://openaccess.thecvf.com/content/ICCV2023/html/Qi_High_Quality_Entity_Segmentation_ICCV_2023_paper.html); no new fusion, filter or crop architecture was trained.

## 6. A negative oracle-fit result is not feature insufficiency

A separate registered diagnostic uses the frozen seed-42 mask-only DINO on three fixed eligible TRAIN windows, one per already seen source, containing 298 crop-reference occurrences. For each known target, a free 256-dimensional mask readout is fitted using balanced sampled logistic supervision, 4,096 positive and 4,096 negative pixels, two fixed starts and 300 Adam updates. GT selects/fits the target readout, so this is an oracle, not RGB-only deployment or matched system comparison.

Native raw availability50 is 278/298; fitted target qualification is 260/298, with one newly qualifying target and 19 lost qualifications. Every loss increases false foreground without increasing target false-negative pixels. Complete-RLE recomputation agrees with stored IoUs and native logits are reproduced exactly. The poorer fit cannot erase the constructive existence of native successful readouts in the same feature family.

For an elementary counterexample, let F=(1,0) on target pixels P and an unsampled negative region B, and F=(−1,0) on sampled negatives. With q=(a,0), a→∞ gives sampled logistic loss→0 but thresholded foreground P∪B. IoU=|P|/(|P|+|B|)<0.5 if |B|>|P|. This is not a theorem about the actual feature field or a novel loss proposal. It explains why the tested sampled fitting failure is not a negative capacity certificate. The probe is closed; no replacement-loss sweep follows.

## 7. Scientific interpretation and limitations

The observed RR benefit outside the task-training source inventory is modest, heterogeneous and accompanied by more reference-relative FPs. Available hypotheses, newly formed masks and final matches are distinct quantities. A policy can increase coverage while losing previous successes, and an optimized diagnostic can reduce the full-mask quality it was meant to probe. Neither outcome identifies a universal representation limitation.

Several limitations are binding: only two spatial components in one German region; non-probability cohort selection; incomplete project/pretraining exposure exclusion; published rather than independently confirmed identity; unavailable original crown area; differing carrier raw-pool definitions, pretraining and scores; and a compound acquisition/scale shift. Three training seeds are computational repetitions, not independent forests. Correlated errors may reflect annotation convention, context, features, supervised optimization or multiple causes. No information-theoretic RGB impossibility is inferred.

Six native RGB images have been prepared as a prediction-blind annotation-feasibility handoff, with overlapping polygon membership, ambiguity/abstention and explicit NOT_STARTED status. **Zero independent human responses have been collected.** The handoff is not evidence of consensus or generalization. It cannot supply prevalence claims or full-source small-crown identities. Protected TEST/reviewer data remain closed; authors or annotators have not been contacted.

## 8. Conclusion

The executed evidence supports a bounded empirical result: an established redundancy-removal adaptation has limited mixed transfer on the registered BAM cohort, and raw hypothesis availability is insufficient to characterize final recovery. A separate oracle shows why fitting failure alone cannot diagnose feature insufficiency. A new effective mechanism, a reference-stable common failure cohort and original-method novelty are not established. Retain the established carriers and their measured operating-point trade-offs; do not open another module from these results alone. The outstanding requirement is qualifying independent reference evidence that can discriminate visual interpretation from model formation errors.

## Reproducibility and evidence links

Inference used only RGB/validity, no optimizer updates, and 1,944 forwards. Summed recorded inference/encoding/NMS time is 766.631 s, excluding checkpoint loading and RGB I/O; it is not total study wall time. Maximum allocated GPU memory is 1,417.416 MiB on a shared GPU, not reserved capacity. The separate oracle ran three carrier forwards and recorded 15.714 s/1,654.46 MiB. No new carrier training or independent reviewer outcome is claimed.

- [Frozen transfer reproduction and environment](../experiments/frozen_bam_transfer/REPRODUCIBILITY.md).
- [Prediction freeze](../experiments/frozen_bam_transfer/PREDICTIONS_FROZEN.json), [actual saved masks](../experiments/frozen_bam_transfer/predictions), [all-area metrics](../experiments/frozen_bam_transfer/results/METRICS.csv).
- [Paired recoveries/losses](../experiments/frozen_bam_transfer/results/RECOVERY_AND_LOSS.csv), [equal-count results](../experiments/frozen_bam_transfer/results/SAME_COUNT_RESULTS.csv), [source-group results](../experiments/frozen_bam_transfer/results/SPATIAL_GROUP_RESULTS.csv).
- [Fixed cross-carrier figure](../experiments/frozen_bam_transfer/figures/FIXED_CROSS_CARRIER_COMPARISON.png), [fixed RR gain/loss figure](../experiments/frozen_bam_transfer/figures/RR_NEW_AND_LOST.png). Selection rules/proxy timing are in the experiment protocol and figure manifest; these are not independent human identity judgments.
- [Oracle report and actual RLEs](../experiments/frozen_readout_capacity/analysis.md).
- [Blank blind handoff and dependency](../experiments/reference_evidence_handoff/README.md).

Citation URLs identify verified primary sources. Four carrier/selection BibTeX entries were fetched from CVF; available additional entries come from Crossref. Any failed metadata fetch remains explicit in CITATION_FETCH_STATUS.json rather than being reconstructed from memory. This draft needs human authorship/venue review and reference-format completion before submission. No accepted-paper or original-method readiness is claimed.
