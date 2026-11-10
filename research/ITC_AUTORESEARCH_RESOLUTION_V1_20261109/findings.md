# Research findings

## Current understanding

The practical task is to form independent visible-crown masks from RGB, preserving small instances without collateral losses. Established positive options already exist: released RR_MASK_INIT as a single carrier, and conventional R5+DINO fusion with extra inference and FP cost. These are inherited results on exposed SAME6, not new method/generalization claims. No original effective mechanism has been established.

## Cycle 1 finding

An actual frozen-DINO readout diagnostic ran on three fixed TRAIN windows, 298 reference occurrences. Native raw availability50 was 278; GT-conditioned fitted target masks qualified 260. One missing target became qualifying, 19 native-qualified targets lost qualification. Every lost case increased false foreground pixels, and none increased false-negative target pixels. Loss fell on sampled pixels while complete mask geometry regressed. Native dot-product reproduction and saved-RLE IoUs were verified.

This reveals a limitation of the probe, not a feature-capacity ceiling. The mask family already contains the native successes, and bounded sampled BCE fitting cannot erase that constructive fact. An elementary counterexample shows sampled loss can approach zero while unsampled foreground makes IoU fail. This is not novelty. Close this probe without another loss/optimizer sweep.

## Independent-evidence work

The quantitative independent-study protocol was amended and its original ledger preserved. Existing metadata does not confirm an eligible new geography, independent annotators or sufficient spatial blocks; it does not prove those resources absent. Public SelvaMask inter-annotator study is explicitly test-zone based and is not opened. ForestSeg/ForestSemantic are candidate public resources with task/provenance conditions still unresolved, not cleared independent references.

## Lessons and constraints

No architecture change, TRAIN auxiliary loss or surrogate improvement is justified merely because a mask is missing. A failed probe must first be checked against the quantity it purports to measure. Do not reopen closed query recycling, selection, local head, KD, pooling or graph recipes. No SAME6 tuning, TEST or protected review. Human consensus is not biological identity. Newly exposed discovery imagery cannot later be called independent method validation.

## Open questions

Which newly accessible reference-stable residual is important enough to justify an intervention? Can an external authorized source support transfer measurement without geometry/annotation-policy confounds? What exact cause links RGB evidence to missed instance formation? These remain unverified. Goal active; no background scheduler or training daemon is claimed.

## Cycle 2: actual external reference-fidelity test

A fixed metadata-selected BAM TRAIN cohort of24spaced tiles/two geographic components was evaluated with9frozen systems (1944forwards), preserving native1.61–1.79cm RGB/alpha. All216image-arm outputs were frozen before references. Task-training inventories contain only SelvaMask; legacy project/pretraining exposure and independent reference stability remain unresolved, so this is not an unqualified independent final test.

On1106reference occurrences RR versus matched continuation gives mean overall Recall50+0.452pp, AP-0.041, FP+64.7/seed; same-output-count Recall50+0.332pp. Seed netTP directions are-2,+9,+8. Of58new final matches, only8lack a qualifying control final raw mask;43old matches are lost. Original-source-small outcome is NOT_ESTIMABLE because COCO lacks trustworthy full-source area/identity; tile-mask area is not substituted.

R5/M2F/DINOseed42 have447common final omissions and320common raw absences in their registered observed pools. Every reference has an oracle one-view support bound>=.75; hard view-containment exclusion cannot explain those absent qualifying masks. Neither this permissive bound nor common error establishes RGB sufficiency, independent visual identity or a unique cause.

Post-hoc fixed-figure analysis finds many query outputs touch internal view edges, including many matched predictions; a strict fragment proxy certifies a small subset of unmatched outputs while others remain unclassified. Do not equate spatial edge contact with instance redundancy or the causal share of all errors. Generic cross-crop query association has CropFormer prior art; no new architecture is justified from these observations.

Actual-output verification checked360653raw masks and46596final masks: alpha validity, view support, equal input pixels, count conservation and unchanged checkpoint hashes. Full source/seed results, masks, source/geometry bindings, figures and reports exist. All study model processes are terminal. Overall goal remains active; a robust new solution/original method is unproved.

## Cycle 3: missing evidence is now a concrete handoff

Six metadata-spaced native RGB cases from the already exposed BAM TRAIN cohort were exported and verified pixel-exact. Blank standard LabelMe templates, uncertainty/abstention semantics and an independent-mask importer are ready. The52.55MB archive excludes reference/prediction/source-case metadata and retains dataset attribution. No participation, response or reference stability result exists. This preparation does not turn two geographic components into independent-site confirmation or solve source-small-area provenance.

The closest Steier2024 multi-annotator resource requires an author request; no labels have been acquired or author contacted. Public SelvaMask repeated annotations remain outside the allowed TEST/reviewer boundary. A concrete availability question for three qualified independent annotators is pending. Additional published-polygon evaluation cannot replace their judgments; no new method branch is justified from the missing information alone.

## Cycle 4 synthesis and stopping boundary

The executed empirical evidence is now an actual-source-linked manuscript, with nine-system tables generated directly from CSV and explicit limits on all common-error/capacity claims. Rechecked278BAM artifacts,31oracle artifacts and8historical decisions. Mei primary full text distinguishes independent true-color annotation from later NIR-assisted fusion; no accessible original masks were established. RandCrowns procedure is not a direct replacement for the stipulated single-RGB polygon references. No labels were acquired.

The outstanding input is actual qualified independent reference evidence. The same absence recurred through cycles2/3/4 while preparation/reporting progressed; now the bounded safe preparations are complete. Completion remains unproved and the native goal is requested BLOCKED at that external evidence gate. This does not close RGB ITC research, declare impossibility or turn a limited transfer result into a new method.

## M1 research after explicit user resume

Original method remains the goal; the empirical draft is evidence archive. Same physical FOV and same input information can be separated using exact nearest-neighbor replication with a known left inverse. This permits a narrow encoding-dependence test without pretending decimation is invertible or that RGB identity is verified. DINO-MS already uses same-location scale consistency while excluding its smallest views; DeepScaleSpaces already treats irreversible scaling by semigroups. The sketched partial assignment loss admits abstention/gate/count collapse unless further constrained; no new loss implemented. Metric physical policy cannot be guaranteed by f(I) alone without calibration/identifiable priors. Research note and NOT_RUN four-arm design are in ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110. No original mechanism established.

## M1 actual encoding diagnostic

The registered12-forward TRAIN pilot now measures large dependence on information-equivalent raster encodings: A(native1024)raw278/final276 versus B(rep2048)raw113/final98; C(coarse512)raw174/final163 versus D(replicatedcoarse1024)raw213/final206. Raw75 and same-count controls agree with a large formation/retention change, not mere count variation. Exact backbone-input left inverse and native-mask reproduction checks pass. This rules out deleted inputRGB for those contrasts. It does not isolate an architecture cause: fixed1024fine-tuning and changed receptive fields/encoder grids remain alternatives. Known input canonicalization can undo the engineeredBinput; generic scale treatment is prior art, not a new method. No observability loss or training launched. Goal active/unachieved.

## Current M1 outcome: established learning control and residuals

Two full-scope398-update runs on796eligibleTRAINwindows completed. Conventional lossless-scale exposure/additional computation reduces rawA/Bgap147→36, but nativeAP−.960/AP75−2.673/FP+59/smallTP−2. The narrow gap screen passes; a practical replacement and original-method entry do not. S1processes2.5×rasterpixels, so exposure and compute are not isolated. No independent validation.

A finite post-hoc analysis on saved masks establishes38grossrawlosses/2gains behind the S1ABnetgap36.20losses start atnativeIoU75;16ofthose retain>=90%ofpublished-tile referenceRLEandare notsource-truncated.35of38independently chosen bestwitnesses have morefalseforeground, but queries/grids differ and this is not a causal decoder trajectory. Native smalllosses endat.499884and.366873, respectively. Thus threshold/fragments alone do not explain the remaining functional phenomenon; specific internal causes and an original remedy remain unverified. The tested control and finite analysis are closed, not the RGBITCproblem. No follow-on module/epochs/thresholds queued.

## Readout-factor diagnostic after user resumes autoresearch

Independentreference annotation unavailability was too broad as a stop for all inference research. A frozen final-readout E/Finterchange now executes on existingTRAINreferences without resolving that separate gate.12forwards,24pools, exactnative reproductions and independent7200raw/2996final-mask verification. S1ABrescues2of20strongresiduals,BA7of20; both lose many existingmasks globally(BA11new/122lostfinal). Samecount/source/originalscorecontrols do not rescue the recipe. A paired channel sign change preserves native logits but changes hybrid logits, so interchange failure cannot certify intrinsic feature deficiency. Same-query witnesses do not establish original localized identity: only1of7BArescuedqueryboxes reachesboxIoU50. No originalmethod or internalcause identified; no alignment/head/loss invented.
