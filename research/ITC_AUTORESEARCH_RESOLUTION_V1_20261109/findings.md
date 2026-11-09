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
