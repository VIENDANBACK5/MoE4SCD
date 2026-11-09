# Frozen BAM TRAIN transfer: reference-fidelity comparison

**Lock before predictions. No training, new method, tuning, TEST or protected reviewer access.** This is not execution of the independent-reference stability study. It measures transfer against existing published polygons on a geographically separate task-training source; complete project-exposure and all pretrained-data independence are not established.

## Rationale and fixed question

RR_MASK_INIT has a replicated exploratory small-coverage benefit on SAME6 relative to matched continuation. Its behavior outside those task-training orthomosaics is unmeasured in the current evidence package. Test whether its frozen output yields net reference recovery/quality gains on existing BAM TRAIN imagery, with R5, Mask2Former and original DINO as contextual frozen carriers. Do not infer a shared cause from common failures.

The source API explicitly supports BAM TRAIN under `legacy_reference`; retain that track. Bundled source Readme documents research use and citation under CC BY-NC-SA 4.0. Do not infer blanket redistribution rights from software terms. No dataset download or migration. Source: Troles et al., Remote Sensing 16(11),1935, https://doi.org/10.3390/rs16111935 . Original source masks, alpha and official split assignments are immutable.

## Sampling and physical geometry before outputs

`prepare_metadata.py` inspected only all 1439 official TRAIN TIFF headers, not RGB, masks or TEST members. CRS EPSG:25832 and explicit pixel spacing/tiepoints establish footprints; filenames are not geospatial truth. Two connected geographic components result from a fixed 2500m center-neighbor graph. Within each, hash order using seed20261109 selects 12 footprints with pairwise gap >=25m: **24 tiles total**, fixed in metadata/SELECTED_COHORT.json. These are spaced spatial blocks within two German geographic groups, not 24 independent sites or verified biological individuals. Historical project-wide use of BAM remains a limitation; this is not advertised as pristine final validation.

Registered checkpoint task-training manifests contain SelvaMask only. Generic pretraining exposure cannot be exhaustively excluded. Do not claim unqualified source independence. TIFF header spacing is approximately 1.61–1.79cm/px, finer than historical delivered4.5cm. Preserve original RGB/alpha and report gx/gy, physical field of view and 30pixel floor. Native1024views span approximately16.5–18.4m. No resampling/harmonized arm. Consequently a cross-source decline cannot identify selection/generalization independently from scale, forest, sensor or annotation changes.

All systems receive the exact same native1024views with starts[0,753,1024] along each2048axis:9views/tile. This is the released query-carrier tiler and an explicit matched-view adaptation for R5 (historical nominal stride768). No inference input depends on references. No inference thresholds/counts are chosen from outcomes.

## Frozen arms and model provenance

- R5 native COCO-initialized R50-FPN, seed42; its own mask scores/box-stage policy, max256returned RoI detections/view. No new voting/fusion arm.
- Mask2Former FP32-uniform R50, seed42,192queries/view; contextual comparison, not new training.
- Original Mask DINO mask-only seed42,300queries/view.
- `NATIVE_MASK_INIT` matched continuation seed42/43/44,300queries/view.
- Released `RR_MASK_INIT` seed42/43/44,300queries/view.

Nine frozen systems,1944view-forwards total. RR versus matched continuation is the principal paired component comparison. Original DINO is a practical reference only. R5 and M2F differ in pretraining/data/budget/formation and support system-level comparisons, not pure architecture causality. Use exact saved hashes; no backbone-only reload or weight adaptation.

## Inference and freeze

Canonical `TreeCrownData.open(... split='train', track='legacy_reference', access_role='inference')` supplies RGB/alpha with all annotation fields withheld. Intrinsic train2023 checks remain active. No direct image-loader bypass or protected-source fallback. Image/alpha inputs are hashed; each carrier runs in a separate process. Model outputs for **all arms and tiles** must be saved/frozen before reference scoring begins.

Mask threshold0.5 (query logit>0); valid-alpha clipping; score>=.05; min30validpixels; cross-view mask-IoU NMS>.5; independent overlapping masks retained. Query score is class probability times mean foreground mask probability; R5 retains native detection score. No cross-carrier score calibration or learned selector. Save nonempty raw mask RLEs, accepted masks, scores, query/view lineage and predicted boxes where available. R5 raw means returned native ROI masks, not its uncomputed full RPN/mask pool.

Use shared existing encoding/COCO evaluator. Its legacy physical-size calculations assume4.5cm: **do not publish those fields for BAM**. The unchanged all-area AP/AP50/AP75, maximum one-to-one recalls and pixel-overlap proxies remain valid; wrapper retains only scale-independent fields. Per-tile physical area summaries use actual header gx*gy separately.

## Reference and size limitations

Read TRAIN references only after prediction freeze. Preserve every nonempty alpha-valid independent reference and log empty/excluded masks. COCO lacks verified original crown IDs/full-source areas. Thus original-source-small Recall50 and its +2pp engineering gate are **NOT_ESTIMABLE**, not replaced by cropped target area. Report tile-mask-area strata as descriptive proxies, along with boundary-contact/unknown-source-size flags. No source-small improvement or independent identity reliability is claimed from them. Count unmatched predictions as FP against the reference convention, not verified biological absence.

## Metrics and analysis

AP/AP50/AP75 and COCO maxDets[1,10,100,400]; all-accepted one-to-one Recall50/75; TP/FP/FN/count; raw qualifying availability50/75; duplicates and significant split/merge proxies; per-block/group results; runtime/allocated memory; new and lost references. Do not infer errors in biological identity or fine boundary accuracy from these unreviewed polygons.

RR comparisons are seed-paired. Same-output-count control uses n=min(number final RR predictions, number final matched-control predictions) per tile, then the top n original-score final masks for both arms. This is fixed before outcomes and uses no GT count. Preserve gross gains/losses and matching-ambiguous reference membership; no arbitrary matched member becomes a definite recovered tree. Seeds and repeated views are not independent sites. Two geographic components allow descriptive transfer findings, not a forest-population significance claim.

## Finite gates and stopping

One frozen evaluation, no threshold/cohort/seed search after outcomes. Do not promote +2pp small-recall claims because required original areas are unavailable. A result can support or fail to support transfer of overall reference recovery; it cannot unlock an original method or settle visual identity ambiguity.

Resource bounds: max45minutes inference,1944forwards,8GiBallocatedGPU/arm,8GiBhostRSS where measured,2GiBnew outputs and keep>=20GiBdiskfree. GPU shared, not reserved. Resume only verified completed arm/tile outputs under identical hashes; never restart a live process or overwrite failed attempts. A coordinate/API error may receive one documented engineering correction without scientific-setting changes. No trained weights are written.

If the registered cohort/model/data cannot be executed, retain the exact blocker; do not substitute SAME6, TEST, another source or a smaller/favorable cohort. If executed, deliver actual masks/tables and one bounded research decision. No new module follows automatically.
