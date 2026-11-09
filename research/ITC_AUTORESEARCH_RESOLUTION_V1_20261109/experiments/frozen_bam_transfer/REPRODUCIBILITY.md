# Reproduce the frozen BAM reference-fidelity comparison

Run from `/home/chung/RS/Image Segmentation` using `cs2_venv/bin/python`. Source data remain in the canonical TreeCrown-MultiSource-V1 root. No download, migration, raw rewrite or TEST access is needed. Existing model checkpoints are bound in every prediction file and rechecked by the verifier.

## Execution order actually used

1. `code/prepare_metadata.py`: official TRAIN-only raster headers, native GSD/metric footprints, fixed spatial/hash sampling. No RGB decode or annotation selection.
2. Protocol/cohort/code commit `20b5086`; source parser and explicit M2F inventory correction `bdf2482` occurred before predictions. No scientific setting changed.
3. `code/run_all.py`: nine separate frozen RGB inference processes, exactly24tiles/9views/system. Complete image-arm outputs are checkpointed as gzip JSON. Final mask lists index the immutable raw RLEs. The source API withholds references and the runner asserts its annotation cache is empty.
4. Scoring/matching implementation commits `32de7cf` and `4666772` preceded reference joins. Fixed example-selection/figure rules were committed in `d6cc84e`. Secondary support addendum `9796883` was recorded after inference began, before reference joins; it is labeled registered-late, not a primary confirmatory endpoint.
5. After all outputs were sealed in `PREDICTIONS_FROZEN.json`, `code/analyze.py` read only TRAIN references. It verifies identical inputs and produces all-area evaluation, raw coverage, matching, paired recovery/loss, same-count and source/group tables. Actual alpha-clipped reference RLEs are stored separately in `results/TRAIN_REFERENCE_MASKS.json.gz`; no model reads them.
6. `code/support_bound.py`: geometric oracle-bound diagnostic, no prediction changes.
7. `code/figures.py`: fixed metadata-selected tiles0/12 and three-seed gain/loss figure. Original image values preserved; only invalid display pixels are gray and individual raster outlines are drawn. Rendering inspected for coordinate/label quality, not used as an expert identity assessment.
8. `code/view_edge_diagnostic.py`: **post-hoc exploratory** quantification motivated by the fixed figures. A strict reference fragment proxy and simple edge contact are distinct. No inference filter or new policy was run.
9. `code/write_report.py`, then `code/verify.py` produce the evidence-grounded report and actual-output checks. The verifier does not establish original-method success or complete the overall research goal.

## Commands

```bash
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/prepare_metadata.py
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/run_all.py
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/analyze.py
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/support_bound.py
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/figures.py
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/view_edge_diagnostic.py
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/write_report.py
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/frozen_bam_transfer/code/verify.py
```

Preparation, inference and primary analysis have completion guards. These commands document the completed sequence; do not delete guards/results to replay it in place. For an independent replay copy only protocol and code into a new sibling experiment of the same directory depth, regenerate the unchanged cohort and compare bindings. Existing verified predictions can be rescored by a separately versioned analysis output; do not overwrite this completion seal.

## Scientific limitations that reproduction does not remove

- Only BAM TRAIN semantics are accessed. This source is external to verified task-training manifests, but legacy project exposure and universal pretrained-data independence are not established.
- References are original benchmark annotations, not three blind independent RGB interpretations or verified biological identities.
- Original-source crown area is unavailable. Do not relabel tile-area proxies as source-small metrics.
- Whole-system query/RoI budgets differ; only RR/matched-continuation contrasts are component-paired. Native GSD and physical support differ from historical SAME6; no resampling rescue or threshold search occurred.
- The original evaluator is reused for scale-independent outcomes. Its4.5cm physical-size and2pixel-boundary fields are intentionally withheld for BAM.
- No new optimizer/model training occurred here. Three seeds are repeated computations, not independent geography; two spatial groups are not a forest-population sampling frame.

Environment versions, exact file/checkpoint hashes and numerical/output validation are retained in FINAL_VERIFICATION.json and ENVIRONMENT.json. No external reviewers, institutional approval or positive novelty claim is fabricated.
