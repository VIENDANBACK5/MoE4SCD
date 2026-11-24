# Before/after fixes and migration
## H01: worker sampling
Before: field dataset had no real shared set_epoch API; failing persistent-worker regression preserved in REGRESSION_BEFORE.log.
Root: worker copies and index-only crop seeds did not expose current epoch.
Patch: clean_benchmark/dataset.py, sampling.py, trainer set_epoch calls. After: same seed/epoch replays; next epoch changes train crop; validation stable. REGRESSION_AFTER.log and final test_persistent_worker_epoch_changes_real_crops.
Migration: future training streams differ; historical data/results unchanged. Existing-directory resume is refused, not certified by this patch.

## H02: native instance targets
Before: nested tiny target fixture returned1 instance instead of2; INSTANCE_REFERENCE_BEFORE.log.
Root: priority raster and minimum16px filtering erased native reference objects.
Patch: clean_benchmark/train_maskrcnn.py, independent original RLE crop masks and per-mask synchronized augmentation. After: both original IDs including one-pixel nested mask retained; INSTANCE_REFERENCE_AFTER.log and final test_instance_loader_keeps_nested_one_pixel_original_target.
Migration: future supervision changes intentionally; old checkpoints/metrics not reinterpreted. Original field supervision remains legacy first-wins and logs invisible IDs.

## H03/H04: execution and source gate
Before: unversioned scripts could load hardcoded test/output/config; helper tests did not prove production enforcement.
Patch: 3 live CLIs require integrity-config and actual runtime-spec match; 59 historical CLIs fail before execution. Original functions remain importable;68 original source entries archived. Hashes, native-source attestations, content roles, destination and confirmed dependency order are checked. Malformed/missing config, unknown critical support and changed manifest fail nonzero.
Migration: old direct CLI invocations intentionally stop. Replaying historical experiments needs a separately reviewed migration; do not remove the gate. This is preventive control, not a reproduced historical defect in every blocked script.

## H10–H13: output, metrics and budget controls
New explicit stage/native lineage, original-reference matching, score-independent grouping, partial-target accounting, query-overflow rejection, final accumulation flush and actual update budget checks. Synthetic tests include empty/duplicate/merge/split/overlap/invalid support/non-square affine and confidence-only changes.
These are newly tested preventive controls; no before/after real AP is asserted.

## Verification limitations encountered
pytest was unavailable; no package installed. Existing standalone9 tests were executed directly. Registered data-root symlink initially failed closed; READ_ROOTS.json binds the existing baseline-derived physical root. DATA_SMOKE_BEFORE.log retained; subsequent16 data-only cases passed.
All63 edited implementation files are listed with before/after SHA256 in AUTHORIZED_SOURCE_CHANGES.json. Historical outputs, original labels and raw data were not rewritten.
