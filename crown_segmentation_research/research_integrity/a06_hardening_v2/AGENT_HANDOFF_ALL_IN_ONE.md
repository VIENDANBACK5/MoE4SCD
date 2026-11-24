# TreeFlowNet A06 v2 — independent hardening and verification
Completed 2026-10-15. Dedicated versioned work area: research_integrity/a06_hardening_v2. No full training, oracle rerun, historical test performance evaluation, calibration fitting, AP optimization, new model architecture, data download or dependency installation.

## I. Executive verdict
Implemented preventive controls in production loaders and registered CLI paths, not only standalone helper tests. Inspected A00–A05 source evidence, original Parquet/prepared COCO/valid masks, F/MR loading/training/evaluation, archived G/Q and diagnostics, matching/grouping/scoring and source provenance.
Freshly verified455 tiles /90462 annotation records against14 available native Parquet shards; actual affine and six-support first-wins replay PASS. No critical native-to-prepared mismatch reported. Original pre-rasterization vector polygons and biological correctness are not verified.
Final executed tests: 87/87 PASS (78 new-suite tests plus9 established synthetic tests), zero failures/errors/skips. Repeated nested self-tests are not extra unique tests.
Preservation: 1038 baseline paths, 975 unchanged, 63 authorized implementation edits, 68 archived original source entries verified;14 raw source hashes and prior independent review/snapshot unchanged. No unauthorized drift.
Level1 available-data infrastructure integrity is ready within declared scope. Complete exploratory FGQ comparison and confirmatory benchmark are not ready.

## II. Findings table
| ID | Severity | Root cause | Scientific impact | Fix | Test | Status |
|---|---|---|---|---|---|---|
| H01 | HIGH | Persistent workers did not share an actual epoch-dependent sampling state | Repeated crops can misrepresent training exposure | Shared epoch state; seed/epoch/tile/crop RNG; fixed validation | test_persistent_worker_epoch_changes_real_crops | FIXED_TESTED |
| H02 | CRITICAL | Instance loader rasterized original overlapping masks and silently filtered tiny targets | Nested/one-pixel native instances disappear from supervision | Independent native RLE masks, IDs retained, synchronized transforms | test_instance_loader_keeps_nested_one_pixel_original_target | FIXED_TESTED |
| H03 | CRITICAL | Historical CLI paths could access old output destinations and test/scorer labels | Historical overwrite or forbidden model-selection evaluation | Mandatory preflight for 3 active CLIs; 59 legacy CLIs refuse direct execution | test_active_entrypoints_require_gate_before_checkpoint_or_data | CONTROLLED_TESTED |
| H04 | HIGH | Declarations and helpers were not sufficient evidence of enforceable source/data provenance | A passing toy gate could authorize an unverified experiment | Hash-bound real source scan, full prepared inventory, native affine and registered raw roots | test_empty_missing_incomplete_and_bad_hash | CONTROLLED_TESTED |
| H05 | HIGH | Native metadata area and native RLE area are distinct | 87 annotations change physical size bins | Native affine determinant times original RLE area for new metrics; preserve metadata/history | test_affine_area_nonsquare_rotated_and_resize | EXPLICIT_TESTED |
| H06 | CRITICAL | Original overlap, unique support and first-wins ownership differ | Single-owner raster is not the full instance reference | Six separate supports; replay old priority exactly; original GT retained for matching | test_overlapping_GT_not_owner_raster | CONTROLLED_TESTED |
| H07 | HIGH | Two groups have byte-identical native masks with different IDs | Cannot establish whether source duplicates or separate intended records | Preserve and flag; no deletion or ontology change | test_six_supports_and_original_preserved | UNRESOLVED_SOURCE_SEMANTICS |
| H08 | CRITICAL | Small border overlaps and spatial proximity do not prove independent sites | Random tile folds/erosion do not establish independent confirmation | Configurable support/buffer policy, unknown cross-grid fail closed, buffered calibration folds | test_spatial_overlap_buffer_and_unknown_crossgrid | CONTROLLED_NOT_INDEPENDENT |
| H09 | CRITICAL | Historical test already informed development; independent acquisition not registered | Confirmatory claims unsupported | Role/content hash restrictions; external holdout requirement | test_mode_mismatch_and_confirmatory_without_holdout | EXTERNAL_HOLDOUT_REQUIRED |
| H10 | HIGH | Query slots, materialized proposals and accepted predictions had different meanings | Raw unmatched slots are not equivalent to detections | Stage0/1/2/3 lineage and explicit objectness/selection contracts | test_inactive_Q_not_materialized | CONTROLLED_SYNTHETIC |
| H11 | HIGH | Confidence and overlapping membership can confound grouping | AP or E2E grouping alone cannot establish identity learning | Structural unique-cover and E2E metrics, abstention, fixed pair/candidate caches | test_score_only_structural_invariant_E2E_changes | CONTROLLED_TESTED |
| H12 | HIGH | Fragment-only Q sanity does not validate complete-crown formation or query capacity | Crop prior, target truncation or overflow can bias F/G/Q | Native crop accounting, common frozen plans, quota infeasibility and overflow rejection | test_shared_planned_window_drives_real_field_and_instance_loaders | CONTROLLED_RECIPE_PENDING |
| H13 | HIGH | Final minibatch/accumulation could change realized optimization exposure | Nominal budgets can differ from actual updates | No drop_last; flush short accumulation; budget assertions; nonfinite loss fails | test_accumulation_flushes_partial_budget | CONTROLLED_TESTED |
| H14 | HIGH | G1/Q benchmark recipes and natural-scene stitching not complete | Three-way comparison and scene-level claims not yet executable/validated | Reject unsupported recipes; same-grid stitch only, explicit parent lineage | test_incomplete_comparison_does_not_become_ready | BLOCKED_DESIGN_SCOPE |
| H15 | MEDIUM | Upstream vector labels, acquisition identity and license evidence unavailable | Biological/reference and external-use claims remain limited | Explicit unknown provenance fields; verify prepared RLE against all 14 raw Parquet shards | source_scan/SOURCE_VERIFICATION_RUNTIME.json | UNRESOLVED_EXTERNAL |

Full classification/source/test/status matrix: ROOT_CAUSE_MATRIX.csv and REQUIREMENT_IMPLEMENTATION_TEST_MAP.csv. No unresolved ontology or holdout issue is called FIXED.

## III. Before/after evidence
H01: real field loader failed missing/shared epoch regression before patch. Shared epoch, worker-independent crop seed and fixed validation now replay correctly; persistent-worker test passed.
H02: old instance loader returned1 target for nested2-target fixture, silently losing tiny original support. Independent native RLE masks now retain both IDs, including one pixel; transformed overlap and source IDs preserved.
H03/H04:3 live CLIs now require a hash-bound actual runtime configuration before reading training/evaluation data or loading checkpoints;59 registered historical CLIs refuse direct execution. Legacy implementation functions remain importable, originals archived. Missing/config mismatch, stale/missing hashes, forbidden roles, unsafe output, unsupported recipe and incomplete critical spatial evidence fail closed.
H10–H13: native/accepted output lineage, inactive query handling, score-separated structural grouping, source-aware crop accounting/query overflow and short accumulation controls tested on synthetic fixtures.
Patch locations, minimal failing examples, logs and compatibility changes are in FIX_LOG.md. AUTHORISED edits are enumerated in AUTHORIZED_SOURCE_CHANGES.json; no metrics have been retroactively rewritten. New training streams/target retention differ from historical ones. Existing-output resume is blocked.

## IV. Verification evidence
Exact final command from workspace:
    ./cs2_venv/bin/python -B -m crown_segmentation_research.research_integrity.a06_hardening_v2.verify --output crown_segmentation_research/research_integrity/a06_hardening_v2/verification_final
Exit0, final test statusPASS, unique cases87, failed0, skipped0; test phase26.37s (not including full preservation hashing). Per-case timings/results and command/environment are in TEST_RESULTS.json; complete logs in verification_final/.
Environment capture: DEPENDENCIES.txt and TEST_RESULTS.json.environment. No pytest package available: established standalone tests executed directly, not reported as skipped pytest.
Source verification commands:
    ./cs2_venv/bin/python -B -m crown_segmentation_research.research_integrity.a06_hardening_v2.rescan
    ./cs2_venv/bin/python -B -m crown_segmentation_research.research_integrity.a06_hardening_v2.support_audit
    ./cs2_venv/bin/python -B -m crown_segmentation_research.research_integrity.a06_hardening_v2.affine_audit
These read-only source scans were executed into the versioned source_scan directory. Do not blindly rerun into already existing artifact paths; fresh scan destinations or a reviewed migration are required.
Native area-bin changes87 (train61/validation4/test22) logged without modifying native area or history. Original outside-valid records35; unique-empty67 and priority-empty14 are distinct. Two identical-mask groups preserved as uncertain semantics.
Same-grid deduplicated cross-split overlap14094/12966/11895 source pixels, predominantly border-localized; erosion1 sensitivity removes same-grid intersections, not spatial dependence. Cross-grid64 pairs include2 residual valid-ground overlaps. Do not describe these findings as extensive test contamination.
16 train-only data/target iterations across8 tiles/two epochs passed, no model fit; observed maximum97 visible targets, not full-corpus maximum. CPU constructors inspected existing F capacity; no model generalization result generated.
Actual preflight synthetic CLI PASS and confirmatory rejection are saved under verification_reports/. Synthetic fixture constants are not scientific defaults or approval for training.
CI workflow added at .github/workflows/treeflownet-integrity.yml; local corresponding tests executed, remote CI not executed. JSON Schema artifact supplied; jsonschema dependency unavailable, full JSON Schema validation NOT EXECUTED (no installation). Runtime native-source/provenance checks are tested separately.
23 original A06 source files were already absent before this task. Prior independently reviewed25-file snapshot and original reports preserved; no claim this task restored the missing files.

## V. Scientific implications
Historical AP39.03, oracleA52.50, oracleB77.86, small recall25.72%, score-only+1.89AP and grouping66.79→72.83 remain historical findings; none recomputed here. Oracle deltas are not additive causal decomposition. Improvements from a future query head cannot establish identity learning from AP alone.
Original annotation count is not an independent biological-tree count. Priority ownership is not unique support or original mask reference. First-wins field supervision can hide native objects; this is now logged, not silently declared biologically correct.
G-merge cannot split a mixed atom; this is not evidence against every relational decoder. Fragment-only16-query sanity is not a benchmark. Query/materialization/accepted detection remain separate.
No generalization, biological boundary accuracy, external independence, matched head capacity or full-scene stitching accuracy has been established by synthetic/infrastructure tests.
V2 E2E grouping tie policy is newly versioned (membership-hash tie), not historical metric reproduction; freeze scientific tie handling before FGQ and retain geometry/score-separated endpoints.

## VI. Readiness decisions
DATA_INTEGRITY_READY=true: available native data derivation, regression suite and historical preservation pass; known support ambiguities and provenance unknowns remain explicit. This does not certify upstream vectors, biological reference or independent sites.
EXPLORATORY_FGQ_READY=false: complete G/Q recipes, frozen supervision/crop quotas/query capacity/budgets, approved selection/tie rules and shared candidate/pair-cache reference remain outstanding. Active F/MR gated adapters do not constitute an executable F–G–Q benchmark.
CONFIRMATORY_BENCHMARK_READY=false: CONFIRMATORY_NOT_READY / EXTERNAL_HOLDOUT_REQUIRED. Need a genuinely appropriate independent evaluation registry and frozen endpoints/margins/model selection/calibration. Inspected tiles cannot be made an unseen lockbox by reshuffling.
Machine decisions and blockers: READINESS_REPORT.json.

## VII. Minimal next-step execution plan
Outstanding scientific choices and safe verification commands are in NEXT_EXPERIMENT_DECISION.md. First freeze a complete FGQ protocol within the approved research definition; bind its data/recipe/source hashes, crop plan, reference caches and explicit limits; run preflight. G/Q architecture implementation is outside this task's authorized scope. Register external holdout before confirmatory claims.
No full-training or benchmarking command is supplied while exploratory FGQ readiness is false.
For a fresh verification only:
    ./cs2_venv/bin/python -B -m crown_segmentation_research.research_integrity.a06_hardening_v2.verify --output crown_segmentation_research/research_integrity/a06_hardening_v2/verification_recheck_001
Use a new output directory each time. For the shipped synthetic-only fixture:
    ./cs2_venv/bin/python -B -m crown_segmentation_research.research_integrity.a06_hardening_v2.preflight --config crown_segmentation_research/research_integrity/a06_hardening_v2/configs/synthetic_smoke.json --mode exploratory


## Full fix log

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


## Machine readiness

{
  "version": "2.0.0",
  "date_UTC": "2026-10-15T13:30:27.229945+00:00",
  "DATA_INTEGRITY_READY": true,
  "EXPLORATORY_FGQ_READY": false,
  "CONFIRMATORY_BENCHMARK_READY": false,
  "data_integrity_scope": "Available native reference/data derivation and tested gated infrastructure; not biological truth, spatial independence or fully validated scene inference",
  "data_evidence": [
    "verification_final/TEST_RESULTS.json",
    "verification_final/PRESERVATION_CHECK.json",
    "source_scan/SOURCE_VERIFICATION_RUNTIME.json",
    "source_scan/AFFINE_VERIFICATION.json",
    "source_scan/SIX_SUPPORTS_VERIFICATION.json"
  ],
  "exploratory_blockers": [
    "Complete executable G/Q benchmark recipes not implemented or scientifically frozen",
    "Matched supervision, sampling quotas, target treatment, query capacity and budgets unresolved",
    "Frozen candidate/pair-cache reference and benchmark tie-selection policy require protocol registration",
    "Natural-scene stitching/GT global identity not certified if Stage3 claims requested"
  ],
  "confirmatory_status": "CONFIRMATORY_NOT_READY",
  "EXTERNAL_HOLDOUT_REQUIRED": true,
  "confirmatory_blockers": [
    "No appropriate independent external dataset and acquisition provenance registered",
    "Historical validation/test influenced development",
    "Primary endpoints/margins, calibration and model selection must be preregistered and frozen"
  ],
  "unknowns": [
    "Upstream polygons/rasterization and biological label identity",
    "Acquisition/site/license provenance",
    "Two duplicate-mask different-ID groups"
  ],
  "synthetic_gate_PASS_is_not_FGQ_readiness": true
}

## Full executed test inventory

- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_GPU_budget_control_without_GPU_execution
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_accumulation_flushes_partial_budget
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_buffered_calibration_excludes_close_neighbour
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_capacity_real_api_rejects_overflow
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_category_planner_reports_infeasible_strata
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_complete_partial_background_invalid_support
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_deterministic_sampler_and_impossible_valid_crop
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_epoch_determinism_validation_locked
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_full_crown_fit_limitation_explicit
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_inactive_Q_not_materialized
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_output_lineage_validity_and_duplicate_selection
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_sampling_overflow_cannot_silently_drop
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_scene_stitch_lineage_order_and_global_coordinates
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_source_edge_crown_not_certified_complete_or_dropped
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_spatial_overlap_buffer_and_unknown_crossgrid
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.CropOutputTests.test_transform_image_mask_alignment_and_resize
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_COCO_mask_ID_permutation_is_irrelevant
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_COCO_perfect_empty_and_score_ranking
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_E2E_ties_do_not_depend_on_renamed_instance_IDs
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_accuracy_coverage_curve_separates_abstention
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_anchor_cores_report_eligibility
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_area_strata_joint_matching_and_NA
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_boundary_metric_uses_ground_affine
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_boundary_small_border_tolerance
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_cardinality_beats_largest_sum_greedy
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_cardinality_matches_bruteforce
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_count_bias_valid_union_not_tile_sum
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_fixed_candidate_strata_and_coverage
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_fractional_indices_not_silently_rounded
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_grouping_NA_counts_and_coverage
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_grouping_perfect_abstain_all_merge_split
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_overlapping_GT_not_owner_raster
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_pair_cache_owner_and_core_validation
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_perfect_empty_duplicate_merge_split
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_physical_AP_bins_and_nonsquare_pixels
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_proposal_coverage_does_not_equate_merge_with_detection
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_score_only_structural_invariant_E2E_changes
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_split_merge_overlap_graph_counts
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_structural_reorder_and_no_score_cap
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.MetricTests.test_union_area_deduplicates_non_square_overlap
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_RLE_invalid_runs_dimensions
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_RLE_roundtrip_and_reencoding
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_affine_area_nonsquare_rotated_and_resize
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_area_bin_boundaries_and_invalid
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_bad_shape_and_values
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_exclusive_artifact_destination
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_hash_missing_empty_changed_and_traversal
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_metric_CRS_rejects_degree_and_feet_axes
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_roles_reject_test_selection_training_calibration
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_six_supports_and_original_preserved
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives.ReferenceTests.test_stable_owner_order_invariant_and_legacy_explicit
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_runtime.ProductionLoaderTests.test_field_validation_epoch_locked_and_workers_consistent
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_runtime.ProductionLoaderTests.test_instance_loader_keeps_nested_one_pixel_original_target
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_runtime.ProductionLoaderTests.test_native_RLE_crop_mask_and_area_match_decode
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_runtime.ProductionLoaderTests.test_persistent_worker_epoch_changes_real_crops
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_runtime.ProductionLoaderTests.test_shared_planned_window_drives_real_field_and_instance_loaders
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_active_entrypoints_require_gate_before_checkpoint_or_data
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_all_registered_entrypoint_gates_exist
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_calibration_test_label_rejected
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_empty_and_incomplete_configs
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_empty_missing_incomplete_and_bad_hash
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_existing_destination_rejected
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_historical_destination_rejected
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_incomplete_comparison_does_not_become_ready
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_legacy_oracle_and_test_scorer_blocked_before_run
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_malformed_config
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_missing_config
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_mode_mismatch_and_confirmatory_without_holdout
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_partial_waiver_is_invalid
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_positive_complete_synthetic_operation
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_query_capacity_rejected
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_raw_query_is_not_accepted_stage
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_real_preflight_CLI_positive_and_negative
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_role_test_training_rejected
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_runtime_tag_path_escape_rejected_without_model_load
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_threshold_on_test_forbidden
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_unknown_sampler_policy_rejected
- PASS: crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight.GateTests.test_unknown_spatial_source_is_not_PASS
- PASS: tests/test_clean_benchmark.py::test_v1_matches_original_targets
- PASS: tests/test_clean_benchmark.py::test_v2_has_no_interior_plateau_and_seam_only_between_crowns
- PASS: tests/test_clean_benchmark.py::test_coco_ap_matches_pycocotools_and_detections_stay_serialisable
- PASS: tests/test_clean_benchmark.py::test_resume_continues_exactly_and_weights_only_resume_is_flagged
- PASS: tests/test_clean_benchmark.py::test_target_margin_gives_full_crown_targets_and_zero_margin_is_unchanged
- PASS: tests/test_clean_benchmark.py::test_error_breakdown_outcomes
- PASS: tests/test_clean_benchmark.py::test_centre_map_rank_and_sdt_source
- PASS: tests/test_clean_benchmark.py::test_crown_weighted_focal_loss
- PASS: tests/test_clean_benchmark.py::test_modern_backbones_and_default_resnet_unchanged

## Verification scope and historical preservation

{
  "status": "PASS",
  "frozen_paths": 1038,
  "unchanged_registered_files": 975,
  "unauthorized_changes": [],
  "archived_source_files_verified": 68,
  "raw_Parquet_checked": 14,
  "raw_Parquet_mismatches": [],
  "prior_review_mismatches": [],
  "A06_snapshot_mismatches": [],
  "historical_data_labels_checkpoints_predictions_reports_preserved": true,
  "inherited_missing_A06_original_files": "23 source files already absent before this task; preserved snapshot retained, not restored over worktree"
}

## Unresolved assumptions

# Remaining uncertainties and decisions
- Upstream polygon vectors and pre-rasterization derivation are unavailable. Raw Parquet RLE/image to prepared data verified; upstream polygon-to-RLE consistency NOT_VERIFIABLE.
- Site, independent survey/acquisition, exact flight time, sensor and license are unknown. sblz1/2/3 do not establish three independent sites.
- Two identical-mask/different-ID groups retained: train image127 IDs44568/44569; image135 IDs47448/47449. Biological interpretation requires source-owner decision.
- Boundary overlap is small; spatial dependence and independent generalization remain unresolved. No universal 30m or one-pixel erosion rule adopted.
- Complete F/G/Q training recipes, supervision, quota distribution, capacity, budgets, cache baseline and selection rules need a frozen scientific protocol. G-merge and G-repartition must remain distinct.
- Same-grid stitching tested synthetically only. Natural-scene association, global GT identity, cross-grid stitching and tile-origin robustness unverified.
- Real GPU deterministic training and allocator limits not exercised; budget API tested without GPU execution. Allocator cap is not OS-wide isolation.
- New sampling/target semantics change future streams. Existing-directory resume is blocked; new epoch-state resume not certified.
- No independent holdout registered: EXTERNAL_HOLDOUT_REQUIRED. Historical test remains development-influenced.
- 23 original A06 source files were already absent before this task; byte-exact prior review snapshot preserved. No restoration over worktree performed.


## Contracts



# Native data contract V3 (implemented in A06 v2)
Each registered dataset binds annotation file, exact tile inventory, valid masks, actual affine/CRS, original source lineage and SHA256. Relative data aliases resolve through READ_ROOTS.json; data is a known symlink to /home/chung/RS/data. Unregistered external roots fail.
Check IDs/association, RLE run sums and dimensions, mask bounds, zero/derived-empty support, metadata area/bbox deviations and valid intersection without deleting records.
New original physical area = decoded native RLE foreground count × abs(a*e-b*d) for affine [a,b,c,d,e,f]. Require metre-based CRS. Crop/resize updates affine and preserves native area separately; valid/unique/priority areas have distinct names. Size bins are half-open [0,1), [1,2), [2,4), [4,9), [9,infinity).
Source scan:455 tiles,90462 annotation records,14 raw Parquet shards. Records are not necessarily unique biological trees. 87 bin changes are logged; historical metrics untouched. Native metadata retained. Informational metadata area/bbox differences are not automatically corrupt labels.


# Reference semantics V3
Original M_i; valid M_i∩V; unique U_i=M_i minus union of other masks; unique valid U_i∩V; priority-owner support; priority-owner∩V are separate arrays.
Default legacy_first_wins replays stored annotation order; priority pixel replay mismatch=0 across455 tiles. Stable-ID priority is a separate explicit alternative; area-ascending policy is experimental, never silently substituted.
All primary one-to-one instance metrics use original independent RLE masks, retaining overlaps. Field supervision retains legacy priority raster; field-invisible annotation IDs are logged instead of dropped from the reference.
Empty unique support train63/validation4/test0; empty priority support train10/validation4/test0. Original pixels outside valid support train29/validation6/test0 annotation records. These are support properties, not automatic annotation deletions.


# Spatial split policy V3
Versioned policy must declare support (full footprint or valid-ground), overlap tolerance, buffer in metres, CRS, spatial units, study scope and rationale. No automatically justified buffer is supplied. Confirmatory unknown support/provenance fails; exploratory exception needs scoped dated waiver with approver, expiry/review trigger and claim limitations. Warnings cannot authorize confirmatory-integrity violations.
Same-source pixel overlap differs from ground-footprint overlap and from byte-identical image content. Fresh scan reproduces deduplicated train–validation14094, train–test12966, validation–test11895 shared same-grid source pixels. Same-grid intersections disappear after1px erosion of both masks as a sensitivity result, not an independence repair. Cross-grid64 pairs include2 residual valid-ground overlaps after erosion.
OOF378 cross-fold pairs/370 GT-positive pairs are inspected historical audit evidence, not newly rerun performance. Buffered calibration excludes neighbours and refuses empty fitting folds.
TRAIN/DEVELOPMENT/CALIBRATION/CONFIRMATORY_TEST roles are explicit. Historical test content cannot become independent by renaming/copied paths. No original split changed.


# Output contract V3
Stage0: architecture-native fields, relations or query logits/objectness/no-object. Persist native source identifiers.
Stage1: materialized mask proposals using explicit architecture-specific rule; no artificial equivalence of raw fields and query slots. Inactive queries are not detections.
Stage2: accepted instances after frozen objectness, threshold, duplicate, invalid-support, minimum-area, overlap and maxDets policies. Native proposal lineage, unique scoped IDs, coordinates, masks/scores remain separable.
Stage3: scene instances with tile parents, affine/CRS/scene bounds and explicit associations. Implemented same-grid stitching supports deterministic overlap association; disjoint edge fragments are retained and flagged. This does not certify natural-scene segmentation or full cross-grid association.
Selection values must be declared, not inferred from fixture values. Structural grouping operates on its frozen stored mask set without score filtering at measurement. Stage and prior filtering are recorded.


# F/G/Q fairness V3
Controlled frozen-feature comparison and practical end-to-end system comparison are separate studies. Record encoder hash, modalities, feature access, crop pixels/physical FOV/GSD, sampling/augmentations, native supervision/loss, optimization and actual update budgets, all seeds, head depth/query count/trainable parameters, GPU/VRAM, selection/calibration, TTA, output stages and evaluator hash.
F uses geometric fields and reconstruction. G0/G-merge groups fixed atoms and cannot split already mixed basins. G1/G-repartition changes the atomic partition; not interchangeable. Q distinguishes query existence, mask geometry, no-object and accepted-set selection. No GT count at inference.
Shared encoder does not imply matched capacity, losses or optimization. CPU constructor inspection finds F trainable head1773189 parameters; Q257537 is a historical sanity report, not new matched-budget evidence.
C0/C1 uses one frozen field reference and stage, identical for all methods. Measure score-independent one-to-one recall, original-mask geometry, grouping/abstention and score-dependent AP. Spatial blocks/annotation IDs, not repeated seeds or anchor pairs, define dependence units.
Active gated recipes currently cover existing F and Mask R-CNN only. G1 and Q full benchmark recipes not implemented in this hardening task: incomplete FGQ configurations fail. EXPLORATORY_FGQ_READY=false.


# Metrics V3 / A06 v2 implementation
IoU uses original binary mask intersection/union. Cardinality-first diagnostic matching maximizes eligible pair count first, then total IoU; assignment bonus min(nGT,nPred)+1 dominates any sum-IoU gain. This differs from COCO's own score-ordered evaluator matching; do not label one as the other.
COCO AP is computed by installed pycocotools for synthetic regression; physical half-open area bins are explicit. No model-performance data was evaluated in this task.
Structural grouping: anchors covered by exactly one stored mask are assigned, zero/multiple cover abstain; score values and score cap are ignored. E2E grouping: frozen threshold/maxDets and highest-score membership. V2 tie rule is deterministic (-score, SHA256(anchor-membership row), stable ID), separately versioned from historical native-index ties. It is not a historical grouping reproduction or a preregistered scientific choice for the future benchmark; freeze the benchmark tie policy explicitly.
Same-tree correct requires two assigned equal instance labels; different-tree correct requires two assigned unequal labels. Report both unconditional correctness/all pairs and conditional correctness/assigned pairs, counts, coverage, abstention, overlap and balanced classes. Missing class/empty denominator returns null/NA, never fabricated perfect accuracy.
Cores r1/r4 require distance>radius inside original mask and unique valid support; adaptive radius=min(4,floor(.25*maximum eligible original distance)). Record eligibility and actual radius. Core distributions differ even with no excluded crowns. Pair cache validates original annotation ownership, integral indices and core support.
Candidate C0/C1 is geometric fixed-reference evidence, not biological detection. Proposal coverage and basin purity include explicit background denominator choices. Overlap diagnostics require declared edge criterion; split/merge graph counts are not causal error counts.
Boundary F1 specifies pixel or affine-metre tolerance. Count signed bias and MAE are separate; density requires deduplicated valid-ground union certificate and validated scene counts, not sum of overlapping tile areas.
Score-only synthetic controls show structural invariance and possible E2E/AP change. No independent anchor-pair bootstrap assumed; use paired tiles/spatial blocks and annotation identities in future inference.


# Calibration policy V3
Declare calibration event (e.g. IoU-threshold correctness), source-role, hashes, spatially buffered fold assignment, exclusion policy, fitting algorithm and scope. Independent test labels may never fit scores, thresholds, preprocessing, architecture or loss choices. Historical OOF tile folds are not automatically spatially independent.
Keep masks, scores, thresholds and fixed-threshold metrics separate. Any scorer output must bind its fitted-data provenance and original prediction geometry. Structural metrics ignore scores at measurement; E2E may respond to them.
preflight rejects forbidden roles, unregistered provenance, insufficient spatial fold policy and empty fitting folds. No calibration fitting was performed.


# Crop/target sampling V3
Eight requested strata: background, one complete, two-to-four complete, crowded, small, multilobed, touching, partial. Quotas and observable definitions are explicit config inputs. Planner reports infeasible strata, query overflow and full-crown DOES_NOT_FIT; never drops targets to satisfy capacity.
Complete means all native visible annotation support contained in crop and no ambiguous source-edge truncation; it does not establish biological completeness. Fragments retain source annotation identity. Record complete/partial/ignored/edge-ambiguous counts, visible targets, field-invisible IDs, valid fraction, physical size, affine/GSD, source, epoch, seed, window and geometry transform.
Common frozen plans are checked against native GT before runtime and drive both field and original-instance loaders. Image/mask transforms are synchronized. Train sampling uses shared actual epoch; validation is fixed. No silent fallback to invalid crops.
16 data-only train smoke iterations across8 tiles/two epochs passed; complete interior annotation masks were present. Observed maximum97 visible targets in1024px crops is not a global maximum. The old16-query fragment fit does not establish capacity for this FOV. This smoke does not certify a balanced eight-stratum empirical training distribution. Freeze scientific quotas and partial-target loss policy before FGQ benchmarks.


# Next decision
Do not start full FGQ or claim independent confirmation yet.
1. Freeze complete G/Q recipes and G-merge versus G-repartition scope, target/loss policy, crop quotas, query capacity, budgets, threshold tie policy, baseline candidate/pair cache and primary endpoints. Current architecture design is outside this infrastructure task.
2. Bind that approved recipe to native source hashes and sample plan; run preflight. Unsupported or incomplete configurations must continue to fail.
3. Register genuinely external, appropriately independent evaluation data and acquisition provenance for confirmation; freeze calibration and model-selection protocol. No new random split of inspected historical tiles creates an unseen lockbox.
Verification commands (from /home/chung/RS/Image Segmentation):
    ./cs2_venv/bin/python -B -m unittest crown_segmentation_research.research_integrity.a06_hardening_v2.test_primitives crown_segmentation_research.research_integrity.a06_hardening_v2.test_runtime crown_segmentation_research.research_integrity.a06_hardening_v2.test_preflight
    ./cs2_venv/bin/python -B -m crown_segmentation_research.research_integrity.a06_hardening_v2.verify --output crown_segmentation_research/research_integrity/a06_hardening_v2/verification_recheck_001
The verify output directory must be new; use another suffix if already present. No training command is authorized by a synthetic smoke PASS.


## Authorized implementation changes

{
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/bootstrap_gaps.py": {
    "before_sha256": "424ac2b7caea87db49d49a5cd2d88a637b6667407eef3bf137825e224693afe0",
    "after_sha256": "2a26b5a2c67f5c02b5c89c324869da932452783375185ae89a11def50fe1bba7",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/boundary_diagnostics.py": {
    "before_sha256": "2110d255a7aff383246713fc1c61da2228a3069b485c5d8063735b1d422c379f",
    "after_sha256": "e1fadef8c15d984de63dfc809d6a50156aaed3988b992f8b9c0c88efb55ee288",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/dataset.py": {
    "before_sha256": "bfc761cbc2a501d18d8886db7e68d83d48a0d1ac9fa8fe63734bc7381303f5ee",
    "after_sha256": "c7fd1e298fff8c1bcd775c5a391d2b66487d34b9366ae84ef8edfd04f359f4ee",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/diagnostics.py": {
    "before_sha256": "aee644a706526a10027d2a317f0bf3ecb30b0d5981677cb5ce306d9dc4cdbca6",
    "after_sha256": "c46c6b6fdae121a4f43e323dfefeeeee54d12046720d0e0cdbe4cb70586c9990",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/error_breakdown.py": {
    "before_sha256": "f63c4448c0d9359a93bd145ca440fc6ffba0b16cd4a6db4536d978d170af9304",
    "after_sha256": "af77ea44c84d7c252354812bfd6cb148cc107c8b56735db2d64a9c3a351cc3af",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/error_gallery.py": {
    "before_sha256": "2689f3a43b99a7184848c56038784dcf0ccfc90e75fed220a6639f0db98d339c",
    "after_sha256": "1d38efca657a29c9a6e17c218d7fc05f580f24343b023ec677fe73c76997b6d8",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/evaluate.py": {
    "before_sha256": "01f0812d58c783e78a55fdc2bf85a00bec5eb3c5e74599e5bb2499ab6103f562",
    "after_sha256": "1deb7e2054c69f8a49579c890ab9a2e9cc0825eb86e3d443b4aa60713f137bba",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp003_decode_variants.py": {
    "before_sha256": "54de1a4e441bfe62d66c864665fdc21256d78cdb9169c95b07062c6d5f77e8f2",
    "after_sha256": "cd5dca8ff507a11c06591995d0be06586dcae3ae17eb070443823613ff9dfbb5",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp003_summary.py": {
    "before_sha256": "98af764d717e777b75dff2449032959dea4a06cf349d84e3257ce6d781961944",
    "after_sha256": "5969856020cd0252d0bd970203c4e0c6e82a53cb2523f0446694ef80dfba68c9",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp003_waterfall.py": {
    "before_sha256": "73a6eb63b7a3e1831a488407142782fb8ccb56eab4afeb6a217caa599e79ac4a",
    "after_sha256": "bfc80fc8dcb10ad619d4fe77d31619cec31e41e291cbf73315f37caad4d2403e",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp007_gate_sweep.py": {
    "before_sha256": "4fe6d9f9b53a9f7eece8e3b77e3d94a58169d4678d5ca2873db5a116875f1e78",
    "after_sha256": "1ea1242ca355b6149fe772c207bd531e7536ef6030cc51f6b1821f91510a3555",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp008b_size_matched.py": {
    "before_sha256": "a0c088fd30c6db048865a259bd52791dfb9c562998888e4b4c2b3d3eccdb884a",
    "after_sha256": "00abe5f72f7d77e669beca0ad701031511112ea15fd96e181aa6b9ec5f563a7e",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp012_preview.py": {
    "before_sha256": "ce25a4c67e764e5fe8682adb5311fd0c991cc162c23755735f4add5152774cba",
    "after_sha256": "c2b010d2487869126dad849ef78684384c2f5e92bcbfb485a156a568e5acde3c",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp012_split_solidity.py": {
    "before_sha256": "3e88a608e4af0148b26eff53437e6c3644a34793852cf9fb19418290ad1952e7",
    "after_sha256": "3691cafc434bb7dbfb05f361c65bade8b8ac81c0dc7b7a541e3b5f81ec1b05a4",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp013_gap_decomposition.py": {
    "before_sha256": "0035f763b255bbef6fc1f017da34da16d5fa5e81536b3ccd74c660184c3fb02c",
    "after_sha256": "d3e2fc8a534a0d216bc36bf0cb2ebcfc760a7c5f5120207fefece40b673d2cf5",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp014_missed_lowscore.py": {
    "before_sha256": "7b489fc1fd1af0dcc0f78044c3fe11bbc0c1c49456e8a0c819e4dde2986b6226",
    "after_sha256": "1bc1254fae64dafba3517715388ac67acb6d03589d4421ae0d17bac11972598c",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp015_split_leak_check.py": {
    "before_sha256": "268026bda29d5c7234da4fd672a957af98e83953054901ec11d7bc5dda828544",
    "after_sha256": "6d9a21bd7abf9a7ff4fe58a6c1a71ff4464c6fcf831385719d14433f7d2d1841",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp016_preview.py": {
    "before_sha256": "340425eba1cab85d20d779c9c9add0cf25f1ef98a1ac1a8bb3b7d527fdade048",
    "after_sha256": "1eff1d539f83710fd626b41a52e04c484c17d31c03c222a0086b18ba3c72e0df",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp016_saddle_merge.py": {
    "before_sha256": "bf0b06e1c32366a42ac11d3e0f6bd6c1c4045e55add7abf4a6d134d7bf6fade1",
    "after_sha256": "c2dd3ba9e7bfd8fcd6b577907920a9ea670af5e338913e05d9353be564895a91",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp017_oracle_radius.py": {
    "before_sha256": "ac6824114749676d7fe51ff0124ae3cf1acf93be87645ac03fb7f3fdc732e479",
    "after_sha256": "79f60182f778dc7063a2d58d35a711c4dd888414131812a74cfce743c2e1e11f",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp018_basin_pairs.py": {
    "before_sha256": "4e6b6e56399df5833722c4c9c4186056fda90a0df243800e5b6a445b85380755",
    "after_sha256": "8265e54edccb9128ad5080dd1bade0b3b639025797383ca8b9c9dd52f9234c38",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp019_blind_visual.py": {
    "before_sha256": "3260d628378cc144b1741e0335dee3bfb99ab49b082da44081300c3fd1823390",
    "after_sha256": "8acec19cc22aaa1442befb2867ca3b64b368bf60ed983a4b154be2629951d932",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp020_pair_merge.py": {
    "before_sha256": "b08f18c693734e991984140a9b010330a867ae933fcbd2a58f46d8f6376c848b",
    "after_sha256": "93739d0dfabb9b2688768165165624683f3b1c1bae2da93e474af4cf581766d6",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp021_veto_merge.py": {
    "before_sha256": "25bd372fef595fb0e8e572a279cbdfc77fbc8b1200ec3b99adfcca46caf660fc",
    "after_sha256": "2d3677722a8843caa1f10481ea165dd44fcb3fd87a5a8023af4a05c96eaa9652",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp022_rescore_ceiling.py": {
    "before_sha256": "5e37665f22a1e36399d893ef10989f83af6da9aa3abaecdc5dde28076e6e0a1f",
    "after_sha256": "31ffd59f836082f9019c610bcd6037a4203c483ee7e0ec9010be490d36e60929",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp023_instance_rescore.py": {
    "before_sha256": "fd94ecdc82650bd732debe423409d6c7209a218933045ff401c23eda6f095150",
    "after_sha256": "022ac933f729fe8a712c4c63d2c31604a06ca9d2a0dbcd3bde7404675e848c5c",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp024_fair_rescore.py": {
    "before_sha256": "09df65948e2f946e7cb7e00a558c8f045098df72e5fe15c089d03bc83df71716",
    "after_sha256": "e75444e68b4ee7657fe26c17bf8aafcca04bb0677e9d946d7ff37bf5102a4673",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp025_gate_rescore.py": {
    "before_sha256": "213f6cf558e57ed51815e04438193b74afb874d754383086fb566bdc5ea8b5d3",
    "after_sha256": "339fc53cfc063a95e5b9cdb4b7f7676e0fd206078bf17954b7cbd86ea6bd094b",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp026_target_ceiling.py": {
    "before_sha256": "f55943ffd26eda3e1e496bbf08587e413e66201b96bb5ed579faead5b3f70f8f",
    "after_sha256": "a0e9c2588951bd8f543dab5852777b173829af696b64b908cc21e185c107b0ea",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/exp027_omnipose_decoder.py": {
    "before_sha256": "e6bd7b9c40e39841c127772c1e4a0b7c201b1c6f1b46ce2b280d18593374f748",
    "after_sha256": "da8fb24c8835953012afeb51fff2a78b8ace90411613ce82db1c70f1107de322",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/make_figures.py": {
    "before_sha256": "ebc8a378d22905c2713de6f3d411f1162b08eed6fdf17b1400b249c910744782",
    "after_sha256": "7ec09759ced9215723b896f9f55cdccb29bf5aef2b459b8bf2c6bcf2bc5fe89e",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/make_relabel_kit.py": {
    "before_sha256": "3cbc6aa9964cbc1bce9e1df371dc7efe0036d7e7e96a9d75e4556375f030c2b1",
    "after_sha256": "02d898d11d7ac68a18e03ed0cd674d4d60115cf546db9f83c4b2db8f8d279384",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/master_table.py": {
    "before_sha256": "1bdf33df76e59147d90c54b80ce0d3659772363564e3edb13c31083dbd0374bf",
    "after_sha256": "220d5ace1ccc88a063b566a13cc906112308a8b10adf4c334a9056d36149c22c",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/oracle_seed_nms.py": {
    "before_sha256": "6013159a5ee3be050894d8bdddd0bacf8ba0d9df872f6890d56529bd717ac3b6",
    "after_sha256": "8d8d3b9ccb6d0b287ef11c8746db5335b921d95bc2c8447db935236ed7ec0407",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/prepare_canopyrs.py": {
    "before_sha256": "3b74bdb9688b90f6ecd91fd3045a4fca56b7dc7fcbfcec40b02edde6741b4853",
    "after_sha256": "feba382919149de212d5e834bd5296ec785ca6668fd73ee20def0825694fcfd2",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/render_preview.py": {
    "before_sha256": "fd9ba16b9ae4d6d0388f978efab3c560743cae3c9ebbdb6094a582592e6b3d03",
    "after_sha256": "2fb0537529078102215cc42ecf03cdbb13724cddc2bc4bb354b70d8493e9eea1",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/render_tables_md.py": {
    "before_sha256": "094ded97185b4b09006041edd98294a30c5451c65fcd9ef92f6e4f63e3768f90",
    "after_sha256": "4cfa479cc629032dc562ce69e1efb20798958d15b6c481fb69c02a8fb4b2f8e7",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/seed_scorer.py": {
    "before_sha256": "3904d3ee139d4335c1e3c34db1c3661ef4fb6d1320970ae9db6fb1c280e960d1",
    "after_sha256": "03af8e2a19ff9302b30b2d82e3cf524c8d69fa7124c856ce985becf221d564bb",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/seed_scorer_test.py": {
    "before_sha256": "c83208667c120838dd61223a112e7aed79024ebf4ff70e8227a53f96d22133ef",
    "after_sha256": "de64988a504a72b906246af3b0eaade1137b0e11b2d88998c106f4f6986d0a0b",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/select_on_val.py": {
    "before_sha256": "84ad45bff1e959b04f7c599fc7697b584f354f4ea3e329a1f5244ea3bbd80e70",
    "after_sha256": "7f014c392f136633eac76619ae9ead05c85db50b2f39442026b89c08af92808f",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/size_extent.py": {
    "before_sha256": "0c705a3761cc8f4e077777c06e25dcf356b89eb2ba37751a203f908e542ae8dc",
    "after_sha256": "93b27f1c0fef9fc168362e432be01e32360e224f4177359913992ddcd6e8c674",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/size_lost_cause.py": {
    "before_sha256": "4fca7f78e683f50037393cf462d7dd26558e061d78921f096bf701b818ceb403",
    "after_sha256": "b615baec0d15d9800dffa9e84ded006515471249d07e89cae41d591532952ba7",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/size_recall.py": {
    "before_sha256": "c748638437a659caf596fa5ec0ceca3a54b99d93f3c02d13dd4214c6bc569927",
    "after_sha256": "4c7c1052b34babf45384786d691b5922d420abed7c3b99db27767dfe308eaa7f",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/stratify_s1s6.py": {
    "before_sha256": "f74509c8824e3ed1900492ad4f88575f3435bef265b3efde72360b0e4c1c052e",
    "after_sha256": "25bb79957bab91f9287e548c12a5347b31d049072c1ce3af103a25412de78c1a",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/synthetic_exp005.py": {
    "before_sha256": "7a254f9e8ed61eb17a470a72b7d0935868c55c6bca6312b800d4f10e10fcc0d4",
    "after_sha256": "67fdf33d316d954e590baf7fbe4564b13426bc34359cba39baba0cceee191582",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/synthetic_maskrcnn.py": {
    "before_sha256": "a5e789dec1d834525e459c8709cc6d010404c91a0da4069bde32ca436ed781c8",
    "after_sha256": "e74604f7555b0d1c4cd07bffd201c40175d409fb21dfbb119e6740cba49fb17f",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/synthetic_world.py": {
    "before_sha256": "f25cd459920692f919ba569d787483d156bd15697f94d30d4a7829ee7506f417",
    "after_sha256": "6168c941b39f7d02e7986ab70dc75969f93f3c048177a14d4e462f4f8852838e",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/train_maskrcnn.py": {
    "before_sha256": "9b94a9e2f3a44fb90ed0694324ddf965ff4d2762ae543b798bd3c724897f3a97",
    "after_sha256": "88333494d17e25ad200c44552975acd0a9cad85b8611a53bc0f0e5db3d45da92",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/clean_benchmark/train_treeflow.py": {
    "before_sha256": "c929a6fe0807e7cbc0505f5bc7a631af1f56b96320cfa42797aa0b1bd4407fd1",
    "after_sha256": "0500d234cbe7843dcb684308c8dfb36556a39e516e055fe83f2d589ce415e789",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/diagnosis_2026-10-11/fgq_revision_2026-10-13/grouping.py": {
    "before_sha256": "fabaae1c25531659c00dcb117626eb526510ea2678ed30a409834edfeccb95f7",
    "after_sha256": "cc901bce6a960f6977892b7f42d971038e56828363daa34d2144e60f98de5a8e",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/diagnosis_2026-10-11/fgq_revision_2026-10-13/q_sanity.py": {
    "before_sha256": "cfcad72411572615296f1da86de375e873642550ae7a62ce41ff7e333a69fd77",
    "after_sha256": "2bfb5b42389bcf7371ae24afd28b016cc141b75de074f2946f371648feaaa0fd",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/diagnosis_2026-10-11/study.py": {
    "before_sha256": "272be1edc43604b92a4483a059042f1770fd24b30343bdf4e96b7c1dee7b6149",
    "after_sha256": "a67401b3a2afbaf190385097e1a0b9f83702a512900e6d89c844fa2481785480",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/benchmark_all_paradigms.py": {
    "before_sha256": "e7875727f4898caadbf06d43f741d00149c17e0b903990f6be8b0e8b855414c1",
    "after_sha256": "149922c9860edf80de9994267a47a19e446f2a314b37003d2686ad98c0970c4f",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/benchmark_boundary_refinement.py": {
    "before_sha256": "7334ced689ef0577485e761c75252d0fa0b33d3c72c9fa1fe9c360c95dba1020",
    "after_sha256": "92647ffac33ee2163304967e365fe436c7cec1501309a4d8acebe611cefd757e",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/canopy_watershed/train.py": {
    "before_sha256": "e611be1ab0cf431aa533ad1b4bfd4822b511e856780deff322046059434f8d0c",
    "after_sha256": "ff2e8b8ddd18ddc8fb690b9dcea41a3839328b8a8d030dba317ab374f031cba3",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/canopy_watershed/train_apex.py": {
    "before_sha256": "ac074b859df368f00255976191e8a440085929f884f096876536da71d1204c77",
    "after_sha256": "c85360ac5c02d5786bce5392a0bc8627605804bab7da47880cbc4b9f7e55612f",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/canopy_watershed/train_calibrated.py": {
    "before_sha256": "f0b4ab4f6b3628d3d8255a421e1cbf0dd9cd7a325f2b4e9c46b753b63fcf9d66",
    "after_sha256": "2edfe43f96dcb38eee30e3f02f4c6c941d0bee681f132a747d781cd5db8066c5",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/canopy_watershed/train_sharp.py": {
    "before_sha256": "a25bb631e643719896da5d04920782efd4a24aa65c1f81d44ebc00904e033979",
    "after_sha256": "e22363e6bcf67fcb36d0bcd2819724138edd0fd333b1d3c4601cf0c57a0585c8",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/canopy_watershed/train_wide.py": {
    "before_sha256": "40009365fc79fb19397447add26f3321fde0e0588ea20bbc65840ed9581d53b8",
    "after_sha256": "01f6e2bb04a4911f09984df626161b166e4c4d549aff64d805a43cee474c1208",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/tree_flow/train.py": {
    "before_sha256": "42ed26bc7ab2acb8c4b95723100cbbe1757b284830dc2a9488849a539c6173a4",
    "after_sha256": "82528a7d2b7d618b2be56ff0f66b5fede1bd4ed7ed47ecaa20152719f1e5f20e",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/tree_flow/train_hr.py": {
    "before_sha256": "52497396f6c82a9b4cafef3552686afb1b3714e22674a83ab797f0e366fd8376",
    "after_sha256": "4cbe72112ef2579aca5963cb819db2d2654a2891003565d43bf3758390d46d75",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/tree_flow/train_v2.py": {
    "before_sha256": "4e356486fb660231af6f5c16fb4fb9530fd79f640c42d0250ae58729e5696632",
    "after_sha256": "45e1f7509d40e06697825f3c4e499b242a66447aeb17120bf2ab09f0c80a4e5a",
    "scope": "authorized implementation repair; original bytes archived"
  },
  "/home/chung/RS/Image Segmentation/crown_segmentation_research/methods/tree_flow/train_v3.py": {
    "before_sha256": "c10d0c9f16b2c0250a6064228a59b24fb332173c21389932f55941500bc83f70",
    "after_sha256": "c2aa00c9f01db510dd3095038305070b50f16afb20127c232f0a6509687b98f7",
    "scope": "authorized implementation repair; original bytes archived"
  }
}
