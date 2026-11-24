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
