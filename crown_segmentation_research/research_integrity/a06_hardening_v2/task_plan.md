# TreeFlowNet preventive hardening v2

## Goal
Implement and verify preventive infrastructure without training, oracle reruns, model-performance evaluation on historical test, or changes to source labels/history.

## Phases
- [x] A. Discover current sources, freeze hashes/Git state, map A00–A05 and failed A06 evidence.
- [x] B. Implement reference/provenance/area/spatial/access controls with failing-first regression tests.
- [x] C. Implement crop accounting/RNG/output/metric contracts and integration tests.
- [x] D. Implement fail-closed preflight and connect active training/evaluation/calibration entrypoints.
- [x] E. Run source integrity scans, synthetic suites, negative/positive CLI smoke checks and preservation checks.
- [x] F. Generate evidence-based reports, requirement matrix and readiness decisions; retain external/scientific blockers.

## Decisions
- Versioned working area: research_integrity/a06_hardening_v2. Do not overwrite failed A06 or independent validation.
- Existing legacy original first-wins ownership is replayable; original overlapping RLE remains instance reference.
- New area uses RLE count times affine determinant. No automatic 30m buffer, confidence threshold, min area or G repartition policy.
- Fix production implementation only with before-source archive and regression; frozen historical artifact hashes distinguish authorized source edits.
- No new model. Missing Q/G benchmark recipes remain explicit blockers rather than invented CLI.

## Errors
- pytest is absent in the environment; preserve failed command log RELATED_TESTS_INITIAL.log. Run the established standalone test runner instead; 9 tests passed without installing dependencies.

## Status
Completed infrastructure scope: 455 tiles/90,462 records and14 raw shards checked; 87 unique tests PASS (78+9); preservation PASS across1,038 registered paths,975 unchanged and63 authorized source edits archived (68 original source entries). Three active CLI gates and59 archival CLI blocks. Synthetic preflight PASS; missing-independent-holdout confirmatory preflight FAIL as required. Level1 ready within native-data scope; full FGQ exploratory and confirmatory readiness false with documented design/external blockers. No prohibited experiment executed.
