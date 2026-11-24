# Task Plan: TreeFlowNet — Data Integrity, Experimental Fairness & Reproducibility Hardening (A06)

## Goal
Establish a hardened, traceable, fail-closed research infrastructure and verification suite for TreeFlowNet before F-G-Q experiments, preserving all historical evidence, fixing confirmed defects with regression protection, and explicitly gating exploratory vs confirmatory readiness.

## Phases
- [x] Phase 1: Planning, Setup & Baseline Protection (Establish task plan, notes, and SHA256 integrity baseline)
- [x] Phase 2: Scientific Audit & Root Cause Analysis (Document audit findings, root cause matrix, evidence levels, and unresolved assumptions)
- [x] Phase 3: Data Integrity & Reference Semantics Hardening (DATA_CONTRACT_V3, REFERENCE_SEMANTICS_V3, SPATIAL_SPLIT_POLICY_V3, PROVENANCE_REGISTRY_SCHEMA)
- [x] Phase 4: Experimental Fairness & Comparison Contracts (OUTPUT_CONTRACT_V3, FGQ_FAIR_COMPARISON_V3, METRIC_DEFINITIONS_V3, CALIBRATION_POLICY_V3, CROP_SAMPLING_POLICY_V3)
- [x] Phase 5: Executable Implementation & Regression Test Suite (Executable preflight tool, comprehensive unit/regression/synthetic test fixtures)
- [x] Phase 6: Verification Execution & Final Reports (Run tests, generate TEST_RESULTS.json, READINESS_REPORT.json, HARDENING_REPORT.md, FIX_LOG.md, NEXT_EXPERIMENT_DECISION.md)

## Key Questions
1. How to strictly enforce separation of support definitions (original, valid, unique original, unique valid, priority owner, priority owner valid)? -> Solved in REFERENCE_SEMANTICS_V3.md and verified in test_suite.py.
2. How to ensure fail-closed spatial split validation with configurable buffers so no boundary leakage can slip into confirmatory splits? -> Solved in preflight.py with fail-closed rejection on confirmatory mode.
3. How to formalize the 4-stage F-G-Q output contract so raw Q slots cannot be conflated with accepted instances? -> Solved in OUTPUT_CONTRACT_V3.md, filtering inactive Q slots at Stage 1/2.
4. How to fix the training crop sampler RNG and full-crown target handling so multi-epoch training doesn't replicate identical crops or drop targets? -> Solved via multi-epoch deterministic hash seeding in CROP_SAMPLING_POLICY_V3.md.
5. What are the explicit blockers preventing Level 3 (Confirmatory) readiness, and why must Level 2 (Exploratory) be kept strictly bounded? -> Solved via Blockers B01, B02, B03, B05 (connected split, prior test inspection, unverified site provenance).

## Decisions Made
- [Phase 1]: Use dedicated versioned directory `crown_segmentation_research/research_integrity/a06_hardening/` to avoid mutating any historical files or existing audit artifacts.
- [Phase 1]: Historical artifacts (e.g., in `execution_plan_audit_2026-10-15/`, `clean_benchmark/`, etc.) remain strictly read-only.
- [Phase 1]: Preflight executable module is runnable directly with `python -m crown_segmentation_research.research_integrity.a06_hardening.preflight`.

## Errors Encountered
- None. All 24 unit/regression tests passed. Baseline SHA256 integrity checks passed.

## Status
**Completed** - All 6 phases finished. Complete hardened deliverables package established in `crown_segmentation_research/research_integrity/a06_hardening/`.
