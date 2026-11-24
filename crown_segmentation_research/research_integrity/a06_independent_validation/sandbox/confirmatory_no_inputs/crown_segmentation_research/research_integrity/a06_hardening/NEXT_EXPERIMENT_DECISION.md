# TreeFlowNet Next Experiment Decision & Execution Protocol

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Reviewer:** Senior Research Software Engineer, Computer Vision Researcher, Scientific Auditor  
**Decision Gate:** A06 Hardening Pre-Flight Review

---

## 1. Readiness Verdicts

| Gate Level | Status | Verdict Summary | Key Justification |
|---|---|---|---|
| **LEVEL 1 — DATA INTEGRITY** | **READY (PASS)** | `DATA_INTEGRITY_READY = true` | All 455 tiles, 90,462 annotations, support definitions, canonical area formulas, and baseline hashes verified without data corruption. |
| **LEVEL 2 — EXPLORATORY F–G–Q** | **READY (PASS)** | `EXPLORATORY_FGQ_READY = true` | Output Contract V3, full-crown sampling policy, epoch RNG fix, and 24-test regression suite verified. Safe for controlled exploratory prototyping. |
| **LEVEL 3 — CONFIRMATORY BENCHMARK** | **NOT READY (BLOCKED)** | `CONFIRMATORY_BENCHMARK_READY = false` | Contiguous Z1 tile splitting (38k boundary pixels, 235 connected tiles) and prior inspection of test tiles permanently prevent confirmatory claims on Z1. |

### Required Action for Level 3:
$$\text{EXTERNAL\_HOLDOUT\_REQUIRED = TRUE}$$
A genuinely independent, uninspected aerial orthomosaic acquisition from a distinct geographic forest site must be registered before authorizing confirmatory evaluation.

---

## 2. Permitted Exploratory Next Steps

With Level 1 and Level 2 readiness confirmed, the team is authorized to proceed with **exploratory F–G–Q prototyping** under Study Type 1 (Controlled Mechanism Comparison) and Study Type 2 (Exploratory System Tuning).

### Safe Exploratory Commands

#### Step 1: Preflight Verification (Always Run First)
```bash
./cs2_venv/bin/python -m crown_segmentation_research.research_integrity.a06_hardening.preflight --mode exploratory
```
*Expected Result:* Exits with code 0 (`PASS`), confirming environment, baseline hashes, and test suite.

#### Step 2: Full-Crown Q Head Prototyping (Train-Only Sanity on 1024x1024 Crops)
To validate query-based complete crown assembly (resolving the fragment-only limitation of historical Q sanity):
```bash
./cs2_venv/bin/python crown_segmentation_research/clean_benchmark/train_treeflow.py \
    --config crown_segmentation_research/clean_benchmark/configs/q_full_crown_exploratory.yaml \
    --crop-size 1024 \
    --target-margin 64 \
    --query-capacity 64 \
    --epochs 5 \
    --device cuda:0
```

#### Step 3: G1 Cut-and-Merge Formulation Prototype
To validate relational repartitioning beyond merge-only G0:
```bash
./cs2_venv/bin/python crown_segmentation_research/clean_benchmark/exp021_veto_merge.py \
    --mode exploratory_repartition \
    --seed 42
```

---

## 3. Strict Experimental Prohibitions

1. **NO Evaluation on Historical Test Set**: The 168 test tiles in `data/itc_benchmarks/canopyrs_quebectrees/prepared/test/` MUST NOT be evaluated to claim model superiority or publication-ready generalization.
2. **NO Unbuffered OOF Calibration**: Do not fit confidence calibration or temperature scaling using random or unbuffered tile cross-validation folds.
3. **NO Conflation of Raw Query Slots with Accepted Instances**: All Q evaluations must apply Stage 2 objectness filtering ($\tau_{obj} \ge 0.5$) before computing instance metrics.
4. **NO Silent Erasure of Nested Crowns**: Never evaluate primary instance matching against single-label priority ownership rasters.
