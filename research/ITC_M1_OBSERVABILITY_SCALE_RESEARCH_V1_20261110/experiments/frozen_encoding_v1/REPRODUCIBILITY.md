# Reproduction

The exact protocol/cohort/inference implementation were committed as b5dc597 before execution, and post-freeze scoring as d0a6846 before reference measurements. Completed outputs are guarded against overwriting. For an independent rerun, copy the isolated experiment's source/protocol/bindings to a new directory at the same repository-relative depth and use new output directories. Do not remove the recorded freeze/results or modify historical checkpoints.

Commands executed from repository root:

    cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/frozen_encoding_v1/code/predict.py
    cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/frozen_encoding_v1/code/analyze.py
    cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/frozen_encoding_v1/code/verify_and_figure.py
    cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/frozen_encoding_v1/code/report.py

The second command's initial console serialization error is retained and explained in ENGINEERING_NOTE.md; all metric files had already been written. No inference/scoring rerun was performed to rescue the result.

INHERITED_CACHE_BINDINGS.json gives checkpoint and RGB/validity/reference-cache hashes. COHORT.json binds the exact3eligible TRAIN windows. PREDICTIONS_FROZEN.json gives all12prediction hashes and the normalized-backbone left-inverse assertions. The256-dimensional official query/pixel head remains unchanged,300query slots per arm. Original source files were read only.

Read input NPZs to reconstruct the exact float32D2means and U2replications; no quantization of the coarse RGB. Each prediction contains all300canonical independent RLEs, class logits/probabilities, mask-quality/output scores, normalized boxes, query-to-selected-site trace, native mask-grid sizes, area/filter/NMS lineage and timings. GT is absent from inference. Dense mask logits are explicitly not archived, per the compact-study storage adaptation; they are reproducible from the exact frozen source but would require a new accounted forward run.

Final results reuse the sealed evaluator helpers from the completed BAM study without changing their functions; only all-area metrics are taken. Custom size results use inherited published-tile raw annotation area, not crop fragment area. No biological or independent geographic claim. Source-truncated flags and all298reference IDs remain in the tables. Source/seed/window repetitions are not independent forests.

VERIFICATION.json recomputes raw IoUs, maximum matching and GT-free NMS from saved RLEs, checks alpha/support, and verifies the original checkpoint. This verifier does not certify model novelty, biological correctness, or generalization.
