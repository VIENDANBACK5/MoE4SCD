# Independent validation of A06 hardening

User asks to check/validate another agent's delivered A06 implementation. Preserve that implementation and its reported outputs as review evidence. No full training, oracle rerun, or historical test-performance evaluation.

- [x] Freeze A06 artifacts and record Git/current historical-source integrity.
- [x] Inspect code/contracts and map executable controls to requirements.
- [x] Run original tests in isolation and adversarial preflight/metric/RNG probes without overwriting A06 reports.
- [x] Reconcile reported claims, readiness, environment and actual behavior; write findings with source lines and reproductions.
- [x] Verify reviewed snapshot unchanged and deliver independent verdict with explicit limitations.

Scope is independent validation, not silently repairing the evidence being reviewed. Scientific choices (ownership, buffer, thresholds, G search space) must not be changed by this review.

External change: after initial checks, 23 A06 source files disappeared outside review operations. All 25 original A06 file contents recovered from the byte-matching sandbox copy into reviewed_a06_snapshot; original source directory was not restored or modified. See external_source_drift.json. Replayed full validation against snapshot successfully; source mutation cause remains unknown pending user response.

Verdict: FAIL; all three readiness booleans false. 24 original tests pass but two are empty; six of eight adversarial CLI cases are accepted incorrectly. 17 findings recorded. Direct compressed-RLE metadata scan covers 90,462 records; no new model-performance result. All 1,026 previously frozen historical files match. Self-contained report includes JSON evidence and review runner source.
