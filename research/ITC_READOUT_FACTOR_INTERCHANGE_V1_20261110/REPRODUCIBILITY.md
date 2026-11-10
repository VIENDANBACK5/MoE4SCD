# Reproducibility

Use the existing repository root and `cs2_venv/bin/python`. Exact commands and bound assets are in REPORT.md, BINDINGS.json and SOURCE_BINDINGS.json. Protocol commit6c1d8a9 precedes all12newforwards; exploratory same-query note/code commitdaa9c3e precedes its supplementary measurement after the primary result was known. No scientific thresholds changed.

Executable stages: code/predict.py → code/analyze.py → code/verify.py → code/same_query.py → code/report.py. Prediction and primary analysis refuse to overwrite completed results. A fresh sibling copy preserving directory depth can reproduce the run, with access to the existing M1pilot/control checkpoints, cached inputs and evaluator. Verifier is immediately rerunnable on retained outputs.

Both completed S0/S1checkpoint bindings are unchanged; no new checkpoint or optimizer is claimed because training did not occur. All24prediction gzipfiles preserve7200independent raw RLEs, query/site/class/box provenance and both score policies. Final E matrices retained in embeddings/. Dense F/logits NOT_STORED, explicitly preregistered; input/checkpoint/code/latent hashes permit reconstruction. Published-reference files are opened only by post-freeze analysis.

Same-count uses per-window lower prediction count and score order, notGT. Original-score counterfactual uses each factual source query's composite score. Raw reference availability, one-to-one feasible matching, accepted/final masks and score-rankedAP are distinct. Figures use raw-best witnesses chosen withGTonly after freezing, according to a fixed first-lexical strong residual per source rule.

DEPENDENCIES.json records installed packages/platform. GPU shared, not reserved. Current environment has no graphical display; HTML/PDF were generated and the mask figure was visually inspected through the image tool, not falsely claimed opened in a desktop browser. No /loop/OpenClawcron tool exists; nativegoal API cannot resume the blocked registry entry. Direct user-requested execution, source control and persistent state provide this turn's continuity; no unattended20-minute scheduler is claimed.
