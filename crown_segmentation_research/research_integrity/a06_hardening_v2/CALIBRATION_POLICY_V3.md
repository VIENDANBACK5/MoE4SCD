# Calibration policy V3
Declare calibration event (e.g. IoU-threshold correctness), source-role, hashes, spatially buffered fold assignment, exclusion policy, fitting algorithm and scope. Independent test labels may never fit scores, thresholds, preprocessing, architecture or loss choices. Historical OOF tile folds are not automatically spatially independent.
Keep masks, scores, thresholds and fixed-threshold metrics separate. Any scorer output must bind its fitted-data provenance and original prediction geometry. Structural metrics ignore scores at measurement; E2E may respond to them.
preflight rejects forbidden roles, unregistered provenance, insufficient spatial fold policy and empty fitting folds. No calibration fitting was performed.
