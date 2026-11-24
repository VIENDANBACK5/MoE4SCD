# Output contract V3
Stage0: architecture-native fields, relations or query logits/objectness/no-object. Persist native source identifiers.
Stage1: materialized mask proposals using explicit architecture-specific rule; no artificial equivalence of raw fields and query slots. Inactive queries are not detections.
Stage2: accepted instances after frozen objectness, threshold, duplicate, invalid-support, minimum-area, overlap and maxDets policies. Native proposal lineage, unique scoped IDs, coordinates, masks/scores remain separable.
Stage3: scene instances with tile parents, affine/CRS/scene bounds and explicit associations. Implemented same-grid stitching supports deterministic overlap association; disjoint edge fragments are retained and flagged. This does not certify natural-scene segmentation or full cross-grid association.
Selection values must be declared, not inferred from fixture values. Structural grouping operates on its frozen stored mask set without score filtering at measurement. Stage and prior filtering are recorded.
