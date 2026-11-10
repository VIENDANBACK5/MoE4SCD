# M1: information loss versus raster-encoding sensitivity

Status: RESEARCH_QUESTION_REFINED; ORIGINAL_MECHANISM_NOT_ESTABLISHED.

The user restored the original-method goal to ACTIVE. The empirical manuscript remains an evidence archive, not the main objective or a substitute for discovering a defensible method. This cycle investigates M1 at research-question level. No new model, loss implementation, dataset, checkpoint, inference run, training run, annotator response or TEST access has been created.

The scientifically testable first question is narrower than "observability-conditioned formation":

> Does a frozen strong carrier lose previously correct, distinct reference matches under a losslessly invertible raster re-encoding of the same RGB observation, with identical physical support and output-coordinate policy?

This can distinguish information deletion from dependence on the input encoding. It cannot alone prove biological identity, a general architectural bottleneck, insufficient training, or novelty. Known scale-equivariance literature already predicts forms of scale sensitivity; a positive diagnostic is an entry to causal analysis, not a method contribution.

## Findings of this research gate

1. DERIVED: fixed pixel field of view and area floor do not define fixed physical policies. This is a verified dimensional relation, not a measured cause of BAM's result.
2. DERIVED: exact physical-scale rules cannot in general be identified from RGB without calibration or an identifiable size prior. GSD is metadata, not an additional spectral modality, but adding it would be a different deployment contract from f(I).
3. DERIVED: replication upsampling is injective and has a known left inverse. It can provide an information-preserving control; downsampling cannot be treated as invertible.
4. DERIVED: the sketched partial-matching consistency objective needs explicit admissible correspondences, unmatched penalties, gate supervision and a segmentation anchor. Otherwise abstention, gate collapse or count collapse can minimize it.
5. VERIFIED PRIOR ART: ScaleMatch, DINO-MS, v-CLR and Deep Scale-spaces overlap with major parts of the proposed direction. In particular DINO-MS already avoids its lowest-resolution consistency views, and Deep Scale-spaces explicitly models irreversible transformations using semigroups. Neither generic partial/scale consistency nor acknowledging information loss establishes novelty.
6. CODE-OBSERVED: the archived DINO logits path normalizes/pads without an input resize; its source permits variable image shapes, although numerical execution at proposed shapes is NOT_RUN. The archived R5 uses min_size=max_size=1024; its native preprocessing would erase the intended input-size contrast. Do not silently bypass it or claim the same manipulation reaches both backbones.

## Decision

Keep M1 as a conditional, narrowly specified research question. Do not implement the proposed consistency loss, an observability head, a metric-scale architecture or a new training program from this review. The first eligible next action would be the finite frozen diagnostic in 03_DISCRIMINATING_TEST_NOT_RUN.md after its geometry/materialization details and resource gate are locked. This document does not claim that diagnostic has run.

An independent-review dependency remains for claims about visually resolved biological/reference identity. It does not prevent the information-equivalence reasoning above. No three-reviewer participation is fabricated, and no fresh blocked status is inferred from the pre-resume history.

Evidence and mathematics: 01_MATHEMATICAL_CHECK.md.
Nearest mechanisms and retrieval limits: 02_NEAREST_PRIOR_ART.md.
