# Locked finite M1 frozen encoding pilot

Confirmatory only for the four registered contrasts on these fixed exposed TRAIN windows; exploratory for any broader ITC mechanism. No carrier training or new architecture. This execution protocol follows the research-only gate without rewriting its NOT_RUN history.

## Cohort / carrier

Reuse exactly the three already selected TRAIN windows from the completed readout diagnostic (one per SelvaMask source;298original reference occurrences). Selection was metadata-based in that earlier study; these images and aggregate native outcomes are already exposed. No new selection by scale response. Their rows must match the canonical854-window manifest, be eligible and have shared_cells=0. Copy exact rows/hashes into COHORT.json before inference. Annotation schema/sample metadata were inspected during feasibility; this is not a claim of never having seen TRAIN labels. No reference-conditioned query, transform, threshold, run-order or output selection.

Completed original mask-only DINO seed42, checkpoint SHA2565611e2c850fdceea01b05840a8bce9010e534f918b06f250189eb584cb154077. Frozen all weights, eval mode,300queries, official backbone/pixel/decoder architecture. Use the archived native top-K tracing adapter, whose native branch preserves the original selection expression. Actual forward-input tensors are inspected through a backbone hook. No R5 arm: native R5 forces1024resize and is not the same input-encoding contrast.

## Inputs / support

For each native1024² RGB image I (0–255 float32), A=I, B=exact2×pixel replication(U2I), C=valid-normalized2×2block mean(D2I), D=U2D2I. C remains float32; block means of valid uint8 pixels may be fractional, with no quantization or clipping. All four retain the same ground footprint. The pairs A/B and C/D have exact left-inverse equality, checked both before inference and on the normalized/padded tensors actually passed to the backbone. Shapes are1024,2048,512,1024; no automatic input resize and no cropping/new view. Input sampling is a synthetic operator, not realistic sensor-GSD simulation.

Coarse footprint validity requires all four original pixels valid. Canonical common domain V is the2×replication of that coarse validity. All arms' canonical prediction masks are clipped to this same V. Report original valid pixels, common valid pixels and removed original-valid pixels. Original reference IDs/denominators are retained across arms; any reference with no remaining common support is flagged, not silently discarded or relabeled as no object.

## Output contract / storage adaptation

Convert each query mask-logit field directly to the same1024²canonical grid using bilinear align_corners=False, then threshold>0 and clip to V. This keeps the coordinate sampling and final pixel-area units common. Class score is sigmoid tree logit. Output score is class probability times mean foreground sigmoid on the canonical valid mask. Accept score>=.05 and canonical area>=30; greedy mask-IoU NMS>.5, stable query-ID tie break. Exactly300raw query records, including empty masks; do not use references to choose a branch.

Save all canonical raw RLEs, raw class logits/probabilities, mask-quality score, native logit-grid shapes, selected encoder sites/spatial levels, predicted boxes and input/domain hashes. Full dense mask-logit dumps are explicitly NOT_STORED to keep this compact pilot within current disk capacity; this is a preregistered storage adaptation from the research sketch. They remain reproducible from the frozen checkpoint/input/code, but probability-level post-hoc analyses would require an explicitly accounted rerun. The planned stage comparisons here need actual RLEs and frozen scores, which are saved. Do not claim full dense logits were archived.

Record protocol/code/checkpoint hashes. Freeze/hash all12predictions BEFORE computing new reference-conditioned measurements. Read RGB/validity only in the inference loader, using verified TRAIN cache bindings; do not call the GT-loading training helper.

## Measurements after freeze

Raw reference availability50/75, final maximum one-to-one Recall50/75, TP/FP/FN, all-area AP/AP50/AP75 with the existing evaluator/maxDets, per-source results and significant merge/split proxies. Original source annotation area is the inherited published-tile raw_area×.045², separately from cropped common-support target area; report source_truncated flags and do not claim uncropped biological crown area.

For A→B, C→D and A→D, count raw and final NEW/LOST separately, matching ambiguity, and source-small(<9m²under that inherited annotation definition) changes. No crop/seed-as-independent-site inference. Add same-final-output-count truncation using min(counts) per window, original score order, no GT counts. Do not infer pure ranking causality from it.

Primary diagnostic target: any registered lossless pair demonstrates gross losses of at least5previously qualifying raw references, including cases in at least2of3sources, after exact input/coordinate checks. This is a finite-pilot relevance screen, not significance, novelty or a method gain. Report all changes even if it fails. Practical original-method gates (+2ppsmall recall with AP/FP constraints) are NOT evaluated by selecting an arm here.

Fixed qualitative rule: for each source, choose the reference with smallest SHA256(reference ID) among A-raw50/B-not-raw50 cases; if none, smallest hash among A raw successes. Show RGB/reference and raw-best mask per arm; all witnesses GT-conditioned post-freeze diagnostics, not deployed choices. No image/policy selection for visual appeal.

## Resource / stopping limits

12forwards maximum,20minutes wall time,4CPU threads,8GiB allocated GPU and8GiB host RSS,192MiBtotal new experiment output,5GiBfree disk floor. Available GPU is shared/notreserved. The5GiBreserve is specific to this new compact pilot, not a change to the already closed BAM experiment's20GiBcontract. No dense-feature/checkpoint copies or new weight download. If a limit/shape/pixel-equivalence check fails, preserve the attempt; do not silently lower input resolution or alter the scientific comparison. Allow only a documented engineering correction, with unchanged arm/cohort/policy; no threshold sweep.

If a lossless relevance screen passes, conclude only information-equivalent encoding dependence in this frozen TRAIN setup. Determine whether it lies in observed raw formation versus acceptance/suppression; do not automatically launch a consistency loss, query/decoder intervention or V2. If it fails, close this specific factor/cohort diagnostic without claiming universal scale invariance.
