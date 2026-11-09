# Reference evidence handoff

**VERIFIED_PREPARED_NOT_COLLECTED**. Six native-resolution RGB cases, zero annotators confirmed, zero actual responses, zero new model forwards. This package is an executed preparation artifact, not executed human research, an independent test, or a solution to ITC.

The immediately usable handoff is [RGB_BLIND_PACKET.zip](RGB_BLIND_PACKET.zip) (52,550,929 bytes). Supply ONLY that archive, separately, to any future independently participating annotators. Do not supply this coordinator directory, private bindings, previous reports, predictions or published polygons. The `private/` directory is logical separation in a shared workspace, not a claim of secure access control. Dataset attribution is deliberately retained; blinding is not a guarantee of unfamiliarity with public images. Prior exposure must be declared.

## What was actually executed

Protocol commit `12624df` preceded extraction. Six positions [0,4,8,12,16,20] from the previously locked 24-block BAM TRAIN cohort were exported by the inference-only canonical API. This metadata rule was chosen after the previous overall experiment, so no prospective independent sampling claim is made. Three images per existing metric-coordinate component; no outcome-based selection, new crop, resampling, model inference or new reference read. PNG pixel/validity hashes agree with the previous frozen input records. One actual PNG was visually inspected for rendering; this inspection did not create identity judgments or reviewer annotations.

Each blank packet contains RGB, validity, GSD, empty LabelMe files, response-status sidecar, visible-crown/uncertainty instructions and the installed source-license text. Neither old masks, model outputs, source annotation counts nor case source names are included. No packet has been sent, no person contacted, and no enrollment or review time is invented.

The importer preserves independent mask overlaps, multiple polygon components per instance, presence/identity/boundary uncertainty, insufficient-evidence regions, explicit empty cases and unknown counts. It rejects incomplete/unblinded declarations, inconsistent identities and self-crossing polygons. Invalid pixels are clipped and recorded. Unmarked pixels are not automatically declared reliable background. Temporary synthetic parser checks were deleted and are explicitly NOT human evidence.

## Evidence dependency and bounded next decision

To proceed requires three qualified independently working annotators for the same six images, or an accessible qualifying pre-existing multi-annotator resource with provenance/permission. No such participation has been confirmed. The specific availability question has been sent to the user after preparing the handoff. It is a missing-information request, not an instruction to recruit or permission inferred from silence.

Six images would test feasibility and reveal whether reference interpretation is usable; they cannot establish the prevalence/replication threshold for a systematic cross-site failure. The larger study still needs the frozen quantitative design and sample support before execution. Consensus would support visual-reference stability, not biological identity. Source-small area remains unavailable from cropped BAM annotations; new polygons alone would not restore full-source identity/area. Acquisition/pretraining independence also remains unresolved. Newly interpreted images would be exposed development evidence for a later method.

No architecture, loss, selection policy, threshold search or V2 follows automatically. Existing RR_MASK_INIT remains an established coverage baseline. A new method is not justified merely because further independent annotation is unavailable.

## Reproduction

From the repository root, using the installed environment:

```bash
# Already executed once; deliberately refuses to overwrite an existing packet.
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/reference_evidence_handoff/code/prepare.py

# Executed on the actual packet; verifies input hashes and parser/storage contracts.
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/reference_evidence_handoff/code/verify.py

# NOT_RUN: use only after an actual independent response exists in a separate copy.
cs2_venv/bin/python research/ITC_AUTORESEARCH_RESOLUTION_V1_20261109/experiments/reference_evidence_handoff/code/import_response.py --packet COMPLETED_PACKET_COPY --output NEW_RESPONSE.json.gz
```

The importer deliberately rejects the shipped blank packet. An imported response remains PENDING human provenance/protocol review, not automatically an independent or adjudicated reference. The full-mask predictions and their frozen policies remain those of cycle 2.

See [verification](VERIFICATION.json), [exact packet hashes](PACKET_FILE_HASHES.json), [registered protocol](protocol.md), and [resource findings](RESOURCE_ACCESS.md).
