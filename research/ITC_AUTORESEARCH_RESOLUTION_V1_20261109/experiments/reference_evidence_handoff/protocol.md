# Six-image reference-evidence handoff — preparation only

Status at registration: DESIGN / PREPARATION. No annotator is enrolled, no response exists, and no new reference stability result may be reported.

Purpose: turn the remaining evidence dependency into a finite, usable local RGB packet. This is an annotation-feasibility pilot, not the geographically independent confirmatory study and not a new model experiment.

Selection is fixed to zero-based cohort positions [0,4,8,12,16,20] in the already frozen BAM TRAIN cohort, three from each header-defined geographic component. These positions are chosen by a metadata spacing rule, without consulting case-level prediction performance. The rule is registered AFTER the cohort-level analysis, so the packet is not retrospectively described as a pristine prospective sample. It must not be used to estimate general prevalence or claim adequate sample size for systematic failure. Images retain native 2048×2048 pixels; no crops, resizing, model forwards or re-training.

Read RGB and validity using the canonical treecrown_data inference role, TRAIN only. Compare decoded RGB/validity hashes to the prior frozen inference record. Strip geotags, source filenames, published masks, predictions, confidence scores, model names and source counts from the rater-facing directory. Retain dataset-level attribution/license. Private source bindings stay OUTSIDE that directory. Blinding is about prediction/reference exposure, not concealment of the dataset's public attribution or guaranteed annotator unfamiliarity.

Prepare one blank packet to be copied separately to each of three qualified, independently working annotators if participation becomes available. No recruitment, contact, upload or annotation collection is authorized by this preparation. Do not send any private binding or carrier figure with the packet. Annotators must disclose prior exposure. Do not fabricate their identities, consent, responses, competence or time requirements.

Visible-crown convention: delineate only RGB-supported visible extent, allow independent overlapping memberships and disconnected components; do not infer hidden extent or biological identity. Include presence, count/identity and boundary ambiguity, partial crowns, and explicit insufficient-evidence regions. An empty NOT_STARTED template is missing data, never evidence of no crowns. No unmarked pixel is certified background by default.

Use standard LabelMe polygon files plus a small response-status sidecar; no bespoke annotation app. Import groups polygons by instance, clips masks only to the supplied valid pixels and preserves per-instance overlaps and uncertainty flags. Require explicit completion/declaration before accepting a response as collected. Parser checks may use temporary synthetic fixtures, always labeled engineering-only and never retained as human evidence.

Verification: exactly six cases; pixel-exact native RGB/validity and original frozen-input hash agreement; no source/model/reference metadata in the rater-facing image files; blank labels/status remain uncollected; importer rejects unfinished, inconsistent and invalid responses while preserving overlapping independent masks. Store exact file hashes and reproducible commands.

Stopping gate: after preparation, the indispensable external input is actual qualified independent participation or qualifying existing multi-annotator labels with verified provenance and permitted access. Another run on the same published polygons cannot replace that evidence. No model architecture/loss/selector branch is queued. The larger research goal remains unachieved.
