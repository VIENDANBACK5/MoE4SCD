# Reproducibility — executed scale-exposure control

## Bound inputs and implementation

Run from `/home/chung/RS/Image Segmentation`, using `cs2_venv/bin/python`. `PLAN.json`, `protocol.md`, `TRAIN_WINDOWS.json` were saved before scientific training. The separate numerical addendum was saved before the bounded correction probe. `PLAN.training_started=false` records that registration time, not current status. See `logs/S0/STATUS.json` and `logs/S1/STATUS.json` for actual completion.

The completed original Mask DINO parent and canonical loader/evaluator are reused; no download, new backbone, new annotation or protected data. `SOURCE_BINDINGS.json` binds 124 actual source/manifest files, captured during training, and `SOURCE_BINDINGS_VERIFIED.json` confirms their hashes after execution. This is not represented as preregistered payload hashing. `ENVIRONMENT.json` and `requirements-observed.txt` record the installed environment. No claim of bitwise reproducibility across GPU/PyTorch versions.

The code imports the immutable prior pilot's materialization/evaluator and exact NPZ inputs. It requires the existing research repository, cached RGB/validity/reference artifacts and both archived package trees; it is not an unrelated standalone benchmark package.

## Exact execution commands

These were executed in sequence; training stdout/stderr is retained in `probe.log`, `probe2.log`, `training.log`, `evaluation.log`, `analysis.log`, `verification.log`, and `report.log`.

```bash
cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/code/probe.py
cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/code/probe2.py
cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/code/run.py
cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/code/evaluate.py
cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/code/analyze.py
cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/code/verify.py
cs2_venv/bin/python research/ITC_M1_OBSERVABILITY_SCALE_RESEARCH_V1_20261110/experiments/scale_exposure_control_v1/code/report.py
```

`probe.py` actually failed the original strict bf16 pointwise-gradient gate, before scientific training. `probe2.py` executes the documented fp32 parity / bf16 rounding correction and a disposable 2048 optimizer step. Seven engineering forwards and one discarded optimizer update total; the two scientific arms reload the original parent and do not use probe weights. This failure/correction is part of provenance, not a passing first attempt.

`run.py` invokes `train.py --arm S0` then `train.py --arm S1`. Each makes 796 microbatch forwards and 398 updates. Fresh optimizer states, sample order, augmentation draws and LR match. The frozen evaluation makes 24 additional forwards; references are loaded only after all predictions are sealed. No model selection follows these metrics.

The completed scientific arm/freeze guards deliberately reject overwriting this directory. For a new numerical reproduction, use a fresh sibling experiment directory containing the registered code/configs and the documented engineering lineage, preserving relative repository depth and access to `../frozen_encoding_v1`. Do not delete completion files or historical masks to bypass guards. Re-run the hardware/numerical gate there, then execute the fixed schedule; no additional run has been launched here. The verification command alone can be rerun immediately against the retained outputs.

## Model state and RGB-only masks

Both `checkpoints/{S0,S1}/final.pt` contain the complete model, AdamW states for 393 parameter tensors, optimizer-update / next-microbatch position and torch/CUDA RNG states. `last.pt` and `final.pt` hard-link the same completed state, preserving a resumable state without double disk use. Each has 43,775,909 trainable parameters. Earlier temporary snapshots were atomically replaced as registered; their individual weights are not claimed retained.

The model and source hashes are listed in `VERIFICATION.json` / `FINAL_ARTIFACTS.json`. `INITIALIZATION.json` in each arm records training/shared source hashes at execution. Checkpoint wrappers alter recomputation, not the state-dict architecture. Report-only edits after scoring do not change training/inference/selection rules.

`evaluations/{S0,S1}/predictions/*.json.gz` contain all 300 independent raw RLE masks, accepted/final IDs, scores, class probabilities, boxes, encoder site trace, input/normalized-backbone hashes and geometry per window/input. Common canonical 1024 validity, logit threshold 0, score .05, area 30 canonical pixels and mask-IoU NMS .5 are fixed. Dense logits are NOT_STORED, as declared in the parent pilot's disk contract.

The 24 saved predictions have 7,200 raw and 3,463 final RLE masks. Final one-to-one recall uses the inherited maximum-cardinality IoU matching; COCO AP uses the inherited score-ranked evaluator. Raw reference availability, simultaneously feasible matching and AP remain distinct. `results/SAME_COUNT.csv` uses the lower prediction count per window, score order only, never GT to choose candidates.

The three qualitative cases/crops were fixed in the parent pilot and reused. GT chooses only the displayed raw-best witness after prediction freeze; these plots are diagnostic, not a deployable selector or independent identity review. One case is a crop fragment. Figure margins were corrected after visual inspection; no case, mask, crop or metric changed.

## Limits and decision

One paired seed, three parent-seen TRAIN windows and 298 reference occurrences. Diagnostic tiles were excluded only from new updates; no geographic independence. Published-tile source area, crop fragments and validity semantics are inherited. S1 processes 2.5× raster pixels and takes more compute; only updates and source supervision are matched. These results cannot identify BAM's cause, validate biological identity, establish independent efficacy, or justify original-method novelty.

See `REPORT.md` and `../../05_MATCHED_SCALE_CONTROL_DECISION.md` for the actual gain/loss and native AP/FP trade-off. No further training, threshold tuning, V2 or protected evaluation is queued.
