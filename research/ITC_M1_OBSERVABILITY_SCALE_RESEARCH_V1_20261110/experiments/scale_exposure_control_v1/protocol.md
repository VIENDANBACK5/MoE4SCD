# Ordinary scale-exposure control — fixed before training

Question: can a known input-scale augmentation reduce the measured lossless-raster sensitivity without a new mechanism or an unacceptable loss of native instances? This is an established learning control, not an original method. Failure within its finite schedule will not prove architecture incapacity.

## Exact arms and data

Both start from completed mask-only DINO seed42/final.pt SHA2565611e2c850fdceea01b05840a8bce9010e534f918b06f250189eb584cb154077. Original trainable scope, classification/box/mask/DN/auxiliary losses, mask-only Hungarian matching (cost_box=cost_giou=0), query budget300, backbone/pixel/decoder architecture and validity semantics are preserved.

S0: matched native1024continuation.
S1: identical continuation except every odd microbatch is exact2×RGB/mask/validity replication to2048; every even microbatch remains1024. No degraded512training or vanished small targets. All independent target memberships are replicated, never selected by reference IoU. No new loss/score/selector.

TRAIN_WINDOWS.json is all796eligible windows from the canonical854-window manifest except the three entire tiles used for the completed diagnostic. Those three tiles have already been seen by the parent carrier; exclusion concerns only these additional optimizer updates, not independent geographic or unseen-carrier validation. Source counts385TBS/297BCI/114ZF2. Fixed SHA256ordering, one complete pass,398updates with accumulation2/effective batch2. Same source instances, per-window D4draws and RNG seeds in both arms. No VAL/TEST exposure or model-selection feedback.

Fresh AdamW states in both arms; LR1e-5nonbackbone/1e-6backbone, weight decay.05 with inherited zero-decay exceptions, clipnorm.01, bf16autocast/FP32weights as in archived training. Linear warmup.1→1over20updates, then constant. Fixed-final checkpoint; no sweeps, early VAL selection or rescue epoch.

## Engineering gate before full execution

Use first fixed TRAIN row train_tile_017_w23, outside diagnostic tiles. On one disposable initialized model, compare native1024loss and every available trainable gradient with/without standard torch nonreentrant activation checkpointing of ResNet stages and transformer layers. Reset torch/CUDA/numpy/Python RNG to identical values. Require loss relative agreement1e-5and gradient allclose atol1e-6/rtol1e-3. This is a numerical engineering check, not a model-quality experiment.

Then execute one real2048replicated training forward/backward and one AdamW update with that checkpointing. Preserve numerical logs/gradient/update/memory statistics; deliberately discard probe weights and optimizer (not used in either scientific arm). Checkpointing recomputes the same forward functions; no parameter/layer/loss replacement or extra freezing. Use it identically in S0/S1. If the gate fails, retain failure and do not silently lower resolution or restrict training scope.

## Resources and checkpoints

One arm at a time on the currently shared RTX5880Ada; at most20GiB allocatedGPU,16GiB hostRSS,4CPU threads,45minutes per arm. No reservation or interruption of other jobs. Before every save require projected free disk after write>=4GiB. Total new study artifacts<=2GiB, including checkpoint temporaries. This new full-training contract's4GiBreserve is explicit and does not rewrite any older closed experiment's5/20GiBlimit.

Save complete model+optimizer+RNG/draw-index checkpoint every100updates and at final; latest state replacement is atomic and final aliases the completed last state rather than duplicating all tensors. Retain per-update losses, update/gradient statistics, augmentation IDs/scales and data/source hashes. Resume only from an actually verified saved state, never from a progress marker. Probe is separate; both scientific arms reload the original seed-specific parent. Record failed attempts; no automatic protocol changes.

## Frozen evaluation and decision

After both fixed398-update arms complete, run the same four A/B/C/Dinputs on the three diagnostic windows, using the prior saved RGB/validity arrays,300queries, identical canonical1024mask materialization and score/area/NMSpolicy. Freeze actual raw/final predictions before reference measurements. No SAME6evaluation or policy selection.

Compare native and re-encoded raw50/75, AP50/75/all-areaAP, one-to-one finalRecall/FP, source-small occurrence metrics, gross NEW/LOST, equal-count and per-source effects. Diagnostic references/crops are the same298seen-TRAIN occurrences, not independent biological trees/sites. Original published-tile source area is separate from crop-fragment area.

Practical screening for conventional-treatment sufficiency: S1's A/B raw50gap is at least50%smaller than S0's gap when S0gap>0, while S1nativeAfinalTP50 is no more than3below S0A. Also report AP/FP and all collateral losses; this gate is not statistical significance, deployment gain, novelty or a substitute for original small-recovery targets.

If ordinary augmentation/canonicalization adequately explains the phenomenon, do not invent an observability module. If not, report only this bounded control's outcome and remaining alternatives; do not automatically open V2, representation/query/decoder branches or longer training.
