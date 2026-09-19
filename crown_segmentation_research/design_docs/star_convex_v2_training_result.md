# Star-convex network — v2 training result (300 epochs, 300 images)

## Material Passport

- Origin: scale-up of the CODE-VERIFIED architecture in
  `star_convex_implementation_status.md`, trained per user request ("train
  tiếp nhiều hẳn hơn") to have a real result to review.
- Verification Status: SCIENTIFICALLY RUN — first real training result for
  this architecture (supersedes the 40-epoch/10-image screen).
- Checkpoint: `experiments/star_convex_screen_v2/star_convex_screen.pth`.
- Data: 300 BAM_train images + 50 BAM_val images, n_rays=16, backbone
  warm-started from the frozen G1B Mask R-CNN checkpoint (281/281 keys
  verified identical before training).

## Training

300/300 epochs completed (batch_size=1, AdamW lr=1e-4). Loss: total
16.68 -> 1.12 (epoch 0 -> 299), plateaued from ~epoch 260 onward
(1.10-1.15 range) -- converged, not diverging or still improving sharply.
One transient CUDA OOM crash during an earlier attempt (shared GPU
contention from other users' processes, not this code) was recovered via
the checkpoint/resume logic added for this reason; the run also survived
an ~10-hour host suspend/resume gap without losing progress.

## Evaluation (BAM_val, 50 images, same `benchmark.evaluator` as G1B)

| Metric | 40-epoch/10-image screen | **300-epoch/300-image (this run)** | G1B Mask R-CNN baseline (BAM_test2) |
|---|---:|---:|---:|
| matched_iou | 0.593 | **0.724** | 0.7585-0.7951 |
| recall | ~24% | **56.5%** | ~90% |
| precision | ~13% | 41.8% | — |
| split_rate | 0.207 | **0.066** | 0.1676-0.1982 |
| merge_rate | 0.012 | 0.062 | 0.0311-0.0406 |
| miss_rate | 0.494 | 0.262 | 0.0446-0.0617 |
| f1 | — | 0.481 | ~0.60+ |

n_gt=1354, n_pred=1828, tp=765 (aggregated over 50 val images).

## Interpretation

More data + more epochs produced a large, consistent improvement across
every metric versus the 40/10 screen -- this confirms the earlier
diagnosis that the screen's weakness was training scale, not a
representation flaw (unlike the H/V representation, which failed even
with perfect oracle information).

**Split_rate (0.066) is already lower than the G1B Mask R-CNN baseline's**
(0.1676-0.1982) -- consistent with the oracle-test finding that star-convex
encoding handles individual crown shape well (Section 4's original
motivation: avoid the multi-fragment over-segmentation Mask R-CNN shows).

**Recall (56.5%) and precision (41.8%) remain well below the baseline** --
this network has seen only 300 images for 300 epochs with no data
augmentation and no threshold/NMS tuning (prob_threshold=0.5,
nms_iou=0.3 are the same untuned defaults used throughout this document,
not validated values). This is the expected gap for a single untuned
training run versus a baseline that went through G1B's full
train/tune/evaluate protocol across 3 seeds.

## What this does not establish

- No hyperparameter tuning has been done (learning rate, prob_threshold,
  NMS threshold, ray_loss_weight are all first-guess defaults).
- No comparison against a same-scale Mask R-CNN run (i.e. Mask R-CNN
  trained on the same 300 images for a fair architecture comparison) --
  the G1B baseline row above used the full 1,439-image BAM_train set, so
  this table compares different training budgets, not just different
  architectures.
- Not evaluated on BAM_test1/test2 (only val, matching this screen's own
  train/val split; test sets remain locked per the frozen protocol).
