# H7 Analysis — The diagnosis holds; the remedy failed the same way the disease did

**Run:** 2026-09-20 · 12 epochs, `train_apex.py`, warm-started from
`best_calibrated_watershed.pth`. Evaluated on the 60 held-out crops.

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | apex-seeded `never_lit` < 15% (from 28.1%) | **60.06%** | **FAILED** |
| 2 | apex seeding beats surface seeding by >= +3 F1@50 | 48.03 vs **68.48** (−20.4) | **FAILED** |
| 3 | best end-to-end >= 73 F1@50 | 68.48 | **FAILED** |

## Results (each seeding field at its own best decode cell)

| seed field | min_dist | min_apex | COCO AP | AP50 | AP75 | F1@50 | seeds/crown | missed | never_lit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **surface** | 140 | 0.35 | **29.07** | 61.61 | 23.98 | **68.48** | 0.93 | 20.98% | 18.48% |
| apex | 100 | 0.10 | 17.58 | 35.33 | 14.51 | 48.03 | **0.40** | 60.06% | 60.06% |

> **CORRECTION (2026-09-20, after H11).** The section title below and its claim that
> the head "never learned" are **wrong**. Measured directly, the apex head reaches
> **AUROC 0.841** against "within 15 px of a crown centre" — almost the surface head's
> 0.863. What it lacks is *magnitude*: mean 0.0041, p99 0.0212, so the `min_apex=0.10`
> seeding threshold sits **above its 99th percentile**. The Tversky flatline at ~1.00
> reflected magnitude collapse, not absence of learning (with beta=0.7 on a ~1% target
> the false-negative term dominates any small true intersection). H7's *conclusion* —
> apex seeding fails, this remedy is refuted — stands. Its *reason* is replaced by the
> corrected one, and H11 shows the head's gradient was worth +1.22 AP to the shared
> trunk despite being unusable as an output. See `../h11-aux-task/analysis.md`.

## Why it failed: the apex head never learned

The apex loss (balanced BCE + Tversky) went **1.745 → 1.015 → 1.003** and flattened
there. A Tversky term near 1.0 means **essentially zero overlap between prediction and
target** — the head is not finding the peaks at all. Consistent with that, apex seeding
emits only **0.40 seeds per crown** even at the loosest threshold swept (`min_apex=0.10`)
and leaves **60%** of crowns unlit.

The cause is the target's sparsity. The seam band that collapsed the original boundary
head covers ~6% of pixels. A Gaussian centroid peak with a **fixed small sigma** covers
a small fraction of one percent. The very property that made the apex target attractive
— a peak whose size does not grow with the crown — also makes it an extremely sparse
target, and the same class-balanced objective that partially rescued the boundary head
(0.105 → 0.186) could not lift this one off the floor at all.

**So H7 reproduced the exact failure it was designed to fix, one level down.** The
diagnosis from the missed-crown analysis is untouched: crowns that go unseeded are small
crowns the surface head never lights, and per-instance normalisation is why. What is
refuted is the specific remedy — *this* apex target, supervised *this* way.

If the idea is pursued, the target has to change before the loss does: a larger sigma,
or a sigma that scales with crown size but a **peak value** that does not (which
preserves scale-invariance where it matters, at the maximum, while giving the head
enough positive area to learn from). That is the same lesson H6 reached from the other
direction — widen the target rather than reweight the loss further.

## An unexpected result, and it is confounded

The **surface** path on this checkpoint scores **AP 29.07**, against H6's best of 27.28
on the identical split and grid (H10) — **+1.8 AP** from a run whose stated purpose
failed entirely.

**This cannot be attributed yet.** The H7 checkpoint had **12 more epochs** of training
than H6 *and* an auxiliary apex task. Either could explain it:
- more training on the same objective, or
- the auxiliary task acting as a regulariser on the shared trunk — which would be
  interesting precisely because the auxiliary head itself is unusable.

H11 separates them by continuing `train_calibrated.py` for 12 further epochs from the
H6 checkpoint with **no** apex head, giving a matched-epoch control.

## Carried forward

H7's checkpoint is currently the best configuration overall by COCO AP (29.07, surface
seeding, `md=140, ma=0.35`, no merge), but that claim is provisional until H11 says
whether it came from the auxiliary task or just from more epochs. Reported here with
the confound attached rather than as a clean win.
