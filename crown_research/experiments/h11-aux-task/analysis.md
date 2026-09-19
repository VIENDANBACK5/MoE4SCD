# H11 Analysis — The auxiliary task does two thirds of the work, and it works *because* the head learned

**Run:** 2026-09-20 · matched-epoch control, three checkpoints over one decode grid on
the 60 held-out crops, surface seeding, no merge.

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | H11 >= 28.5 AP (extra epochs explain >= 70%) | **27.85**, recovers **32%** | **FAILED** |
| 2 | \|H7 − H11\| < 1.0 (auxiliary task adds little) | **1.22** | **FAILED** |
| 3 | H11 >= H6 (no regression) | 27.85 >= 27.28 | CONFIRMED |

Both substantive predictions failed, in the direction that makes the result interesting.

## Results

| checkpoint | epochs past H6 | apex head | best COCO AP | AP50 | AP75 | F1@50 |
|---|---:|---|---:|---:|---:|---:|
| H6 calibrated | 0 | no | 27.28 | 59.54 | 20.38 | 66.72 |
| **H11** | +12 | **no** | 27.85 | 59.54 | 21.38 | 66.73 |
| **H7** | +12 | **yes** | **29.07** | 61.61 | 23.98 | 68.48 |

Decomposition of H7's +1.79 over H6:

- **more epochs: +0.57 (32%)**
- **auxiliary apex task: +1.22 (68%)**

The same ordering holds at a common decode cell (27.28 / 27.82 / 28.81), so it is not
an artefact of each checkpoint picking a different optimum.

## This forces a correction to H7's analysis

H7's write-up said the apex head "never learned", reading the Tversky term's flatline at
~1.00 as zero overlap. **That was wrong**, and H11's result prompted the check:

| field | mean | p99 | max | AUROC vs "near crown centre" | AUROC vs foreground |
|---|---:|---:|---:|---:|---:|
| apex head | **0.0041** | 0.0212 | 0.973 | **0.841** | 0.578 |
| surface head | — | — | — | 0.863 | 0.925 |

**The apex head ranks crown centres almost as well as the surface head (0.841 vs
0.863).** What it lacks is magnitude: mean 0.0041, p99 0.0212. The seeding threshold
`min_apex = 0.10` sits *above its 99th percentile*, which is the entire reason apex
seeding emitted 0.40 seeds per crown. Tversky stayed near 1.0 because with beta=0.7 on a
~1% target the false-negative term dominates any small true intersection — the flatline
measured magnitude, not learning.

So H7's *conclusion* (apex seeding fails, the remedy is refuted) stands, but its
*reason* was wrong. The corrected reason is far more useful, because it is the project's
central pathology appearing a third time, and now with a dose-response:

| head | positive rate | mean output | AUROC | usable as magnitude? |
|---|---:|---:|---:|---|
| canopy | ~60% | 0.85 | 0.96 | yes |
| boundary (pre-H6) | ~6% | 0.105 | 0.866 | barely — H2/H4 both failed on it |
| boundary (post-H6) | ~6% | 0.186 | 0.872 | yes, and it narrowed the domain gap (H10) |
| **apex** | **~1.2%** | **0.0041** | **0.841** | **no** |

**Ranking survives sparsity; magnitude does not, and the collapse scales with the
positive rate.** Every head in this pipeline learns *where* its structure is. Only the
ones with enough positive area learn *how strongly* to say so — and every downstream
consumer here reads magnitude.

## Why the auxiliary task helps

This also explains H11's headline. The apex head is useless as an output but it did
learn the ranking, so its gradient carries real information about crown-centre
localisation into the shared trunk. That is a textbook auxiliary-task benefit, and it is
worth **+1.22 AP** — more than the entire H5 decode-side programme and about half of
H6's gain.

**Training a head you never read is, here, a better use of 12 epochs than training
without it.** Cheap, requires no change at inference, and the target already existed in
the codebase.

## Carried forward

H7's checkpoint (`best_apex_watershed.pth`) is now the best configuration on the
validation hold-out by COCO AP — **29.07**, surface seeding, `md=140, ma=0.35`, no
merge — and the attribution is settled: two thirds auxiliary task, one third extra
epochs. It should still be checked on `test1`/`test2` before being called best overall,
since H10 showed in-domain and out-of-domain gains are not proportional.

## A prediction this creates

If magnitude collapse scales with target sparsity, then widening the *apex* target
(larger sigma, peak value held fixed) should do for apex seeding what H12 is currently
testing for the boundary head. That is the natural follow-up and is not run here.
