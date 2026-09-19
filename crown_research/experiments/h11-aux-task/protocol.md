# H11 — Was H7's +1.8 AP the auxiliary task, or just more epochs?

**Status:** LOCKED (pre-registration) · 2026-09-20
**Type:** CONFIRMATORY · matched-epoch control for a confound in H7

## The confound

H7 added an apex head and failed on all three of its predictions — the head never
learned (Tversky flat at ~1.00, 60% of crowns unlit). Yet the **surface** path of that
checkpoint scores **AP 29.07** against H6's **27.28** on the identical split and decode
grid: **+1.8 AP** from a run whose purpose failed.

Two explanations, not separable from H7 alone:
- **(a) more epochs** — H7 trained 12 epochs beyond H6's 12, on the same objective;
- **(b) auxiliary-task regularisation** — the apex head, though useless as a seeding
  field, may still shape the shared trunk.

(b) would be the more interesting result precisely because the auxiliary head is
unusable on its own.

## Design

Continue `train_calibrated.py` for **12 further epochs** warm-started from
`best_calibrated_watershed.pth`, **with no apex head and no other change** —
same loss, optimiser, schedule, augmentation, seeds. Output
`best_calibrated_x2_watershed.pth`.

That gives a three-way comparison at matched epoch budget:

| checkpoint | epochs from H6 start | apex head |
|---|---:|---|
| H6 `best_calibrated` | 0 | no |
| **H11 `best_calibrated_x2`** | **+12** | **no** |
| H7 `best_apex` | +12 | yes |

All three scored over the same decode grid on `evalhold60`, surface seeding, no merge.

## Predictions (committed before running)

1. **More epochs explains most of it.** H11 reaches **AP >= 28.5**, i.e. it recovers
   >= 70% of H7's +1.8 over H6.
2. **The auxiliary task adds little.** `|AP(H7) − AP(H11)| < 1.0`.
3. **H11 does not regress in-domain.** AP(H11) >= AP(H6) = 27.28.

If 1 fails — H11 lands near H6 while H7 sits ~1.8 above — then the auxiliary apex task
*is* doing real work on the trunk despite being unusable as an output, which would make
"train a head you never read" a finding worth its own follow-up.

Also re-scored: `test1` and `test2` for whichever checkpoint wins, since H10 showed
in-domain gains and out-of-domain gains are not proportional and should not be assumed
to transfer.

## Outputs
- `results/h11_sweep.csv` — three checkpoints x decode grid
- `results/h11_summary.csv`
