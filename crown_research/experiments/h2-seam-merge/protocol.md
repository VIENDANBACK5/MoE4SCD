# H2 — Escaping the seeding frontier by over-seeding then merging on seam evidence

**Status:** LOCKED (pre-registration) · 2026-09-19
**Type:** CONFIRMATORY
**Tuning split:** `eval30` · **Hold-out split:** `evalhold60` (eval idx 100-159, never tuned on)

## Mechanism

H1's seed audit showed global NMS is trapped on a precision/recall frontier:

| suppression | crowns hit | duplicate seeds/crown | crowns missed |
|---|---:|---:|---:|
| loose (md=24, ma=0.25) | **83.1%** | 0.820 | 16.9% |
| tight (md=112, ma=0.55) | 62.5% | 0.039 | 37.5% |

Loose seeding already *finds* 83% of crowns. Its only defect is that each found crown
receives ~1.8 seeds and is therefore split. That defect is repairable **after**
flooding, where NMS's information deficit does not apply: once basins exist, the
predicted boundary field B can be read *along the actual shared seam* between two
basins instead of guessing from peak spacing alone.

H0 cell C established that B is informative about geometry (oracle seams lifted SQ
67.7 -> 74.3 and B-IoU 10.3 -> 24.4) even though it cannot drive recognition alone.
A seam-integrated statistic is exactly the use B should be good for.

## Method

1. Seed loosely; flood to an over-segmented basin map.
2. Fill watershed lines to nearest label; build the region-adjacency graph.
3. For each adjacent pair (a,b): `seam_B(a,b)` = mean predicted boundary value over
   the shared seam pixels; `seam_len(a,b)` = number of such pixels.
4. Union-find merge every pair with `seam_B < tau` and `seam_len >= L_min`.
5. Re-label, drop components below `min_area`.

## Arms

| Arm | Seeds | Post-process |
|---|---|---|
| A | loose (24, 0.25) | none — published baseline |
| B | tuned (160, 0.50) | none — H1 best |
| **C** | loose (24, 0.25) | **seam-B merge**, sweep `tau` x `L_min` |
| **D** | loose (24, 0.25) | **oracle merge** — merge iff both basins' dominant GT label agrees |
| E | tuned (160, 0.50) | seam-B merge at C's best `tau` |

Arm **D** is the diagnostic that separates *framework* from *criterion*: it is the best
any merge rule could do given these basins. If C falls far short of D, the merge
framework is right and the criterion needs learning; if C ~= D, the framework is spent.

Grid for C: `tau in {0.15, 0.25, 0.35, 0.45, 0.55, 0.65}` x `L_min in {5, 15, 30}`.

## Predictions (committed before running)

1. **C beats B by >= 3 mAP@50 on the hold-out.** (B = 65.54; so C >= 68.5.)
2. **D >= 80 mAP@50 on the hold-out**, i.e. the over-seed-then-merge *framework* has a
   ceiling well above the tuned-NMS 65.5 and above H0's oracle-seed 70.8, because it
   retains loose seeding's 83% hit rate rather than the oracle's one-seed-per-crown.
3. **C recovers >= 40% of the gap between B and D**, i.e. `(C - B) / (D - B) >= 0.40`.

If prediction 1 fails but 2 holds, the framework is validated and the project moves to
a *learned* merge criterion (features: seam B statistics, basin size, surface dip
across the seam) rather than a single threshold on B.
If prediction 2 also fails, over-seeding is not recoverable and the project returns to
the seeding stage — likely a supervised scale/count head, since H1 showed scale cannot
be recovered at decode time.

## Selection rule
Best `tau`/`L_min` chosen on `eval30` by mean mAP@50 (ties -> PQ), then **frozen** and
re-scored on `evalhold60`. All headline claims quote the hold-out number.

## Outputs
- `results/h2_merge_sweep.csv`, `results/h2_summary.csv`, `results/h2_seed_audit.csv`
