# H5 Analysis — Rank relief fails (gradient noise); Percentile Seam Merge sets new SOTA (66.91% mAP@50)

**Run:** 2026-09-20 · Full calibration & rank sweep on `eval30`, validated on `evalhold60` (60 disjoint crops).

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | Rank relief (B) beats raw-B control by >= +2 mAP@50 | 56.10 vs 65.54 -> **-9.44** | **FAILED** |
| 2 | Percentile merge (C) >= 67.96 mAP@50 | 66.91 mAP@50 (+1.37 over control) | **FAILED** (by 1.05 pt) |
| 3 | Canopy gate covers >= 30% of GT seam pixels (Arm D) | **32.11%** (vs 5.24% for interior) | **CONFIRMED** |

---

## Results on Held-out Validation (60 Crops)

| Method / Configuration | mAP@50 (%) | mAP@75 (%) | PQ (%) | SQ (%) | RQ (%) | Boundary-IoU (%) | Dao ED | Seed Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Historical Baseline (md=24, ma=0.25)** | 44.29 | 16.13 | 31.01 | 69.17 | 44.29 | 11.22 | 0.598 | 1.455 |
| **H1 Tuned NMS (md=160, ma=0.50)** | 65.54 | 28.47 | 47.42 | 72.13 | 65.54 | 11.65 | 0.644 | 0.756 |
| **H5-B Rank Relief ($\gamma=2.0$)** | 56.10 | 14.85 | 38.32 | 66.93 | 56.00 | 6.87 | 0.733 | 0.756 |
| **H5-C Percentile Seam Merge ($q=50$)** | **66.91** | **30.45** | **48.82** | **72.78** | **66.91** | **12.84** | **0.638** | **0.739** |
| **H3 Oracle Ceiling (Seeds + Energy + Canopy)** | **97.30** | **93.10** | **89.69** | **92.19** | **97.30** | **59.20** | **0.103** | **0.986** |

---

## Key Insights

1. **Why Rank-Relief Failed Mechanically:**
   - Transforming $B$ to its empirical CDF stretches even tiny background noise in flat crown interiors into a non-zero slope ($[0, 1]$ uniform distribution).
   - The watershed algorithm treats every micro-variation in the background as an energy gradient, causing the flooding front to stall or deviate from true crown boundaries.
   - **Conclusion:** Watershed relief *requires* genuine flat basins with zero internal gradient; raw low-magnitude $B$ is far better than forced continuous rank energy.

2. **Why Percentile Merge ($q=50$) Succeeds:**
   - Instead of using a fixed global cutoff $\tau$ (which failed in H2 because $B$ was magnitude-collapsed), dynamic per-crop rank percentile thresholding ($q=50$) merges the bottom 50% lowest-barrier seams.
   - This achieves **66.91% mAP@50** and **48.82% PQ** on 60 held-out crops, establishing the highest non-oracle score in the entire project.

3. **Validation of the Canopy Barrier Mechanism (Arm D):**
   - 32.11% of all ground-truth crown contact seams coincide with predicted canopy gaps ($C < 0.25$), compared to only 5.24% of crown interiors.
   - This explains why the binary canopy gate provides strong geometric separation in natural forest canopies.
