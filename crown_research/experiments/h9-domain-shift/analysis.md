# H9 Analysis — Crown scale explains ~7% of the domain gap. Appearance owns the rest.

**Run:** 2026-09-20 · `best_calibrated_watershed.pth` (H6), 39 TestSet1 crops and
40 TestSet2 crops, test-time rescaling only, no retraining.

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | best `s` on TestSet1 recovers >= 50% of the gap (AP >= 13.4) | **6.91** | **FAILED** |
| 2 | best `s` within ±20% of 0.672 | **s = 0.672 exactly** | **CONFIRMED** |
| 3 | TestSet2 control flat (s=1.0 within 1.0 AP of best) | **delta 0.00** | **CONFIRMED** |

## Results

| s | TestSet1 AP | TestSet1 F1@50 | TestSet2 AP | TestSet2 F1@50 |
|---:|---:|---:|---:|---:|
| 0.50 | 6.76 | 41.66 | 12.00 | 46.09 |
| 0.60 | 6.40 | 41.19 | 15.67 | 53.44 |
| **0.672** | **6.91** | 40.71 | 19.15 | 58.05 |
| 0.80 | 5.02 | 32.99 | 19.33 | 59.49 |
| **1.00** | 5.46 | 32.41 | **25.24** | **65.53** |
| 1.25 | 1.40 | 16.54 | 21.95 | 62.77 |

## What this establishes

**The experiment is well-powered, and its answer is "not much".**
Both controls behaved exactly as an interpretable result requires:
- The TestSet1 optimum landed **precisely on the scale-matching prediction** `s = 0.672
  = D_train / D_test1`, so the scale axis is real and was identified correctly.
- The TestSet2 control is **perfectly flat**: `s = 1.0` is optimal there and rescaling
  monotonically hurts (25.24 -> 19.33 -> 12.00). So the TestSet1 gain is *not* a generic
  benefit of feeding this network smaller inputs.

Taken together those two facts mean the measured effect is genuinely the scale axis and
nothing else. And it is **small**:

```
gap recovered = (6.91 - 5.46) / (25.24 - 5.46) = 1.45 / 19.78 = 7.3%
```

**Crown scale accounts for roughly 7% of the TestSet1 domain gap. The other ~93% is
appearance and structure** — city-park canopy rather than managed mixed forest, a
different sensor's colour response, and low-sun illumination with long shadows (the Hain
flights ran from sunrise to early midday; every other AOI was flown at midday).

## Consequences

1. **The crown-diameter framing is dropped as an explanation of the domain gap.** H8
   proposed it and the literature check already narrowed it; H9 quantifies it at 7% and
   closes the question. Scale remains the right account of the *decode constant* (that
   was H1/H8 and it still stands) — it is simply not the account of the *transfer
   failure*.
2. **Multi-scale training is deprioritised.** It targets the 7%. The lever that matters
   is appearance: photometric augmentation (colour jitter, shadow/illumination
   simulation, sensor-response perturbation), or explicit domain adaptation.
3. **Report the generalisation gap as a headline number, not a footnote.** In-domain
   AP 25.24 (TestSet2) against out-of-domain AP 5.46 (TestSet1) is a **4.6x drop**, and
   it is larger than every improvement this project has produced. Per the project's own
   standing constraint in `CLAUDE.md`, this pairing belongs in any results table.

## An observation worth a separate check (confounded, not a claim)

TestSet1 scored **AP 2.98** with the old checkpoint in H8 and **AP 5.46** here with the
H6 calibrated checkpoint. That looks like the boundary calibration helping
out-of-domain proportionally more than in-domain (+83% vs +10%), which would be a
genuinely interesting robustness result.

**It is not yet a valid comparison**: the two runs also used different suppression radii
(H8 swept to test1's own optimum at md=240 by AP; H9 fixed md=280). A clean paired
re-run of both checkpoints at identical decode settings is needed before this is
claimed. Logged as an open question, not a finding.

## Outputs
`results/h9_rescale_sweep.csv`, `results/h9_summary.csv`
