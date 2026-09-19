# H10 Analysis — Calibration is an out-of-domain robustness intervention, and the reason is an absolute usability floor

**Run:** 2026-09-20 · both checkpoints swept over the same 15-cell decode grid on the
same crops. Only the weights differ.

## Pre-registered predictions vs outcome

| # | Prediction | Outcome | Verdict |
|---|---|---|---|
| 1 | relative gain OOD > in-domain | **83.0% vs 18.6%** | **CONFIRMED** |
| 2 | OOD absolute gain >= +1.0 AP | **+2.47** | **CONFIRMED** |
| 3 | controlled gain **< 83%** (my bet against my own earlier observation) | **83.0%** | **FAILED** |

Prediction 3 was deliberately set against H9's uncontrolled observation, expecting part
of the +83% to be a decode artefact. **It was not.** The effect survives decode control
essentially unchanged.

## Results

| split | domain | old AP | H6 AP | ΔAP | Δ relative |
|---|---|---:|---:|---:|---:|
| evalhold60 | validation (in-domain) | 24.12 | 27.28 | +3.17 | +13.1% |
| test2 | in-domain | 21.32 | 25.28 | +3.96 | **+18.6%** |
| **test1** | **OUT-of-domain** | **2.98** | **5.46** | +2.47 | **+83.0%** |

At a single common cell (`md=200, ma=0.50`) the pattern holds: +12.8% / +17.2% / +63.8%.

**The generalisation gap itself narrows:**

| checkpoint | in-domain (test2) | out-of-domain (test1) | ratio |
|---|---:|---:|---:|
| old | 21.32 | 2.98 | **7.15x** |
| H6 calibrated | 25.28 | 5.46 | **4.63x** |

This is the first intervention in the project that narrows that gap at all.

## Mechanism — not what I assumed

My working explanation was that a calibrated head *degrades more gracefully* under
shift. Measuring it directly refutes that:

| checkpoint | B on seams, test2 → test1 | change | B AUROC, test2 → test1 |
|---|---|---:|---|
| old | 0.0990 → 0.0566 | **−43%** | 0.853 → 0.766 |
| H6 | 0.1791 → 0.1075 | **−40%** | 0.857 → 0.778 |

**Both heads lose the same proportion of their seam magnitude — about 40% — and their
ranking degrades identically.** Calibration buys no extra robustness in the head itself.

What differs is where that leaves them in absolute terms. The watershed floods on
`boundary * 255`, so mean seam response converts directly to barrier height:

| checkpoint | test1 barrier height |
|---|---|
| old | 0.0566 × 255 ≈ **14 / 255** |
| H6 | 0.1075 × 255 ≈ **27 / 255** |

**There is an absolute usability floor for the decoder.** Below some ridge height the
flood simply cannot hold a boundary, and the old head falls through it out-of-domain
while the calibrated head stays (barely) above. The domain shift costs both heads the
same 40% — **only one of them could afford it.**

Two details confirm the reading:
- The old head's seam-to-background *ratio* is actually **higher** out-of-domain (8.69)
  than in-domain (7.23). Ranking contrast is not the problem; absolute magnitude is.
- The canopy head barely degrades (AUROC 0.929 → 0.921), so the fragility is specific to
  the boundary head, which is exactly the head that was collapsed.

## What this predicts

If the mechanism is a floor rather than a slope, then:
1. The AP benefit of calibration should **grow with the severity of the shift**, since
   more severe shifts push more of the ridge field under the floor.
2. Pushing seam magnitude higher still (H6 reached 0.186 against the 0.40 originally
   aimed at) should buy **disproportionately more out-of-domain than in-domain**.
3. Any other head whose output is consumed as an *absolute* magnitude — rather than by
   rank or by threshold-crossing — carries the same fragility.

Point 2 makes the pre-registered H6 fallback (widen the seam target, since a 3 px band
blurred at sigma 0.8 may be too thin to supervise) more valuable than it looked: it is
now a robustness lever, not only an accuracy one.

## Caveat

TestSet1 absolute numbers are small (AP 2.98 → 5.46), so an +83% relative gain is
change on a low base and should be quoted with the absolute figures beside it. The
direction is solid — the same ordering appears at every one of the 15 decode cells and
at both the per-split optimum and the common cell — but this is one out-of-domain split
of one benchmark, and it is not evidence about domain shift in general.
