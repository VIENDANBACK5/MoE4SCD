# Literature notes

Started 2026-09-19, after H8 — later than it should have been. I entered an existing
project and went straight to experiments without ever establishing what the benchmark's
published numbers are. Two findings below materially change how earlier results should
be scoped.

---

## 1. Troles, Schmid, Fan & Tian (2024) — BAMFORESTS dataset paper

*Remote Sensing* 16(11), 1935. https://www.mdpi.com/2072-4292/16/11/1935
(open PDF: https://elib.dlr.de/205658/1/remotesensing-16-01935-v2_opt.pdf)

27,160 delineated crowns over 105 ha of UAV imagery, four AOIs within 20 km of Bamberg.
GSD 5-7 cm. Published in COCO format at two tile sizes. **It is a pure dataset paper —
it reports no instance-segmentation baselines**, so it cannot be used for comparison.

**The finding that matters for this project — TestSet1 is a deliberate domain shift.**
Quoting the split design directly:

> "Test-Set-1 (15 ha, 14%) solely consists of the complete Hain-AOI. It is 5-20 km away
> from the other AOIs and is of **severely different structure**. Furthermore, it was
> captured with **another UAV and sensor**. All of these factors are good for estimating
> the real-world performance of methods."

> "we define Test-Set-2 (14 ha, 13%), which is taken from the **same AOIs** as the
> training and validation set."

Additional differences for Hain/TestSet1: it is a **city park**, not managed mixed
forest; flown on a different date (3 Aug vs 5-6 Jul); flown **after sunrise into early
midday** rather than at midday, so illumination and shadow geometry differ; and flown
with a DJI Phantom 4 at 85 m AGL rather than a Trinity F90+ at 120 m AGL.

**Consequence for H8.** The collapse I measured on TestSet1 (COCO AP 2.98 vs 23.85
in-domain, TestSet2 21.29) is not an anomaly I discovered — it is the benchmark's
designated out-of-domain test doing exactly what it was built to do, and the model
failing it. My crown-diameter statistic (D = 314 px vs 211 px) is one *symptom* of a
documented multi-factor shift (site structure, sensor, platform, altitude, date,
time of day), not the cause. **H8's analysis over-attributed the collapse to scale
alone and must be corrected.**

This also connects to the project's own standing constraint in `CLAUDE.md`: *"Always
evaluate the Generalization Gap (In-domain vs Out-domain)."* TestSet2 vs TestSet1 is
precisely that pairing, and the gap is roughly 7x in AP.

---

## 2. TreeCoG — over-segmentation plus GNN contour merging

Reported on BAMFORESTS: **AP 53.21, AP@50 73.14, AP@70 43.23.**
Method: a contour-detection network deliberately **over-segments**, then contours are
merged by a **graph neural network** over contour features.

**This is the same paradigm as my H2, and it is the published state of the art.**
H2 tested "over-seed then merge" and I wrote that the framework "has a ceiling of ~70"
based on an oracle merge reaching 69.74 F1@50. **That conclusion was scoped too widely.**
What the oracle bounded was merging *of my watershed basins* — basins produced by a
boundary head whose mean response on true seams was 0.105. It bounds my instantiation,
not the paradigm. TreeCoG differs in both components that mattered:
- its over-segmentation comes from a dedicated **contour network**, not from apex NMS
  flooded against a collapsed ridge field;
- its merge criterion is **learned** (GNN over contour features), not a threshold on a
  single scalar, which is exactly the failure mode H2 and H5 diagnosed (no seam's mean
  B ever reached 0.35; the best threshold was the smallest one swept).

H6 is consistent with this reading: strengthening the ridge changed merging from
mildly helpful to actively harmful, i.e. the value of merging depends entirely on the
quality of the basins it is given.

## 3. FG-TreeSeg — flow-guided segmentation without instance annotations

arXiv 2602.00470. Reports **mAP@50 67.31** on BAMFORESTS. Relevant because the repo
already contains a flow paradigm (`methods/tree_flow/`) which scored poorly in the
project's earlier cross-paradigm table (2.48 mAP@50) — worth revisiting whether that
was the paradigm or, as with the watershed, its decode parameters.

---

## Where this project actually stands

| method | AP | AP@50 |
|---|---:|---:|
| TreeCoG (published) | **53.21** | 73.14 |
| FG-TreeSeg (published) | — | 67.31 |
| **this project, best so far** | **26.84** | 58.06 |
| this project, oracle heads / same decoder | 84.48 | 96.00 |

**Caveat on comparability:** the published protocols are not verified to match mine.
I evaluate 1024 px crops taken from the 2048 px COCO tiles, on 60 held-out validation
crops, with confidence taken as peak surface response. Published numbers are presumably
over full official test splits. The numbers should be treated as indicative of a
**large gap**, not as a like-for-like ranking — but a 2x gap in AP is too wide to be
explained by protocol alone.

**Honest position:** this project's contribution is diagnostic, not competitive. It
explains *where* a watershed ITC pipeline loses its accuracy and shows the decoder is
not the limit (84.48 AP with correct heads). It does not currently produce a
state-of-the-art detector, and any write-up must say so plainly.
