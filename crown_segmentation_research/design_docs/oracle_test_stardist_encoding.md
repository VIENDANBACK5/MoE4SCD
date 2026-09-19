# Oracle test: star-convex K-ray polygon representation — PROMOTE

## Material Passport

- Origin: direct follow-up to `oracle_test_hv_decode_result.md` (H/V KILL).
  Before implementing StarDist/CPP-Net, tested the star-convexity assumption
  itself on real BAM GT, applying the same "verify the representation before
  building the network" discipline that caught the H/V failure.
- Verification Status: VERIFIED on real BAM_val GT polygons (500-instance
  sample, seed 0).

## Test 1 — is the star-convexity assumption even plausible here?

For each polygon, cast 36 rays from the centroid and check whether each ray
crosses the polygon boundary exactly once (star-convex from that point) or
more than once (not star-convex from the centroid).

**Result: 442/498 (88.8%) of real BAM crowns are star-convex from their own
centroid.** This is a much weaker, more easily satisfied condition than what
the H/V representation actually needed (which requires smooth *monotonic*
per-axis behavior, not just single-crossing star-convexity) — this
difference in strictness is the likely reason StarDist-style encoding
survives where H/V did not, even though both are nominally in the same
"single-center representation" family flagged as a risk in
`literature/synthesis/00_SYNTHESIS.md`.

## Test 2 — how well does K-ray encoding reconstruct the true polygon?

For each polygon, cast K evenly-spaced rays from the centroid, take the
*farthest* boundary crossing along each ray (defined this way so the test
also covers non-star-convex instances with a reasonable rule, not just the
88.8% that are exactly star-convex), reconstruct a polygon from the K
resulting points, and measure IoU against the true GT polygon.

| K (rays) | mean IoU | median IoU | % IoU>=0.5 | % IoU>=0.7 |
|---:|---:|---:|---:|---:|
| 16 | 0.917 | 0.937 | 99.6% | 98.6% |
| 32 | 0.965 | 0.978 | 99.6% | 99.6% |
| 64 | 0.984 | 0.993 | 99.6% | 99.6% |

For reference, this is well above the current Mask R-CNN baseline's matched
IoU (0.7585-0.7951 depending on split) — though not a fair direct
comparison yet, since this is an *encoding* ceiling test (perfect centroid
+ perfect boundary access from GT), not a detection-and-prediction result;
a trained network's predicted rays will have error the oracle does not.

## Verdict

```text
STAR_CONVEX_K_RAY_REPRESENTATION = PROMOTE (oracle-level, K=32 recommended)
```

Unlike H/V (`oracle_test_hv_decode_result.md`), this representation's
encoding ceiling is high enough on real BAM crowns to justify building a
prediction network on top of it. K=32 is a reasonable default (matches
common StarDist practice) balancing reconstruction fidelity (0.965 mean IoU)
against head output size.

## Test 3 — robustness to centroid localization error

A trained network must predict its own centroid, not use the true one. Perturbed
the true centroid with Gaussian noise (std expressed as a fraction of the
instance's equivalent radius `sqrt(area/pi)`) before re-running the K=32
encode/decode:

| Centroid noise (% of equiv. radius) | mean IoU | % IoU>=0.5 |
|---:|---:|---:|
| 0% | 0.969 | 100.0% |
| 5% | 0.968 | 100.0% |
| 10% | 0.968 | 100.0% |
| 20% | 0.966 | 100.0% |
| 30% | 0.959 | 100.0% |

**The representation is robust to centroid error** -- even at 30% noise
(a large, realistic-worst-case offset), mean IoU only drops 0.010 and every
instance stays above IoU 0.5. This substantially de-risks concern (b) from
the section above: a network's centroid-prediction error is unlikely to be
the dominant error source for this representation, so effort should
prioritize per-ray distance regression quality over centroid precision.

## What this does not yet establish

- This tests **encoding fidelity only** — centroid + rays from a known true
  polygon. It does not test: (a) a trained network's actual per-ray
  regression error, (b) centroid *localization* error (this test used the
  true centroid; a network must also predict it, revisiting the same
  "centroid-pixel-alone lacks context" problem CPP-Net targets), (c) NMS
  behavior when many predicted star-convex polygons overlap in a dense
  canopy, or (d) the 11.2% of crowns that are not star-convex from their
  centroid at all -- their IoU here reflects a specific reconstruction rule
  (farthest crossing), not a validated way to handle them at decode time in
  a live pipeline.
- The multi-seed merge module (Section 4, already validated with a real
  positive result on BAM_test2) remains relevant regardless of this result
  -- star-convexity assumes one center per instance, and large/multi-lobed
  crowns can still violate that even when nominally star-convex from a
  single point chosen as centroid.

## Suggested next step

CPP-Net's context-sampling fix (already scoped for the classifier-confidence
problem in `method_design_crown_confidence_v2.md` Section 3) is now doubly
motivated: it addresses both the original classifier-score problem and the
star-convex encoding's centroid-localization sensitivity noted in (b) above.
Before training a full network, the next oracle-adjacent check worth running
cheaply: perturb the true centroid by a few pixels (simulating realistic
localization error) and re-measure reconstruction IoU, to see how sensitive
the K=32 result is to centroid error specifically -- this determines how
much of the eventual network's effort must go toward centroid precision
versus ray-distance regression.
