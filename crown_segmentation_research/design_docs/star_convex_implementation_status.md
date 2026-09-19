# Star-convex network implementation status

## Material Passport

- Origin: direct follow-up to `oracle_test_stardist_encoding.md` (PROMOTE
  verdict) — implements the architecture the oracle test justified building.
- Verification Status: CODE-VERIFIED (unit tests + real-image smoke test)
  — **NOT scientifically run**: no training has happened. This mirrors this
  project's own established distinction (see `experiments/g4a_physbound/`'s
  "CODE-VERIFIED; SCIENTIFICALLY UNVERIFIED" status for a prior module).

## What is done

1. **NMS safety check** (real GT, no network): among 1,690 genuinely
   touching/adjacent GT crown pairs across 100 BAM_val images, mean mutual
   IoU is 0.0002 (max 0.0019) — 0% exceed even IoU>=0.1. A standard IoU-NMS
   at typical thresholds (0.3-0.5) will not spuriously suppress a real
   adjacent tree in favor of its neighbour.
2. **`star_convex_targets.py`** — per-pixel object-probability (normalized
   distance-to-background, StarDist convention) and per-pixel K-ray
   boundary-distance targets, computed per-instance on a cropped bounding
   box (not full-image scale). This crop fix was necessary, not optional:
   full-image-scale computation for one 125,629-px instance measured at
   ~99s; cropped, ~4s. Full-image scale would have made real training-data
   generation impractical (~525 GPU-independent CPU-hours estimated for one
   epoch's worth of val-set instances alone).
3. **`star_convex_model.py`** — ResNet50-FPN backbone (same family as the
   frozen G1B Mask R-CNN checkpoint) + two dense conv heads (probability,
   n_rays=32) at stride-4 resolution, bilinearly upsampled to input
   resolution. No RPN/ROI stage — dense per-pixel prediction throughout, so
   this architecture cannot reproduce the proposal-drop failure mode G4D
   diagnosed in Mask R-CNN (by construction, not by measurement yet).
4. **`star_convex_decode.py`** — probability-map peak-finding (local maxima
   above threshold) -> per-peak ray-to-polygon construction -> greedy
   polygon-IoU NMS.
5. **Tests**: 4/4 unit tests on synthetic disks (ray-distance correctness,
   probability-peak correctness, two-disk end-to-end separation, single-disk
   non-fragmentation) all pass.
6. **Real-image smoke test**: full forward pass + decode on one real BAM_val
   2048x2048 image with an *untrained* (random-weight) network — 0.19s
   forward, 0.07s decode, 7 polygons produced (expected to be meaningless
   noise at this stage; this test verifies the pipeline is wired correctly
   end-to-end and runs fast enough, not that it is accurate).

## What is explicitly NOT done

- **No training has run.** There is no loss function, no optimizer, no
  training loop, no checkpoint. `pretrained_backbone=True` is available
  (ImageNet-pretrained ResNet50) but was not exercised in the smoke test
  (used `False` to avoid a network weights download during a shape-check).
- No accuracy claim of any kind exists for this network. Every number in
  this document is either a pure-geometry oracle result (no network) or a
  code/shape/speed check (untrained network).
- Target-generation performance (~4s per large instance, faster for
  smaller/typical ones) has not been profiled across a full training epoch;
  it may still need further optimization (e.g. vectorizing the radius loop,
  or precomputing targets once to disk rather than on-the-fly) before being
  practical for repeated-epoch training at full BAM_train scale (1,439
  images, up to tens of instances each).

## Next step if training is authorized

1. Precompute targets to disk for BAM_train (avoid recomputing per epoch).
2. Loss: BCE or focal loss on probability map + smooth-L1 (or similar) on
   ray distances, masked to foreground only, matching StarDist's published
   loss design (not yet implemented here).
3. Same warm-start convention as the rest of this track: consider
   initializing the backbone from the frozen G1B Mask R-CNN checkpoint's
   backbone weights rather than from scratch or generic ImageNet, since it
   is already fine-tuned on this exact dataset's visual domain.
4. Screen on the same deterministic subset convention as G4A (small
   train/val slice, fixed seed) before any full-scale run.
