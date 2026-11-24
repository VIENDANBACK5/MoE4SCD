"""Build the fixed TRAIN RoI cache. Training data only; validation is never read here.

For every train crop (148 tiles × 4 quadrants), run the frozen Mask R-CNN s42 and keep candidates with score >= 0.45.
Each mask is ANDed with valid, and empty masks are dropped. Labels (common.candidate_labels):
- k = 0: FP, skipped;
- k = 1: KEEP;
- k >= 2: REFORM.

Kept for training:
- every REFORM;
- KEEP with probability P_KEEP = 0.26, drawn from an RNG seeded by crop index. This aims at the protocol's 2:1 KEEP:REFORM
  ratio, using the 35-tile measurement of KEEP/REFORM ≈ 7.65.

The natural label counts of ALL candidates are recorded so the decision can be prior-corrected at evaluation.

Stored per kept candidate (memmaps in ../results/cache/):
- feat fp16 [256,32,32]: RoIAlign of FPN '0' on the RoI;
- mask fp16 [1,32,32]: the candidate;
- tgt fp16 [2,32,32]: the 2 covered GT with the largest |g ∩ P| (KEEP: the 1 GT, then zeros);
- ign uint8 [32,32]: other covered GT and invalid pixels;
- label int8 (1 = REFORM);
- meta: crop id, box, k.
"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from common import crops, load_maskrcnn, maskrcnn_forward, roi_box, crop_resample, roi_features, candidate_labels, R, STUDY  # noqa: E402

P_KEEP, NMAX = 0.26, 20000
OUT = STUDY / "results/cache"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    model, ckpt = load_maskrcnn(42)
    mm = lambda n, shape, dt: np.lib.format.open_memmap(OUT / f"{n}.npy", mode="w+", dtype=dt, shape=(NMAX,) + shape)
    feat, mask, tgt, ign = mm("feat", (256, R, R), np.float16), mm("mask", (1, R, R), np.float16), mm("tgt", (2, R, R), np.float16), mm("ign", (R, R), np.uint8)
    label = np.zeros(NMAX, np.int8); meta = []; nat = np.zeros(3, int); n = 0; t0 = time.time()
    for ci, (cid, img, valid, gt) in enumerate(crops("train")):
        dets, f0 = maskrcnn_forward(model, np.ascontiguousarray(img))
        ms = [m & valid for m, _ in dets]; ms = [m for m in ms if m.any()]
        labs = candidate_labels(gt, ms); rng = np.random.default_rng(ci)
        sel = []
        for j, (k, ids) in enumerate(labs):
            nat[min(k, 2)] += 1
            if k >= 2 or (k == 1 and rng.random() < P_KEEP): sel.append(j)
        if not sel: continue
        boxes = [roi_box(ms[j]) for j in sel]; fr = roi_features(f0, boxes)
        for b, j, fj in zip(boxes, sel, fr):
            assert n < NMAX, "cache overflow"
            k, ids = labs[j]
            feat[n] = fj.numpy(); mask[n, 0] = crop_resample(ms[j].astype(np.float32), b)
            for t, g in enumerate(ids[:2]): tgt[n, t] = crop_resample((gt == g).astype(np.float32), b)
            other = np.isin(gt, ids[2:]) if len(ids) > 2 else np.zeros_like(valid)
            ign[n] = (crop_resample((other | ~valid).astype(np.float32), b) > 0.5).astype(np.uint8)
            label[n] = int(k >= 2); meta.append({"crop": cid, "box": b.tolist(), "k": int(k)}); n += 1
        if ci % 40 == 0: print(ci, cid, "kept", n, "natural FP/KEEP/REFORM", nat.tolist(), f"{time.time() - t0:.0f}s", flush=True)
    for a in (feat, mask, tgt, ign): a.flush()
    np.save(OUT / "label.npy", label[:n])
    json.dump({"n": n, "natural_counts_FP_KEEP_REFORM": nat.tolist(), "p_keep": P_KEEP, "ckpt": ckpt, "meta": meta,
               "seconds": round(time.time() - t0)}, open(OUT / "index.json", "w"))
    print("done n", n, "natural", nat.tolist(), "reform kept", int(label[:n].sum()), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
