"""Preview for D3q SelvaMask: 8 random (seed 0) GT crowns absorbed by Mask2Former s42 (d = 2).
Each crop is 500 px (22.5 m), shown at 0.64×.
- Left half: Mask2Former s42. Right half: R5 Mask R-CNN on the same crop.
- Yellow: GT outlines. Red: the absorbed crown. Magenta: the prediction covering it.
- Each title says whether R5 also absorbs it.
Coarse eye judgement only."""
import json, pickle, sys
from pathlib import Path
import numpy as np, cv2
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from d3q_selvamask_families import TILES_DIR, systems, load, rle
R = HERE.parent / "results"
rows = {k: {(x["tile"], x["id"]): x for x in pickle.load(open(R / f"rows_selvamask_{k}.pkl", "rb"))} for k in ("Mask2Former_s42", "R5_MaskRCNN")}
ab = [k for k, x in rows["Mask2Former_s42"].items() if x["M"][2]]; rng = np.random.default_rng(0); pick = [ab[i] for i in rng.choice(len(ab), 8, replace=False)]
S = systems(); tiles = []
for t, g in pick:
    refs = json.load(open(TILES_DIR / t / "references.json"))["references"]; H, W = refs[0]["rle"]["size"]; gt = np.zeros((H, W), np.int32)
    for k, r in enumerate(refs, 1): gt[mu.decode(rle(r["rle"])).astype(bool)] = k
    rgb0 = cv2.cvtColor(cv2.imread(str(TILES_DIR / t / "RGB.png")), cv2.COLOR_BGR2RGB)[..., ::-1].copy()
    own = gt == g; cy, cx = np.argwhere(own).mean(0).astype(int); y0, x0 = np.clip(cy - 250, 0, H - 500), np.clip(cx - 250, 0, W - 500)
    halves = []
    for name in ("Mask2Former_s42", "R5_MaskRCNN"):
        rgb = rgb0.copy(); dets = load(S[name][t]); m = [mu.decode(d[0]).astype(bool) for d in dets]
        def ol(mask, col, th):
            cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(rgb, cs, -1, col, th)
        for v in np.unique(gt[y0:y0 + 500, x0:x0 + 500])[1:]: ol(gt == v, (0, 220, 220), 2)
        covs = [(mm & own).sum() / own.sum() for mm in m]
        if covs and max(covs) >= .5: ol(m[int(np.argmax(covs))], (255, 0, 255), 3)
        ol(own, (0, 0, 255), 3)
        c = cv2.resize(rgb[y0:y0 + 500, x0:x0 + 500], (320, 320))
        cv2.putText(c, f"{t} {name[:6]} {'ABS' if rows[name][(t, g)]['M'][2] else 'ok'}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1); halves.append(c)
    tiles.append(np.hstack(halves))
grid = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, 8, 2)])
p = HERE.parent / "previews" / "selvamask_m2f_vs_r5_absorbed.png"; cv2.imwrite(str(p), grid); print(p)
