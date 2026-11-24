"""Feasibility count: labels of Mask R-CNN s42 candidates (score >= the IEO validation F1 cut) on the existing
train (35 tiles × 4 quadrants) and validation crops.
Label of candidate P = number of GT g with |g ∩ P| / |g| >= 0.5:
- 0 = FP;
- 1 = KEEP;
- >= 2 = REFORM (and >= 3 = multi)."""
import json, sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
REPO = Path(__file__).resolve().parents[3]; PREP = REPO / "data/itc_benchmarks/canopyrs_quebectrees/prepared"
IEO = REPO / "research/ITC_IDENTITY_ERROR_ORIGIN_V1_20261119"; Q = 1666
cut = json.load(open(IEO / "results.json"))["cutoffs"]["mrcnn_s42"]; P = json.load(open(IEO / "preds/mrcnn_s42.json"))["crops"]
def gt_of(split, cid):
    if split == "validation":
        coco = json.load(open(PREP / "validation_coco.json")); info = {f"v{i['id']}": i for i in coco["images"]}[cid]
        return np.load(PREP / "validation" / f"{Path(info['file_name']).stem}.npz")["label"]
    tid, q = cid[1:].split("_"); coco = json.load(open(PREP / "train_coco.json")); info = {str(i["id"]): i for i in coco["images"]}[tid]
    lab = np.load(PREP / "train" / f"{Path(info['file_name']).stem}.npz")["label"]; qy, qx = int(q[0]), int(q[1])
    return lab[qy * Q:(qy + 1) * Q, qx * Q:(qx + 1) * Q]
cnt = {"validation": np.zeros(4, int), "train": np.zeros(4, int)}
for cid, c in P.items():
    gt = gt_of(c["split"], cid); ids = [g for g in np.unique(gt) if g]
    dets = [{"size": d["size"], "counts": d["counts"].encode()} for d in c["dets"] if d["score"] >= cut]
    if not dets or not ids: continue
    grle = [mu.encode(np.asfortranarray((gt == g).astype(np.uint8))) for g in ids]
    cov = mu.iou(grle, dets, [1] * len(dets))               # |g ∩ P| / |g|
    k = (cov >= .5).sum(0); cnt[c["split"]] += [(k == 0).sum(), (k == 1).sum(), (k == 2).sum(), (k >= 3).sum()]
print("cut", cut)
for s, v in cnt.items(): print(s, "FP", v[0], "KEEP", v[1], "REFORM2", v[2], "REFORM>=3", v[3], "reform share", round((v[2] + v[3]) / v.sum(), 3))
