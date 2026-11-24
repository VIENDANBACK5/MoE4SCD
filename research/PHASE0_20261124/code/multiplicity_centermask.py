"""Post-hoc check of an unverified claim in RESULTS_D3_QUERY_CELLSAM.md ("CenterMask2's absorption is mostly pairwise").
For every CenterMask2-absorbed cell (d = 2, cut 0.55, thick >= 3), count how many GT cells the absorbing prediction
covers by >= 50%. The absorbing prediction is the one covering >= 50% of the cell with the largest coverage count.
Same counting as cellsam_absorbing_mask_multiplicity.json; there the CellSAM label-map instance is used.
Run with system python3 + the detectron2 PYTHONPATH, as d3_livecell_centermask.py.
Writes ../results/centermask_absorbing_mask_multiplicity.json."""
import json, pickle, sys, zipfile
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, "/home/chung/RS/envs/src/centermask2")
LC = Path("/home/chung/RS/data/livecell"); MD = LC / "models/anchor_free"; CUT = 0.55


def main():
    import cv2, torch
    from centermask.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg(); cfg.merge_from_file(str(MD / "livecell_config.yaml")); cfg.MODEL.WEIGHTS = str(MD / "LIVECell_anchor_free_model.pth"); cfg.MODEL.DEVICE = "cuda"; cfg.freeze()
    pred = DefaultPredictor(cfg)
    rows = pickle.load(open(HERE.parent / "results/rows_livecell_centermask.pkl", "rb")); absd = {}
    for x in rows:
        if x["M"][2] and x["thick"] >= 3: absd.setdefault(x["img"], []).append(x["id"])
    d = json.load(open(LC / "annotations/livecell_coco_val.json"))
    anns = d["annotations"].values() if isinstance(d["annotations"], dict) else d["annotations"]; by = {}
    for a in anns: by.setdefault(a["image_id"], []).append(a)
    imgs = list(d["images"].values() if isinstance(d["images"], dict) else d["images"])
    zf = zipfile.ZipFile(LC / "images.zip"); names = {Path(n).name: n for n in zf.namelist() if n.endswith(".tif")}
    ncov = []; byct = {}
    for im in imgs:
        if im["id"] not in absd: continue
        H, W = im["height"], im["width"]; gt = np.zeros((H, W), np.int32); n = 0
        for a in by.get(im["id"], []):
            seg = a["segmentation"]; rle = mu.merge(mu.frPyObjects(seg, H, W)) if isinstance(seg, list) else seg
            m = mu.decode(rle).astype(bool)
            if m.any(): n += 1; gt[m] = n
        g = cv2.imdecode(np.frombuffer(zf.read(names[im["file_name"]]), np.uint8), cv2.IMREAD_UNCHANGED)
        with torch.no_grad(): inst = pred(np.stack([g] * 3, -1))["instances"].to("cpu")
        ms = inst.pred_masks.numpy()[inst.scores.numpy() >= CUT]
        if not len(ms): continue
        grle = [mu.encode(np.asfortranarray((gt == k).astype(np.uint8))) for k in range(1, n + 1)]
        prle = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in ms]
        cov = mu.iou(grle, prle, [1] * len(prle))                     # |g ∩ P| / |g|
        cnt = (cov >= .5).sum(0)
        for gid in absd[im["id"]]:
            own = np.where(cov[gid - 1] >= .5)[0]
            if len(own): c = int(cnt[own].max()); ncov.append(c); byct.setdefault(im["file_name"].split("_")[0], []).append(c)
    ncov = np.array(ncov)
    out = {"n": len(ncov), "2": float((ncov == 2).mean()), "3-4": float(((ncov >= 3) & (ncov <= 4)).mean()), "5-9": float(((ncov >= 5) & (ncov <= 9)).mean()),
           ">=10": float((ncov >= 10).mean()), "median": float(np.median(ncov)), "by_ctype_median": {k: float(np.median(v)) for k, v in byct.items()}}
    print(out); json.dump(out, open(HERE.parent / "results/centermask_absorbing_mask_multiplicity.json", "w"), indent=1)


if __name__ == "__main__":
    main()
