"""Phase 0 D3 on LIVECell VAL with the released LIVECell anchor-free model (CenterMask2 V-99-eSE, trained on all of
LIVECell by the dataset authors; task baseline, detection-based family).
Run with the system python3 (torch 2.4) and PYTHONPATH=/home/chung/RS/envs/src/detectron2:/home/chung/RS/envs/src/centermask2:/home/chung/RS/envs/d2deps (detectron2 built in place; no install into the shared environment).

- Config: the authors' model/anchor_free/livecell_config.yaml unchanged, except WEIGHTS = the downloaded checkpoint.
- Score cut-off: the validation F1@0.5 maximum on the same grid as QuebecTrees (d2d3_quebectrees.f1_cut).
- Absorption definition, exclusion of border instances, and size / density standardisation: identical to
  d3_livecell_cellpose.py.

Writes ../results/livecell_d3_centermask.json."""
import json, sys, zipfile
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/chung/RS/envs/src/centermask2")
from absorption import image_rows, summarise, D_TOUCH  # noqa: E402
from d2d3_quebectrees import f1_cut  # noqa: E402
LC = Path("/home/chung/RS/data/livecell"); MD = LC / "models/anchor_free"


def main():
    import cv2, torch
    from centermask.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg(); cfg.merge_from_file(str(MD / "livecell_config.yaml"))
    cfg.MODEL.WEIGHTS = str(MD / "LIVECell_anchor_free_model.pth"); cfg.MODEL.DEVICE = "cuda"; cfg.freeze()
    pred = DefaultPredictor(cfg)
    d = json.load(open(LC / "annotations/livecell_coco_val.json"))
    anns = d["annotations"].values() if isinstance(d["annotations"], dict) else d["annotations"]
    by = {}
    for a in anns: by.setdefault(a["image_id"], []).append(a)
    imgs = list(d["images"].values() if isinstance(d["images"], dict) else d["images"])
    zf = zipfile.ZipFile(LC / "images.zip"); names = {Path(n).name: n for n in zf.namelist() if n.endswith(".tif")}
    per = []
    for k, im in enumerate(imgs):
        H, W = im["height"], im["width"]; gt = np.zeros((H, W), np.int32); n = 0
        for a in by.get(im["id"], []):
            seg = a["segmentation"]; rle = mu.merge(mu.frPyObjects(seg, H, W)) if isinstance(seg, list) else seg
            m = mu.decode(rle).astype(bool)
            if m.any(): n += 1; gt[m] = n
        g = cv2.imdecode(np.frombuffer(zf.read(names[im["file_name"]]), np.uint8), cv2.IMREAD_UNCHANGED)
        img = np.stack([g] * 3, -1) if g.ndim == 2 else g
        with torch.no_grad(): inst = pred(img)["instances"].to("cpu")
        ms, sc = inst.pred_masks.numpy(), inst.scores.numpy()
        dets = [(mu.encode(np.asfortranarray(ms[i].astype(np.uint8))), float(sc[i])) for i in range(len(sc))]
        per.append((im, gt, dets))
        if k % 50 == 0: print(k, len(imgs), len(dets), flush=True)
    cut = f1_cut([(gt, dets) for _, gt, dets in per]); print("cut", cut, flush=True)
    rows = []
    for im, gt, dets in per:
        n = int(gt.max()); excl = np.zeros(n + 1, bool); b = np.unique(np.concatenate([gt[0], gt[-1], gt[:, 0], gt[:, -1]])); excl[b[b > 0]] = True
        for x in image_rows(gt, excl, [(r, s) for r, s in dets if s >= cut]):
            x["img"] = im["id"]; x["ctype"] = im["file_name"].split("_")[0]; rows.append(x)
    import pickle; pickle.dump(rows, open(HERE.parent / "results/rows_livecell_centermask.pkl", "wb"))
    area = np.array([x["area"] for x in rows]); dens = np.array([x["density"] for x in rows])
    se, de = np.quantile(area, [.2, .4, .6, .8]), np.quantile(dens, [1 / 3, 2 / 3])
    out = {"model": "LIVECell anchor-free CenterMask2 V-99-eSE (ALL)", "cut": cut, "size_edges_px": se.tolist(), "density_edges": de.tolist(),
           "all": {d_: summarise([dict(x) for x in rows], "img", se, de, d=d_) for d_ in D_TOUCH}, "by_ctype": {}}
    for ct in sorted({x["ctype"] for x in rows}):
        rr = [dict(x) for x in rows if x["ctype"] == ct]; a2 = np.array([x["area"] for x in rr]); d2 = np.array([x["density"] for x in rr])
        out["by_ctype"][ct] = summarise(rr, "img", np.quantile(a2, [.2, .4, .6, .8]), np.quantile(d2, [1 / 3, 2 / 3]), d=2)
    json.dump(out, open(HERE.parent / "results/livecell_d3_centermask.json", "w"), indent=1, default=float)
    for d_, s in out["all"].items():
        st = s["standardised"]; print("d", d_, {k: (v["n"], None if v["rate"] is None else round(v["rate"], 3)) for k, v in s["strata"].items()}, "std diff", st["diff"], st["diff_ci"], flush=True)
    for ct, v in out["by_ctype"].items():
        print(ct, {k: (v2["n"], None if v2["rate"] is None else round(v2["rate"], 3)) for k, v2 in v["strata"].items()}, {k: (round(v2, 3) if isinstance(v2, float) else v2) for k, v2 in v["standardised"].items()}, flush=True)


if __name__ == "__main__":
    main()
