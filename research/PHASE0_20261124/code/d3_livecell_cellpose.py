"""Phase 0 D3 on LIVECell VAL with the released Cellpose 3 'livecell_cp3' model (mechanism baseline: flow family).
Run with /home/chung/RS/envs/phase0/bin/python (cellpose 3.1.1.2, isolated venv).

- Default model settings: diameter = the model's diam_mean, channels [0, 0].
- Cellpose has no per-instance score, so every mask gets score 1. Masks do not overlap, so one-to-one matching does not
  depend on the order.
- Same absorption definition as QuebecTrees. Instances touching the image border are excluded.
- The size / density standardisation uses quantiles computed over all LIVECell val instances.
- Results are reported overall and per cell type.

Writes ../results/livecell_d3_cellpose.json."""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from absorption import image_rows, summarise, D_TOUCH  # noqa: E402
LC = Path("/home/chung/RS/data/livecell")


def main():
    from cellpose import models
    import cv2
    d = json.load(open(LC / "annotations/livecell_coco_val.json"))
    anns = d["annotations"].values() if isinstance(d["annotations"], dict) else d["annotations"]
    by = {}
    for a in anns: by.setdefault(a["image_id"], []).append(a)
    imgs = list(d["images"].values() if isinstance(d["images"], dict) else d["images"])
    zf = zipfile.ZipFile(LC / "images.zip"); names = {Path(n).name: n for n in zf.namelist() if n.endswith(".tif")}
    model = models.CellposeModel(gpu=True, model_type="livecell_cp3"); print("diam_mean", model.diam_mean, flush=True)
    rows = []
    for k, im in enumerate(imgs):
        H, W = im["height"], im["width"]; gt = np.zeros((H, W), np.int32); n = 0
        for a in by.get(im["id"], []):
            seg = a["segmentation"]; rle = mu.merge(mu.frPyObjects(seg, H, W)) if isinstance(seg, list) else seg
            m = mu.decode(rle).astype(bool)
            if m.any(): n += 1; gt[m] = n
        excl = np.zeros(n + 1, bool); b = np.unique(np.concatenate([gt[0], gt[-1], gt[:, 0], gt[:, -1]])); excl[b[b > 0]] = True
        img = cv2.imdecode(np.frombuffer(zf.read(names[im["file_name"]]), np.uint8), cv2.IMREAD_UNCHANGED)
        masks, _, _ = model.eval(img, channels=[0, 0], diameter=None)
        dets = [(mu.encode(np.asfortranarray((masks == i).astype(np.uint8))), 1.0) for i in range(1, int(masks.max()) + 1)]
        ct = im["file_name"].split("_")[0]
        for x in image_rows(gt, excl, dets):
            x["img"] = im["id"]; x["ctype"] = ct; rows.append(x)
        if k % 50 == 0: print(k, len(imgs), len(rows), flush=True)
    import pickle; pickle.dump(rows, open(HERE.parent / "results/rows_livecell_cellpose.pkl", "wb"))
    area = np.array([x["area"] for x in rows]); dens = np.array([x["density"] for x in rows])
    se, de = np.quantile(area, [.2, .4, .6, .8]), np.quantile(dens, [1 / 3, 2 / 3])
    out = {"model": "cellpose3 livecell_cp3", "size_edges_px": se.tolist(), "density_edges": de.tolist(),
           "all": {d_: summarise([dict(x) for x in rows], "img", se, de, d=d_) for d_ in D_TOUCH}, "by_ctype": {}}
    for ct in sorted({x["ctype"] for x in rows}):
        rr = [dict(x) for x in rows if x["ctype"] == ct]; a2 = np.array([x["area"] for x in rr]); d2 = np.array([x["density"] for x in rr])
        out["by_ctype"][ct] = summarise(rr, "img", np.quantile(a2, [.2, .4, .6, .8]), np.quantile(d2, [1 / 3, 2 / 3]), d=2)
    json.dump(out, open(HERE.parent / "results/livecell_d3_cellpose.json", "w"), indent=1, default=float)
    s = out["all"][2]; print(json.dumps({"strata": s["strata"], "standardised": s["standardised"], "consequence": s["consequence"]}, indent=1, default=float))
    for ct, v in out["by_ctype"].items():
        print(ct, {k: (v2["n"], None if v2["rate"] is None else round(v2["rate"], 3)) for k, v2 in v["strata"].items()}, {k: (round(v2, 3) if isinstance(v2, float) else v2) for k, v2 in v["standardised"].items()})


if __name__ == "__main__":
    main()
