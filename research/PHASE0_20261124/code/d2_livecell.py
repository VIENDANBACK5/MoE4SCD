"""Phase 0 D2 on LIVECell annotations only (train + val; the test split is not read).

Same definitions as QuebecTrees (absorption.image_rows with no detections): neighbour asymmetry r at d = 1, 2, 5 px; area;
density. Instances touching the image border are excluded (cut objects).

Per cell type and overall:
- counts per stratum;
- area quantiles (px; LIVECell's own size cut-offs are 500 / 1,500 px).

Writes ../results/livecell_d2.json."""
import json, sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from absorption import image_rows, D_TOUCH  # noqa: E402
ANN = Path("/home/chung/RS/data/livecell/annotations")


def rows_for(split):
    d = json.load(open(ANN / f"livecell_coco_{split}.json"))
    by = {}
    for a in (d["annotations"].values() if isinstance(d["annotations"], dict) else d["annotations"]):
        by.setdefault(a["image_id"], []).append(a)
    imgs = d["images"].values() if isinstance(d["images"], dict) else d["images"]
    out = []
    for im in imgs:
        H, W = im["height"], im["width"]; gt = np.zeros((H, W), np.int32); k = 0
        for a in by.get(im["id"], []):
            seg = a["segmentation"]
            rle = mu.merge(mu.frPyObjects(seg, H, W)) if isinstance(seg, list) else seg
            m = mu.decode(rle).astype(bool)
            if not m.any(): continue
            k += 1; gt[m] = k     # overlapping polygons: the later one wins (rare; reported below)
        n = int(gt.max()); excl = np.zeros(n + 1, bool)
        border = np.unique(np.concatenate([gt[0], gt[-1], gt[:, 0], gt[:, -1]])); excl[border[border > 0]] = True
        ctype = im["file_name"].split("_")[0]
        for x in image_rows(gt, excl, []):
            out.append({"img": im["id"], "ctype": ctype, "area": x["area"], "r": x["r"], "density": x["density"]})
    return out


def summary(rows):
    s = {"n": len(rows)}
    if not rows: return s
    a = np.array([x["area"] for x in rows]); s["area_px_quantiles"] = {q: float(np.quantile(a, q)) for q in (.05, .25, .5, .75, .95)}
    for d in D_TOUCH:
        r = np.array([x["r"][d] for x in rows])
        s[f"d{d}"] = {"non_touching": int((r == 0).sum()), "balanced": int(((r > 0) & (r < 2)).sum()),
                      "moderate": int(((r >= 2) & (r < 4)).sum()), "strong": int((r >= 4).sum()), "strong_share": float((r >= 4).mean()),
                      "touching_share": float((r > 0).mean())}
    return s


def main():
    res = {}
    for split in ("val", "train"):
        rows = rows_for(split); print(split, len(rows), flush=True)
        res[split] = {"all": summary(rows)}
        for ct in sorted({x["ctype"] for x in rows}):
            res[split][ct] = summary([x for x in rows if x["ctype"] == ct])
        print(json.dumps({ct: (v["n"], v.get("d2", {}).get("strong"), round(v.get("d2", {}).get("strong_share", 0), 3)) for ct, v in res[split].items()}), flush=True)
    json.dump(res, open(HERE.parent / "results/livecell_d2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
