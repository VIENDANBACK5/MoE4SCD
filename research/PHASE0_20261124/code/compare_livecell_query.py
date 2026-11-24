"""G3 comparison table on LIVECell val (d = 2). Same GT, same border exclusion, same instances for every model, so the
per-instance absorption rate needs no reweighting. Also reports the rate inside size-quintile × density-tertile cells
(edges from the shared GT), which compares models on identical instance populations. All-instance and thick >= 3
(post-hoc sliver filter) versions are given.

Writes ../results/livecell_query_comparison.json and also writes previews (CellSAM label map; 8 random thick >= 3
absorbed cells, seed 0; crops of 160 px at 2×):
- yellow: GT outlines;
- red: the absorbed cell;
- magenta: the CellSAM label-map instance that covers it;
- each crop's title gives the decomposition category.

Run with /home/chung/RS/envs/phase0/bin/python."""
import json, pickle, sys, zipfile
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
R = HERE.parent / "results"; LC = Path("/home/chung/RS/data/livecell")
MODELS = {"Cellpose3 livecell_cp3 (in-domain)": "rows_livecell_cellpose.pkl", "CenterMask2 (in-domain)": "rows_livecell_centermask.pkl",
          "CellSAM labelmap (zero-shot)": "rows_livecell_cellsam_labelmap.pkl", "CellSAM instances, F1 cut (zero-shot)": "rows_livecell_cellsam_instances.pkl",
          "CellSAM GT-box prompts (oracle hypotheses)": "rows_livecell_cellsam_gtbox.pkl"}


def stats(rows, se, de):
    n = len(rows); hit = sum(x["hit"] for x in rows); M = sum(x["M"][2] for x in rows); miss = n - hit
    cells = {}
    for x in rows:
        c = (int(np.searchsorted(se, x["area"], side="right")), int(np.searchsorted(de, x["density"], side="right")))
        v = cells.setdefault(c, [0, 0]); v[0] += 1; v[1] += x["M"][2]
    return {"n": n, "recall": hit / n, "absorption_rate": M / n, "share_of_misses": M / miss if miss else None,
            "strong_absorption_rate": sum(x["M"][2] for x in rows if x["r"][2] >= 4) / n,
            "cells": {f"{a}{b}": v[1] / v[0] for (a, b), v in sorted(cells.items())}}


def main():
    rows = {k: pickle.load(open(R / f, "rb")) for k, f in MODELS.items()}
    key = lambda x: (x["img"], x["id"])
    base = {key(x) for x in rows["CenterMask2 (in-domain)"]}
    for k, rr in rows.items(): assert {key(x) for x in rr} == base, f"instance set differs: {k}"
    ref = rows["CenterMask2 (in-domain)"]
    out = {}
    for tag, filt in (("all", lambda x: True), ("thick>=3", lambda x: x["thick"] >= 3)):
        rr0 = [x for x in ref if filt(x)]
        se = np.quantile([x["area"] for x in rr0], [.2, .4, .6, .8]); de = np.quantile([x["density"] for x in rr0], [1 / 3, 2 / 3])
        out[tag] = {k: stats([x for x in rr if filt(x)], se, de) for k, rr in rows.items()}
        print(f"\n== {tag} (n = {len(rr0)})")
        print(f"{'model':45s} recall  abs_rate  share_miss  strong_rate")
        for k, s in out[tag].items():
            print(f"{k:45s} {s['recall']:.3f}  {s['absorption_rate']:.3f}     {s['share_of_misses']:.3f}       {s['strong_absorption_rate']:.3f}")
        print("cell-wise absorption rate (size quintile, density tertile):")
        for c in out[tag]["CenterMask2 (in-domain)"]["cells"]:
            print("  ", c, "  ".join(f"{out[tag][k]['cells'][c]:.3f}" for k in MODELS))
    # per cell type, labelmap vs CenterMask2, thick >= 3
    out["by_ctype_thick"] = {}
    for ct in sorted({x["ctype"] for x in ref}):
        out["by_ctype_thick"][ct] = {k: stats([x for x in rows[k] if x["ctype"] == ct and x["thick"] >= 3], np.array([np.inf] * 4), np.array([np.inf] * 2))
                                     for k in ("CenterMask2 (in-domain)", "CellSAM labelmap (zero-shot)")}
        a, b = out["by_ctype_thick"][ct].values()
        print(ct, f"CM2 recall {a['recall']:.3f} abs {a['absorption_rate']:.3f} | CellSAM recall {b['recall']:.3f} abs {b['absorption_rate']:.3f} share {b['share_of_misses'] or 0:.3f}")
    json.dump(out, open(R / "livecell_query_comparison.json", "w"), indent=1)
    preview()


def preview():
    import cv2
    dec = [x for x in pickle.load(open(R / "cellsam_decomposition.pkl", "rb")) if x["thick"] >= 3]
    rng = np.random.default_rng(0); pick = [dec[i] for i in rng.choice(len(dec), min(8, len(dec)), replace=False)]
    d = json.load(open(LC / "annotations/livecell_coco_val.json")); imgs = {im["id"]: im for im in (d["images"].values() if isinstance(d["images"], dict) else d["images"])}
    anns = d["annotations"].values() if isinstance(d["annotations"], dict) else d["annotations"]; by = {}
    for a in anns: by.setdefault(a["image_id"], []).append(a)
    from d3_livecell_cellsam import gt_raster
    zf = zipfile.ZipFile(LC / "images.zip"); names = {Path(n).name: n for n in zf.namelist() if n.endswith(".tif")}
    tiles = []
    for x in pick:
        im = imgs[x["img"]]; gt = gt_raster(im, by); lab = pickle.load(open(R / "cellsam_cache" / f"{im['id']}.pkl", "rb"))["labelmap"]
        g = cv2.imdecode(np.frombuffer(zf.read(names[im["file_name"]]), np.uint8), cv2.IMREAD_UNCHANGED)
        rgb = cv2.cvtColor(cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        def outline(m, col, th=1):
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(rgb, cs, -1, col, th)
        for v in np.unique(gt)[1:]: outline(gt == v, (0, 220, 220))
        own = gt == x["id"]; vals, cnt = np.unique(lab[own], return_counts=True); cnt = cnt[vals > 0]; vals = vals[vals > 0]
        if len(vals): outline(lab == vals[np.argmax(cnt)], (255, 0, 255), 1)
        outline(own, (0, 0, 255), 1)
        cy, cx = np.argwhere(own).mean(0).astype(int); H, W = gt.shape
        y0, x0 = np.clip(cy - 80, 0, H - 160), np.clip(cx - 80, 0, W - 160)
        crop = cv2.resize(rgb[y0:y0 + 160, x0:x0 + 160], (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.putText(crop, f"{x['ctype']} {x['cat']} r={x['r']:.1f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        tiles.append(crop)
    while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    p = HERE.parent / "previews" / "livecell_cellsam_absorbed_thick3.png"; p.parent.mkdir(exist_ok=True); cv2.imwrite(str(p), grid); print("preview", p)


if __name__ == "__main__":
    main()
