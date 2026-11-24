"""D3-query Step 1 (external review, 2026-11-24): query-SELECTION diagnostic for CellSAM on LIVECell val. No training.

Question: of the CellSAM absorptions whose cell already had a box candidate that was scored too low, how many are
recovered by keeping more candidates, and at what cost (false positives, duplicates, AP)?

What changes and what stays fixed:
- The only change is the box-score threshold T on CellFinder's 3,500 raw candidates:
  T ∈ {per-image dynamic default, 0.25, 0.20, 0.15, 0.10, 0.05}.
- Everything else is as in d3_livecell_cellsam.py: predicted-IoU >= 0.5 mask filter, pixel threshold 0.4, same
  embeddings. Every candidate with score >= 0.05 is decoded once and cached.
- CellSAM has no NMS, so lowering T can create duplicates.

Two outputs per T:
- "inst": per-candidate masks (may overlap), scored by box score. This is primary for the sweep.
- "lab": the package's label-map compositing. Candidates are in descending score order and a later (lower-scored) mask
  overwrites an earlier one; holes are filled and objects < 25 px removed.

Metrics per T (d = 2; the absorption metric is the pre-registered one, unchanged):
- number of predictions;
- recall and absorption rate (absorption.image_rows, border GT excluded);
- unmatched predictions (FP);
- duplicates: unmatched predictions with IoU >= 0.5 to some GT;
- COCO mask AP and AP50 (inst only, maxDets 3,500, all GT);
- rescued: GT absorbed in the default label map that are hits at T.

Also: for the default "no_box" absorbed cells, the best raw-box IoU at any score, a sensitivity check on the 0.5
box-matching cut.

Writes ../results/d3q_threshold_sweep.json and ../results/cellsam_cache_lowthr/ (per-image decoded candidates).
Run with /home/chung/RS/envs/phase0/bin/python."""
import json, pickle, sys, time, zipfile
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from absorption import image_rows  # noqa: E402
from d3_livecell_cellsam import Runner, load_val, gt_raster, border_excl, box_iou, LC  # noqa: E402
R = HERE.parent / "results"; C1 = R / "cellsam_cache"; C2 = R / "cellsam_cache_lowthr"
TS = ["dyn", 0.25, 0.20, 0.15, 0.10, 0.05]; FLOOR = 0.05


def decode_low(Rn, g):
    t = Rn.prep(g); H, W = g.shape[:2]
    import torch
    with torch.no_grad(): emb, _ = Rn.m.generate_embeddings(t)
    boxes, sc, thr = Rn.raw_boxes(t); idx = np.where(sc >= FLOOR)[0]
    dec = Rn.decode(emb, boxes[idx], H, W) if len(idx) else []
    return {"idx": idx, "score": sc[idx].astype(np.float32), "thr": float(thr),
            "rle": [mu.encode(np.asfortranarray(m)) for m, _ in dec], "iou": np.array([i for _, i in dec], np.float32)}


def labelmap(rles, H, W):
    from cellSAM.utils import fill_holes_and_remove_small_masks
    lab = np.zeros((H, W), np.int64)
    for j, r in enumerate(rles): lab = np.maximum(lab, mu.decode(r).astype(np.int64) * (j + 1))
    if rles: lab = fill_holes_and_remove_small_masks(lab, min_size=25)
    return [(mu.encode(np.asfortranarray((lab == v).astype(np.uint8))), 1.0) for v in np.unique(lab) if v > 0]


def main():
    imgs, by = load_val(); C2.mkdir(exist_ok=True)
    zf = zipfile.ZipFile(LC / "images.zip"); names = {Path(x).name: x for x in zf.namelist() if x.endswith(".tif")}
    Rn = None; t0 = time.time()
    absorbed0 = {(x["img"], x["id"]) for x in pickle.load(open(R / "rows_livecell_cellsam_labelmap.pkl", "rb")) if x["M"][2]}
    acc = {(v, str(T)): {"n_pred": 0, "fp": 0, "dup": 0, "rows": []} for v in ("inst", "lab") for T in TS}
    coco_gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "cell"}]}; coco_dt = {str(T): [] for T in TS}; aid = 0
    for k, im in enumerate(imgs):
        gt = gt_raster(im, by); H, W = gt.shape; excl = border_excl(gt); n = int(gt.max())
        cp = C2 / f"{im['id']}.pkl"
        if cp.exists(): r = pickle.load(open(cp, "rb"))
        else:
            if Rn is None: Rn = Runner()
            g = __import__("cv2").imdecode(np.frombuffer(zf.read(names[im["file_name"]]), np.uint8), -1)
            r = decode_low(Rn, g); pickle.dump(r, open(cp, "wb"))
        grle = [mu.encode(np.asfortranarray((gt == g_).astype(np.uint8))) for g_ in range(1, n + 1)]
        coco_gt["images"].append({"id": im["id"], "height": H, "width": W})
        for g_, rl in enumerate(grle, 1):
            aid += 1; rr = dict(rl); rr["counts"] = rr["counts"].decode()
            coco_gt["annotations"].append({"id": aid, "image_id": im["id"], "category_id": 1, "segmentation": rr, "area": float(mu.area(rl)),
                                           "bbox": mu.toBbox(rl).tolist(), "iscrowd": 0})
        for T in TS:
            thr = r["thr"] if T == "dyn" else T
            sel = [j for j in range(len(r["rle"])) if r["score"][j] > thr and r["iou"][j] >= 0.5]   # score-descending order kept
            dets = [(r["rle"][j], float(r["score"][j])) for j in sel]
            for j in sel:
                rr = dict(r["rle"][j]); rr["counts"] = rr["counts"].decode()
                coco_dt[str(T)].append({"image_id": im["id"], "category_id": 1, "segmentation": rr, "score": float(r["score"][j])})
            for v, dd in (("inst", dets), ("lab", labelmap([d[0] for d in dets], H, W))):
                a = acc[(v, str(T))]; a["n_pred"] += len(dd)
                rows = image_rows(gt, excl, dd)
                for x in rows: x["img"] = im["id"]; x["ctype"] = im["file_name"].split("_")[0]
                a["rows"] += [{"img": x["img"], "id": x["id"], "hit": x["hit"], "M": x["M"][2], "thick": x["thick"], "ctype": x["ctype"]} for x in rows]
                if dd and grle:                                       # FP and duplicates via one-to-one greedy by score
                    srt = sorted(dd, key=lambda z: -z[1]); iou = mu.iou([z[0] for z in srt], grle, [0] * n); used = np.zeros(n, bool)
                    for p in range(len(srt)):
                        c = np.where(~used & (iou[p] >= .5))[0]
                        if len(c): used[c[np.argmax(iou[p, c])]] = True
                        else:
                            a["fp"] += 1; a["dup"] += int((iou[p] >= .5).any())
        if k % 50 == 0: print(k, len(imgs), "cands>=.05", len(r["rle"]), f"{time.time() - t0:.0f}s", flush=True)
    out = {"thresholds": [str(T) for T in TS], "sweep": {}}
    for (v, T), a in acc.items():
        rr = a["rows"]; nn = len(rr); hit = sum(x["hit"] for x in rr); M = sum(x["M"] for x in rr)
        th = [x for x in rr if x["thick"] >= 3]
        resc = sum(1 for x in rr if (x["img"], x["id"]) in absorbed0 and x["hit"])
        out["sweep"].setdefault(v, {})[T] = {"n_pred": a["n_pred"], "fp": a["fp"], "dup": a["dup"], "recall": hit / nn, "absorption_rate": M / nn,
                                            "share_of_misses": M / max(nn - hit, 1), "recall_thick3": sum(x["hit"] for x in th) / len(th),
                                            "absorption_rate_thick3": sum(x["M"] for x in th) / len(th), "rescued_of_default_absorbed": resc, "n_default_absorbed": len(absorbed0)}
    cg = COCO(); cg.dataset = coco_gt; cg.createIndex()
    for T in TS:
        if not coco_dt[str(T)]: continue
        cd = cg.loadRes(coco_dt[str(T)]); E = COCOeval(cg, cd, "segm"); E.params.maxDets = [1, 100, 3500]; E.evaluate(); E.accumulate(); E.summarize()
        out["sweep"]["inst"][str(T)]["AP"] = float(E.stats[0]); out["sweep"]["inst"][str(T)]["AP50"] = float(E.stats[1])
    # no_box sensitivity on the default decomposition
    dec = pickle.load(open(R / "cellsam_decomposition.pkl", "rb")); best = []
    for x in dec:
        if x["cat"] != "no_box": continue
        c = pickle.load(open(C1 / f"{x['img']}.pkl", "rb")); best.append(float(box_iou(c["gt_boxes"][x["id"] - 1:x["id"]], c["raw_boxes"])[0].max()))
    best = np.array(best)
    out["no_box_best_raw_box_iou"] = {"n": len(best), "ge_.4": float((best >= .4).mean()), "ge_.3": float((best >= .3).mean()), "lt_.2": float((best < .2).mean()), "median": float(np.median(best))}
    json.dump(out, open(R / "d3q_threshold_sweep.json", "w"), indent=1)
    for v in ("inst", "lab"):
        print(f"\n[{v}] T      n_pred     FP     dup   recall  abs_rate  rec_t3  abs_t3  rescued   AP     AP50")
        for T in TS:
            s = out["sweep"][v][str(T)]
            print(f"  {str(T):5s} {s['n_pred']:8d} {s['fp']:7d} {s['dup']:6d}  {s['recall']:.3f}   {s['absorption_rate']:.3f}   {s['recall_thick3']:.3f}  {s['absorption_rate_thick3']:.3f}  {s['rescued_of_default_absorbed']:6d}  "
                  f"{s.get('AP', float('nan')):.3f}  {s.get('AP50', float('nan')):.3f}")
    print("no_box best raw-box IoU", out["no_box_best_raw_box_iou"])


if __name__ == "__main__":
    main()
