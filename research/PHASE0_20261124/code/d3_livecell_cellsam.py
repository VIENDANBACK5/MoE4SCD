"""Phase 0 D3-query (G3): absorption of a query / set-prediction model on LIVECell VAL. Inference only.

Model and environment:
- CellSAM general v1.2 (Marks, Israel et al., Nature Methods 2025). Weights are taken from the owner's ~/.deepcell cache;
  the md5 matches checksums.md5.
- Run with /home/chung/RS/envs/phase0/bin/python (cellSAM 0.0.dev1).
- ZERO-SHOT: LIVECell is held out from CellSAM training. The pre-declared reading is in PHASE0_SYNTHESIS.md §6.

Inference reproduces cellSAM.model.segment_cellular_image (normalize=True, no WSI tiling, bbox_threshold 0.4) step by
step, so the intermediate states can be recorded:
  1. Input: format_image_shape puts the single phase channel in the last (whole-cell) channel, then normalize_image
     (99.9th-percentile clip + CLAHE).
  2. CellFinder returns 3,500 raw (box, score) pairs. Threshold = 0.66·0.4 + 0.33·mean(k-means k=2 centres), as in
     CellSAM.generate_bounding_boxes.
  3. Each kept box prompts the SAM decoder. A mask is dropped if the predicted IoU is < 0.5 (CellSAM.iou_threshold).
     Pixel threshold sigmoid > 0.4 (CellSAM.mask_threshold; the code uses 0.4, the paper says 0.5).
  4. Label map = max over (mask_k · k), so a later index overwrites an earlier one, then
     fill_holes_and_remove_small_masks(min_size=25). This is the package's output.
  Deviation: prompts are decoded in batches of up to 256 rather than one by one. The pilot checks agreement with
  segment_cellular_image (--pilot).

Evaluations (same absorption.py, border-touching GT excluded, identical GT construction to the other LIVECell runs):
  A. "labelmap": the package output, every instance scored 1. This is primary, the published operating point.
  B. "instances": the per-box masks before compositing (they may overlap), scored by the predicted IoU, with a
     validation F1@0.5 cut on the same grid as the other models.
  C. "gtbox": every GT box (all GT, including border instances) is a prompt.
     - For each instance: IoU of its own prompted mask and max coverage of a d=2 neighbour.
     - Absorption computed on these masks.
     - This separates "no hypothesis formed" from "mask assignment given the right hypothesis".

Decomposition of every labelmap-absorbed instance B (pre-declared). "composited" is tested first; otherwise the box
chain decides, in this order:
  no_box        no raw box (any score) with box-IoU >= 0.5 to B's GT box
  thresholded   such a box exists but every one scored below the threshold
  iou_rejected  kept box(es) exist, but every decoded mask was dropped for predicted IoU < 0.5
  mask_wrong    a kept, decoded mask exists, but none has mask-IoU >= 0.5 with B
  composited    a kept instance mask has IoU >= 0.5 with B, but B is lost in the label map (overwritten)

Writes ../results/livecell_d3_cellsam.json and ../results/rows_livecell_cellsam_{labelmap,instances,gtbox}.pkl, plus
../results/cellsam_cache/ (per-image raw boxes, scores and RLE masks for re-analysis)."""
import argparse, json, pickle, sys, time, zipfile
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from absorption import image_rows, summarise, D_TOUCH, neighbours  # noqa: E402
from d2d3_quebectrees import f1_cut  # noqa: E402
LC = Path("/home/chung/RS/data/livecell"); OUT = HERE.parent / "results"; CACHE = OUT / "cellsam_cache"
BATCH = 256


def load_val():
    d = json.load(open(LC / "annotations/livecell_coco_val.json"))
    anns = d["annotations"].values() if isinstance(d["annotations"], dict) else d["annotations"]
    by = {}
    for a in anns: by.setdefault(a["image_id"], []).append(a)
    imgs = list(d["images"].values() if isinstance(d["images"], dict) else d["images"])
    return imgs, by


def gt_raster(im, by):
    H, W = im["height"], im["width"]; gt = np.zeros((H, W), np.int32); n = 0
    for a in by.get(im["id"], []):
        seg = a["segmentation"]; rle = mu.merge(mu.frPyObjects(seg, H, W)) if isinstance(seg, list) else seg
        m = mu.decode(rle).astype(bool)
        if m.any(): n += 1; gt[m] = n
    return gt


def border_excl(gt):
    n = int(gt.max()); excl = np.zeros(n + 1, bool); b = np.unique(np.concatenate([gt[0], gt[-1], gt[:, 0], gt[:, -1]])); excl[b[b > 0]] = True
    return excl


def box_iou(a, b):
    """a (n,4), b (m,4) xyxy -> (n,m)."""
    if len(a) == 0 or len(b) == 0: return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)


class Runner:
    def __init__(self, device="cuda"):
        import torch
        from cellSAM.model import get_model
        self.torch = torch; self.dev = device
        self.m = get_model("cellsam_general").to(device).eval(); self.m.bbox_threshold = 0.4
        self.mdl = self.m.model_cp if self.m.adv_mode else self.m.model
        print("adv_mode", self.m.adv_mode, flush=True)

    def prep(self, g):
        from cellSAM.utils import normalize_image
        img = normalize_image(g.astype(np.float32))                       # (H, W, 3), phase in channel 2
        return self.torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(self.dev)

    def raw_boxes(self, t):
        from sklearn.cluster import KMeans
        res = self.m.cellfinder.forward_inference(self.m.sam_bbox_preprocessing(t, percentile=False))[0]
        boxes = res["boxes"].detach().cpu().numpy(); sc = res["scores"].detach().cpu().numpy()
        thr = self.m.bbox_threshold
        try:
            cc = KMeans(n_clusters=2, random_state=42).fit(sc.reshape(-1, 1)).cluster_centers_
            thr = 0.66 * self.m.bbox_threshold + 0.33 * float(np.mean(cc))
        except Exception:
            pass
        return boxes, sc, thr                                             # boxes in the 1024x1024 stretched frame

    def decode(self, emb, boxes1024, H, W):
        """-> list of (binary mask HxW uint8 or None, predicted IoU). Order preserved."""
        torch = self.torch; out = []
        for s in range(0, len(boxes1024), BATCH):
            b = torch.as_tensor(boxes1024[s:s + BATCH], dtype=torch.float32, device=self.dev).unsqueeze(1)  # (B,1,4)
            with torch.no_grad():
                sp, de = self.mdl.prompt_encoder(points=None, boxes=b, masks=None)
                lr, iou = self.mdl.mask_decoder(image_embeddings=emb, image_pe=self.mdl.prompt_encoder.get_dense_pe(),
                                                sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=False)
                up = self.m.model.postprocess_masks(lr, input_size=(1024, 1024), original_size=(H, W))
                ms = (torch.sigmoid(up[:, 0]) > self.m.mask_threshold).to(torch.uint8).cpu().numpy()
            iou = iou[:, 0].float().cpu().numpy()
            out += [(ms[k], float(iou[k])) for k in range(len(iou))]
        return out

    def run_image(self, g, gt):
        torch = self.torch; H, W = g.shape[:2]
        t = self.prep(g)
        with torch.no_grad():
            emb, _ = self.m.generate_embeddings(t)
        boxes, sc, thr = self.raw_boxes(t)
        keep = np.where(sc > thr)[0]
        dec = self.decode(emb, boxes[keep], H, W) if len(keep) else []
        inst = [(keep[k], m, i) for k, (m, i) in enumerate(dec) if i >= self.m.iou_threshold]
        lab = np.zeros((H, W), np.int64)
        for j, (_, m, _) in enumerate(inst): lab = np.maximum(lab, m.astype(np.int64) * (j + 1))
        if inst:
            from cellSAM.utils import fill_holes_and_remove_small_masks
            lab = fill_holes_and_remove_small_masks(lab, min_size=25)
        # GT-box prompts (all GT instances), scaled to the 1024 stretched frame
        from scipy.ndimage import find_objects
        ids = list(range(1, int(gt.max()) + 1)); sl = find_objects(gt)
        gtb = np.array([[s[1].start, s[0].start, s[1].stop, s[0].stop] if s is not None else [0, 0, 0, 0] for s in sl], float)
        gtb1024 = gtb * np.array([1024 / W, 1024 / H, 1024 / W, 1024 / H])
        gdec = self.decode(emb, gtb1024, H, W) if len(ids) else []
        sx = np.array([W / 1024, H / 1024, W / 1024, H / 1024])
        return {"raw_boxes": (boxes * sx).astype(np.float32), "raw_scores": sc.astype(np.float32), "thr": float(thr),
                "keep": keep, "dec_iou": np.array([i for _, i in dec], np.float32),
                "dec_rle": [mu.encode(np.asfortranarray(m)) for m, _ in dec],
                "inst": [(int(q), mu.encode(np.asfortranarray(m)), float(i)) for q, m, i in inst],
                "labelmap": lab.astype(np.int32), "gt_boxes": gtb.astype(np.float32),
                "gt_rle": [mu.encode(np.asfortranarray(m)) for m, _ in gdec], "gt_iou": np.array([i for _, i in gdec], np.float32)}


def labelmap_dets(lab):
    return [(mu.encode(np.asfortranarray((lab == v).astype(np.uint8))), 1.0) for v in np.unique(lab) if v > 0]


def decompose(gt, r, absorbed_ids):
    """Category for every absorbed GT id (see docstring)."""
    out = {}
    gtb = r["gt_boxes"]; raw_b = r["raw_boxes"]
    keep = set(r["keep"].tolist()); inst_q = {q for q, _, _ in r["inst"]}; inst_rle = [rl for _, rl, _ in r["inst"]]
    for g in absorbed_ids:
        grle = mu.encode(np.asfortranarray((gt == g).astype(np.uint8)))
        if inst_rle and mu.iou(inst_rle, [grle], [0]).max() >= .5: out[g] = "composited"; continue
        cand = set(np.where(box_iou(gtb[g - 1:g], raw_b)[0] >= .5)[0].tolist())
        if not cand: out[g] = "no_box"
        elif not cand & keep: out[g] = "thresholded"
        elif not cand & inst_q: out[g] = "iou_rejected"
        else: out[g] = "mask_wrong"
    return out


def pilot(n=6):
    """Agreement of this step-by-step reimplementation with cellSAM.segment_cellular_image on n val images."""
    import cv2
    from cellSAM.model import segment_cellular_image
    imgs, by = load_val(); zf = zipfile.ZipFile(LC / "images.zip"); names = {Path(x).name: x for x in zf.namelist() if x.endswith(".tif")}
    R = Runner()
    for im in imgs[:: max(1, len(imgs) // n)][:n]:
        g = cv2.imdecode(np.frombuffer(zf.read(names[im["file_name"]]), np.uint8), cv2.IMREAD_UNCHANGED); gt = gt_raster(im, by)
        t0 = time.time(); r = R.run_image(g, gt); t1 = time.time()
        try: ref, _, _ = segment_cellular_image(g, R.m, normalize=True, device="cuda")
        except AttributeError: ref = np.zeros_like(gt)            # package bug: predict returns (None,)*4 when every mask is rejected
        t2 = time.time()
        a, b = r["labelmap"] > 0, ref > 0
        print(im["file_name"], "GT", int(gt.max()), "raw>thr", len(r["keep"]), "inst", len(r["inst"]), "lab", len(np.unique(r["labelmap"])) - 1,
              "ref", len(np.unique(ref)) - 1, "fg IoU", round((a & b).sum() / max((a | b).sum(), 1), 4), "thr", round(r["thr"], 3),
              "t_ours", round(t1 - t0, 1), "t_ref", round(t2 - t1, 1), flush=True)


def main():
    import cv2
    imgs, by = load_val(); CACHE.mkdir(parents=True, exist_ok=True)
    zf = zipfile.ZipFile(LC / "images.zip"); names = {Path(x).name: x for x in zf.namelist() if x.endswith(".tif")}
    R = Runner(); per = []
    for k, im in enumerate(imgs):
        cp = CACHE / f"{im['id']}.pkl"; gt = gt_raster(im, by)
        if cp.exists(): r = pickle.load(open(cp, "rb"))
        else:
            g = cv2.imdecode(np.frombuffer(zf.read(names[im["file_name"]]), np.uint8), cv2.IMREAD_UNCHANGED)
            r = R.run_image(g, gt); pickle.dump(r, open(cp, "wb"))
        per.append((im, gt, r))
        if k % 25 == 0: print(k, len(imgs), "GT", int(gt.max()), "kept", len(r["keep"]), "inst", len(r["inst"]), "thr", round(r["thr"], 3), flush=True)
    cut = f1_cut([(gt, [(rl, s) for _, rl, s in r["inst"]]) for _, gt, r in per]); print("instances F1 cut", cut, flush=True)
    rows = {"labelmap": [], "instances": [], "gtbox": []}; decomp = []; gtbox_stats = []
    for im, gt, r in per:
        excl = border_excl(gt); ct = im["file_name"].split("_")[0]
        variants = {"labelmap": labelmap_dets(r["labelmap"]), "instances": [(rl, s) for _, rl, s in r["inst"] if s >= cut],
                    "gtbox": [(rl, 1.0) for rl in r["gt_rle"]]}
        for v, dets in variants.items():
            rr = image_rows(gt, excl, dets)
            for x in rr: x["img"] = im["id"]; x["ctype"] = ct
            rows[v] += rr
            if v == "labelmap":
                dc = decompose(gt, r, [x["id"] for x in rr if x["M"][2]])
                for x in rr:
                    if x["M"][2]: decomp.append({"img": im["id"], "ctype": ct, "id": x["id"], "r": x["r"][2], "area": x["area"], "thick": x["thick"], "cat": dc[x["id"]]})
        if r["gt_rle"]:                                                # own-prompt quality and neighbour coverage
            n = int(gt.max()); grle = [mu.encode(np.asfortranarray((gt == g).astype(np.uint8))) for g in range(1, n + 1)]
            own = np.diag(mu.iou(r["gt_rle"], grle, [0] * n)); cov = mu.iou(grle, r["gt_rle"], [1] * n)   # cov[g, p] = |g ∩ P| / |g|
            nb = neighbours(gt, 2)
            for g in range(1, n + 1):
                if excl[g]: continue
                nbc = max([cov[b - 1, g - 1] for b in nb.get(g, ())], default=0.0)
                gtbox_stats.append({"img": im["id"], "id": g, "own_iou": float(own[g - 1]), "max_nb_cov": float(nbc)})
    for v, rr in rows.items(): pickle.dump(rr, open(OUT / f"rows_livecell_cellsam_{v}.pkl", "wb"))
    area = np.array([x["area"] for x in rows["labelmap"]]); dens = np.array([x["density"] for x in rows["labelmap"]])
    se, de = np.quantile(area, [.2, .4, .6, .8]), np.quantile(dens, [1 / 3, 2 / 3])
    out = {"model": "CellSAM general v1.2, zero-shot", "instances_cut": cut, "size_edges_px": se.tolist(), "density_edges": de.tolist(),
           "eval": {v: {d_: summarise([dict(x) for x in rr], "img", se, de, d=d_) for d_ in D_TOUCH} for v, rr in rows.items()}}
    thick = [x for x in rows["labelmap"] if x["thick"] >= 3]
    a2 = np.array([x["area"] for x in thick]); d2 = np.array([x["density"] for x in thick])
    out["labelmap_thick>=3"] = summarise([dict(x) for x in thick], "img", np.quantile(a2, [.2, .4, .6, .8]), np.quantile(d2, [1 / 3, 2 / 3]), d=2)
    cats = ["no_box", "thresholded", "iou_rejected", "mask_wrong", "composited"]
    def share(dd): return {c: sum(x["cat"] == c for x in dd) for c in cats} | {"n": len(dd)}
    out["decomposition"] = {"all": share(decomp), "thick>=3": share([x for x in decomp if x["thick"] >= 3]),
                            "strong(r>=4)": share([x for x in decomp if x["r"] >= 4]),
                            "by_ctype": {ct: share([x for x in decomp if x["ctype"] == ct]) for ct in sorted({x["ctype"] for x in decomp})}}
    gs = gtbox_stats
    out["gtbox_prompt"] = {"n": len(gs), "own_iou_ge_.5": float(np.mean([x["own_iou"] >= .5 for x in gs])),
                           "nb_cov_ge_.5": float(np.mean([x["max_nb_cov"] >= .5 for x in gs])),
                           "own_ge_.5_and_nb_ge_.5": float(np.mean([x["own_iou"] >= .5 and x["max_nb_cov"] >= .5 for x in gs]))}
    pickle.dump(decomp, open(OUT / "cellsam_decomposition.pkl", "wb"))
    json.dump(out, open(OUT / "livecell_d3_cellsam.json", "w"), indent=1, default=float)
    for v in rows:
        s = out["eval"][v][2]; c = s["consequence"]; st = s["standardised"]
        print(v, "recall", round(c["recall"], 3), "abs share of misses", round(c["share_of_misses_that_are_absorption"], 3),
              "abs rate", round(c["recall_gain_if_all_absorptions_were_hits"], 3), "strong gain", round(c["recall_gain_if_strong_absorptions_were_hits"], 3),
              "std diff", st["diff"], st["diff_ci"], flush=True)
    c = out["labelmap_thick>=3"]["consequence"]; print("labelmap thick>=3", {k: round(v, 3) for k, v in c.items() if isinstance(v, float)})
    print("decomposition", json.dumps(out["decomposition"]["all"]), "\nthick>=3", json.dumps(out["decomposition"]["thick>=3"]))
    print("gtbox prompt", out["gtbox_prompt"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--pilot", action="store_true"); a = ap.parse_args()
    pilot() if a.pilot else main()
