"""Validation evaluation of Phase 1 · Prototype 1 (../PROTOCOL_M2_PROTOTYPE.md). All seeds and arms in one pass.

Variants:
- B0: candidates unchanged;
- {B1, B2, M2} × seed × {learned decision, ORACLE decision (REFORM iff k >= 2)}.

The decision logit is prior-corrected from the training-cache ratio to the natural train KEEP/REFORM prior; the threshold
is 0.5.

Metrics (d = 2, world-coordinate de-duplication as in PHASE0 d2d3_quebectrees):
- recall; absorption rate; recovered / lost / net vs B0;
- FP; COCO mask AP over the 139 crops;
- decision rates by label;
- P(trigger hit | absorbed) and P(recovered | trigger hit, absorbed);
- the auxiliary U / C / O stratification of the B0 misses.

Writes ../results/EVAL.json. Run with system python3."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from pycocotools import mask as mu
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from common import (crops, load_maskrcnn, maskrcnn_forward, roi_box, crop_resample, roi_features, candidate_labels,  # noqa: E402
                    REPO, PREP, STUDY, R)
from arms import ARMS  # noqa: E402
sys.path.insert(0, str(REPO / "research/PHASE0_20261124/code"))
from absorption import neighbours  # noqa: E402
from crown_segmentation_research.clean_benchmark.boundary_diagnostics import edge_and_adjacency  # noqa: E402
MIN_CHILD = 100


# ---------- GT-side precomputation + per-variant rows (identical to absorption.image_rows, d = 2) ----------
def gt_pre(gt, excl):
    ids = [g for g in range(1, int(gt.max()) + 1)]
    area = np.bincount(gt.ravel(), minlength=len(ids) + 1).astype(np.int64)
    ids = [g for g in ids if area[g] > 0]
    grle = [mu.encode(np.asfortranarray((gt == g).astype(np.uint8))) for g in ids]
    nb = neighbours(gt, 2); com = np.array(ndi.center_of_mass(np.ones_like(gt), gt, ids))
    return {"ids": ids, "area": area, "grle": grle, "pos": {g: k for k, g in enumerate(ids)}, "nb": nb, "com": com, "excl": excl}


def rows_fast(pre, dets):
    ids, grle, pos = pre["ids"], pre["grle"], pre["pos"]
    dets = sorted(dets, key=lambda x: -x[1]); prle = [d[0] for d in dets]
    if prle:
        iou = mu.iou(prle, grle, [0] * len(grle)); cov = mu.iou(grle, prle, [1] * len(prle))
    else:
        iou = np.zeros((0, len(ids))); cov = np.zeros((len(ids), 0))
    hit = np.zeros(len(ids), bool)
    for p in range(len(prle)):
        c = np.where(~hit & (iou[p] >= .5))[0]
        if len(c): hit[c[np.argmax(iou[p, c])]] = True
    out = {}
    for k, g in enumerate(ids):
        if pre["excl"][g]: continue
        nb = [b for b in pre["nb"].get(g, ()) if b in pos]
        M = False
        if not hit[k] and nb and cov.shape[1]:
            own = cov[k] >= .5
            if own.any(): M = any((own & (cov[pos[b]] >= .5)).any() for b in nb)
        r = max(pre["area"][b] for b in nb) / pre["area"][g] if nb else 0.0
        out[g] = {"hit": bool(hit[k]), "M": bool(M), "r": float(r), "maxcov": float(cov[k].max()) if cov.shape[1] else 0.0}
    tp = int(hit.sum()); return out, len(prle) - tp


# ---------- children ----------
def paste_box(arr2d, box, H, W):
    x0, y0, x1, y1 = box; out = np.zeros((H, W), bool)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
    if sy1 > sy0 and sx1 > sx0: out[sy0:sy1, sx0:sx1] = arr2d[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    return out


def up(t, box):
    import cv2
    x0, y0, x1, y1 = box
    return cv2.resize(t.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)


def children_B2(ml, box, H, W, valid):
    return [paste_box(up(1 / (1 + np.exp(-ml[c])), box) > 0.5, box, H, W) & valid for c in (0, 1)]


def children_M2(lg, box, H, W, valid):
    a = np.stack([up(lg[c], box) for c in range(3)]).argmax(0)
    return [paste_box(a == c, box, H, W) & valid for c in (1, 2)]


def children_B1(parent):
    from sklearn.mixture import GaussianMixture
    ys, xs = np.nonzero(parent); pts = np.stack([ys, xs], 1).astype(np.float64)
    if len(pts) < 2 * MIN_CHILD: return None
    rng = np.random.default_rng(0); sub = pts[rng.choice(len(pts), min(4000, len(pts)), replace=False)]
    lab = GaussianMixture(2, random_state=0).fit(sub).predict(pts)
    out = []
    for c in (0, 1):
        m = np.zeros(parent.shape, bool); m[ys[lab == c], xs[lab == c]] = True; out.append(m)
    return out


def finalize(kids, parent):
    kids = [k for k in (kids or []) if k.sum() >= MIN_CHILD]
    return kids if len(kids) >= 2 else None


def enc(m):
    return mu.encode(np.asfortranarray(m.astype(np.uint8)))


def main():
    cfg = json.load(open(STUDY / "CONFIG.json")); idx = json.load(open(STUDY / "results/cache/index.json"))
    lab = np.load(STUDY / "results/cache/label.npy"); nR, nK = int(lab.sum()), int((lab == 0).sum())
    nat = idx["natural_counts_FP_KEEP_REFORM"]; pi = nat[2] / (nat[1] + nat[2])
    shift = np.log(pi / (1 - pi)) - np.log(nR / nK)
    models = {}
    for s in cfg["seeds"]:
        for a, cls in ARMS.items():
            m = cls().cuda().eval(); m.load_state_dict(torch.load(STUDY / f"results/runs/seed{s}/{a}.pt", map_location="cuda")); models[(a, s)] = m
    mr, _ = load_maskrcnn(42)
    variants = ["B0"] + [f"{a}_s{s}_{d}" for s in cfg["seeds"] for a in ARMS for d in ("learned", "oracle")]
    coco = json.load(open(PREP / "validation_coco.json")); bounds = {f"v{i['id']}": i["bounds"] for i in coco["images"]}
    allrows = {v: [] for v in variants}; fp = {v: 0 for v in variants}; dt = {v: [] for v in variants}
    dec = {v: np.zeros((3, 2), int) for v in variants}                  # rows: label FP/KEEP/REFORM; cols: keep/reform
    trig = {v: [] for v in variants}                                     # per B0-absorbed occurrence: (trigger hit, recovered)
    gt_coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "tree"}]}; aid = 0; t0 = time.time()
    for ci, (cid, img, valid, gt) in enumerate(crops("validation")):
        H, W = gt.shape; iid = ci + 1
        edge, _ = edge_and_adjacency(gt, valid); pre = gt_pre(gt, edge)
        gt_coco["images"].append({"id": iid, "height": H, "width": W})
        for g, rl in zip(pre["ids"], pre["grle"]):
            aid += 1; rr = dict(rl); rr["counts"] = rr["counts"].decode()
            gt_coco["annotations"].append({"id": aid, "image_id": iid, "category_id": 1, "segmentation": rr, "area": float(mu.area(rl)), "bbox": mu.toBbox(rl).tolist(), "iscrowd": 0})
        dets, f0 = maskrcnn_forward(mr, np.ascontiguousarray(img))
        cand = [(m & valid, s) for m, s in dets]; cand = [(m, s) for m, s in cand if m.any()]
        ms = [m for m, _ in cand]; sc = [s for _, s in cand]; labs = candidate_labels(gt, ms)
        kcat = [min(k, 2) for k, _ in labs]
        boxes = [roi_box(m) for m in ms]
        if ms:
            F = roi_features(f0, boxes).cuda().float()
            Mm = torch.from_numpy(np.stack([crop_resample(m.astype(np.float32), b) for m, b in zip(ms, boxes)])[:, None]).cuda()
        outs = {}
        with torch.no_grad():
            for key, m in models.items():
                outs[key] = {kk: vv.cpu().numpy() for kk, vv in m(F, Mm).items()} if ms else {}
        crle = [enc(m) for m in ms]; b1_cache = {}
        per = {"B0": [(r, s) for r, s in zip(crle, sc)]}; reformed = {"B0": set()}
        for (a, s), o in outs.items():
            for d in ("learned", "oracle"):
                v = f"{a}_s{s}_{d}"; preds = []; rf = set()
                for j, (m, sj) in enumerate(zip(ms, sc)):
                    want = (o["logit"][j] + shift > 0) if d == "learned" else (kcat[j] == 2)
                    kids = None
                    if want:
                        if a == "B1":
                            if j not in b1_cache: b1_cache[j] = finalize(children_B1(m), m)
                            kids = b1_cache[j]
                        elif a == "B2": kids = finalize(children_B2(o["mask_logits"][j], boxes[j], H, W, valid), m)
                        else: kids = finalize(children_M2(o["member_logits"][j], boxes[j], H, W, valid), m)
                    dec[v][kcat[j], int(want)] += 1
                    if kids: rf.add(j); preds += [(enc(k), sj) for k in kids]
                    else: preds.append((crle[j], sj))
                per[v] = preds; reformed[v] = rf
        if ms:
            cov0 = mu.iou(pre["grle"], crle, [1] * len(crle))
        minx, _, _, maxy = bounds[cid]
        rows0 = None
        for v in variants:
            rows, nfp = rows_fast(pre, per[v]); fp[v] += nfp
            for p, s in per[v]:
                rr = dict(p); rr["counts"] = rr["counts"].decode() if isinstance(rr["counts"], bytes) else rr["counts"]
                dt[v].append({"image_id": iid, "category_id": 1, "segmentation": rr, "score": float(s)})
            if v == "B0": rows0 = rows
            for g, x in rows.items():
                k = pre["pos"][g]; cy, cx = pre["com"][k]
                x.update({"tile": cid, "id": g, "key": (round(minx + (cx + .5) * .03), round(maxy - (cy + .5) * .03)),
                          "ctr": float(np.hypot(cy - H / 2, cx - W / 2))})
                allrows[v].append(x)
                if v != "B0" and rows0[g]["M"] and ms:
                    covering = set(np.where(cov0[k] >= .5)[0].tolist())
                    trig[v].append((bool(covering & reformed[v]), x["hit"]))
        if ci % 10 == 0: print(ci, cid, "cands", len(ms), f"{time.time() - t0:.0f}s", flush=True)
    # de-duplicate per variant (keep the occurrence nearest its crop centre; identical choice across variants)
    def dedup(rows):
        best = {}
        for x in rows:
            if x["key"] not in best or x["ctr"] < best[x["key"]]["ctr"]: best[x["key"]] = x
        return best
    U = {v: dedup(allrows[v]) for v in variants}; base = U["B0"]; N = len(base)
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    cg = COCO(); cg.dataset = gt_coco; cg.createIndex()
    res = {"n_unique": N, "prior_shift": float(shift), "variants": {}}
    for v in variants:
        u = U[v]; assert u.keys() == base.keys()
        rec = sum(1 for k in u if base[k]["M"] and u[k]["hit"]); lost = sum(1 for k in u if base[k]["hit"] and not u[k]["hit"])
        E = COCOeval(cg, cg.loadRes(dt[v]), "segm"); E.params.maxDets = [1, 100, 1000]; E.evaluate(); E.accumulate(); E.summarize()
        t = np.array(trig[v]) if trig[v] else np.zeros((0, 2), bool)
        res["variants"][v] = {"recall": sum(x["hit"] for x in u.values()) / N, "absorption_rate": sum(x["M"] for x in u.values()) / N,
                              "recovered": rec, "lost": lost, "net_pp": 100 * (rec - lost) / N, "FP": fp[v], "AP": float(E.stats[0]), "AP50": float(E.stats[1]),
                              "decisions_FP_KEEP_REFORM_x_keep_reform": dec[v].tolist(),
                              "P_trigger_given_absorbed_occ": float(t[:, 0].mean()) if len(t) else None,
                              "P_recovered_given_trigger_absorbed_occ": float(t[t[:, 0], 1].mean()) if len(t) and t[:, 0].any() else None}
    miss = [x for x in base.values() if not x["hit"]]
    res["UCO_B0"] = {"U_uncovered": sum(1 for x in miss if x["maxcov"] < .5), "C_absorbed": sum(1 for x in miss if x["M"]),
                     "O_other": sum(1 for x in miss if x["maxcov"] >= .5 and not x["M"]), "misses": len(miss)}
    json.dump(res, open(STUDY / "results/EVAL.json", "w"), indent=1)
    for v, x in res["variants"].items():
        print(f"{v:22s} recall {x['recall']:.4f} abs {x['absorption_rate']:.4f} rec {x['recovered']:4d} lost {x['lost']:4d} net {x['net_pp']:+.2f}pp FP {x['FP']:6d} AP {x['AP']:.4f}")
    print("UCO", res["UCO_B0"])


if __name__ == "__main__":
    main()
