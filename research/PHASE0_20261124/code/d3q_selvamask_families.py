"""D3-query Step 2a on SelvaMask SAME6, as specified in ../PROTOCOL_D3Q_SELVAMASK.md (written before running).
In-domain query (Mask2Former × 3 seeds) vs NMS (R5 Mask R-CNN) vs flow (TreeFlowNet PP200), using saved frozen
predictions; CPU only. Writes ../results/d3q_selvamask.json and rows_selvamask_<system>.pkl."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from scipy import ndimage as ndi
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from absorption import image_rows, D_TOUCH  # noqa: E402
RS = HERE.parents[1]; OUT = HERE.parent / "results"
TILES_DIR = RS / "diagnostics/D1_M0_BOTTLENECK/execution_v2/tiles"
M2F = RS / "ITC_CARRIER_FORMATION_V1_20261103/evaluation"; R5 = RS / "method_development/R5_INSTANCE_METHOD/evaluation"
TILES = ["v025", "v072", "v101", "v125", "v130", "v188"]


def rle(r):
    r = dict(r); r["counts"] = r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]; return r


def load(path, key="predictions"):
    o = json.load(open(path)); return [(rle(p["rle"]), float(p["score"])) for p in o[key]]


def systems():
    s = {f"Mask2Former_s{k}": {t: M2F / f"mask2former_seed{k}" / t / "predictions.json" for t in TILES} for k in (42, 43, 44)}
    s["R5_MaskRCNN"] = {t: R5 / t / "predictions.json" for t in TILES}
    s["TFN_PP200"] = {t: TILES_DIR / t / "PP200_predictions.json" for t in TILES}
    return s


def main():
    tiles = {}
    for t in TILES:
        refs = json.load(open(TILES_DIR / t / "references.json"))["references"]
        H, W = refs[0]["rle"]["size"]; gt = np.zeros((H, W), np.int32)
        for k, r in enumerate(refs, 1): gt[mu.decode(rle(r["rle"])).astype(bool)] = k
        valid = np.asarray(Image.open(TILES_DIR / t / "valid.png")) > 0
        inv = ndi.binary_dilation(~valid, iterations=1)
        excl = np.zeros(len(refs) + 1, bool)
        b = np.unique(np.concatenate([gt[0], gt[-1], gt[:, 0], gt[:, -1], gt[inv]])); excl[b[b > 0]] = True
        tiles[t] = (gt, excl, refs)
    print("GT", sum(len(v[2]) for v in tiles.values()), "non-excluded", sum(int((~v[1][1:]).sum()) for v in tiles.values()), flush=True)
    res = {"systems": {}}; allrows = {}
    for name, paths in systems().items():
        rows = []
        for t, (gt, excl, _) in tiles.items():
            for x in image_rows(gt, excl, load(paths[t])): x["tile"] = t; rows.append(x)
        allrows[name] = rows; pickle.dump(rows, open(OUT / f"rows_selvamask_{name}.pkl", "wb"))
        n = len(rows); hit = sum(x["hit"] for x in rows)
        res["systems"][name] = {"n": n, "recall": hit / n, **{f"abs_rate_d{d}": sum(x["M"][d] for x in rows) / n for d in D_TOUCH},
                                "share_of_misses": sum(x["M"][2] for x in rows) / max(n - hit, 1),
                                "strong_rate": sum(x["M"][2] for x in rows if x["r"][2] >= 4) / n}
    # primary: paired difference vs R5 per Mask2Former seed, tile-cluster bootstrap and McNemar
    key = lambda x: (x["tile"], x["id"]); ref = {key(x): x["M"][2] for x in allrows["R5_MaskRCNN"]}
    rng = np.random.default_rng(0); res["primary"] = {}
    for s in (42, 43, 44):
        name = f"Mask2Former_s{s}"; q = {key(x): x["M"][2] for x in allrows[name]}; assert q.keys() == ref.keys()
        per = {t: [(q[k], ref[k]) for k in q if k[0] == t] for t in TILES}
        def diff(ts):
            a = [v for t in ts for v in per[t]]; return np.mean([u for u, _ in a]) - np.mean([w for _, w in a])
        boots = [diff(rng.choice(TILES, len(TILES))) for _ in range(1000)]
        b01 = sum(1 for k in q if q[k] and not ref[k]); b10 = sum(1 for k in q if ref[k] and not q[k])
        from scipy.stats import binomtest
        p = binomtest(b01, b01 + b10, .5).pvalue if b01 + b10 else 1.0
        rq, rr = res["systems"][name]["abs_rate_d2"], res["systems"]["R5_MaskRCNN"]["abs_rate_d2"]
        res["primary"][name] = {"rate_m2f": rq, "rate_r5": rr, "ratio": rq / rr if rr else None, "diff": rq - rr,
                                "diff_ci_tile_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                "mcnemar_m2f_only": b01, "mcnemar_r5_only": b10, "mcnemar_p": p}
    ratios = [res["primary"][f"Mask2Former_s{s}"]["ratio"] for s in (42, 43, 44)]
    res["reading"] = ("SET_PREDICTION_DOES_NOT_REMOVE_ABSORPTION" if all(r >= .8 for r in ratios) else
                      "SET_PREDICTION_SUBSTANTIALLY_REDUCES" if all(r <= .5 for r in ratios) else "PARTIAL")
    # secondary: multiplicity of the absorbing prediction, and the Mask2Former stage decomposition
    res["multiplicity"] = {}; res["m2f_stage"] = {}
    for name, paths in systems().items():
        ncov = []; stage = {"no_raw_iou50": 0, "raw_not_accepted": 0, "accepted_lost_in_matching": 0}
        for t, (gt, excl, _) in tiles.items():
            ab = [x["id"] for x in allrows[name] if x["tile"] == t and x["M"][2]]
            if not ab: continue
            n = int(gt.max()); grle = [mu.encode(np.asfortranarray((gt == k).astype(np.uint8))) for k in range(1, n + 1)]
            dets = load(paths[t]); cov = mu.iou(grle, [d[0] for d in dets], [1] * len(dets)); cnt = (cov >= .5).sum(0)
            for g in ab:
                own = np.where(cov[g - 1] >= .5)[0]
                if len(own): ncov.append(int(cnt[own].max()))
            if name.startswith("Mask2Former"):
                raw = load(paths[t], "raw_predictions"); acc = [d[0] for d in dets]
                iraw = mu.iou([d[0] for d in raw], [grle[g - 1] for g in ab], [0] * len(ab)).max(0) if raw else np.zeros(len(ab))
                iacc = mu.iou(acc, [grle[g - 1] for g in ab], [0] * len(ab)).max(0) if acc else np.zeros(len(ab))
                for j in range(len(ab)):
                    stage["no_raw_iou50" if iraw[j] < .5 else ("raw_not_accepted" if iacc[j] < .5 else "accepted_lost_in_matching")] += 1
        ncov = np.array(ncov)
        res["multiplicity"][name] = {"n": len(ncov), "2": float((ncov == 2).mean()) if len(ncov) else None,
                                     "3-4": float(((ncov >= 3) & (ncov <= 4)).mean()) if len(ncov) else None,
                                     ">=5": float((ncov >= 5).mean()) if len(ncov) else None}
        if name.startswith("Mask2Former"): res["m2f_stage"][name] = stage
    json.dump(res, open(OUT / "d3q_selvamask.json", "w"), indent=1, default=float)
    print(f"{'system':18s} recall  abs_d1  abs_d2  abs_d5  share_miss  strong")
    for k, v in res["systems"].items():
        print(f"{k:18s} {v['recall']:.3f}  {v['abs_rate_d1']:.3f}   {v['abs_rate_d2']:.3f}   {v['abs_rate_d5']:.3f}   {v['share_of_misses']:.3f}      {v['strong_rate']:.3f}")
    for k, v in res["primary"].items(): print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    print("READING:", res["reading"]); print("multiplicity", res["multiplicity"]); print("m2f stage", res["m2f_stage"])


if __name__ == "__main__":
    main()
