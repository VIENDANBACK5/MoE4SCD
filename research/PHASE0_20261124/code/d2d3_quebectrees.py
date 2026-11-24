"""Phase 0 D2 + D3 on QuebecTrees validation (139 tiles of 1,666 px, 3 cm).
Existing predictions only (no inference):
- Mask R-CNN s42/43/44 and TFN-c5 s42/43/44 (ITC_IDENTITY_ERROR_ORIGIN_V1, with validation F1@0.5 cut-offs);
- TFN-M0' s42/43/44 (track3 cached, selected snapshots; F1@0.5 cut-off computed here on validation with the same grid).

Exclusion: GT crowns touching the tile border or invalid area (boundary_diagnostics.edge_and_adjacency).

Overlap: validation tiles overlap by 50%, so each crown occurrence is de-duplicated by world coordinates (1 m grid). The
occurrence whose centroid is closest to its tile centre (most context) is kept. The same rule applies to every model.

Writes ../results/quebectrees_d2.json and ../results/quebectrees_d3.json."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
REPO = HERE.parents[2]; sys.path.insert(0, str(REPO))
from absorption import image_rows, summarise, D_TOUCH  # noqa: E402
from crown_segmentation_research.clean_benchmark.boundary_diagnostics import edge_and_adjacency  # noqa: E402
PREP = REPO / "data/itc_benchmarks/canopyrs_quebectrees/prepared"
IEO = REPO / "research/ITC_IDENTITY_ERROR_ORIGIN_V1_20261119"
T3P = REPO / "research/ITC_THREE_TRACKS_V1_20261120/track3_formation/preds"
M0P = {42: "exp028b_m0prime_s42_last", 43: "exp035b_m0prime_s43_last", 44: "exp035_m0prime_s44_snap_ep25"}  # EXP-035 selected
GRID = np.round(np.arange(.05, .96, .05), 2)
OUT = HERE.parent / "results"


def f1_cut(per_tile):
    """per_tile: list of (gt label raster, [(rle, score)]) -> score cut maximising F1@0.5 (one-to-one greedy)."""
    tp = np.zeros(len(GRID)); nd = np.zeros(len(GRID)); ng = 0
    for gt, dets in per_tile:
        ids = [g for g in np.unique(gt) if g]; ng += len(ids)
        if not dets or not ids: nd += [sum(s >= t for _, s in dets) for t in GRID]; continue
        grle = [mu.encode(np.asfortranarray((gt == g).astype(np.uint8))) for g in ids]
        dets = sorted(dets, key=lambda x: -x[1]); iou = mu.iou([d[0] for d in dets], grle, [0] * len(grle))
        sc = np.array([d[1] for d in dets])
        for k, t in enumerate(GRID):
            used = np.zeros(len(ids), bool)
            for p in np.where(sc >= t)[0]:
                c = np.where(~used & (iou[p] >= .5))[0]
                if len(c): used[c[np.argmax(iou[p, c])]] = True
            tp[k] += used.sum(); nd[k] += (sc >= t).sum()
    f1 = 2 * tp / np.maximum(nd + ng, 1)
    return float(GRID[int(np.argmax(f1))])


def main():
    coco = json.load(open(PREP / "validation_coco.json")); imgs = coco["images"]
    tiles = {}
    for info in imgs:
        z = np.load(PREP / "validation" / f"{Path(info['file_name']).stem}.npz"); gt, valid = z["label"], z["valid"].astype(bool)
        edge, _ = edge_and_adjacency(gt, valid); tiles[info["id"]] = (gt, edge, info["bounds"])
    models = {}
    cut = json.load(open(IEO / "results.json"))["cutoffs"]
    for fam in ("mrcnn", "tfn"):
        for s in (42, 43, 44):
            P = json.load(open(IEO / "preds" / f"{fam}_s{s}.json"))["crops"]
            models[f"{'MaskRCNN' if fam == 'mrcnn' else 'TFN-c5'}_s{s}"] = (
                {i: [({"size": d["size"], "counts": d["counts"].encode()}, d["score"]) for d in P[f"v{i}"]["dets"]] for i in tiles}, cut[f"{fam}_s{s}"])
    for s, f in M0P.items():
        dets = pickle.load(open(T3P / f"{f}.pkl", "rb")); by = {i: [] for i in tiles}
        for d in dets:
            r = dict(d["segmentation"]); r["counts"] = r["counts"].encode() if isinstance(r["counts"], str) else r["counts"]
            by[d["image_id"]].append((r, float(d["score"])))
        c = f1_cut([(tiles[i][0], by[i]) for i in tiles]); models[f"TFN-M0p_s{s}"] = (by, c); print("M0' cut", s, c, flush=True)
    # rows per model, then world-coordinate de-duplication (keep the occurrence nearest its tile centre)
    allrows = {}
    for name, (by, c) in models.items():
        rows = []
        for i, (gt, edge, b) in tiles.items():
            for x in image_rows(gt, edge, [(r, s) for r, s in by[i] if s >= c]):
                minx, _, _, maxy = b
                x["tile"] = i; x["key"] = (round(minx + (x["cx"] + .5) * .03), round(maxy - (x["cy"] + .5) * .03))
                x["ctr"] = float(np.hypot(x["cy"] - gt.shape[0] / 2, x["cx"] - gt.shape[1] / 2)); rows.append(x)
        best = {}
        for x in rows:
            if x["key"] not in best or x["ctr"] < best[x["key"]]["ctr"]: best[x["key"]] = x
        allrows[name] = (rows, list(best.values()), c)
        pickle.dump(list(best.values()), open(OUT / f"rows_quebectrees_{name}.pkl", "wb"))
        print(name, "cut", c, "occurrences", len(rows), "unique", len(best), flush=True)
    # D2: annotation-only statistics (identical GT for all models; taken from the first model's de-duplicated rows)
    u = next(iter(allrows.values()))[1]; area = np.array([x["area"] for x in u]) * 0.0009
    d2 = {"n_unique_crowns": len(u), "area_m2_quantiles": {q: float(np.quantile(area, q)) for q in (.05, .25, .5, .75, .95)}}
    for d in D_TOUCH:
        r = np.array([x["r"][d] for x in u])
        d2[f"d{d}"] = {"non_touching": int((r == 0).sum()), "balanced": int(((r > 0) & (r < 2)).sum()),
                       "moderate": int(((r >= 2) & (r < 4)).sum()), "strong": int((r >= 4).sum()),
                       "strong_share": float((r >= 4).mean()), "strong_lt2m2": int(((r >= 4) & (area < 2)).sum())}
    json.dump(d2, open(OUT / "quebectrees_d2.json", "w"), indent=1); print(json.dumps(d2, indent=1))
    size_edges = np.quantile([x["area"] for x in u], [.2, .4, .6, .8]); dens_edges = np.quantile([x["density"] for x in u], [1 / 3, 2 / 3])
    d3 = {"size_edges_px": size_edges.tolist(), "density_edges": dens_edges.tolist(), "models": {}}
    for name, (rows, uniq, c) in allrows.items():
        d3["models"][name] = {"cut": c, "by_d": {d: summarise([dict(x) for x in uniq], "tile", size_edges, dens_edges, d=d) for d in D_TOUCH},
                              "lt2m2_d2": summarise([dict(x) for x in uniq if x["area"] * .0009 < 2], "tile", size_edges, dens_edges, d=2)}
        s = d3["models"][name]["by_d"][2]
        print(name, {k: (v["n"], round(v["rate"], 3) if v["rate"] is not None else None) for k, v in s["strata"].items()},
              {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s["standardised"].items()}, flush=True)
    json.dump(d3, open(OUT / "quebectrees_d3.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
