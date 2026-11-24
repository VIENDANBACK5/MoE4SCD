"""Action-consequence diagnostic (no retraining), answering the review.

For every validation candidate P_j and every arm+seed, reform ONLY P_j (replace it by that arm's two children, all other
B0 predictions unchanged) and recompute, locally and exactly, the marginal change on the border-excluded crowns of that
crop:
- d_rec  = crowns that were absorbed (B0 M) and are now hit;
- d_lost = crowns that were hit in B0 and are now not;
- dnet   = d_rec - d_lost.

This is the ΔU_i the review asked for (with U = hits). From the per-candidate (logit, dnet, k) we derive, per arm:
1. action-benefit oracle net = sum of dnet over candidates with dnet > 0 (perfect per-candidate action decision);
2. multi-object oracle net   = sum of dnet over candidates with k >= 2 (what the main eval used);
3. learned net               = sum of dnet over candidates the arm actually reforms (logit + shift > 0);
   -- this validates the marginal-sum approximation against the headline EVAL net;
4. threshold-matched curves: net(t) and splits vs a sweep of the decision threshold, so B1/B2/M2 are compared at the
   SAME false-split rate;
5. AUC of the decision logit for predicting dnet > 0 (does the head track action benefit?).

Non-additivity caveat: the marginal sum assumes candidate actions are independent. Reforms are sparse and local, so this
is a good approximation; point (3) checks it against the true joint eval.

Per-crop raw crowns (not world-deduplicated); this is a relative operator diagnostic, reported separately from the
de-duplicated headline. Writes ../results/ACTION.json. Run with PYTHONNOUSERSITE=1 python3."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from common import crops, load_maskrcnn, maskrcnn_forward, roi_box, crop_resample, roi_features, candidate_labels, STUDY
from arms import ARMS
from evaluate import gt_pre, children_B1, children_B2, children_M2, finalize, enc
from crown_segmentation_research.clean_benchmark.boundary_diagnostics import edge_and_adjacency
SEEDS = (42, 43, 44)


def greedy_hits(iou, scores):
    """iou [nP,nG], scores [nP] -> hit[nG] bool by score-ordered one-to-one at IoU>=0.5."""
    nG = iou.shape[1]; hit = np.zeros(nG, bool)
    if iou.shape[0] == 0: return hit
    for p in np.argsort(-scores):
        c = np.where(~hit & (iou[p] >= .5))[0]
        if len(c): hit[c[np.argmax(iou[p, c])]] = True
    return hit


def main():
    shift = json.load(open(STUDY / "results/EVAL.json"))["prior_shift"]
    models = {}
    for s in SEEDS:
        for a, cls in ARMS.items():
            m = cls().cuda().eval(); m.load_state_dict(torch.load(STUDY / f"results/runs/seed{s}/{a}.pt", map_location="cuda")); models[(a, s)] = m
    mr, _ = load_maskrcnn(42)
    data = {(a, s): {"logit": [], "dnet": [], "drec": [], "dlost": [], "k": []} for a in ARMS for s in SEEDS}
    t0 = time.time()
    for ci, (cid, img, valid, gt) in enumerate(crops("validation")):
        H, W = gt.shape; edge, _ = edge_and_adjacency(gt, valid); pre = gt_pre(gt, edge)
        ids = pre["ids"]; grle = pre["grle"]; keep_g = [g for g in ids if not edge[g]]
        if not keep_g: continue
        gidx = {g: i for i, g in enumerate(ids)}
        dets, f0 = maskrcnn_forward(mr, np.ascontiguousarray(img))
        cand = [(m & valid, sscore) for m, sscore in dets]; cand = [(m, sscore) for m, sscore in cand if m.any()]
        ms = [m for m, _ in cand]; sc = np.array([s for _, s in cand], float)
        if not ms: continue
        labs = candidate_labels(gt, ms); kcat = [min(k, 2) for k, _ in labs]
        crle = [enc(m) for m in ms]
        iou0 = mu.iou(crle, grle, [0] * len(grle))                      # [nP, nG]
        hit0 = greedy_hits(iou0, sc)
        absorbed0 = np.array([pre["ids"][i] in [g for g in ids] and (not hit0[i]) for i in range(len(ids))])  # placeholder
        # B0 absorption M per crown (reuse rows_fast semantics via pre + cov)
        cov0 = mu.iou(grle, crle, [1] * len(crle))                      # [nG, nP]
        b0M = np.zeros(len(ids), bool)
        for i, g in enumerate(ids):
            if edge[g] or hit0[i]: continue
            nb = [b for b in pre["nb"].get(g, ()) if b in gidx]
            if not nb: continue
            own = cov0[i] >= .5
            if own.any() and any((own & (cov0[gidx[b]] >= .5)).any() for b in nb): b0M[i] = True
        boxes = [roi_box(m) for m in ms]
        F = roi_features(f0, boxes).cuda().float()
        Mm = torch.from_numpy(np.stack([crop_resample(m.astype(np.float32), b) for m, b in zip(ms, boxes)])[:, None]).cuda()
        outs = {}
        with torch.no_grad():
            for key, m in models.items(): outs[key] = {kk: vv.cpu().numpy() for kk, vv in m(F, Mm).items()}
        keepmask = np.array([not edge[g] for g in ids])
        for j in range(len(ms)):
            for s in SEEDS:
                for a in ARMS:
                    o = outs[(a, s)]
                    if a == "B1": kids = finalize(children_B1(ms[j]), ms[j])
                    elif a == "B2": kids = finalize(children_B2(o["mask_logits"][j], boxes[j], H, W, valid), ms[j])
                    else: kids = finalize(children_M2(o["member_logits"][j], boxes[j], H, W, valid), ms[j])
                    if kids is None:
                        drec = dlost = 0
                    else:
                        krle = [enc(k) for k in kids]
                        iou_k = mu.iou(krle, grle, [0] * len(grle))      # [2, nG]
                        rows = [iou0[p] for p in range(len(ms)) if p != j] + [iou_k[0], iou_k[1]]
                        scj = np.array([sc[p] for p in range(len(ms)) if p != j] + [sc[j], sc[j]])
                        hitn = greedy_hits(np.array(rows), scj)
                        chg = keepmask & (hitn != hit0)
                        drec = int((chg & b0M & hitn).sum()); dlost = int((chg & hit0 & ~hitn).sum())
                    d = data[(a, s)]; d["logit"].append(float(o["logit"][j])); d["dnet"].append(drec - dlost)
                    d["drec"].append(drec); d["dlost"].append(dlost); d["k"].append(kcat[j])
        if ci % 20 == 0: print(ci, cid, f"{time.time() - t0:.0f}s", flush=True)
    # aggregate
    N = json.load(open(STUDY / "results/EVAL.json"))["n_unique"]
    def auc(score, y):
        y = np.array(y); score = np.array(score)
        if y.sum() == 0 or y.sum() == len(y): return None
        order = np.argsort(score); r = np.empty(len(score)); r[order] = np.arange(1, len(score) + 1)
        return float((r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (y == 0).sum()))
    res = {"note": "per-crop raw crowns; dnet is marginal (reform one candidate vs B0); net in pp uses N_unique=%d" % N, "arms": {}}
    thr_grid = list(np.round(np.arange(-6, 4.01, 0.25), 2))
    for (a, s), d in data.items():
        lg = np.array(d["logit"]); dn = np.array(d["dnet"]); k = np.array(d["k"])
        pp = lambda m: round(100 * dn[m].sum() / N, 3)
        key = f"{a}_s{s}"
        res["arms"][key] = {
            "n_cand": len(dn), "raw_crowns_recovered_ceiling": int(dn[dn > 0].sum()),
            "action_oracle_net_pp": pp(dn > 0), "multiobject_oracle_net_pp": pp(k >= 2),
            "learned_net_pp": pp(lg + shift > 0), "total_marginal_net_pp": pp(np.ones_like(dn, bool)),
            "auc_logit_predicts_benefit": auc(lg, (dn > 0).astype(int)),
            "sweep": [{"thr": float(t), "net_pp": round(100 * dn[lg > t].sum() / N, 3),
                       "false_split_rate": round(float((k[lg > t] < 2).mean()) if (lg > t).any() else 0.0, 4),
                       "true_split_rate": round(float((k[lg > t] >= 2).mean()) if (k >= 2).any() else 0.0, 4),
                       "frac_reformed_of_REFORM": round(float(((lg > t) & (k >= 2)).sum() / max((k >= 2).sum(), 1)), 4)} for t in thr_grid]}
    json.dump(res, open(STUDY / "results/ACTION.json", "w"), indent=1)
    print("\narm      action_oracle  multiobj_oracle  learned   AUC(logit->benefit)")
    for a in ARMS:
        ao = np.mean([res["arms"][f"{a}_s{s}"]["action_oracle_net_pp"] for s in SEEDS])
        mo = np.mean([res["arms"][f"{a}_s{s}"]["multiobject_oracle_net_pp"] for s in SEEDS])
        le = np.mean([res["arms"][f"{a}_s{s}"]["learned_net_pp"] for s in SEEDS])
        au = np.mean([res["arms"][f"{a}_s{s}"]["auc_logit_predicts_benefit"] for s in SEEDS])
        print(f"{a:6s}   {ao:+.2f}pp       {mo:+.2f}pp        {le:+.2f}pp    {au:.3f}")
    # threshold-matched: net at the false-split rate M2 uses (~0.029), for each arm (mean over seeds)
    print("\nthreshold-matched net at false-split ~= 0.03 and ~= 0.07:")
    for target in (0.03, 0.07):
        row = []
        for a in ARMS:
            nets = []
            for s in SEEDS:
                sw = res["arms"][f"{a}_s{s}"]["sweep"]; best = min(sw, key=lambda x: abs(x["false_split_rate"] - target))
                nets.append(best["net_pp"])
            row.append(f"{a} {np.mean(nets):+.2f}")
        print(f"  fs~{target}: " + "   ".join(row))


if __name__ == "__main__":
    main()
