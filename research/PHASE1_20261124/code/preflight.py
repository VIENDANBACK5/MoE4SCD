"""Pre-flight for Phase 1 · Prototype 1. Must PASS before train.py will run. Writes ../results/PREFLIGHT.json."""
import json, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from common import tile_names, crops, load_maskrcnn, maskrcnn_forward, REPO  # noqa: E402
from arms import ARMS, n_params  # noqa: E402
from train import canonical_hash, load_cache, train, batch  # noqa: E402
STUDY = HERE.parent; checks = {}


def check(name, ok, info=None):
    checks[name] = {"ok": bool(ok), "info": info}; print(("PASS " if ok else "FAIL ") + name, info if info is not None else "", flush=True)


def main():
    cfg = json.load(open(STUDY / "CONFIG.json"))
    tr, va = set(tile_names("train")), set(tile_names("validation"))
    check("train_val_tiles_disjoint", not (tr & va), {"train": len(tr), "validation": len(va)})
    check("no_test_in_config", "test" not in json.dumps(cfg["dataset"]).lower().replace("test not touched", ""))
    n, arr, label, idx = load_cache()
    check("cache_complete", n > 1000 and len(idx["meta"]) == n, {"n": n, "reform": int(label.sum()), "keep": int((label == 0).sum()), "natural": idx["natural_counts_FP_KEEP_REFORM"]})
    check("cache_from_train_only", all(m["crop"].startswith("t") for m in idx["meta"]))
    rs = np.where(label == 1)[0][:200]; ks = np.where(label == 0)[0][:200]
    t = np.asarray(arr["tgt"][np.sort(np.concatenate([rs, ks]))]).astype(np.float32); lb = label[np.sort(np.concatenate([rs, ks]))]
    check("targets_consistent", (t[lb == 1].sum((2, 3)) > 0).all() and (t[lb == 0][:, 0].sum((1, 2)) > 0).all() and (t[lb == 0][:, 1].sum((1, 2)) == 0).all())
    check("finite_features", np.isfinite(np.asarray(arr["feat"][:500]).astype(np.float32)).all())
    # rows_fast == absorption.image_rows (d = 2) on one validation crop with B0 candidates (evaluation-side parity only)
    from evaluate import gt_pre, rows_fast, enc
    sys.path.insert(0, str(REPO / "research/PHASE0_20261124/code")); from absorption import image_rows
    from crown_segmentation_research.clean_benchmark.boundary_diagnostics import edge_and_adjacency
    mr, _ = load_maskrcnn(42); cid, img, valid, gt = next(crops("validation"))
    dets, _ = maskrcnn_forward(mr, np.ascontiguousarray(img)); dd = [(enc(m & valid), s) for m, s in dets if (m & valid).any()]
    edge, _ = edge_and_adjacency(gt, valid); fast, _ = rows_fast(gt_pre(gt, edge), dd); ref = {x["id"]: x for x in image_rows(gt, edge, dd)}
    same = fast.keys() == ref.keys() and all(fast[g]["hit"] == ref[g]["hit"] and fast[g]["M"] == ref[g]["M"][2] for g in ref)
    check("rows_fast_equals_image_rows", same, {"crop": cid, "n": len(ref), "absorbed": sum(x["M"][2] for x in ref.values())})
    del mr; torch.cuda.empty_cache()
    # M2 hypothesis-permutation invariance: flipping the start vector flips u, which swaps hypotheses 1 and 2
    torch.manual_seed(0); m = ARMS["M2"]().cuda().eval(); ids = np.arange(16)
    F, M, T, I, y = batch(arr, label, ids, "cuda")
    with torch.no_grad():
        o1 = m(F, M); m.v0.data.mul_(-1); o2 = m(F, M)
    inv = torch.allclose(o1["logit"], o2["logit"], atol=1e-4) and torch.allclose(o1["member_logits"][:, 1], o2["member_logits"][:, 2], atol=1e-4)
    check("M2_permutation_invariant", inv, {"max_logit_diff": float((o1["logit"] - o2["logit"]).abs().max())})
    pc = {k: n_params(c()) for k, c in ARMS.items()}
    check("param_parity_B2_M2_within_25pct", abs(pc["B2"] - pc["M2"]) / pc["M2"] < 0.25, pc)
    # capacity sanity: full-batch, deterministic (no augmentation), 300 steps on 64 balanced items.
    # Each arm must memorise the decision and (B2/M2) the masks. This isolates fitting capacity from minibatch noise.
    order = np.sort(np.concatenate([np.where(label == 1)[0][:32], np.where(label == 0)[0][:32]]))
    small = {k: np.asarray(arr[k][order]) for k in ("feat", "mask", "tgt", "ign")}; lab_small = label[order]
    F = torch.from_numpy(small["feat"]).cuda().float(); Mm = torch.from_numpy(small["mask"]).cuda().float()
    Tt = torch.from_numpy(small["tgt"]).cuda().float(); Ii = torch.from_numpy(small["ign"]).cuda().float()
    yy = torch.from_numpy(lab_small.astype(np.float32)).cuda(); rsel = yy > 0.5
    from arms import MASK_LOSS
    fit = {}
    for name, cls in ARMS.items():
        torch.manual_seed(0); mdl = cls().cuda(); opt = torch.optim.AdamW(mdl.parameters(), lr=2e-3, weight_decay=1e-4)
        for _ in range(300):
            out = mdl(F, Mm); ld = torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], yy); lm = torch.zeros((), device="cuda")
            if MASK_LOSS[name] is not None: lm = MASK_LOSS[name]({k: v[rsel] for k, v in out.items() if k != "logit"}, Tt[rsel], Ii[rsel])
            opt.zero_grad(); (ld + lm).backward(); opt.step()
        fit[name] = [round(float(ld), 4), round(float(lm), 4)]
    ok = all(fit[a][0] < 0.05 for a in ARMS) and all(fit[a][1] < 0.45 for a in ("B2", "M2"))
    check("capacity_overfit", ok, fit)
    status = "PASS" if all(c["ok"] for c in checks.values()) else "FAIL"
    json.dump({"status": status, "config_hash": canonical_hash(STUDY / "CONFIG.json"), "checks": checks}, open(STUDY / "results/PREFLIGHT.json", "w"), indent=1, default=str)
    print("PREFLIGHT", status)


if __name__ == "__main__":
    main()
