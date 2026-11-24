"""Train B1, B2 and M2 jointly for one seed: identical batches, order and dihedral augmentation; each arm has its own
optimiser. Refuses to run unless ../results/PREFLIGHT.json is PASS and ../CONFIG.json matches its recorded hash.
Writes ../results/runs/seed{s}/{arm}.pt, TRAIN_LOG.json and PARAMS.json."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from arms import ARMS, MASK_LOSS, n_params  # noqa: E402
STUDY = HERE.parent; CACHE = STUDY / "results/cache"


def canonical_hash(path):
    return hashlib.sha256(json.dumps(json.load(open(path)), sort_keys=True).encode()).hexdigest()


def load_cache():
    idx = json.load(open(CACHE / "index.json")); n = idx["n"]
    arr = {k: np.load(CACHE / f"{k}.npy", mmap_mode="r") for k in ("feat", "mask", "tgt", "ign")}
    return n, arr, np.load(CACHE / "label.npy"), idx


def dihedral(ts, k, flip):
    out = []
    for t in ts:
        t = torch.rot90(t, k, (-2, -1)); out.append(t.flip(-1) if flip else t)
    return out


def batch(arr, label, ids, dev):
    ids = np.sort(ids)
    F = torch.from_numpy(np.ascontiguousarray(arr["feat"][ids])).to(dev).float()
    M = torch.from_numpy(np.ascontiguousarray(arr["mask"][ids])).to(dev).float()
    T = torch.from_numpy(np.ascontiguousarray(arr["tgt"][ids])).to(dev).float()
    I = torch.from_numpy(np.ascontiguousarray(arr["ign"][ids])).to(dev).float()
    y = torch.from_numpy(label[ids].astype(np.float32)).to(dev)
    return F, M, T, I, y


def train(seed, cfg, n_items=None, epochs=None, out_dir=None, log_every=50):
    dev = "cuda"; n, arr, label, idx = load_cache()
    if n_items: n = n_items
    epochs = epochs or cfg["epochs"]; bs = cfg["batch"]
    models, opts = {}, {}
    for name, cls in ARMS.items():
        torch.manual_seed(seed); models[name] = cls().to(dev)
        opts[name] = torch.optim.AdamW(models[name].parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    log, step, t0 = [], 0, time.time()
    for ep in range(epochs):
        perm = rng.permutation(n)
        for b0 in range(0, n - bs + 1, bs):
            F, M, T, I, y = batch(arr, label, perm[b0:b0 + bs], dev)
            k = int(torch.randint(4, (1,), generator=g)); fl = bool(torch.randint(2, (1,), generator=g))
            F, M, T, I = dihedral([F, M, T, I], k, fl)
            rec = {"step": step, "epoch": ep}
            for name, m in models.items():
                out = m(F, M)
                ld = torch.nn.functional.binary_cross_entropy_with_logits(out["logit"], y)
                loss = ld; lm = torch.zeros((), device=dev)
                if MASK_LOSS[name] is not None and y.sum() > 0:
                    r = y > 0.5
                    sub = {kk: vv[r] for kk, vv in out.items() if kk != "logit"}
                    lm = MASK_LOSS[name](sub, T[r], I[r]); loss = ld + lm
                opts[name].zero_grad(set_to_none=True); loss.backward(); opts[name].step()
                rec[name] = [round(float(ld), 5), round(float(lm), 5)]
            if step % log_every == 0:
                print(json.dumps(rec), f"{time.time() - t0:.0f}s", flush=True); log.append(rec)
            step += 1
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=False)
        for name, m in models.items(): torch.save(m.state_dict(), out_dir / f"{name}.pt")
        json.dump({"log": log, "steps": step, "seconds": round(time.time() - t0)}, open(out_dir / "TRAIN_LOG.json", "w"))
        json.dump({k: n_params(m) for k, m in models.items()}, open(out_dir / "PARAMS.json", "w"))
    return models, log


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); a = ap.parse_args()
    pf = json.load(open(STUDY / "results/PREFLIGHT.json"))
    assert pf["status"] == "PASS", "preflight not PASS"
    assert pf["config_hash"] == canonical_hash(STUDY / "CONFIG.json"), "CONFIG.json changed after preflight"
    cfg = json.load(open(STUDY / "CONFIG.json"))
    assert a.seed in cfg["seeds"]
    train(a.seed, cfg, out_dir=STUDY / f"results/runs/seed{a.seed}")


if __name__ == "__main__":
    main()
