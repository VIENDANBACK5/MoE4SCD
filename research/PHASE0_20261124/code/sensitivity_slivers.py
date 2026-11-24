"""POST-HOC sensitivity (added 2026-11-24 after the LIVECell previews showed 2–30 px sliver annotations in the strong
stratum). Not part of the registered Phase 0 definitions; reported alongside them.

Filter: keep GT instances whose maximum inscribed radius ('thick', px) is ≥ 3. Typical sizes:
- a real BV2 cell is about 6 px;
- a 1 m² QuebecTrees crown is about 19 px;
- slivers are ≤ 2 px.

Size / density edges are recomputed on the kept set; the standardisation is otherwise unchanged (d = 2).

Limitation: neighbour asymmetry r and absorption M were computed with slivers still present as neighbours. Slivers
rarely produce r ≥ 4 for a real cell, but M could in rare cases count a merge with a sliver.

Reads ../results/rows_*.pkl; writes ../results/sensitivity_slivers.json."""
import json, pickle, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from absorption import summarise  # noqa: E402
R = HERE.parent / "results"; THICK = 3.0


def run(rows, key):
    out = {}
    for tag, rr in (("all", rows), ("thick>=3", [x for x in rows if x["thick"] >= THICK])):
        a = np.array([x["area"] for x in rr]); dn = np.array([x["density"] for x in rr])
        s = summarise([dict(x) for x in rr], key, np.quantile(a, [.2, .4, .6, .8]), np.quantile(dn, [1 / 3, 2 / 3]), d=2)
        out[tag] = {"n": len(rr), "strata": {k: (v["n"], v["rate"]) for k, v in s["strata"].items()}, "standardised": s["standardised"],
                    "consequence": s["consequence"]}
    st_all = [x for x in rows if x["r"][2] >= 4]
    out["strong_share_slivers"] = float(np.mean([x["thick"] < THICK for x in st_all])) if st_all else None
    out["absorbed_strong_share_slivers"] = float(np.mean([x["thick"] < THICK for x in st_all if x["M"][2]])) if any(x["M"][2] for x in st_all) else None
    return out


def main():
    res = {}
    for f in sorted(R.glob("rows_*.pkl")):
        rows = pickle.load(open(f, "rb")); key = "tile" if "quebectrees" in f.name else "img"
        res[f.stem] = run(rows, key)
        if "livecell" in f.stem:
            res[f.stem]["by_ctype_thick"] = {}
            for ct in sorted({x["ctype"] for x in rows}):
                rr = [x for x in rows if x["ctype"] == ct and x["thick"] >= THICK]
                a = np.array([x["area"] for x in rr]); dn = np.array([x["density"] for x in rr])
                s = summarise([dict(x) for x in rr], key, np.quantile(a, [.2, .4, .6, .8]), np.quantile(dn, [1 / 3, 2 / 3]), d=2)
                res[f.stem]["by_ctype_thick"][ct] = {"n_strong": s["strata"]["strong"]["n"], "standardised": s["standardised"]}
        v = res[f.stem]
        print(f.stem, "sliver share of strong", round(v["strong_share_slivers"], 3), "of absorbed strong", None if v["absorbed_strong_share_slivers"] is None else round(v["absorbed_strong_share_slivers"], 3))
        for tag in ("all", "thick>=3"):
            st = v[tag]["standardised"]
            print("  ", tag, "n", v[tag]["n"], {k: (n, None if r is None else round(r, 3)) for k, (n, r) in v[tag]["strata"].items()},
                  "std diff", None if st["diff"] is None else round(st["diff"], 3), None if st["diff_ci"] is None else [round(x, 3) for x in st["diff_ci"]], flush=True)
    json.dump(res, open(R / "sensitivity_slivers.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
