"""
phase4_package.py — Package the SECOND-OC benchmark into final distribution files.

Reads all phase outputs → writes:
    SECOND-OC/benchmark.json       (full benchmark, all 1694 pairs)
    SECOND-OC/tier1/tier1.json     (high-confidence subset for VLM eval)
    SECOND-OC/stats.json           (dataset statistics for paper)

Tier 1 criteria (high-confidence subset):
    - change_type in {appeared, disappeared, semantic_change}
    - t2_conf >= TIER1_CONF
    - area_px  >= TIER1_MIN_AREA
    - sam2_score >= TIER1_SAM2_SCORE
"""
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from config import ANN_DIR, TIER1_DIR, OUT_ROOT

TIER1_CONF       = 0.65
TIER1_MIN_AREA   = 300
TIER1_SAM2_SCORE = 0.85


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    # ── Load all phase outputs ─────────────────────────────────────────────────
    for p in [ANN_DIR / "instances_T1.json",
              ANN_DIR / "instances_T2.json",
              ANN_DIR / "change_annotations.json",
              ANN_DIR / "captions.jsonl"]:
        assert p.exists(), f"Missing: {p} — run previous phases first"

    with open(ANN_DIR / "instances_T1.json") as f:
        inst_T1 = json.load(f)
    with open(ANN_DIR / "instances_T2.json") as f:
        inst_T2 = json.load(f)
    with open(ANN_DIR / "change_annotations.json") as f:
        change_data = json.load(f)

    captions_by_change_id = {
        r["change_id"]: r["caption"]
        for r in load_jsonl(ANN_DIR / "captions.jsonl")
    }

    # ── Build per-stem records ─────────────────────────────────────────────────
    benchmark: dict = {}
    tier1_records: list[dict] = []
    type_counter: Counter = Counter()
    class_counter: Counter = Counter()

    for stem, changes in change_data["changes"].items():
        stem_record = {
            "stem":        stem,
            "instances_T1": inst_T1["instances"].get(stem, []),
            "instances_T2": inst_T2["instances"].get(stem, []),
            "changes":     [],
        }

        for ch in changes:
            caption = captions_by_change_id.get(ch["change_id"], "")
            record = {**ch, "caption": caption}
            stem_record["changes"].append(record)
            type_counter[ch["change_type"]] += 1
            if ch["change_type"] != "unchanged":
                class_counter[f"{ch['class_name_T1']}→{ch['class_name_T2']}"] += 1

            # Tier 1 filter
            if (ch["change_type"] != "unchanged"
                    and ch["t2_conf"] >= TIER1_CONF
                    and ch["area_px"] >= TIER1_MIN_AREA
                    and ch["sam2_score"] >= TIER1_SAM2_SCORE
                    and caption):
                tier1_records.append(record)

        benchmark[stem] = stem_record

    # ── Save benchmark.json ────────────────────────────────────────────────────
    out_benchmark = OUT_ROOT / "benchmark.json"
    with open(out_benchmark, "w") as f:
        json.dump({
            "name":        "SECOND-OC",
            "version":     "1.0",
            "n_images":    len(benchmark),
            "n_changes":   sum(type_counter.values()),
            "benchmark":   benchmark,
        }, f, indent=2)
    print(f"✅ benchmark.json: {len(benchmark)} images, "
          f"{sum(type_counter.values())} change events")

    # ── Save tier1.json ────────────────────────────────────────────────────────
    TIER1_DIR.mkdir(parents=True, exist_ok=True)
    out_tier1 = TIER1_DIR / "tier1.json"
    with open(out_tier1, "w") as f:
        json.dump({
            "name":     "SECOND-OC Tier1",
            "version":  "1.0",
            "criteria": {
                "t2_conf":       TIER1_CONF,
                "min_area_px":   TIER1_MIN_AREA,
                "sam2_score":    TIER1_SAM2_SCORE,
            },
            "n_records": len(tier1_records),
            "records":   tier1_records,
        }, f, indent=2)
    print(f"✅ tier1.json: {len(tier1_records)} high-confidence records")

    # ── Compute & save stats ───────────────────────────────────────────────────
    total = sum(type_counter.values())
    stats = {
        "n_images":           len(benchmark),
        "n_change_events":    total,
        "change_type_dist":   dict(type_counter),
        "change_type_pct":    {k: round(v / total * 100, 1)
                               for k, v in type_counter.items()},
        "top_class_transitions": dict(class_counter.most_common(20)),
        "tier1_n":            len(tier1_records),
        "tier1_pct":          round(len(tier1_records) / max(total, 1) * 100, 1),
    }

    out_stats = OUT_ROOT / "stats.json"
    with open(out_stats, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n── Dataset Statistics ──────────────────────────")
    for ct, n in sorted(type_counter.items()):
        print(f"   {ct:<20}: {n:>6}  ({n/total*100:.1f}%)")
    print(f"   {'Tier1 subset':<20}: {len(tier1_records):>6}  ({stats['tier1_pct']}%)")
    print(f"\n   Saved: {out_benchmark}")
    print(f"   Saved: {out_tier1}")
    print(f"   Saved: {out_stats}")

    assert len(benchmark) == change_data["n_images"], "Image count mismatch"
    assert len(tier1_records) > 100, \
        f"[FAIL] Only {len(tier1_records)} tier1 records — check TIER1 thresholds"
    print("\n   ✅ Validation passed")


if __name__ == "__main__":
    main()
