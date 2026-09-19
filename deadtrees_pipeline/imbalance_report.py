"""Report class imbalance for the DeadTrees object table, per site and pooled.

This is a standalone diagnostic: `classify_objects.py` already *handles*
imbalance internally via `class_weight="balanced_subsample"`, but nothing
previously measured and published the imbalance itself as a number. Run
after `classify_objects.py` has produced `object_features.csv`:

    python -m deadtrees_pipeline.imbalance_report \
        --object-table DeadTrees/experiments/classification_alive_dead_v1/object_features.csv \
        --output DeadTrees/experiments/classification_alive_dead_v1/imbalance_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_OBJECT_TABLE = Path(
    "DeadTrees/experiments/classification_alive_dead_v1/object_features.csv"
)
DEFAULT_OUTPUT = Path(
    "DeadTrees/experiments/classification_alive_dead_v1/imbalance_report.json"
)


def _counts_and_ratio(frame: pd.DataFrame, column: str, positive: str, negative: str) -> dict:
    n_pos = int((frame[column] == positive).sum())
    n_neg = int((frame[column] == negative).sum())
    return {
        f"n_{positive}": n_pos,
        f"n_{negative}": n_neg,
        "imbalance_ratio_majority_to_minority": (
            max(n_pos, n_neg) / min(n_pos, n_neg) if min(n_pos, n_neg) > 0 else None
        ),
        "minority_class": positive if n_pos <= n_neg else negative,
        "minority_fraction": (
            min(n_pos, n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else None
        ),
    }


def build_report(object_table_path: Path) -> dict:
    frame = pd.read_csv(object_table_path)
    sam2 = frame[frame["source"] == "sam2"]
    sites = sorted(int(site) for site in frame["dataset_id"].unique())

    report = {
        "source_object_table": str(object_table_path),
        "n_rows_total": int(len(frame)),
        "deadwood_vs_other": {
            "pooled": _counts_and_ratio(sam2, "label_name", "positive", "clean_negative"),
            "by_site": {
                str(site): _counts_and_ratio(
                    sam2[sam2["dataset_id"] == site], "label_name", "positive", "clean_negative"
                )
                for site in sites
            },
        },
        "alive_vs_dead": {
            "pooled": _counts_and_ratio(sam2, "condition_label_name", "dead", "alive"),
            "by_site": {
                str(site): _counts_and_ratio(
                    sam2[sam2["dataset_id"] == site], "condition_label_name", "dead", "alive"
                )
                for site in sites
            },
        },
        "condition_label_full_breakdown_by_site": {
            str(site): sam2[sam2["dataset_id"] == site]["condition_label_name"]
            .value_counts()
            .to_dict()
            for site in sites
        },
        "caveat": (
            "alive_vs_dead counts depend on the classify_condition proxy "
            "(tree_cover overlap minus standing_deadwood overlap), which has not "
            "been semantically audited for individual-tree identity "
            "(see audit/annotation_schema.md). Treat these ratios as the "
            "imbalance of the proxy labels, not of biologically confirmed "
            "alive/dead trees."
        ),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-table", type=Path, default=DEFAULT_OBJECT_TABLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.object_table)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
