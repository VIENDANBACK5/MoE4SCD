#!/usr/bin/env python3
"""Headless variant of review_deadtrees_crowns.py.

review_deadtrees_crowns.py opens a live matplotlib window and reads
keypresses. That requires a real X11 display ($DISPLAY). In an environment
without one, matplotlib silently falls back to the non-interactive "Agg"
backend: the script exits immediately with no error and no window, which
looks like "nothing happened".

This script does the same job (one polygon at a time, same audit CSV, same
7 labels, same auto-save/resume semantics) but as two separate commands
instead of a live GUI loop, meant to be driven turn-by-turn:

    # 1. Render the next not-yet-reviewed polygon to a PNG and print its info
    python3 review_deadtrees_crowns_headless.py show \
        --output-csv audit/polygon_semantic_audit_chung.csv --reviewer Chung

    # 2. Look at the saved PNG, then commit a decision for that same polygon
    python3 review_deadtrees_crowns_headless.py commit \
        --output-csv audit/polygon_semantic_audit_chung.csv --reviewer Chung \
        --label 1 --confidence H

Repeating show -> commit -> show -> commit ... works through the queue.
Labels: 1=SINGLE_CROWN 2=PARTIAL_CROWN_DIEBACK 3=MULTI_CROWN_GROUP
4=NON_CROWN_DEADWOOD 5=TRUNCATED_AT_TILE_EDGE 6=AMBIGUOUS 7=INVALID_OR_MISALIGNED
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from deadtrees_gate0.audit import AUDIT_DIR, AUDIT_LABELS, _geometry_for_row


DEFAULT_IMAGE_PATH = AUDIT_DIR / "review_current.png"
CONFIDENCE_MAP = {"H": "HIGH", "M": "MEDIUM", "L": "LOW"}


def _load_or_init(source_path: Path, output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.read_csv(source_path, keep_default_na=False)
        frame["audit_label"] = ""
        frame["confidence"] = ""
        frame["notes"] = ""
        frame["reviewer"] = ""
        frame["timestamp"] = ""
        frame["audit_status"] = "PENDING_HUMAN_REVIEW"
        frame.to_csv(output_path, index=False)
        return frame
    return pd.read_csv(output_path, keep_default_na=False)


def _pending_index(frame: pd.DataFrame, site: int | None) -> int | None:
    mask = ~frame["audit_label"].isin(AUDIT_LABELS)
    if site is not None:
        mask &= frame["site_id"].astype(int).eq(site)
    remaining = frame.index[mask]
    return int(remaining[0]) if len(remaining) else None


def cmd_show(args: argparse.Namespace) -> None:
    frame = _load_or_init(args.audit_csv, args.output_csv)
    index = _pending_index(frame, args.site)
    if index is None:
        print(json.dumps({"done": True, "message": "Review queue complete"}, indent=2))
        return
    row = frame.loc[index]
    crop, target_rings, neighbor_rings, cover_rings, _ = _geometry_for_row(row)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(crop)
    for ring in cover_rings:
        ax.plot(ring[:, 0], ring[:, 1], color="#00d4d8", linewidth=0.8, alpha=0.65)
    for ring in neighbor_rings:
        ax.plot(ring[:, 0], ring[:, 1], color="#ffbf00", linewidth=1.0, alpha=0.8)
    for ring in target_rings:
        ax.plot(ring[:, 0], ring[:, 1], color="#ff1744", linewidth=2.5)
    ax.set_xlim(0, crop.shape[1])
    ax.set_ylim(crop.shape[0], 0)
    ax.axis("off")
    args.image_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.image_out, dpi=130, bbox_inches="tight")
    plt.close(fig)

    remaining = int((~frame["audit_label"].isin(AUDIT_LABELS)).sum())
    print(json.dumps({
        "done": False,
        "image_path": str(args.image_out),
        "legend": "red=this polygon, orange=neighbouring standing_deadwood, cyan=tree_cover context",
        "polygon_id": row["polygon_id"],
        "site_id": int(row["site_id"]),
        "gsd_cm": round(float(row["gsd_x_m"]) * 100, 2),
        "area_m2": round(float(row["geometry_area_m2"]), 2),
        "remaining_in_queue": remaining,
        "labels": {i + 1: name for i, name in enumerate(AUDIT_LABELS)},
    }, indent=2))


def cmd_commit(args: argparse.Namespace) -> None:
    frame = _load_or_init(args.audit_csv, args.output_csv)
    index = _pending_index(frame, args.site)
    if index is None:
        print(json.dumps({"done": True, "message": "Nothing pending to commit"}, indent=2))
        return
    label = AUDIT_LABELS[args.label - 1]
    frame.loc[index, "audit_label"] = label
    frame.loc[index, "confidence"] = CONFIDENCE_MAP[args.confidence.upper()]
    frame.loc[index, "notes"] = args.notes or ""
    frame.loc[index, "reviewer"] = args.reviewer
    frame.loc[index, "timestamp"] = datetime.now(timezone.utc).isoformat()
    frame.loc[index, "audit_status"] = "REVIEWED"
    frame.to_csv(args.output_csv, index=False)
    remaining = int((~frame["audit_label"].isin(AUDIT_LABELS)).sum())
    print(json.dumps({
        "done": False,
        "committed_polygon_id": str(frame.loc[index, "polygon_id"]),
        "label": label,
        "remaining_in_queue": remaining,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--audit-csv", type=Path, default=AUDIT_DIR / "polygon_semantic_audit.csv")
    common.add_argument("--output-csv", type=Path, required=True)
    common.add_argument("--reviewer", required=True)
    common.add_argument("--site", type=int)

    show = sub.add_parser("show", parents=[common])
    show.add_argument("--image-out", type=Path, default=DEFAULT_IMAGE_PATH)
    show.set_defaults(func=cmd_show)

    commit = sub.add_parser("commit", parents=[common])
    commit.add_argument("--label", type=int, required=True, choices=range(1, 8))
    commit.add_argument("--confidence", required=True, choices=["H", "M", "L", "h", "m", "l"])
    commit.add_argument("--notes", default="")
    commit.set_defaults(func=cmd_commit)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
