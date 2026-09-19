#!/usr/bin/env python3
"""CLI for the fail-closed DeadTrees Gate 0 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deadtrees_gate0.audit import (
    AUDIT_DIR,
    build_inventory,
    build_verified_benchmark,
    make_audit_plots,
    prepare_semantic_sample,
    render_audit_examples,
    write_readiness_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Run G0A and create the G0B review queue")
    prepare.add_argument("--per-site", type=int, default=60)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--overwrite-review-queue", action="store_true")
    subparsers.add_parser("report", help="Refresh plots and fail-closed gate reports")
    build = subparsers.add_parser("build", help="Build verified benchmark after an explicit G0B PASS")
    build.add_argument("--confirm-g0b-pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        inventory, sites, overlap = build_inventory()
        audit = prepare_semantic_sample(
            per_site=args.per_site,
            seed=args.seed,
            overwrite_review_queue=args.overwrite_review_queue,
        )
        make_audit_plots()
        render_audit_examples()
        decision = write_readiness_reports()
        print(json.dumps({
            "images": len(inventory), "sites": len(sites), "audit_candidates": len(audit),
            "overlap_pairs": overlap["n_positive_area_overlap_pairs"], **decision,
        }, indent=2))
    elif args.command == "report":
        make_audit_plots()
        render_audit_examples()
        print(json.dumps(write_readiness_reports(), indent=2))
    else:
        result = build_verified_benchmark(confirm_g0b_pass=args.confirm_g0b_pass)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
