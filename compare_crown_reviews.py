#!/usr/bin/env python3
"""Compare two independent DeadTrees crown-semantic review files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from deadtrees_gate0.audit import AUDIT_LABELS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_a", type=Path)
    parser.add_argument("review_b", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("audit/reviewer_agreement"))
    return parser.parse_args()


def main():
    args = parse_args()
    left = pd.read_csv(args.review_a, keep_default_na=False)
    right = pd.read_csv(args.review_b, keep_default_na=False)
    required = {"polygon_id", "audit_label", "reviewer"}
    for path, frame in [(args.review_a, left), (args.review_b, right)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} lacks columns: {sorted(missing)}")
        if frame["polygon_id"].duplicated().any():
            raise ValueError(f"{path} contains duplicate polygon_id values")
    merged = left.merge(
        right, on="polygon_id", suffixes=("_a", "_b"), how="inner", validate="one_to_one"
    )
    reviewed = merged[
        merged["audit_label_a"].isin(AUDIT_LABELS)
        & merged["audit_label_b"].isin(AUDIT_LABELS)
    ].copy()
    if reviewed.empty:
        raise ValueError("No polygons have valid decisions from both reviewers")
    reviewed["agrees"] = reviewed["audit_label_a"].eq(reviewed["audit_label_b"])
    raw_agreement = float(reviewed["agrees"].mean())
    kappa = float(cohen_kappa_score(reviewed["audit_label_a"], reviewed["audit_label_b"], labels=list(AUDIT_LABELS)))
    matrix = confusion_matrix(
        reviewed["audit_label_a"], reviewed["audit_label_b"], labels=list(AUDIT_LABELS)
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    disagreements = reviewed[~reviewed["agrees"]].copy()
    disagreements.to_csv(args.output_dir / "disagreements.csv", index=False)
    pd.DataFrame(matrix, index=AUDIT_LABELS, columns=AUDIT_LABELS).to_csv(
        args.output_dir / "confusion_matrix.csv"
    )
    summary = {
        "review_a": str(args.review_a),
        "review_b": str(args.review_b),
        "n_double_reviewed": int(len(reviewed)),
        "n_agreements": int(reviewed["agrees"].sum()),
        "n_disagreements": int((~reviewed["agrees"]).sum()),
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
    }
    (args.output_dir / "agreement.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "agreement.md").write_text(
        "# Crown semantic inter-reviewer agreement\n\n"
        f"- Double-reviewed polygons: {summary['n_double_reviewed']}\n"
        f"- Raw agreement: {raw_agreement:.4f}\n"
        f"- Cohen's kappa: {kappa:.4f}\n"
        f"- Disagreements requiring adjudication: {summary['n_disagreements']}\n\n"
        "Agreement does not by itself establish validity; category definitions, confidence, and adjudication remain part of G0B.\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
