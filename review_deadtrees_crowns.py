#!/usr/bin/env python3
"""Keyboard-driven visual review for the Gate 0 polygon queue.

Workflow per polygon:
1. Press 1–7 to choose a semantic category.
2. Press H/M/L to commit with a confidence level.
3. Escape saves and exits; X skips the current polygon.

This utility records decisions but does not decide whether G0B passes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from deadtrees_gate0.audit import AUDIT_DIR, AUDIT_LABELS, _geometry_for_row


KEY_TO_LABEL = {str(index + 1): label for index, label in enumerate(AUDIT_LABELS)}
KEY_TO_CONFIDENCE = {"h": "HIGH", "m": "MEDIUM", "l": "LOW"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_DIR / "polygon_semantic_audit.csv")
    parser.add_argument("--output-csv", type=Path, help="Independent reviewer file; source queue is copied on first use")
    parser.add_argument("--reviewer", required=True, help="Named human reviewer identifier")
    parser.add_argument("--site", type=int)
    parser.add_argument("--include-reviewed", action="store_true")
    return parser.parse_args()


class ReviewSession:
    def __init__(self, source_path: Path, output_path: Path | None, reviewer: str, site: int | None, include_reviewed: bool):
        self.path = output_path or source_path
        self.reviewer = reviewer
        if output_path is not None and not output_path.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self.frame = pd.read_csv(source_path, keep_default_na=False)
            self.frame["audit_label"] = ""
            self.frame["confidence"] = ""
            self.frame["notes"] = ""
            self.frame["reviewer"] = ""
            self.frame["timestamp"] = ""
            self.frame["audit_status"] = "PENDING_HUMAN_REVIEW"
            self.frame.to_csv(output_path, index=False)
        else:
            self.frame = pd.read_csv(self.path, keep_default_na=False)
        mask = pd.Series(True, index=self.frame.index)
        if site is not None:
            mask &= self.frame["site_id"].astype(int).eq(site)
        if not include_reviewed:
            mask &= ~self.frame["audit_label"].isin(AUDIT_LABELS)
        self.indices = self.frame.index[mask].tolist()
        self.position = 0
        self.pending_label = None
        self.pending_notes = ""
        self.fig, self.ax = plt.subplots(figsize=(10, 9))
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

    def save(self):
        self.frame.to_csv(self.path, index=False)

    def draw(self):
        self.ax.clear()
        if self.position >= len(self.indices):
            self.ax.text(0.5, 0.5, "Review queue complete", ha="center", va="center", fontsize=18)
            self.ax.axis("off")
            self.save()
            self.fig.canvas.draw_idle()
            return
        index = self.indices[self.position]
        row = self.frame.loc[index]
        crop, target, neighbors, cover, _ = _geometry_for_row(row)
        self.ax.imshow(crop)
        for ring in cover:
            self.ax.plot(ring[:, 0], ring[:, 1], color="#00d4d8", linewidth=0.8, alpha=0.65)
        for ring in neighbors:
            self.ax.plot(ring[:, 0], ring[:, 1], color="#ffbf00", linewidth=1.0, alpha=0.8)
        for ring in target:
            self.ax.plot(ring[:, 0], ring[:, 1], color="#ff1744", linewidth=2.5)
        self.ax.set_xlim(0, crop.shape[1])
        self.ax.set_ylim(crop.shape[0], 0)
        state = self.pending_label or "choose 1–7"
        self.ax.set_title(
            f"{self.position + 1}/{len(self.indices)}  {row['polygon_id']}\n"
            f"site={row['site_id']}  GSD={float(row['gsd_x_m']) * 100:.2f} cm  "
            f"area={float(row['geometry_area_m2']):.2f} m²\n{state}; then H/M/L | N notes | X skip | Esc save+exit"
        )
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        key = (event.key or "").lower()
        if key in KEY_TO_LABEL:
            self.pending_label = KEY_TO_LABEL[key]
            self.draw()
            return
        if key in KEY_TO_CONFIDENCE and self.pending_label is not None:
            index = self.indices[self.position]
            self.frame.loc[index, "audit_label"] = self.pending_label
            self.frame.loc[index, "confidence"] = KEY_TO_CONFIDENCE[key]
            self.frame.loc[index, "notes"] = self.pending_notes
            self.frame.loc[index, "reviewer"] = self.reviewer
            self.frame.loc[index, "timestamp"] = datetime.now(timezone.utc).isoformat()
            self.frame.loc[index, "audit_status"] = "REVIEWED"
            self.pending_label = None
            self.pending_notes = ""
            self.position += 1
            self.save()
            self.draw()
            return
        if key == "n":
            self.pending_notes = input("Notes for current polygon: ").strip()
            self.draw()
            return
        if key == "x":
            self.pending_label = None
            self.position += 1
            self.draw()
            return
        if key == "escape":
            self.save()
            plt.close(self.fig)

    def run(self):
        if not self.indices:
            print("No matching polygons remain in the review queue.")
            return
        self.draw()
        plt.show()


def main():
    args = parse_args()
    ReviewSession(
        args.audit_csv, args.output_csv, args.reviewer, args.site, args.include_reviewed
    ).run()


if __name__ == "__main__":
    main()
