"""Score-aware mask NMS and conservative containment suppression for SAM2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

from deadtrees_pipeline.metrics import mask_bboxes


DEFAULT_INPUT = Path("DeadTrees/sam2_masks_high_recall_v1")
DEFAULT_OUTPUT = Path("DeadTrees/sam2_masks_improved_v1")


def suppress_overlapping_masks(
    masks: np.ndarray,
    quality_scores: np.ndarray,
    min_area_px: int = 50,
    mask_iou_threshold: float = 0.70,
    containment_threshold: float = 0.90,
    containment_min_area_ratio: float = 0.50,
) -> tuple[np.ndarray, dict]:
    """Return retained original indices and auditable removal statistics.

    Masks are visited from highest to lowest SAM quality. A lower-scored mask is
    suppressed if its mask IoU with a retained mask is high, or if the pair is
    nearly contained and similarly sized. The area-ratio guard preserves small
    objects that happen to sit inside a much larger proposal.
    """
    masks = np.asarray(masks, dtype=bool)
    quality_scores = np.asarray(quality_scores, dtype=float)
    if len(masks) != len(quality_scores):
        raise ValueError("masks and quality_scores must have equal length")

    areas = (
        masks.reshape(len(masks), -1).sum(axis=1)
        if len(masks) else np.zeros(0, dtype=np.int64)
    )
    eligible = np.flatnonzero(areas >= min_area_px)
    # Stable secondary area ordering makes exact score ties reproducible.
    order = sorted(
        eligible.tolist(),
        key=lambda index: (quality_scores[index], areas[index], -index),
        reverse=True,
    )
    boxes = mask_bboxes(masks)
    kept: list[int] = []
    reasons = Counter()

    for candidate in order:
        candidate_box = boxes[candidate]
        if candidate_box is None:
            reasons["empty"] += 1
            continue
        cx1, cy1, cx2, cy2 = candidate_box
        suppressed = False
        for winner in kept:
            winner_box = boxes[winner]
            if winner_box is None:
                continue
            wx1, wy1, wx2, wy2 = winner_box
            x1, y1 = max(cx1, wx1), max(cy1, wy1)
            x2, y2 = min(cx2, wx2), min(cy2, wy2)
            if x2 < x1 or y2 < y1:
                continue
            intersection = int(np.logical_and(
                masks[candidate, y1:y2 + 1, x1:x2 + 1],
                masks[winner, y1:y2 + 1, x1:x2 + 1],
            ).sum())
            if intersection == 0:
                continue
            smaller = min(int(areas[candidate]), int(areas[winner]))
            larger = max(int(areas[candidate]), int(areas[winner]))
            union = int(areas[candidate] + areas[winner] - intersection)
            mask_iou = intersection / max(union, 1)
            containment = intersection / max(smaller, 1)
            area_ratio = smaller / max(larger, 1)
            if mask_iou >= mask_iou_threshold:
                reasons["mask_iou_nms"] += 1
                suppressed = True
                break
            if (
                containment >= containment_threshold
                and area_ratio >= containment_min_area_ratio
            ):
                reasons["containment"] += 1
                suppressed = True
                break
        if not suppressed:
            kept.append(candidate)

    reasons["below_min_area"] = int((areas < min_area_px).sum())
    reasons["input"] = len(masks)
    reasons["retained"] = len(kept)
    return np.asarray(kept, dtype=int), dict(reasons)


def process_directory(
    input_dir: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    min_area_px: int = 50,
    mask_iou_threshold: float = 0.70,
    containment_threshold: float = 0.90,
    containment_min_area_ratio: float = 0.50,
    overwrite: bool = False,
) -> dict:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output directories must differ")
    paths = sorted(input_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No NPZ masks under {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    total = Counter()
    per_image = {}

    for input_path in tqdm(paths, desc="Suppressing duplicate/nested masks"):
        output_path = output_dir / input_path.name
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output exists: {output_path}; use --overwrite for a deliberate rerun"
            )
        with np.load(input_path) as data:
            masks = data["masks"].astype(bool)
            stability = data[
                "stability_scores" if "stability_scores" in data.files else "scores"
            ].astype(np.float32)
            predicted_iou = data[
                "predicted_iou" if "predicted_iou" in data.files else "scores"
            ].astype(np.float32)
        quality = predicted_iou.astype(float) * stability.astype(float)
        keep, stats = suppress_overlapping_masks(
            masks,
            quality,
            min_area_px=min_area_px,
            mask_iou_threshold=mask_iou_threshold,
            containment_threshold=containment_threshold,
            containment_min_area_ratio=containment_min_area_ratio,
        )
        kept_masks = masks[keep]
        kept_stability = stability[keep]
        kept_predicted_iou = predicted_iou[keep]
        np.savez_compressed(
            output_path,
            masks=kept_masks,
            scores=kept_stability,
            stability_scores=kept_stability,
            predicted_iou=kept_predicted_iou,
            quality_scores=quality[keep].astype(np.float32),
            areas=(
                kept_masks.reshape(len(kept_masks), -1).sum(axis=1).astype(np.int32)
                if len(kept_masks) else np.zeros(0, dtype=np.int32)
            ),
            source_indices=keep.astype(np.int32),
        )
        per_image[input_path.stem] = stats
        total.update(stats)

    manifest = {
        "method": "score-aware mask IoU NMS plus conservative containment suppression",
        "score": "predicted_iou * stability_score",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "n_images": len(paths),
        "min_area_px": min_area_px,
        "mask_iou_threshold": mask_iou_threshold,
        "containment_threshold": containment_threshold,
        "containment_min_area_ratio": containment_min_area_ratio,
        "totals": dict(total),
        "per_image": per_image,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-area", type=int, default=50)
    parser.add_argument("--mask-iou", type=float, default=0.70)
    parser.add_argument("--containment", type=float, default=0.90)
    parser.add_argument("--containment-min-area-ratio", type=float, default=0.50)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = process_directory(
        input_dir=args.input,
        output_dir=args.output,
        min_area_px=args.min_area,
        mask_iou_threshold=args.mask_iou,
        containment_threshold=args.containment,
        containment_min_area_ratio=args.containment_min_area_ratio,
        overwrite=args.overwrite,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_image"}, indent=2))


if __name__ == "__main__":
    main()
