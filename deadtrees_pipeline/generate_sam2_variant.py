"""Generate a reproducible SAM2 proposal variant without overwriting baseline masks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
from tqdm import tqdm


DEFAULT_IMAGE_ROOT = Path(
    "DeadTrees/raw/image-tiles-1024-global-aerial-sampled-20-random"
)
DEFAULT_OUTPUT = Path("DeadTrees/sam2_masks_high_recall_v1")
DEFAULT_SAM2_REPO = Path("sam2")
DEFAULT_CHECKPOINT = Path("sam2/checkpoints/sam2.1_hiera_large.pt")
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _rgb_uint8(image_path: Path) -> np.ndarray:
    with rasterio.open(image_path) as src:
        image = np.transpose(src.read([1, 2, 3]), (1, 2, 0))
    if not np.isfinite(image).all():
        raise ValueError(f"Image contains NaN/Inf: {image_path}")
    low, high = float(image.min()), float(image.max())
    if low < 0 or high > 255:
        raise ValueError(f"Expected RGB [0,255], got [{low},{high}]: {image_path}")
    return np.rint(image).astype(np.uint8)


def generate(args: argparse.Namespace) -> dict:
    if args.output.resolve() == Path("DeadTrees/sam2_masks").resolve():
        raise ValueError("Refusing to overwrite the raw baseline directory")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.sam2_repo.exists():
        raise FileNotFoundError(args.sam2_repo)
    image_paths = sorted(args.image_root.glob("**/*.tif"))
    if not image_paths:
        raise FileNotFoundError(f"No GeoTIFFs under {args.image_root}")

    cuda_available = torch.cuda.is_available()
    if args.device == "auto":
        device = "cuda" if cuda_available else "cpu"
    else:
        device = args.device
    if device == "cuda" and not cuda_available:
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu" and not args.allow_cpu:
        raise RuntimeError(
            "High-recall SAM2 Hiera-L generation on CPU is intentionally blocked. "
            "Run on a CUDA machine or pass --allow-cpu knowingly."
        )

    sys.path.insert(0, str(args.sam2_repo.resolve()))
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    model = build_sam2(args.model_config, str(args.checkpoint), device=device)
    generator = SAM2AutomaticMaskGenerator(
        model,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        output_mode="binary_mask",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    per_image = {}

    for image_path in tqdm(image_paths, desc=f"Generating SAM2 proposals on {device}"):
        output_path = args.output / f"{image_path.stem}.npz"
        if output_path.exists() and not args.overwrite:
            with np.load(output_path) as existing:
                per_image[image_path.stem] = int(len(existing["masks"]))
            continue
        image = _rgb_uint8(image_path)
        with torch.inference_mode():
            generated = generator.generate(image)
        if generated:
            masks = np.stack([item["segmentation"] for item in generated]).astype(bool)
            stability = np.asarray(
                [item.get("stability_score", 0.0) for item in generated],
                dtype=np.float32,
            )
            predicted_iou = np.asarray(
                [item.get("predicted_iou", 0.0) for item in generated],
                dtype=np.float32,
            )
            areas = np.asarray([item.get("area", 0) for item in generated], dtype=np.int32)
        else:
            masks = np.zeros((0, image.shape[0], image.shape[1]), dtype=bool)
            stability = np.zeros(0, dtype=np.float32)
            predicted_iou = np.zeros(0, dtype=np.float32)
            areas = np.zeros(0, dtype=np.int32)
        np.savez_compressed(
            output_path,
            masks=masks,
            scores=stability,
            stability_scores=stability,
            predicted_iou=predicted_iou,
            areas=areas,
        )
        per_image[image_path.stem] = len(masks)

    manifest = {
        "variant": args.output.name,
        "image_root": str(args.image_root),
        "output": str(args.output),
        "sam2_repo": str(args.sam2_repo),
        "checkpoint": str(args.checkpoint),
        "model_config": args.model_config,
        "device": device,
        "points_per_side": args.points_per_side,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_score_thresh,
        "min_mask_region_area": args.min_mask_region_area,
        "n_images": len(image_paths),
        "n_proposals": sum(per_image.values()),
        "elapsed_seconds": time.time() - started,
        "proposals_per_image": per_image,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sam2-repo", type=Path, default=DEFAULT_SAM2_REPO)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--points-per-side", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.75)
    parser.add_argument("--stability-score-thresh", type=float, default=0.85)
    parser.add_argument("--min-mask-region-area", type=int, default=20)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate(args)
    print(json.dumps({k: v for k, v in manifest.items() if k != "proposals_per_image"}, indent=2))


if __name__ == "__main__":
    main()
