"""Cache raw (probability, rays) forward-pass outputs for a trained
StarConvexNet checkpoint on a precomputed image set, so decode-threshold
sweeps (code/sweep_star_convex_decode.py) can run decode-only without
repeating the forward pass.

Same "cache once, sweep cheaply" pattern as code/evaluate_merge.py. Written
as a reusable script (rather than one-off inline commands) because
star_convex_v2_decode_sweep.md's own caveat is that the adopted threshold
config is checkpoint-specific and must be re-swept for every future
checkpoint -- i.e. this step is not a one-time thing.
"""

from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from crown_segmentation_research.methods.star_convex.model import StarConvexNet


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = StarConvexNet(
        n_rays=args.n_rays, pretrained_backbone=False, use_canopy_head=args.use_canopy_head,
        use_embedding_head=args.use_embedding_head, embedding_dim=args.embedding_dim,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    manifest = pd.read_csv(args.target_dir / "manifest.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i, image_id in enumerate(manifest["image_id"].astype(str)):
            npz_path = args.target_dir / f"{image_id.replace(':', '__')}.npz"
            if not npz_path.exists():
                continue
            out_path = args.output_dir / f"{image_id.replace(':', '__')}.npz"
            if out_path.exists() and not args.overwrite:
                continue
            data = np.load(npz_path)
            image = torch.from_numpy(data["image"]).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            output = model(image)
            probability = output["probability"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float16)
            rays = output["rays"].squeeze(0).cpu().numpy().astype(np.float16)
            save_kwargs = {"probability": probability, "rays": rays}
            if args.use_canopy_head:
                save_kwargs["canopy"] = output["canopy"].squeeze(0).squeeze(0).cpu().numpy().astype(np.float16)
            if args.use_embedding_head:
                save_kwargs["embedding"] = output["embedding"].squeeze(0).cpu().numpy().astype(np.float16)
            np.savez_compressed(out_path, **save_kwargs)
            print(f"[{i + 1}/{len(manifest)}] {image_id}: cached", flush=True)

    print(f"done, raw outputs at {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True, help="precomputed dir with manifest.csv + npz images (e.g. .../val)")
    parser.add_argument("--n-rays", type=int, default=16)
    parser.add_argument("--use-canopy-head", action="store_true", help="checkpoint was trained with a canopy head (code/train_star_convex.py); also cache its output")
    parser.add_argument("--use-embedding-head", action="store_true", help="checkpoint was trained with a discriminative-embedding head; also cache its output")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
