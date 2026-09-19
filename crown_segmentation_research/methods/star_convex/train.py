"""First training screen for StarConvexNet on a small precomputed BAM subset.

This is a screening run (small subset, matching this project's own
"minimal screen before full commitment" convention -- see G4A's 300/50-image
screen), not a final model. Its purpose is to answer one question: does the
architecture + representation combination actually learn a usable signal

# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from real BAM images, at all -- before any larger-scale training investment.

Backbone warm-started from the frozen G1B Mask R-CNN checkpoint (state_dict
keys verified identical, 281/281, before writing this script) rather than
from scratch or generic ImageNet weights, since it is already adapted to
this exact dataset's visual domain.

Loss: BCE on the probability map (dense, all pixels) + smooth-L1 on ray
distances (masked to foreground pixels only, since background pixels have
no meaningful ray target).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from crown_segmentation_research.legacy.discriminative_loss import discriminative_loss
from crown_segmentation_research.methods.star_convex.model import StarConvexNet


def compute_centroid_sdt_targets(
    instance_label: np.ndarray, sdt_max_distance: float = 20.0, centroid_sigma: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """On-the-fly TreeMort-3T-UNet-style centroid heatmap + signed-distance-
    transform targets (arXiv:2503.21438, research.md Addendum 6/8) from an
    (H,W) int instance_label array (0=background) already cached by
    precompute_star_targets.py/precompute_deadtrees_targets.py -- no new
    precompute pass needed, computed per-sample in the DataLoader worker.

    Computed per-instance (looping distance_transform_edt over each
    instance's own mask), not over the combined foreground blob: this is
    what makes the SDT dip toward its boundary value at the shared ridge
    between two *touching* instances, not only at the true canopy/
    background edge -- the explicit inter-instance signal star-convex's
    NMS-only separation lacks (this file's own Stage 2 discussion, and the
    original v6 boundary-weight-loss work this project already did).
    """
    from scipy import ndimage

    height, width = instance_label.shape
    centroid_heatmap = np.zeros((height, width), dtype=np.float32)
    inside_dist = np.zeros((height, width), dtype=np.float32)

    labels = np.unique(instance_label)
    labels = labels[labels != 0]
    if len(labels):
        yy, xx = np.mgrid[0:height, 0:width]
        for label in labels:
            mask = instance_label == label
            inside_dist[mask] = ndimage.distance_transform_edt(mask)[mask]
            ys, xs = np.nonzero(mask)
            cy, cx = ys.mean(), xs.mean()
            gaussian = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * centroid_sigma**2))
            np.maximum(centroid_heatmap, gaussian, out=centroid_heatmap)

    outside_dist = ndimage.distance_transform_edt(instance_label == 0)
    sdt = inside_dist - outside_dist
    sdt = np.clip(sdt, -sdt_max_distance, sdt_max_distance) / sdt_max_distance
    return centroid_heatmap.astype(np.float32), sdt.astype(np.float32)


class PrecomputedStarDataset(torch.utils.data.Dataset):
    def __init__(self, target_dir: Path, compute_centroid_sdt: bool = False, crop_size: int = 1024, random_crop: bool = True):
        self.crop_size = crop_size
        self.random_crop = random_crop
        manifest_path = target_dir / "manifest.csv"
        if manifest_path.exists():
            manifest = pd.read_csv(manifest_path)
            self.paths = [
                target_dir / f"{image_id.replace(':', '__')}.npz"
                for image_id in manifest["image_id"].astype(str)
            ]
        else:
            self.paths = sorted(target_dir.glob("*.npz"))
        self.paths = [p for p in self.paths if p.exists()]
        self.compute_centroid_sdt = compute_centroid_sdt

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.paths[index])
        image = data["image"]  # (H, W, 3) uint8
        probability = data["probability"].astype(np.float32)
        rays = data["rays"].astype(np.float32)

        H, W = image.shape[:2]
        CS = self.crop_size

        if H != CS or W != CS:
            if H >= CS and W >= CS:
                if self.random_crop:
                    y0 = np.random.randint(0, H - CS + 1)
                    x0 = np.random.randint(0, W - CS + 1)
                else:
                    y0 = (H - CS) // 2
                    x0 = (W - CS) // 2
                image = image[y0:y0 + CS, x0:x0 + CS]
                probability = probability[y0:y0 + CS, x0:x0 + CS]
                rays = rays[:, y0:y0 + CS, x0:x0 + CS]
                if "instance_label" in data:
                    inst_crop = data["instance_label"][y0:y0 + CS, x0:x0 + CS]
                else:
                    inst_crop = None
            else:
                pad_h = max(0, CS - H)
                pad_w = max(0, CS - W)
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")[:CS, :CS]
                probability = np.pad(probability, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)[:CS, :CS]
                rays = np.pad(rays, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)[:, :CS, :CS]
                if "instance_label" in data:
                    inst_crop = np.pad(data["instance_label"], ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)[:CS, :CS]
                else:
                    inst_crop = None
        else:
            inst_crop = data.get("instance_label", None)

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        prob_t = torch.from_numpy(probability)
        rays_t = torch.from_numpy(rays)
        canopy_t = (prob_t > 0).float()
        boundary_weight_t = torch.zeros_like(prob_t)

        item = {
            "image": image_t,
            "probability": prob_t,
            "rays": rays_t,
            "canopy": canopy_t,
            "boundary_weight": boundary_weight_t,
        }

        if inst_crop is not None:
            instance_label = inst_crop.astype(np.int64)
            item["instance_label"] = torch.from_numpy(instance_label)
            if self.compute_centroid_sdt:
                centroid_heatmap, sdt = compute_centroid_sdt_targets(inst_crop)
                item["centroid_target"] = torch.from_numpy(centroid_heatmap)
                item["sdt_target"] = torch.from_numpy(sdt)

        return item


def focal_bce_loss(pred: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
    """Per-pixel soft-target generalization of binary focal loss (Lin et al.
    2017): returns a weight that down-weights pixels the model already
    predicts correctly, focusing gradient on hard pixels -- both hard
    negatives (background the model currently over-predicts on) and hard
    positives (touching-instance boundaries, where the target falls to 0 by
    construction but visually similar neighbors give the model little cue).

    Added per star_convex_v3_failure_diagnosis.md: plain (unweighted) BCE
    is dominated by the vast majority of easy background pixels in a
    2048x2048 image where crowns cover a small fraction of the area, which
    dilutes gradient signal on exactly the two failure modes diagnosed
    there (background false positives, dense-stand misses).
    """
    p_correct = target * pred + (1.0 - target) * (1.0 - pred)
    return (1.0 - p_correct).clamp(min=1e-6).pow(gamma)


def compute_loss(
    output: dict[str, torch.Tensor],
    probability_target: torch.Tensor,
    rays_target: torch.Tensor,
    ray_loss_weight: float,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    canopy_target: torch.Tensor | None = None,
    canopy_loss_weight: float = 1.0,
    boundary_weight: torch.Tensor | None = None,
    instance_label: torch.Tensor | None = None,
    embedding_loss_weight: float = 1.0,
    centroid_target: torch.Tensor | None = None,
    centroid_loss_weight: float = 1.0,
    sdt_target: torch.Tensor | None = None,
    sdt_loss_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    probability_pred = output["probability"].squeeze(1)
    rays_pred = output["rays"]

    probability_bce = nn.functional.binary_cross_entropy(probability_pred, probability_target, reduction="none")
    pixel_weight = focal_bce_loss(probability_pred, probability_target, focal_gamma) if use_focal_loss else torch.ones_like(probability_bce)
    if boundary_weight is not None:
        # Additive on top of 1.0 (not multiplicative on the focal weight
        # alone): boundary_weight is 0 almost everywhere (only non-zero in
        # the thin background ridge between two close instances, see
        # star_convex_targets.py::boundary_weight_map), so multiplying it
        # directly against a possibly-already-small focal weight would make
        # its effect vanish exactly where it's meant to matter most.
        pixel_weight = pixel_weight * (1.0 + boundary_weight)
    probability_loss = (pixel_weight * probability_bce).mean()

    foreground = probability_target > 0
    if foreground.any():
        foreground_expanded = foreground.unsqueeze(1).expand_as(rays_pred)
        ray_loss = nn.functional.smooth_l1_loss(
            rays_pred[foreground_expanded], rays_target[foreground_expanded]
        )
    else:
        ray_loss = torch.tensor(0.0, device=rays_pred.device)

    total = probability_loss + ray_loss_weight * ray_loss
    losses = {"total": total, "probability_loss": probability_loss, "ray_loss": ray_loss}

    if canopy_target is not None:
        canopy_pred = output["canopy"].squeeze(1)
        # Plain BCE here, not focal: canopy covers a much larger fraction of
        # the image than any single instance, so the extreme class
        # imbalance focal loss targets is far less severe for this target.
        canopy_loss = nn.functional.binary_cross_entropy(canopy_pred, canopy_target)
        losses["total"] = losses["total"] + canopy_loss_weight * canopy_loss
        losses["canopy_loss"] = canopy_loss

    if centroid_target is not None:
        centroid_pred = output["centroid"].squeeze(1)
        # MSE, matching TreeMort-3T-UNet's centroid-heatmap loss (arXiv:2503.21438):
        # this is a regression target (a soft Gaussian, not a hard 0/1 class), so
        # MSE rather than BCE.
        centroid_loss = nn.functional.mse_loss(centroid_pred, centroid_target)
        losses["total"] = losses["total"] + centroid_loss_weight * centroid_loss
        losses["centroid_loss"] = centroid_loss

    if sdt_target is not None:
        sdt_pred = output["sdt"].squeeze(1)
        # Smooth-L1 over the full image (not foreground-masked): unlike the ray
        # head, the SDT target is meaningful everywhere -- it is exactly what
        # carries the "how far to the nearest instance boundary, including a
        # neighbor's" signal into background-adjacent pixels too.
        sdt_loss = nn.functional.smooth_l1_loss(sdt_pred, sdt_target)
        losses["total"] = losses["total"] + sdt_loss_weight * sdt_loss
        losses["sdt_loss"] = sdt_loss

    if instance_label is not None:
        # discriminative_loss operates on one image at a time (no batch
        # dim); batch_size=1 throughout this project (GPU memory), so
        # index out that dim rather than adding batched-loss plumbing for
        # a case that never occurs here.
        embedding_losses = discriminative_loss(output["embedding"][0], instance_label[0])
        losses["total"] = losses["total"] + embedding_loss_weight * embedding_losses["total"]
        losses["embedding_loss"] = embedding_losses["total"]

    return losses


def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Addendum 11 (research.md): this phase is general crown segmentation,
    # not dead-tree-specific -- BAM's live crowns and DeadTrees' tree_cover
    # instances are both valid positive training signal here, combined with
    # standing_deadwood via a plain ConcatDataset (each source keeps its own
    # ground-truth instance labels; no cross-source geometric merging).
    target_dirs = [args.target_dir] + list(args.extra_target_dirs or [])
    datasets = []
    for target_dir in target_dirs:
        sub_dataset = PrecomputedStarDataset(
            target_dir, compute_centroid_sdt=args.use_centroid_head or args.use_sdt_head,
        )
        print(f"Loaded {len(sub_dataset)} precomputed training images from {target_dir}")
        datasets.append(sub_dataset)
    dataset = datasets[0] if len(datasets) == 1 else torch.utils.data.ConcatDataset(datasets)
    print(f"Total combined training images: {len(dataset)}")
    if len(dataset) == 0:
        raise RuntimeError(f"No precomputed targets found in {target_dirs}")

    model = StarConvexNet(
        n_rays=args.n_rays, pretrained_backbone=False, use_canopy_head=args.use_canopy_head,
        use_embedding_head=args.use_embedding_head, embedding_dim=args.embedding_dim,
        use_centroid_head=args.use_centroid_head, use_sdt_head=args.use_sdt_head,
    ).to(device)
    if args.init_checkpoint is not None:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device, weights_only=True))
        print(f"Initialized full model from {args.init_checkpoint} (fine-tuning)")
    elif args.warm_start_checkpoint is not None:
        mrcnn_state = torch.load(args.warm_start_checkpoint, map_location=device, weights_only=True)
        backbone_state = {
            key[len("backbone."):]: value
            for key, value in mrcnn_state.items()
            if key.startswith("backbone.")
        }
        missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=True)
        print(f"Warm-started backbone from {args.warm_start_checkpoint} (strict=True, no missing/unexpected keys)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=args.num_workers > 0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "training_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    start_epoch = history[-1]["epoch"] + 1 if history else 0
    resume_path = args.output_dir / "star_convex_screen_latest.pth"
    if resume_path.exists() and start_epoch > 0:
        model.load_state_dict(torch.load(resume_path, map_location=device, weights_only=True))
        print(f"Resumed model weights from {resume_path} at epoch {start_epoch}")

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []
        for batch in loader:
            images = batch["image"].to(device)
            probability_target = batch["probability"].to(device)
            rays_target = batch["rays"].to(device)
            canopy_target = batch["canopy"].to(device) if args.use_canopy_head else None
            boundary_weight = batch["boundary_weight"].to(device) if args.use_boundary_weight else None
            instance_label = batch["instance_label"].to(device) if args.use_embedding_head else None
            centroid_target = batch["centroid_target"].to(device) if args.use_centroid_head else None
            sdt_target = batch["sdt_target"].to(device) if args.use_sdt_head else None

            try:
                output = model(images)
                losses = compute_loss(
                    output, probability_target, rays_target, args.ray_loss_weight,
                    use_focal_loss=args.use_focal_loss, focal_gamma=args.focal_gamma,
                    canopy_target=canopy_target, canopy_loss_weight=args.canopy_loss_weight,
                    boundary_weight=boundary_weight,
                    instance_label=instance_label, embedding_loss_weight=args.embedding_loss_weight,
                    centroid_target=centroid_target, centroid_loss_weight=args.centroid_loss_weight,
                    sdt_target=sdt_target, sdt_loss_weight=args.sdt_loss_weight,
                )
                optimizer.zero_grad()
                losses["total"].backward()
                optimizer.step()
                epoch_losses.append({k: float(v) for k, v in losses.items()})
            except torch.cuda.OutOfMemoryError:
                # Long unattended runs on a shared GPU can hit transient
                # memory pressure from *other* processes (measured once
                # already, see design_docs/star_convex_implementation_status.md).
                # Skip this batch rather than crash the whole run; do not
                # silently continue forever if it keeps happening (that
                # would just waste the run), but a handful of skips is fine.
                print(f"  epoch {epoch}: OOM on one batch, skipping it", flush=True)
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue

        if not epoch_losses:
            print(f"epoch {epoch}: every batch OOM'd, stopping early", flush=True)
            break
        mean_losses = {
            key: float(np.mean([entry[key] for entry in epoch_losses]))
            for key in epoch_losses[0]
        }
        history.append({"epoch": epoch, **mean_losses})
        history_path.write_text(json.dumps(history, indent=2))
        canopy_part = f" canopy={mean_losses['canopy_loss']:.4f}" if "canopy_loss" in mean_losses else ""
        embedding_part = f" embed={mean_losses['embedding_loss']:.4f}" if "embedding_loss" in mean_losses else ""
        centroid_part = f" centroid={mean_losses['centroid_loss']:.4f}" if "centroid_loss" in mean_losses else ""
        sdt_part = f" sdt={mean_losses['sdt_loss']:.4f}" if "sdt_loss" in mean_losses else ""
        print(f"epoch {epoch}: total={mean_losses['total']:.4f} "
              f"prob={mean_losses['probability_loss']:.4f} ray={mean_losses['ray_loss']:.4f}{canopy_part}{embedding_part}{centroid_part}{sdt_part}", flush=True)

        if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), resume_path)
            if args.save_epoch_checkpoints:
                epoch_dir = args.output_dir / "epoch_checkpoints"
                epoch_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), epoch_dir / f"epoch_{epoch:04d}.pth")

    torch.save(model.state_dict(), args.output_dir / "star_convex_screen.pth")
    history_path.write_text(json.dumps(history, indent=2))
    print(f"Saved final checkpoint + history to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--extra-target-dirs", type=Path, nargs="*", default=None, help="additional precomputed-target directories to combine via ConcatDataset (e.g. BAM's cache alongside DeadTrees') for general (source-agnostic) crown segmentation training, research.md Addendum 11")
    parser.add_argument("--n-rays", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker processes to overlap disk I/O with GPU compute; 0=synchronous (measured as a real bottleneck once instance_label/boundary_weight channels made cached targets larger -- see star_convex_v7 training result)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ray-loss-weight", type=float, default=0.1)
    parser.add_argument("--use-focal-loss", action="store_true", help="soft-target focal loss for probability head instead of plain BCE (see focal_bce_loss docstring); off by default to preserve exact reproducibility of v2/v3 runs")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--use-canopy-head", action="store_true", help="add a binary canopy/background head (target derived for free from probability>0) to gate out background false positives at decode time; off by default to preserve v2-v4 reproducibility")
    parser.add_argument("--canopy-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-boundary-weight", action="store_true", help="up-weight the probability-head loss on the thin background ridge between close instances (code/add_boundary_weights.py must have been run on --target-dir first); off by default to preserve v2-v5 reproducibility")
    parser.add_argument("--use-embedding-head", action="store_true", help="add a per-pixel instance-embedding head trained with the De Brabandere discriminative (pull/push) loss (code/add_instance_labels.py must have been run on --target-dir first); off by default to preserve v2-v6 reproducibility")
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--embedding-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-centroid-head", action="store_true", help="add a Gaussian instance-centroid heatmap head (MSE loss), TreeMort-3T-UNet-style (arXiv:2503.21438, research.md Addendum 6/8); target computed on-the-fly from instance_label, no separate precompute needed; off by default")
    parser.add_argument("--centroid-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-sdt-head", action="store_true", help="add a signed-distance-transform/boundary head (smooth-L1 loss), TreeMort-3T-UNet-style; target computed on-the-fly from instance_label, no separate precompute needed; off by default")
    parser.add_argument("--sdt-loss-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--save-epoch-checkpoints", action="store_true", help="also save a numbered snapshot at every --checkpoint-every interval (for post-hoc best-epoch selection on a small dataset), instead of only overwriting the single latest/final checkpoint; off by default to preserve BAM v2-v8 disk usage/behavior")
    parser.add_argument(
        "--warm-start-checkpoint", type=Path,
        default=Path("experiments/g1b_baselines/checkpoints/maskrcnn_seed42_best.pth"),
    )
    parser.add_argument(
        "--init-checkpoint", type=Path, default=None,
        help="full StarConvexNet checkpoint (e.g. an earlier run's best_model) to initialize every weight from, "
        "for fine-tuning on a new target-dir; takes priority over --warm-start-checkpoint (backbone-only) when given",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("crown_segmentation_research/experiments/star_convex_screen"))
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
