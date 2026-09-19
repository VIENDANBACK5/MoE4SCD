"""Training script for TreeFlowNet / OmniCrown on combined multi-source datasets.

Trains the multi-task flow, SDT boundary, centroid heatmap, and canopy gating network.
Supports combining BAM (1,439 tiles), DeadTrees TreeCover (787 tiles), and
DeadTrees Deadwood (3,600 tiles).
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
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.targets import compute_tree_flow_targets


class TreeFlowDataset(Dataset):
    """Dataset that loads (image, instance_label), crops/pads to crop_size, and computes flow targets on-the-fly."""

    def __init__(self, target_dir: Path, crop_size: int = 1024, random_crop: bool = True):
        self.target_dir = target_dir
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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.paths[index])
        image = data["image"]  # (H, W, 3) uint8

        if "instance_label" in data:
            instance_label = data["instance_label"].astype(np.int32)
        else:
            prob = data.get("probability", np.zeros(image.shape[:2], dtype=np.float32))
            instance_label = (prob > 0.5).astype(np.int32)

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
                instance_label = instance_label[y0:y0 + CS, x0:x0 + CS]
            else:
                pad_h = max(0, CS - H)
                pad_w = max(0, CS - W)
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                instance_label = np.pad(instance_label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)
                image = image[:CS, :CS]
                instance_label = instance_label[:CS, :CS]

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        flow_t, sdt_t, centroid_t, canopy_t = compute_tree_flow_targets(instance_label)

        return {
            "image": image_t,
            "flow_target": torch.from_numpy(flow_t),
            "sdt_target": torch.from_numpy(sdt_t),
            "centroid_target": torch.from_numpy(centroid_t),
            "canopy_target": torch.from_numpy(canopy_t),
        }


def tree_flow_loss(
    predictions: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    flow_weight: float = 1.0,
    sdt_weight: float = 2.0,
    centroid_weight: float = 5.0,
    canopy_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute combined multi-task loss for TreeFlowNet."""
    flow_pred = predictions["flow"]
    sdt_pred = predictions["sdt"]
    centroid_pred = predictions["centroid"]
    canopy_pred = predictions["canopy"]

    flow_gt = targets["flow_target"]
    sdt_gt = targets["sdt_target"]
    centroid_gt = targets["centroid_target"]
    canopy_gt = targets["canopy_target"]

    # 1. Flow loss: Cosine similarity + L1 on foreground canopy pixels
    fg_mask = (canopy_gt > 0.5).expand_as(flow_pred)
    if fg_mask.sum() > 0:
        flow_p_fg = flow_pred[fg_mask]
        flow_g_fg = flow_gt[fg_mask]
        # Cosine distance: 1 - dot product
        cos_sim = F.cosine_similarity(flow_pred, flow_gt, dim=1, eps=1e-6)
        fg_single = canopy_gt[:, 0] > 0.5
        cos_loss = (1.0 - cos_sim[fg_single]).mean() if fg_single.sum() > 0 else torch.tensor(0.0, device=flow_pred.device)
        l1_loss = F.l1_loss(flow_p_fg, flow_g_fg)
        loss_flow = cos_loss + l1_loss
    else:
        loss_flow = torch.tensor(0.0, device=flow_pred.device)

    # 2. SDT loss: Smooth L1 across all pixels
    loss_sdt = F.smooth_l1_loss(sdt_pred, sdt_gt)

    # 3. Centroid loss: MSE on Gaussian centroid heatmap
    loss_centroid = F.mse_loss(centroid_pred, centroid_gt)

    # 4. Canopy loss: BCE + Dice loss
    bce_loss = F.binary_cross_entropy(canopy_pred, canopy_gt)
    intersection = (canopy_pred * canopy_gt).sum()
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (canopy_pred.sum() + canopy_gt.sum() + 1.0)
    loss_canopy = bce_loss + dice_loss

    total_loss = (
        flow_weight * loss_flow
        + sdt_weight * loss_sdt
        + centroid_weight * loss_centroid
        + canopy_weight * loss_canopy
    )

    return {
        "total": total_loss,
        "flow_loss": loss_flow,
        "sdt_loss": loss_sdt,
        "centroid_loss": loss_centroid,
        "canopy_loss": loss_canopy,
    }


def run_training(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)

    # Build multi-source dataset
    all_dirs = [args.target_dir] + (args.extra_target_dirs or [])
    datasets = []
    for d in all_dirs:
        ds = TreeFlowDataset(d)
        print(f"Loaded {len(ds)} samples from {d}", flush=True)
        datasets.append(ds)

    combined_dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    print(f"Total training samples combined: {len(combined_dataset)}", flush=True)

    loader = DataLoader(
        combined_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = TreeFlowNet(pretrained_backbone=not args.no_pretrained).to(device)

    # Warm-start backbone from G1B Mask R-CNN if requested
    if args.warm_start_from and Path(args.warm_start_from).exists():
        ckpt = torch.load(args.warm_start_from, map_location=device, weights_only=True)
        state_dict = ckpt.get("model", ckpt)
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                backbone_dict[k.replace("backbone.", "")] = v
        missing, unexpected = model.backbone.load_state_dict(backbone_dict, strict=False)
        print(f"Warm-started backbone from {args.warm_start_from} (loaded {len(backbone_dict)} keys)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "epoch_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    history = []

    # Auto-resume if latest checkpoint exists
    existing_ckpts = sorted(ckpt_dir.glob("epoch_*.pth"))
    if existing_ckpts and args.auto_resume:
        latest = existing_ckpts[-1]
        start_epoch = int(latest.stem.split("_")[1]) + 1
        model.load_state_dict(torch.load(latest, map_location=device, weights_only=True))
        print(f"Auto-resumed model from {latest} at epoch {start_epoch}", flush=True)
        history_path = output_dir / "training_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text())

    print(f"\nStarting TreeFlowNet training for {args.epochs} epochs...", flush=True)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        model.train()
        epoch_losses = {"total": 0.0, "flow_loss": 0.0, "sdt_loss": 0.0, "centroid_loss": 0.0, "canopy_loss": 0.0}
        n_batches = 0

        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in batch.items() if k != "image"}

            optimizer.zero_grad()
            predictions = model(images)
            losses = tree_flow_loss(predictions, targets)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            for k, v in losses.items():
                epoch_losses[k] += v.item()
            n_batches += 1

        scheduler.step()
        elapsed = time.time() - t0
        avg_losses = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}
        avg_losses["epoch"] = epoch
        avg_losses["elapsed_s"] = elapsed
        avg_losses["lr"] = optimizer.param_groups[0]["lr"]
        history.append(avg_losses)

        print(
            f"Epoch {epoch:03d}/{args.epochs} [{elapsed:.1f}s]: "
            f"total={avg_losses['total']:.4f} | "
            f"flow={avg_losses['flow_loss']:.4f} | "
            f"sdt={avg_losses['sdt_loss']:.4f} | "
            f"centroid={avg_losses['centroid_loss']:.4f} | "
            f"canopy={avg_losses['canopy_loss']:.4f}",
            flush=True,
        )

        # Save checkpoint
        if (epoch + 1) % args.checkpoint_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = ckpt_dir / f"epoch_{epoch:04d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    print(f"Training completed. Checkpoints saved in {ckpt_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TreeFlowNet on combined remote sensing datasets")
    parser.add_argument("--target-dir", type=Path, default=Path("DeadTrees/star_convex_targets_v1/train"))
    parser.add_argument("--extra-target-dirs", type=Path, nargs="*", default=[
        Path("crown_segmentation_research/experiments/star_convex_targets_v2/train"),
        Path("DeadTrees/star_convex_targets_v1/train_treecover"),
    ])
    parser.add_argument("--output-dir", type=Path, default=Path("DeadTrees/experiments/tree_flow_unified_v1"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--warm-start-from", type=Path, default=Path("experiments/g1b_baselines/checkpoints/maskrcnn_seed42_best.pth"))
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--auto-resume", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
