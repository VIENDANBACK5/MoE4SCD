"""Training Script for Neuro-Algorithmic Deadwood Segmentation on DeadTrees.

Trains NeuroDeadwoodNet with:
  - Anisotropic eps-Kernel Directional Extents (SumFormer Linear Attention)
  - Spectral Modularity Graph Partitioning
  - Medial Axis Bellman-Ford Shortest Path Relaxation
  - Composite NeuroDeadwoodLoss (Focal + Extent + Modularity + Sparsity)
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
import torch
from torch.utils.data import DataLoader, Dataset

from crown_segmentation_research.methods.dynamic_mask.neuro_deadwood_net import NeuroDeadwoodNet
from crown_segmentation_research.methods.dynamic_mask.neuro_deadwood_loss import NeuroDeadwoodLoss


class DeadTreesNpzDataset(Dataset):
    def __init__(self, target_dir: Path, crop_size: int = 512, random_crop: bool = True):
        self.target_dir = Path(target_dir)
        self.paths = sorted(self.target_dir.glob("*.npz"))
        self.crop_size = crop_size
        self.random_crop = random_crop

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = np.load(self.paths[index])
        image = data["image"] # (H, W, 3) uint8
        probability = data["probability"].astype(np.float32)
        rays = data["rays"].astype(np.float32) # (K, H, W) or (K,)
        
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
                if rays.ndim == 3:
                    rays = rays[:, y0:y0 + CS, x0:x0 + CS]
            else:
                pad_h = max(0, CS - H)
                pad_w = max(0, CS - W)
                image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")[:CS, :CS]
                probability = np.pad(probability, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)[:CS, :CS]
                if rays.ndim == 3:
                    rays = np.pad(rays, ((0, 0), (0, pad_h), (0, pad_w)), mode="constant", constant_values=0)[:, :CS, :CS]

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        prob_t = torch.from_numpy(probability).unsqueeze(0) # (1, H, W)
        rays_t = torch.from_numpy(rays)
        canopy_t = (prob_t > 0).float()
        
        return {
            "image": image_t,
            "probability": prob_t,
            "rays": rays_t,
            "canopy": canopy_t,
        }


def train_one_epoch(
    model: NeuroDeadwoodNet,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NeuroDeadwoodLoss,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    focal_loss_sum = 0.0
    extent_loss_sum = 0.0
    modularity_loss_sum = 0.0
    n_batches = len(dataloader)

    for step, batch in enumerate(dataloader, 1):
        images = batch["image"].to(device, non_blocking=True)
        targets = {
            "probability": batch["probability"].to(device, non_blocking=True),
            "rays": batch["rays"].to(device, non_blocking=True),
            "canopy": batch["canopy"].to(device, non_blocking=True),
        }
        
        optimizer.zero_grad()
        if scaler is not None:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict["total_loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        total_loss += loss.item()
        focal_loss_sum += loss_dict.get("loss_focal", torch.tensor(0.0)).item()
        extent_loss_sum += loss_dict.get("loss_extent", torch.tensor(0.0)).item()
        modularity_loss_sum += loss_dict.get("loss_modularity", torch.tensor(0.0)).item()

        if step % 100 == 0 or step == n_batches:
            print(f"  [Step {step:04d}/{n_batches:04d}] Batch Loss: {loss.item():.4f} (Focal: {loss_dict.get('loss_focal', torch.tensor(0.0)).item():.4f}, Ext: {loss_dict.get('loss_extent', torch.tensor(0.0)).item():.4f})", flush=True)

    return {
        "total_loss": total_loss / max(n_batches, 1),
        "loss_focal": focal_loss_sum / max(n_batches, 1),
        "loss_extent": extent_loss_sum / max(n_batches, 1),
        "loss_modularity": modularity_loss_sum / max(n_batches, 1),
    }


def evaluate(
    model: NeuroDeadwoodNet,
    dataloader: DataLoader,
    criterion: NeuroDeadwoodLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    focal_loss_sum = 0.0
    extent_loss_sum = 0.0
    n_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = {
                "probability": batch["probability"].to(device),
                "rays": batch["rays"].to(device),
                "canopy": batch["canopy"].to(device),
            }
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            
            total_loss += loss_dict["total_loss"].item()
            focal_loss_sum += loss_dict.get("loss_focal", torch.tensor(0.0)).item()
            extent_loss_sum += loss_dict.get("loss_extent", torch.tensor(0.0)).item()

    return {
        "val_total_loss": total_loss / max(n_batches, 1),
        "val_loss_focal": focal_loss_sum / max(n_batches, 1),
        "val_loss_extent": extent_loss_sum / max(n_batches, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Train NeuroDeadwoodNet on DeadTrees")
    parser.add_argument("--train-dir", type=str, default="DeadTrees/star_convex_targets_v1/val") # default to val for fast screen
    parser.add_argument("--val-dir", type=str, default="DeadTrees/star_convex_targets_v1/val")
    parser.add_argument("--output-dir", type=str, default="DeadTrees/experiments/neuro_deadwood_screen")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument("--n-directions", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-eps-kernel", action="store_true")
    parser.add_argument("--no-modularity", action="store_true")
    parser.add_argument("--no-bellman-ford", action="store_true")
    parser.add_argument("--backbone-ckpt", type=str, default="DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth", help="Optional pretrained TreeFlowNet checkpoint for backbone")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "epoch_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Datasets
    train_dataset = DeadTreesNpzDataset(args.train_dir, crop_size=args.crop_size, random_crop=True)
    val_dataset = DeadTreesNpzDataset(args.val_dir, crop_size=args.crop_size, random_crop=False)
    print(f"Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples")

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    # Model
    model = NeuroDeadwoodNet(
        n_directions=args.n_directions,
        pretrained_backbone=True,
        backbone_ckpt=args.backbone_ckpt,
        use_eps_kernel=not args.no_eps_kernel,
        use_modularity=not args.no_modularity,
        use_bellman_ford=not args.no_bellman_ford,
    ).to(device)

    start_epoch = 1
    best_val_loss = float("inf")
    history = []

    if args.resume and Path(args.resume).is_file():
        print(f"Loading checkpoint from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt)
        # Attempt to infer start epoch from filename (e.g. epoch_0036.pth -> 37)
        stem = Path(args.resume).stem
        if stem.startswith("epoch_"):
            try:
                start_epoch = int(stem.split("_")[1]) + 1
                print(f"Resuming from epoch {start_epoch}...")
            except Exception:
                pass

    hist_file = out_dir / "training_history.json"
    if hist_file.is_file():
        try:
            history = json.loads(hist_file.read_text())
            best_val_loss = min([h.get("val_total_loss", float("inf")) for h in history], default=float("inf"))
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = NeuroDeadwoodLoss()
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    print(f"\nStarting training from epoch {start_epoch} to {args.epochs}...")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        record = {"epoch": epoch, **train_metrics, **val_metrics, "time_s": elapsed}
        history.append(record)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} [{elapsed:.1f}s] "
            f"Train Loss: {train_metrics['total_loss']:.4f} (Focal: {train_metrics['loss_focal']:.4f}, Ext: {train_metrics['loss_extent']:.4f}, Mod: {train_metrics['loss_modularity']:.4f}) | "
            f"Val Loss: {val_metrics['val_total_loss']:.4f} (ValFocal: {val_metrics['val_loss_focal']:.4f}, ValExt: {val_metrics['val_loss_extent']:.4f})",
            flush=True
        )

        # Save checkpoint
        torch.save(model.state_dict(), ckpt_dir / f"epoch_{epoch:04d}.pth")
        if val_metrics["val_total_loss"] <= best_val_loss:
            best_val_loss = val_metrics["val_total_loss"]
            torch.save(model.state_dict(), out_dir / "best_model.pth")
            print(f"  --> Saved new best model (val_loss: {best_val_loss:.4f})")

        hist_file.write_text(json.dumps(history, indent=2))

    print(f"\nTraining complete! Results saved to {out_dir}")


if __name__ == "__main__":
    main()
