"""
Training script for TreeFlowNet on exhaustive all-tree crown data.

Train on combined BAM + Quebec + (optionally DeadTrees tree_cover) dataset.
Warm-starts from existing ep39 checkpoint (DeadTrees-trained).
Evaluates on BAM eval split after every N epochs using COCO AP.

Usage (quick dev run):
    python crown_segmentation_research/methods/tree_flow/train.py \
        --warm-start DeadTrees/experiments/tree_flow_unified_v1/epoch_checkpoints/epoch_0039.pth \
        --epochs 60 \
        --batch 4 \
        --lr 3e-4 \
        --out-dir DeadTrees/experiments/tree_flow_exhaustive_v1

Usage (from scratch, no warm-start):
    python crown_segmentation_research/methods/tree_flow/train.py \
        --epochs 80 --batch 4 --lr 1e-3 \
        --out-dir DeadTrees/experiments/tree_flow_exhaustive_scratch
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ── path setup ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
from crown_segmentation_research.methods.tree_flow.loss import TreeFlowLoss
from crown_segmentation_research.datasets.exhaustive_tree_dataset import (
    build_exhaustive_tree_dataset,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper (lightweight F1@0.5)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def quick_eval_f1(model: TreeFlowNet, val_loader: DataLoader, decode_cfg: dict) -> float:
    """
    Fast F1@IoU0.5 on a small validation sample.
    Returns F1 score (0-1).
    """
    from crown_segmentation_research.methods.tree_flow.decode import decode

    model.eval()
    tp = fp = fn = 0

    for batch in val_loader:
        imgs    = batch["image"].to(DEVICE)         # [B, 3, H, W]
        gt_can  = batch["canopy"].to(DEVICE)        # [B, 1, H, W]
        gt_sdt  = batch["sdt"].to(DEVICE)           # [B, 1, H, W]

        outputs = model(imgs)

        B = imgs.shape[0]
        for b in range(B):
            out_b = {k: v[b] for k, v in outputs.items()}
            preds = decode(out_b, device=DEVICE, **decode_cfg)

            # Count GT instances from SDT (connected components of sdt > 0)
            from scipy import ndimage
            sdt_b  = gt_sdt[b, 0].cpu().numpy()
            fg_b   = sdt_b > 0
            labeled, n_gt = ndimage.label(fg_b)
            n_pred = len(preds)

            # Match each pred to a GT component
            matched_gt = set()
            for inst in preds:
                pmask = inst["mask"]
                # Which GT component does this overlap most?
                if pmask.shape != fg_b.shape:
                    continue
                overlap_ids = labeled[pmask & fg_b]
                if len(overlap_ids) == 0:
                    fp += 1
                    continue
                best_gt = int(overlap_ids.max())
                # Compute IoU with that GT region
                gt_mask = labeled == best_gt
                inter = (pmask & gt_mask).sum()
                union = (pmask | gt_mask).sum()
                iou = inter / (union + 1e-6)
                if iou >= 0.5 and best_gt not in matched_gt:
                    tp += 1
                    matched_gt.add(best_gt)
                else:
                    fp += 1

            fn += n_gt - len(matched_gt)

    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)
    model.train()
    return float(f1)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    import random
    import numpy as np
    from crown_segmentation_research.lineage import check_clean_init
    check_clean_init(args.warm_start, args.allow_leaked_init)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "epoch_checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}", flush=True)
    print(f"TreeFlowNet — Exhaustive All-Tree Training", flush=True)
    print(f"  device  : {DEVICE}", flush=True)
    print(f"  epochs  : {args.epochs}", flush=True)
    print(f"  batch   : {args.batch}", flush=True)
    print(f"  lr      : {args.lr}", flush=True)
    print(f"  out_dir : {out_dir}", flush=True)
    print(f"{'='*70}\n", flush=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    print("Building training dataset...", flush=True)
    train_ds = build_exhaustive_tree_dataset(
        patch_size   = args.patch_size,
        augment      = True,
        use_bam      = True,
        use_quebec   = True,
        use_deadtrees= False,
        # Z1 only by default. V1 itself read every cached chip of Z1, Z2 and Z3 (the test zone): --legacy-v1-data
        quebec_zones = ("Z1", "Z2", "Z3") if args.legacy_v1_data else ("Z1",),
        quebec_legacy_v1 = args.legacy_v1_data,
    )
    print(f"  Total training samples: {len(train_ds):,}", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size  = args.batch,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = False,
        drop_last   = True,
    )

    # Small val subset for quick F1 monitoring (use first 50 BAM eval images)
    print("\nBuilding val dataset (BAM eval subset)...", flush=True)
    from crown_segmentation_research.datasets.exhaustive_tree_dataset import BAMCocoDataset
    bam_dir   = REPO / "data/itc_benchmarks/extracted/bam_coco2048"
    bam_arch  = REPO / "data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"
    val_ds = BAMCocoDataset(
        json_path    = bam_dir / "instances_tree_eval2023.json",
        archive_path = bam_arch,
        patch_size   = args.patch_size,
        augment      = False,
    )
    # Only use first 50 for speed
    from torch.utils.data import Subset
    val_ds = Subset(val_ds, list(range(min(50, len(val_ds)))))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"  Val samples: {len(val_ds)}", flush=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TreeFlowNet(pretrained_backbone=not args.warm_start)  # default was a randomly initialised ResNet-50

    if args.warm_start and Path(args.warm_start).exists():
        print(f"\nWarm-starting from {args.warm_start} ...", flush=True)
        sd = torch.load(args.warm_start, map_location="cpu", weights_only=False)
        state = sd.get("model_state_dict", sd.get("state_dict", sd))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"  Missing keys (new heads): {len(missing)}", flush=True)
        print("  Warm-start complete.", flush=True)
    else:
        print("\nTraining from scratch (no warm-start).")

    model = model.to(DEVICE)
    model.train()

    # ── Loss + Optimizer + AMP ────────────────────────────────────────────────
    criterion = TreeFlowLoss(w_flow=1.0, w_sdt=2.0, w_centroid=1.0, w_canopy=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    use_bf16  = DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.05
    )

    # ── Decode config for val eval ────────────────────────────────────────────
    decode_cfg = dict(
        canopy_thresh = 0.40,
        sdt_thresh    = -0.50,
        nms_iou       = 0.30,
        min_area      = 30,
        n_steps       = 20,  # fewer steps for speed during training eval
        eta           = 1.0,
    )

    # ── Training history ──────────────────────────────────────────────────────
    history = []
    best_f1 = 0.0
    best_ckpt = out_dir / "best_model.pth"

    # ── Loop ──────────────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        loss_comp  = {"flow": 0.0, "sdt": 0.0, "centroid": 0.0, "canopy": 0.0}
        n_batches  = 0

        for batch in train_loader:
            imgs = batch["image"].to(DEVICE)
            targets = {
                k: batch[k].to(DEVICE)
                for k in ("flow", "sdt", "centroid", "canopy", "fg_mask")
            }

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=amp_dtype):
                outputs = model(imgs)
                losses  = criterion(outputs, targets)

            if torch.isnan(losses["total"]) or torch.isinf(losses["total"]):
                print(f"  [Warning] NaN/Inf loss: { {k: v.item() for k, v in losses.items()} }", flush=True)
                optimizer.zero_grad()
                continue

            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += losses["total"].item()
            for k in loss_comp:
                loss_comp[k] += losses[k].item()
            n_batches += 1

            if n_batches % 50 == 0 or n_batches == len(train_loader):
                print(f"  [Epoch {epoch:2d}/{args.epochs}] step {n_batches:3d}/{len(train_loader)}: total_loss={losses['total'].item():.4f} (flow={losses['flow'].item():.3f}, sdt={losses['sdt'].item():.3f}, cent={losses['centroid'].item():.3f}, can={losses['canopy'].item():.3f})", flush=True)

        scheduler.step()
        elapsed = time.time() - t0

        avg_loss = total_loss / max(n_batches, 1)
        avg_comp = {k: v / max(n_batches, 1) for k, v in loss_comp.items()}

        # Eval every eval_every epochs
        val_f1 = None
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_f1 = quick_eval_f1(model, val_loader, decode_cfg)
            is_best = val_f1 > best_f1
            if is_best:
                best_f1 = val_f1
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "f1": best_f1}, str(best_ckpt))
            print(
                f"Epoch {epoch:3d}/{args.epochs} [{elapsed:.0f}s] "
                f"loss={avg_loss:.4f} "
                f"(flow={avg_comp['flow']:.3f} sdt={avg_comp['sdt']:.3f} "
                f"cent={avg_comp['centroid']:.3f} can={avg_comp['canopy']:.3f}) "
                f"  val_F1={val_f1:.4f} {'← BEST' if is_best else ''}",
                flush=True
            )
        else:
            print(
                f"Epoch {epoch:3d}/{args.epochs} [{elapsed:.0f}s] "
                f"loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}",
                flush=True
            )

        # Save epoch checkpoint every 5 epochs
        if epoch % 5 == 0:
            ep_ckpt = ckpt_dir / f"epoch_{epoch:04d}.pth"
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, str(ep_ckpt))

        history.append({
            "epoch": epoch,
            "total_loss": avg_loss,
            **avg_comp,
            "val_f1": val_f1,
            "lr": scheduler.get_last_lr()[0],
        })

        # Save history json periodically
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # ── Save final ────────────────────────────────────────────────────────────
    final_ckpt = ckpt_dir / f"epoch_{args.epochs:04d}_final.pth"
    torch.save({"model_state_dict": model.state_dict(), "epoch": args.epochs}, str(final_ckpt))
    print(f"\nTraining complete! Best val F1: {best_f1:.4f}", flush=True)

    with open(out_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Training complete.")
    print(f"  Best val F1@0.5 : {best_f1:.4f}")
    print(f"  Best checkpoint : {best_ckpt}")
    print(f"{'='*70}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--warm-start",   default=None, help="Path to warm-start checkpoint")
    p.add_argument("--epochs",       type=int,   default=60)
    p.add_argument("--batch",        type=int,   default=8)
    p.add_argument("--num-workers",  type=int,   default=4)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--patch-size",   type=int,   default=512)
    p.add_argument("--eval-every",   type=int,   default=5)
    p.add_argument("--out-dir",      default="crown_segmentation_research/experiments/tree_flow_exhaustive_v1")
    p.add_argument("--legacy-v1-data", action="store_true",
                   help="reproduce V1's data (Quebec Z1+Z2+Z3 cached chips = test zone included); old runs only")
    p.add_argument("--allow-leaked-init", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    from crown_segmentation_research.research_integrity.a06_hardening_v2.runtime import reject_legacy_entrypoint
    reject_legacy_entrypoint(__file__)
    train(parse_args())
