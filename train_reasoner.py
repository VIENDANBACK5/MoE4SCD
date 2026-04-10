"""
train_reasoner.py
=================
Stage 4A / 4B / 4C training loop for the Token Change Reasoner.

Use --model_type base  for Stage 4A (Transformer only)
Use --model_type graph for Stage 4B (Transformer + GraphSAGE)
Use --model_type moe   for Stage 4C (Transformer + GraphSAGE + MoE)

Loads Stage 3 match outputs + Stage 2 token files and trains
ChangeReasonerModel with proxy labels derived from embedding delta norms.

Usage:
    # Quick smoke test (50 pairs, 5 epochs)
    python train_reasoner.py \\
        --tokens_T1 SECOND/tokens_T1 \\
        --tokens_T2 SECOND/tokens_T2 \\
        --matches   SECOND/matches   \\
        --output    SECOND/stage4    \\
        --epochs 5 --batch_size 4 --n_samples 50 --device cuda

    # Full training
    python train_reasoner.py \\
        --tokens_T1 SECOND/tokens_T1 \\
        --tokens_T2 SECOND/tokens_T2 \\
        --matches   SECOND/matches   \\
        --output    SECOND/stage4    \\
        --epochs 30 --batch_size 8 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from token_change_reasoner import (
    ChangeReasonerModel,
    ReasonerConfig,
    SampleData,
    build_batch,
    build_model,
    compute_loss,
    count_parameters,
)
from token_change_reasoner_graph import (
    GraphReasonerConfig,
    TokenChangeReasonerGraph,
    build_graph_model,
)
from token_change_reasoner_moe import (
    MoEConfig,
    TokenChangeReasonerMoE,
    build_moe_model,
    compute_moe_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class MatchDataset(Dataset):
    """
    Loads Stage 2 tokens + Stage 3 matches for each image pair.

    Directory structure expected:
        tokens_T1/  {stem}.pt  → {"tokens": [N1,256], "centroids": [N1,2], "areas": [N1]}
        tokens_T2/  {stem}.pt  → same for T2
        matches/    {stem}_matches.pt → {"pairs": [[i,j,score], ...], ...}

    Optional:
        labels_dir/ {stem}_labels.pt → {"labels": [N1+N2]} float 0/1
    """

    def __init__(
        self,
        t1_dir: Path,
        t2_dir: Path,
        match_dir: Path,
        labels_dir: Optional[Path] = None,
        semantic_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        self.t1_dir    = t1_dir
        self.t2_dir    = t2_dir
        self.match_dir = match_dir
        self.labels_dir = labels_dir
        self.semantic_dir = semantic_dir


        # Collect valid stems
        stems = []
        for mp in sorted(match_dir.glob("*_matches.pt")):
            stem = mp.stem.replace("_matches", "")
            if (t1_dir / f"{stem}.pt").exists() and (t2_dir / f"{stem}.pt").exists():
                stems.append(stem)

        if max_samples is not None and max_samples < len(stems):
            rng = random.Random(seed)
            stems = rng.sample(stems, max_samples)

        self.stems = stems
        log.info(f"Dataset: {len(self.stems)} samples")

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> SampleData:
        stem = self.stems[idx]

        t1   = torch.load(self.t1_dir    / f"{stem}.pt",     weights_only=True)
        t2   = torch.load(self.t2_dir    / f"{stem}.pt",     weights_only=True)
        mtch = torch.load(self.match_dir / f"{stem}_matches.pt", weights_only=False)

        pairs = mtch.get("pairs", [])
        if isinstance(pairs, list):
            if len(pairs) > 0:
                pairs = torch.tensor([[float(p[0]), float(p[1]), float(p[2])]
                                      for p in pairs])
            else:
                pairs = torch.zeros(0, 3)
        elif isinstance(pairs, torch.Tensor):
            pairs = pairs.float()

        # ── GT labels ──────────────────────────────────
        labels = None
        if self.labels_dir is not None:
            lp = self.labels_dir / f"{stem}_labels.pt"
            if lp.exists():
                labels = torch.load(lp, weights_only=True).float()

        # ── Semantic labels ────────────────────────────
        semantic_labels = None
        if self.semantic_dir is not None:
            sp = self.semantic_dir / f"{stem}.png"
            if sp.exists():
                import numpy as np
                from PIL import Image
                
                # Load PIL image, ensure mode L or P
                lbl_pil = Image.open(sp)
                if lbl_pil.mode not in ("L", "P"):
                    lbl_pil = lbl_pil.convert("L")
                lbl_img = np.array(lbl_pil)
                img_h, img_w = lbl_img.shape[:2]
                
                # Extract values for T1 + T2 centroids
                c1 = t1["centroids"].numpy()
                c2 = t2["centroids"].numpy()
                c_all = np.concatenate([c1, c2], axis=0) # [N1+N2, 2]
                
                sem_list = []
                for cx, cy in c_all:
                    px = max(0, min(int(round(cx * (img_w - 1))), img_w - 1))
                    py = max(0, min(int(round(cy * (img_h - 1))), img_h - 1))
                    val = lbl_img[py, px]
                    
                    if isinstance(val, (np.ndarray, list)) and len(val) > 1:
                        raw_val = float(val[0])
                    else:
                        raw_val = float(val)
                    
                    # Map 0-255 intensity to 0-6 class index
                    cls = int(round(raw_val / 255.0 * 6))
                    if cls >= 7:
                        cls = 0
                    sem_list.append(cls)
                
                semantic_labels = torch.tensor(sem_list, dtype=torch.long)

        return SampleData(
            tokens_t1    = t1["tokens"].float(),
            tokens_t2    = t2["tokens"].float(),
            centroids_t1 = t1["centroids"].float(),
            centroids_t2 = t2["centroids"].float(),
            areas_t1     = t1["areas"].float(),
            areas_t2     = t2["areas"].float(),
            match_pairs  = pairs,
            change_labels= labels,
            semantic_labels= semantic_labels,
        )



def collate_fn(cfg: ReasonerConfig, device: torch.device):
    """Returns a collate function that packs samples into a batch on device."""
    def _collate(samples: List[SampleData]) -> Dict[str, torch.Tensor]:
        return build_batch(samples, cfg, device)
    return _collate


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    train: bool,
    loss_fn=None,
) -> Dict[str, float]:
    """
    Run one full epoch. Returns mean losses.

    loss_fn: callable(outputs, batch, cfg) → dict.  Defaults to compute_loss.
             Pass compute_moe_loss for Stage 4C.
    """
    if loss_fn is None:
        loss_fn = compute_loss

    model.train(train)
    # Seed totals with base keys; MoE keys added dynamically on first batch.
    totals: Dict[str, float] = {}
    n_batches = 0
    expert_counts: Optional[torch.Tensor] = None  # accumulated [E] for logging

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            cfg = model.cfg
            if train:
                optimizer.zero_grad()

            if scaler is not None and train:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(batch)
                    losses  = loss_fn(outputs, batch, cfg)
                scaler.scale(losses["total_loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(batch)
                losses  = loss_fn(outputs, batch, cfg)
                if train:
                    losses["total_loss"].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

            for k, v in losses.items():
                totals[k] = totals.get(k, 0.0) + float(v.detach())

            # Accumulate expert usage (MoE only)
            if "tokens_per_expert" in outputs:
                tpe = outputs["tokens_per_expert"].detach().cpu()
                if expert_counts is None:
                    expert_counts = tpe
                else:
                    expert_counts += tpe

            # Accumulate evaluation metrics if GT labels are present
            if not train and "change_labels" in batch:
                logits = outputs["change_logits"].detach()
                labels = batch["change_labels"].detach()
                mask = ~batch["padding_mask"]
                
                pred = (logits[mask] > 0).long()
                targ = labels[mask].long()
                
                tp = ((pred == 1) & (targ == 1)).sum().item()
                fp = ((pred == 1) & (targ == 0)).sum().item()
                fn = ((pred == 0) & (targ == 1)).sum().item()
                tn = ((pred == 0) & (targ == 0)).sum().item()
                
                totals["val_tp"] = totals.get("val_tp", 0) + tp
                totals["val_fp"] = totals.get("val_fp", 0) + fp
                totals["val_fn"] = totals.get("val_fn", 0) + fn
                totals["val_tn"] = totals.get("val_tn", 0) + tn

            n_batches += 1

    means = {k: v / max(n_batches, 1) for k, v in totals.items() if not k.startswith("val_")}
    if expert_counts is not None:
        # Log as fraction so it's always comparable
        total_tok = expert_counts.sum().clamp(min=1)
        means["expert_fracs"] = (expert_counts / total_tok).tolist()
    
    # Calculate global metrics
    if not train and "val_tp" in totals:
        tp, fp = totals["val_tp"], totals["val_fp"]
        fn, tn = totals["val_fn"], totals["val_tn"]
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
        iou = tp / max(tp + fp + fn, 1)
        
        means["f1"] = f1
        means["iou"] = iou
        means["precision"] = precision
        means["recall"] = recall

    return means


def save_checkpoint(
    model: ChangeReasonerModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    out_dir: Path,
    name: str = "checkpoint.pt",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": model.cfg.__dict__,
    }, out_dir / name)


def load_checkpoint(path: Path, model: ChangeReasonerModel, optimizer=None):
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))


# ─────────────────────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train(args):
    # ── Setup ──────────────────────────────────────────────────────────────
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    use_graph = (args.model_type == "graph")
    use_moe   = (args.model_type == "moe")

    if use_moe:
        cfg = MoEConfig(
            hidden_dim            = args.hidden_dim,
            num_layers            = args.num_layers,
            num_heads             = args.num_heads,
            dropout               = args.dropout,
            delta_loss_weight     = args.delta_weight,
            proxy_delta_threshold = args.proxy_threshold,
            graph_k               = args.graph_k,
            graph_layers          = args.graph_layers,
            moe_num_experts       = args.moe_num_experts,
            moe_expert_dim        = args.moe_expert_dim,
            lambda_balance        = args.lambda_balance,
            lambda_entropy        = args.lambda_entropy,
            # Stage 4D router improvements
            router_version        = args.router_version,
            expert_dropout_prob   = args.expert_dropout,
            use_top2              = args.use_top2,
        )
    elif use_graph:
        cfg = GraphReasonerConfig(
            hidden_dim            = args.hidden_dim,
            num_layers            = args.num_layers,
            num_heads             = args.num_heads,
            dropout               = args.dropout,
            delta_loss_weight     = args.delta_weight,
            proxy_delta_threshold = args.proxy_threshold,
            graph_k               = args.graph_k,
            graph_layers          = args.graph_layers,
        )
    else:
        cfg = ReasonerConfig(
            hidden_dim            = args.hidden_dim,
            num_layers            = args.num_layers,
            num_heads             = args.num_heads,
            dropout               = args.dropout,
            delta_loss_weight     = args.delta_weight,
            proxy_delta_threshold = args.proxy_threshold,
        )

    # Save config
    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    if use_moe:
        is_4d = (getattr(args, 'router_version', 'v1') == 'v2'
                 or getattr(args, 'use_top2', False)
                 or getattr(args, 'expert_dropout', 0.0) > 0)
        stage = "MoE (4D)" if is_4d else "MoE (4C)"
    elif use_graph:
        stage = "Graph (4B)"
    else:
        stage = "Base (4A)"
    log.info(f"Model type: {stage}")

    # ── Dataset ────────────────────────────────────────────────────────────
    labels_dir = Path(args.labels) if args.labels else None
    semantic_dir = Path(args.semantic_dir) if getattr(args, "semantic_dir", None) else None
    
    dataset = MatchDataset(
        t1_dir     = Path(args.tokens_T1),
        t2_dir     = Path(args.tokens_T2),
        match_dir  = Path(args.matches),
        labels_dir = labels_dir,
        semantic_dir=semantic_dir,
        max_samples= args.n_samples,
        seed       = args.seed,
    )

    val_size  = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    log.info(f"Train: {train_size} | Val: {val_size}")

    _collate = collate_fn(cfg, device)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )

    # ── Model + loss function ─────────────────────────────────────────────
    if use_moe:
        model = build_moe_model(cfg).to(device)
        loss_fn = compute_moe_loss
    elif use_graph:
        model = build_graph_model(cfg).to(device)
        loss_fn = compute_loss
    else:
        model = build_model(cfg).to(device)
        loss_fn = compute_loss
    log.info(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and args.amp) else None

    if args.resume and (out_dir / "checkpoint.pt").exists():
        start_epoch, best_val = load_checkpoint(out_dir / "checkpoint.pt", model, optimizer)
        log.info(f"Resumed from epoch {start_epoch}, val_loss={best_val:.4f}")
    else:
        start_epoch, best_val = 0, float("inf")

    # ── CSV log ────────────────────────────────────────────────────────────
    log_path = out_dir / "training_log.csv"
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["epoch", "train_total", "train_change", "train_delta",
              "val_total",   "val_change",   "val_delta",
              "val_f1", "val_iou", "lr", "time_s"]
    if use_moe:
        header += ["train_balance", "train_entropy", "expert_fracs"]
    csv_writer.writerow(header)

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()

        train_losses = run_epoch(model, train_loader, optimizer, scaler, train=True,  loss_fn=loss_fn)
        val_losses   = run_epoch(model, val_loader,   optimizer, scaler, train=False, loss_fn=loss_fn)

        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr_now  = scheduler.get_last_lr()[0]

        # Expert usage logging (MoE only)
        expert_info = ""
        if "expert_fracs" in train_losses:
            fracs = [f"{f:.2f}" for f in train_losses["expert_fracs"]]
            expert_info = f" | experts=[{','.join(fracs)}]"

        # Aux loss logging (MoE only)
        bal_info = ""
        if "balance_loss" in train_losses:
            bal_info = (
                f" bal={train_losses['balance_loss']:.4f}"
                f" ent={train_losses['entropy_loss']:.4f}"
            )

        log.info(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"train_loss={train_losses['total_loss']:.4f} "
            f"(chg={train_losses['change_loss']:.4f} Δ={train_losses['delta_loss']:.4f}"
            f"{bal_info}) | "
            f"val_loss={val_losses['total_loss']:.4f} "
            f"(chg={val_losses['change_loss']:.4f} Δ={val_losses['delta_loss']:.4f}) "
            f"[F1:{val_losses.get('f1', 0):.4f} IoU:{val_losses.get('iou', 0):.4f}] | "
            f"lr={lr_now:.2e} | {elapsed:.1f}s{expert_info}"
        )

        row = [
            epoch + 1,
            f"{train_losses['total_loss']:.6f}",
            f"{train_losses['change_loss']:.6f}",
            f"{train_losses['delta_loss']:.6f}",
            f"{val_losses['total_loss']:.6f}",
            f"{val_losses['change_loss']:.6f}",
            f"{val_losses['delta_loss']:.6f}",
            f"{val_losses.get('f1', 0):.6f}",
            f"{val_losses.get('iou', 0):.6f}",
            f"{lr_now:.2e}",
            f"{elapsed:.2f}",
        ]
        # Append MoE cols if present
        if "balance_loss" in train_losses:
            row += [
                f"{train_losses['balance_loss']:.6f}",
                f"{train_losses['entropy_loss']:.6f}",
            ]
        if "expert_fracs" in train_losses:
            row.append("|".join(f"{f:.4f}" for f in train_losses["expert_fracs"]))
        csv_writer.writerow(row)
        csv_file.flush()

        # Save best checkpoint
        if val_losses["total_loss"] < best_val:
            best_val = val_losses["total_loss"]
            save_checkpoint(model, optimizer, epoch + 1, best_val, out_dir, "best_model.pt")

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1,
                            val_losses["total_loss"], out_dir, "checkpoint.pt")

    csv_file.close()
    # Final checkpoint
    save_checkpoint(model, optimizer, args.epochs, best_val, out_dir, "final_model.pt")
    log.info(f"Training complete. Best val_loss: {best_val:.4f}")
    log.info(f"Outputs saved to {out_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 4A/4B — Train Token Change Reasoner")

    # Data
    p.add_argument("--tokens_T1",  default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2",  default="SECOND/tokens_T2")
    p.add_argument("--matches",    default="SECOND/matches")
    p.add_argument("--labels",     default=None,
                   help="Optional dir with *_labels.pt for GT change labels")
    p.add_argument("--semantic_dir", default=None,
                   help="Optional dir with Semantic label RGB pngs")
    p.add_argument("--output",     default="SECOND/stage4")
    p.add_argument("--n_samples",  type=int, default=None,
                   help="Limit number of training samples (None = all)")
    p.add_argument("--val_split",  type=float, default=0.1)

    # Model
    p.add_argument("--model_type", default="base", choices=["base", "graph", "moe"],
                   help="base = Stage 4A  |  graph = Stage 4B  |  moe = Stage 4C")
    p.add_argument("--hidden_dim",  type=int,   default=384)
    p.add_argument("--num_layers",  type=int,   default=4)
    p.add_argument("--num_heads",   type=int,   default=8)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--proxy_threshold", type=float, default=9.56,
                   help="delta_norm threshold for proxy changed labels")
    # Graph-specific (Stage 4B / 4C)
    p.add_argument("--graph_k",      type=int, default=6,
                   help="k-NN neighbours per token node")
    p.add_argument("--graph_layers", type=int, default=2,
                   help="Number of stacked GraphSAGE layers")
    # MoE-specific (Stage 4C only)
    p.add_argument("--moe_num_experts", type=int,   default=4,
                   help="Number of MoE experts")
    p.add_argument("--moe_expert_dim",  type=int,   default=512,
                   help="Hidden dim inside each expert (H → expert_dim → H)")
    p.add_argument("--lambda_balance",  type=float, default=0.01,
                   help="Load-balancing aux loss weight")
    p.add_argument("--lambda_entropy",  type=float, default=0.001,
                   help="Entropy regularisation aux loss weight")
    # Stage 4D router improvements
    p.add_argument("--router_version",  default="v1", choices=["v1", "v2", "v3"],
                   help="v1=h only (4C)  v2=concat(h,area,delta) for specialization (4D) v3=concat(h, semantic) for 5.6")
    p.add_argument("--expert_dropout",  type=float, default=0.0,
                   help="Prob of disabling one expert per step (0=off, 0.1 for 4D)")
    p.add_argument("--use_top2",        action="store_true", default=False,
                   help="Top-2 routing: weighted sum of two expert outputs per token")

    # Training
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--delta_weight", type=float, default=0.2)
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--no_amp",       dest="amp", action="store_false")
    p.add_argument("--device",       default="cuda")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--save_every",   type=int,   default=5)
    p.add_argument("--resume",       action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
