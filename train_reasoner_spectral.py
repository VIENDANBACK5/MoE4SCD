# train_reasoner_spectral.py
from __future__ import annotations

import torch
for i in range(1, 8):
    attr = f"int{i}"
    if not hasattr(torch, attr):
        setattr(torch, attr, torch.int8)

import sys
from types import ModuleType
dummy_schema = ModuleType("torch._library.infer_schema")
dummy_schema.infer_schema = lambda *args, **kwargs: ""
sys.modules["torch._library.infer_schema"] = dummy_schema



import torch.serialization
if not hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals = lambda *args, **kwargs: None

import torch._dynamo.utils
if not hasattr(torch._dynamo.utils, 'warn_once'):
    torch._dynamo.utils.warn_once = lambda *args, **kwargs: None

import torch._inductor.config
if not hasattr(torch._inductor.config, 'max_autotune_gemm_search_space'):
    object.__setattr__(torch._inductor.config, 'max_autotune_gemm_search_space', False)


"""
train_reasoner_spectral.py
==========================
Stage 4A / 4B / 4C training loop for the Token Change Reasoner.
Includes support for LoRA on SAM2 encoder and spectral feature injection.
"""



import argparse
import csv
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from stage2_quality_diagnostics import load_gt_masks, diagnostics
import numpy as np

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
from token_hierarchical_reasoner import (
    HierarchicalConfig,
    HierarchicalChangeReasoner,
    compute_hierarchical_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# remap Class 7 "other/farmland" to 0 (background)
_SECOND_RGB_TO_CLASS: Dict[tuple, int] = {
    (0,   0,   0):   0,
    (0,   128, 0):   1,
    (128, 0,   0):   2,
    (0,   0,   255): 3,
    (128, 128, 128): 4,
    (255, 255, 255): 5,
    (0,   255, 0):   6,
    (255, 0,   0):   0,
}


def _centroid_class(label_rgb: "np.ndarray", cx: float, cy: float) -> int:
    H, W = label_rgb.shape[:2]
    px = max(0, min(int(round(cx * (W - 1))), W - 1))
    py = max(0, min(int(round(cy * (H - 1))), H - 1))
    rgb = tuple(label_rgb[py, px].tolist())
    return _SECOND_RGB_TO_CLASS.get(rgb, 0)


class MatchDataset(Dataset):
    def __init__(
        self,
        t1_dir: Path,
        t2_dir: Path,
        match_dir: Path,
        labels_dir: Optional[Path] = None,
        semantic_dir: Optional[Path] = None,
        semantic_dir_t2: Optional[Path] = None,
        max_samples: Optional[int] = None,
        seed: int = 42,
        gt_change_labels: bool = False,
    ):
        self.t1_dir          = t1_dir
        self.t2_dir          = t2_dir
        self.match_dir       = match_dir
        self.labels_dir      = labels_dir
        self.semantic_dir    = semantic_dir
        self.semantic_dir_t2 = semantic_dir_t2
        self.gt_change_labels = gt_change_labels

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

        labels = None
        if self.gt_change_labels and self.semantic_dir is not None:
            import numpy as np
            from PIL import Image as _PIL
            BG_CLS = {0, 4, 6}
            sp1 = self.semantic_dir / f"{stem}.png"
            sp2 = (self.semantic_dir_t2 / f"{stem}.png"
                   if self.semantic_dir_t2 is not None else sp1)
            if sp1.exists():
                lab1 = np.array(_PIL.open(sp1).convert("RGB"))
                lab2 = np.array(_PIL.open(sp2).convert("RGB")) if sp2.exists() else lab1
                c1 = t1["centroids"].numpy()
                c2 = t2["centroids"].numpy()
                gt_ch_t1 = []
                for cx, cy in c1:
                    cls_t1 = _centroid_class(lab1, cx, cy)
                    cls_t2 = _centroid_class(lab2, cx, cy)
                    changed = int(cls_t1 != cls_t2 and cls_t1 not in BG_CLS)
                    gt_ch_t1.append(changed)
                gt_ch_t2 = [0] * len(c2)
                labels = torch.tensor(gt_ch_t1 + gt_ch_t2, dtype=torch.float32)
        elif self.labels_dir is not None:
            lp = self.labels_dir / f"{stem}_labels.pt"
            if lp.exists():
                labels = torch.load(lp, weights_only=True).float()

        semantic_labels    = None
        transition_labels  = None
        if self.semantic_dir is not None:
            import numpy as np
            from PIL import Image

            sp1 = self.semantic_dir / f"{stem}.png"
            sp2 = (self.semantic_dir_t2 / f"{stem}.png"
                   if self.semantic_dir_t2 is not None else sp1)

            if sp1.exists():
                lab_t1_rgb = np.array(Image.open(sp1).convert("RGB"))
                lab_t2_rgb = (np.array(Image.open(sp2).convert("RGB"))
                              if sp2.exists() else lab_t1_rgb)

                c1 = t1["centroids"].numpy()
                c2 = t2["centroids"].numpy()

                sem_t1 = [_centroid_class(lab_t1_rgb, cx, cy) for cx, cy in c1]
                sem_t2 = [_centroid_class(lab_t2_rgb, cx, cy) for cx, cy in c2]

                semantic_labels = torch.tensor(sem_t1 + sem_t2, dtype=torch.long)

                BG_CLS    = {0, 4, 6}
                NUM_CLS   = 7
                trans_t1  = []
                for (cx, cy), c1_cls in zip(c1, sem_t1):
                    c2_cls = _centroid_class(lab_t2_rgb, cx, cy)
                    if c1_cls in BG_CLS or c1_cls == c2_cls:
                        trans_t1.append(-1)
                    else:
                        trans_t1.append(c1_cls * NUM_CLS + c2_cls)
                transition_labels = torch.tensor(trans_t1, dtype=torch.long)

        return SampleData(
            tokens_t1         = t1["tokens"].float(),
            tokens_t2         = t2["tokens"].float(),
            centroids_t1      = t1["centroids"].float(),
            centroids_t2      = t2["centroids"].float(),
            areas_t1          = t1["areas"].float(),
            areas_t2          = t2["areas"].float(),
            cvs_t1            = t1.get("cvs", None),
            cvs_t2            = t2.get("cvs", None),
            spectral_t1       = t1.get("spectral", None),
            spectral_t2       = t2.get("spectral", None),
            match_pairs       = pairs,
            change_labels     = labels,
            semantic_labels   = semantic_labels,
            transition_labels = transition_labels,
        )


def collate_fn(cfg: ReasonerConfig, device: torch.device):
    def _collate(samples: List[SampleData]) -> Dict[str, torch.Tensor]:
        return build_batch(samples, cfg, device)
    return _collate


def run_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    train: bool,
    args,
    loss_fn=None,
) -> Dict[str, float]:
    if loss_fn is None:
        loss_fn = compute_loss

    model.train(train)
    totals: Dict[str, float] = {}
    n_batches = 0
    expert_counts: Optional[torch.Tensor] = None

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

            if "tokens_per_expert" in outputs:
                tpe = outputs["tokens_per_expert"].detach().cpu()
                if expert_counts is None:
                    expert_counts = tpe
                else:
                    expert_counts += tpe

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
        total_tok = expert_counts.sum().clamp(min=1)
        means["expert_fracs"] = (expert_counts / total_tok).tolist()
    
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


def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)

    use_graph = (args.model_type == "graph")
    use_moe   = (args.model_type == "moe")

    if use_moe:
        cfg = MoEConfig(
            token_dim             = args.token_dim,
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
            lambda_semantic        = args.lambda_semantic,
            lambda_transition     = args.lambda_transition,
            router_version        = args.router_version,
            expert_dropout_prob   = args.expert_dropout,
            use_top2              = args.use_top2,
            use_spectral          = args.use_spectral,
        )
    elif args.model_type == "hierarchical":
        cfg = HierarchicalConfig(
            token_dim             = args.token_dim,
            hidden_dim            = args.hidden_dim,
            num_layers            = args.num_layers,
            num_heads             = args.num_heads,
            dropout               = args.dropout,
            delta_loss_weight     = args.delta_weight,
            proxy_delta_threshold = args.proxy_threshold,
            graph_k               = args.graph_k,
            graph_layers          = args.graph_layers,
            num_clusters          = args.num_clusters,
            smoothness_weight     = args.smoothness_weight,
        )
    elif use_graph:
        cfg = GraphReasonerConfig(
            token_dim             = args.token_dim,
            hidden_dim            = args.hidden_dim,
            num_layers            = args.num_layers,
            num_heads             = args.num_heads,
            dropout               = args.dropout,
            delta_loss_weight     = args.delta_weight,
            proxy_delta_threshold = args.proxy_threshold,
            graph_k               = args.graph_k,
            graph_layers          = args.graph_layers,
            alpha_cv              = args.alpha_cv,
        )
    else:
        cfg = ReasonerConfig(
            token_dim             = args.token_dim,
            hidden_dim            = args.hidden_dim,
            num_layers            = args.num_layers,
            num_heads             = args.num_heads,
            dropout               = args.dropout,
            proxy_delta_threshold = args.proxy_threshold,
            delta_loss_weight     = args.delta_weight,
            alpha_cv              = args.alpha_cv,
        )

    (out_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))
    if use_moe:
        is_4d = (getattr(args, 'router_version', 'v1') == 'v2'
                 or getattr(args, 'use_top2', False)
                 or getattr(args, 'expert_dropout', 0.0) > 0)
        stage = "MoE (4D)" if is_4d else "MoE (4C)"
    elif use_graph:
        stage = "Graph (4B)"
    elif args.model_type == "hierarchical":
        stage = "Hierarchical (MOB-GCN)"
    else:
        stage = "Base (4A)"
    log.info(f"Model type: {stage}")

    labels_dir       = Path(args.labels) if args.labels else None
    semantic_dir     = Path(args.semantic_dir)    if getattr(args, "semantic_dir",    None) else None
    semantic_dir_t2  = Path(args.semantic_dir_t2) if getattr(args, "semantic_dir_t2", None) else None

    dataset = MatchDataset(
        t1_dir           = Path(args.tokens_T1),
        t2_dir           = Path(args.tokens_T2),
        match_dir        = Path(args.matches),
        labels_dir       = labels_dir,
        semantic_dir     = semantic_dir,
        semantic_dir_t2  = semantic_dir_t2,
        max_samples      = args.n_samples,
        seed             = args.seed,
        gt_change_labels = args.gt_change_labels,
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

    if use_moe:
        model = build_moe_model(cfg).to(device)
        loss_fn = compute_moe_loss
    elif use_graph:
        model = build_graph_model(cfg).to(device)
        loss_fn = compute_loss
    elif args.model_type == "hierarchical":
        model = HierarchicalChangeReasoner(cfg).to(device)
        loss_fn = compute_hierarchical_loss
    else:
        model = build_model(cfg).to(device)
        loss_fn = compute_loss
    log.info(f"Model parameters: {count_parameters(model):,}")

    # Load pretrain weights if specified before creating optimizer
    if args.pretrain:
        ckpt = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt.get("model_state", ckpt))
        missing, unexpected = model.load_state_dict(state, strict=False)
        log.info(f"Loaded pretrain weights from {args.pretrain}")
        if missing:    log.info(f"  Missing keys (new heads): {missing}")
        if unexpected: log.info(f"  Unexpected keys (ignored): {unexpected}")

    if args.use_lora:
        # SAM2 is never called during forward (tokens are pre-computed), so LoRA params
        # won't receive gradients. We attempt to load anyway for completeness; if peft/
        # torchao import fails on this PyTorch version, fall back gracefully.
        try:
            import sys as _sys
            SAM2_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam2")
            if os.path.isdir(SAM2_REPO) and SAM2_REPO not in _sys.path:
                _sys.path.insert(0, SAM2_REPO)
            from sam2.build_sam import build_sam2
            SAM2_CKPT   = os.path.join(SAM2_REPO, "checkpoints", "sam2.1_hiera_large.pt")
            SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
            sam2_model = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=device)

            from lora_sam2 import apply_lora_to_sam2
            sam2_model = apply_lora_to_sam2(
                sam2_model,
                rank=args.lora_rank,
                alpha=args.lora_alpha
            )

            lora_params = []
            other_params = []
            for m in [sam2_model, model]:
                for name, param in m.named_parameters():
                    if not param.requires_grad:
                        continue
                    if "lora_" in name:
                        lora_params.append(param)
                    else:
                        other_params.append(param)

            param_groups = [
                {"params": lora_params,  "lr": args.lora_lr, "name": "lora"},
                {"params": other_params, "lr": args.lr,      "name": "other"},
            ]
            optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
            log.info(f"LoRA applied to SAM2 (rank={args.lora_rank}, alpha={args.lora_alpha})")
        except Exception as _e:
            log.warning(f"LoRA on SAM2 unavailable ({_e}). Training reasoner only (tokens are pre-computed, SAM2 not in forward pass).")
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and args.amp) else None

    if args.resume and (out_dir / "checkpoint.pt").exists():
        start_epoch, best_val = load_checkpoint(out_dir / "checkpoint.pt", model, optimizer)
        log.info(f"Resumed from epoch {start_epoch}, best_val_f1={best_val:.4f}")
    else:
        start_epoch, best_val = 0, 0.0

    log_path = out_dir / "training_log.csv"
    csv_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    header = ["epoch", "train_total", "train_change", "train_delta",
              "val_total",   "val_change",   "val_delta",
              "val_f1", "val_iou", "lr", "time_s"]
    if use_moe:
        header += ["train_balance", "train_entropy", "train_semantic", "expert_fracs"]
    csv_writer.writerow(header)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.perf_counter()

        train_losses = run_epoch(model, train_loader, optimizer, scaler, train=True,  args=args, loss_fn=loss_fn)
        val_losses   = run_epoch(model, val_loader,   optimizer, scaler, train=False, args=args, loss_fn=loss_fn)

        scheduler.step()
        elapsed = time.perf_counter() - t0
        lr_now  = scheduler.get_last_lr()[0]

        expert_info = ""
        if "expert_fracs" in train_losses:
            fracs = [f"{f:.2f}" for f in train_losses["expert_fracs"]]
            expert_info = f" | experts=[{','.join(fracs)}]"

        bal_info = ""
        if "balance_loss" in train_losses:
            sem = train_losses.get("semantic_loss", 0)
            bal_info = (
                f" bal={train_losses['balance_loss']:.4f}"
                f" ent={train_losses['entropy_loss']:.4f}"
                f" sem={sem:.4f}"
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
        if "balance_loss" in train_losses:
            row += [
                f"{train_losses['balance_loss']:.6f}",
                f"{train_losses['entropy_loss']:.6f}",
                f"{train_losses.get('semantic_loss', 0):.6f}",
            ]
        if "expert_fracs" in train_losses:
            row.append("|".join(f"{f:.4f}" for f in train_losses["expert_fracs"]))
        csv_writer.writerow(row)
        csv_file.flush()

        val_f1 = val_losses.get("f1", 0.0)
        if val_f1 > best_val:
            best_val = val_f1
            save_checkpoint(model, optimizer, epoch + 1, best_val, out_dir, "best_model.pt")

        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, epoch + 1,
                            val_f1, out_dir, "checkpoint.pt")

    csv_file.close()
    save_checkpoint(model, optimizer, args.epochs, best_val, out_dir, "final_model.pt")
    
    log.info(f"Training complete. Best val_f1: {best_val:.4f}")
    log.info(f"Outputs saved to {out_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4 — Train Token Change Reasoner (with Spectral/LoRA)")

    # Data
    p.add_argument("--tokens_T1",  default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2",  default="SECOND/tokens_T2")
    p.add_argument("--matches",    default="SECOND/matches")
    p.add_argument("--labels",     default=None)
    p.add_argument("--semantic_dir", default=None)
    p.add_argument("--semantic_dir_t2", default=None)
    p.add_argument("--output",     default="SECOND/stage4")
    p.add_argument("--n_samples",  type=int, default=None)
    p.add_argument("--val_split",  type=float, default=0.1)

    # Model
    p.add_argument("--model_type", default="base", choices=["base", "graph", "moe", "hierarchical"])
    p.add_argument("--token_dim",  type=int,   default=256)
    p.add_argument("--hidden_dim",  type=int,   default=384)
    p.add_argument("--num_layers",  type=int,   default=4)
    p.add_argument("--num_heads",   type=int,   default=8)
    p.add_argument("--dropout",     type=float, default=0.1)
    p.add_argument("--proxy_threshold", type=float, default=9.56)
    p.add_argument("--alpha_cv", type=float, default=1.0)
    p.add_argument("--gt_dir", type=str, default=None)

    # Graph
    p.add_argument("--graph_k",      type=int, default=6)
    p.add_argument("--graph_layers", type=int, default=2)

    # Hierarchical
    p.add_argument("--num_clusters", type=int, default=20)
    p.add_argument("--smoothness_weight", type=float, default=0.1)

    # MoE
    p.add_argument("--moe_num_experts", type=int,   default=4)
    p.add_argument("--moe_expert_dim",  type=int,   default=512)
    p.add_argument("--lambda_balance",   type=float, default=0.01)
    p.add_argument("--lambda_entropy",   type=float, default=0.001)
    p.add_argument("--lambda_semantic",    type=float, default=0.3)
    p.add_argument("--lambda_transition",  type=float, default=0.0)
    p.add_argument("--router_version",  default="v1", choices=["v1", "v2", "v3"])
    p.add_argument("--expert_dropout",  type=float, default=0.0)
    p.add_argument("--use_top2",        action="store_true", default=False)

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
    p.add_argument("--resume",            action="store_true")
    p.add_argument("--pretrain",          default=None)
    p.add_argument("--gt_change_labels",  action="store_true")
    p.add_argument("--use_spectral",      action="store_true", default=False)

    # LoRA Specific
    p.add_argument("--use_lora",          action="store_true", default=False)
    p.add_argument("--lora_rank",         type=int, default=4)
    p.add_argument("--lora_alpha",        type=float, default=8.0)
    p.add_argument("--lora_lr",           type=float, default=1e-4)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
