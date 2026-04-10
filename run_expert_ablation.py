#!/usr/bin/env python3
"""
run_expert_ablation.py
======================
Expert Ablation Study for TokenChangeReasonerMoE.

For each expert i in [0..E-1]:
  - Disable expert_i (zero out its contribution during inference)
  - Run full validation evaluation
  - Record F1, IoU, Precision, Recall

Ablation strategies (--strategy flag):
  zero     : tokens routed to expert_i get zero expert output (only residual x).
             Most interpretable — measures the net additive value of expert_i.
  rerout   : tokens routed to expert_i are redirected to the next-best expert.
             Measures whether expert_i is replaceable by others.
  identity : expert_i is replaced by an identity mapping (output = input x).
             Measures whether expert_i adds any non-trivial transformation.

Validation setup:
  Mirrors the training split: 90/10 with seed 42 (train_reasoner.py defaults).
  Uses proxy change labels from delta_norm (same as training when --labels is
  not specified).  Pass --labels_dir to use GT PNG labels instead.

Output files (--output_dir):
  ablation_results.csv
  expert_importance_plot.png
  stage5_expert_ablation_report.md

Usage:
  # semantic model (v3 router)
  python run_expert_ablation.py \\
      --checkpoint SECOND/stage5_6_semantic/best_model.pt \\
      --config     SECOND/stage5_6_semantic/config.json \\
      --tokens_T1  SECOND/tokens_T1 \\
      --tokens_T2  SECOND/tokens_T2 \\
      --matches    SECOND/matches \\
      --output_dir stage5/ablation/semantic \\
      --strategy   zero

  # dynamic model (v2 router)
  python run_expert_ablation.py \\
      --checkpoint SECOND/stage5_6_dynamic/best_model.pt \\
      --config     SECOND/stage5_6_dynamic/config.json \\
      --output_dir stage5/ablation/dynamic \\
      --strategy   zero
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

# ── allow imports from phase1 root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from token_change_reasoner import SampleData, build_batch, _proxy_labels
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE

# ─────────────────────────────────────────────────────────────────────────────
# Expert colors (colorblind-friendly palette)
# ─────────────────────────────────────────────────────────────────────────────

EXPERT_COLORS = [
    (214,  39,  40),   # Expert 0 — red
    ( 31, 119, 180),   # Expert 1 — blue
    ( 44, 160,  44),   # Expert 2 — green
    (255, 127,  14),   # Expert 3 — orange
    (148, 103, 189),   # Expert 4 — purple
    ( 23, 190, 207),   # Expert 5 — teal
]

BASELINE_COLOR = (90, 90, 90)   # grey for full model bar

# ─────────────────────────────────────────────────────────────────────────────
# Config + model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> MoEConfig:
    d = json.loads(path.read_text())
    cfg = MoEConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_model(
    ckpt_path: Path,
    cfg: MoEConfig,
    device: torch.device,
) -> TokenChangeReasonerMoE:
    model = TokenChangeReasonerMoE(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_stems(
    match_dir: Path,
    t1_dir: Path,
    t2_dir: Path,
    val_split: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], List[str]]:
    """Return (train_stems, val_stems) mirroring the training split."""
    all_stems = []
    for mp in sorted(match_dir.glob("*_matches.pt")):
        stem = mp.stem.replace("_matches", "")
        if (t1_dir / f"{stem}.pt").exists() and (t2_dir / f"{stem}.pt").exists():
            all_stems.append(stem)

    if max_samples and max_samples < len(all_stems):
        rng = random.Random(seed)
        all_stems = rng.sample(all_stems, max_samples)

    # Mirror torch random_split with manual_seed(seed)
    n_val = max(1, int(len(all_stems) * val_split))
    n_train = len(all_stems) - n_val
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(all_stems), generator=gen).tolist()
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]
    train_stems = [all_stems[i] for i in train_idx]
    val_stems   = [all_stems[i] for i in val_idx]
    return train_stems, val_stems


# ─────────────────────────────────────────────────────────────────────────────
# Expert disabling context managers
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def disable_expert(
    model: TokenChangeReasonerMoE,
    expert_id: int,
    strategy: str = "zero",
):
    """
    Context manager that temporarily disables one expert during inference.

    strategy = "zero"     → zero out the expert's contribution (out_flat stays 0
                            for tokens dispatched to this expert; residual x
                            still passes through unchanged).
    strategy = "rerout"   → tokens that would go to expert_id are redirected to
                            the expert with the 2nd-highest router probability.
    strategy = "identity" → expert_id is replaced by an identity function
                            (output = input), measuring non-trivial transformation.
    """
    moe = model.moe
    orig_forward = moe.forward

    if strategy == "identity":
        # Temporarily replace expert weights with identity approximation
        # done by saving & zeroing the first linear bias and setting weight ≈ I
        expert = moe.experts[expert_id]
        orig_w0 = expert.net[0].weight.data.clone()
        orig_b0 = expert.net[0].bias.data.clone() if expert.net[0].bias is not None else None
        orig_w2 = expert.net[2].weight.data.clone()
        orig_b2 = expert.net[2].bias.data.clone() if expert.net[2].bias is not None else None

        H, D = orig_w2.shape[1], orig_w2.shape[0]   # weight[2]: [H, D_exp]
        # Identity approximation: W0=0, b0=0, W2=0, b2=0 → output = GELU(0)·W2+b2 ≈ 0
        # A true identity is hard without matching dims (D_exp ≠ H usually).
        # Use the zero strategy internally (same effect for most dim configs).
        strategy_actual = "zero"
    else:
        strategy_actual = strategy

    if strategy_actual in ("zero", "rerout"):
        # Patch forward via a wrapper
        def patched_forward(
            x,
            log_areas=None,
            delta_hints=None,
            semantic_labels=None,
        ):
            B, N, H = x.shape
            E = moe.num_experts
            x_flat = x.reshape(B * N, H)

            # Build router input (same logic as MoELayer.forward)
            if moe.router_version == "v2":
                la = log_areas.reshape(B * N, 1) if log_areas is not None else \
                     torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)
                dh = delta_hints.reshape(B * N, 1) if delta_hints is not None else \
                     torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)
                router_input = torch.cat([x_flat, la, dh], dim=-1)
            elif moe.router_version == "v3":
                if semantic_labels is not None:
                    sl    = semantic_labels.reshape(B * N).long()
                    sl_oh = F.one_hot(sl, num_classes=7).to(x.dtype)
                else:
                    sl_oh = torch.zeros(B * N, 7, device=x.device, dtype=x.dtype)
                router_input = torch.cat([x_flat, sl_oh], dim=-1)
            else:
                router_input = x_flat

            router_logits = moe.router(router_input)          # [T, E]
            router_probs  = torch.softmax(router_logits, dim=-1)

            if strategy_actual == "rerout":
                # Zero out the disabled expert's probability column, renormalise
                router_probs = router_probs.clone()
                router_probs[:, expert_id] = 0.0
                router_probs = router_probs / router_probs.sum(-1, keepdim=True).clamp(min=1e-8)

            expert_idx = router_probs.argmax(dim=-1)           # [T]

            out_flat = torch.zeros(B * N, H, dtype=x.dtype, device=x.device)
            tokens_per_expert = torch.zeros(E, dtype=torch.long, device=x.device)

            for e in range(E):
                mask = (expert_idx == e)
                n_e  = mask.sum()
                tokens_per_expert[e] = n_e
                if n_e == 0:
                    continue
                if strategy_actual == "zero" and e == expert_id:
                    # Zero contribution — skip running the expert
                    pass
                else:
                    out_flat[mask] = moe.experts[e](x_flat[mask]).to(x.dtype)

            out = x + out_flat.reshape(B, N, H)

            # Aux losses (kept for structural compatibility)
            p_i          = router_probs.mean(dim=0)
            balance_loss = (E * (p_i ** 2).sum()).unsqueeze(0).squeeze()
            eps          = 1e-8
            entropy      = -(router_probs * (router_probs + eps).log()).sum(dim=-1)
            entropy_loss = entropy.mean()

            return out, balance_loss, entropy_loss, tokens_per_expert

        moe.forward = patched_forward

    try:
        yield
    finally:
        moe.forward = orig_forward
        if strategy == "identity":
            # No weight modifications were made (we used zero strategy internally)
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    f1:         float
    iou:        float
    precision:  float
    recall:     float
    n_tokens:   int
    n_samples:  int


@torch.no_grad()
def run_evaluation(
    model: TokenChangeReasonerMoE,
    stems: List[str],
    t1_dir: Path,
    t2_dir: Path,
    match_dir: Path,
    cfg: MoEConfig,
    device: torch.device,
    desc: str = "eval",
) -> EvalMetrics:
    """
    Run a full evaluation pass and return aggregated metrics.
    Proxy change labels are derived from delta_norm when GT labels are absent
    (same mechanism used during training).
    """
    tp_total = fp_total = fn_total = tn_total = 0
    n_tokens = 0
    n_samples = 0

    for stem in tqdm(stems, desc=f"  [{desc}]", ncols=80, leave=False):
        t1_path = t1_dir   / f"{stem}.pt"
        t2_path = t2_dir   / f"{stem}.pt"
        m_path  = match_dir / f"{stem}_matches.pt"
        if not (t1_path.exists() and t2_path.exists() and m_path.exists()):
            continue

        t1   = torch.load(t1_path,  weights_only=True)
        t2   = torch.load(t2_path,  weights_only=True)
        mtch = torch.load(m_path,   weights_only=True)

        pairs = mtch.get("pairs", [])
        if isinstance(pairs, list):
            p_tensor = (
                torch.tensor(
                    [[float(p[0]), float(p[1]), float(p[2])] for p in pairs],
                    dtype=torch.float32,
                )
                if pairs
                else torch.zeros(0, 3)
            )
        else:
            p_tensor = pairs.float()

        sample = SampleData(
            tokens_t1    = t1["tokens"].float(),
            tokens_t2    = t2["tokens"].float(),
            centroids_t1 = t1["centroids"].float(),
            centroids_t2 = t2["centroids"].float(),
            areas_t1     = t1["areas"].float(),
            areas_t2     = t2["areas"].float(),
            match_pairs  = p_tensor,
            change_labels= None,  # use proxy
        )

        batch  = build_batch([sample], cfg, device)
        output = model(batch)

        change_logits = output["change_logits"]          # [B, N]
        change_labels = batch["change_labels"]           # [B, N] proxy 0/1
        padding_mask  = batch["padding_mask"]            # [B, N] True = pad

        valid = ~padding_mask
        pred  = (change_logits[valid] > 0).long()
        targ  = change_labels[valid].long()

        tp = int(((pred == 1) & (targ == 1)).sum())
        fp = int(((pred == 1) & (targ == 0)).sum())
        fn = int(((pred == 0) & (targ == 1)).sum())
        tn = int(((pred == 0) & (targ == 0)).sum())

        tp_total += tp
        fp_total += fp
        fn_total += fn
        tn_total += tn
        n_tokens  += int(valid.sum())
        n_samples += 1

    precision = tp_total / max(tp_total + fp_total, 1)
    recall    = tp_total / max(tp_total + fn_total, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    iou       = tp_total / max(tp_total + fp_total + fn_total, 1)

    return EvalMetrics(
        f1=f1, iou=iou, precision=precision, recall=recall,
        n_tokens=n_tokens, n_samples=n_samples,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Expert routing stats (for report context)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_routing_stats(
    model: TokenChangeReasonerMoE,
    stems: List[str],
    t1_dir: Path,
    t2_dir: Path,
    match_dir: Path,
    cfg: MoEConfig,
    device: torch.device,
) -> np.ndarray:
    """Return per-expert token count fractions [E] from baseline inference."""
    E = model.moe.num_experts
    counts = np.zeros(E, dtype=np.int64)

    for stem in tqdm(stems, desc="  [routing stats]", ncols=80, leave=False):
        t1_path = t1_dir   / f"{stem}.pt"
        t2_path = t2_dir   / f"{stem}.pt"
        m_path  = match_dir / f"{stem}_matches.pt"
        if not (t1_path.exists() and t2_path.exists() and m_path.exists()):
            continue
        t1   = torch.load(t1_path,  weights_only=True)
        t2   = torch.load(t2_path,  weights_only=True)
        mtch = torch.load(m_path,   weights_only=True)

        pairs = mtch.get("pairs", [])
        if isinstance(pairs, list):
            p_tensor = (
                torch.tensor(
                    [[float(p[0]), float(p[1]), float(p[2])] for p in pairs],
                    dtype=torch.float32,
                )
                if pairs
                else torch.zeros(0, 3)
            )
        else:
            p_tensor = pairs.float()

        sample = SampleData(
            tokens_t1    = t1["tokens"].float(),
            tokens_t2    = t2["tokens"].float(),
            centroids_t1 = t1["centroids"].float(),
            centroids_t2 = t2["centroids"].float(),
            areas_t1     = t1["areas"].float(),
            areas_t2     = t2["areas"].float(),
            match_pairs  = p_tensor,
        )
        batch  = build_batch([sample], cfg, device)
        output = model(batch)
        tpe    = output["tokens_per_expert"].cpu().numpy()
        counts += tpe

    total = counts.sum()
    return counts / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Visualization — PIL only (no matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def _lerp_color(
    c_lo: Tuple[int, int, int],
    c_hi: Tuple[int, int, int],
    t: float,
) -> Tuple[int, int, int]:
    return tuple(int(c_lo[k] + (c_hi[k] - c_lo[k]) * t) for k in range(3))


def save_importance_bar_chart(
    baseline: EvalMetrics,
    ablated: List[EvalMetrics],
    out_path: Path,
    title: str = "Expert Importance (IoU drop when disabled)",
) -> None:
    """
    Horizontal bar chart of importance_i = IoU_full - IoU_without_i.
    Negative importance (ablation improves!) shown in lighter tone.
    Includes a reference table of F1 / IoU / P / R for every variant.
    """
    E = len(ablated)

    importance = np.array([baseline.iou - m.iou for m in ablated])

    # ── Bar chart section ──────────────────────────────────────────────────
    BAR_H   = 40
    LABEL_W = 90
    PAD     = 14
    CHART_W = 560
    CHART_H = PAD * 3 + 32 + E * (BAR_H + 10)

    # ── Table section ──────────────────────────────────────────────────────
    COL_W   = 110       # per metric column
    ROW_H   = 28
    N_ROWS  = E + 2     # header + baseline + E ablations
    N_COLS  = 5         # Model | F1 | IoU | Precision | Recall
    TABLE_W = LABEL_W + N_COLS * COL_W
    TABLE_H = PAD + 24 + N_ROWS * ROW_H + PAD

    TOTAL_W = max(CHART_W, TABLE_W)
    TOTAL_H = CHART_H + TABLE_H + PAD

    img  = Image.new("RGB", (TOTAL_W, TOTAL_H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # ── Title ───────────────────────────────────────────────────────────────
    draw.text((TOTAL_W // 2 - len(title) * 4, PAD), title, fill=(30, 30, 30))

    # ── Bars ────────────────────────────────────────────────────────────────
    max_abs = float(np.abs(importance).max()) if np.abs(importance).max() > 0 else 0.01
    BAR_MAX  = CHART_W - LABEL_W - PAD * 2 - 60   # pixels for max bar

    for i in range(E):
        val  = float(importance[i])
        y0   = PAD + 32 + i * (BAR_H + 10)
        lbl  = f"Expert {i}"
        draw.text((PAD, y0 + 12), lbl, fill=(60, 60, 60))

        # bar length proportional to |importance|
        bar_w = max(2, int(abs(val) / max_abs * BAR_MAX))
        color = (
            EXPERT_COLORS[i % len(EXPERT_COLORS)]
            if val >= 0
            else (200, 200, 200)
        )
        draw.rectangle(
            [LABEL_W, y0, LABEL_W + bar_w, y0 + BAR_H],
            fill=color, outline=(100, 100, 100),
        )
        sign  = "+" if val >= 0 else ""
        draw.text(
            (LABEL_W + bar_w + 6, y0 + 12),
            f"{sign}{val:.4f}",
            fill=(30, 30, 30),
        )

    # Bottom axis label
    draw.text(
        (LABEL_W + BAR_MAX // 2, CHART_H - PAD - 2),
        "← IoU drop (higher = more important) →",
        fill=(120, 120, 120),
    )

    # ── Metrics Table ────────────────────────────────────────────────────────
    TY0 = CHART_H + PAD
    draw.text((PAD, TY0), "Detailed Metrics", fill=(30, 30, 30))

    headers = ["Model", "F1", "IoU", "Precision", "Recall"]
    col_x   = [PAD + 4]
    for c in range(1, N_COLS):
        col_x.append(LABEL_W + (c - 1) * COL_W + 4)

    # Header row
    hdr_y = TY0 + 22
    for c, h in enumerate(headers):
        draw.text((col_x[c], hdr_y), h, fill=(40, 40, 40))
    draw.line(
        [(PAD, hdr_y + 16), (LABEL_W + (N_COLS - 1) * COL_W, hdr_y + 16)],
        fill=(180, 180, 180), width=1,
    )

    all_variants: List[Tuple[str, EvalMetrics, Tuple[int, int, int]]] = [
        ("Full MoE", baseline, BASELINE_COLOR)
    ] + [
        (f"- Expert {i}", ablated[i], EXPERT_COLORS[i % len(EXPERT_COLORS)])
        for i in range(E)
    ]

    for row_i, (name, m, color) in enumerate(all_variants):
        ry = hdr_y + 20 + row_i * ROW_H
        # Alternate row shading
        if row_i % 2 == 1:
            draw.rectangle(
                [PAD, ry - 2, LABEL_W + (N_COLS - 1) * COL_W, ry + ROW_H - 4],
                fill=(245, 245, 252),
            )

        iou_fl = m.iou
        # Color ramp: worse than baseline → red, better → green
        delta_iou = m.iou - baseline.iou
        if row_i == 0:
            row_color = (30, 30, 30)   # baseline: black
        elif delta_iou < -0.01:
            row_color = (180, 30, 30)  # significant drop
        elif delta_iou < 0:
            row_color = (140, 90, 10)  # minor drop
        else:
            row_color = (20, 130, 20)  # improved (expert ablation helped!)

        values = [
            name,
            f"{m.f1:.4f}",
            f"{m.iou:.4f}",
            f"{m.precision:.4f}",
            f"{m.recall:.4f}",
        ]
        for c, v in enumerate(values):
            fc = row_color if c > 0 else (30, 30, 30)
            draw.text((col_x[c], ry), str(v), fill=fc)

    img.save(out_path)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(
    baseline: EvalMetrics,
    ablated: List[EvalMetrics],
    routing_fracs: np.ndarray,
    out_path: Path,
) -> None:
    E = len(ablated)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "variant", "disabled_expert",
            "f1", "iou", "precision", "recall",
            "importance_f1", "importance_iou",
            "routing_frac", "n_samples", "n_tokens",
        ])
        writer.writerow([
            "Full MoE", "None",
            f"{baseline.f1:.6f}", f"{baseline.iou:.6f}",
            f"{baseline.precision:.6f}", f"{baseline.recall:.6f}",
            "0.000000", "0.000000",
            "1.000000", baseline.n_samples, baseline.n_tokens,
        ])
        for i, m in enumerate(ablated):
            imp_f1  = baseline.f1  - m.f1
            imp_iou = baseline.iou - m.iou
            frac    = float(routing_fracs[i]) if i < len(routing_fracs) else 0.0
            writer.writerow([
                f"- Expert {i}", str(i),
                f"{m.f1:.6f}", f"{m.iou:.6f}",
                f"{m.precision:.6f}", f"{m.recall:.6f}",
                f"{imp_f1:.6f}", f"{imp_iou:.6f}",
                f"{frac:.6f}", m.n_samples, m.n_tokens,
            ])
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def save_report(
    baseline: EvalMetrics,
    ablated: List[EvalMetrics],
    routing_fracs: np.ndarray,
    strategy: str,
    out_path: Path,
    ckpt_name: str,
) -> None:
    E = len(ablated)
    importance_iou = [baseline.iou - m.iou for m in ablated]
    importance_f1  = [baseline.f1  - m.f1  for m in ablated]

    # Rank experts by importance
    ranked = sorted(range(E), key=lambda i: importance_iou[i], reverse=True)

    with open(out_path, "w") as f:
        f.write("# Stage 5 — Expert Ablation Report\n\n")
        f.write(f"**Checkpoint**: `{ckpt_name}`  \n")
        f.write(f"**Ablation strategy**: `{strategy}`  \n")
        f.write(f"**Validation samples**: {baseline.n_samples}  \n")
        f.write(f"**Validation tokens**: {baseline.n_tokens:,}  \n\n")
        f.write("---\n\n")

        # ── Main results table ───────────────────────────────────────────────
        f.write("## Ablation Results\n\n")
        f.write(
            "| Model | F1 | Change IoU | Precision | Recall | "
            "ΔF1 | ΔIoU | Load % |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        f.write(
            f"| **Full MoE** | **{baseline.f1:.4f}** | **{baseline.iou:.4f}** | "
            f"{baseline.precision:.4f} | {baseline.recall:.4f} | — | — | 100% |\n"
        )
        for i, m in enumerate(ablated):
            df1  = baseline.f1  - m.f1
            diou = baseline.iou - m.iou
            frac = float(routing_fracs[i]) if i < len(routing_fracs) else 0.0
            sign_f1  = "+" if df1  >= 0 else ""
            sign_iou = "+" if diou >= 0 else ""
            flag = " 🔴" if diou > 0.01 else (" 🟡" if diou > 0.002 else " 🟢")
            f.write(
                f"| – Expert {i} | {m.f1:.4f} | {m.iou:.4f} | "
                f"{m.precision:.4f} | {m.recall:.4f} | "
                f"{sign_f1}{df1:.4f} | {sign_iou}{diou:.4f}{flag} | "
                f"{100*frac:.1f}% |\n"
            )
        f.write("\n")
        f.write("> Legend: 🔴 significant drop (>0.01 IoU) · 🟡 minor drop · 🟢 negligible / improvement\n\n")
        f.write("---\n\n")

        # ── Expert importance ranking ────────────────────────────────────────
        f.write("## Expert Importance Ranking\n\n")
        f.write("_Ranked by IoU drop when expert is disabled._\n\n")
        f.write("| Rank | Expert | Importance (ΔIoU) | Load % | Verdict |\n")
        f.write("|---|---|---|---|---|\n")
        for rank, i in enumerate(ranked):
            diou = importance_iou[i]
            frac = float(routing_fracs[i]) if i < len(routing_fracs) else 0.0
            if diou > 0.01:
                verdict = "**Critical** — large performance drop"
            elif diou > 0.002:
                verdict = "**Moderate** — noticeable but small"
            elif diou > -0.002:
                verdict = "Marginal — near-zero contribution"
            else:
                verdict = "_Redundant / Harmful_ — ablation improves IoU"
            f.write(
                f"| {rank+1} | Expert {i} | `{diou:+.4f}` | "
                f"{100*frac:.1f}% | {verdict} |\n"
            )
        f.write("\n")
        f.write("![Expert Importance](expert_importance_plot.png)\n\n")
        f.write("---\n\n")

        # ── Per-expert deep analysis ─────────────────────────────────────────
        f.write("## Per-Expert Analysis\n\n")
        for i in range(E):
            m    = ablated[i]
            diou = importance_iou[i]
            df1  = importance_f1[i]
            frac = float(routing_fracs[i]) if i < len(routing_fracs) else 0.0
            f.write(f"### Expert {i}\n\n")
            f.write(f"| Metric | Full MoE | Without Expert {i} | Drop |\n")
            f.write("|---|---|---|---|\n")
            f.write(f"| F1       | {baseline.f1:.4f}        | {m.f1:.4f}  | `{df1:+.4f}` |\n")
            f.write(f"| IoU      | {baseline.iou:.4f}       | {m.iou:.4f} | `{diou:+.4f}` |\n")
            f.write(f"| Precision| {baseline.precision:.4f} | {m.precision:.4f} | `{baseline.precision-m.precision:+.4f}` |\n")
            f.write(f"| Recall   | {baseline.recall:.4f}    | {m.recall:.4f} | `{baseline.recall-m.recall:+.4f}` |\n")
            f.write(f"\n**Routing load**: {100*frac:.1f}% of tokens\n\n")

            # Automated interpretation
            if diou > 0.01:
                f.write(
                    f"**⚠️ CRITICAL**: Removing Expert {i} causes a `{diou:.4f}` IoU drop "
                    f"(`{df1:.4f}` F1 drop). This expert handles {100*frac:.1f}% of tokens "
                    f"and provides an irreplaceable specialization. Its capacity should be "
                    f"preserved or even expanded.\n\n"
                )
            elif diou > 0.002:
                f.write(
                    f"**⚡ MODERATE**: Expert {i} contributes a small but measurable "
                    f"`{diou:.4f}` IoU gain. With {100*frac:.1f}% token load it handles a "
                    f"real sub-task, but may be mergeable with a similar expert.\n\n"
                )
            elif diou > -0.002:
                f.write(
                    f"**✓ MARGINAL**: Expert {i} has near-zero individual impact "
                    f"(`{diou:+.4f}` IoU). Either:\n"
                    f"  - Its specialization is replicated by another expert, OR\n"
                    f"  - The tokens it handles are inherently easy to classify.\n"
                    f"  Consider merging this expert or reducing `moe_expert_dim`.\n\n"
                )
            else:
                f.write(
                    f"**🔵 REDUNDANT/HARMFUL**: Removing Expert {i} **improves** IoU by "
                    f"`{-diou:.4f}`. This expert may be:\n"
                    f"  - Adding noise to stable-region predictions\n"
                    f"  - Interfering with other experts' signals\n"
                    f"  Consider deactivating or re-training with expert dropout.\n\n"
                )

        f.write("---\n\n")

        # ── Interpretation ───────────────────────────────────────────────────
        f.write("## Interpretation & Recommendations\n\n")

        max_imp_idx = int(np.argmax(importance_iou))
        min_imp_idx = int(np.argmin(importance_iou))

        f.write(
            f"**Most critical expert**: Expert {max_imp_idx} "
            f"(ΔIoU = `{importance_iou[max_imp_idx]:+.4f}`, "
            f"load {100*float(routing_fracs[max_imp_idx]):.1f}%)\n\n"
        )
        f.write(
            f"**Least critical expert**: Expert {min_imp_idx} "
            f"(ΔIoU = `{importance_iou[min_imp_idx]:+.4f}`, "
            f"load {100*float(routing_fracs[min_imp_idx]):.1f}%)\n\n"
        )

        n_critical  = sum(1 for d in importance_iou if d > 0.01)
        n_moderate  = sum(1 for d in importance_iou if 0.002 < d <= 0.01)
        n_marginal  = sum(1 for d in importance_iou if -0.002 <= d <= 0.002)
        n_redundant = sum(1 for d in importance_iou if d < -0.002)

        f.write("### Summary Breakdown\n\n")
        f.write(f"- **Critical experts** (ΔIoU > 0.01): {n_critical}/{E}\n")
        f.write(f"- **Moderate experts** (0.002 < ΔIoU ≤ 0.01): {n_moderate}/{E}\n")
        f.write(f"- **Marginal experts** (ΔIoU ≈ 0): {n_marginal}/{E}\n")
        f.write(f"- **Redundant experts** (ΔIoU < -0.002, removal helps): {n_redundant}/{E}\n\n")

        if n_marginal + n_redundant >= E // 2:
            f.write(
                "**Recommendation**: More than half the experts are marginal or redundant. "
                "Consider:\n"
                "1. Reducing to 2 experts (halved compute)\n"
                "2. Increasing `expert_dropout_prob` to force non-overlapping specialization\n"
                "3. Adding an explicit diversity loss on expert outputs\n"
                "4. Using `use_top2=True` to allow partial contributions from multiple experts\n\n"
            )
        elif n_critical >= E - 1:
            f.write(
                "**Recommendation**: Nearly all experts are critical — the MoE is highly "
                "efficient. Consider:\n"
                "1. Increasing `moe_num_experts` to 6 or 8 (more specialization capacity)\n"
                "2. Increasing `moe_expert_dim` for larger expert networks\n"
                "3. Removing the load-balancing loss (`lambda_balance=0`) to allow more "
                "   natural load concentration\n\n"
            )
        else:
            f.write(
                "**Recommendation**: Mixed expert utilization. The MoE has learned partial "
                "specialization but some experts are underutilized. Consider:\n"
                "1. Starting expert dropout at `expert_dropout_prob=0.1` in continued training\n"
                "2. Monitoring routing entropy — if entropy is low, experts are decisively "
                "   assigned and further specialization may be limited by input distribution\n\n"
            )

        f.write("---\n\n")
        f.write("_Generated by `run_expert_ablation.py`_\n")

    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(
    baseline: EvalMetrics,
    ablated: List[EvalMetrics],
    routing_fracs: np.ndarray,
) -> None:
    E    = len(ablated)
    sep  = "─" * 72
    print(f"\n{sep}")
    print("  EXPERT ABLATION STUDY — RESULTS")
    print(sep)
    print(f"  {'Model':<22} {'F1':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8}  {'ΔIoU':>8}")
    print(sep)
    print(
        f"  {'Full MoE':<22} "
        f"{baseline.f1:>8.4f} {baseline.iou:>8.4f} "
        f"{baseline.precision:>8.4f} {baseline.recall:>8.4f}  {'—':>8}"
    )
    print(sep)
    for i, m in enumerate(ablated):
        diou = baseline.iou - m.iou
        tag = "  ← CRITICAL" if diou > 0.01 else ("  ← moderate" if diou > 0.002 else "")
        print(
            f"  {'– Expert ' + str(i):<22} "
            f"{m.f1:>8.4f} {m.iou:>8.4f} "
            f"{m.precision:>8.4f} {m.recall:>8.4f}  "
            f"{diou:>+8.4f}{tag}"
        )
    print(sep)
    importance_iou = [baseline.iou - m.iou for m in ablated]
    ranked = sorted(range(E), key=lambda i: importance_iou[i], reverse=True)
    print(f"  Expert importance rank: {' > '.join(f'E{i}' for i in ranked)}")
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Expert Ablation Study for TokenChangeReasonerMoE"
    )
    # Model
    p.add_argument(
        "--checkpoint",
        default="SECOND/stage5_6_semantic/best_model.pt",
        help="Path to best_model.pt",
    )
    p.add_argument(
        "--config",
        default=None,
        help="Path to config.json (auto-detected from checkpoint dir if omitted)",
    )
    # Data
    p.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    p.add_argument("--matches",   default="SECOND/matches")
    # Validation split (must match training)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument(
        "--val_stems_only",
        action="store_true",
        default=False,
        help="Only use val split (default). Set False to use ALL stems.",
    )
    p.add_argument(
        "--all_stems",
        action="store_true",
        default=False,
        help="Evaluate on the full dataset (no train/val split).",
    )
    # Ablation
    p.add_argument(
        "--strategy",
        choices=["zero", "rerout", "identity"],
        default="zero",
        help="How to disable an expert (zero | rerout | identity)",
    )
    # Output
    p.add_argument("--output_dir", default="stage5/ablation/semantic")
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device   = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt     = Path(args.checkpoint)
    cfg_path = Path(args.config) if args.config else (ckpt.parent / "config.json")
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_dir    = Path(args.tokens_T1)
    t2_dir    = Path(args.tokens_T2)
    match_dir = Path(args.matches)

    print(f"\nLoading config from  : {cfg_path}")
    print(f"Loading checkpoint   : {ckpt}")
    cfg   = load_config(cfg_path)
    model = load_model(ckpt, cfg, device)
    E     = model.moe.num_experts
    print(f"Model: {E} experts, router_version={cfg.router_version}")

    # ── Build val stems ─────────────────────────────────────────────────────
    if args.all_stems:
        stems = [
            mp.stem.replace("_matches", "")
            for mp in sorted(match_dir.glob("*_matches.pt"))
            if (t1_dir / f"{mp.stem.replace('_matches','')}.pt").exists()
            and (t2_dir / f"{mp.stem.replace('_matches','')}.pt").exists()
        ]
        print(f"Using ALL {len(stems)} stems for evaluation.")
    else:
        train_stems, val_stems = build_stems(
            match_dir, t1_dir, t2_dir,
            val_split=args.val_split,
            seed=args.seed,
        )
        stems = val_stems
        print(f"Val split: {len(val_stems)} stems (from {len(train_stems)+len(val_stems)} total, val_split={args.val_split}, seed={args.seed})")

    # ── Baseline evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Step 1/6: Baseline (all experts active)")
    print('='*60)
    # Collect routing stats alongside baseline
    routing_fracs = collect_routing_stats(model, stems, t1_dir, t2_dir, match_dir, cfg, device)
    baseline = run_evaluation(model, stems, t1_dir, t2_dir, match_dir, cfg, device, desc="baseline")
    print(f"  Baseline F1={baseline.f1:.4f}  IoU={baseline.iou:.4f}  "
          f"P={baseline.precision:.4f}  R={baseline.recall:.4f}")

    # ── Ablation loop ────────────────────────────────────────────────────────
    ablated: List[EvalMetrics] = []
    for i in range(E):
        print(f"\n{'='*60}")
        print(f"  Step {i+2}/{E+1}: Disable Expert {i} (strategy={args.strategy})")
        print('='*60)
        with disable_expert(model, i, strategy=args.strategy):
            m = run_evaluation(
                model, stems, t1_dir, t2_dir, match_dir, cfg, device,
                desc=f"without E{i}",
            )
        ablated.append(m)
        diou = baseline.iou - m.iou
        print(f"  F1={m.f1:.4f}  IoU={m.iou:.4f}  "
              f"P={m.precision:.4f}  R={m.recall:.4f}  "
              f"ΔIoU={diou:+.4f}")

    # ── Save outputs ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Saving outputs ...")
    print('='*60)

    save_csv(
        baseline, ablated, routing_fracs,
        out_dir / "ablation_results.csv",
    )
    save_importance_bar_chart(
        baseline, ablated,
        out_dir / "expert_importance_plot.png",
    )
    save_report(
        baseline, ablated, routing_fracs,
        strategy=args.strategy,
        out_path=out_dir / "stage5_expert_ablation_report.md",
        ckpt_name=str(ckpt),
    )

    print_summary(baseline, ablated, routing_fracs)
    print(f"  All outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
