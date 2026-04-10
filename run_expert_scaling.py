#!/usr/bin/env python3
"""
run_expert_scaling.py
=====================
Expert Scaling Experiment for TokenChangeReasonerMoE.

Trains (or reuses) one model per expert count, then evaluates all of them on the
same validation split and produces a comparison table, line-chart, and report.

Expert counts to sweep: 2, 4, 8, 16  (configurable via --experts).

Design rules (controlled variables):
  - Same dataset          (tokens_T1 / tokens_T2 / matches)
  - Same backbone         (Transformer + GraphReasoner, identical dims)
  - Same training schedule (epochs, lr, schedule, batch_size)
  - Same random seed
  - Same loss functions   (BCE change loss + delta loss + balance + entropy)
  - Only moe_num_experts varies

Baseline reuse:
  If --baseline_dir is provided (default: SECOND/stage5_6_dynamic), the E=4
  checkpoint is reused and training is skipped for that configuration.

Output (--output_dir):
  scaling_results.csv
  experts_vs_iou_plot.png
  expert_scaling_report.md

Usage:
  python run_expert_scaling.py \\
      --experts      2 4 8 16 \\
      --baseline_dir SECOND/stage5_6_dynamic \\
      --output_dir   stage5/scaling \\
      --epochs       60 \\
      --device       cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from tqdm import tqdm

# ── allow imports from phase1 root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

# One color per expert count in [2, 4, 8, 16]
N_COLORS = {
    2:  ( 31, 119, 180),   # blue
    4:  ( 44, 160,  44),   # green  ← baseline
    8:  (214,  39,  40),   # red
    16: (255, 127,  14),   # orange
}

# ─────────────────────────────────────────────────────────────────────────────
# Model loading
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
# Validation split (mirrors train_reasoner.py defaults)
# ─────────────────────────────────────────────────────────────────────────────

def build_val_stems(
    match_dir: Path,
    t1_dir:    Path,
    t2_dir:    Path,
    val_split: float = 0.1,
    seed:      int   = 42,
) -> List[str]:
    """Return the val stems produced by the training split."""
    all_stems = []
    for mp in sorted(match_dir.glob("*_matches.pt")):
        stem = mp.stem.replace("_matches", "")
        if (t1_dir / f"{stem}.pt").exists() and (t2_dir / f"{stem}.pt").exists():
            all_stems.append(stem)

    n_val   = max(1, int(len(all_stems) * val_split))
    n_train = len(all_stems) - n_val
    gen     = torch.Generator().manual_seed(seed)
    perm    = torch.randperm(len(all_stems), generator=gen).tolist()
    return [all_stems[i] for i in perm[n_train:]]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Metrics:
    n_experts:  int
    f1:         float
    iou:        float
    precision:  float
    recall:     float
    n_tokens:   int
    n_samples:  int
    train_time_s: float = 0.0   # filled from training log if available


@torch.no_grad()
def evaluate(
    model:     TokenChangeReasonerMoE,
    stems:     List[str],
    t1_dir:    Path,
    t2_dir:    Path,
    match_dir: Path,
    cfg:       MoEConfig,
    device:    torch.device,
    desc:      str = "eval",
) -> Tuple[float, float, float, float, int, int]:
    """Returns (f1, iou, precision, recall, n_tokens, n_samples)."""
    tp = fp = fn = tn = 0
    n_tokens  = 0
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

        raw_pairs = mtch.get("pairs", [])
        if isinstance(raw_pairs, list):
            p_tensor = (
                torch.tensor(
                    [[float(p[0]), float(p[1]), float(p[2])] for p in raw_pairs],
                    dtype=torch.float32,
                )
                if raw_pairs else torch.zeros(0, 3)
            )
        else:
            p_tensor = raw_pairs.float()

        sample = SampleData(
            tokens_t1    = t1["tokens"].float(),
            tokens_t2    = t2["tokens"].float(),
            centroids_t1 = t1["centroids"].float(),
            centroids_t2 = t2["centroids"].float(),
            areas_t1     = t1["areas"].float(),
            areas_t2     = t2["areas"].float(),
            match_pairs  = p_tensor,
            change_labels= None,   # use proxy labels
        )

        batch  = build_batch([sample], cfg, device)
        output = model(batch)

        logits  = output["change_logits"]  # [B, N]
        labels  = batch["change_labels"]   # [B, N] proxy
        valid   = ~batch["padding_mask"]

        pred = (logits[valid] > 0).long()
        targ = labels[valid].long()

        tp += int(((pred == 1) & (targ == 1)).sum())
        fp += int(((pred == 1) & (targ == 0)).sum())
        fn += int(((pred == 0) & (targ == 1)).sum())
        tn += int(((pred == 0) & (targ == 0)).sum())
        n_tokens  += int(valid.sum())
        n_samples += 1

    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2 * prec * rec / max(prec + rec, 1e-8)
    iou  = tp / max(tp + fp + fn, 1)
    return f1, iou, prec, rec, n_tokens, n_samples


def read_train_time(log_path: Path) -> float:
    """Sum time_s column from training log CSV."""
    if not log_path.exists():
        return 0.0
    total = 0.0
    try:
        with open(log_path) as f:
            for row in csv.DictReader(f):
                total += float(row.get("time_s", 0))
    except Exception:
        pass
    return total


# ─────────────────────────────────────────────────────────────────────────────
# Training via subprocess
# ─────────────────────────────────────────────────────────────────────────────

def train_config(
    n_experts:    int,
    out_dir:      Path,
    t1_dir:       Path,
    t2_dir:       Path,
    match_dir:    Path,
    epochs:       int,
    batch_size:   int,
    lr:           float,
    seed:         int,
    device:       str,
    base_cfg:     MoEConfig,
    resume:       bool = True,
) -> None:
    """Launch train_reasoner.py as a subprocess with live output."""
    ckpt = out_dir / "best_model.pt"
    if resume and ckpt.exists():
        print(f"  [skip] checkpoint exists: {ckpt}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(Path(__file__).parent / "train_reasoner.py"),
        "--model_type",       "moe",
        "--tokens_T1",        str(t1_dir),
        "--tokens_T2",        str(t2_dir),
        "--matches",          str(match_dir),
        "--output",           str(out_dir),
        # Controlled variables — match baseline exactly
        "--hidden_dim",       str(base_cfg.hidden_dim),
        "--num_layers",       str(base_cfg.num_layers),
        "--num_heads",        str(base_cfg.num_heads),
        "--dropout",          str(base_cfg.dropout),
        "--graph_k",          str(base_cfg.graph_k),
        "--graph_layers",     str(base_cfg.graph_layers),
        "--proxy_threshold",  str(base_cfg.proxy_delta_threshold),
        "--delta_weight",     str(base_cfg.delta_loss_weight),
        "--router_version",   base_cfg.router_version,
        "--moe_expert_dim",   str(base_cfg.moe_expert_dim),
        "--lambda_balance",   str(base_cfg.lambda_balance),
        "--lambda_entropy",   str(base_cfg.lambda_entropy),
        # Variable
        "--moe_num_experts",  str(n_experts),
        # Training schedule
        "--epochs",           str(epochs),
        "--batch_size",       str(batch_size),
        "--lr",               str(lr),
        "--weight_decay",     "0.01",
        "--val_split",        "0.1",
        "--seed",             str(seed),
        "--save_every",       "10",
        "--device",           device,
        "--amp",
    ]

    print(f"\n  [train] E={n_experts}  →  {out_dir}")
    print(f"  cmd: {' '.join(cmd)}\n")

    # Stream output live so user can follow progress
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"Training failed for E={n_experts} (exit code {proc.returncode})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Visualization — PIL only
# ─────────────────────────────────────────────────────────────────────────────

def _lerp_color(c0, c1, t):
    return tuple(int(c0[k] + (c1[k] - c0[k]) * t) for k in range(3))


def save_scaling_plot(
    results:  List[Metrics],
    out_path: Path,
) -> None:
    """
    Line chart: Experts (x) vs Change IoU and F1 (y).
    X axis uses equal spacing with tick labels [2, 4, 8, 16].
    Two lines: IoU (solid) and F1 (dashed-ish).
    Baseline (E=4) marked with a ●.
    """
    W, H   = 860, 560
    ML, MR = 80, 40    # margin left/right
    MT, MB = 50, 80    # margin top/bottom
    PW     = W - ML - MR
    PH     = H - MT - MB

    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    title = "Expert Scaling: Number of Experts vs Performance"
    draw.text((W // 2 - len(title) * 4, 14), title, fill=(30, 30, 30))

    # Sort by n_experts
    results_sorted = sorted(results, key=lambda m: m.n_experts)
    xs_raw = [m.n_experts for m in results_sorted]
    iou_vals = [m.iou       for m in results_sorted]
    f1_vals  = [m.f1        for m in results_sorted]

    # Equal spacing on x axis
    N = len(results_sorted)
    x_px = [ML + int(i / max(N - 1, 1) * PW) for i in range(N)]

    # Y axis range: include 0 baseline visually
    all_vals = iou_vals + f1_vals
    y_min = max(0.0, min(all_vals) - 0.05)
    y_max = min(1.0, max(all_vals) + 0.05)
    y_span = max(y_max - y_min, 0.01)

    def to_px(v: float) -> int:
        return int(MT + PH - (v - y_min) / y_span * PH)

    # ── Grid lines ──────────────────────────────────────────────────────────
    n_yticks = 6
    for k in range(n_yticks + 1):
        yv = y_min + k / n_yticks * y_span
        yp = to_px(yv)
        draw.line([(ML, yp), (ML + PW, yp)], fill=(220, 220, 220), width=1)
        draw.text((ML - 48, yp - 6), f"{yv:.3f}", fill=(100, 100, 100))

    # ── Axes ────────────────────────────────────────────────────────────────
    draw.line([(ML, MT), (ML, MT + PH)], fill=(80, 80, 80), width=2)
    draw.line([(ML, MT + PH), (ML + PW, MT + PH)], fill=(80, 80, 80), width=2)

    # Axis labels
    draw.text((ML + PW // 2 - 60, H - 22), "Number of Experts", fill=(50, 50, 50))
    draw.text((8, MT + PH // 2 - 10), "Score", fill=(50, 50, 50))

    # X tick labels
    for i, (xp, ne) in enumerate(zip(x_px, xs_raw)):
        draw.line([(xp, MT + PH - 4), (xp, MT + PH + 4)], fill=(80, 80, 80))
        draw.text((xp - 6, MT + PH + 10), str(ne), fill=(50, 50, 50))

    # ── IoU line (blue, thick) ───────────────────────────────────────────────
    iou_pts = [(x_px[i], to_px(iou_vals[i])) for i in range(N)]
    for i in range(N - 1):
        draw.line([iou_pts[i], iou_pts[i + 1]], fill=(31, 119, 180), width=3)

    # ── F1 line (orange) ────────────────────────────────────────────────────
    f1_pts = [(x_px[i], to_px(f1_vals[i])) for i in range(N)]
    for i in range(N - 1):
        # Dashed: alternate segments of 8px
        x0, y0 = f1_pts[i]
        x1, y1 = f1_pts[i + 1]
        length = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        steps  = max(int(length / 8), 1)
        for s in range(steps):
            if s % 2 == 0:
                tx0 = x0 + (x1 - x0) * s / steps
                ty0 = y0 + (y1 - y0) * s / steps
                tx1 = x0 + (x1 - x0) * (s + 1) / steps
                ty1 = y0 + (y1 - y0) * (s + 1) / steps
                draw.line(
                    [(int(tx0), int(ty0)), (int(tx1), int(ty1))],
                    fill=(255, 127, 14), width=2,
                )

    # ── Markers ─────────────────────────────────────────────────────────────
    for i, m in enumerate(results_sorted):
        xp      = x_px[i]
        iou_yp  = to_px(m.iou)
        f1_yp   = to_px(m.f1)
        is_base = (m.n_experts == 4)
        R       = 7 if is_base else 5

        # IoU circle
        draw.ellipse(
            [xp - R, iou_yp - R, xp + R, iou_yp + R],
            fill=(31, 119, 180), outline=(255, 255, 255),
        )
        # F1 square
        draw.rectangle(
            [xp - R + 1, f1_yp - R + 1, xp + R - 1, f1_yp + R - 1],
            fill=(255, 127, 14), outline=(255, 255, 255),
        )
        # Value labels
        draw.text((xp + R + 3, iou_yp - 7),  f"{m.iou:.3f}", fill=(31, 119, 180))
        draw.text((xp + R + 3, f1_yp  - 7),  f"{m.f1:.3f}",  fill=(255, 127, 14))
        # Baseline star annotation
        if is_base:
            draw.text((xp - 14, iou_yp - 20), "★ baseline", fill=(80, 80, 80))

    # ── Legend ───────────────────────────────────────────────────────────────
    lx, ly = ML + PW - 140, MT + 10
    draw.rectangle([lx - 6, ly - 6, lx + 136, ly + 54], fill=(245, 245, 245), outline=(200, 200, 200))
    # IoU line + circle
    draw.line([(lx, ly + 10), (lx + 24, ly + 10)], fill=(31, 119, 180), width=3)
    draw.ellipse([lx + 8, ly + 6, lx + 18, ly + 14], fill=(31, 119, 180))
    draw.text((lx + 30, ly + 4), "Change IoU", fill=(30, 30, 30))
    # F1 dashed line + square
    draw.line([(lx, ly + 32), (lx + 24, ly + 32)], fill=(255, 127, 14), width=2)
    draw.rectangle([lx + 9, ly + 27, lx + 17, ly + 37], fill=(255, 127, 14))
    draw.text((lx + 30, ly + 26), "F1 Score", fill=(30, 30, 30))

    img.save(out_path)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results: List[Metrics], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "n_experts", "f1", "iou", "precision", "recall",
            "n_samples", "n_tokens", "train_time_s",
            "params_moe_million",
        ])
        for m in sorted(results, key=lambda x: x.n_experts):
            # Rough MoE param count: 2 × E × (H × D_exp + D_exp + D_exp × H + H)
            # = 2 × E × H × (2 * D_exp + 1)  (ignoring biases)
            H, D = 384, 512
            E    = m.n_experts
            moe_params = E * (H * D + D + D * H + H)   # net[0] + net[2] weights+biases
            router_params = E * (H + 1)                # router linear
            total_new = moe_params + router_params
            writer.writerow([
                m.n_experts,
                f"{m.f1:.6f}", f"{m.iou:.6f}",
                f"{m.precision:.6f}", f"{m.recall:.6f}",
                m.n_samples, m.n_tokens,
                f"{m.train_time_s:.1f}",
                f"{total_new / 1e6:.2f}",
            ])
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def save_report(results: List[Metrics], out_path: Path, base_cfg: MoEConfig) -> None:
    rs = sorted(results, key=lambda m: m.n_experts)
    baseline = next((m for m in rs if m.n_experts == 4), rs[0])

    best_iou = max(rs, key=lambda m: m.iou)
    best_f1  = max(rs, key=lambda m: m.f1)

    with open(out_path, "w") as f:
        f.write("# Expert Scaling Experiment Report\n\n")
        f.write(f"**Router version**: `{base_cfg.router_version}`  \n")
        f.write(f"**Expert dim** (D_exp): `{base_cfg.moe_expert_dim}`  \n")
        f.write(f"**Hidden dim** (H): `{base_cfg.hidden_dim}`  \n")
        f.write(f"**Validation samples**: {baseline.n_samples}  \n\n")
        f.write("---\n\n")

        # ── Results table ────────────────────────────────────────────────────
        f.write("## Results Table\n\n")
        f.write(
            "| Experts | F1 | Change IoU | Precision | Recall | "
            "ΔIoU vs E4 | MoE Params (M) |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        H, D = base_cfg.hidden_dim, base_cfg.moe_expert_dim
        for m in rs:
            E       = m.n_experts
            delta   = m.iou - baseline.iou
            sign    = "+" if delta >= 0 else ""
            mark_f1  = " **★**" if m.n_experts == best_f1.n_experts else ""
            mark_iou = " **★**" if m.n_experts == best_iou.n_experts else ""
            moe_p   = E * (H * D + D + D * H + H + E * (H + 1))
            moe_p_m = moe_p / 1e6
            base_row = " ← baseline" if m.n_experts == 4 else ""
            f.write(
                f"| **{E}**{base_row} | {m.f1:.4f}{mark_f1} | "
                f"{m.iou:.4f}{mark_iou} | "
                f"{m.precision:.4f} | {m.recall:.4f} | "
                f"`{sign}{delta:.4f}` | {moe_p_m:.2f} |\n"
            )
        f.write("\n> ★ = best value across all configurations\n\n")
        f.write("![Scaling Plot](experts_vs_iou_plot.png)\n\n")
        f.write("---\n\n")

        # ── Expert importance chart ──────────────────────────────────────────
        f.write("## Expert Importance (IoU improvement vs E=2)\n\n")
        e2 = next((m for m in rs if m.n_experts == 2), None)
        if e2:
            f.write("| Experts | ΔIoU vs E=2 | ΔF1 vs E=2 |\n|---|---|---|\n")
            for m in rs:
                d_iou = m.iou - e2.iou
                d_f1  = m.f1  - e2.f1
                f.write(f"| {m.n_experts} | `{d_iou:+.4f}` | `{d_f1:+.4f}` |\n")
            f.write("\n")

        # ── Analysis ─────────────────────────────────────────────────────────
        f.write("## Analysis\n\n")

        ious      = [m.iou for m in rs]
        n_experts = [m.n_experts for m in rs]

        # 1. Does performance increase with more experts?
        mono_up = all(ious[i] <= ious[i + 1] for i in range(len(ious) - 1))
        mono_dn = all(ious[i] >= ious[i + 1] for i in range(len(ious) - 1))
        peak_idx  = int(np.argmax(ious))
        best_ne   = n_experts[peak_idx]

        f.write("### 1. Does performance increase with more experts?\n\n")
        if mono_up:
            f.write(
                f"**Yes — monotonically increasing.** IoU improves at every step from "
                f"E=2 ({rs[0].iou:.4f}) to E={rs[-1].n_experts} ({rs[-1].iou:.4f}). "
                f"More experts consistently provide complementary specialization.\n\n"
            )
        elif mono_dn:
            f.write(
                f"**Counter-intuitively, performance decreases** with more experts. "
                f"The best configuration is E={best_ne} (IoU={ious[peak_idx]:.4f}). "
                f"More experts may introduce routing noise or require more training data "
                f"than available.\n\n"
            )
        else:
            f.write(
                f"**Non-monotonic.** Performance peaks at E={best_ne} "
                f"(IoU={ious[peak_idx]:.4f}), then "
                f"{'increases' if ious[-1] > ious[peak_idx - 1] else 'decreases'} "
                f"with {n_experts[-1]} experts.\n\n"
            )

        # 2. Saturation?
        iou_range = max(ious) - min(ious)
        f.write("### 2. Does performance saturate?\n\n")
        if iou_range < 0.005:
            f.write(
                f"**Yes — strong saturation.** The full IoU range across all configs is "
                f"only `{iou_range:.4f}`. The model's bottleneck lies elsewhere "
                f"(token features, label noise, or the proxy label quality), not in the "
                f"number of experts.\n\n"
            )
        elif iou_range < 0.02:
            f.write(
                f"**Partial saturation.** IoU range across configs is `{iou_range:.4f}` "
                f"— small but detectable. Going from E=2 to E={best_ne} provides real "
                f"gains, but further increases yield diminishing returns.\n\n"
            )
        else:
            f.write(
                f"**Not saturated.** IoU range `{iou_range:.4f}` is substantial. "
                f"There is a clear benefit to increasing experts up to E={best_ne}.\n\n"
            )

        # 3. Can too many experts hurt?
        last_iou = ious[-1]
        second_last_iou = ious[-2] if len(ious) > 1 else ious[-1]
        f.write("### 3. Can too many experts hurt performance?\n\n")
        if last_iou < second_last_iou - 0.002:
            f.write(
                f"**Yes.** Going from E={n_experts[-2]} to E={n_experts[-1]} drops IoU "
                f"by `{second_last_iou - last_iou:.4f}`. Possible causes:\n"
                f"- With {n_experts[-1]} experts and only ~{baseline.n_samples} val samples, "
                f"each expert sees fewer tokens → underfitting\n"
                f"- The load balancing loss spreads tokens too thinly\n"
                f"- Gradient updates for rarely-used experts are noisy\n\n"
            )
        else:
            f.write(
                f"**Marginally** or not at all. The highest expert count "
                f"(E={n_experts[-1]}, IoU={last_iou:.4f}) does not significantly "
                f"underperform E={n_experts[-2]} (IoU={second_last_iou:.4f}). "
                f"The architecture scales gracefully, though the gains are diminishing.\n\n"
            )

        # 4. Recommendation
        f.write("### 4. Recommendation\n\n")
        if best_ne == 4:
            f.write(
                "**E=4 (baseline) is optimal** for this dataset and training schedule. "
                "Increasing to 8 or 16 experts adds parameters without proportional "
                "performance gain.  \n\n"
                "**Suggestion**: Instead of more experts, try:\n"
                "- Larger `moe_expert_dim` (512 → 1024) to give each expert more capacity\n"
                "- `use_top2=True` to allow soft expert blending\n"
                "- More epochs (current 60 may not be enough for E=16)\n\n"
            )
        elif best_ne == 8:
            f.write(
                f"**E=8 gives the best performance** (IoU={ious[peak_idx]:.4f} vs "
                f"E=4 baseline IoU={baseline.iou:.4f}). The MoE benefits from finer "
                f"specialization with 8 experts — but E=16 shows diminishing returns.  \n\n"
                "**Suggestion**: Adopt E=8 as the new baseline with:\n"
                "- `expert_dropout_prob=0.1` to ensure all 8 experts are actively used\n"
                "- Monitor routing load — E=16 may collapse to effectively 4-6 active experts\n\n"
            )
        elif best_ne > 8:
            f.write(
                f"**E={best_ne} gives the best performance** (IoU={ious[peak_idx]:.4f}). "
                f"Large-scale MoE works for this task.  \n\n"
                "**Important**: Verify that all experts are actually used (routing load "
                "should be roughly uniform). Consider reducing `lambda_balance` if "
                "routing has collapsed to fewer experts.\n\n"
            )
        else:
            f.write(
                f"**E={best_ne} gives the best performance** (IoU={ious[peak_idx]:.4f}). "
                f"Fewer experts may work better when training data is limited.\n\n"
            )

        # 5. Compute/performance tradeoff
        f.write("### 5. Compute-Performance Tradeoff\n\n")
        f.write("| Experts | MoE Params (M) | Relative cost | IoU gain vs E=2 |\n")
        f.write("|---|---|---|---|\n")
        H, D = base_cfg.hidden_dim, base_cfg.moe_expert_dim
        e2_iou = rs[0].iou  # smallest expert count as reference
        e2_p   = 2 * (H * D + D + D * H + H)
        for m in rs:
            E      = m.n_experts
            p      = E * (H * D + D + D * H + H)
            rel    = p / e2_p
            d_iou  = m.iou - e2_iou
            sign   = "+" if d_iou >= 0 else ""
            f.write(
                f"| {E} | {p/1e6:.2f}M | ×{rel:.1f} | `{sign}{d_iou:.4f}` |\n"
            )
        f.write("\n")

        f.write("---\n\n")
        f.write("_Generated by `run_expert_scaling.py`_\n")

    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(results: List[Metrics]) -> None:
    rs = sorted(results, key=lambda m: m.n_experts)
    sep = "─" * 70
    print(f"\n{sep}")
    print("  EXPERT SCALING — RESULTS")
    print(sep)
    print(f"  {'Experts':>8} {'F1':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8}  {'ΔIoU':>8}")
    print(sep)
    baseline_iou = next((m.iou for m in rs if m.n_experts == 4), rs[0].iou)
    for m in rs:
        delta = m.iou - baseline_iou
        flag  = " ← baseline" if m.n_experts == 4 else ""
        best  = " ← BEST" if m.iou == max(r.iou for r in rs) and m.n_experts != 4 else ""
        print(
            f"  {m.n_experts:>8}   {m.f1:>6.4f}  {m.iou:>6.4f}  "
            f"{m.precision:>6.4f}  {m.recall:>6.4f}  {delta:>+7.4f}{flag}{best}"
        )
    print(sep + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Expert Scaling Experiment")
    p.add_argument(
        "--experts", nargs="+", type=int, default=[2, 4, 8, 16],
        help="List of expert counts to sweep",
    )
    p.add_argument(
        "--baseline_dir",
        default="SECOND/stage5_6_dynamic",
        help="Dir of existing 4-expert checkpoint (skips re-training for E=4)",
    )
    p.add_argument(
        "--baseline_experts", type=int, default=4,
        help="Which expert count the baseline_dir corresponds to",
    )
    p.add_argument("--tokens_T1",  default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2",  default="SECOND/tokens_T2")
    p.add_argument("--matches",    default="SECOND/matches")
    p.add_argument("--output_dir", default="stage5/scaling")
    p.add_argument(
        "--checkpoints_root", default="SECOND/scaling",
        help="Root dir where per-expert training output dirs are created",
    )
    p.add_argument("--epochs",     type=int,   default=60)
    p.add_argument("--batch_size", type=int,   default=8)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--val_split",  type=float, default=0.1)
    p.add_argument("--device",     default="cuda")
    p.add_argument(
        "--skip_train", action="store_true", default=False,
        help="Skip training; only evaluate existing checkpoints",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_dir    = Path(args.tokens_T1)
    t2_dir    = Path(args.tokens_T2)
    match_dir = Path(args.matches)

    baseline_dir = Path(args.baseline_dir)
    base_cfg_path = baseline_dir / "config.json"
    print(f"Loading baseline config from: {base_cfg_path}")
    base_cfg = load_config(base_cfg_path)

    # ── Build validation split (same across all configs) ────────────────────
    val_stems = build_val_stems(match_dir, t1_dir, t2_dir, args.val_split, args.seed)
    print(f"Validation stems: {len(val_stems)} (val_split={args.val_split}, seed={args.seed})")

    # ── Step 1: Train / locate checkpoints ──────────────────────────────────
    ckpt_map: Dict[int, Path] = {}   # n_experts → best_model.pt path

    for ne in sorted(set(args.experts)):
        if ne == args.baseline_experts and baseline_dir.exists():
            ckpt = baseline_dir / "best_model.pt"
            if ckpt.exists():
                print(f"\n  [E={ne}] → reusing baseline: {ckpt}")
                ckpt_map[ne] = ckpt
                continue
            # Fall through to training if baseline checkpoint missing

        ckpt_dir = Path(args.checkpoints_root) / f"experts_{ne}"
        ckpt     = ckpt_dir / "best_model.pt"
        ckpt_map[ne] = ckpt

        if not args.skip_train:
            train_config(
                n_experts  = ne,
                out_dir    = ckpt_dir,
                t1_dir     = t1_dir,
                t2_dir     = t2_dir,
                match_dir  = match_dir,
                epochs     = args.epochs,
                batch_size = args.batch_size,
                lr         = args.lr,
                seed       = args.seed,
                device     = args.device,
                base_cfg   = base_cfg,
                resume     = True,
            )
        else:
            if not ckpt.exists():
                print(f"  [E={ne}] WARNING: checkpoint not found at {ckpt}, skipping eval.")

    # ── Step 2: Evaluate all checkpoints ────────────────────────────────────
    results: List[Metrics] = []

    for ne in sorted(ckpt_map.keys()):
        ckpt = ckpt_map[ne]
        if not ckpt.exists():
            print(f"  [E={ne}] checkpoint missing, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"  Evaluating E={ne}  ←  {ckpt}")
        print('='*60)

        # Load the saved config for this checkpoint
        cfg_path = ckpt.parent / "config.json"
        if cfg_path.exists():
            cfg = load_config(cfg_path)
        else:
            # Construct config by patching baseline
            import copy
            cfg = copy.deepcopy(base_cfg)
            cfg.moe_num_experts = ne

        model = load_model(ckpt, cfg, device)

        f1, iou, prec, rec, n_tok, n_sam = evaluate(
            model, val_stems, t1_dir, t2_dir, match_dir, cfg, device,
            desc=f"E={ne}",
        )
        print(f"  E={ne}: F1={f1:.4f}  IoU={iou:.4f}  P={prec:.4f}  R={rec:.4f}")

        # Read training time from log if available
        log_path = ckpt.parent / "training_log.csv"
        t_s = read_train_time(log_path)

        results.append(Metrics(
            n_experts=ne, f1=f1, iou=iou,
            precision=prec, recall=rec,
            n_tokens=n_tok, n_samples=n_sam,
            train_time_s=t_s,
        ))

        # Clean up GPU memory between models
        del model
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if not results:
        print("No results to report — all checkpoints missing.")
        return

    # ── Step 3: Save outputs ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Saving outputs ...")
    print('='*60)

    save_csv(results, out_dir / "scaling_results.csv")
    save_scaling_plot(results, out_dir / "experts_vs_iou_plot.png")
    save_report(results, out_dir / "expert_scaling_report.md", base_cfg)
    print_summary(results)
    print(f"  All outputs saved to: {out_dir}/")


if __name__ == "__main__":
    main()
