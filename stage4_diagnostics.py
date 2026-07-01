"""
stage4_diagnostics.py
=====================
Stage 4A pre-training diagnostic checks.

Checks:
  1. Delta separation     — ||emb_T2 − emb_T1|| for changed vs unchanged tokens
  2. Label distribution   — class balance (expect ~15-20% changed)
  3. Token count dist     — min/mean/max/p95 per pair; warn if >300
  4. Overfit sanity test  — 2 pairs, 200 iters, loss → ~0

Usage:
    python stage4_diagnostics.py \\
        --tokens_T1 SECOND/tokens_T1 \\
        --tokens_T2 SECOND/tokens_T2 \\
        --matches   SECOND/matches   \\
        --model     SECOND/stage4/best_model.pt \\
        --out       SECOND/diagnostics/stage4_diag.png \\
        --n_samples 200
"""

from __future__ import annotations

import argparse
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from token_change_reasoner import (
    ChangeReasonerModel,
    ReasonerConfig,
    SampleData,
    build_batch,
    build_model,
    compute_loss,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_samples(
    t1_dir: Path,
    t2_dir: Path,
    match_dir: Path,
    n: Optional[int] = None,
    seed: int = 42,
) -> List[SampleData]:
    """Load up to n SampleData objects."""
    stems = []
    for mp in sorted(match_dir.glob("*_matches.pt")):
        stem = mp.stem.replace("_matches", "")
        if (t1_dir / f"{stem}.pt").exists() and (t2_dir / f"{stem}.pt").exists():
            stems.append(stem)

    if n is not None and n < len(stems):
        stems = random.Random(seed).sample(stems, n)

    samples = []
    for stem in stems:
        try:
            t1   = torch.load(t1_dir    / f"{stem}.pt",          weights_only=True)
            t2   = torch.load(t2_dir    / f"{stem}.pt",          weights_only=True)
            mtch = torch.load(match_dir / f"{stem}_matches.pt",  weights_only=False)

            pairs_raw = mtch.get("pairs", [])
            if isinstance(pairs_raw, list) and len(pairs_raw) > 0:
                pairs = torch.tensor(
                    [[float(p[0]), float(p[1]), float(p[2])] for p in pairs_raw]
                )
            elif isinstance(pairs_raw, torch.Tensor) and len(pairs_raw) > 0:
                pairs = pairs_raw.float()
            else:
                pairs = torch.zeros(0, 3)

            samples.append(SampleData(
                tokens_t1    = t1["tokens"].float(),
                tokens_t2    = t2["tokens"].float(),
                centroids_t1 = t1["centroids"].float(),
                centroids_t2 = t2["centroids"].float(),
                areas_t1     = t1["areas"].float(),
                areas_t2     = t2["areas"].float(),
                match_pairs  = pairs,
                change_labels= None,
            ))
        except Exception as e:
            log.warning(f"Skipping {stem}: {e}")

    log.info(f"Loaded {len(samples)} samples")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 1 — Delta Separation
# ─────────────────────────────────────────────────────────────────────────────

def check_delta_separation(
    samples: List[SampleData],
    cfg: ReasonerConfig,
    ax: plt.Axes,
) -> Dict:
    """
    For matched pairs split by proxy label, compute ||emb_T2_j − emb_T1_i||.
    Expect: mean_delta_changed > mean_delta_unchanged (clear separation).
    """
    deltas_changed   = []
    deltas_unchanged = []

    for s in samples:
        if len(s.match_pairs) == 0:
            continue
        for p in s.match_pairs:
            i, j = int(p[0]), int(p[1])
            if i >= len(s.tokens_t1) or j >= len(s.tokens_t2):
                continue
            dn  = (s.tokens_t2[j] - s.tokens_t1[i]).norm().item()
            lbl = 1 if dn > cfg.proxy_delta_threshold else 0
            if lbl == 1:
                deltas_changed.append(dn)
            else:
                deltas_unchanged.append(dn)

    dc = np.array(deltas_changed)
    du = np.array(deltas_unchanged)

    mean_c = dc.mean() if len(dc) else 0.0
    mean_u = du.mean() if len(du) else 0.0
    ratio  = mean_c / (mean_u + 1e-6)

    # ── Plot ──────────────────────────────────────────────────────────────
    bins = np.linspace(0, max(dc.max() if len(dc) else 1,
                              du.max() if len(du) else 1) * 1.05, 60)

    if len(du):
        ax.hist(du, bins=bins, color="#51cf66", alpha=0.65,
                label=f"Unchanged  n={len(du)}\nmean={mean_u:.2f}", edgecolor="none")
    if len(dc):
        ax.hist(dc, bins=bins, color="#ff6b6b", alpha=0.75,
                label=f"Changed    n={len(dc)}\nmean={mean_c:.2f}", edgecolor="none")

    ax.axvline(cfg.proxy_delta_threshold, color="white", lw=1.5, ls="--",
               label=f"proxy thr = {cfg.proxy_delta_threshold:.1f}")
    ax.set_xlabel("‖ emb_T2 − emb_T1 ‖  (delta norm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("① Delta Separation  (Changed vs Unchanged)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")

    # Stats box
    sep_ok = ratio > 1.5
    verdict = "✅ Good separation" if sep_ok else "⚠️  Weak separation"
    stats_txt = (f"mean Δ changed:   {mean_c:.3f}\n"
                 f"mean Δ unchanged: {mean_u:.3f}\n"
                 f"ratio:            {ratio:.2f}x\n"
                 f"→ {verdict}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="right", color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                      edgecolor="#444466", alpha=0.9))

    result = {"mean_changed": mean_c, "mean_unchanged": mean_u,
              "ratio": ratio, "sep_ok": sep_ok,
              "n_changed": len(dc), "n_unchanged": len(du)}

    # ── Print ─────────────────────────────────────────────────────────────
    print("\n① Delta Separation")
    print(f"   n_changed   = {len(dc):>8,}")
    print(f"   n_unchanged = {len(du):>8,}")
    print(f"   mean Δ changed   = {mean_c:.4f}")
    print(f"   mean Δ unchanged = {mean_u:.4f}")
    print(f"   ratio            = {ratio:.2f}x")
    print(f"   → {'✅ Good separation (ratio > 1.5)' if sep_ok else '⚠️  Weak separation — check proxy threshold'}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 2 — Label Distribution
# ─────────────────────────────────────────────────────────────────────────────

def check_label_distribution(
    samples: List[SampleData],
    cfg: ReasonerConfig,
    ax: plt.Axes,
) -> Dict:
    """
    Compute overall change label statistics.
    Expect ~15-20% changed tokens. Warn if <5%.
    """
    total_changed   = 0
    total_unchanged = 0

    pair_changed_fracs = []

    for s in samples:
        n1 = len(s.tokens_t1)
        n2 = len(s.tokens_t2)
        n  = n1 + n2
        changed_in_pair = 0

        for p in s.match_pairs:
            i, j = int(p[0]), int(p[1])
            if i < n1 and j < n2:
                dn  = (s.tokens_t2[j] - s.tokens_t1[i]).norm().item()
                if dn > cfg.proxy_delta_threshold:
                    changed_in_pair += 2   # T1 + T2 token both flagged

        unchanged_in_pair = n - changed_in_pair
        total_changed   += changed_in_pair
        total_unchanged += unchanged_in_pair
        pair_changed_fracs.append(changed_in_pair / max(n, 1))

    total = total_changed + total_unchanged
    pct_changed = total_changed / max(total, 1) * 100
    pct_unc     = 100 - pct_changed
    pcf         = np.array(pair_changed_fracs) * 100

    # ── Plot ──────────────────────────────────────────────────────────────
    ax.hist(pcf, bins=40, color="#7c4dff", edgecolor="#9965ff",
            alpha=0.85, linewidth=0.5)
    ax.axvline(pcf.mean(), color="white", lw=1.5, ls=":",
               label=f"Mean = {pcf.mean():.1f}%")
    ax.axvline(5,  color="#ff6b6b", lw=1.2, ls="--", alpha=0.7, label="5% warn")
    ax.axvline(20, color="#51cf66", lw=1.2, ls="--", alpha=0.7, label="20% target")
    ax.set_xlabel("% Changed Tokens per Image Pair", fontsize=11)
    ax.set_ylabel("Count (pairs)", fontsize=11)
    ax.set_title("② Change Label Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")

    ok  = 5 <= pct_changed <= 30
    verdict = "✅ Healthy balance" if ok else ("⚠️  Severely imbalanced (<5% changed)"
                                                if pct_changed < 5 else "⚠️  Very high changed ratio")
    stats_txt = (f"total tokens:   {total:,}\n"
                 f"changed:        {total_changed:,}  ({pct_changed:.1f}%)\n"
                 f"unchanged:      {total_unchanged:,}  ({pct_unc:.1f}%)\n"
                 f"→ {verdict}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="right", color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                      edgecolor="#444466", alpha=0.9))

    result = {"pct_changed": pct_changed, "pct_unchanged": pct_unc,
              "total": total, "n_changed": total_changed, "n_unchanged": total_unchanged,
              "balance_ok": ok}

    print("\n② Label Distribution")
    print(f"   total tokens = {total:,}")
    print(f"   changed      = {total_changed:,}  ({pct_changed:.1f}%)")
    print(f"   unchanged    = {total_unchanged:,}  ({pct_unc:.1f}%)")
    if pct_changed < 5:
        print("   ⚠️  WARNING: Very low changed ratio — consider lowering proxy_threshold")
    elif pct_changed > 30:
        print("   ⚠️  WARNING: High changed ratio — check proxy_threshold")
    else:
        print(f"   ✅ Balance within expected range (5–30%)")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 3 — Token Count Distribution
# ─────────────────────────────────────────────────────────────────────────────

def check_token_count(
    samples: List[SampleData],
    ax: plt.Axes,
) -> Dict:
    """
    Per-pair token count (N1 + N2). Warn if p95 > 300 (Transformer memory).
    """
    counts = np.array([len(s.tokens_t1) + len(s.tokens_t2) for s in samples])

    mn, mx, mu = int(counts.min()), int(counts.max()), counts.mean()
    p50  = float(np.percentile(counts, 50))
    p95  = float(np.percentile(counts, 95))
    p99  = float(np.percentile(counts, 99))

    # ── Plot ──────────────────────────────────────────────────────────────
    ax.hist(counts, bins=40, color="#00b4d8", edgecolor="#48cae4",
            alpha=0.85, linewidth=0.5)
    ax.axvline(mu,  color="white",   lw=1.5, ls=":", label=f"Mean = {mu:.1f}")
    ax.axvline(p95, color="#ffd600", lw=1.5, ls="--", label=f"p95  = {p95:.0f}")
    ax.axvline(300, color="#ff6b6b", lw=1.5, ls="--", alpha=0.8, label="Warn = 300")
    ax.set_xlabel("Tokens per Pair  (N1 + N2)", fontsize=11)
    ax.set_ylabel("Count (pairs)", fontsize=11)
    ax.set_title("③ Token Count Distribution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")

    warn_mem = p95 > 300
    stats_txt = (f"min:  {mn}\n"
                 f"mean: {mu:.1f}\n"
                 f"p50:  {p50:.0f}\n"
                 f"p95:  {p95:.0f}  {'⚠️ >300' if warn_mem else '✅'}\n"
                 f"max:  {mx}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="right", color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                      edgecolor="#444466", alpha=0.9))

    result = {"min": mn, "max": mx, "mean": float(mu),
              "p50": p50, "p95": p95, "p99": p99, "mem_warn": warn_mem}

    print("\n③ Token Count Distribution")
    print(f"   min:  {mn}")
    print(f"   mean: {mu:.1f}")
    print(f"   p50:  {p50:.0f}")
    print(f"   p95:  {p95:.0f}")
    print(f"   p99:  {p99:.0f}")
    print(f"   max:  {mx}")
    if warn_mem:
        print("   ⚠️  WARNING: p95 > 300 — consider chunking long sequences or limiting max_tokens")
    else:
        print("   ✅ Token counts within safe Transformer range")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CHECK 4 — Overfit Sanity Test
# ─────────────────────────────────────────────────────────────────────────────

def check_overfit_sanity(
    samples: List[SampleData],
    cfg: ReasonerConfig,
    ax: plt.Axes,
    device: torch.device,
    n_iters: int = 200,
    lr: float = 3e-3,
) -> Dict:
    """
    Train a fresh model on exactly 2 pairs for n_iters iterations.
    Expect: loss → near zero (model can overfit a tiny set).
    Failure = architecture problem or loss is ill-conditioned.
    """
    log.info(f"Overfit sanity: {n_iters} iters on 2 pairs …")
    tiny_samples = samples[:2]
    batch = build_batch(tiny_samples, cfg, device)

    model = build_model(cfg).to(device)
    model.train()
    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

    loss_curve = []
    time_curve = []
    t0 = time.perf_counter()

    for it in range(n_iters):
        opt.zero_grad()
        outputs = model(batch)
        losses  = compute_loss(outputs, batch, cfg)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_curve.append(float(losses["total_loss"].detach()))
        time_curve.append(time.perf_counter() - t0)

    loss_init  = loss_curve[0]
    loss_final = loss_curve[-1]
    drop_pct   = (loss_init - loss_final) / (loss_init + 1e-8) * 100
    overfit_ok = drop_pct > 70   # should drop >70% on 2 samples

    # ── Plot ──────────────────────────────────────────────────────────────
    iters = np.arange(1, n_iters + 1)
    ax.plot(iters, loss_curve, color="#f9a825", lw=1.5, label="total_loss")
    ax.axhline(loss_curve[-1], color="#51cf66" if overfit_ok else "#ff6b6b",
               lw=1.2, ls="--",
               label=f"final = {loss_final:.4f}  ({'✅' if overfit_ok else '⚠️'})")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("④ Overfit Sanity (2 pairs, fresh model)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466", labelcolor="white")
    ax.set_yscale("log")

    stats_txt = (f"iter 1:    {loss_init:.4f}\n"
                 f"iter {n_iters}: {loss_final:.4f}\n"
                 f"drop:      {drop_pct:.1f}%\n"
                 f"→ {'✅ Model CAN overfit' if overfit_ok else '⚠️  Loss not decreasing enough'}")
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=8, va="top", ha="right", color="white", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                      edgecolor="#444466", alpha=0.9))

    result = {"loss_init": loss_init, "loss_final": loss_final,
              "drop_pct": drop_pct, "overfit_ok": overfit_ok}

    print("\n④ Overfit Sanity Test  (2 pairs, 200 iters)")
    print(f"   loss start : {loss_init:.4f}")
    print(f"   loss final : {loss_final:.4f}")
    print(f"   drop       : {drop_pct:.1f}%")
    if overfit_ok:
        print("   ✅ Model CAN overfit — architecture and loss are healthy")
    else:
        print("   ⚠️  Loss did not drop >70% — possible issues:")
        print("      - learning rate too low  → try lr=1e-2")
        print("      - loss scale mismatch    → check BCE / MSE balance")
        print("      - data/label issue       → inspect proxy labels")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _style_axes(axes):
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")


def main(args):
    device = torch.device(args.device if torch.cuda.is_available()
                          or args.device == "cpu" else "cpu")

    # ── Load config / model ────────────────────────────────────────────────
    cfg = ReasonerConfig(proxy_delta_threshold=args.proxy_threshold)

    if args.model and Path(args.model).exists():
        ckpt = torch.load(args.model, weights_only=False, map_location="cpu")
        cfg_dict = ckpt.get("config", {})
        for k, v in cfg_dict.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        log.info(f"Loaded config from {args.model}")
    else:
        log.info("No checkpoint provided — using default config")

    # ── Load samples ───────────────────────────────────────────────────────
    samples = load_samples(
        t1_dir    = Path(args.tokens_T1),
        t2_dir    = Path(args.tokens_T2),
        match_dir = Path(args.matches),
        n         = args.n_samples,
        seed      = 42,
    )
    if len(samples) < 2:
        log.error("Need at least 2 samples. Exiting.")
        return

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.patch.set_facecolor("#0f0f1a")
    _style_axes(axes.flatten())

    print("\n" + "─" * 60)
    print("  STAGE 4A — DIAGNOSTIC REPORT")
    print("─" * 60)
    print(f"  Samples    : {len(samples)}")
    print(f"  Device     : {device}")
    print(f"  proxy_thr  : {cfg.proxy_delta_threshold}")

    # ── Run checks ─────────────────────────────────────────────────────────
    r1 = check_delta_separation(samples, cfg, axes[0, 0])
    r2 = check_label_distribution(samples, cfg, axes[0, 1])
    r3 = check_token_count(samples, axes[1, 0])
    r4 = check_overfit_sanity(samples, cfg, axes[1, 1], device,
                              n_iters=args.overfit_iters, lr=args.overfit_lr)

    # ── Overall verdict ────────────────────────────────────────────────────
    all_ok = r1["sep_ok"] and r2["balance_ok"] and r4["overfit_ok"]
    print("\n" + "─" * 60)
    print("  OVERALL VERDICT")
    print("─" * 60)
    checks = [
        ("Delta separation",   r1["sep_ok"]),
        ("Label balance",      r2["balance_ok"]),
        ("Token count safe",   not r3["mem_warn"]),
        ("Overfit sanity",     r4["overfit_ok"]),
    ]
    for name, ok in checks:
        print(f"  {'✅' if ok else '⚠️ '} {name}")

    if all_ok:
        print("\n  ✅ ALL CHECKS PASS — safe to run full training")
    else:
        print("\n  ⚠️  Some checks failed — review warnings above before full training")
    print("─" * 60)

    # ── Save figure ────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.suptitle("Stage 4A — Diagnostic Report",
                 color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Plot saved → {out_path}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 4A diagnostics")
    p.add_argument("--tokens_T1",  default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2",  default="SECOND/tokens_T2")
    p.add_argument("--matches",    default="SECOND/matches")
    p.add_argument("--model",      default=None,
                   help="Path to checkpoint (optional — for config)")
    p.add_argument("--out",        default="SECOND/diagnostics/stage4_diag.png")
    p.add_argument("--n_samples",  type=int, default=200,
                   help="Max samples to load (None = all)")
    p.add_argument("--proxy_threshold", type=float, default=9.56)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--overfit_iters",   type=int,   default=200)
    p.add_argument("--overfit_lr",      type=float, default=3e-3)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
