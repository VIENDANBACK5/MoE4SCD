"""
matching_diagnostics.py
=======================
Diagnostics for Stage 3 token matching.

Panels:
  ① Cosine score distribution histogram
  ② Centroid distance histogram
  ③ Match count per pair
  ④ Δtoken norm  ||T2 − T1||  histogram
  ⑤ Cosine vs centroid distance scatter

Usage:
    python matching_diagnostics.py \
        --matches SECOND/matches \
        --tokens_T1 SECOND/tokens_T1 \
        --tokens_T2 SECOND/tokens_T2 \
        --out SECOND/diagnostics/matching_diagnostics.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_all_matches(matches_dir: Path):
    """Load all *_matches.pt files and aggregate pair data."""
    cosine_scores = []
    centroid_dists = []
    pair_counts = []

    files = sorted(matches_dir.glob("*_matches.pt"))
    print(f"Found {len(files)} match files in {matches_dir}")

    for pt_path in files:
        try:
            res = torch.load(pt_path, weights_only=False)
            pairs = res.get("pairs", [])
            pair_counts.append(len(pairs))

            for p in pairs:
                cosine_scores.append(float(p[2]))

        except Exception as e:
            print(f"  Warning: could not load {pt_path.name}: {e}")

    return cosine_scores, centroid_dists, pair_counts


def compute_per_match_data(matches_dir: Path, t1_dir: Path, t2_dir: Path):
    """
    Single pass over all match files.
    Returns parallel arrays:
      centroid_dists  — || centroid_T1[i] - centroid_T2[j] ||
      delta_norms     — || token_T2[j]   - token_T1[i]    ||  (Δtoken norm)
      cosine_scores2  — match scores (aligned with above)
    """
    centroid_dists = []
    delta_norms    = []
    cosine_scores2 = []

    files = sorted(matches_dir.glob("*_matches.pt"))
    print(f"Computing per-match data from {len(files)} files...")

    for pt_path in files:
        stem = pt_path.stem.replace("_matches", "")
        t1_path = t1_dir / f"{stem}.pt"
        t2_path = t2_dir / f"{stem}.pt"

        if not t1_path.exists() or not t2_path.exists():
            continue

        try:
            res     = torch.load(pt_path, weights_only=False)
            data_t1 = torch.load(t1_path, weights_only=True)
            data_t2 = torch.load(t2_path, weights_only=True)

            cen1 = data_t1["centroids"].float()   # [N1, 2]
            cen2 = data_t2["centroids"].float()   # [N2, 2]
            tok1 = data_t1["tokens"].float()       # [N1, D]
            tok2 = data_t2["tokens"].float()       # [N2, D]

            for p in res.get("pairs", []):
                i, j, sc = int(p[0]), int(p[1]), float(p[2])
                if i < len(cen1) and j < len(cen2):
                    # centroid distance
                    d = float((cen1[i] - cen2[j]).norm())
                    centroid_dists.append(d)
                    # Δtoken norm  ||T2 - T1||
                    dn = float((tok2[j] - tok1[i]).norm())
                    delta_norms.append(dn)
                    # score
                    cosine_scores2.append(sc)

        except Exception as e:
            print(f"  Warning: {stem}: {e}")

    return centroid_dists, delta_norms, cosine_scores2


def plot_diagnostics(
    cosine_scores, centroid_dists, pair_counts,
    delta_norms, cosine_scores2,
    out_path: Path,
):
    """Generate 5-panel diagnostic figure (2-row grid)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()           # index 0-5; axes[5] will be hidden
    fig.patch.set_facecolor("#0f0f1a")
    for ax in axes:
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444466")

    # ── Panel 1: Cosine score distribution ─────────────────────────────────
    ax = axes[0]
    if cosine_scores:
        cos_arr = np.array(cosine_scores)
        ax.hist(cos_arr, bins=50, color="#7c4dff", edgecolor="#9965ff",
                alpha=0.85, linewidth=0.5)

        # Bimodal reference lines
        ax.axvline(0.3, color="#ff6b6b", lw=1.5, ls="--", alpha=0.8,
                   label="Change boundary (~0.3)")
        ax.axvline(0.6, color="#51cf66", lw=1.5, ls="--", alpha=0.8,
                   label="Stable boundary (~0.6)")

        mean_cos = cos_arr.mean()
        ax.axvline(mean_cos, color="white", lw=1.5, ls=":",
                   label=f"Mean = {mean_cos:.3f}")

        ax.set_xlabel("Cosine Score (match quality)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("① Cosine Score Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white")

        # Stats annotation
        pct_low  = (cos_arr < 0.3).mean() * 100
        pct_mid  = ((cos_arr >= 0.3) & (cos_arr < 0.6)).mean() * 100
        pct_high = (cos_arr >= 0.6).mean() * 100
        stats_txt = (f"n={len(cos_arr)}\n"
                     f"<0.3 (change?):  {pct_low:.1f}%\n"
                     f"0.3–0.6 (ambig): {pct_mid:.1f}%\n"
                     f"≥0.6 (stable?):  {pct_high:.1f}%")
        ax.text(0.97, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color="white", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                          edgecolor="#444466", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color="white", transform=ax.transAxes)

    # ── Panel 2: Centroid distance distribution ─────────────────────────────
    ax = axes[1]
    if centroid_dists:
        dist_arr = np.array(centroid_dists)
        ax.hist(dist_arr, bins=50, color="#00b4d8", edgecolor="#48cae4",
                alpha=0.85, linewidth=0.5)

        ax.axvline(0.15, color="#ff6b6b", lw=1.5, ls="--", alpha=0.8,
                   label="Local threshold (0.15)")
        ax.axvline(dist_arr.mean(), color="white", lw=1.5, ls=":",
                   label=f"Mean = {dist_arr.mean():.3f}")

        ax.set_xlabel("Centroid Distance (normalized)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("② Match Centroid Distance", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white")

        pct_local = (dist_arr < 0.15).mean() * 100
        stats_txt = (f"n={len(dist_arr)}\n"
                     f"<0.15 (local):    {pct_local:.1f}%\n"
                     f"≥0.15 (shift?):   {100-pct_local:.1f}%\n"
                     f"max dist: {dist_arr.max():.3f}")
        ax.text(0.97, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color="white", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                          edgecolor="#444466", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No centroid data\n(check token dirs)",
                ha="center", va="center", color="white",
                transform=ax.transAxes)

    # ── Panel 3: Matches per pair ──────────────────────────────────────────
    ax = axes[2]
    if pair_counts:
        cnt_arr = np.array(pair_counts)
        ax.hist(cnt_arr, bins=max(5, len(cnt_arr)//2), color="#f9a825",
                edgecolor="#ffcc02", alpha=0.85, linewidth=0.5)
        ax.axvline(cnt_arr.mean(), color="white", lw=1.5, ls=":",
                   label=f"Mean = {cnt_arr.mean():.1f}")

        ax.set_xlabel("Matches per Image Pair", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("③ Match Count Distribution", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white")

        stats_txt = (f"n_pairs={len(cnt_arr)}\n"
                     f"min:  {cnt_arr.min()}\n"
                     f"mean: {cnt_arr.mean():.1f}\n"
                     f"max:  {cnt_arr.max()}")
        ax.text(0.97, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color="white", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                          edgecolor="#444466", alpha=0.9))

    # ── Panel 4: Δtoken norm  ||T2 − T1||  ────────────────────────────────
    ax = axes[3]
    if delta_norms:
        dn_arr = np.array(delta_norms)
        ax.hist(dn_arr, bins=60, color="#e91e8c", edgecolor="#ff4db8",
                alpha=0.85, linewidth=0.5)

        mean_dn = dn_arr.mean()
        ax.axvline(mean_dn, color="white", lw=1.5, ls=":",
                   label=f"Mean = {mean_dn:.2f}")
        # Heuristic boundary: tokens with large norm are likely changed
        thr = mean_dn + dn_arr.std()
        ax.axvline(thr, color="#ff6b6b", lw=1.5, ls="--", alpha=0.8,
                   label=f"Mean+σ = {thr:.2f}  (changed?)")

        ax.set_xlabel("‖ token_T2 − token_T1 ‖  (Δ embedding norm)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("④ Δ Token Norm  ‖T2 − T1‖", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white")

        pct_large = (dn_arr > thr).mean() * 100
        stats_txt = (f"n={len(dn_arr)}\n"
                     f"mean: {mean_dn:.3f}\n"
                     f"std:  {dn_arr.std():.3f}\n"
                     f"> mean+σ: {pct_large:.1f}%  (likely changed)")
        ax.text(0.97, 0.97, stats_txt,
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                color="white", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                          edgecolor="#444466", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No Δnorm data", ha="center", va="center",
                color="white", transform=ax.transAxes)

    # ── Panel 5: Cosine vs centroid distance scatter ───────────────────────
    ax = axes[4]
    if cosine_scores2 and centroid_dists:
        cos_arr2 = np.array(cosine_scores2)
        dist_arr = np.array(centroid_dists)

        # Subsample for scatter (keep up to 30k points)
        MAX_PTS = 30_000
        if len(cos_arr2) > MAX_PTS:
            idx = np.random.default_rng(42).choice(len(cos_arr2), MAX_PTS, replace=False)
            cos_arr2 = cos_arr2[idx]
            dist_arr = dist_arr[idx]

        # Colour by cosine: low=changed (red), high=stable (blue)
        sc = ax.scatter(
            dist_arr, cos_arr2,
            c=cos_arr2, cmap="RdYlGn",
            s=3, alpha=0.35, linewidths=0,
            vmin=0.2, vmax=1.0,
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label("Cosine score", color="white", fontsize=8)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white", fontsize=7)

        # Reference lines
        ax.axhline(0.3, color="#ff6b6b", lw=1.2, ls="--", alpha=0.7,
                   label="cos = 0.3 (change)")
        ax.axhline(0.6, color="#51cf66", lw=1.2, ls="--", alpha=0.7,
                   label="cos = 0.6 (stable)")
        ax.axvline(0.15, color="#ffd600", lw=1.2, ls=":", alpha=0.7,
                   label="dist = 0.15 (local)")

        ax.set_xlabel("Centroid Distance (normalized)", fontsize=11)
        ax.set_ylabel("Cosine Score", fontsize=11)
        ax.set_title("⑤ Cosine vs Centroid Distance", fontsize=12, fontweight="bold")
        ax.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white", markerscale=3)
        ax.set_xlim(left=0)
        ax.set_ylim(0.15, 1.02)

        # Correlation
        corr = float(np.corrcoef(dist_arr, cos_arr2)[0, 1])
        ax.text(0.03, 0.04,
                f"Pearson r = {corr:.3f}",
                transform=ax.transAxes, fontsize=9, color="white",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0f0f1a",
                          edgecolor="#444466", alpha=0.9))
    else:
        ax.text(0.5, 0.5, "No scatter data", ha="center", va="center",
                color="white", transform=ax.transAxes)

    # ── Hide unused 6th cell ───────────────────────────────────────────────
    axes[5].set_visible(False)

    plt.suptitle("Stage 3 — Token Matching Diagnostics",
                 color="white", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nDiagnostic plot saved → {out_path}")


def print_summary(cosine_scores, centroid_dists, pair_counts, delta_norms, cosine_scores2):
    """Print text summary to stdout."""
    print("\n" + "─" * 60)
    print("  DIAGNOSTIC SUMMARY")
    print("─" * 60)

    if cosine_scores:
        cos = np.array(cosine_scores)
        print(f"\n① Cosine Score Distribution  (n={len(cos)})")
        print(f"   mean ± std : {cos.mean():.4f} ± {cos.std():.4f}")
        print(f"   min / max  : {cos.min():.4f} / {cos.max():.4f}")
        print(f"   < 0.3  (change?) : {(cos < 0.3).mean()*100:5.1f}%")
        print(f"   0.3–0.6 (ambig)  : {((cos>=0.3)&(cos<0.6)).mean()*100:5.1f}%")
        print(f"   ≥ 0.6  (stable?) : {(cos >= 0.6).mean()*100:5.1f}%")
        bimodal_ok = (cos < 0.3).mean() > 0.05 and (cos >= 0.6).mean() > 0.05
        verdict = "✅ Bimodal signal detected" if bimodal_ok else "⚠️  Single-mode (expected for SCD)"
        print(f"   → {verdict}")

    if centroid_dists:
        d = np.array(centroid_dists)
        print(f"\n② Centroid Distance of Matches  (n={len(d)})")
        print(f"   mean ± std : {d.mean():.4f} ± {d.std():.4f}")
        print(f"   < 0.15 (local) : {(d < 0.15).mean()*100:5.1f}%")
        local_ok = (d < 0.15).mean() > 0.5
        verdict = "✅ Local matches dominant" if local_ok else "⚠️  Many long-range matches"
        print(f"   → {verdict}")

    if pair_counts:
        c = np.array(pair_counts)
        print(f"\n③ Match count per pair")
        print(f"   mean: {c.mean():.1f}  min: {c.min()}  max: {c.max()}")

    if delta_norms:
        dn = np.array(delta_norms)
        thr = dn.mean() + dn.std()
        pct_changed = (dn > thr).mean() * 100
        print(f"\n④ Δ Token Norm  ‖T2 − T1‖  (n={len(dn)})")
        print(f"   mean ± std : {dn.mean():.4f} ± {dn.std():.4f}")
        print(f"   min / max  : {dn.min():.4f} / {dn.max():.4f}")
        print(f"   > mean+σ ({thr:.2f}) : {pct_changed:.1f}%  ← likely changed tokens")

    if cosine_scores2 and centroid_dists:
        cos2 = np.array(cosine_scores2)
        dist = np.array(centroid_dists)
        corr = float(np.corrcoef(dist, cos2)[0, 1])
        print(f"\n⑤ Cosine vs Centroid Distance")
        print(f"   Pearson r = {corr:.4f}")
        verdict = "✅ Negative corr: local = high cosine" if corr < -0.05 else "⚠️  No clear spatial-semantic correlation"
        print(f"   → {verdict}")

    print("\n" + "─" * 60)


def main():
    p = argparse.ArgumentParser(description="Stage 3 pre-run diagnostics")
    p.add_argument("--matches",   default="SECOND/matches",
                   help="Directory containing *_matches.pt files")
    p.add_argument("--tokens_T1", default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2", default="SECOND/tokens_T2")
    p.add_argument("--out", default="SECOND/diagnostics/matching_diagnostics.png",
                   help="Output path for the diagnostic plot")
    args = p.parse_args()

    matches_dir = Path(args.matches)
    t1_dir      = Path(args.tokens_T1)
    t2_dir      = Path(args.tokens_T2)
    out_path    = Path(args.out)

    print("Loading match files...")
    cosine_scores, _, pair_counts = load_all_matches(matches_dir)

    print("Computing per-match spatial + embedding data...")
    centroid_dists, delta_norms, cosine_scores2 = compute_per_match_data(
        matches_dir, t1_dir, t2_dir
    )

    print_summary(cosine_scores, centroid_dists, pair_counts, delta_norms, cosine_scores2)
    plot_diagnostics(
        cosine_scores, centroid_dists, pair_counts,
        delta_norms, cosine_scores2,
        out_path,
    )

    # List existing visualizations
    diag_match_dir = Path("SECOND/diagnostics/matches")
    if diag_match_dir.exists():
        imgs = sorted(diag_match_dir.glob("*.png"))
        print(f"\nFound {len(imgs)} diagnostic images in {diag_match_dir}:")
        for img in imgs:
            print(f"  {img.name}")


if __name__ == "__main__":
    main()
