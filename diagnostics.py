"""
Stage 2.5: Dataset Diagnostics and Validation
==============================================
Validates and analyzes the token dataset produced by Stage 2.

Outputs (saved to --output_dir, default: SECOND/diagnostics/):
    token_count_histogram.png
    token_norm_distribution.png
    area_coverage_histogram.png
    token_outliers.txt
    low_coverage_images.txt
    temporal_inconsistency.txt
    dataset_report.json
    diagnostics_visualizations/sample_XXXX.png  (20 random samples)

Usage:
    python diagnostics.py --dataset_root /path/to/SECOND
    python diagnostics.py --dataset_root /path/to/SECOND --splits train test
"""

import argparse
import json
import logging
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Helpers
# ============================================================================

def load_token_file(path: Path):
    """Load .pt token file. Returns dict or None on error."""
    try:
        d = torch.load(path, weights_only=True)
        assert "tokens" in d and "centroids" in d and "areas" in d
        return d
    except Exception as e:
        log.warning(f"Cannot load {path.name}: {e}")
        return None


def stats_dict(values: list, name: str = "") -> dict:
    """Compute basic stats from a list of numbers."""
    if not values:
        return {}
    a = np.array(values, dtype=np.float64)
    return {
        "count": int(len(a)),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "std": float(a.std()),
        "p5": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
    }


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {path.name}")


# ============================================================================
# 1. Collect per-image statistics
# ============================================================================

def collect_stats(t1_dir: Path, t2_dir: Path, split_label: str):
    """
    Iterate all T1 files, load T1+T2 pair, collect:
      - token counts N1, N2
      - all token L2 norms
      - total coverage (sum of areas)
      - temporal difference ratio
    Returns a dict of lists.
    """
    files = sorted(t1_dir.glob("*.pt"))
    if not files:
        log.warning(f"No .pt files found in {t1_dir}")
        return None

    records = {
        "stems":            [],
        "n1":               [],   # T1 token counts
        "n2":               [],   # T2 token counts
        "norms_t1":         [],   # flat list of all T1 token norms
        "norms_t2":         [],   # flat list of all T2 token norms
        "coverage_t1":      [],   # sum(areas) per image
        "coverage_t2":      [],
        "diff_ratio":       [],   # |N1-N2|/max(N1,N2)
        "corrupt":          [],
    }

    for f in tqdm(files, desc=f"Collecting stats ({split_label})", unit="file"):
        stem = f.stem
        t2_path = t2_dir / f.name

        d1 = load_token_file(f)
        d2 = load_token_file(t2_path) if t2_path.exists() else None

        if d1 is None or d2 is None:
            records["corrupt"].append(stem)
            continue

        n1 = d1["tokens"].shape[0]
        n2 = d2["tokens"].shape[0]

        records["stems"].append(stem)
        records["n1"].append(n1)
        records["n2"].append(n2)

        # L2 norms — compute on float32
        norms1 = d1["tokens"].float().norm(dim=1).tolist()
        norms2 = d2["tokens"].float().norm(dim=1).tolist()
        records["norms_t1"].extend(norms1)
        records["norms_t2"].extend(norms2)

        cov1 = float(d1["areas"].sum().item())
        cov2 = float(d2["areas"].sum().item())
        records["coverage_t1"].append(cov1)
        records["coverage_t2"].append(cov2)

        diff_ratio = abs(n1 - n2) / max(n1, n2, 1)
        records["diff_ratio"].append(diff_ratio)

    return records


# ============================================================================
# 2. Histograms
# ============================================================================

def plot_token_count_histogram(n1: list, n2: list, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in zip(axes, [n1, n2], ["T1 (before change)", "T2 (after change)"]):
        ax.hist(data, bins=50, color="steelblue", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(data):.1f}")
        ax.axvline(np.median(data), color="orange", linestyle="--", linewidth=1.5, label=f"median={np.median(data):.1f}")
        ax.set_title(f"Token Count per Image — {label}")
        ax.set_xlabel("# tokens")
        ax.set_ylabel("# images")
        ax.legend(fontsize=8)
    savefig(fig, out_path)


def plot_norm_distribution(norms_t1: list, norms_t2: list, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in zip(axes, [norms_t1, norms_t2], ["T1", "T2"]):
        ax.hist(data, bins=80, color="coral", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(data):.2f}")
        ax.set_title(f"Token L2 Norm Distribution — {label}")
        ax.set_xlabel("L2 norm")
        ax.set_ylabel("# tokens")
        ax.legend(fontsize=8)
    savefig(fig, out_path)


def plot_coverage_histogram(cov_t1: list, cov_t2: list, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, label in zip(axes, [cov_t1, cov_t2], ["T1", "T2"]):
        ax.hist(data, bins=50, color="mediumseagreen", alpha=0.8, edgecolor="white")
        ax.axvline(np.mean(data), color="red", linestyle="--", linewidth=1.5, label=f"mean={np.mean(data):.3f}")
        ax.axvline(0.4, color="black", linestyle=":", linewidth=1.5, label="threshold=0.4")
        ax.set_title(f"Image Coverage (sum of mask areas) — {label}")
        ax.set_xlabel("coverage fraction")
        ax.set_ylabel("# images")
        ax.legend(fontsize=8)
    savefig(fig, out_path)


# ============================================================================
# 3. Issue detection
# ============================================================================

def find_outliers(stems: list, n1: list, n2: list, high_thresh=500, low_thresh=5):
    """Return (high_count_list, low_count_list)."""
    high = [(s, n, "T1") for s, n in zip(stems, n1) if n > high_thresh] + \
           [(s, n, "T2") for s, n in zip(stems, n2) if n > high_thresh]
    low  = [(s, n, "T1") for s, n in zip(stems, n1) if n < low_thresh] + \
           [(s, n, "T2") for s, n in zip(stems, n2) if n < low_thresh]
    return high, low


def find_low_coverage(stems: list, cov_t1: list, cov_t2: list, threshold=0.4):
    issues = []
    for s, c1, c2 in zip(stems, cov_t1, cov_t2):
        if c1 < threshold:
            issues.append((s, "T1", round(c1, 4)))
        if c2 < threshold:
            issues.append((s, "T2", round(c2, 4)))
    return issues


def find_temporal_inconsistency(stems: list, diff_ratio: list, threshold=0.5):
    return [(s, round(r, 4)) for s, r in zip(stems, diff_ratio) if r > threshold]


# ============================================================================
# 4. Visualizations — centroids over image
# ============================================================================

def visualize_samples(
    t1_dir: Path, t2_dir: Path,
    im1_dir: Path, im2_dir: Path,
    out_dir: Path, n_samples: int = 20, seed: int = 42
):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(t1_dir.glob("*.pt"))
    if not files:
        return

    random.seed(seed)
    samples = random.sample(files, min(n_samples, len(files)))

    for i, f in enumerate(tqdm(samples, desc="Visualizing samples", unit="img")):
        stem = f.stem

        d1 = load_token_file(f)
        d2 = load_token_file(t2_dir / f.name)
        if d1 is None or d2 is None:
            continue

        # Load images (try .png and .jpg)
        def load_img(d, stem):
            for ext in (".png", ".jpg", ".tif"):
                p = d / f"{stem}{ext}"
                if p.exists():
                    from PIL import Image
                    return np.array(Image.open(p).convert("RGB"))
            return np.zeros((512, 512, 3), dtype=np.uint8)

        img1 = load_img(im1_dir, stem)
        img2 = load_img(im2_dir, stem)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{stem}  |  T1 tokens={d1['tokens'].shape[0]}  T2 tokens={d2['tokens'].shape[0]}", fontsize=11)

        for ax, img, data, hw, label in [
            (axes[0], img1, d1, (h1, w1), "T1 (before)"),
            (axes[1], img2, d2, (h2, w2), "T2 (after)"),
        ]:
            ax.imshow(img)
            cx = data["centroids"][:, 0].numpy() * (hw[1] - 1)
            cy = data["centroids"][:, 1].numpy() * (hw[0] - 1)
            areas = data["areas"].numpy()
            # Scale scatter size by area (min 10, max 200)
            sizes = np.clip(areas * 5000, 10, 200)
            sc = ax.scatter(cx, cy, s=sizes, c=areas, cmap="YlOrRd", alpha=0.75,
                            edgecolors="white", linewidths=0.5)
            ax.set_title(f"{label}  ({data['tokens'].shape[0]} tokens)", fontsize=10)
            ax.axis("off")
            plt.colorbar(sc, ax=ax, shrink=0.6, label="area")

        savefig(fig, out_dir / f"sample_{stem}.png")


# ============================================================================
# 5. Main
# ============================================================================

def run_diagnostics(args):
    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.output_dir) if args.output_dir else dataset_root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {out_dir}")

    all_report = {}

    for split in args.splits:
        log.info(f"\n{'='*60}")
        log.info(f"Processing split: {split}")

        suffix = "_test" if split == "test" else ""
        t1_dir = dataset_root / f"tokens_T1{suffix}"
        t2_dir = dataset_root / f"tokens_T2{suffix}"
        im1_dir = (dataset_root / "test" / "im1") if split == "test" else (dataset_root / "im1")
        im2_dir = (dataset_root / "test" / "im2") if split == "test" else (dataset_root / "im2")

        if not t1_dir.is_dir():
            log.warning(f"tokens_T1{suffix}/ not found — skipping {split} split")
            continue

        # Per-split output subdirectory
        split_out = out_dir / split
        split_out.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------------------------
        # Collect statistics
        # -------------------------------------------------------------------
        rec = collect_stats(t1_dir, t2_dir, split)
        if rec is None or not rec["stems"]:
            log.warning(f"No valid pairs found for {split}")
            continue

        n = len(rec["stems"])
        log.info(f"Valid pairs: {n}  |  Corrupt: {len(rec['corrupt'])}")

        # -------------------------------------------------------------------
        # 1. Token count histogram
        # -------------------------------------------------------------------
        plot_token_count_histogram(
            rec["n1"], rec["n2"],
            split_out / "token_count_histogram.png"
        )

        # -------------------------------------------------------------------
        # 2. Norm distribution
        # -------------------------------------------------------------------
        plot_norm_distribution(
            rec["norms_t1"], rec["norms_t2"],
            split_out / "token_norm_distribution.png"
        )
        mean_norm = float(np.mean(rec["norms_t1"] + rec["norms_t2"]))
        if mean_norm < 0.5:
            log.warning(f"[{split}] Mean token norm = {mean_norm:.3f} — embeddings may be near-zero!")
        if mean_norm > 100:
            log.warning(f"[{split}] Mean token norm = {mean_norm:.3f} — embeddings may be exploding!")

        # -------------------------------------------------------------------
        # 3. Coverage histogram
        # -------------------------------------------------------------------
        plot_coverage_histogram(
            rec["coverage_t1"], rec["coverage_t2"],
            split_out / "area_coverage_histogram.png"
        )

        # -------------------------------------------------------------------
        # 4. Outlier detection
        # -------------------------------------------------------------------
        high_out, low_out = find_outliers(
            rec["stems"], rec["n1"], rec["n2"],
            high_thresh=args.high_token_thresh,
            low_thresh=args.low_token_thresh,
        )
        with open(split_out / "token_outliers.txt", "w") as fp:
            fp.write(f"# Token count outliers (>{args.high_token_thresh} or <{args.low_token_thresh})\n")
            fp.write(f"# High count ({len(high_out)}):\n")
            for s, n_tok, side in high_out:
                fp.write(f"  {s}  {side}  tokens={n_tok}\n")
            fp.write(f"# Low count ({len(low_out)}):\n")
            for s, n_tok, side in low_out:
                fp.write(f"  {s}  {side}  tokens={n_tok}\n")
        log.info(f"[{split}] Outliers — high: {len(high_out)}  low: {len(low_out)}")

        # -------------------------------------------------------------------
        # 5. Low coverage detection
        # -------------------------------------------------------------------
        low_cov = find_low_coverage(
            rec["stems"], rec["coverage_t1"], rec["coverage_t2"],
            threshold=args.coverage_thresh
        )
        with open(split_out / "low_coverage_images.txt", "w") as fp:
            fp.write(f"# Images with coverage < {args.coverage_thresh} ({len(low_cov)} cases)\n")
            for s, side, cov in low_cov:
                fp.write(f"  {s}  {side}  coverage={cov}\n")
        log.info(f"[{split}] Low coverage images: {len(low_cov)}")

        # -------------------------------------------------------------------
        # 6. Temporal inconsistency
        # -------------------------------------------------------------------
        temporal = find_temporal_inconsistency(
            rec["stems"], rec["diff_ratio"],
            threshold=args.diff_ratio_thresh
        )
        with open(split_out / "temporal_inconsistency.txt", "w") as fp:
            fp.write(f"# Pairs with |N1-N2|/max(N1,N2) > {args.diff_ratio_thresh} ({len(temporal)} cases)\n")
            for s, r in temporal:
                fp.write(f"  {s}  diff_ratio={r}\n")
        log.info(f"[{split}] Temporal inconsistency (>{args.diff_ratio_thresh}): {len(temporal)}")

        # -------------------------------------------------------------------
        # 7. Visualizations
        # -------------------------------------------------------------------
        viz_dir = split_out / "diagnostics_visualizations"
        visualize_samples(t1_dir, t2_dir, im1_dir, im2_dir, viz_dir,
                          n_samples=args.n_viz_samples)

        # -------------------------------------------------------------------
        # 8. Build split report
        # -------------------------------------------------------------------
        split_report = {
            "split": split,
            "num_images": n,
            "corrupt_files": len(rec["corrupt"]),
            "token_count_stats": {
                "T1": stats_dict(rec["n1"]),
                "T2": stats_dict(rec["n2"]),
            },
            "token_norm_stats": {
                "T1": stats_dict(rec["norms_t1"]),
                "T2": stats_dict(rec["norms_t2"]),
                "combined_mean": round(mean_norm, 4),
            },
            "coverage_stats": {
                "T1": stats_dict(rec["coverage_t1"]),
                "T2": stats_dict(rec["coverage_t2"]),
                "low_coverage_count": len(low_cov),
                "low_coverage_threshold": args.coverage_thresh,
            },
            "temporal_consistency": {
                "diff_ratio_stats": stats_dict(rec["diff_ratio"]),
                "inconsistent_pairs_count": len(temporal),
                "inconsistency_threshold": args.diff_ratio_thresh,
            },
            "outliers": {
                "high_token_count": len(high_out),
                "low_token_count": len(low_out),
                "high_threshold": args.high_token_thresh,
                "low_threshold": args.low_token_thresh,
            },
        }
        all_report[split] = split_report

        # Print quick summary
        log.info(f"\n[{split}] Token counts T1: mean={np.mean(rec['n1']):.1f} "
                 f"min={min(rec['n1'])} max={max(rec['n1'])}")
        log.info(f"[{split}] Token counts T2: mean={np.mean(rec['n2']):.1f} "
                 f"min={min(rec['n2'])} max={max(rec['n2'])}")
        log.info(f"[{split}] Norm: mean={mean_norm:.3f}")
        log.info(f"[{split}] Coverage T1: mean={np.mean(rec['coverage_t1']):.3f}"
                 f"  T2: mean={np.mean(rec['coverage_t2']):.3f}")

    # -----------------------------------------------------------------------
    # Save combined report
    # -----------------------------------------------------------------------
    report_path = out_dir / "dataset_report.json"
    with open(report_path, "w") as fp:
        json.dump(all_report, fp, indent=2)
    log.info(f"\nDataset report saved to: {report_path}")
    log.info("Diagnostics complete.")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 2.5: Dataset diagnostics and validation"
    )
    parser.add_argument("--dataset_root", type=str,
                        default="/home/chung/RS/phase1/SECOND")
    parser.add_argument("--splits", nargs="+", default=["train"],
                        choices=["train", "test"],
                        help="Which splits to diagnose")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: dataset_root/diagnostics)")
    parser.add_argument("--high_token_thresh", type=int, default=500,
                        help="Flag images with more than N tokens")
    parser.add_argument("--low_token_thresh", type=int, default=5,
                        help="Flag images with fewer than N tokens")
    parser.add_argument("--coverage_thresh", type=float, default=0.40,
                        help="Flag images where sum(areas) < threshold")
    parser.add_argument("--diff_ratio_thresh", type=float, default=0.50,
                        help="Flag pairs where |N1-N2|/max > threshold")
    parser.add_argument("--n_viz_samples", type=int, default=20,
                        help="Number of visualization samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_diagnostics(args)
