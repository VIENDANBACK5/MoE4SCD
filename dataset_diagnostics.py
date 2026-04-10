"""
Stage 2.5: Full Dataset Diagnostics and Validation
====================================================
Validates SAM2 embeddings (Stage 1) and region tokens (Stage 2) before
training downstream change detection models.

Sections:
  1. Token count statistics + histogram
  2. Token L2 norm distribution
  3. Embedding variance (Stage 1 quality)
  4. Token diversity (intra-image cosine similarity)
  5. Spatial coverage (mask area)
  6. Temporal consistency (T1 vs T2 token count ratio)
  7. Temporal feature stability (T1↔T2 nearest-neighbor similarity)
  8. Feature clustering (K-Means k=10, pure NumPy)
  9. Spatial visualizations (centroids overlay)
  10. JSON dataset report

Usage:
    python dataset_diagnostics.py --dataset_root /path/to/SECOND
    python dataset_diagnostics.py --dataset_root /path/to/SECOND --splits train test
"""

import argparse
import json
import logging
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Sampling sizes — keep memory bounded
N_NORM_SAMPLE    = 50_000   # max tokens to collect for norm/diversity stats
N_EMB_SAMPLE     = 500      # max embedding files to sample for variance
N_SIM_PAIRS      = 300      # random pairs for temporal similarity
N_CLUSTER_TOKENS = 20_000   # tokens fed to k-means
N_VIZ_SAMPLES    = 20       # visualization images


# ============================================================================
# I/O helpers
# ============================================================================

def load_token(path: Path):
    try:
        d = torch.load(path, weights_only=True)
        assert "tokens" in d and "centroids" in d and "areas" in d
        assert d["tokens"].ndim == 2 and d["tokens"].shape[1] == 256
        return d
    except Exception as e:
        log.debug(f"Skip {path.name}: {e}")
        return None


def load_embedding(path: Path):
    try:
        t = torch.load(path, weights_only=True)
        if t.dim() == 4:
            t = t.squeeze(0)
        assert t.shape == (256, 64, 64)
        return t
    except Exception as e:
        log.debug(f"Skip {path.name}: {e}")
        return None


def stats(values, label=""):
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(a)),
        "min": round(float(a.min()), 4),
        "max": round(float(a.max()), 4),
        "mean": round(float(a.mean()), 4),
        "median": round(float(np.median(a)), 4),
        "std": round(float(a.std()), 4),
        "p5": round(float(np.percentile(a, 5)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
    }


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {path.name}")


# ============================================================================
# Data collection
# ============================================================================

def collect_all(t1_dir: Path, t2_dir: Path, emb_t1_dir: Path, label: str):
    """
    Single pass over all T1 files. Collects everything needed for all
    sections to avoid re-reading files repeatedly.
    Returns a namespace-like dict.
    """
    files = sorted(t1_dir.glob("*.pt"))
    if not files:
        raise RuntimeError(f"No .pt files in {t1_dir}")

    rec = dict(
        stems=[], corrupt=[],
        n1=[], n2=[],
        cov_t1=[], cov_t2=[],
        diff_ratio=[],
        # sampled tensors (bounded memory)
        norms_t1=[], norms_t2=[],
        tokens_pool=[],           # flat pool for diversity & clustering
        t1t2_pairs=[],            # (tok_t1, tok_t2) for temporal sim
    )

    rng = random.Random(42)

    for f in tqdm(files, desc=f"Loading data ({label})", unit="file"):
        stem = f.stem
        t2_path = t2_dir / f.name

        d1 = load_token(f)
        d2 = load_token(t2_path) if t2_path.exists() else None
        if d1 is None or d2 is None:
            rec["corrupt"].append(stem)
            continue

        rec["stems"].append(stem)
        n1, n2 = d1["tokens"].shape[0], d2["tokens"].shape[0]
        rec["n1"].append(n1)
        rec["n2"].append(n2)
        rec["cov_t1"].append(float(d1["areas"].sum()))
        rec["cov_t2"].append(float(d2["areas"].sum()))
        rec["diff_ratio"].append(abs(n1 - n2) / max(n1, n2, 1))

        # Collect norms (sample to bound memory)
        if len(rec["norms_t1"]) < N_NORM_SAMPLE:
            rec["norms_t1"].extend(d1["tokens"].float().norm(dim=1).tolist())
            rec["norms_t2"].extend(d2["tokens"].float().norm(dim=1).tolist())

        # Pool tokens for diversity & clustering
        if len(rec["tokens_pool"]) < N_CLUSTER_TOKENS:
            pick = d1["tokens"].float()
            rec["tokens_pool"].append(pick)

        # Collect T1/T2 pairs for temporal similarity (random subset)
        if len(rec["t1t2_pairs"]) < N_SIM_PAIRS:
            rec["t1t2_pairs"].append((
                d1["tokens"].float(),
                d2["tokens"].float(),
            ))

    rec["tokens_pool"] = torch.cat(rec["tokens_pool"])[:N_CLUSTER_TOKENS]
    return rec


def sample_embedding_variances(emb_dir: Path, n: int = N_EMB_SAMPLE):
    """Sample embedding files and compute per-file spatial variance."""
    files = sorted(emb_dir.glob("*.pt"))
    if not files:
        return []
    chosen = random.sample(files, min(n, len(files)))
    variances = []
    for f in tqdm(chosen, desc="Embedding variance", unit="file", leave=False):
        emb = load_embedding(f)
        if emb is not None:
            variances.append(float(emb.float().var().item()))
    return variances


# ============================================================================
# Section helpers
# ============================================================================

# ── §1 Token count histogram ─────────────────────────────────────────────────
def plot_token_hist(n1, n2, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, data, lbl, col in zip(
        axes, [n1, n2],
        ["T1 (before change)", "T2 (after change)"],
        ["steelblue", "darkorange"]
    ):
        ax.hist(data, bins=60, color=col, alpha=0.85, edgecolor="none")
        ax.axvline(np.mean(data),   color="red",    ls="--", lw=1.5, label=f"mean {np.mean(data):.1f}")
        ax.axvline(np.median(data), color="yellow", ls="--", lw=1.5, label=f"median {np.median(data):.1f}")
        ax.set(title=f"Token Count — {lbl}", xlabel="# tokens", ylabel="# images")
        ax.legend(fontsize=8)
    savefig(fig, out)


# ── §2 Token L2 norm ─────────────────────────────────────────────────────────
def plot_norm_hist(norms_t1, norms_t2, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, data, lbl in zip(axes, [norms_t1, norms_t2], ["T1", "T2"]):
        ax.hist(data, bins=80, color="coral", alpha=0.85, edgecolor="none")
        ax.axvline(np.mean(data), color="red", ls="--", lw=1.5, label=f"mean {np.mean(data):.2f}")
        ax.set(title=f"Token L2 Norm — {lbl}", xlabel="L2 norm", ylabel="# tokens")
        ax.legend(fontsize=8)
    savefig(fig, out)


# ── §3 Embedding variance ─────────────────────────────────────────────────────
def plot_emb_variance(variances, out):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(variances, bins=60, color="mediumslateblue", alpha=0.85, edgecolor="none")
    ax.axvline(np.mean(variances), color="red", ls="--", lw=1.5, label=f"mean {np.mean(variances):.4f}")
    ax.set(title="SAM2 Embedding Spatial Variance per Image",
           xlabel="var(embedding)", ylabel="# images")
    ax.legend(fontsize=8)
    savefig(fig, out)


# ── §4 Token diversity (intra-image cosine similarity) ───────────────────────
def compute_intra_cosine(tokens_pool: torch.Tensor, n_pairs: int = 10_000):
    """Sample random pairs from the token pool and compute cosine similarity."""
    pool = F.normalize(tokens_pool, dim=1)
    n = pool.shape[0]
    idx_a = torch.randint(0, n, (n_pairs,))
    idx_b = torch.randint(0, n, (n_pairs,))
    sims = (pool[idx_a] * pool[idx_b]).sum(dim=1).clamp(-1, 1).tolist()
    return sims


def plot_sim_hist(sims, out, title, xlabel, color):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sims, bins=80, color=color, alpha=0.85, edgecolor="none")
    ax.axvline(np.mean(sims), color="red", ls="--", lw=1.5, label=f"mean {np.mean(sims):.3f}")
    ax.set(title=title, xlabel=xlabel, ylabel="# pairs")
    ax.legend(fontsize=8)
    savefig(fig, out)


# ── §7 Temporal feature stability ────────────────────────────────────────────
def compute_temporal_sim(t1t2_pairs):
    """
    For each (T1_tokens, T2_tokens) pair compute the mean nearest-neighbour
    cosine similarity from T1 → T2.
    """
    sims = []
    for tok1, tok2 in t1t2_pairs:
        n1_norm = F.normalize(tok1, dim=1)   # (N1, 256)
        n2_norm = F.normalize(tok2, dim=1)   # (N2, 256)
        sim_mat = n1_norm @ n2_norm.T        # (N1, N2)
        nn_sim = sim_mat.max(dim=1).values   # nearest neighbour per T1 token
        sims.append(float(nn_sim.mean()))
    return sims


# ── §8 K-Means (pure NumPy, no sklearn) ──────────────────────────────────────
def numpy_kmeans(X: np.ndarray, k: int = 10, max_iter: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=np.int32)
    for iteration in range(max_iter):
        # Assign: (N, k) distances via broadcasting in chunks to save memory
        chunk = 2000
        new_labels = np.empty(len(X), dtype=np.int32)
        for start in range(0, len(X), chunk):
            diff = X[start:start+chunk, None, :] - centers[None, :, :]  # (c, k, 256)
            dists = (diff * diff).sum(axis=2)
            new_labels[start:start+chunk] = dists.argmin(axis=1)
        # Update centers
        new_centers = np.stack([
            X[new_labels == kk].mean(0) if (new_labels == kk).any() else centers[kk]
            for kk in range(k)
        ])
        if np.allclose(new_labels, labels):
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def plot_cluster_distribution(labels, k, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Bar chart of cluster sizes
    counts = np.bincount(labels, minlength=k)
    axes[0].bar(range(k), counts, color="teal", edgecolor="white", alpha=0.85)
    axes[0].set(title="Token Cluster Sizes (K-Means k=10)",
                xlabel="Cluster ID", ylabel="# tokens")
    for i, c in enumerate(counts):
        axes[0].text(i, c + max(counts)*0.01, str(c), ha="center", fontsize=7)

    # PCA 2D projection
    from numpy.linalg import svd
    X_mean = np.zeros(2)  # placeholder computed below
    axes[1].set_title("Token Clusters — PCA 2D Projection")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].text(0.5, 0.5, "PCA projection\n(see below)",
                 ha="center", va="center", transform=axes[1].transAxes, alpha=0.3)
    savefig(fig, out)


def plot_cluster_pca(tokens_pool: np.ndarray, labels: np.ndarray, k: int, out: Path):
    """PCA 2D + cluster scatter."""
    # Center
    mu = tokens_pool.mean(0)
    X = (tokens_pool - mu).astype(np.float32)
    # Randomized SVD via power iteration (avoid full SVD on 20k×256)
    rng = np.random.default_rng(0)
    omega = rng.standard_normal((X.shape[1], 2)).astype(np.float32)
    Y = X @ omega
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ X
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    proj = X @ Vt[:2].T   # (N, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(k)
    for kk in range(k):
        mask = labels == kk
        ax.scatter(proj[mask, 0], proj[mask, 1],
                   c=[cmap(kk)], s=3, alpha=0.4, label=f"C{kk}")
    ax.set(title=f"Token PCA 2D — K-Means clusters (k={k})",
           xlabel="PC1", ylabel="PC2")
    ax.legend(markerscale=4, fontsize=7, ncol=2)
    savefig(fig, out)


# ── §9 Spatial visualizations ────────────────────────────────────────────────
def visualize_samples(t1_dir, t2_dir, im1_dir, im2_dir, out_dir, n=N_VIZ_SAMPLES):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(t1_dir.glob("*.pt"))
    random.seed(42)
    samples = random.sample(files, min(n, len(files)))
    IMG_H, IMG_W = 512, 512

    def load_img(d, stem):
        for ext in (".png", ".jpg", ".tif"):
            p = d / f"{stem}{ext}"
            if p.exists():
                from PIL import Image
                return np.array(Image.open(p).convert("RGB"))
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    for f in tqdm(samples, desc="Visualizing", unit="img", leave=False):
        stem = f.stem
        d1 = load_token(f)
        d2 = load_token(t2_dir / f.name)
        if d1 is None or d2 is None:
            continue

        img1 = load_img(im1_dir, stem)
        img2 = load_img(im2_dir, stem)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"{stem}  ·  T1={d1['tokens'].shape[0]} tokens  T2={d2['tokens'].shape[0]} tokens",
            fontsize=11
        )
        for ax, img, d, lbl in [
            (axes[0], img1, d1, "T1 (before)"),
            (axes[1], img2, d2, "T2 (after)"),
        ]:
            ax.imshow(img)
            cx = d["centroids"][:, 0].numpy() * (img.shape[1] - 1)
            cy = d["centroids"][:, 1].numpy() * (img.shape[0] - 1)
            sz = np.clip(d["areas"].numpy() * 6000, 15, 250)
            sc = ax.scatter(cx, cy, s=sz, c=d["areas"].numpy(),
                            cmap="YlOrRd", alpha=0.75,
                            edgecolors="white", linewidths=0.4)
            ax.set(title=f"{lbl}")
            ax.axis("off")
            plt.colorbar(sc, ax=ax, shrink=0.55, label="area")
        savefig(fig, out_dir / f"sample_{stem}.png")


# ============================================================================
# Main
# ============================================================================

def run(args):
    root    = Path(args.dataset_root)
    out_dir = Path(args.output_dir) if args.output_dir else root / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output: {out_dir}")

    full_report = {}

    for split in args.splits:
        log.info(f"\n{'='*65}\nSplit: {split}\n{'='*65}")
        suf = "_test" if split == "test" else ""

        t1_dir    = root / f"tokens_T1{suf}"
        t2_dir    = root / f"tokens_T2{suf}"
        emb_t1    = root / f"embeddings_T1{suf}"
        im1_dir   = (root / "test" / "im1") if split == "test" else root / "im1"
        im2_dir   = (root / "test" / "im2") if split == "test" else root / "im2"

        if not t1_dir.is_dir():
            log.warning(f"tokens_T1{suf}/ not found — skipping {split}")
            continue

        sd = out_dir / split          # per-split sub-directory
        sd.mkdir(parents=True, exist_ok=True)

        # ── Load everything ─────────────────────────────────────────────────
        rec = collect_all(t1_dir, t2_dir, emb_t1, split)
        n   = len(rec["stems"])
        log.info(f"Valid pairs: {n}  |  Corrupt/missing: {len(rec['corrupt'])}")

        # ── §1 Token count ───────────────────────────────────────────────────
        log.info("§1 Token count statistics")
        plot_token_hist(rec["n1"], rec["n2"], sd / "token_count_histogram.png")

        high_thr, low_thr = args.high_token_thresh, args.low_token_thresh
        high_out = [(s, n1, "T1") for s, n1 in zip(rec["stems"], rec["n1"]) if n1 > high_thr] + \
                   [(s, n2, "T2") for s, n2 in zip(rec["stems"], rec["n2"]) if n2 > high_thr]
        low_out  = [(s, n1, "T1") for s, n1 in zip(rec["stems"], rec["n1"]) if n1 < low_thr] + \
                   [(s, n2, "T2") for s, n2 in zip(rec["stems"], rec["n2"]) if n2 < low_thr]
        with open(sd / "token_outliers.txt", "w") as fp:
            fp.write(f"# High (>{high_thr}): {len(high_out)}\n")
            for s, n_, side in high_out:
                fp.write(f"  {s}  {side}  tokens={n_}\n")
            fp.write(f"# Low (<{low_thr}): {len(low_out)}\n")
            for s, n_, side in low_out:
                fp.write(f"  {s}  {side}  tokens={n_}\n")
        log.info(f"  Outliers high={len(high_out)}  low={len(low_out)}")

        # ── §2 Token L2 norm ─────────────────────────────────────────────────
        log.info("§2 Token L2 norm distribution")
        plot_norm_hist(rec["norms_t1"], rec["norms_t2"],
                       sd / "token_norm_distribution.png")
        mean_norm = float(np.mean(rec["norms_t1"] + rec["norms_t2"]))
        if mean_norm < 0.5:
            log.warning(f"  ⚠ Mean norm {mean_norm:.3f} — possible embedding collapse!")
        elif mean_norm > 200:
            log.warning(f"  ⚠ Mean norm {mean_norm:.3f} — possible embedding explosion!")
        else:
            log.info(f"  Mean norm: {mean_norm:.3f} ✓")

        # ── §3 Embedding variance ────────────────────────────────────────────
        log.info("§3 SAM2 embedding variance")
        emb_vars = sample_embedding_variances(emb_t1, N_EMB_SAMPLE) if emb_t1.is_dir() else []
        if emb_vars:
            plot_emb_variance(emb_vars, sd / "embedding_variance_distribution.png")
            mean_var = float(np.mean(emb_vars))
            if mean_var < 1e-4:
                log.warning(f"  ⚠ Very low embedding variance {mean_var:.2e} — feature collapse?")
            else:
                log.info(f"  Mean embedding variance: {mean_var:.4f} ✓")
        else:
            log.warning("  Embeddings not found — skipping §3")
            emb_vars = []

        # ── §4 Token diversity ───────────────────────────────────────────────
        log.info("§4 Token diversity (intra-dataset cosine similarity)")
        intra_sims = compute_intra_cosine(rec["tokens_pool"])
        plot_sim_hist(intra_sims, sd / "token_similarity_distribution.png",
                      "Token Cosine Similarity (random pairs)",
                      "cosine similarity", "mediumpurple")
        mean_intra = float(np.mean(intra_sims))
        if mean_intra > 0.95:
            log.warning(f"  ⚠ Mean cosine sim {mean_intra:.3f} — tokens may be collapsed!")
        else:
            log.info(f"  Mean intra cosine sim: {mean_intra:.3f} ✓")

        # ── §5 Spatial coverage ──────────────────────────────────────────────
        log.info("§5 Spatial coverage")
        cov_thr = args.coverage_thresh
        low_cov = (
            [(s, "T1", round(c, 4)) for s, c in zip(rec["stems"], rec["cov_t1"]) if c < cov_thr] +
            [(s, "T2", round(c, 4)) for s, c in zip(rec["stems"], rec["cov_t2"]) if c < cov_thr]
        )
        with open(sd / "low_coverage_images.txt", "w") as fp:
            fp.write(f"# Images with coverage < {cov_thr}  ({len(low_cov)} cases)\n")
            for s, side, cov in low_cov:
                fp.write(f"  {s}  {side}  coverage={cov}\n")
        log.info(f"  Low coverage images: {len(low_cov)}")

        # Coverage histogram (simple, reuse norm histogram style)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for ax, data, lbl in zip(axes, [rec["cov_t1"], rec["cov_t2"]], ["T1", "T2"]):
            ax.hist(data, bins=50, color="mediumseagreen", alpha=0.85, edgecolor="none")
            ax.axvline(float(np.mean(data)), color="red", ls="--", lw=1.5,
                       label=f"mean {np.mean(data):.3f}")
            ax.axvline(cov_thr, color="black", ls=":", lw=1.5,
                       label=f"threshold {cov_thr}")
            ax.set(title=f"Coverage — {lbl}", xlabel="sum(areas)", ylabel="# images")
            ax.legend(fontsize=8)
        savefig(fig, sd / "area_coverage_histogram.png")

        # ── §6 Temporal consistency ──────────────────────────────────────────
        log.info("§6 Temporal consistency (token count ratio)")
        diff_thr = args.diff_ratio_thresh
        temporal_inc = [(s, round(r, 4))
                        for s, r in zip(rec["stems"], rec["diff_ratio"])
                        if r > diff_thr]
        with open(sd / "temporal_inconsistency.txt", "w") as fp:
            fp.write(f"# |N1-N2|/max > {diff_thr}  ({len(temporal_inc)} cases)\n")
            for s, r in temporal_inc:
                fp.write(f"  {s}  diff_ratio={r}\n")
        log.info(f"  Temporal inconsistency pairs: {len(temporal_inc)}")

        # ── §7 Temporal feature stability ────────────────────────────────────
        log.info("§7 Temporal feature stability (T1↔T2 nearest-neighbour sim)")
        temp_sims = compute_temporal_sim(rec["t1t2_pairs"])
        plot_sim_hist(temp_sims, sd / "temporal_similarity_distribution.png",
                      "T1→T2 Nearest-Neighbour Cosine Similarity (per image mean)",
                      "mean NN cosine similarity", "goldenrod")
        mean_temp = float(np.mean(temp_sims))
        log.info(f"  Mean temporal NN similarity: {mean_temp:.3f}")
        if mean_temp < 0.3:
            log.warning("  ⚠ Low temporal similarity — large domain shift between T1/T2?")

        # ── §8 K-Means clustering ────────────────────────────────────────────
        log.info(f"§8 K-Means clustering (k={args.n_clusters})")
        pool_np = rec["tokens_pool"].numpy().astype(np.float32)
        # Normalize before clustering
        norms = np.linalg.norm(pool_np, axis=1, keepdims=True).clip(1e-6)
        pool_norm = pool_np / norms
        labels_km, _ = numpy_kmeans(pool_norm, k=args.n_clusters, max_iter=50)
        plot_cluster_distribution(labels_km, args.n_clusters,
                                  sd / "token_cluster_visualization.png")
        plot_cluster_pca(pool_norm, labels_km, args.n_clusters,
                         sd / "token_cluster_pca.png")
        counts_km = np.bincount(labels_km, minlength=args.n_clusters)
        log.info(f"  Cluster sizes: {counts_km.tolist()}")

        # ── §9 Spatial visualizations ─────────────────────────────────────────
        log.info("§9 Token spatial visualizations")
        visualize_samples(t1_dir, t2_dir, im1_dir, im2_dir,
                          sd / "diagnostics_visualizations",
                          n=args.n_viz_samples)

        # ── §10 JSON report ──────────────────────────────────────────────────
        split_report = {
            "split": split,
            "num_images": n,
            "corrupt_files": len(rec["corrupt"]),
            "token_count_stats": {
                "T1": stats(rec["n1"]),
                "T2": stats(rec["n2"]),
            },
            "token_norm_stats": {
                "T1": stats(rec["norms_t1"]),
                "T2": stats(rec["norms_t2"]),
                "combined_mean_norm": round(mean_norm, 4),
                "status": "ok" if 0.5 < mean_norm < 200 else "warning",
            },
            "embedding_variance_stats": stats(emb_vars) if emb_vars else "not_available",
            "token_diversity": {
                "mean_intra_cosine_sim": round(mean_intra, 4),
                "std": round(float(np.std(intra_sims)), 4),
                "status": "ok" if mean_intra < 0.95 else "collapsed",
            },
            "coverage_stats": {
                "T1": stats(rec["cov_t1"]),
                "T2": stats(rec["cov_t2"]),
                "low_coverage_count": len(low_cov),
                "threshold": cov_thr,
            },
            "temporal_consistency": {
                "diff_ratio_stats": stats(rec["diff_ratio"]),
                "inconsistent_pairs": len(temporal_inc),
                "threshold": diff_thr,
            },
            "temporal_feature_stability": {
                "mean_nn_cosine_sim": round(mean_temp, 4),
                "std": round(float(np.std(temp_sims)), 4),
                "n_pairs_sampled": len(temp_sims),
            },
            "token_clustering": {
                "k": args.n_clusters,
                "cluster_sizes": counts_km.tolist(),
                "n_tokens_used": int(len(pool_np)),
            },
            "outliers": {
                "high_count": len(high_out),
                "low_count": len(low_out),
                "high_thresh": high_thr,
                "low_thresh": low_thr,
            },
        }
        full_report[split] = split_report

        # Print quick summary
        log.info(
            f"\n  ── Summary ({split}) ─────────────────\n"
            f"  Valid pairs:     {n}\n"
            f"  Token count:     T1 mean={np.mean(rec['n1']):.1f}  T2 mean={np.mean(rec['n2']):.1f}\n"
            f"  L2 norm mean:    {mean_norm:.3f}\n"
            f"  Intra cosine:    {mean_intra:.3f}\n"
            f"  Coverage T1:     {np.mean(rec['cov_t1']):.3f}  T2: {np.mean(rec['cov_t2']):.3f}\n"
            f"  Temporal sim:    {mean_temp:.3f}\n"
            f"  Low coverage:    {len(low_cov)}\n"
            f"  Inconsistent:    {len(temporal_inc)}\n"
        )

    report_path = out_dir / "dataset_report.json"
    with open(report_path, "w") as fp:
        json.dump(full_report, fp, indent=2)
    log.info(f"Report: {report_path}")
    log.info("Done.")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Stage 2.5 — Full dataset diagnostics")
    p.add_argument("--dataset_root",   default="/home/chung/RS/phase1/SECOND")
    p.add_argument("--splits",         nargs="+", default=["train"],
                                       choices=["train", "test"])
    p.add_argument("--output_dir",     default=None)
    p.add_argument("--high_token_thresh", type=int,   default=500)
    p.add_argument("--low_token_thresh",  type=int,   default=5)
    p.add_argument("--coverage_thresh",   type=float, default=0.40)
    p.add_argument("--diff_ratio_thresh", type=float, default=0.50)
    p.add_argument("--n_clusters",        type=int,   default=10)
    p.add_argument("--n_viz_samples",     type=int,   default=20)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
