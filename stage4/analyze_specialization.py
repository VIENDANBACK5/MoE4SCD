"""
stage4/analyze_specialization.py
=================================
Stage 4D — Expert Specialization Analysis

Runs inference on validation data using a trained Stage 4C checkpoint and
produces the following outputs in stage5/:

    expert_class_matrix.csv    — M[expert, semantic_class] token counts
    expert_purity.md           — purity metric table + interpretation
    expert_stats.csv           — per-expert: tokens, change_ratio, delta, disp
    top_tokens/expert_{e}/     — top-20 token crops per expert (PNG)
    expert_analysis_report.md  — narrative summary

Usage:
    python stage4/analyze_specialization.py \\
        --checkpoint SECOND/stage4C/best_model.pt \\
        --config     SECOND/stage4C/config.json \\
        --tokens_T1  SECOND/tokens_T1 \\
        --tokens_T2  SECOND/tokens_T2 \\
        --matches    SECOND/matches \\
        --labels     SECOND/label1 \\
        --images     SECOND/im1 \\
        --output     stage5 \\
        --device     cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ── make parent searchable ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE

# ─────────────────────────────────────────────────────────────────────────────
# SECOND semantic classes
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = {
    0: "background",
    1: "water",
    2: "soil/impervious",
    3: "vegetation",
    4: "building",
    5: "farmland",
    6: "low_veg",
}
N_CLASSES = len(CLASS_NAMES)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> MoEConfig:
    """Load MoEConfig from config.json saved during training."""
    d = json.loads(path.read_text())
    cfg = MoEConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_model(ckpt_path: Path, cfg: MoEConfig, device: torch.device) -> TokenChangeReasonerMoE:
    model = TokenChangeReasonerMoE(cfg).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def get_label_at_centroid(
    label_img: np.ndarray,   # [H_img, W_img] uint8
    centroids: torch.Tensor, # [N, 2]  float in [0,1]
    img_h: int, img_w: int,
) -> np.ndarray:             # [N] int  –1 if out-of-bounds
    """Map fractional centroids to pixel-space class labels."""
    cx = (centroids[:, 0].numpy() * (img_w - 1)).round().astype(int)
    cy = (centroids[:, 1].numpy() * (img_h - 1)).round().astype(int)
    valid = (cx >= 0) & (cx < img_w) & (cy >= 0) & (cy < img_h)
    labels = np.full(len(centroids), -1, dtype=int)
    labels[valid] = label_img[cy[valid], cx[valid]]
    return labels


def get_router_assignments(
    model: TokenChangeReasonerMoE,
    batch: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Run a forward pass and collect per-token routing information.

    Returns dict with:
        expert_ids  : [B, N]  long
        router_probs: [B, N, E]  float
        repr_ctx    : [B, N, H]  float  (post-MoE representations)
        change_logits:[B, N] float
        delta_pred  : [M] float
    """
    with torch.no_grad():
        B, N, _ = batch["tokens_pad"].shape

        # ── re-implement forward to capture router probs ──────────────────
        flat_emb  = batch["tokens_pad"].reshape(B * N, -1)
        flat_tid  = batch["time_ids_pad"].reshape(B * N)
        flat_cen  = batch["centroids_pad"].reshape(B * N, 2)
        flat_area = batch["log_areas_pad"].reshape(B * N)

        flat_repr = model.token_encoder(flat_emb, flat_tid, flat_cen, flat_area)
        repr_pad  = flat_repr.reshape(B, N, -1)

        repr_ctx  = model.reasoner(repr_pad, batch["padding_mask"])
        repr_ctx  = model.graph(
            repr_ctx, batch["padding_mask"],
            batch["centroids_pad"], batch["time_ids_pad"],
        )

        # Router
        x_flat       = repr_ctx.reshape(B * N, -1)  # [T, H]
        
        # In v3, the router needs semantic_labels
        if model.cfg.router_version == "v3":
            semantic_labels = batch.get("semantic_labels_pad")
            if semantic_labels is not None:
                sl = semantic_labels.reshape(B * N).long()
                sl_onehot = F.one_hot(sl, num_classes=7).to(repr_ctx.dtype)
            else:
                sl_onehot = torch.zeros(B * N, 7, device=repr_ctx.device, dtype=repr_ctx.dtype)
            router_input = torch.cat([x_flat, sl_onehot], dim=-1)
        # v2 needs log_areas and delta_hints
        elif model.cfg.router_version == "v2":
            la = batch["log_areas_pad"].reshape(B * N, 1)
            # Rough delta hint (0 for unassigned, proper norm for assigned)
            dh = torch.zeros(B * N, 1, device=repr_ctx.device, dtype=repr_ctx.dtype)
            pair_b, pair_i, pair_j = batch["pair_b"], batch["pair_i"], batch["pair_j"]
            if len(pair_b) > 0:
                diff = repr_ctx[pair_b, pair_i] - repr_ctx[pair_b, pair_j]
                dist = diff.norm(dim=-1)
                # Note: this ignores batch dimension for single-sample inference, 
                # but works because B=1
                dh[pair_i] = dist.unsqueeze(-1)
            router_input = torch.cat([x_flat, la, dh], dim=-1)
        else:
            router_input = x_flat
            
        router_logits = model.moe.router(router_input)
        router_probs  = torch.softmax(router_logits, dim=-1)   # [T, E]
        expert_ids    = router_probs.argmax(dim=-1)             # [T]

        # Expert outputs (to get post-MoE context for change head)
        out_flat = repr_ctx.reshape(B * N, -1).clone()
        E = model.moe.num_experts
        for e in range(E):
            mask = (expert_ids == e)
            if mask.sum() > 0:
                out_flat[mask] = model.moe.experts[e](x_flat[mask]).to(repr_ctx.dtype)
        repr_out = repr_ctx + out_flat.reshape(B, N, -1)

        change_logits = model.change_head(repr_out)

        pair_b, pair_i, pair_j = batch["pair_b"], batch["pair_i"], batch["pair_j"]
        if len(pair_b) > 0:
            delta_pred = model.delta_head(repr_out[pair_b, pair_i], repr_out[pair_b, pair_j])
        else:
            delta_pred = torch.zeros(0, device=repr_ctx.device)

    return {
        "expert_ids":   expert_ids.reshape(B, N),           # [B, N]
        "router_probs": router_probs.reshape(B, N, E),      # [B, N, E]
        "repr_ctx":     repr_out,                           # [B, N, H]
        "change_logits": change_logits,                     # [B, N]
        "delta_pred":   delta_pred,                         # [M]
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Dirs ─────────────────────────────────────────────────────────────────
    t1_dir    = Path(args.tokens_T1)
    t2_dir    = Path(args.tokens_T2)
    match_dir = Path(args.matches)
    label_dir = Path(args.labels)
    image_dir = Path(args.images) if args.images else None

    # ── Model ────────────────────────────────────────────────────────────────
    cfg   = load_config(Path(args.config))
    model = load_model(Path(args.checkpoint), cfg, device)
    E     = cfg.moe_num_experts
    print(f"[analyze] Model loaded  ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"          Experts: {E}   Device: {device}")

    # ── Collect all stems ────────────────────────────────────────────────────
    all_stems = sorted([
        p.stem.replace("_matches", "")
        for p in match_dir.glob("*_matches.pt")
        if (t1_dir / f"{p.stem.replace('_matches','')}.pt").exists()
    ])

    # Use a val split: last 10% (consistent with training)
    n_val  = max(1, len(all_stems) // 10)
    stems  = all_stems[-n_val:] if not args.all_data else all_stems
    print(f"          Samples: {len(stems)} ({'all' if args.all_data else 'val split'})")

    print(f"          Samples: {len(stems)} ({'all' if args.all_data else 'val split'})")

    # ── Accumulators ─────────────────────────────────────────────────────────
    # expert_class_matrix[e, c] = count of tokens from class c routed to expert e
    matrix         = np.zeros((E, N_CLASSES), dtype=np.int64)
    # per-expert stats
    expert_counts  = np.zeros(E, dtype=np.int64)
    expert_changed = np.zeros(E, dtype=np.float64)   # sum of change probs
    expert_delta   = np.zeros(E, dtype=np.float64)   # sum of delta distances
    expert_disp    = np.zeros(E, dtype=np.float64)   # sum of spatial displacement

    # For top-K tokens per expert: store (confidence, crop_info) tuples
    top_tokens: Dict[int, List[tuple]] = {e: [] for e in range(E)}

    # ── Per-sample inference ──────────────────────────────────────────────────
    print(f"[analyze] Starting inference loop...")
    for stem in tqdm(stems, desc="Analyzing specialization"):
        t1_path = t1_dir / f"{stem}.pt"
        t2_path = t2_dir / f"{stem}.pt"
        mtch_path = match_dir / f"{stem}_matches.pt"

        if not (t1_path.exists() and t2_path.exists() and mtch_path.exists()):
            continue

        t1   = torch.load(t1_path,     weights_only=True)
        t2   = torch.load(t2_path,     weights_only=True)
        mtch_raw = torch.load(mtch_path, weights_only=False)

        pairs = mtch_raw.get("pairs", [])
        if isinstance(pairs, list):
            pairs = torch.tensor([[float(p[0]),float(p[1]),float(p[2])] for p in pairs]) \
                    if len(pairs) > 0 else torch.zeros(0, 3)
        else:
            pairs = pairs.float()

        # ── Load semantic label ───────────────────────────────────────────
        lbl_path = label_dir / f"{stem}.png"
        lbl_img = None
        if lbl_path.exists():
            lbl_pil = Image.open(lbl_path)
            # Ensure it's in a mode where indexing works as expected (L or P)
            if lbl_pil.mode not in ("L", "P"):
                lbl_pil = lbl_pil.convert("L")
            lbl_img = np.array(lbl_pil)
            img_h, img_w = lbl_img.shape[:2]

        # ── Extract semantic labels for token centroids ───────────────────
        c1 = t1["centroids"].numpy()
        c2 = t2["centroids"].numpy()
        c_all = np.concatenate([c1, c2], axis=0) # [N1+N2, 2]
        
        semantic_list = []
        if lbl_img is not None:
            for cx, cy in c_all:
                px = max(0, min(int(round(cx * (img_w - 1))), img_w - 1))
                py = max(0, min(int(round(cy * (img_h - 1))), img_h - 1))
                val = lbl_img[py, px]
                
                if isinstance(val, (np.ndarray, list)) and len(val) > 1:
                    raw_val = float(val[0])
                else:
                    raw_val = float(val)
                cls = int(round(raw_val / 255.0 * 6))
                if cls >= N_CLASSES:
                    cls = 0
                semantic_list.append(cls)
        else:
            semantic_list = [0] * len(c_all)
            
        semantic_labels = torch.tensor(semantic_list, dtype=torch.long)

        sample = SampleData(
            tokens_t1    = t1["tokens"].float(),
            tokens_t2    = t2["tokens"].float(),
            centroids_t1 = t1["centroids"].float(),
            centroids_t2 = t2["centroids"].float(),
            areas_t1     = t1["areas"].float(),
            areas_t2     = t2["areas"].float(),
            match_pairs  = pairs,
            semantic_labels=semantic_labels,
        )
        batch = build_batch([sample], cfg, device)
        info  = get_router_assignments(model, batch)

        B, N = info["expert_ids"].shape  # B==1 always here
        padding_mask = batch["padding_mask"][0]  # [N]


        # Centroids for this sample (un-padded)
        centroids = batch["centroids_pad"][0]  # [N, 2]
        time_ids  = batch["time_ids_pad"][0]   # [N]
        log_areas = batch["log_areas_pad"][0]  # [N]

        expert_ids_n  = info["expert_ids"][0]           # [N]
        router_probs_n = info["router_probs"][0]        # [N, E]
        change_prob   = torch.sigmoid(info["change_logits"][0])  # [N]

        for n in range(N):
            if padding_mask[n]:
                continue
            e = int(expert_ids_n[n])
            conf = float(router_probs_n[n, e])
            cp   = float(change_prob[n])
            area = float(log_areas[n])
            cx   = float(centroids[n, 0])
            cy   = float(centroids[n, 1])

            # Semantic class was precomputed
            cls = int(semantic_labels[n].item())

            matrix[e, cls] += 1

            expert_counts[e]  += 1
            expert_changed[e] += cp
            expert_disp[e]    += (cx**2 + cy**2) ** 0.5  # dist from origin

            # Top-K accumulation
            top_tokens[e].append((conf, stem, n, cx, cy, cls, cp, area))

    # ── Sort top-K per expert ─────────────────────────────────────────────────
    for e in range(E):
        top_tokens[e].sort(key=lambda x: x[0], reverse=True)
        top_tokens[e] = top_tokens[e][:20]

    # ─────────────────────────────────────────────────────────────────────────
    # PART 1: Expert-class matrix CSV
    # ─────────────────────────────────────────────────────────────────────────
    cls_cols = [CLASS_NAMES[c] for c in range(N_CLASSES)]
    matrix_path = out_dir / "expert_class_matrix.csv"
    with open(matrix_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expert"] + cls_cols)
        for e in range(E):
            w.writerow([f"expert_{e}"] + matrix[e].tolist())
    print(f"[✓] {matrix_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 2: Expert purity
    # ─────────────────────────────────────────────────────────────────────────
    purity_rows = []
    for e in range(E):
        total = int(matrix[e].sum())
        if total == 0:
            purity_rows.append((e, 0, "—", 0.0, "⚠ dead expert"))
            continue
        dominant_cls = int(matrix[e].argmax())
        purity = float(matrix[e, dominant_cls]) / total
        if purity < 0.3:
            interp = "weak specialization"
        elif purity < 0.5:
            interp = "moderate specialization"
        else:
            interp = "strong specialization ✓"
        purity_rows.append((e, total, CLASS_NAMES[dominant_cls], purity, interp))

    purity_path = out_dir / "expert_purity.md"
    with open(purity_path, "w") as f:
        f.write("# Expert Purity Report\n\n")
        f.write("**Purity** = max_class_count / total_tokens_routed_to_expert\n\n")
        f.write("| expert | token_count | dominant_class | purity | interpretation |\n")
        f.write("|--------|------------|----------------|--------|----------------|\n")
        for e, total, dom, pur, interp in purity_rows:
            f.write(f"| expert_{e} | {total:,} | {dom} | {pur:.3f} | {interp} |\n")
        f.write("\n## Purity Scale\n")
        f.write("- purity < 0.30 → weak specialization\n")
        f.write("- purity 0.30–0.50 → moderate specialization\n")
        f.write("- purity > 0.50 → strong specialization\n")
    print(f"[✓] {purity_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 3: Expert statistics CSV
    # ─────────────────────────────────────────────────────────────────────────
    stats_path = out_dir / "expert_stats.csv"
    with open(stats_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expert", "token_count", "change_ratio", "mean_disp", "dominant_class", "purity"])
        for e in range(E):
            total = int(expert_counts[e])
            if total == 0:
                w.writerow([f"expert_{e}", 0, 0.0, 0.0, "—", 0.0])
                continue
            change_ratio = expert_changed[e] / total
            mean_disp    = expert_disp[e]    / total
            dom_cls = CLASS_NAMES[int(matrix[e].argmax())]
            pur     = float(matrix[e].max()) / total
            w.writerow([f"expert_{e}", total,
                        f"{change_ratio:.4f}", f"{mean_disp:.4f}",
                        dom_cls, f"{pur:.4f}"])
    print(f"[✓] {stats_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 5: Top-20 token record cards per expert
    # ─────────────────────────────────────────────────────────────────────────
    COLORS = ["red", "blue", "green", "goldenrod"]

    for e in range(E):
        eout = out_dir / "top_tokens" / f"expert_{e}"
        eout.mkdir(parents=True, exist_ok=True)

        for rank, (conf, stem, n, cx, cy, cls, cp, area) in enumerate(top_tokens[e]):
            # Load satellite image patch if available
            patch = None
            if image_dir is not None:
                impath = image_dir / f"{stem}.png"
                if not impath.exists():
                    impath = image_dir / f"{stem}.jpg"
                if impath.exists():
                    img = Image.open(impath).convert("RGB")
                    iw, ih = img.size
                    pw = max(32, int(np.exp(area) ** 0.5))  # approx patch size
                    px = int(round(cx * (iw - 1)))
                    py = int(round(cy * (ih - 1)))
                    x0, y0 = max(0, px - pw), max(0, py - pw)
                    x1, y1 = min(iw, px + pw), min(ih, py + pw)
                    patch = img.crop((x0, y0, x1, y1)).resize((128, 128), Image.BILINEAR)

            # Create a card image
            card_w, card_h = 200, 160 + (128 if patch else 0)
            card = Image.new("RGB", (card_w, card_h), "white")
            draw = ImageDraw.Draw(card)

            # Color band
            draw.rectangle([0, 0, card_w, 8], fill=COLORS[e % len(COLORS)])

            # Paste patch
            if patch:
                card.paste(patch, (36, 12))

            yoff = (128 + 16) if patch else 12
            draw.text((4, yoff),      f"Expert {e}",               fill="black")
            draw.text((4, yoff + 16), f"Rank #{rank+1}",           fill="gray")
            draw.text((4, yoff + 32), f"Class: {CLASS_NAMES.get(cls,'?')}",  fill="navy")
            draw.text((4, yoff + 48), f"Conf:  {conf:.3f}",        fill="darkgreen")
            draw.text((4, yoff + 64), f"Change:{cp:.2f}",          fill="firebrick")
            draw.text((4, yoff + 80), f"Image: {stem}",            fill="dimgray")

            card.save(eout / f"rank{rank+1:02d}_{stem}_n{n}.png")

    print(f"[✓] stage5/top_tokens/  (top-20 cards per expert)")

    # ─────────────────────────────────────────────────────────────────────────
    # PART 5b: Expert analysis report (narrative)
    # ─────────────────────────────────────────────────────────────────────────
    report_path = out_dir / "expert_analysis_report.md"
    with open(report_path, "w") as f:
        f.write("# Expert Analysis Report — Stage 4C\n\n")

        f.write("## Expert Load Distribution\n\n")
        total_all = int(expert_counts.sum())
        f.write("| expert | tokens | fraction |\n|--------|--------|----------|\n")
        for e in range(E):
            t = int(expert_counts[e])
            frac = t / total_all if total_all > 0 else 0
            f.write(f"| expert_{e} | {t:,} | {frac:.1%} |\n")
        f.write("\n")

        f.write("## Expert–Class Matrix\n\n")
        f.write("| expert | " + " | ".join(CLASS_NAMES[c] for c in range(N_CLASSES)) + " |\n")
        f.write("|" + "--------|" * (N_CLASSES + 1) + "\n")
        for e in range(E):
            row_total = max(1, matrix[e].sum())
            cells = " | ".join(f"{matrix[e, c]} ({100*matrix[e,c]/row_total:.0f}%)"
                               for c in range(N_CLASSES))
            f.write(f"| expert_{e} | {cells} |\n")
        f.write("\n")

        f.write("## Purity Summary\n\n")
        f.write("| expert | dominant_class | purity | interpretation |\n")
        f.write("|--------|----------------|--------|----------------|\n")
        for e, total, dom, pur, interp in purity_rows:
            f.write(f"| expert_{e} | {dom} | {pur:.3f} | {interp} |\n")
        f.write("\n")

        # Determine dominant class per expert
        f.write("## Specialization Conclusions\n\n")
        dom_classes = {}
        for e in range(E):
            if expert_counts[e] == 0:
                f.write(f"- **Expert {e}**: ⚠ DEAD — no tokens assigned\n")
            else:
                dom = CLASS_NAMES[int(matrix[e].argmax())]
                pur = float(matrix[e].max()) / max(1, int(expert_counts[e]))
                dom_classes[e] = dom
                frac = expert_counts[e] / max(1, total_all)
                cr   = expert_changed[e] / max(1, expert_counts[e])
                level = "strong" if pur > 0.5 else ("moderate" if pur > 0.3 else "weak")
                f.write(f"- **Expert {e}** ({frac:.0%} of tokens): "
                        f"dominant={dom}, purity={pur:.3f} → **{level}** specialization, "
                        f"change_ratio={cr:.2f}\n")

        # Check for noise expert
        fracs = expert_counts / max(1, total_all)
        if fracs.max() > 0.45:
            noise_e = int(fracs.argmax())
            f.write(f"\n> ⚠ Expert {noise_e} handles {fracs[noise_e]:.0%} of all tokens — "
                    f"possible **hub/noise expert**. Router improvements (Part 6–8) "
                    f"should redistribute load.\n")

        f.write("\n## Recommendations\n\n")
        f.write("- Train Stage 4D with `--router_version v2` and `--expert_dropout 0.1`\n")
        f.write("- Monitor `expert_fracs` in training log for convergence toward uniform load\n")
        f.write("- Re-run this analysis after Stage 4D training to compare purity scores\n")

    print(f"[✓] {report_path}")
    print("\n[analyze_specialization] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Stage 4D Expert Specialization Analysis")
    p.add_argument("--checkpoint", default="SECOND/stage4C/best_model.pt")
    p.add_argument("--config",     default="SECOND/stage4C/config.json")
    p.add_argument("--tokens_T1",  default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2",  default="SECOND/tokens_T2")
    p.add_argument("--matches",    default="SECOND/matches")
    p.add_argument("--labels",     default="SECOND/label1",
                   help="Directory of semantic label PNGs (label1/)")
    p.add_argument("--images",     default="SECOND/im1",
                   help="Directory of satellite images for token crops")
    p.add_argument("--output",     default="stage5")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--all_data",   action="store_true",
                   help="Run on all data (not just val split)")
    return p.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
