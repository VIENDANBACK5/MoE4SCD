"""
stage4/visualize_expert_map.py
================================
Stage 4D — Expert Assignment Visualization

For each image pair in the validation set, overlays expert-coloured
token regions on the satellite image and saves to stage5/visualizations/.

Expert colour mapping:
    Expert 0 → red
    Expert 1 → blue
    Expert 2 → green
    Expert 3 → yellow / goldenrod

Usage:
    python stage4/visualize_expert_map.py \\
        --checkpoint SECOND/stage4C/best_model.pt \\
        --config     SECOND/stage4C/config.json \\
        --tokens_T1  SECOND/tokens_T1 \\
        --tokens_T2  SECOND/tokens_T2 \\
        --matches    SECOND/matches \\
        --labels     SECOND/label1 \\
        --images     SECOND/im1 \\
        --output     stage5/visualizations \\
        --n_images   30 \\
        --device     cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# RGBA: expert colours with alpha=140 (semi-transparent overlay)
EXPERT_COLORS = [
    (220, 50,  50,  140),   # Expert 0 → red
    (50,  100, 220, 140),   # Expert 1 → blue
    (50,  180, 50,  140),   # Expert 2 → green
    (220, 190, 20,  140),   # Expert 3 → yellow/gold
]

CLASS_NAMES = {
    0: "bg", 1: "water", 2: "soil", 3: "veg",
    4: "building", 5: "farmland", 6: "low_veg",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (re-used from analyze_specialization)
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: Path) -> MoEConfig:
    import json
    d = json.loads(path.read_text())
    cfg = MoEConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def load_model(ckpt: Path, cfg: MoEConfig, device: torch.device) -> TokenChangeReasonerMoE:
    model = TokenChangeReasonerMoE(cfg).to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    if "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def get_expert_ids(model: TokenChangeReasonerMoE,
                   batch: dict,
                   device: torch.device):
    """Return expert_ids[B,N] and change_probs[B,N] for a batch."""
    B, N, _ = batch["tokens_pad"].shape

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

    x_flat       = repr_ctx.reshape(B * N, -1)
    router_logits = model.moe.router(x_flat)
    router_probs  = torch.softmax(router_logits, dim=-1)
    expert_ids    = router_probs.argmax(dim=-1).reshape(B, N)

    # Quick change prediction
    out_flat = repr_ctx.reshape(B * N, -1).clone()
    for e in range(model.moe.num_experts):
        mask = (expert_ids.reshape(-1) == e)
        if mask.sum() > 0:
            out_flat[mask] = model.moe.experts[e](x_flat[mask]).to(repr_ctx.dtype)
    repr_out     = repr_ctx + out_flat.reshape(B, N, -1)
    change_probs = torch.sigmoid(model.change_head(repr_out))  # [B,N]

    return expert_ids, change_probs, router_probs.reshape(B, N, -1)


def draw_token_overlay(
    sat_img: Image.Image,
    centroids: torch.Tensor,         # [N, 2] float in [0,1]
    areas: torch.Tensor,             # [N] float log-area
    expert_ids: torch.Tensor,        # [N] long
    change_probs: torch.Tensor,      # [N] float
    padding_mask: torch.Tensor,      # [N] bool True=pad
    label_img: np.ndarray | None,
    E: int,
) -> Image.Image:
    """
    Overlay semi-transparent coloured circles on satellite image.
    Circle size ∝ token area. Border thickness ∝ change probability.
    """
    W, H = sat_img.size
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    for n in range(len(centroids)):
        if padding_mask[n]:
            continue
        e  = int(expert_ids[n])
        cp = float(change_probs[n])
        cx = int(round(float(centroids[n, 0]) * (W - 1)))
        cy = int(round(float(centroids[n, 1]) * (H - 1)))
        r  = max(4, min(20, int(np.exp(float(areas[n])) ** 0.5 * 0.5)))

        color = EXPERT_COLORS[e % len(EXPERT_COLORS)]
        # Fill circle
        draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)], fill=color)

        # Bright border if likely changed
        if cp > 0.5:
            border_color = (255, 255, 255, 200)
            draw.ellipse([(cx - r, cy - r), (cx + r, cy + r)],
                         outline=border_color, width=2)

    # Composite onto satellite image
    sat_rgba = sat_img.convert("RGBA")
    result   = Image.alpha_composite(sat_rgba, overlay)
    return result.convert("RGB")


def add_legend(img: Image.Image, E: int) -> Image.Image:
    """Add a small legend strip at the bottom."""
    lh  = 30
    W, H = img.size
    legend = Image.new("RGB", (W, H + lh), (30, 30, 30))
    legend.paste(img, (0, 0))
    draw = ImageDraw.Draw(legend)
    labels = [f"E{e}" for e in range(E)]
    sw     = W // E
    for e in range(E):
        rgb = EXPERT_COLORS[e][:3]
        x0 = e * sw
        draw.rectangle([x0, H, x0 + sw - 2, H + lh - 2], fill=rgb)
        draw.text((x0 + 4, H + 6), labels[e], fill="white")
    return legend


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def visualize(args):
    device  = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    t1_dir    = Path(args.tokens_T1)
    t2_dir    = Path(args.tokens_T2)
    match_dir = Path(args.matches)
    image_dir = Path(args.images)
    label_dir = Path(args.labels) if args.labels else None

    # Load model
    cfg   = load_config(Path(args.config))
    model = load_model(Path(args.checkpoint), cfg, device)
    E     = cfg.moe_num_experts
    print(f"[visualize] Model loaded  E={E}  device={device}")

    # Val stems
    all_stems = sorted([
        p.stem.replace("_matches", "")
        for p in match_dir.glob("*_matches.pt")
        if (t1_dir / f"{p.stem.replace('_matches','')}.pt").exists()
    ])
    n_val  = max(1, len(all_stems) // 10)
    stems  = all_stems[-n_val:][:args.n_images]
    print(f"[visualize] Visualising {len(stems)} image pairs → {out_dir}")

    saved = 0
    for stem in stems:
        # ── Try to find satellite image ────────────────────────────────────
        sat_path = None
        for ext in [".png", ".jpg", ".tif"]:
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                sat_path = p
                break
        if sat_path is None:
            print(f"  [skip] no image for {stem}")
            continue

        sat_img = Image.open(sat_path).convert("RGB")

        # ── Load tokens ────────────────────────────────────────────────────
        t1   = torch.load(t1_dir    / f"{stem}.pt", weights_only=True)
        t2   = torch.load(t2_dir    / f"{stem}.pt", weights_only=True)
        mtch = torch.load(match_dir / f"{stem}_matches.pt", weights_only=False)

        pairs = mtch.get("pairs", [])
        if isinstance(pairs, list):
            pairs = (torch.tensor([[float(p[0]),float(p[1]),float(p[2])]
                                    for p in pairs])
                     if len(pairs) > 0 else torch.zeros(0, 3))
        else:
            pairs = pairs.float()

        sample = SampleData(
            tokens_t1    = t1["tokens"].float(),
            tokens_t2    = t2["tokens"].float(),
            centroids_t1 = t1["centroids"].float(),
            centroids_t2 = t2["centroids"].float(),
            areas_t1     = t1["areas"].float(),
            areas_t2     = t2["areas"].float(),
            match_pairs  = pairs,
        )
        batch = build_batch([sample], cfg, device)

        expert_ids_bn, change_probs_bn, _ = get_expert_ids(model, batch, device)

        N           = batch["tokens_pad"].shape[1]
        expert_ids  = expert_ids_bn[0].cpu()      # [N]
        change_probs= change_probs_bn[0].cpu()    # [N]
        centroids   = batch["centroids_pad"][0].cpu()   # [N, 2]
        log_areas   = batch["log_areas_pad"][0].cpu()   # [N]
        pad_mask    = batch["padding_mask"][0].cpu()    # [N]

        # Load semantic label if available
        lbl_img = None
        if label_dir is not None:
            lp = label_dir / f"{stem}.png"
            if lp.exists():
                lbl_img = np.array(Image.open(lp))

        # ── Draw expert overlay ────────────────────────────────────────────
        result = draw_token_overlay(
            sat_img, centroids, log_areas, expert_ids,
            change_probs, pad_mask, lbl_img, E,
        )
        result = add_legend(result, E)

        out_path = out_dir / f"{stem}_expert_map.png"
        result.save(out_path)
        saved += 1

    print(f"[✓] Saved {saved} expert maps → {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Stage 4D Expert Visualization")
    p.add_argument("--checkpoint", default="SECOND/stage4C/best_model.pt")
    p.add_argument("--config",     default="SECOND/stage4C/config.json")
    p.add_argument("--tokens_T1",  default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2",  default="SECOND/tokens_T2")
    p.add_argument("--matches",    default="SECOND/matches")
    p.add_argument("--labels",     default="SECOND/label1")
    p.add_argument("--images",     default="SECOND/im1")
    p.add_argument("--output",     default="stage5/visualizations")
    p.add_argument("--n_images",   type=int, default=30)
    p.add_argument("--device",     default="cuda")
    return p.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
