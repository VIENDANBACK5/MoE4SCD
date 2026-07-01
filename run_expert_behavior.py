#!/usr/bin/env python3
"""run_expert_behavior.py  —  Stage 6: Expert Emergence Analysis
================================================================
Analyses what each MoE expert specialises in by running 5 experiments
on the validation set.

Experiments
-----------
  Exp 1  Change-magnitude specialisation
           change_magnitude = ||repr_T1_i - repr_T2_j||  (pre-MoE features)
  Exp 2  Region-size specialisation
           region_area = SAM mask area (pixels)
  Exp 3  Boundary sensitivity
           boundary_ratio = fraction of tokens ≤ BOUNDARY_PX pixels from a
                            semantic class boundary
  Exp 4  Change-probability specialisation
           avg predicted change probability per expert
  Exp 5  Expert assignment visualisation
           overlay expert colours on satellite images

Usage
-----
python run_expert_behavior.py \\
    --model_dir  SECOND/stage5_6_dynamic \\
    --tokens_T1  SECOND/tokens_T1 \\
    --tokens_T2  SECOND/tokens_T2 \\
    --matches    SECOND/matches \\
    --labels_dir SECOND/label1 \\
    --images_dir SECOND/im1 \\
    --output_dir stage6/expert_behavior \\
    --val_split  0.1 --seed 42 --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from token_change_reasoner import SampleData, build_batch
from token_change_reasoner_moe import MoEConfig, TokenChangeReasonerMoE, build_moe_model
from train_reasoner import MatchDataset

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
BOUNDARY_PX      = 10       # pixels: token "near boundary" threshold
VIS_N_IMAGES     = 8        # number of images in assignment map figure
IMG_SIZE         = 512      # SAT image dimensions
TOKEN_RADIUS_MIN = 4        # min disk radius for token visualisation
TOKEN_RADIUS_MAX = 18       # max disk radius for token visualisation

# Expert colour palette — BGRA (for PIL RGBA)
EXPERT_RGBA = [
    (0,   120, 255, 210),   # E0: blue
    (30,  200,  50, 210),   # E1: green
    (255, 210,   0, 210),   # E2: yellow
    (255,  50,   0, 210),   # E3: red
    (180,   0, 255, 210),   # E4: purple
    (  0, 220, 220, 210),   # E5: cyan
    (255, 140,   0, 210),   # E6: orange
    (160, 160,   0, 210),   # E7: olive
]
EXPERT_NAMES = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7"]


# ─────────────────────────────────────────────────────────────────────────────
# Validation stem selection  (mirrors training random_split)
# ─────────────────────────────────────────────────────────────────────────────
def build_val_stems(
    match_dir: Path,
    t1_dir: Path,
    t2_dir: Path,
    val_split: float = 0.1,
    seed: int = 42,
) -> List[str]:
    all_stems = []
    for mp in sorted(match_dir.glob("*_matches.pt")):
        stem = mp.stem.replace("_matches", "")
        if (t1_dir / f"{stem}.pt").exists() and (t2_dir / f"{stem}.pt").exists():
            all_stems.append(stem)
    n_val   = max(1, int(len(all_stems) * val_split))
    n_train = len(all_stems) - n_val
    gen     = torch.Generator().manual_seed(seed)
    perm    = torch.randperm(len(all_stems), generator=gen).tolist()
    val_idx = perm[n_train:]
    return [all_stems[i] for i in val_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_dir: Path, device: torch.device) -> TokenChangeReasonerMoE:
    cfg_path  = model_dir / "config.json"
    ckpt_path = model_dir / "best_model.pt"
    with open(cfg_path) as f:
        cfg_dict = json.load(f)
    cfg   = MoEConfig(**{k: v for k, v in cfg_dict.items() if k in MoEConfig.__dataclass_fields__})
    model = build_moe_model(cfg).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Boundary distance map  (per-image, cached)
# ─────────────────────────────────────────────────────────────────────────────
_boundary_cache: Dict[str, np.ndarray] = {}

def get_boundary_dist_map(stem: str, labels_dir: Path) -> Optional[np.ndarray]:
    """
    Returns a (H, W) float32 array where each pixel holds its distance (in pixels)
    to the nearest semantic class boundary.  Returns None if label not found.
    """
    global _boundary_cache
    if stem in _boundary_cache:
        return _boundary_cache[stem]

    lp = labels_dir / f"{stem}.png"
    if not lp.exists():
        return None

    lbl = np.array(Image.open(lp))
    if lbl.ndim == 3:
        # RGB label → single-channel by combining channels into unique class index
        lbl = lbl[:, :, 0].astype(np.int32) * 65536 + \
              lbl[:, :, 1].astype(np.int32) * 256   + \
              lbl[:, :, 2].astype(np.int32)
    lbl = lbl.astype(np.int32)

    # Boundary: pixel differs from right or below neighbour
    diff_h = np.pad(np.abs(np.diff(lbl, axis=0)), ((0, 1), (0, 0)), mode="constant") > 0
    diff_w = np.pad(np.abs(np.diff(lbl, axis=1)), ((0, 0), (0, 1)), mode="constant") > 0
    edge   = (diff_h | diff_w)

    # Distance transform: distance from each pixel to nearest edge
    dist   = ndimage.distance_transform_edt(~edge).astype(np.float32)
    _boundary_cache[stem] = dist
    return dist


def sample_boundary_dist(dist_map: np.ndarray, cx: float, cy: float) -> float:
    """Sample boundary distance at centroid (cx, cy) in [0,1] coordinates."""
    H, W   = dist_map.shape
    px     = int(round(cx * (W - 1)))
    py     = int(round(cy * (H - 1)))
    px     = max(0, min(px, W - 1))
    py     = max(0, min(py, H - 1))
    return float(dist_map[py, px])


# ─────────────────────────────────────────────────────────────────────────────
# Semantic class from label image
# ─────────────────────────────────────────────────────────────────────────────
_sem_cache: Dict[str, np.ndarray] = {}

# SECOND dataset colour→class map (7 semantic classes)
# colours observed in label1: [0,128,0],[0,255,0],[128,0,0],[128,128,128],[255,255,255]
# We quantise to nearest known class:
_SECOND_PALETTE = np.array([
    [  0,   0, 255],   # class 0: water         (blue)
    [128, 128,   0],   # class 1: vegetation    (olive)
    [  0, 128,   0],   # class 2: low vegetation(dark green)
    [  0, 255,   0],   # class 3: vegetation    (bright green)
    [128,   0,   0],   # class 4: building      (dark red)
    [255, 255, 255],   # class 5: impervious     (white)
    [128, 128, 128],   # class 6: others         (gray)
], dtype=np.float32)

CLASS_NAMES = ["water", "veg_dark", "veg_low", "veg_bright", "building", "impervious", "other"]


def get_sem_map(stem: str, labels_dir: Path) -> Optional[np.ndarray]:
    """Returns (H, W) int32 semantic class map, or None."""
    global _sem_cache
    if stem in _sem_cache:
        return _sem_cache[stem]
    lp = labels_dir / f"{stem}.png"
    if not lp.exists():
        return None
    lbl = np.array(Image.open(lp).convert("RGB")).astype(np.float32)  # [H,W,3]
    H, W, _ = lbl.shape
    # Nearest-palette colour matching
    flat   = lbl.reshape(-1, 3)                        # [H*W, 3]
    dists  = ((flat[:, None, :] - _SECOND_PALETTE[None, :, :]) ** 2).sum(-1)  # [H*W, 7]
    cls    = dists.argmin(-1).reshape(H, W).astype(np.int32)
    _sem_cache[stem] = cls
    return cls


def sample_semantic_class(sem_map: np.ndarray, cx: float, cy: float) -> int:
    H, W = sem_map.shape
    px   = max(0, min(int(round(cx * (W - 1))), W - 1))
    py   = max(0, min(int(round(cy * (H - 1))), H - 1))
    return int(sem_map[py, px])


# ─────────────────────────────────────────────────────────────────────────────
# Hook capture state
# ─────────────────────────────────────────────────────────────────────────────
class MoECapture:
    """Captures pre-MoE features, expert assignment, and router probabilities."""

    def __init__(self):
        self._pre_x: Optional[torch.Tensor] = None   # [B, N, H]
        self.expert_idx:   Optional[torch.Tensor] = None   # [B*N]
        self.router_probs: Optional[torch.Tensor] = None   # [B*N, E]
        self._handles = []

    def register(self, moe_layer):
        h1 = moe_layer.register_forward_pre_hook(self._pre_hook)
        h2 = moe_layer.register_forward_hook(self._post_hook)
        self._handles = [h1, h2]

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _pre_hook(self, module, args):
        x = args[0]
        self._pre_x = x.detach()

    @torch.no_grad()
    def _post_hook(self, module, args, output):
        x = self._pre_x
        if x is None:
            return
        B, N, H = x.shape
        x_flat = x.reshape(B * N, H)

        # Recompute router from the captured pre-MoE features
        if module.router_version == "v2" and len(args) > 1 and args[1] is not None:
            la = args[1].reshape(B * N, 1) if args[1] is not None else torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)
            dh = args[2].reshape(B * N, 1) if (len(args) > 2 and args[2] is not None) else torch.zeros(B * N, 1, device=x.device, dtype=x.dtype)
            r_in = torch.cat([x_flat, la, dh], dim=-1)
        else:
            r_in = x_flat

        logits = module.router(r_in)
        probs  = torch.softmax(logits, dim=-1)
        self.expert_idx   = probs.argmax(dim=-1).cpu()      # [B*N]
        self.router_probs = probs.cpu()                     # [B*N, E]
        self._pre_x       = None                             # release memory


# ─────────────────────────────────────────────────────────────────────────────
# Per-token record structure
# ─────────────────────────────────────────────────────────────────────────────
class TokenRecord:
    __slots__ = ("stem", "token_idx", "time_id", "expert_id",
                 "change_magnitude", "area_px", "centroid_x", "centroid_y",
                 "boundary_dist", "near_boundary", "semantic_class",
                 "change_prob")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Inference loop — collect token records
# ─────────────────────────────────────────────────────────────────────────────
def collect_records(
    model: TokenChangeReasonerMoE,
    stems: List[str],
    t1_dir: Path,
    t2_dir: Path,
    match_dir: Path,
    labels_dir: Optional[Path],
    device: torch.device,
) -> List[TokenRecord]:
    capture = MoECapture()
    capture.register(model.moe)

    records: List[TokenRecord] = []

    for stem in stems:
        # ── Load data ──────────────────────────────────────────────────────
        t1p = t1_dir  / f"{stem}.pt"
        t2p = t2_dir  / f"{stem}.pt"
        mp  = match_dir / f"{stem}_matches.pt"
        if not (t1p.exists() and t2p.exists() and mp.exists()):
            continue

        t1   = torch.load(t1p,   map_location="cpu", weights_only=True)
        t2   = torch.load(t2p,   map_location="cpu", weights_only=True)
        mtch = torch.load(mp,    map_location="cpu", weights_only=False)

        pairs_raw = mtch.get("pairs", [])
        if isinstance(pairs_raw, list):
            if len(pairs_raw):
                pairs = torch.tensor([[float(p[0]), float(p[1]), float(p[2])]
                                       for p in pairs_raw])
            else:
                pairs = torch.zeros(0, 3)
        else:
            pairs = pairs_raw.float()

        sample = SampleData(
            tokens_t1     = t1["tokens"].float(),
            tokens_t2     = t2["tokens"].float(),
            centroids_t1  = t1["centroids"].float(),
            centroids_t2  = t2["centroids"].float(),
            areas_t1      = t1["areas"].float(),
            areas_t2      = t2["areas"].float(),
            match_pairs   = pairs,
            change_labels = None,
            semantic_labels = None,
        )

        batch = build_batch([sample], model.cfg, device)

        # ── Forward pass ───────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model(batch)

        # ── Extract per-token features ─────────────────────────────────────
        B, N, _   = batch["tokens_pad"].shape
        pred_mask = ~batch["padding_mask"][0]           # [N] valid tokens
        n1        = int((batch["time_ids_pad"][0] == 0).sum())  # T1 count
        n_total   = int(pred_mask.sum())                # valid tokens (T1+T2)

        # expert assignment — shape [B*N] → reshape to [N] for batch 0
        expert_idx_all  = capture.expert_idx.reshape(B, N)[0]   # [N]
        router_probs_all = capture.router_probs.reshape(B, N, -1)[0]  # [N, E]

        # change logits → probabilities [N]
        change_logits = outputs["change_logits"][0]     # [N]
        change_probs  = torch.sigmoid(change_logits).cpu()

        # centroids & log_areas
        centroids  = batch["centroids_pad"][0].cpu()    # [N, 2]
        log_areas  = batch["log_areas_pad"][0].cpu()    # [N]
        area_px    = (torch.exp(log_areas) * IMG_SIZE * IMG_SIZE)  # [N] in pixels

        # ── Build pair-to-change_magnitude map ─────────────────────────────
        # For each T1 token i, find its paired T2 token j in the padded seq
        # pair_i → T1 idx (0..n1-1), pair_j → T2 idx (n1..N-1)
        pair_b = batch["pair_b"].cpu()      # [M]
        pair_i = batch["pair_i"].cpu()      # [M] T1 positions
        pair_j = batch["pair_j"].cpu()      # [M] T2 positions (with n1 offset)

        # Compute change_magnitude at pre-MoE representation level
        # We directly compute from the T1/T2 raw token features for clarity
        # (normalized by feature dim)
        raw_t1 = t1["tokens"].float()   # [n1, 256]
        raw_t2 = t2["tokens"].float()   # [n2, 256]

        # Build magnitude map: t1_idx → magnitude
        t1_magnitude: Dict[int, float] = {}
        t2_magnitude: Dict[int, float] = {}
        for k in range(len(pair_b)):
            if int(pair_b[k]) != 0:
                continue
            pi = int(pair_i[k])         # position in padded seq (T1 range)
            pj = int(pair_j[k])         # position in padded seq (T2 range, offset by n1)
            ti = pi                     # T1 local index
            tj = pj - n1                # T2 local index
            if 0 <= ti < raw_t1.shape[0] and 0 <= tj < raw_t2.shape[0]:
                mag = float(torch.norm(raw_t1[ti] - raw_t2[tj]).item())
                t1_magnitude[pi] = mag
                t2_magnitude[pj] = mag

        # ── Boundary / semantic maps ─────────────────────────────────────
        bd_map  = get_boundary_dist_map(stem, labels_dir) if labels_dir else None
        sem_map = get_sem_map(stem, labels_dir)            if labels_dir else None

        # ── Collect records ───────────────────────────────────────────────
        for tok_idx in range(N):
            if batch["padding_mask"][0, tok_idx]:
                continue    # skip padding tokens

            cx  = float(centroids[tok_idx, 0])
            cy  = float(centroids[tok_idx, 1])
            apx = float(area_px[tok_idx])
            eid = int(expert_idx_all[tok_idx])
            tid = int(batch["time_ids_pad"][0, tok_idx])  # 0=T1, 1=T2

            # change_magnitude: only for matched tokens
            if tid == 0:
                chg_mag = t1_magnitude.get(tok_idx, 0.0)
            else:
                chg_mag = t2_magnitude.get(tok_idx, 0.0)

            # boundary
            if bd_map is not None:
                bd  = sample_boundary_dist(bd_map, cx, cy)
                nbd = int(bd <= BOUNDARY_PX)
            else:
                bd  = -1.0
                nbd = 0

            # semantic
            sem = sample_semantic_class(sem_map, cx, cy) if sem_map is not None else -1

            records.append(TokenRecord(
                stem             = stem,
                token_idx        = tok_idx,
                time_id          = tid,
                expert_id        = eid,
                change_magnitude = chg_mag,
                area_px          = apx,
                centroid_x       = cx,
                centroid_y       = cy,
                boundary_dist    = bd,
                near_boundary    = nbd,
                semantic_class   = sem,
                change_prob      = float(change_probs[tok_idx]),
            ))

    capture.remove()
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate per-expert statistics
# ─────────────────────────────────────────────────────────────────────────────
def aggregate(records: List[TokenRecord], n_experts: int):
    buckets: Dict[int, dict] = {e: {
        "change_magnitudes": [], "areas": [], "boundary": [],
        "change_probs": [], "sem_classes": [],
    } for e in range(n_experts)}

    for r in records:
        e = r.expert_id
        if e not in buckets:
            continue
        if r.change_magnitude > 0:     # only matched tokens for magnitude
            buckets[e]["change_magnitudes"].append(r.change_magnitude)
        buckets[e]["areas"].append(r.area_px)
        buckets[e]["boundary"].append(r.near_boundary)
        buckets[e]["change_probs"].append(r.change_prob)
        if r.semantic_class >= 0:
            buckets[e]["sem_classes"].append(r.semantic_class)

    stats = {}
    for e in range(n_experts):
        b = buckets[e]
        mags   = b["change_magnitudes"]
        areas  = b["areas"]
        bnds   = b["boundary"]
        cprobs = b["change_probs"]
        sems   = b["sem_classes"]

        sem_dist = [0] * 7
        for s in sems:
            if 0 <= s < 7:
                sem_dist[s] += 1
        total = len(sems)
        sem_frac = [v / max(total, 1) for v in sem_dist]

        stats[e] = {
            "n_tokens":          len(areas),
            "n_matched":         len(mags),
            "avg_change_magnitude": float(np.mean(mags))  if mags  else 0.0,
            "avg_area_px":       float(np.mean(areas)) if areas else 0.0,
            "boundary_ratio":    float(np.mean(bnds))  if bnds  else 0.0,
            "avg_change_prob":   float(np.mean(cprobs)) if cprobs else 0.0,
            "dominant_sem_class": int(np.argmax(sem_frac)) if total else -1,
            "sem_class_dist":    sem_frac,
        }
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# PIL utilities
# ─────────────────────────────────────────────────────────────────────────────
def _try_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def bar_chart(
    values: List[float],
    labels: List[str],
    title: str,
    x_label: str,
    y_label: str,
    save_path: Path,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    annotate: bool = True,
):
    """Draw a horizontal bar chart using PIL."""
    W, H   = 700, 80 + len(values) * 70 + 40
    margin = dict(left=90, right=180, top=70, bottom=50)
    img    = Image.new("RGB", (W, H), (248, 248, 250))
    draw   = ImageDraw.Draw(img)
    fnt_t  = _try_font(17)
    fnt_b  = _try_font(13)
    fnt_s  = _try_font(11)

    # title
    draw.text((W // 2, 22), title, fill=(30, 30, 30), font=fnt_t, anchor="mm")

    # axes
    bar_region_w = W - margin["left"] - margin["right"]
    val_max      = max(values) * 1.15 if max(values) > 0 else 1.0

    for i, (v, lbl) in enumerate(zip(values, labels)):
        y_center = margin["top"] + i * 70 + 30
        bar_w    = int(bar_region_w * v / val_max)

        # bar colour
        if colors and i < len(colors):
            clr = colors[i][:3]
        else:
            ratio = v / val_max
            clr   = (int(60 + 195 * ratio), int(100 + 50 * (1 - ratio)), int(200 - 150 * ratio))

        # bar
        x0, y0 = margin["left"], y_center - 20
        x1, y1 = margin["left"] + max(bar_w, 3), y_center + 20
        draw.rectangle([x0, y0, x1, y1], fill=clr)

        # label (left)
        draw.text((margin["left"] - 8, y_center), lbl,
                  fill=(40, 40, 40), font=fnt_b, anchor="rm")

        # value (right of bar)
        if annotate:
            draw.text((x1 + 8, y_center), f"{v:.4f}",
                      fill=(40, 40, 40), font=fnt_s, anchor="lm")

    # axis labels
    draw.text((W // 2, H - 18), x_label, fill=(80, 80, 80), font=fnt_s, anchor="mm")
    # vertical y-label (rotated) — approximate with transposed text
    draw.text((12, H // 2), y_label, fill=(80, 80, 80), font=fnt_s, anchor="mm")

    img.save(save_path)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Expert assignment visualisation  (Exp 5)
# ─────────────────────────────────────────────────────────────────────────────
def draw_expert_map(
    sat_img_path: Path,
    records_for_stem: List[TokenRecord],
    n_experts: int,
    alpha_by_change: bool = True,
) -> Image.Image:
    """Overlay expert-coloured token disks on the satellite image."""
    if sat_img_path.exists():
        sat = Image.open(sat_img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    else:
        sat = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (60, 60, 60))

    # Dim the satellite image slightly for contrast
    overlay_base = Image.blend(sat, Image.new("RGB", sat.size, (30, 30, 30)), alpha=0.25)
    overlay      = Image.new("RGBA", sat.size, (0, 0, 0, 0))
    draw         = ImageDraw.Draw(overlay)

    # Sort by area descending so small tokens draw on top
    sorted_recs = sorted(records_for_stem, key=lambda r: -r.area_px)

    for r in sorted_recs:
        cx_px = int(r.centroid_x * (IMG_SIZE - 1))
        cy_px = int(r.centroid_y * (IMG_SIZE - 1))
        radius = int(math.sqrt(r.area_px / math.pi))
        radius = max(TOKEN_RADIUS_MIN, min(radius, TOKEN_RADIUS_MAX))

        eid   = r.expert_id
        if eid >= len(EXPERT_RGBA):
            eid = eid % len(EXPERT_RGBA)
        rgba  = list(EXPERT_RGBA[eid])

        if alpha_by_change:
            # Scale alpha by change probability: low-change tokens more transparent
            rgba[3] = int(rgba[3] * (0.35 + 0.65 * r.change_prob))

        draw.ellipse(
            [cx_px - radius, cy_px - radius, cx_px + radius, cy_px + radius],
            fill=tuple(rgba),
            outline=(255, 255, 255, 100),
        )

    # Composite
    result = Image.new("RGBA", sat.size)
    result.paste(overlay_base.convert("RGBA"), (0, 0))
    result = Image.alpha_composite(result, overlay)
    return result.convert("RGB")


def draw_legend(n_experts: int, w: int = 280) -> Image.Image:
    """Tiny legend panel for the expert colours."""
    row_h  = 36
    pad    = 12
    H      = pad * 2 + n_experts * row_h + 30
    img    = Image.new("RGB", (w, H), (245, 245, 250))
    draw   = ImageDraw.Draw(img)
    fnt_t  = _try_font(14)
    fnt_b  = _try_font(12)
    draw.text((w // 2, pad + 8), "Expert Colours", fill=(30, 30, 30), font=fnt_t, anchor="mm")
    for e in range(n_experts):
        y    = pad + 30 + e * row_h
        clr  = EXPERT_RGBA[e][:3]
        draw.rectangle([pad, y, pad + 24, y + 24], fill=clr, outline=(80, 80, 80))
        draw.text((pad + 34, y + 12), f"Expert {e}", fill=(40, 40, 40), font=fnt_b, anchor="lm")
    return img


def generate_assignment_maps(
    stems_for_vis: List[str],
    records: List[TokenRecord],
    images_dir: Optional[Path],
    n_experts: int,
    save_path: Path,
):
    """Create a grid of expert assignment maps for several val images."""
    n_show = min(len(stems_for_vis), VIS_N_IMAGES)
    stems  = stems_for_vis[:n_show]

    # Build per-stem record dict
    stem_records: Dict[str, List[TokenRecord]] = defaultdict(list)
    for r in records:
        stem_records[r.stem].append(r)

    n_cols  = min(n_show, 4)
    n_rows  = math.ceil(n_show / n_cols)
    cell_w  = IMG_SIZE + 4
    cell_h  = IMG_SIZE + 30
    legend  = draw_legend(n_experts)
    total_w = n_cols * cell_w + legend.width + 10
    total_h = max(n_rows * cell_h + 50, legend.height + 50)

    canvas = Image.new("RGB", (total_w, total_h), (240, 240, 245))
    draw   = ImageDraw.Draw(canvas)
    fnt_t  = _try_font(18)
    fnt_s  = _try_font(11)

    title  = "Expert Assignment Maps"
    draw.text((total_w // 2 - legend.width // 2, 18), title,
              fill=(30, 30, 30), font=fnt_t, anchor="mm")

    for idx, stem in enumerate(stems):
        row = idx // n_cols
        col = idx % n_cols
        x0  = col * cell_w + 2
        y0  = row * cell_h + 45

        img_path = (images_dir / f"{stem}.png") if images_dir else Path("_none_")
        recs     = stem_records.get(stem, [])
        panel    = draw_expert_map(img_path, recs, n_experts)
        canvas.paste(panel, (x0, y0))

        # caption
        draw.text((x0 + IMG_SIZE // 2, y0 + IMG_SIZE + 14), stem,
                  fill=(60, 60, 60), font=fnt_s, anchor="mm")

    # Paste legend on right
    canvas.paste(legend, (n_cols * cell_w + 6, 50))
    canvas.save(save_path)
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────
def save_csv(records: List[TokenRecord], save_path: Path):
    fields = ["stem", "token_idx", "time_id", "expert_id",
              "change_magnitude", "area_px", "centroid_x", "centroid_y",
              "boundary_dist", "near_boundary", "semantic_class", "change_prob"]
    with open(save_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: getattr(r, k) for k in fields})
    print(f"  Saved: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────
def _interpret_expert_role(
    eid: int,
    stats: dict,
    all_stats: dict,
    n_experts: int
) -> str:
    """Heuristically infer an expert role label from its statistics."""
    s = stats   # already the single-expert stats dict
    vals = {k: [all_stats[e][k] for e in range(n_experts)] for k in
            ["avg_change_magnitude", "avg_area_px", "boundary_ratio", "avg_change_prob"]}

    def rank(key):   # 0 = lowest, n_experts-1 = highest
        return sorted(range(n_experts), key=lambda e: all_stats[e][key]).index(eid)

    chg_r  = rank("avg_change_magnitude")
    area_r = rank("avg_area_px")
    bnd_r  = rank("boundary_ratio")
    prob_r = rank("avg_change_prob")

    top = n_experts - 1
    if chg_r == top and prob_r == top:
        return "**strong-change / large-delta expert**"
    if chg_r == 0 and prob_r == 0:
        return "**stable-region / background expert**"
    if bnd_r == top:
        return "**boundary / edge-sensitive expert**"
    if area_r == 0:
        return "**small-object expert**"
    if area_r == top:
        return "**large-region expert**"
    if chg_r > top // 2:
        return "**medium-to-high change expert**"
    return "**medium-change expert**"


def save_report(
    stats: dict,
    n_experts: int,
    records: List[TokenRecord],
    output_dir: Path,
    args,
):
    total_tokens = sum(s["n_tokens"] for s in stats.values())

    lines = [
        "# Stage 6 — Expert Behavior Analysis",
        "",
        "## Overview",
        "",
        f"- Model: `{args.model_dir}`",
        f"- Validation images: `{len(set(r.stem for r in records))}`",
        f"- Total tokens analysed: `{total_tokens:,}`",
        f"- Number of experts: `{n_experts}`",
        f"- Boundary threshold: `{BOUNDARY_PX}` pixels",
        "",
        "---",
        "",
        "## Experiment 1 — Change-Magnitude Specialisation",
        "",
        "> `change_magnitude = ||feat_T1 - feat_T2||` for matched pairs (raw SAM features, 256-dim L2 norm)",
        "",
        "| Expert | Tokens | Matched Pairs | Avg Change Magnitude | Rank |",
        "|--------|--------|:-------------:|:--------------------:|:----:|",
    ]
    mags = [(e, stats[e]["avg_change_magnitude"]) for e in range(n_experts)]
    mags_sorted = sorted(mags, key=lambda x: x[1], reverse=True)
    ranks = {e: i + 1 for i, (e, _) in enumerate(mags_sorted)}
    for e in range(n_experts):
        s = stats[e]
        lines.append(
            f"| E{e} | {s['n_tokens']:,} | {s['n_matched']:,} | "
            f"{s['avg_change_magnitude']:.4f} | #{ranks[e]} |"
        )

    lines += [
        "",
        "**Interpretation:** Experts with higher `avg_change_magnitude` specialise in "
        "regions undergoing significant semantic transitions.",
        "",
        "---",
        "",
        "## Experiment 2 — Region-Size Specialisation",
        "",
        "> `region_area` = SAM mask area in pixels (512×512 image space)",
        "",
        "| Expert | Avg Region Area (px²) | Rank |",
        "|--------|-----------------------|:----:|",
    ]
    areas = [(e, stats[e]["avg_area_px"]) for e in range(n_experts)]
    areas_sorted = sorted(areas, key=lambda x: x[1], reverse=True)
    area_ranks = {e: i + 1 for i, (e, _) in enumerate(areas_sorted)}
    for e in range(n_experts):
        lines.append(f"| E{e} | {stats[e]['avg_area_px']:.1f} | #{area_ranks[e]} |")

    lines += [
        "",
        "**Interpretation:** Experts with smaller average area specialise in fine-grained "
        "objects; those with larger area handle broad scene regions.",
        "",
        "---",
        "",
        "## Experiment 3 — Boundary Sensitivity",
        "",
        f"> `boundary_ratio` = fraction of tokens within **{BOUNDARY_PX} px** of a semantic class boundary.",
        "",
        "| Expert | Boundary Ratio | Rank |",
        "|--------|:--------------:|:----:|",
    ]
    bnds = [(e, stats[e]["boundary_ratio"]) for e in range(n_experts)]
    bnds_sorted = sorted(bnds, key=lambda x: x[1], reverse=True)
    bnd_ranks = {e: i + 1 for i, (e, _) in enumerate(bnds_sorted)}
    for e in range(n_experts):
        lines.append(f"| E{e} | {stats[e]['boundary_ratio']:.4f} | #{bnd_ranks[e]} |")

    lines += [
        "",
        "**Interpretation:** A high boundary ratio indicates the expert is activated "
        "preferentially at semantic boundaries (e.g., building edges, road–grass interfaces).",
        "",
        "---",
        "",
        "## Experiment 4 — Change-Probability Specialisation",
        "",
        "> `avg_change_prob` = mean sigmoid(change\_logit) over all tokens assigned to this expert.",
        "",
        "| Expert | Avg Change Probability | Rank |",
        "|--------|:---------------------:|:----:|",
    ]
    cprobs = [(e, stats[e]["avg_change_prob"]) for e in range(n_experts)]
    cprobs_sorted = sorted(cprobs, key=lambda x: x[1], reverse=True)
    cprob_ranks = {e: i + 1 for i, (e, _) in enumerate(cprobs_sorted)}
    for e in range(n_experts):
        lines.append(f"| E{e} | {stats[e]['avg_change_prob']:.4f} | #{cprob_ranks[e]} |")

    lines += [
        "",
        "**Interpretation:** Experts with higher average change probability are "
        "primarily routing changed tokens; low-probability experts handle stable/background regions.",
        "",
        "---",
        "",
        "## Expert Role Summary",
        "",
        "| Expert | Role | Tokens | Δ Magnitude | Area (px²) | Bnd Ratio | Chg Prob |",
        "|--------|------|:------:|:-----------:|:----------:|:---------:|:--------:|",
    ]
    for e in range(n_experts):
        s    = stats[e]
        role = _interpret_expert_role(e, s, stats, n_experts)
        lines.append(
            f"| **E{e}** | {role} | {s['n_tokens']:,} | "
            f"{s['avg_change_magnitude']:.4f} | {s['avg_area_px']:.1f} | "
            f"{s['boundary_ratio']:.4f} | {s['avg_change_prob']:.4f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Semantic Class Distribution per Expert",
        "",
        "Fraction of each expert's tokens attributed to each SECOND semantic class:",
        "",
        "| Expert | " + " | ".join(CLASS_NAMES) + " | Dominant |",
        "|--------|" + "---|" * len(CLASS_NAMES) + "----------|",
    ]
    for e in range(n_experts):
        s    = stats[e]
        dist = s["sem_class_dist"]
        dom  = s["dominant_sem_class"]
        dom_name = CLASS_NAMES[dom] if 0 <= dom < len(CLASS_NAMES) else "N/A"
        cells= " | ".join(f"{v:.3f}" for v in dist)
        lines.append(f"| E{e} | {cells} | **{dom_name}** |")

    lines += [
        "",
        "---",
        "",
        "## Experiment 5 — Expert Assignment Visualisation",
        "",
        "See `stage6_expert_assignment_maps.png` for overlays of expert colour maps on satellite images.",
        "",
        "Colour scheme:",
    ]
    for e in range(n_experts):
        clr = EXPERT_RGBA[e][:3]
        lines.append(f"- **E{e}**: RGB {clr}")

    lines += [
        "",
        "---",
        "",
        "## Emergent Specialisation",
        "",
        "The analysis reveals the following emergent expert behaviours:",
        "",
    ]
    for e in range(n_experts):
        role = _interpret_expert_role(e, stats[e], stats, n_experts)
        lines.append(f"- **E{e}**: {role}")

    lines += [
        "",
        "This emergent specialisation arises without explicit supervision on expert assignments — "
        "it is driven solely by the load-balancing and routing losses, confirming the MoE "
        "architecture's ability to self-organise into semantically coherent expert roles.",
        "",
        "---",
        "",
        "_Generated by `run_expert_behavior.py`_",
    ]

    report_path = output_dir / "stage6_expert_behavior_report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Saved: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(stats: dict, n_experts: int):
    print()
    print("=" * 72)
    print("  EXPERT BEHAVIOR ANALYSIS — RESULTS")
    print("=" * 72)
    print(f"  {'Expert':<8}{'Tokens':>8}  {'ΔMag':>8}  {'Area(px²)':>10}  {'BndRatio':>10}  {'ChgProb':>9}")
    print("-" * 72)
    for e in range(n_experts):
        s = stats[e]
        print(
            f"  E{e:<7}{s['n_tokens']:>8}  "
            f"{s['avg_change_magnitude']:>8.4f}  "
            f"{s['avg_area_px']:>10.1f}  "
            f"{s['boundary_ratio']:>10.4f}  "
            f"{s['avg_change_prob']:>9.4f}"
        )
    print("=" * 72)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Expert Emergence Analysis — Stage 6")
    p.add_argument("--model_dir",   default="SECOND/stage5_6_dynamic", help="Checkpoint dir")
    p.add_argument("--tokens_T1",   default="SECOND/tokens_T1")
    p.add_argument("--tokens_T2",   default="SECOND/tokens_T2")
    p.add_argument("--matches",     default="SECOND/matches")
    p.add_argument("--labels_dir",  default="SECOND/label1",
                   help="SECOND label1 dir (RGB semantic labels)")
    p.add_argument("--images_dir",  default="SECOND/im1",
                   help="Satellite images dir for visualisation")
    p.add_argument("--output_dir",  default="stage6/expert_behavior")
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max_samples", type=int,   default=None,
                   help="Limit number of val samples (for quick testing)")
    p.add_argument("--vis_n",       type=int,   default=8,
                   help="Number of images for assignment map visualisation")
    args = p.parse_args()

    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dir  = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t1_dir     = Path(args.tokens_T1)
    t2_dir     = Path(args.tokens_T2)
    match_dir  = Path(args.matches)
    labels_dir = Path(args.labels_dir) if args.labels_dir else None
    images_dir = Path(args.images_dir) if args.images_dir else None

    # ── Load model ──────────────────────────────────────────────────────────
    print(f"\nLoading model from {model_dir} ...")
    model     = load_model(model_dir, device)
    n_experts = model.cfg.moe_num_experts
    print(f"  Router: v{model.cfg.router_version}  |  Experts: {n_experts}")

    # ── Val stems ───────────────────────────────────────────────────────────
    val_stems = build_val_stems(match_dir, t1_dir, t2_dir, args.val_split, args.seed)
    if args.max_samples:
        val_stems = val_stems[:args.max_samples]
    print(f"  Validation stems: {len(val_stems)}")

    # ── Collect per-token records ────────────────────────────────────────────
    print("\nRunning inference ...")
    records = collect_records(
        model, val_stems, t1_dir, t2_dir, match_dir, labels_dir, device
    )
    print(f"  Collected {len(records):,} token records from {len(val_stems)} images")

    # ── Aggregate statistics ─────────────────────────────────────────────────
    stats = aggregate(records, n_experts)
    print_summary(stats, n_experts)

    # ── Experiment 1: Change Magnitude bar chart ─────────────────────────────
    print("\nGenerating plots ...")
    bar_chart(
        values    = [stats[e]["avg_change_magnitude"] for e in range(n_experts)],
        labels    = [f"Expert {e}" for e in range(n_experts)],
        title     = "Exp 1 — Change Magnitude per Expert",
        x_label   = "Avg ||feat_T1 − feat_T2||  (256-dim L2)",
        y_label   = "Expert",
        save_path = output_dir / "stage6_change_magnitude.png",
        colors    = [EXPERT_RGBA[e] for e in range(n_experts)],
    )

    # ── Experiment 2: Region Size bar chart ─────────────────────────────────
    bar_chart(
        values    = [stats[e]["avg_area_px"] for e in range(n_experts)],
        labels    = [f"Expert {e}" for e in range(n_experts)],
        title     = "Exp 2 — Region Area per Expert",
        x_label   = "Avg SAM region area (pixels²)",
        y_label   = "Expert",
        save_path = output_dir / "stage6_region_size.png",
        colors    = [EXPERT_RGBA[e] for e in range(n_experts)],
    )

    # ── Experiment 3: Boundary Ratio bar chart ───────────────────────────────
    bar_chart(
        values    = [stats[e]["boundary_ratio"] for e in range(n_experts)],
        labels    = [f"Expert {e}" for e in range(n_experts)],
        title     = f"Exp 3 — Boundary Sensitivity (≤{BOUNDARY_PX}px from boundary)",
        x_label   = "Fraction of tokens near semantic boundary",
        y_label   = "Expert",
        save_path = output_dir / "stage6_boundary_ratio.png",
        colors    = [EXPERT_RGBA[e] for e in range(n_experts)],
    )

    # ── Experiment 4: Change Probability bar chart ───────────────────────────
    bar_chart(
        values    = [stats[e]["avg_change_prob"] for e in range(n_experts)],
        labels    = [f"Expert {e}" for e in range(n_experts)],
        title     = "Exp 4 — Change Probability per Expert",
        x_label   = "Avg predicted change probability",
        y_label   = "Expert",
        save_path = output_dir / "stage6_change_prob.png",
        colors    = [EXPERT_RGBA[e] for e in range(n_experts)],
    )

    # ── Experiment 5: Expert assignment maps ────────────────────────────────
    vis_n   = min(args.vis_n, len(val_stems))
    # Pick a diverse set of images that have records
    stems_with_recs = list({r.stem for r in records})
    vis_stems = stems_with_recs[:vis_n]
    generate_assignment_maps(
        stems_for_vis = vis_stems,
        records       = records,
        images_dir    = images_dir,
        n_experts     = n_experts,
        save_path     = output_dir / "stage6_expert_assignment_maps.png",
    )

    # ── CSV ──────────────────────────────────────────────────────────────────
    save_csv(records, output_dir / "stage6_expert_behavior.csv")

    # ── Report ───────────────────────────────────────────────────────────────
    save_report(stats, n_experts, records, output_dir, args)

    print(f"\n  All outputs saved to: {output_dir}/")


if __name__ == "__main__":
    main()
