"""H9 — test-time rescaling separates the scale axis from the rest of the domain shift.

The network is fully convolutional, so resizing the input by `s`, running the heads and
resizing the outputs back isolates crown scale without touching appearance. See
../protocol.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

_c = '/home/ubuntu/anaconda3/envs/cs2/lib/python3.11/site-packages'
_l = '/home/chung/.local/lib/python3.11/site-packages'
sys.path = [p for p in sys.path if p not in (_l, _c)]
sys.path.insert(0, _c); sys.path.append(_l)
sys.path.insert(0, "/home/chung/RS/Image Segmentation")

import argparse
import cv2
import numpy as np
import pandas as pd
import torch

from crown_research.src.coco_ap import APAccumulator, instance_scores_from_surface
from crown_research.src.seeding import flood, global_nms_seeds
from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet
from crown_segmentation_research.methods.tree_flow.hamiltonian_watershed import evaluate_metrics

OUT = Path("crown_research/experiments/h9-domain-shift/results")
CKPT = "DeadTrees/experiments/canopy_watershed/best_calibrated_watershed.pth"
# Each split's own decode optimum at s=1 (H8), scaled by s inside the loop.
BASE_MD = {"test1": 280.0, "test2": 160.0}
SCALES = [0.5, 0.6, 0.672, 0.8, 1.0, 1.25]


def heads_at_scale(model, img_t, s, device):
    """Run the heads on a rescaled input, return outputs at the original resolution."""
    _, _, H, W = img_t.shape
    if abs(s - 1.0) > 1e-6:
        h, w = int(round(H * s)), int(round(W * s))
        x = torch.nn.functional.interpolate(img_t, size=(h, w), mode="bilinear", align_corners=False)
    else:
        x = img_t
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        p = model(x)
    out = {}
    for k in ("surface", "boundary", "canopy"):
        v = p[k].float()
        if v.shape[-2:] != (H, W):
            v = torch.nn.functional.interpolate(v, size=(H, W), mode="bilinear", align_corners=False)
        out[k] = v.squeeze().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_crops", type=int, default=40)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0); np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanopyWatershedNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False)["model_state_dict"])
    model.eval()

    rows = []
    for split in ("test1", "test2"):
        ds = BAMCanopyWatershedDataset(split=split, crop_size=1024, augment=False, compute_targets=False)
        n = min(a.num_crops, len(ds))
        items = []
        for i in range(n):
            it = ds[i]
            raw = it["instance_label"].squeeze().numpy().astype(np.int32)
            gt = np.zeros_like(raw); gt[raw > 0] = raw[raw > 0] + 1
            if len(np.unique(raw[raw > 0])) == 0:
                continue
            items.append((it["image"].unsqueeze(0).to(device), gt))
        print(f"[H9] {split}: {len(items)} crops", flush=True)

        for s in SCALES:
            acc = APAccumulator(); f1 = []; ratios = []
            md = BASE_MD[split] * s
            for img_t, gt in items:
                h = heads_at_scale(model, img_t, s, device)
                # decode at the ORIGINAL resolution; only the suppression radius follows s,
                # because rescaling changes how many pixels a crown subtends in the heads.
                seeds = global_nms_seeds(h["surface"], h["canopy"], BASE_MD[split], 0.50)
                m, insts = flood(seeds, h["boundary"], h["canopy"])
                insts = instance_scores_from_surface(m, insts, h["surface"])
                acc.add(m, insts, gt)
                ev = evaluate_metrics(m, insts, gt)
                f1.append(ev["ap50"])
                ratios.append(len(insts) / max(1, int((np.unique(gt) >= 2).sum())))
            r = acc.ap()
            rows.append({"split": split, "scale": s, "COCO AP": r["AP"], "AP50": r["AP50"],
                         "AP75": r["AP75"], "F1@50": float(np.mean(f1)),
                         "ratio": float(np.mean(ratios))})
            print(f"  {split} s={s:<5} -> AP {r['AP']:5.2f} | AP50 {r['AP50']:5.2f} | "
                  f"F1@50 {np.mean(f1):5.2f} | ratio {np.mean(ratios):.2f}", flush=True)

    df = pd.DataFrame(rows).round(3)
    df.to_csv(OUT / "h9_rescale_sweep.csv", index=False)
    print("\n" + "=" * 96)
    print(df.to_string(index=False))
    print("=" * 96)

    t1 = df[df.split == "test1"]
    t2 = df[df.split == "test2"]
    b1 = t1.sort_values("COCO AP", ascending=False).iloc[0]
    b2 = t2.sort_values("COCO AP", ascending=False).iloc[0]
    s1 = float(t1[t1.scale == 1.0]["COCO AP"].iloc[0])
    s2 = float(t2[t2.scale == 1.0]["COCO AP"].iloc[0])
    print(f"\nP1 test1 best AP >= 13.4 : {b1['COCO AP']:.2f} (s={b1.scale}, from {s1:.2f}) -> "
          f"{'PASS' if b1['COCO AP'] >= 13.4 else 'FAIL'}")
    print(f"P2 best s in [0.54, 0.81] : s={b1.scale} -> "
          f"{'PASS' if 0.54 <= b1.scale <= 0.81 else 'FAIL'}")
    print(f"P3 test2 control flat     : best {b2['COCO AP']:.2f} (s={b2.scale}) vs s=1.0 {s2:.2f}, "
          f"delta {b2['COCO AP'] - s2:.2f} -> {'PASS' if b2['COCO AP'] - s2 <= 1.0 else 'FAIL'}")
    pd.DataFrame([{"split": "test1", "best_scale": b1.scale, "best_AP": b1["COCO AP"], "AP_at_s1": s1},
                  {"split": "test2", "best_scale": b2.scale, "best_AP": b2["COCO AP"], "AP_at_s1": s2}
                  ]).round(3).to_csv(OUT / "h9_summary.csv", index=False)


if __name__ == "__main__":
    main()
