"""Preview of strong-stratum (r >= 4, d = 2) absorption events on LIVECell val, CenterMask2 (cut 0.55).
Cells are drawn at random (seed 0) from BV2 and MCF7 images, 8 per type. Each crop of 160 px is shown at 2× scale:
- yellow: all GT outlines;
- red: the absorbed small cell;
- cyan: its larger GT neighbour(s);
- magenta: the prediction that covers both.
Purpose: is the larger neighbour a single cell or an annotated multi-cell blob? Coarse visual only.
Run as d3_livecell_centermask.py (system python3 + PYTHONPATH)."""
import json, sys, zipfile
from pathlib import Path
import numpy as np, cv2, torch
from pycocotools import mask as mu
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, "/home/chung/RS/envs/src/centermask2")
from absorption import neighbours  # noqa: E402
LC = Path("/home/chung/RS/data/livecell"); MD = LC / "models/anchor_free"; CUT = 0.55
OUT = HERE.parent / "previews"; OUT.mkdir(exist_ok=True)


def main():
    from centermask.config import get_cfg
    from detectron2.engine import DefaultPredictor
    cfg = get_cfg(); cfg.merge_from_file(str(MD / "livecell_config.yaml")); cfg.MODEL.WEIGHTS = str(MD / "LIVECell_anchor_free_model.pth"); cfg.freeze()
    pred = DefaultPredictor(cfg)
    d = json.load(open(LC / "annotations/livecell_coco_val.json"))
    anns = d["annotations"].values() if isinstance(d["annotations"], dict) else d["annotations"]; by = {}
    for a in anns: by.setdefault(a["image_id"], []).append(a)
    imgs = list(d["images"].values() if isinstance(d["images"], dict) else d["images"])
    zf = zipfile.ZipFile(LC / "images.zip"); names = {Path(n).name: n for n in zf.namelist() if n.endswith(".tif")}
    rng = np.random.default_rng(0)
    for ct in ("BV2", "MCF7"):
        pool = [im for im in imgs if im["file_name"].startswith(ct + "_")]; rng.shuffle(pool); tiles = []
        for im in pool:
            if len(tiles) >= 8: break
            H, W = im["height"], im["width"]; gt = np.zeros((H, W), np.int32); n = 0
            for a in by.get(im["id"], []):
                seg = a["segmentation"]; rle = mu.merge(mu.frPyObjects(seg, H, W)) if isinstance(seg, list) else seg
                m = mu.decode(rle).astype(bool)
                if m.any(): n += 1; gt[m] = n
            g = cv2.imdecode(np.frombuffer(zf.read(names[im["file_name"]]), np.uint8), cv2.IMREAD_UNCHANGED)
            with torch.no_grad(): inst = pred(np.stack([g] * 3, -1))["instances"].to("cpu")
            P = inst.pred_masks.numpy()[inst.scores.numpy() >= CUT]
            area = np.bincount(gt.ravel(), minlength=n + 1); nb = neighbours(gt, 2)
            border = set(np.unique(np.concatenate([gt[0], gt[-1], gt[:, 0], gt[:, -1]])).tolist())
            cands = []
            for s in range(1, n + 1):
                if s in border or not nb.get(s): continue
                big = [b for b in nb[s] if area[b] >= 4 * area[s]]
                if not big: continue
                ms = gt == s; iou_hit = any(((p & ms).sum() / (p | ms).sum()) >= .5 for p in P)
                if iou_hit: continue
                for p in P:
                    if (p & ms).sum() >= .5 * ms.sum() and any((p & (gt == b)).sum() >= .5 * area[b] for b in big):
                        cands.append((s, big, p)); break
            if not cands: continue
            s, big, p = cands[rng.integers(len(cands))]
            cy, cx = np.argwhere(gt == s).mean(0).astype(int); R = 80
            y0, x0 = int(np.clip(cy - R, 0, gt.shape[0] - 2 * R)), int(np.clip(cx - R, 0, gt.shape[1] - 2 * R)); sl = (slice(y0, y0 + 2 * R), slice(x0, x0 + 2 * R))
            c = cv2.cvtColor(g[sl], cv2.COLOR_GRAY2BGR); c = cv2.resize(c, (4 * R, 4 * R), interpolation=cv2.INTER_NEAREST)
            def draw(mask, col, w):
                cs, _ = cv2.findContours(cv2.resize(mask[sl].astype(np.uint8), (4 * R, 4 * R), interpolation=cv2.INTER_NEAREST), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(c, cs, -1, col, w)
            for gid in np.unique(gt[sl]):
                if gid and gid != s and gid not in big: draw(gt == gid, (0, 220, 220), 1)
            for b in big: draw(gt == b, (255, 255, 0), 2)
            draw(gt == s, (0, 0, 255), 2); draw(p, (255, 0, 255), 1)
            cv2.putText(c, f"{ct} img{im['id']} s={area[s]}px big={max(area[b] for b in big)}px", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1)
            tiles.append(c)
        rows = [np.concatenate(tiles[i:i + 4], 1) for i in range(0, len(tiles) - len(tiles) % 4, 4)]
        if rows: cv2.imwrite(str(OUT / f"livecell_strong_absorbed_{ct}.png"), np.concatenate(rows, 0)); print(ct, len(tiles), flush=True)


if __name__ == "__main__":
    main()
