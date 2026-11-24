"""Shared pieces for the Phase 1 · Prototype 1 study. See ../PROTOCOL_M2_PROTOTYPE.md.

Contents:
- QuebecTrees crop iteration: train = 148 tiles × 4 quadrants; validation = 139 crops; TEST is never touched;
- the frozen Mask R-CNN s42 (validation-selected checkpoint), with a forward pass that returns candidate masks and FPN
  level '0' together;
- RoI geometry and mask resampling consistent with roi_align(aligned=True) on integer boxes;
- candidate labels.
"""
import json, sys
from pathlib import Path
import numpy as np
import cv2
import torch
from pycocotools import mask as mu

REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
PREP = REPO / "data/itc_benchmarks/canopyrs_quebectrees/prepared"
CB = REPO / "crown_segmentation_research/clean_benchmark"
HERE = Path(__file__).resolve().parent; STUDY = HERE.parent
Q = 1666                     # quadrant size used by the IEO predictions (train tiles are 3,333 px)
SCORE_CUT = 0.45             # IEO validation F1@0.5 cut for mrcnn_s42 (research/ITC_IDENTITY_ERROR_ORIGIN_V1_20261119/results.json)
R = 32                       # RoI grid
EXPAND, MIN_SIDE = 1.25, 24  # RoI = tight mask box enlarged 1.25× about its centre, side >= 24 px
FEAT_STRIDE = 4              # FPN level '0'


def crops(split):
    """Yield (crop_id, image, valid, label) for split in {'train', 'validation'}."""
    assert split in ("train", "validation")
    coco = json.load(open(PREP / f"{split}_coco.json"))
    for info in coco["images"]:
        z = np.load(PREP / split / f"{Path(info['file_name']).stem}.npz")
        if split == "validation":
            yield f"v{info['id']}", z["image"], z["valid"], z["label"]
        else:
            for qy in (0, 1):
                for qx in (0, 1):
                    sl = (slice(qy * Q, (qy + 1) * Q), slice(qx * Q, (qx + 1) * Q))
                    yield f"t{info['id']}_{qy}{qx}", z["image"][sl], z["valid"][sl], z["label"][sl]


def tile_names(split):
    return sorted(Path(i["file_name"]).stem for i in json.load(open(PREP / f"{split}_coco.json"))["images"])


def load_maskrcnn(seed=42):
    from crown_segmentation_research.clean_benchmark.train_maskrcnn import build_model
    sel = json.load(open(CB / f"results/mrcnn_s{seed}_val_selection.json"))
    ck = torch.load(REPO / sel["val_at_choice"]["ckpt"], map_location="cpu", weights_only=False)
    m = build_model(); m.load_state_dict(ck["model_state_dict"]); m = m.cuda().eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m, sel["val_at_choice"]["ckpt"]


@torch.no_grad()
def maskrcnn_forward(model, img, chunk=50):
    """Same detections as evaluate.maskrcnn_predict (no test-time rescaling).
    Returns (list of (bool mask HxW, score) with score >= SCORE_CUT, FPN '0' feature tensor [1,256,h,w])."""
    from torchvision.models.detection.roi_heads import paste_masks_in_image
    H, W = img.shape[:2]
    model.transform.min_size, model.transform.max_size = (min(H, W),), max(H, W)
    x = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.0).cuda()
    images, _ = model.transform([x])
    feats = model.backbone(images.tensors)
    props, _ = model.rpn(images, feats)
    d = model.roi_heads(feats, props, images.image_sizes)[0][0]
    keep = d["scores"] >= SCORE_CUT
    boxes, scores, probs = d["boxes"][keep], d["scores"][keep], d["masks"][keep]
    out = []
    for i in range(0, len(boxes), chunk):
        ms = (paste_masks_in_image(probs[i:i + chunk], boxes[i:i + chunk], (H, W))[:, 0] > 0.5).cpu().numpy()
        out += [(m, float(s)) for m, s in zip(ms, scores[i:i + chunk].cpu())]
    return out, feats["0"]


def roi_box(mask):
    """Integer RoI [x0, y0, x1, y1) (may extend outside the image) from a boolean mask."""
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max() + 1) / 2, (xs.min() + xs.max() + 1) / 2
    h = max((ys.max() + 1 - ys.min()) * EXPAND, MIN_SIDE); w = max((xs.max() + 1 - xs.min()) * EXPAND, MIN_SIDE)
    return np.array([int(np.floor(cx - w / 2)), int(np.floor(cy - h / 2)), int(np.ceil(cx + w / 2)), int(np.ceil(cy + h / 2))])


def crop_resample(arr, box, out=R):
    """Area-average arr[y0:y1, x0:x1] (zero-padded outside the image) to out×out; float32 in [0, 1] for masks.
    Equivalent to roi_align(aligned=True) bins on the integer box."""
    x0, y0, x1, y1 = box; H, W = arr.shape[:2]
    pad = np.zeros((y1 - y0, x1 - x0) + arr.shape[2:], np.float32)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
    if sy1 > sy0 and sx1 > sx0: pad[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = arr[sy0:sy1, sx0:sx1]
    return cv2.resize(pad, (out, out), interpolation=cv2.INTER_AREA)


def roi_features(feat, boxes):
    """feat [1,256,h,w] (stride 4) -> [n,256,R,R] fp16 via roi_align(aligned=True)."""
    from torchvision.ops import roi_align
    if len(boxes) == 0: return torch.zeros((0, feat.shape[1], R, R), dtype=torch.float16)
    b = torch.as_tensor(np.asarray(boxes, np.float32), device=feat.device)
    b = torch.cat([torch.zeros((len(b), 1), device=feat.device), b], 1)
    return roi_align(feat, b, output_size=R, spatial_scale=1 / FEAT_STRIDE, sampling_ratio=2, aligned=True).half().cpu()


def paste(prob_or_bool, box, H, W, valid=None):
    """Upsample an R×R child (float probability or bool) to its RoI and paste into an H×W canvas (bool)."""
    x0, y0, x1, y1 = box
    a = cv2.resize(prob_or_bool.astype(np.float32), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR) > 0.5
    out = np.zeros((H, W), bool)
    sy0, sy1, sx0, sx1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
    if sy1 > sy0 and sx1 > sx0: out[sy0:sy1, sx0:sx1] = a[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
    return out & valid if valid is not None else out


def candidate_labels(gt, masks):
    """For each candidate mask: (k = |G(P)|, ids of the covered GT sorted by |g ∩ P| descending)."""
    ids = [int(g) for g in np.unique(gt) if g]
    if not ids or not masks: return [(0, [])] * len(masks)
    grle = [mu.encode(np.asfortranarray((gt == g).astype(np.uint8))) for g in ids]
    prle = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
    cov = mu.iou(grle, prle, [1] * len(prle))                         # |g ∩ P| / |g|
    area = np.array([mu.area(r) for r in grle], np.float64)
    out = []
    for j in range(len(masks)):
        c = np.where(cov[:, j] >= 0.5)[0]
        c = c[np.argsort(-(cov[c, j] * area[c]))]                     # by |g ∩ P|
        out.append((len(c), [ids[i] for i in c]))
    return out
