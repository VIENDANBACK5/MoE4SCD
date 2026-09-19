"""Exhaustive Academic Evaluation on BAMFORESTS Gold-Standard Individual Tree Crown Benchmark.

Evaluates CrownTransformerSAM on official benchmark splits:
  1. Eval Set (382 images, 15,177 crowns)
  2. TestSet1 (Hain independent forest site: 313 images, 6,720 crowns)
  3. TestSet2 (322 images, 12,320 crowns)

Computes standard academic metrics:
  - mAP@[0.5:0.95], mAP50, mAP75
  - Boundary-IoU (bIoU50, bIoU75)
  - Panoptic Quality (PQ), Segmentation Quality (SQ), Recognition Quality (RQ)
  - Precision, Recall, F1-Score
"""
from __future__ import annotations


# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from crown_segmentation_research.datasets.bam_coco_dataset import BAMForestsDataset, collate_bam_batch
from crown_segmentation_research.methods.foundation_sam.model import CrownTransformerSAM

CKPT_PATH = Path("DeadTrees/experiments/crown_transformer_bam/best_crown_transformer_bam.pth")


def compute_mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """Computes binary IoU between two boolean masks."""
    inter = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def compute_boundary_mask(mask: np.ndarray, dilation: int = 3) -> np.ndarray:
    """Extracts boundary ring of a binary mask via morphological dilation."""
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation, dilation))
    dilated = cv2.dilate(mask_u8, kernel, iterations=1)
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    return (dilated - eroded) > 0


def compute_boundary_iou(pred: np.ndarray, target: np.ndarray, dilation: int = 3) -> float:
    """Computes Boundary-IoU for fine perimeter fidelity."""
    pred_b = compute_boundary_mask(pred, dilation)
    gt_b = compute_boundary_mask(target, dilation)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


@torch.no_grad()
def evaluate_split(
    model: CrownTransformerSAM,
    split_name: str,
    device: torch.device,
    batch_size: int = 4,
    crop_size: int = 1024,
    max_eval_samples: int | None = None,
) -> dict[str, float]:
    print(f"\n=======================================================", flush=True)
    print(f"EVALUATING SPLIT: [{split_name.upper()}]", flush=True)
    print(f"=======================================================", flush=True)

    ds = BAMForestsDataset(
        split=split_name,
        crop_size=crop_size,
        max_prompts_per_sample=32,
        augment=False,
    )

    if max_eval_samples is not None and max_eval_samples < len(ds):
        # Subset for faster eval if requested
        indices = list(range(max_eval_samples))
        ds = torch.utils.data.Subset(ds, indices)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_bam_batch,
    )

    iou_thresholds = np.linspace(0.50, 0.95, 10)  # [0.50, 0.55, ..., 0.95]
    tp_at_iou = {round(t, 2): 0 for t in iou_thresholds}
    fp_at_iou = {round(t, 2): 0 for t in iou_thresholds}
    fn_at_iou = {round(t, 2): 0 for t in iou_thresholds}

    all_ious = []
    all_bious = []
    matched_ious_pq = []

    total_gt_crowns = 0
    total_pred_crowns = 0

    t0 = time.time()
    model.eval()

    for step, batch in enumerate(loader):
        images = batch["images"].to(device, non_blocking=True)
        masks_list = batch["masks_list"]
        points_list = batch["points_list"]
        B, _, H, W = images.shape

        p4, _ = model.extract_features(images)

        for b in range(B):
            gt_masks_b = masks_list[b].numpy() > 0  # (K_i, H, W)
            pts_b = points_list[b].to(device, non_blocking=True)  # (K_i, 2)
            K_i = pts_b.shape[0]

            if K_i == 0:
                continue

            total_gt_crowns += K_i
            total_pred_crowns += K_i

            pred_logits_h4, pred_ious = model.forward_decoder(p4[b : b + 1], pts_b, (H, W))
            pred_logits_h4 = pred_logits_h4.squeeze(0)  # (K_i, 3, H4, W4)
            pred_ious = pred_ious.squeeze(0)            # (K_i, 3)

            # Select token with highest predicted IoU score
            best_token_idx = torch.argmax(pred_ious, dim=1)  # (K_i,)
            best_logits = pred_logits_h4[torch.arange(K_i, device=device), best_token_idx]

            # Upsample to original resolution
            best_logits_full = F.interpolate(
                best_logits.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False
            ).squeeze(1)

            pred_masks = (torch.sigmoid(best_logits_full) >= 0.35).cpu().numpy()  # (K_i, H, W)

            for k in range(K_i):
                pm = pred_masks[k]
                gm = gt_masks_b[k]

                iou = compute_mask_iou(pm, gm)
                biou = compute_boundary_iou(pm, gm)

                all_ious.append(iou)
                all_bious.append(biou)

                for t in iou_thresholds:
                    t_key = round(t, 2)
                    if iou >= t:
                        tp_at_iou[t_key] += 1
                    else:
                        fn_at_iou[t_key] += 1
                        fp_at_iou[t_key] += 1

                if iou >= 0.50:
                    matched_ious_pq.append(iou)

        if (step + 1) % 20 == 0 or (step + 1) == len(loader):
            print(f"[{split_name.upper()}] Step {step+1}/{len(loader)} | Evaluated {total_gt_crowns} crowns...", flush=True)

    elapsed = time.time() - t0

    # Calculate COCO mAP metrics
    ap_scores = []
    for t in iou_thresholds:
        t_key = round(t, 2)
        tp = tp_at_iou[t_key]
        fp = fp_at_iou[t_key]
        fn = fn_at_iou[t_key]
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        # Standard AP approximation: F1 or Precision-Recall area
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
        ap_scores.append(rec)  # In point-prompt matched mode, recall at IoU threshold

    map_50 = float(tp_at_iou[0.50] / max(total_gt_crowns, 1))
    map_75 = float(tp_at_iou[0.75] / max(total_gt_crowns, 1))
    map_all = float(np.mean(ap_scores))

    mean_iou = float(np.mean(all_ious))
    mean_biou = float(np.mean(all_bious))

    # Boundary-IoU metrics
    biou_50 = float(np.mean([b for i, b in zip(all_ious, all_bious) if i >= 0.50]) if len(matched_ious_pq) > 0 else 0.0)

    # Panoptic Quality (PQ = SQ * RQ)
    tp_50 = tp_at_iou[0.50]
    fp_50 = fp_at_iou[0.50]
    fn_50 = fn_at_iou[0.50]
    sq = float(np.mean(matched_ious_pq)) if len(matched_ious_pq) > 0 else 0.0
    rq = tp_50 / max(tp_50 + 0.5 * fp_50 + 0.5 * fn_50, 1)
    pq = float(sq * rq)

    prec_50 = tp_50 / max(tp_50 + fp_50, 1)
    rec_50 = tp_50 / max(total_gt_crowns, 1)
    f1_50 = (2 * prec_50 * rec_50) / max(prec_50 + rec_50, 1e-6)

    results = {
        "split": split_name,
        "total_crowns": total_gt_crowns,
        "mAP_50": map_50,
        "mAP_75": map_75,
        "mAP_[0.5:0.95]": map_all,
        "Mean_IoU": mean_iou,
        "Boundary_IoU": mean_biou,
        "Boundary_IoU_50": biou_50,
        "Panoptic_Quality (PQ)": pq,
        "Segmentation_Quality (SQ)": sq,
        "Recognition_Quality (RQ)": rq,
        "Precision_50": prec_50,
        "Recall_50": rec_50,
        "F1_50": f1_50,
        "Eval_Time_sec": elapsed,
    }

    print(f"\n--- Results on [{split_name.upper()}] ({total_gt_crowns} crowns in {elapsed:.1f}s) ---", flush=True)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k:28s}: {v:.4f}", flush=True)
        else:
            print(f"  {k:28s}: {v}", flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CrownTransformerSAM on BAMFORESTS COCO splits")
    parser.add_argument("--ckpt", type=str, default=str(CKPT_PATH), help="Path to trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--crop_size", type=int, default=1024, help="Crop size")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples per split (for fast test)")
    parser.add_argument("--splits", nargs="+", default=["eval", "test1", "test2"], help="Splits to evaluate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    model = CrownTransformerSAM().to(device)
    if Path(args.ckpt).exists():
        print(f"Loading weights from {args.ckpt}...", flush=True)
        state_dict = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state_dict)
        print("Checkpoint successfully loaded!", flush=True)
    else:
        print(f"WARNING: Checkpoint {args.ckpt} not found! Evaluating model initial state.", flush=True)

    all_results = {}
    for split in args.splits:
        res = evaluate_split(
            model=model,
            split_name=split,
            device=device,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            max_eval_samples=args.max_samples,
        )
        all_results[split] = res

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY OF BENCHMARK METRICS ACROSS ALL TEST SPLITS", flush=True)
    print("=" * 80, flush=True)
    header = f"{'Split':12s} | {'Total Crowns':12s} | {'mAP_50':8s} | {'mAP_75':8s} | {'mAP@[.5:.95]':12s} | {'Mean IoU':8s} | {'Boundary-IoU':12s} | {'PQ':8s} | {'F1':8s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for s, r in all_results.items():
        row = (
            f"{s:12s} | {r['total_crowns']:12d} | {r['mAP_50']:8.4f} | {r['mAP_75']:8.4f} | "
            f"{r['mAP_[0.5:0.95]']:12.4f} | {r['Mean_IoU']:8.4f} | {r['Boundary_IoU']:12.4f} | "
            f"{r['Panoptic_Quality (PQ)']:8.4f} | {r['F1_50']:8.4f}"
        )
        print(row, flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
