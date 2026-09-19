"""Test surface thresholding inside watershed basins to see how IoU changes."""
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
import sys
from pathlib import Path
for _p in Path(__file__).resolve().parents:
    if (_p / "crown_segmentation_research").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break
import cv2
import numpy as np
import torch

from crown_segmentation_research.methods.canopy_watershed.dataset import BAMCanopyWatershedDataset
from crown_segmentation_research.methods.canopy_watershed.decode import extract_persistent_apices
from crown_segmentation_research.methods.canopy_watershed.model import CanopyWatershedNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CanopyWatershedNet(pretrained=False).to(device)
    ckpt = torch.load("DeadTrees/experiments/canopy_watershed/best_canopy_watershed.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    val_dataset = BAMCanopyWatershedDataset(split="eval", crop_size=1024, augment=False, compute_targets=True)
    item = val_dataset[0]
    img = item["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img)

    surf_p = preds["surface"].squeeze().float().cpu().numpy()
    bound_p = preds["boundary"].squeeze().float().cpu().numpy()
    canopy_p = preds["canopy"].squeeze().float().cpu().numpy()
    gt_inst = item["instance_label"].squeeze().numpy()
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]

    peaks = extract_persistent_apices(
        surface=surf_p,
        canopy=canopy_p,
        kernel_size=35,
        min_apex_val=0.35,
        min_canopy_val=0.40,
        pers_thresh=0.15,
        min_distance=70.0,
    )
    print(f"Extracted {len(peaks)} peaks (GT has {len(gt_ids)} crowns).")

    # Watershed relief
    H, W = surf_p.shape
    markers = np.zeros((H, W), dtype=np.int32)
    markers[canopy_p < 0.40] = 1
    for i, (y, x, _) in enumerate(peaks):
        inst_id = i + 2
        markers[max(0, y - 2):min(H, y + 3), max(0, x - 2):min(W, x + 3)] = inst_id

    W_surf = (1.0 - surf_p) + 1.5 * bound_p
    W_u8 = np.clip((W_surf / 2.5) * 255.0, 0, 255).astype(np.uint8)
    W_3c = cv2.merge([W_u8, W_u8, W_u8])
    cv2.watershed(W_3c, markers)

    print("\n--- Testing Potential Surface Cutoff inside Basins ---")
    for cutoff in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        ious = []
        for gid in gt_ids:
            gt_mask = (gt_inst == gid)
            best_iou = 0.0
            for i, (y, x, s) in enumerate(peaks):
                inst_id = i + 2
                basin_mask = (markers == inst_id)
                if cutoff > 0:
                    pred_mask = basin_mask & (surf_p >= cutoff)
                else:
                    pred_mask = basin_mask

                inter = np.logical_and(pred_mask, gt_mask).sum()
                union = np.logical_or(pred_mask, gt_mask).sum()
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
            ious.append(best_iou)

        mean_iou = np.mean(ious)
        above_50 = sum(x >= 0.5 for x in ious)
        above_30 = sum(x >= 0.3 for x in ious)
        print(f"Cutoff {cutoff:4.2f} => Mean Max IoU: {mean_iou:.3f} | IoU>=0.5: {above_50:2d}/{len(gt_ids)} | IoU>=0.3: {above_30:2d}/{len(gt_ids)} | Max IoU: {max(ious):.3f}")


if __name__ == "__main__":
    main()
