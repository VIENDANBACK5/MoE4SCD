"""Diagnostic script to inspect ground truth vs predicted instances for sample 0 in validation set."""
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
from crown_segmentation_research.methods.canopy_watershed.decode import decode_canopy_watershed
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

    surf_t = item["surface"].squeeze().numpy()
    bound_t = item["boundary"].squeeze().numpy()
    canopy_t = item["canopy"].squeeze().numpy()
    gt_inst = item["instance_label"].squeeze().numpy()

    print(f"Sample 0 Info:")
    print(f"Image shape: {img.shape}")
    print(f"Surface Pred min={surf_p.min():.3f}, max={surf_p.max():.3f}, mean={surf_p.mean():.3f}")
    print(f"Surface Target min={surf_t.min():.3f}, max={surf_t.max():.3f}, mean={surf_t.mean():.3f}")
    print(f"Canopy Pred min={canopy_p.min():.3f}, max={canopy_p.max():.3f}, mean={canopy_p.mean():.3f}")
    print(f"Canopy Target min={canopy_t.min():.3f}, max={canopy_t.max():.3f}, mean={canopy_t.mean():.3f}")

    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]
    print(f"Ground truth instances count: {len(gt_ids)}")
    gt_areas = [int((gt_inst == gid).sum()) for gid in gt_ids]
    print(f"Ground truth instance areas: {gt_areas}")

    # Test decoding with different min_distances
    for dist in [10, 25, 50, 75, 100]:
        markers, insts = decode_canopy_watershed(
            surface=surf_p,
            boundary=bound_p,
            canopy=canopy_p,
            kernel_size=15,
            min_apex_val=0.35,
            min_canopy_val=0.40,
            pers_thresh=0.10,
            min_distance=dist,
            bound_weight=1.5,
            min_area=100,
        )
        print(f"\n--- min_distance={dist} ---")
        print(f"Predicted instances count: {len(insts)}")
        pred_areas = [p["area"] for p in insts]
        print(f"Pred areas (first 10): {pred_areas[:10]}")

        # Compute IoU with each GT
        if len(insts) > 0 and len(gt_ids) > 0:
            ious = []
            for gid in gt_ids:
                gt_mask = (gt_inst == gid)
                best_iou = 0.0
                for p in insts:
                    inter = np.logical_and(p["mask"], gt_mask).sum()
                    union = np.logical_or(p["mask"], gt_mask).sum()
                    iou = inter / union if union > 0 else 0
                    if iou > best_iou:
                        best_iou = iou
                ious.append(best_iou)
            print(f"Max IoU per GT crown: {[round(x, 3) for x in ious]}")
            print(f"Mean Max IoU: {np.mean(ious):.3f}, IoU >= 0.5: {sum(x >= 0.5 for x in ious)} / {len(gt_ids)}")


if __name__ == "__main__":
    main()
