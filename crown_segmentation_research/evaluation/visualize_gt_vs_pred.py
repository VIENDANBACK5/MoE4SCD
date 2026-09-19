"""Visualize Ground Truth vs Predicted Crowns for diagnostic inspection."""
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
import matplotlib.pyplot as plt
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

    out_dir = Path("crown_segmentation_research/images/diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(3):
        item = val_dataset[idx]
        img_t = item["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(img_t)

        surf_p = preds["surface"].squeeze().float().cpu().numpy()
        bound_p = preds["boundary"].squeeze().float().cpu().numpy()
        canopy_p = preds["canopy"].squeeze().float().cpu().numpy()

        img_rgb = (item["image"].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        gt_inst = item["instance_label"].squeeze().numpy().astype(np.int32)

        # Decode watershed
        markers, insts = decode_canopy_watershed(
            surface=surf_p,
            boundary=bound_p,
            canopy=canopy_p,
            kernel_size=25,
            min_apex_val=0.35,
            min_canopy_val=0.40,
            pers_thresh=0.15,
            min_distance=60.0,
            bound_weight=1.5,
            min_area=500,
        )

        fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)

        # 1. RGB Image
        axes[0].imshow(img_rgb)
        axes[0].set_title(f"Sample {idx}: RGB Input (1024x1024)", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # 2. Ground Truth Instances
        gt_overlay = img_rgb.copy()
        gt_ids = np.unique(gt_inst)
        gt_ids = gt_ids[gt_ids > 0]
        np.random.seed(42)
        for gid in gt_ids:
            mask = (gt_inst == gid).astype(np.uint8)
            color = np.random.randint(50, 255, size=3).tolist()
            colored_mask = np.zeros_like(img_rgb)
            colored_mask[mask > 0] = color
            gt_overlay = cv2.addWeighted(gt_overlay, 1.0, colored_mask, 0.45, 0)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(gt_overlay, contours, -1, (255, 255, 255), 2)

        axes[1].imshow(gt_overlay)
        axes[1].set_title(f"Ground Truth ({len(gt_ids)} Crowns)", fontsize=12, fontweight="bold")
        axes[1].axis("off")

        # 3. Predicted Potential Surface + Apices
        axes[2].imshow(surf_p, cmap="inferno", vmin=0, vmax=1)
        for p in insts:
            y, x = p["apex"]
            axes[2].plot(x, y, "c*", markersize=8)
        axes[2].set_title("Neural Potential Surface U(y, x) + Apices", fontsize=12, fontweight="bold")
        axes[2].axis("off")

        # 4. Predicted Watershed Crowns
        pred_overlay = img_rgb.copy()
        for p in insts:
            mask = p["mask"].astype(np.uint8)
            color = np.random.randint(50, 255, size=3).tolist()
            colored_mask = np.zeros_like(img_rgb)
            colored_mask[mask > 0] = color
            pred_overlay = cv2.addWeighted(pred_overlay, 1.0, colored_mask, 0.45, 0)
            if "polygon" in p and len(p["polygon"]) >= 3:
                cv2.drawContours(pred_overlay, [p["polygon"]], -1, (255, 255, 255), 2)

        axes[3].imshow(pred_overlay)
        axes[3].set_title(f"Canopy Watershed Delineation ({len(insts)} Crowns)", fontsize=12, fontweight="bold")
        axes[3].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"diagnostic_sample_{idx}.png"
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
        print(f"Saved diagnostic image to {save_path}")


if __name__ == "__main__":
    main()
