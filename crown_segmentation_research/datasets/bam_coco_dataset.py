"""PyTorch Dataset for BAMFORESTS Individual Tree Crown Instance Benchmark.

Loads 2048x2048 aerial TIFF crops directly from zip archive with COCO polygon annotations,
rasterizing instance masks and providing ground-truth point prompts.
"""
from __future__ import annotations

import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ZIP_ARCHIVE = Path("/home/chung/RS/data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip")
EXTRACTED_JSON_DIR = Path("/home/chung/RS/data/itc_benchmarks/extracted/bam_coco2048")


MANIFEST_CACHE = Path("crown_segmentation_research/datasets/bam_zip_manifest.json")
_GLOBAL_NAME_TO_ZIP: dict[str, str] | None = None


def get_zip_manifest(zip_path: Path) -> dict[str, str]:
    global _GLOBAL_NAME_TO_ZIP
    if _GLOBAL_NAME_TO_ZIP is not None:
        return _GLOBAL_NAME_TO_ZIP

    if MANIFEST_CACHE.exists():
        with open(MANIFEST_CACHE, "r") as f:
            _GLOBAL_NAME_TO_ZIP = json.load(f)
            return _GLOBAL_NAME_TO_ZIP

    print(f"Generating zip manifest cache from {zip_path}...", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        tifs = [n for n in zf.namelist() if n.endswith(".tif")]
        _GLOBAL_NAME_TO_ZIP = {Path(n).name: n for n in tifs}

    MANIFEST_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_CACHE, "w") as f:
        json.dump(_GLOBAL_NAME_TO_ZIP, f)
    return _GLOBAL_NAME_TO_ZIP


class BAMForestsDataset(Dataset):
    """BAMFORESTS COCO Dataset streaming directly from Bamberg_coco2048.zip."""

    def __init__(
        self,
        split: str = "train",  # "train", "eval", "test1", "test2"
        crop_size: int = 1024,
        max_prompts_per_sample: int = 32,
        augment: bool = True,
        zip_path: Path = ZIP_ARCHIVE,
    ) -> None:
        super().__init__()
        self.split = split
        self.crop_size = crop_size
        self.max_prompts = max_prompts_per_sample
        self.augment = augment
        self.zip_path = zip_path

        # Select corresponding annotation file
        split_map = {
            "train": EXTRACTED_JSON_DIR / "instances_tree_train2023.json",
            "eval": EXTRACTED_JSON_DIR / "instances_tree_eval2023.json",
            "test1": EXTRACTED_JSON_DIR / "instances_tree_TestSet12023.json",
            "test2": EXTRACTED_JSON_DIR / "instances_tree_TestSet22023.json",
        }
        json_file = split_map[split]

        print(f"Loading {split} annotations from {json_file}...", flush=True)
        with open(json_file, "r") as f:
            coco_data = json.load(f)

        self.images = coco_data["images"]
        # Group annotations by image_id for O(1) lookup
        self.img_to_anns: dict[int, list[dict[str, Any]]] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Exact zip manifest lookup
        self._name_to_zip = get_zip_manifest(self.zip_path)

        # Filter images that exist in zip and have >= 1 tree crown
        self.images = [
            img for img in self.images
            if img["file_name"] in self._name_to_zip and len(self.img_to_anns.get(img["id"], [])) > 0
        ]
        print(f"[{split.upper()}] Initialized: {len(self.images)} images with valid tree crown annotations.", flush=True)

        # Thread-local zip handle
        self._zipfile = None

    def _get_zipfile(self) -> zipfile.ZipFile:
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self.zip_path, "r")
        return self._zipfile

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        img_info = self.images[idx]
        img_id = img_info["id"]
        base_name = img_info["file_name"]
        anns = self.img_to_anns.get(img_id, [])

        zf = self._get_zipfile()
        zip_entry = self._name_to_zip[base_name]
        data = zf.read(zip_entry)

        # Decode image
        pil_img = Image.open(io.BytesIO(data))
        img_arr = np.array(pil_img)[:, :, :3]  # (H, W, 3) RGB uint8
        H_orig, W_orig = img_arr.shape[:2]

        # Rasterize tree crown masks & extract prompt points
        crown_masks_list = []
        prompt_pts_list = []

        for ann in anns:
            seg = ann.get("segmentation", [])
            if not seg:
                continue

            # Rasterize binary mask for this tree crown
            mask = np.zeros((H_orig, W_orig), dtype=np.uint8)
            all_pts = []
            for poly in seg:
                pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 1)
                    all_pts.append(pts)

            if mask.sum() < 15 or len(all_pts) == 0:
                continue

            # Center prompt point (mean of polygon vertices)
            combined_pts = np.concatenate(all_pts, axis=0)
            cx, cy = int(combined_pts[:, 0].mean()), int(combined_pts[:, 1].mean())

            # If centroid falls outside mask (irregular concave crown), sample any point inside
            if mask[min(cy, H_orig - 1), min(cx, W_orig - 1)] == 0:
                ys, xs = np.where(mask > 0)
                if len(ys) > 0:
                    ridx = random.randint(0, len(ys) - 1)
                    cy, cx = int(ys[ridx]), int(xs[ridx])

            crown_masks_list.append(mask > 0)
            prompt_pts_list.append((cy, cx))

        if len(crown_masks_list) == 0:
            # Fallback if no crowns rasterized
            crown_masks_list.append(np.zeros((H_orig, W_orig), dtype=bool))
            prompt_pts_list.append((H_orig // 2, W_orig // 2))

        # Random Crop or Resize to crop_size
        if self.augment and (H_orig > self.crop_size or W_orig > self.crop_size):
            oy = random.randint(0, H_orig - self.crop_size)
            ox = random.randint(0, W_orig - self.crop_size)

            crop_rgb = img_arr[oy : oy + self.crop_size, ox : ox + self.crop_size]

            valid_masks = []
            valid_prompts = []

            for m, (py, px) in zip(crown_masks_list, prompt_pts_list):
                if oy <= py < oy + self.crop_size and ox <= px < ox + self.crop_size:
                    m_crop = m[oy : oy + self.crop_size, ox : ox + self.crop_size]
                    if m_crop.sum() >= 15:
                        valid_masks.append(m_crop)
                        valid_prompts.append((py - oy, px - ox))

            if len(valid_masks) > 0:
                crown_masks_list = valid_masks
                prompt_pts_list = valid_prompts
            else:
                crown_masks_list = [crown_masks_list[0][oy : oy + self.crop_size, ox : ox + self.crop_size]]
                prompt_pts_list = [(self.crop_size // 2, self.crop_size // 2)]
        else:
            if H_orig != self.crop_size or W_orig != self.crop_size:
                crop_rgb = cv2.resize(img_arr, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
                sy = self.crop_size / H_orig
                sx = self.crop_size / W_orig
                crown_masks_list = [
                    cv2.resize(m.astype(np.uint8), (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST) > 0
                    for m in crown_masks_list
                ]
                prompt_pts_list = [(int(py * sy), int(px * sx)) for (py, px) in prompt_pts_list]
            else:
                crop_rgb = img_arr

        # Data Augmentation: Horizontal / Vertical Flip
        if self.augment:
            if random.random() > 0.5:
                crop_rgb = np.fliplr(crop_rgb).copy()
                crown_masks_list = [np.fliplr(m).copy() for m in crown_masks_list]
                prompt_pts_list = [(py, self.crop_size - 1 - px) for (py, px) in prompt_pts_list]
            if random.random() > 0.5:
                crop_rgb = np.flipud(crop_rgb).copy()
                crown_masks_list = [np.flipud(m).copy() for m in crown_masks_list]
                prompt_pts_list = [(self.crop_size - 1 - py, px) for (py, px) in prompt_pts_list]

        # Subsample positive tree prompts up to max_prompts
        if len(crown_masks_list) > self.max_prompts:
            indices = random.sample(range(len(crown_masks_list)), self.max_prompts)
            crown_masks_list = [crown_masks_list[i] for i in indices]
            prompt_pts_list = [prompt_pts_list[i] for i in indices]

        labels_list = [1] * len(crown_masks_list)
        H_c, W_c = crop_rgb.shape[:2]

        # Fast O(1) Negative Background Sampling
        if len(crown_masks_list) > 0:
            tree_union = np.logical_or.reduce(crown_masks_list)
            n_neg = max(2, min(6, len(crown_masks_list) // 3))
            for _ in range(n_neg):
                for attempt in range(15):
                    by = random.randint(0, H_c - 1)
                    bx = random.randint(0, W_c - 1)
                    if not tree_union[by, bx]:
                        crown_masks_list.append(np.zeros((H_c, W_c), dtype=bool))
                        prompt_pts_list.append((by, bx))
                        labels_list.append(0)
                        break

        # Convert to Tensors
        img_t = torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)  # (3, H, W)
        masks_t = torch.from_numpy(np.stack(crown_masks_list, axis=0)).float()          # (K, H, W)
        prompts_t = torch.tensor(prompt_pts_list, dtype=torch.float32)                   # (K, 2) [y, x]
        labels_t = torch.tensor(labels_list, dtype=torch.long)                          # (K,)

        return {
            "image": img_t,
            "masks": masks_t,
            "points": prompts_t,
            "labels": labels_t,
            "num_crowns": len(crown_masks_list),
            "file_name": base_name,
        }


def collate_bam_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collator handling variable number of tree prompt points per tile."""
    images = torch.stack([b["image"] for b in batch], dim=0)  # (B, 3, H, W)
    masks_list = [b["masks"] for b in batch]                  # List of (K_i, H, W)
    points_list = [b["points"] for b in batch]                # List of (K_i, 2)
    labels_list = [b["labels"] for b in batch]                # List of (K_i,)
    file_names = [b["file_name"] for b in batch]

    return {
        "images": images,
        "masks_list": masks_list,
        "points_list": points_list,
        "labels_list": labels_list,
        "file_names": file_names,
    }


if __name__ == "__main__":
    ds = BAMForestsDataset(split="train", crop_size=1024, max_prompts_per_sample=16, augment=True)
    sample = ds[0]
    print(f"Sample 0 -> Image: {sample['image'].shape}, Masks: {sample['masks'].shape}, Points: {sample['points'].shape}, Labels: {sample['labels'].shape}")
    print(f"Prompts:\n{sample['points'][:5]}")
    print(f"Labels:\n{sample['labels']}")
