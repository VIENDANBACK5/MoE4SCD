"""BAMFORESTS Dataset for Option A: Neural Potential Surface + Persistence Watershed.

Streams directly from Bamberg_coco2048.zip, crops to 1024x1024 with polygon rasterization,
and computes high-fidelity potential surface, boundary ridge, and canopy support gate.
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

from crown_segmentation_research.methods.canopy_watershed.targets import compute_canopy_watershed_targets

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


class BAMCanopyWatershedDataset(Dataset):
    """BAMFORESTS Dataset for Canopy Watershed model training & evaluation."""

    def __init__(
        self,
        split: str = "train",  # "train", "eval", "test1", "test2"
        crop_size: int = 1024,
        augment: bool = True,
        compute_targets: bool = True,
        zip_path: Path = ZIP_ARCHIVE,
    ) -> None:
        super().__init__()
        self.split = split
        self.crop_size = crop_size
        self.augment = augment
        self.compute_targets = compute_targets
        self.zip_path = zip_path

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
        self.img_to_anns: dict[int, list[dict[str, Any]]] = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        self._name_to_zip = get_zip_manifest(self.zip_path)
        self.images = [
            img for img in self.images
            if img["file_name"] in self._name_to_zip and len(self.img_to_anns.get(img["id"], [])) > 0
        ]
        print(f"[{split.upper()}] Initialized: {len(self.images)} images with valid tree crowns.", flush=True)
        self._zipfiles: dict[int, zipfile.ZipFile] = {}

    def _get_zipfile(self) -> zipfile.ZipFile:
        import os
        pid = os.getpid()
        if pid not in self._zipfiles:
            self._zipfiles[pid] = zipfile.ZipFile(self.zip_path, "r")
        return self._zipfiles[pid]

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

        pil_img = Image.open(io.BytesIO(data))
        img_arr = np.array(pil_img)[:, :, :3]  # (H, W, 3) RGB uint8
        H_orig, W_orig = img_arr.shape[:2]

        # 1:1 Native Resolution Crop
        if H_orig > self.crop_size or W_orig > self.crop_size:
            if self.augment:
                oy = random.randint(0, max(0, H_orig - self.crop_size))
                ox = random.randint(0, max(0, W_orig - self.crop_size))
            else:
                oy = max(0, (H_orig - self.crop_size) // 2)
                ox = max(0, (W_orig - self.crop_size) // 2)

            crop_rgb = img_arr[oy : oy + self.crop_size, ox : ox + self.crop_size]
            crop_inst = np.zeros((self.crop_size, self.crop_size), dtype=np.int32)
            inst_id = 1

            for ann in anns:
                seg = ann.get("segmentation", [])
                if not seg:
                    continue
                crown_mask = np.zeros((self.crop_size, self.crop_size), dtype=np.uint8)
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    pts_shifted = pts - np.array([ox, oy])
                    if len(pts_shifted) >= 3:
                        cv2.fillPoly(crown_mask, [pts_shifted], 1)
                if crown_mask.sum() >= 15:
                    crop_inst[crown_mask > 0] = inst_id
                    inst_id += 1
        else:
            crop_rgb = img_arr
            crop_inst = np.zeros((H_orig, W_orig), dtype=np.int32)
            inst_id = 1
            for ann in anns:
                seg = ann.get("segmentation", [])
                if not seg:
                    continue
                crown_mask = np.zeros((H_orig, W_orig), dtype=np.uint8)
                for poly in seg:
                    pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
                    if len(pts) >= 3:
                        cv2.fillPoly(crown_mask, [pts], 1)
                if crown_mask.sum() >= 15:
                    crop_inst[crown_mask > 0] = inst_id
                    inst_id += 1

        # Data Augmentations
        if self.augment:
            if random.random() > 0.5:
                crop_rgb = np.fliplr(crop_rgb).copy()
                crop_inst = np.fliplr(crop_inst).copy()
            if random.random() > 0.5:
                crop_rgb = np.flipud(crop_rgb).copy()
                crop_inst = np.flipud(crop_inst).copy()

        # Compute targets if requested
        if self.compute_targets:
            surf_tgt, bound_tgt, canopy_tgt = compute_canopy_watershed_targets(crop_inst)
            surf_t = torch.from_numpy(surf_tgt)
            bound_t = torch.from_numpy(bound_tgt)
            canopy_t = torch.from_numpy(canopy_tgt)
        else:
            surf_t = torch.empty(0)
            bound_t = torch.empty(0)
            canopy_t = torch.empty(0)

        return {
            "image": torch.from_numpy(crop_rgb.astype(np.float32) / 255.0).permute(2, 0, 1),
            "surface": surf_t,
            "boundary": bound_t,
            "canopy": canopy_t,
            "instance_label": torch.from_numpy(crop_inst),
            "image_name": base_name,
        }
