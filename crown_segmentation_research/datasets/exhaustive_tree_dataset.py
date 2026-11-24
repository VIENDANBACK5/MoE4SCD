"""
Unified exhaustive all-tree crown dataset.

Combines three sources of exhaustive (all-trees-annotated) ground truth:
  1. BAMFORESTS (BAM) — 58K train / 15K eval crowns, COCO JSON + zip archive
  2. Quebec 2021   — 23K crowns, GeoPackage polygons + GeoTIFF RGB
  3. DeadTrees tree_cover — 87K crowns, GeoPackage instances layer

All sources produce: (image_patch, targets) where targets has keys:
    flow      [2, H, W]  centripetal unit flow
    sdt       [1, H, W]  signed distance transform [-1,1]
    centroid  [1, H, W]  Gaussian centroid heatmap
    canopy    [1, H, W]  binary tree cover
    fg_mask   [1, H, W]  foreground (crown interior) mask
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T

# Geo packages: the GeoTIFF paths below use rasterio / geopandas / Window, which were never imported (the fallback
# only ran when no npz cache existed, so the NameError went unnoticed)
try:
    import geopandas as gpd
    import rasterio
    from rasterio.windows import Window
    _GEO_AVAILABLE = True
except ImportError:
    _GEO_AVAILABLE = False

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


# ─────────────────────────────────────────────────────────────────────────────
# Target computation (shared across all sources)
# ─────────────────────────────────────────────────────────────────────────────

def compute_targets(
    masks: list[np.ndarray],   # list of bool arrays [H, W], one per instance
    H: int, W: int,
    sigma_fraction: float = 0.08,
) -> dict[str, np.ndarray]:
    """
    Compute flow/SDT/centroid/canopy targets from binary instance masks.
    Optimized: computes background distance transform once and local instance crops.
    """
    from scipy.ndimage import distance_transform_edt

    canopy        = np.zeros((H, W), dtype=bool)
    for m in masks:
        canopy |= m

    dist_bg = distance_transform_edt(~canopy)
    sdt_out = -np.clip(dist_bg / 15.0, 0.0, 1.0).astype(np.float32)
    centroid_heat = np.zeros((H, W), dtype=np.float32)
    flow_y        = np.zeros((H, W), dtype=np.float32)
    flow_x        = np.zeros((H, W), dtype=np.float32)

    for m in masks:
        rows, cols = np.where(m)
        if len(rows) < 5:
            continue
        r0, r1 = rows.min(), rows.max() + 1
        c0, c1 = cols.min(), cols.max() + 1
        m_crop = m[r0:r1, c0:c1]

        # SDT inside crown
        dist_in = distance_transform_edt(m_crop)
        max_d   = dist_in.max() + 1e-6
        sdt_crop = (dist_in / max_d).astype(np.float32)
        sdt_out[r0:r1, c0:c1][m_crop] = np.maximum(sdt_out[r0:r1, c0:c1][m_crop], sdt_crop[m_crop])

        # Centroid Gaussian heatmap
        cy, cx = rows.mean(), cols.mean()
        area   = len(rows)
        sigma  = max(sigma_fraction * np.sqrt(float(area)), 3.0)
        ys, xs = np.mgrid[r0:r1, c0:c1].astype(np.float32)
        dist_sq = (ys - cy) ** 2 + (xs - cx) ** 2
        gauss   = np.exp(-dist_sq / (2 * sigma ** 2)).astype(np.float32)
        centroid_heat[r0:r1, c0:c1][m_crop] = np.maximum(centroid_heat[r0:r1, c0:c1][m_crop], gauss[m_crop])

        # Flow unit vectors toward centroid
        dy = cy - ys
        dx = cx - xs
        mag = np.sqrt(dy ** 2 + dx ** 2 + 1e-6)
        flow_y[r0:r1, c0:c1][m_crop] = (dy / mag)[m_crop]
        flow_x[r0:r1, c0:c1][m_crop] = (dx / mag)[m_crop]

    fg_mask = (sdt_out > -0.5).astype(np.float32)

    return {
        "flow":     np.stack([flow_y, flow_x], axis=0),  # [2, H, W]
        "sdt":      sdt_out[None],                        # [1, H, W]
        "centroid": centroid_heat[None],                  # [1, H, W]
        "canopy":   canopy[None].astype(np.float32),      # [1, H, W]
        "fg_mask":  fg_mask[None],                        # [1, H, W]
    }


def targets_to_tensors(targets: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {k: torch.from_numpy(v).float() for k, v in targets.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Augmentation (shared)
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def normalize_image(img: torch.Tensor) -> torch.Tensor:
    """Safely normalize [0, 255] RGB image tensor to ImageNet standardized float32."""
    img = img.float()
    if img.max() > 1.0:
        img = img / 255.0
    return (img - _IMAGENET_MEAN) / _IMAGENET_STD


def augment_pair(
    img: np.ndarray,           # [H, W, 3] uint8
    masks: list[np.ndarray],   # list of bool [H, W]
    flip_h: bool = True,
    flip_v: bool = True,
    rot90: bool = True,
    color_jitter: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Random spatial and color augmentation, consistent across image and masks."""
    if flip_h and random.random() < 0.5:
        img = img[:, ::-1, :].copy()
        masks = [m[:, ::-1].copy() for m in masks]

    if flip_v and random.random() < 0.5:
        img = img[::-1, :, :].copy()
        masks = [m[::-1, :].copy() for m in masks]

    if rot90 and random.random() < 0.5:
        k = random.randint(1, 3)
        img = np.rot90(img, k).copy()
        masks = [np.rot90(m, k).copy() for m in masks]

    if color_jitter and random.random() < 0.5:
        pil = Image.fromarray(img)
        jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
        img = np.array(jitter(pil))

    return img, masks


# ─────────────────────────────────────────────────────────────────────────────
# Source 1: BAMFORESTS (BAM) COCO dataset
# ─────────────────────────────────────────────────────────────────────────────

class BAMCocoDataset(Dataset):
    """
    BAMFORESTS COCO 2048px dataset.
    Loads images from the zip archive on-the-fly, crops to `patch_size`.
    """

    def __init__(
        self,
        json_path: str | Path,
        archive_path: str | Path,
        patch_size: int = 1024,
        augment: bool = True,
        area_min_px: int = 30,
        area_max_px: int = 500_000,
    ):
        from pycocotools.coco import COCO
        from pycocotools import mask as coco_mask_util

        self.coco        = COCO(str(json_path))
        self.archive     = Path(archive_path)
        self.patch_size  = patch_size
        self.augment     = augment
        self.area_min    = area_min_px
        self.area_max    = area_max_px
        self._rle_util   = coco_mask_util

        self.img_ids = self.coco.getImgIds()
        # Pre-build reverse index: img_id → list of ann dicts
        self._img_anns: dict[int, list] = {}
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            self._img_anns[img_id] = self.coco.loadAnns(ann_ids)

        # Load zip manifest if present
        manifest_p = Path(json_path).parent / "zip_manifest.json"
        self._manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else None

        # Open zip once and keep handle (thread-unsafe for multiprocessing, use num_workers=0)
        self._zf: Optional[zipfile.ZipFile] = None
        self._zf_names: set[str] = set()

    def _ensure_zip(self):
        import os
        pid = os.getpid()
        if self._zf is None or getattr(self, "_zf_pid", None) != pid:
            self._zf = zipfile.ZipFile(self.archive, "r")
            self._zf_names = set(self._zf.namelist())
            self._zf_pid = pid

    def _load_image(self, file_name: str) -> np.ndarray:
        self._ensure_zip()
        if self._manifest and file_name in self._manifest:
            try:
                data = self._zf.read(self._manifest[file_name])
                return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
            except KeyError:
                pass
        # Try common layouts
        for prefix in ("", "images/", "Bamberg_coco2048/images/", "coco2048/images/", "coco2048/train2023/", "coco2048/val2023/", "coco2048/test2023/Test-Set-1/", "coco2048/test2023/Test-Set-2/"):
            key = prefix + file_name
            if key in self._zf_names:
                data = self._zf.read(key)
                return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
        raise FileNotFoundError(f"{file_name} not in archive (tried several prefixes)")

    def _rle_to_mask(self, seg, H: int, W: int) -> np.ndarray:
        if isinstance(seg, dict):
            return self._rle_util.decode(seg).astype(bool)
        rle = self._rle_util.frPyObjects(seg, H, W)
        rle = self._rle_util.merge(rle)
        return self._rle_util.decode(rle).astype(bool)

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_id   = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        file_name = img_info["file_name"]
        anns      = self._img_anns[img_id]

        img_np = self._load_image(file_name)
        H, W   = img_np.shape[:2]

        # Convert annotations to mask list, filter by area
        masks = []
        for ann in anns:
            area = ann.get("area", 0)
            if area < self.area_min or area > self.area_max:
                continue
            m = self._rle_to_mask(ann["segmentation"], H, W)
            if m.sum() >= self.area_min:
                masks.append(m)

        # Random crop if larger than patch_size
        if H > self.patch_size or W > self.patch_size:
            r0 = random.randint(0, max(0, H - self.patch_size))
            c0 = random.randint(0, max(0, W - self.patch_size))
            img_np = img_np[r0:r0 + self.patch_size, c0:c0 + self.patch_size]
            masks  = [m[r0:r0 + self.patch_size, c0:c0 + self.patch_size] for m in masks]
            masks  = [m for m in masks if m.sum() >= self.area_min]

        # Crop / Pad to exact (patch_size, patch_size)
        H2, W2 = img_np.shape[:2]
        if H2 != self.patch_size or W2 != self.patch_size:
            img_np = img_np[:self.patch_size, :self.patch_size]
            masks  = [m[:self.patch_size, :self.patch_size] for m in masks]
            H3, W3 = img_np.shape[:2]
            if H3 < self.patch_size or W3 < self.patch_size:
                pad_h = max(0, self.patch_size - H3)
                pad_w = max(0, self.patch_size - W3)
                img_np = np.pad(img_np, ((0, pad_h), (0, pad_w), (0, 0)))
                masks  = [np.pad(m, ((0, pad_h), (0, pad_w))) for m in masks]

        if self.augment:
            img_np, masks = augment_pair(img_np, masks)

        H_p, W_p = img_np.shape[:2]
        targets  = compute_targets(masks, H_p, W_p)
        targets  = targets_to_tensors(targets)

        img_t = normalize_image(torch.from_numpy(img_np.copy()).permute(2, 0, 1))
        targets["image"] = img_t
        targets["source"] = "bam"
        return targets

    def __del__(self):
        if self._zf is not None:
            self._zf.close()


# ─────────────────────────────────────────────────────────────────────────────
# Source 2: Quebec 2021 (GeoPackage + GeoTIFF)
# ─────────────────────────────────────────────────────────────────────────────

class QuebecWindowSampler:
    """Random windows from the Quebec 2021-09-02 orthomosaics of the given zones, with instance labels rasterised
    over the window grown by `margin_px` on every side.

    Replaces the pre-cut `quebec_cached_npz` chips for training (audit 28/09/2026):
      - the cache holds 200 fixed 512-px windows of EVERY zone, Z3 (test) included, and was read unfiltered;
      - its labels are crowns already clipped to the window, so flow/centroid/SDT targets of a crown cut by the edge
        point at the fragment's own centre (41.7% of crowns in a 1024-px crop touch its edge on CanopyRS);
      - the windows never change between epochs.
    Here a new window is drawn at every call, only from `zones`, and the caller computes targets on the grown label
    grid and keeps the centre, so crowns within `margin_px` of the edge keep full-crown targets.
    Default margin 256 px = 4.8 m at 1.87 cm: covers the full extent of ~97% of crowns cut by the window (the p99.9
    equivalent crown radius is 5.6 m on CanopyRS; crowns reaching further stay truncated)."""

    def __init__(self, data_dir, zones=("Z1",), patch_size: int = 512, margin_px: int = 256,
                 min_label_fraction: float = 0.1, max_tries: int = 10):
        import geopandas as gpd
        import rasterio
        self.data_dir, self.zones, self.P, self.m = Path(data_dir), tuple(zones), patch_size, margin_px
        self.min_label_fraction, self.max_tries = min_label_fraction, max_tries
        self.scenes = []
        for z in self.zones:
            tif, gpkg = self.data_dir / f"2021-09-02-sbl-{z.lower()}-rgb-cog.tif", self.data_dir / f"{z}_polygons.gpkg"
            if not (tif.exists() and gpkg.exists()):
                raise FileNotFoundError(f"missing {tif} or {gpkg}")
            with rasterio.open(tif) as src:
                gdf = gpd.read_file(str(gpkg)).to_crs(src.crs)
                self.scenes.append({"tif": tif, "H": src.height, "W": src.width, "gdf": gdf, "sindex": gdf.sindex})

    def sample(self, rng: random.Random):
        """Returns (image uint8 [P, P, 3], label int32 [P + 2m, P + 2m]); label 0 = background."""
        import rasterio
        from rasterio.features import rasterize
        from rasterio.windows import Window
        from shapely.geometry import box
        P, m = self.P, self.m
        for attempt in range(self.max_tries):
            sc = self.scenes[rng.randrange(len(self.scenes))]
            r0, c0 = rng.randint(0, sc["H"] - P), rng.randint(0, sc["W"] - P)
            with rasterio.open(sc["tif"]) as src:
                img = src.read([1, 2, 3], window=Window(c0, r0, P, P)).transpose(1, 2, 0)
                big_T = src.window_transform(Window(c0 - m, r0 - m, P + 2 * m, P + 2 * m))
            x0, y0 = big_T * (0, 0)
            x1, y1 = big_T * (P + 2 * m, P + 2 * m)
            bb = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            idx = [i for i in sc["sindex"].query(bb) if sc["gdf"].geometry.iloc[i].intersects(bb)]
            shapes = [(sc["gdf"].geometry.iloc[i], k + 1) for k, i in enumerate(idx)]
            label = (rasterize(shapes, out_shape=(P + 2 * m, P + 2 * m), transform=big_T, fill=0, dtype="int32")
                     if shapes else np.zeros((P + 2 * m, P + 2 * m), np.int32))
            if (label[m:m + P, m:m + P] > 0).mean() >= self.min_label_fraction or attempt == self.max_tries - 1:
                if img.dtype != np.uint8:
                    img = np.clip(img / max(img.max(), 1) * 255, 0, 255).astype(np.uint8)
                return np.ascontiguousarray(img), label


class QuebecDataset(Dataset):
    """
    Quebec boreal forest crown dataset.
    Loads from fast pre-cached .npz chips when available, with fallback to GeoTIFF.
    """

    ZONES = ["Z1", "Z2", "Z3"]

    def __init__(
        self,
        data_dir: str | Path,
        patch_size: int = 512,
        patches_per_scene: int = 200,
        augment: bool = True,
        area_min_m2: float = 2.0,
        area_max_m2: float = 3000.0,
        zones: tuple = ("Z1",),
        margin_px: int = 256,
        legacy_v1: bool = False,
    ):
        """Default: training zone Z1 only, a fresh random window per sample, full-crown targets (see
        QuebecWindowSampler). `legacy_v1=True` reproduces what V1 was trained on (every cached npz chip of Z1, Z2
        and Z3 -- i.e. the test zone -- with clipped crowns); use it only to reproduce that run."""
        self.data_dir   = Path(data_dir)
        self.patch_size = patch_size
        self.augment    = augment
        self.area_min   = area_min_m2
        self.area_max   = area_max_m2
        self.margin     = margin_px
        self.legacy_v1  = legacy_v1
        self._npz_files = []
        if not legacy_v1:
            self._sampler = QuebecWindowSampler(self.data_dir, zones, patch_size, margin_px)
            self._n = patches_per_scene * len(tuple(zones))
            return
        print("WARNING: QuebecDataset(legacy_v1=True) reads Z1+Z2+Z3 (test) chips with clipped crowns", flush=True)

        # Check for pre-cached npz directory
        cached_dir = self.data_dir.parent / "quebec_cached_npz"
        self._npz_files = sorted(cached_dir.glob("*.npz")) if cached_dir.exists() else []

        if not self._npz_files:
            if not _GEO_AVAILABLE:
                raise ImportError("geopandas and rasterio are required for QuebecDataset")
            # Build list of (scene_path, gpkg_path) pairs
            self.scenes = []
            for z in self.ZONES:
                tif = self.data_dir / f"2021-09-02-sbl-{z.lower()}-rgb-cog.tif"
                gpkg = self.data_dir / f"{z}_polygons.gpkg"
                if tif.exists() and gpkg.exists():
                    self.scenes.append((tif, gpkg))

            if not self.scenes:
                raise FileNotFoundError(f"No Quebec scenes found in {data_dir}")

            self._patches = []
            for si, (tif, _) in enumerate(self.scenes):
                with rasterio.open(tif) as src:
                    H, W = src.height, src.width
                for _ in range(patches_per_scene):
                    r = random.randint(0, max(0, H - patch_size))
                    c = random.randint(0, max(0, W - patch_size))
                    self._patches.append((si, r, c))
            self._gdfs: dict[int, Any] = {}

    def _get_gdf(self, si: int):
        if si not in self._gdfs:
            tif, gpkg = self.scenes[si]
            gdf = gpd.read_file(str(gpkg))
            with rasterio.open(tif) as src:
                raster_crs = src.crs
            if gdf.crs != raster_crs:
                gdf = gdf.to_crs(raster_crs)
            self._gdfs[si] = gdf
        return self._gdfs[si]

    def __len__(self) -> int:
        if not self.legacy_v1:
            return self._n
        return len(self._npz_files) if self._npz_files else len(self._patches)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if not self.legacy_v1:
            img_np, big = self._sampler.sample(random)  # `random` is seeded per DataLoader worker by torch
            masks = [big == u for u in np.unique(big) if u > 0]
            masks = [mk for mk in masks if mk.sum() >= 5]
            if self.augment:  # flips / rot90 of the centred square grid stay aligned with the image
                img_np, masks = augment_pair(img_np, masks)
            B = big.shape[0]
            m, P = self.margin, self.patch_size
            targets = compute_targets(masks, B, B)
            targets = {k: v[..., m:m + P, m:m + P].copy() if isinstance(v, np.ndarray) and v.shape[-2:] == (B, B) else v
                       for k, v in targets.items()}
            targets = targets_to_tensors(targets)
            targets["image"] = normalize_image(torch.from_numpy(img_np.copy()).permute(2, 0, 1))
            targets["source"] = "quebec"
            return targets
        if self._npz_files:
            data = np.load(self._npz_files[idx])
            img_np = data["image"]
            label_map = data["instance_label"]

            masks = []
            for uid in np.unique(label_map):
                if uid == 0:
                    continue
                m = (label_map == uid)
                if m.sum() >= 5:
                    masks.append(m)

            if self.augment:
                img_np, masks = augment_pair(img_np, masks)

            H, W = img_np.shape[:2]
            targets = compute_targets(masks, H, W)
            targets = targets_to_tensors(targets)
            img_t = normalize_image(torch.from_numpy(img_np.copy()).permute(2, 0, 1))
            targets["image"]  = img_t
            targets["source"] = "quebec"
            return targets

        # Fallback to direct GeoTIFF reading
        si, r0, c0 = self._patches[idx]
        tif, _ = self.scenes[si]
        gdf = self._get_gdf(si)
        P = self.patch_size
        with rasterio.open(tif) as src:
            window = Window(c0, r0, P, P)
            img_np = src.read([1, 2, 3], window=window).transpose(1, 2, 0)
            if img_np.dtype != np.uint8:
                img_np = np.clip(img_np / img_np.max() * 255, 0, 255).astype(np.uint8)
            transform = src.window_transform(window)

        from rasterio.features import rasterize
        from shapely.geometry import box
        H, W = img_np.shape[:2]
        ul = transform * (0, 0)
        lr = transform * (W, H)
        patch_box = box(min(ul[0], lr[0]), min(ul[1], lr[1]),
                        max(ul[0], lr[0]), max(ul[1], lr[1]))

        # Clip crowns to patch extent
        patch_gdf = gdf[gdf.intersects(patch_box)].copy()

        masks = []
        for _, row in patch_gdf.iterrows():
            geom = row.geometry.intersection(patch_box)
            if geom.is_empty:
                continue
            m = rasterize(
                [(geom, 1)],
                out_shape=(H, W),
                transform=transform,
                fill=0,
                dtype=np.uint8,
            ).astype(bool)
            if m.sum() >= 5:
                masks.append(m)

        if self.augment:
            img_np, masks = augment_pair(img_np, masks)

        H2, W2 = img_np.shape[:2]
        targets = compute_targets(masks, H2, W2)
        targets = targets_to_tensors(targets)

        img_t = normalize_image(torch.from_numpy(img_np.copy()).permute(2, 0, 1))
        targets["image"]  = img_t
        targets["source"] = "quebec"
        return targets


# ─────────────────────────────────────────────────────────────────────────────
# Source 3: DeadTrees tree_cover (GeoPackage layer)
# ─────────────────────────────────────────────────────────────────────────────

class DeadTreesTreeCoverDataset(Dataset):
    """
    DeadTrees tree_cover instances — exhaustive crown polygons across 181 sites.
    Uses the pre-extracted BAM COCO-format images (tiles) as the RGB source.
    Alternatively, reads site RGB from DeadTrees image tiles if they exist.
    """

    def __init__(
        self,
        bam_json_path: str | Path,     # use BAM train images as the underlying images
        bam_archive:   str | Path,
        patch_size: int = 1024,
        augment: bool = True,
        area_min_px: int = 30,
    ):
        """
        For now, this source reuses BAM train images but with tree_cover GT.
        This is a placeholder — will switch to actual DeadTrees site imagery in Step 1.2.
        """
        # Delegate to BAM dataset with same images
        self._delegate = BAMCocoDataset(
            json_path=bam_json_path,
            archive_path=bam_archive,
            patch_size=patch_size,
            augment=augment,
            area_min_px=area_min_px,
        )

    def __len__(self) -> int:
        return len(self._delegate)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self._delegate[idx]
        item["source"] = "deadtrees_tree_cover"
        return item


# ─────────────────────────────────────────────────────────────────────────────
# Factory: build combined ConcatDataset
# ─────────────────────────────────────────────────────────────────────────────

def build_exhaustive_tree_dataset(
    bam_coco_dir:   str | Path = None,
    bam_archive:    str | Path = None,
    quebec_dir:     str | Path = None,
    patch_size: int = 1024,
    augment: bool = True,
    use_bam:     bool = True,
    use_quebec:  bool = True,
    use_deadtrees: bool = False,  # placeholder, enable later
    quebec_zones: tuple = ("Z1",),
    quebec_legacy_v1: bool = False,
) -> Dataset:
    """Build and return the unified exhaustive all-tree training dataset."""
    bam_coco_dir = Path(bam_coco_dir or (REPO / "data/itc_benchmarks/extracted/bam_coco2048"))
    bam_archive  = Path(bam_archive  or (REPO / "data/itc_benchmarks/raw_archives/Bamberg_coco2048.zip"))
    quebec_dir   = Path(quebec_dir   or (REPO / "data/itc_benchmarks/extracted/quebec_2021_09_02"))

    sources = []

    if use_bam and bam_archive.exists() and (bam_coco_dir / "instances_tree_train2023.json").exists():
        bam_ds = BAMCocoDataset(
            json_path     = bam_coco_dir / "instances_tree_train2023.json",
            archive_path  = bam_archive,
            patch_size    = patch_size,
            augment       = augment,
        )
        sources.append(bam_ds)
        print(f"  [ExhaustiveTreeDataset] BAM: {len(bam_ds)} images")
    else:
        print("  [ExhaustiveTreeDataset] BAM source not found / disabled — skipping")

    if use_quebec and _GEO_AVAILABLE and quebec_dir.exists():
        try:
            qc_ds = QuebecDataset(
                data_dir         = quebec_dir,
                patch_size       = patch_size,
                patches_per_scene= 300,
                augment          = augment,
                zones            = quebec_zones,
                legacy_v1        = quebec_legacy_v1,
            )
            sources.append(qc_ds)
            print(f"  [ExhaustiveTreeDataset] Quebec: {len(qc_ds)} patches")
        except Exception as e:
            print(f"  [ExhaustiveTreeDataset] Quebec failed: {e}")

    if not sources:
        raise RuntimeError("No training sources available!")

    if len(sources) == 1:
        return sources[0]
    return ConcatDataset(sources)
