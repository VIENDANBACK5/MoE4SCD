"""
tests/test_legacy_training_fixes.py
===================================
The training errors found by the 28/09/2026 audit must stay fixed in the legacy TreeFlowNet trainers
(methods/tree_flow/train*.py, datasets/exhaustive_tree_dataset.py):
  1. warm-starting from a checkpoint of the leaked V1 chain is refused unless explicitly allowed;
  2. Quebec training windows come only from the requested zone, are redrawn at every call, and carry labels over a
     grown grid, so a crown cut by the window edge keeps its full extent for target computation;
  3. the default QuebecDataset never touches the Z2 / Z3 chips; the legacy path still reproduces V1's 600 chips;
  4. the V2 / V3 / HR dataset classes return centre-cropped targets of the image size.

Run: python tests/test_legacy_training_fixes.py   (reads the Quebec orthomosaics; ~1 min)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

QUEBEC = REPO / "data/itc_benchmarks/extracted/quebec_2021_09_02"


def test_leaked_checkpoints_are_refused():
    from crown_segmentation_research.lineage import check_clean_init
    for leaked in ("crown_segmentation_research/experiments/tree_flow_exhaustive_v1/best_model.pth",
                   "crown_segmentation_research/experiments/tree_flow_v3/best_model_v3.pth",
                   "crown_segmentation_research/experiments/tree_flow_hr/best_model_hr.pth",
                   "crown_segmentation_research/experiments/ssl_in_domain_weights/simmim_best_encoder.pth"):
        try:
            check_clean_init(leaked)
        except SystemExit:
            pass
        else:
            raise AssertionError(f"{leaked} was accepted")
        check_clean_init(leaked, allow_leaked=True)  # explicit reproduction: warns, does not raise
    check_clean_init(None)                            # ImageNet init
    check_clean_init("crown_segmentation_research/clean_benchmark/runs/c0_v1_s42/last.pth")


def test_sampler_zone_redraw_and_full_crowns():
    from crown_segmentation_research.datasets.exhaustive_tree_dataset import QuebecWindowSampler
    s = QuebecWindowSampler(QUEBEC, zones=("Z1",), patch_size=256, margin_px=128)
    assert [sc["tif"].name for sc in s.scenes] == ["2021-09-02-sbl-z1-rgb-cog.tif"]
    rng = random.Random(0)
    first_img, _ = s.sample(rng)
    found_extension = False
    for _ in range(6):
        img, lab = s.sample(rng)
        assert img.shape == (256, 256, 3) and img.dtype == np.uint8 and lab.shape == (512, 512)
        ctr = lab[128:384, 128:384]
        edge_ids = set(np.unique(np.concatenate([ctr[0], ctr[-1], ctr[:, 0], ctr[:, -1]]))) - {0}
        # a crown touching the window edge must continue into the margin (it is not clipped to the window)
        for u in edge_ids:
            if (lab == u).sum() > (ctr == u).sum():
                found_extension = True
    assert found_extension, "no crown cut by the window edge extends into the margin"
    assert not np.array_equal(first_img, img), "windows are not redrawn"


def test_default_quebec_dataset_is_z1_only_and_legacy_is_v1():
    from crown_segmentation_research.datasets.exhaustive_tree_dataset import QuebecDataset
    ds = QuebecDataset(QUEBEC, patch_size=256, patches_per_scene=10, augment=True, margin_px=128)
    assert len(ds) == 10 and [sc["tif"].name for sc in ds._sampler.scenes] == ["2021-09-02-sbl-z1-rgb-cog.tif"]
    t = ds[0]
    for k in ("flow", "sdt", "centroid", "canopy", "fg_mask"):
        assert tuple(t[k].shape[-2:]) == (256, 256), (k, t[k].shape)
    assert tuple(t["image"].shape) == (3, 256, 256)
    legacy = QuebecDataset(QUEBEC, legacy_v1=True)
    zones = {f.name.split("_patch")[0] for f in legacy._npz_files}
    assert len(legacy) == 600 and zones == {"quebec_z1", "quebec_z2", "quebec_z3"}


def test_v2_v3_hr_datasets_crop_targets_to_image():
    from crown_segmentation_research.datasets.exhaustive_tree_dataset import QuebecWindowSampler
    from crown_segmentation_research.methods.tree_flow import train_hr, train_v2, train_v3
    s = QuebecWindowSampler(QUEBEC, zones=("Z1",), patch_size=256, margin_px=128)
    for mod, cls in ((train_v2, "QuebecNpzDataset"), (train_v3, "QuebecNpzDataset"), (train_hr, "QuebecHRDataset")):
        ds = getattr(mod, cls)(split="train", augment=True, sampler=s, n_samples=3)
        assert len(ds) == 3
        t = ds[0]
        for k, v in t.items():
            assert tuple(v.shape[-2:]) == (256, 256), (mod.__name__, k, tuple(v.shape))
        legacy = getattr(mod, cls)(split="train", augment=False)  # --legacy-cache path: the Z1 chips, unchanged
        assert len(legacy) == 200 and all("_z1_" in f.name for f in legacy.files)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
