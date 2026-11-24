"""
tests/test_clean_benchmark.py
=============================
The fast per-instance targets must reproduce the original whole-image targets (so the clean baseline "C0"
trains on the same target definition as V1), and the v2 targets must have the properties they were
designed for (no interior plateau, seam only where crowns touch).

Run: python tests/test_clean_benchmark.py   (or python -m pytest tests/test_clean_benchmark.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from crown_segmentation_research.clean_benchmark.targets_fast import targets_v1, targets_v2
from crown_segmentation_research.methods.tree_flow.targets import compute_tree_flow_targets


def _labels():
    rng = np.random.default_rng(0)
    lab = np.zeros((160, 200), np.int32)
    k = 1
    for _ in range(9):
        cy, cx, r = rng.integers(10, 150), rng.integers(10, 190), rng.integers(8, 30)
        yy, xx = np.ogrid[:160, :200]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 < r * r
        lab[m & (lab == 0)] = k
        k += 1
    return lab


def test_v1_matches_original_targets():
    lab = _labels()
    f0, s0, c0, k0 = compute_tree_flow_targets(lab)
    f1, s1, c1, k1 = targets_v1(lab)
    assert np.allclose(f0, f1, atol=1e-5)
    assert np.allclose(s0, s1, atol=1e-5)
    assert np.allclose(c0, c1, atol=1e-3)  # Gaussian truncated at 4 sigma in the fast version
    assert np.array_equal(k0, k1)


def test_v2_has_no_interior_plateau_and_seam_only_between_crowns():
    lab = np.zeros((60, 100), np.int32)
    lab[10:50, 10:50] = 1
    lab[10:50, 50:90] = 2          # touches crown 1 along x = 49/50
    flow, sdt, cent, can, seam = targets_v2(lab)
    inside = sdt[0][lab == 1]
    assert inside.max() == 1.0 and len(np.unique(np.round(inside, 3))) > 15  # graded, not clamped flat
    assert seam[0][:, 45:55].any() and not seam[0][:, 10:40].any()          # seam at the shared edge only
    assert not seam[0][0:12, 10:40].any()     # not on crown edges that face background (away from the seam)


def _toy_coco():
    """Two 64x64 images, 3 GT squares each; detections: exact hits, a shifted box, and a spurious one."""
    from pycocotools import mask as mu
    from pycocotools.coco import COCO

    def rle(y, x, h, w):
        m = np.zeros((64, 64), np.uint8)
        m[y:y + h, x:x + w] = 1
        r = mu.encode(np.asfortranarray(m))
        return {"size": r["size"], "counts": r["counts"].decode()}

    anns, dts = [], []
    for img in (1, 2):
        for k, (y, x) in enumerate([(2, 2), (2, 34), (34, 2)]):
            anns.append({"id": len(anns) + 1, "image_id": img, "category_id": 1, "segmentation": rle(y, x, 20, 20),
                         "area": 400.0, "bbox": [x, y, 20, 20], "iscrowd": 0})
        dts += [{"image_id": img, "category_id": 1, "segmentation": rle(2, 2, 20, 20), "score": 0.9},
                {"image_id": img, "category_id": 1, "segmentation": rle(5, 36, 20, 20), "score": 0.8},
                {"image_id": img, "category_id": 1, "segmentation": rle(40, 40, 10, 10), "score": 0.7}]
    gt = COCO()
    gt.dataset = {"images": [{"id": i, "height": 64, "width": 64} for i in (1, 2)], "annotations": anns,
                  "categories": [{"id": 1, "name": "tree"}]}
    gt.createIndex()
    return gt, dts


def test_coco_ap_matches_pycocotools_and_detections_stay_serialisable():
    """coco_ap exists because summarize() reads AP at a hard-coded maxDets=100; with maxDets [1, 10, 100]
    it must reproduce stats[0..2] exactly. loadRes must not leave numpy fields in our own result dicts."""
    import json
    from pycocotools.cocoeval import COCOeval
    from crown_segmentation_research.clean_benchmark.evaluate import coco_ap
    gt, dts = _toy_coco()
    E = COCOeval(gt, gt.loadRes([dict(d) for d in dts]), "segm")
    E.params.maxDets = [1, 10, 100]
    E.evaluate(); E.accumulate(); E.summarize()
    ours = coco_ap(E, max_det=100)
    assert np.allclose([ours["AP"], ours["AP50"], ours["AP75"]], 100 * E.stats[:3]), (ours, E.stats[:3])
    json.dumps(dts)  # raises TypeError if loadRes wrote numpy values into these dicts


def _toy_run(epochs, start_state=None, save_after=None, path=None):
    """Tiny CPU training loop that draws from every RNG the trainers use (torch, numpy, python, loader generator)."""
    import random
    import torch
    from crown_segmentation_research.clean_benchmark.checkpointing import load_state, save_state
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    model = torch.nn.Linear(4, 1)
    opt = torch.optim.AdamW(model.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=0.005)
    gen = torch.Generator().manual_seed(1)
    start, hist, best = 1, [], float("inf")
    if start_state is not None:
        torch.manual_seed(123); gen.manual_seed(999)  # scramble, so only a correct restore can match
        start, hist, best, exact = load_state(start_state, model, opt, sched, gen, Path(start_state).parent / "none.json", "v")
        assert exact
    for ep in range(start, epochs + 1):
        x = torch.randn(8, 4) + float(np.random.rand()) + random.random()
        idx = torch.randperm(8, generator=gen)
        loss = (model(x[idx]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        hist.append({"epoch": ep, "v": float(loss)}); best = min(best, float(loss))
        if ep == save_after:
            save_state(path, model, opt, sched, gen, ep, hist, best, {})
    return torch.cat([q.detach().flatten() for q in model.parameters()]), hist


def test_resume_continues_exactly_and_weights_only_resume_is_flagged():
    import tempfile
    import torch
    from crown_segmentation_research.clean_benchmark.checkpointing import load_state
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d) / "last.pth"
        straight, h_straight = _toy_run(4)
        _toy_run(4, save_after=2, path=ck)
        resumed, h_resumed = _toy_run(4, start_state=ck)
        assert torch.equal(straight, resumed) and h_straight == h_resumed
        # weights-only checkpoint (format of runs started before checkpointing.py): resumes, flagged inexact,
        # and the scheduler lands on the same LR as the uninterrupted run
        torch.save({"model_state_dict": torch.nn.Linear(4, 1).state_dict(), "epoch": 2}, ck)
        m = torch.nn.Linear(4, 1); opt = torch.optim.AdamW(m.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=4, eta_min=0.005)
        start, hist, best, exact = load_state(ck, m, opt, sched, torch.Generator(), Path(d) / "none.json", "v")
        ref = torch.optim.lr_scheduler.CosineAnnealingLR(torch.optim.AdamW(torch.nn.Linear(4, 1).parameters(), lr=0.1),
                                                         T_max=4, eta_min=0.005)
        ref.step(); ref.step()
        assert (start, exact) == (3, False) and np.isclose(opt.param_groups[0]["lr"], ref.get_last_lr()[0])


def test_target_margin_gives_full_crown_targets_and_zero_margin_is_unchanged():
    """C2's grown-window targets: with a margin covering the whole tile, crop targets must equal targets computed on
    the whole tile and then cropped (crowns cut by the crop edge keep full-crown targets); margin 0 must equal targets
    computed on the crop itself (the C0/C1 behaviour)."""
    import tempfile
    import crown_segmentation_research.clean_benchmark.dataset as D
    rng = np.random.default_rng(3)
    lab = np.zeros((96, 96), np.int32)
    for k, (cy, cx, r) in enumerate([(20, 20, 14), (48, 60, 18), (80, 30, 12), (70, 85, 10)], start=1):
        yy, xx = np.ogrid[:96, :96]
        lab[((yy - cy) ** 2 + (xx - cx) ** 2 < r * r) & (lab == 0)] = k
    img = rng.integers(0, 255, (96, 96, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "validation").mkdir()
        np.savez(Path(d) / "validation" / "t.npz", image=img, valid=np.ones((96, 96), bool), label=lab)
        old = D.PREP
        D.PREP = Path(d)
        try:
            for tv, fn in (("v1", targets_v1), ("v2", targets_v2)):
                big = D.CanopyCropDataset("validation", 64, 1, tv, augment=False, fixed=True, target_margin=96)[0]
                zero = D.CanopyCropDataset("validation", 64, 1, tv, augment=False, fixed=True)[0]
                im = (big["image"].numpy().transpose(1, 2, 0) * D.STD + D.MEAN) * 255
                y0, x0 = next((y, x) for y in range(33) for x in range(33)
                              if np.abs(img[y:y + 64, x:x + 64] - im).max() < 1e-2)
                full = fn(lab)
                crop = lab[y0:y0 + 64, x0:x0 + 64]
                u, inv = np.unique(crop, return_inverse=True)
                own = fn((inv + (0 if u[0] == 0 else 1)).reshape(64, 64).astype(np.int32))
                for j, key in enumerate(("flow", "sdt", "centroid", "canopy")):
                    assert np.allclose(big[key].numpy(), full[j][:, y0:y0 + 64, x0:x0 + 64], atol=1e-5), (tv, key)
                    assert np.allclose(zero[key].numpy(), own[j], atol=1e-5), (tv, key)
                # a crown cut by this crop edge really gets different (full-crown) targets with the margin
                assert not np.allclose(big["flow"].numpy(), zero["flow"].numpy())
        finally:
            D.PREP = old


def test_error_breakdown_outcomes():
    """One toy tile per outcome: an exact hit, two crowns covered by one prediction, a crown cut in two, a crown
    with no prediction on it."""
    from pycocotools import mask as mu
    from crown_segmentation_research.clean_benchmark.error_breakdown import outcomes_for_tile

    def box(y, x, h, w):
        m = np.zeros((100, 100), np.uint8)
        m[y:y + h, x:x + w] = 1
        r = mu.encode(np.asfortranarray(m))
        return r, float(mu.area(r))

    gts = {1: box(0, 0, 20, 20), 2: box(0, 40, 20, 20), 3: box(0, 62, 20, 20), 4: box(50, 0, 20, 44), 5: box(80, 80, 15, 15)}
    gt = [(k, r, a) for k, (r, a) in gts.items()]
    dts = [box(0, 0, 20, 20), box(0, 40, 20, 42), box(50, 0, 20, 20), box(50, 24, 20, 20)]  # halves: IoU 0.45 each
    dt = [(r, 0.9 - 0.1 * i, a) for i, (r, a) in enumerate(dts)]
    res = outcomes_for_tile(gt, dt)
    assert res == {1: "matched", 2: "merged", 3: "merged", 4: "split", 5: "missed"}, res


def test_centre_map_rank_and_sdt_source():
    import torch
    from crown_segmentation_research.clean_benchmark.evaluate import centre_map
    g = torch.Generator().manual_seed(0)
    o = {"centroid": torch.rand(1, 20, 30, generator=g) * 0.08, "sdt": torch.rand(1, 20, 30, generator=g) * 2 - 1}
    r = centre_map(o, "head", "rank")
    assert float(r.min()) == 0.0 and float(r.max()) == 1.0
    assert torch.equal(torch.argsort(r.flatten()), torch.argsort(o["centroid"].flatten()))  # order preserved
    assert torch.equal(centre_map(o, "sdt", "none"), o["sdt"].clamp(0, 1))
    assert torch.equal(centre_map(o, "head", "none"), o["centroid"])                      # default unchanged


def test_crown_weighted_focal_loss():
    """C4: each crown's centre pixels weigh 1 in total whatever its size; a crown-free crop stays finite and small;
    with weight 1 per positive pixel and n_norm = #positives it equals the original CenterNet loss."""
    import tempfile
    import torch
    import crown_segmentation_research.clean_benchmark.dataset as D
    from crown_segmentation_research.clean_benchmark.train_treeflow import crown_focal_loss
    from crown_segmentation_research.methods.tree_flow.loss import centroid_focal_loss
    lab = np.zeros((200, 200), np.int32)
    yy, xx = np.ogrid[:200, :200]
    lab[(yy - 90) ** 2 + (xx - 90) ** 2 < 80 ** 2] = 1   # large crown: several centre pixels
    lab[(yy - 185) ** 2 + (xx - 185) ** 2 < 8 ** 2] = 2  # small crown: one centre pixel
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "validation").mkdir()
        np.savez(Path(d) / "validation" / "t.npz", image=np.zeros((200, 200, 3), np.uint8),
                 valid=np.ones((200, 200), bool), label=lab)
        old, D.PREP = D.PREP, Path(d)
        try:
            b = D.CanopyCropDataset("validation", 200, 1, "v2", augment=False, fixed=True, centroid_crown_weights=True)[0]
        finally:
            D.PREP = old
    w, ce = b["cent_w"][0].numpy(), b["centroid"][0].numpy()
    assert float(b["n_crowns"]) == 2
    for k in (1, 2):
        assert abs(w[lab == k].sum() - 1.0) < 1e-5, (k, w[lab == k].sum())
    assert (ce[lab == 1] >= 0.99).sum() > (ce[lab == 2] >= 0.99).sum()  # size-scaled Gaussians: big crown has more
    p = torch.full((1, 1, 200, 200), 0.5)
    gt = torch.from_numpy(ce)[None, None]
    pos = (gt >= 0.99).float()
    assert torch.isclose(crown_focal_loss(p, gt, pos, float(pos.sum())), centroid_focal_loss(p, gt), rtol=1e-5)
    empty = torch.zeros_like(gt)
    assert float(crown_focal_loss(torch.full_like(gt, 0.02), empty, empty, 30.0)) < 1.0


def test_modern_backbones_and_default_resnet_unchanged():
    """Swin-T / ConvNeXt-T must give the same output structure as the ResNet-50 default (strides 4/8/16/32 FPN, heads read
    P2, outputs at input size), the ResNet-50 state-dict keys must be exactly those of the C0-C5 checkpoints, and a
    checkpoint trained with --backbone must be rebuildable by evaluate.py from its recorded args alone."""
    import torch
    from crown_segmentation_research.methods.tree_flow.model import TreeFlowNet
    x = torch.randn(1, 3, 128, 128)
    base = TreeFlowNet(pretrained_backbone=False)
    ref = {k: v.shape for k, v in base.state_dict().items()}
    assert all(k.startswith(("backbone.", "flow_head.", "sdt_head.", "centroid_head.", "canopy_head.")) for k in ref)
    assert any(k.startswith("backbone.body.layer4") for k in ref)          # the ResNet-50 layout of C0-C5, unchanged
    for name in ("swin_t", "convnext_tiny"):
        m = TreeFlowNet(pretrained_backbone=False, backbone=name).eval()
        with torch.no_grad():
            o = m(x)
            feats = m.backbone(x)
        assert set(o) == set(base.eval()(x)), name
        for k in ("flow", "sdt", "centroid", "canopy"):
            assert o[k].shape[-2:] == (128, 128), (name, k)
        assert [feats[str(i)].shape[-1] for i in range(4)] == [32, 16, 8, 4] and feats["0"].shape[1] == 256, name
        assert not any(k.startswith("backbone.body.layer") for k in m.state_dict()), name
        # the head weights carry no backbone dependence: same shapes as ResNet-50's
        assert {k: v.shape for k, v in m.state_dict().items() if not k.startswith("backbone.")} == \
               {k: v for k, v in ref.items() if not k.startswith("backbone.")}
        m2 = TreeFlowNet(backbone=name)                                     # what evaluate.py builds from args
        m2.load_state_dict(m.state_dict())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
