"""
eval_test_set.py
================
Evaluate the best MoE model on the SECOND test set, and optionally export
pixel-level semantic predictions (PNG) for object-level conversion.

Supports two modes for pixel map reconstruction:
1. Pure Binary Mode (Default): Uses dummy classes (1 for T1, 2 for T2 on change)
   to prevent ground-truth leakage and evaluate Binary-Object-F1 purely.
2. Oracle Semantics Mode: Uses ground-truth T1/T2 semantic labels to wrap
   the model's binary predictions for upper-bound reference.

Usage:
    python eval_test_set.py \
        --checkpoint SECOND/stage5_6_semantic/best_model.pt \
        --tokens_T1  SECOND/tokens_T1_test \
        --tokens_T2  SECOND/tokens_T2_test \
        --matches    SECOND/matches_test \
        --device cuda \
        --save-preds SECOND-OC/predictions/tokenmoe
"""

import argparse
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from token_change_reasoner_moe import MoEConfig, build_moe_model
from train_reasoner_spectral import MatchDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ── SECOND Label Mapping (only needed for oracle mode) ───────────────────────
RGB_TO_CLASS = {
    (0,   0,   0):   0,  # background / unlabeled
    (0,   128, 0):   1,  # tree
    (128, 0,   0):   2,  # buildings
    (0,   0,   255): 3,  # water
    (128, 128, 128): 4,  # non_veg_ground (impervious surface)
    (255, 255, 255): 5,  # playground
    (0,   255, 0):   6,  # low_vegetation
    (255, 0,   0):   7,  # other
}

_PALETTE_RGB = np.array(list(RGB_TO_CLASS.keys()), dtype=np.float32)
_PALETTE_CLS = np.array(list(RGB_TO_CLASS.values()), dtype=np.int32)


def rgb_to_class(rgb_img: np.ndarray) -> np.ndarray:
    """Convert RGB label image to 0-7 class IDs."""
    H, W, _ = rgb_img.shape
    flat = rgb_img.reshape(-1, 3)
    out  = np.full(len(flat), -1, dtype=np.int32)

    for rgb_tuple, cls_id in RGB_TO_CLASS.items():
        match = np.all(flat == rgb_tuple, axis=1)
        out[match] = cls_id

    unknown_mask = out == -1
    if unknown_mask.any():
        unk = flat[unknown_mask].astype(np.float32)
        dists = ((unk[:, None, :] - _PALETTE_RGB[None, :, :]) ** 2).sum(-1)
        out[unknown_mask] = _PALETTE_CLS[dists.argmin(-1)]

    return out.reshape(H, W)


def dominant_class(pixel_map: np.ndarray, mask: np.ndarray) -> int:
    """Return dominant class id inside mask (ignoring value 0 = background)."""
    pixels = pixel_map[mask]
    fg = pixels[pixels != 0]
    if len(fg) == 0:
        return 0
    vals, counts = np.unique(fg, return_counts=True)
    return int(vals[counts.argmax()])


def evaluate(model, loader, device, save_preds: Optional[Path] = None,
             dataset: Optional[MatchDataset] = None, oracle_semantics: bool = False,
             masks_T1_dir: str = "SECOND/sam2_masks_T1_test",
             masks_T2_dir: str = "SECOND/sam2_masks_T2_test"):
    model.eval()
    tp = fp = fn = tn = 0
    n_samples = 0

    global_idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            outputs = model(batch)
            logits = outputs["change_logits"].detach().cpu()
            labels = batch["change_labels"].detach().cpu()
            mask = ~batch["padding_mask"].cpu()
            sem_logits_T1 = outputs.get("class_logits_T1", None)
            sem_logits_T2 = outputs.get("class_logits_T2", None)
            if sem_logits_T1 is not None:
                sem_logits_T1 = sem_logits_T1.detach().cpu()
                sem_logits_T2 = sem_logits_T2.detach().cpu()

            pred = (logits[mask] > 0).long()
            targ = labels[mask].long()

            tp += ((pred == 1) & (targ == 1)).sum().item()
            fp += ((pred == 1) & (targ == 0)).sum().item()
            fn += ((pred == 0) & (targ == 1)).sum().item()
            tn += ((pred == 0) & (targ == 0)).sum().item()
            n_samples += batch["tokens_pad"].shape[0]

            # Save pixel predictions if save_preds is specified
            if save_preds is not None and dataset is not None:
                B = logits.shape[0]
                for b in range(B):
                    stem = dataset.stems[global_idx]
                    
                    sample_data = dataset[global_idx]
                    n1 = len(sample_data.tokens_t1)
                    
                    # Extract change predictions for T1 tokens
                    change_preds = (logits[b, :n1] > 0).numpy() # bool array of length n1

                    # Load SAM2 masks directly from NPZ (100% index aligned)
                    m1_path = Path(masks_T1_dir) / f"{stem}.npz"
                    m2_path = Path(masks_T2_dir) / f"{stem}.npz"
                    
                    masks_t1 = np.load(m1_path)["masks"] # (n1, 512, 512) bool
                    masks_t2 = np.load(m2_path)["masks"] # (n2, 512, 512) bool
                    
                    t1_areas = sample_data.areas_t1.numpy()

                    # Parse match pairs
                    t1_to_t2 = {}
                    for p in sample_data.match_pairs:
                        t1_to_t2[int(p[0])] = int(p[1])

                    # Determine T1 and T2 predicted classes for each token
                    c1_all = []
                    c2_all = []

                    if oracle_semantics:
                        # Load GT label images
                        lbl1_img = np.array(Image.open(Path("SECOND/test/label1") / f"{stem}.png").convert("RGB"))
                        lbl2_img = np.array(Image.open(Path("SECOND/test/label2") / f"{stem}.png").convert("RGB"))
                        
                        class1 = rgb_to_class(lbl1_img)
                        class2 = rgb_to_class(lbl2_img)

                        for i in range(n1):
                            c1 = dominant_class(class1, masks_t1[i])
                            if change_preds[i]:
                                j = t1_to_t2.get(i)
                                if j is not None and j < len(masks_t2):
                                    c2 = dominant_class(class2, masks_t2[j])
                                else:
                                    c2 = dominant_class(class2, masks_t1[i])
                            else:
                                c2 = c1
                            c1_all.append(c1)
                            c2_all.append(c2)
                    elif sem_logits_T1 is not None:
                        # Use predicted semantic classes from class_logits_T1/T2
                        sem_c1 = sem_logits_T1[b, :n1].argmax(dim=-1).numpy()  # (n1,)
                        sem_c2 = sem_logits_T2[b, :n1].argmax(dim=-1).numpy()  # (n1,)
                        for i in range(n1):
                            c1_all.append(int(sem_c1[i]))
                            c2_all.append(int(sem_c2[i]) if change_preds[i] else int(sem_c1[i]))
                    else:
                        # Pure binary mode: use dummy classes to avoid GT leakage
                        for i in range(n1):
                            c1 = 1  # Dummy foreground class
                            if change_preds[i]:
                                c2 = 2  # Different dummy foreground class
                            else:
                                c2 = 1  # Unchanged
                            c1_all.append(c1)
                            c2_all.append(c2)

                    # Reconstruct pixel maps
                    pred_T1 = np.zeros((512, 512), dtype=np.uint8)
                    pred_T2 = np.zeros((512, 512), dtype=np.uint8)
                    
                    # Sort by area in descending order so smaller masks overwrite larger ones
                    order = np.argsort(t1_areas)[::-1]
                    for i in order:
                        mask_t1 = masks_t1[i]
                        pred_T1[mask_t1] = c1_all[i]
                        pred_T2[mask_t1] = c2_all[i]

                    # Save PNGs
                    Image.fromarray(pred_T1).save(save_preds / "im1" / f"{stem}.png")
                    Image.fromarray(pred_T2).save(save_preds / "im2" / f"{stem}.png")

                    # Save per-token predictions for direct object-level eval
                    token_pred_path = save_preds / "tokens" / f"{stem}.json"
                    token_pred_path.parent.mkdir(exist_ok=True)
                    raw_logits = logits[b, :n1].numpy()
                    trans_logits = outputs.get("transition_logits", None)
                    if trans_logits is not None:
                        trans_logits_np = trans_logits[b, :n1].detach().cpu().numpy()
                    token_records = []
                    for i in range(n1):
                        rec = {
                            "token_idx":    i,
                            "change_logit": float(raw_logits[i]),
                            "change_pred":  bool(change_preds[i]),
                            "class_T1":     int(c1_all[i]),
                            "class_T2":     int(c2_all[i]),
                        }
                        if trans_logits is not None:
                            rec["transition_pred"] = int(trans_logits_np[i].argmax())
                        token_records.append(rec)
                    import json as _json
                    token_pred_path.write_text(_json.dumps(token_records))

                    global_idx += 1

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    iou       = tp / max(tp + fp + fn, 1)
    accuracy  = (tp + tn) / max(tp + fp + fn + tn, 1)

    return {
        "samples": n_samples,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "iou":       iou,
        "accuracy":  accuracy,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="SECOND/stage5_6_semantic/best_model.pt")
    p.add_argument("--tokens_T1",  default="SECOND/tokens_T1_test")
    p.add_argument("--tokens_T2",  default="SECOND/tokens_T2_test")
    p.add_argument("--matches",    default="SECOND/matches_test")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--save-preds", default=None, help="Directory to save reconstructed pixel maps")
    p.add_argument("--oracle-semantics", action="store_true",
                   help="Use GT labels for T1/T2 class assignments (for reference only)")
    p.add_argument("--masks_T1", default="SECOND/sam2_masks_T1_test")
    p.add_argument("--masks_T2", default="SECOND/sam2_masks_T2_test")
    # Add Spectral and LoRA CLI arguments to match train command CLI signatures
    p.add_argument("--use_spectral", action="store_true", default=False)
    p.add_argument("--use_lora",     action="store_true", default=False)
    p.add_argument("--lora_rank",    type=int, default=4)
    p.add_argument("--lora_alpha",   type=float, default=8.0)
    p.add_argument("--lora_lr",      type=float, default=1e-4)
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / "config.json"

    # Load config
    cfg_dict = json.loads(config_path.read_text())
    cfg = MoEConfig(**{k: v for k, v in cfg_dict.items() if k in MoEConfig.__dataclass_fields__})
    if args.use_spectral:
        cfg.use_spectral = True

    # Build and load model
    model = build_moe_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    log.info(f"Loaded checkpoint from {ckpt_path}")

    # Dataset
    dataset = MatchDataset(
        t1_dir    = Path(args.tokens_T1),
        t2_dir    = Path(args.tokens_T2),
        match_dir = Path(args.matches),
    )
    log.info(f"Test samples: {len(dataset)}")

    _collate = collate_fn(cfg, device)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=_collate, num_workers=0)

    save_preds_path = Path(args.save_preds) if args.save_preds else None
    if save_preds_path:
        (save_preds_path / "im1").mkdir(parents=True, exist_ok=True)
        (save_preds_path / "im2").mkdir(parents=True, exist_ok=True)
        log.info(f"Saving predictions to {save_preds_path}")

    metrics = evaluate(model, loader, device, save_preds=save_preds_path,
                       dataset=dataset, oracle_semantics=args.oracle_semantics,
                       masks_T1_dir=args.masks_T1, masks_T2_dir=args.masks_T2)

    print("\n" + "="*50)
    print("TEST SET RESULTS (proxy labels)")
    print("="*50)
    print(f"  Samples   : {metrics['samples']}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1        : {metrics['f1']:.4f}")
    print(f"  IoU       : {metrics['iou']:.4f}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  TN={metrics['tn']}")
    print("="*50)

    # Save results
    out_path = ckpt_path.parent / "test_results.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
