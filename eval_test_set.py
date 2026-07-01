"""
eval_test_set.py
================
Evaluate the best MoE model on the SECOND test set.

Usage:
    python eval_test_set.py \
        --checkpoint SECOND/stage5_6_semantic/best_model.pt \
        --tokens_T1  SECOND/tokens_T1_test \
        --tokens_T2  SECOND/tokens_T2_test \
        --matches    SECOND/matches_test \
        --device cuda
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from token_change_reasoner_moe import MoEConfig, build_moe_model
from train_reasoner import MatchDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = tn = 0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            outputs = model(batch)
            logits = outputs["change_logits"].detach()
            labels = batch["change_labels"].detach()
            mask = ~batch["padding_mask"]

            pred = (logits[mask] > 0).long()
            targ = labels[mask].long()

            tp += ((pred == 1) & (targ == 1)).sum().item()
            fp += ((pred == 1) & (targ == 0)).sum().item()
            fn += ((pred == 0) & (targ == 1)).sum().item()
            tn += ((pred == 0) & (targ == 0)).sum().item()
            n_samples += batch["tokens_pad"].shape[0]

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
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(args.checkpoint)
    config_path = ckpt_path.parent / "config.json"

    # Load config
    cfg_dict = json.loads(config_path.read_text())
    cfg = MoEConfig(**{k: v for k, v in cfg_dict.items() if k in MoEConfig.__dataclass_fields__})

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

    metrics = evaluate(model, loader, device)

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
