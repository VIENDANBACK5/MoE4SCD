#!/usr/bin/env python3
"""DTE-aerial Gate-0 parity runner.

Three tasks in one invocation:
  1. QC: verify 25 sites / 525 tiles / multi-resolution structure / label set.
  2. Inference: run official MiT-B3 checkpoint, freeze per-tile predictions.
  3. Parity: site-macro F1, per-resolution breakdown, PASS/FAIL verdict.

Usage (background):
    nohup python scripts/dte_g0_parity.py \\
        --bench  datasets/DTE-aerial-bench \\
        --meta   datasets/DTE-aerial-bench/DTE-aerial-bench-meta.csv \\
        --ckpt   DTE-aerial-model/DTE_aerial_model.safetensors \\
        --cfg    DTE-aerial-official/config/evaluation.yml \\
        --out    dte_runs/g0_parity \\
        > dte_runs/logs/g0_parity.log 2>&1 &
    echo $! > dte_runs/logs/g0_parity.pid
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from safetensors.torch import load_file

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EXPECTED_SITES = 25
EXPECTED_TILES = 525
VALID_LABEL_VALUES = {0, 1, 2, 255}
RESOLUTIONS = ["5cm", "10cm", "20cm"]

# Published targets (Sharma et al. 2026, Table 2)
PARITY_TARGETS = {
    "mortality_f1_overall":   0.59,
    "mortality_iou_overall":  0.45,
    "mortality_prec_overall": 0.72,
    "mortality_rec_overall":  0.54,
    "treecover_f1_overall":   0.89,
    "mortality_f1_5cm":       0.60,
    "mortality_f1_10cm":      0.55,
    "mortality_f1_20cm":      0.45,
}
TOLERANCE = 0.03   # absolute; investigate if exceeded


# ──────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ──────────────────────────────────────────────
# Task 1 — Dataset QC
# ──────────────────────────────────────────────
def qc_benchmark(bench_dir: Path, meta_csv: Path) -> dict:
    log("=== QC START ===")
    issues = []

    df = pd.read_csv(meta_csv)
    required_cols = {"site", "biome", "resolution", "tile_path", "mask_path"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        issues.append(f"MISSING COLUMNS in meta CSV: {missing_cols}")

    n_sites = df["site"].nunique() if "site" in df.columns else -1
    n_tiles = len(df)
    log(f"  Sites: {n_sites}  (expected {EXPECTED_SITES})")
    log(f"  Tiles: {n_tiles}  (expected {EXPECTED_TILES})")
    if n_sites != EXPECTED_SITES:
        issues.append(f"Expected {EXPECTED_SITES} sites, got {n_sites}")
    if n_tiles != EXPECTED_TILES:
        issues.append(f"Expected {EXPECTED_TILES} tiles, got {n_tiles}")

    # Resolution distribution
    if "resolution" in df.columns:
        res_counts = df["resolution"].value_counts().to_dict()
        log(f"  Resolution counts: {res_counts}")
        expected_per_res = {"5cm": EXPECTED_SITES * 16, "10cm": EXPECTED_SITES * 4, "20cm": EXPECTED_SITES * 1}
        for res, expected in expected_per_res.items():
            got = res_counts.get(res, 0)
            if got != expected:
                issues.append(f"Resolution {res}: expected {expected} tiles, got {got}")

    # Multi-resolution pairing per site
    if "site" in df.columns and "resolution" in df.columns:
        grouped = df.groupby("site")["resolution"].apply(set)
        bad = grouped[grouped.apply(lambda s: not s.issuperset({"5cm","10cm","20cm"}))].index.tolist()
        if bad:
            issues.append(f"Sites missing some resolutions: {bad[:5]}")

    if "biome" in df.columns:
        log(f"  Biomes: {df['biome'].unique().tolist()}")

    # File existence + label QC on 10% sample
    sample_df = df.sample(frac=0.1, random_state=42) if len(df) > 20 else df
    bad_labels, missing_files = [], []
    for _, row in sample_df.iterrows():
        img_path = bench_dir / row["tile_path"]
        msk_path = bench_dir / row["mask_path"]
        if not img_path.exists():
            missing_files.append(str(row["tile_path"])); continue
        if not msk_path.exists():
            missing_files.append(str(row["mask_path"])); continue
        try:
            img = Image.open(img_path)
            if img.mode not in ("RGB", "RGBA"):
                issues.append(f"Non-RGB image: {img_path}")
        except Exception as e:
            issues.append(f"Cannot open image {img_path}: {e}")
        try:
            msk = np.array(Image.open(msk_path))
            illegal = set(np.unique(msk).tolist()) - VALID_LABEL_VALUES
            if illegal:
                bad_labels.append((str(row["mask_path"]), illegal))
        except Exception as e:
            issues.append(f"Cannot open mask {msk_path}: {e}")

    if missing_files:
        issues.append(f"Missing files (first 5): {missing_files[:5]}")
    if bad_labels:
        issues.append(f"Illegal label values: {bad_labels[:3]}")

    status = "PASS" if not issues else "FAIL"
    log(f"  QC status: {status}")
    for iss in issues:
        log(f"  ISSUE: {iss}")

    return {"status": status, "n_sites": n_sites, "n_tiles": n_tiles, "issues": issues}


# ──────────────────────────────────────────────
# Task 2 — Inference
# ──────────────────────────────────────────────
def build_model(cfg_path: Path, ckpt_path: Path, device: torch.device):
    repo = cfg_path.parents[1]
    sys.path.insert(0, str(repo))
    from src.model import build_model as _build
    from src.utils import get_config

    class _Args:
        cfg = str(cfg_path)
        checkpoint = str(ckpt_path)
        output = "dummy"

    config = get_config(_Args())
    model = _build(config.model).to(device)
    ckpt = load_file(str(ckpt_path), device="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(state_dict, strict=True)
    model = model.to(memory_format=torch.channels_last, device=device)
    model.eval()
    log(f"  Model loaded from {ckpt_path.name}")
    return model, config


@torch.inference_mode()
def run_inference(model, bench_dir: Path, meta_csv: Path, out_dir: Path, device) -> Path:
    from torchvision import transforms
    df = pd.read_csv(meta_csv)
    records = []
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    log(f"  Running inference on {len(df)} tiles …")
    t0 = time.time()

    for i, (_, row) in enumerate(df.iterrows()):
        img_path = bench_dir / row["tile_path"]
        msk_path = bench_dir / row["mask_path"]
        img = Image.open(img_path).convert("RGB")
        msk = np.array(Image.open(msk_path))
        x = preprocess(img).unsqueeze(0).to(device=device, memory_format=torch.channels_last)
        pred = model(x).softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        tile_name = Path(row["tile_path"]).stem
        np.save(pred_dir / f"{tile_name}_pred.npy", pred)

        gt_flat   = msk.ravel().astype(np.int64)
        pred_flat = pred.ravel().astype(np.int64)
        valid     = gt_flat != 255
        gt_flat, pred_flat = gt_flat[valid], pred_flat[valid]
        K = 3
        cm = np.zeros((K, K), dtype=np.int64)
        np.add.at(cm, (gt_flat, pred_flat), 1)

        record = {
            "tile":       tile_name,
            "ortho_id":   str(row["site"]),
            "resolution": str(row["resolution"]),
            "biome":      str(row["biome"]),
        }
        for c in range(K):
            record[f"tp_{c}"] = int(cm[c, c])
            record[f"fp_{c}"] = int(cm[:, c].sum() - cm[c, c])
            record[f"fn_{c}"] = int(cm[c, :].sum() - cm[c, c])
        records.append(record)

        if (i + 1) % 50 == 0:
            log(f"  {i+1}/{len(df)} tiles | {time.time()-t0:.0f}s elapsed")

    parquet_path = out_dir / "per_tile_metrics.parquet"
    pd.DataFrame(records).to_parquet(parquet_path, index=False)
    log(f"  Saved per-tile metrics → {parquet_path}")
    return parquet_path


# ──────────────────────────────────────────────
# Task 3 — Parity check
# ──────────────────────────────────────────────
def compute_f1(tp, fp, fn, eps=1e-8):
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    return float(2 * p * r / (p + r + eps))

def compute_iou(tp, fp, fn, eps=1e-8):
    return float(tp / (tp + fp + fn + eps))

def site_macro_f1(df: pd.DataFrame, class_idx: int) -> float:
    f1s = []
    for _, grp in df.groupby("ortho_id"):
        tp = grp[f"tp_{class_idx}"].sum()
        fp = grp[f"fp_{class_idx}"].sum()
        fn = grp[f"fn_{class_idx}"].sum()
        f1s.append(compute_f1(tp, fp, fn))
    return float(np.mean(f1s))


def parity_check(parquet_path: Path) -> dict:
    log("=== PARITY CHECK ===")
    df = pd.read_parquet(parquet_path)
    results = {}

    results["mortality_f1_overall"]   = site_macro_f1(df, 2)
    results["treecover_f1_overall"]   = site_macro_f1(df, 1)
    tp2 = df["tp_2"].sum(); fp2 = df["fp_2"].sum(); fn2 = df["fn_2"].sum()
    results["mortality_iou_overall"]  = compute_iou(tp2, fp2, fn2)
    results["mortality_prec_overall"] = float(tp2 / (tp2 + fp2 + 1e-8))
    results["mortality_rec_overall"]  = float(tp2 / (tp2 + fn2 + 1e-8))

    for res in RESOLUTIONS:
        sub = df[df["resolution"] == res]
        results[f"mortality_f1_{res}"] = site_macro_f1(sub, 2) if len(sub) > 0 else float("nan")

    trend_ok = (
        results.get("mortality_f1_5cm",0) >
        results.get("mortality_f1_10cm",0) >
        results.get("mortality_f1_20cm",0)
    )

    log(f"\n{'─'*60}")
    log(f"  {'Metric':<38} {'Got':>7}  {'Target':>7}  {'Δ':>6}  OK?")
    log(f"{'─'*60}")
    verdict_ok = True
    for key, target in PARITY_TARGETS.items():
        got = results.get(key, float("nan"))
        delta = got - target
        ok = abs(delta) <= TOLERANCE
        if not ok:
            verdict_ok = False
        log(f"  {key:<38} {got:>7.4f}  {target:>7.4f}  {delta:>+6.4f}  {'✓' if ok else '✗'}")
    log(f"{'─'*60}")
    log(f"  5cm > 10cm > 20cm trend: {'✓' if trend_ok else '✗'}")

    overall = verdict_ok and trend_ok
    log(f"\n  ═══  G0 VERDICT: {'PASS' if overall else 'FAIL'}  ═══\n")
    results["trend_ok"] = trend_ok
    results["verdict"]  = "PASS" if overall else "FAIL"
    results["tolerance"] = TOLERANCE
    return results


# ──────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────
def write_report(qc: dict, parity: dict, ckpt_sha: str, out_dir: Path):
    verdict = parity.get("verdict", "FAIL")
    tol = parity.get("tolerance", TOLERANCE)
    lines = [
        f"# G0 Parity Report", f"",
        f"**Verdict: {verdict}**", f"",
        f"## Checkpoint",
        f"- SHA-256: `{ckpt_sha}`", f"",
        f"## QC",
        f"- Status: {qc['status']}",
        f"- Sites: {qc['n_sites']}  (expected {EXPECTED_SITES})",
        f"- Tiles: {qc['n_tiles']}  (expected {EXPECTED_TILES})",
    ]
    if qc["issues"]:
        lines += ["- Issues:"] + [f"  - {iss}" for iss in qc["issues"]]
    else:
        lines += ["- No issues found."]

    lines += [f"", f"## Parity", f"",
              f"| Metric | Got | Target | Δ | OK? |",
              f"|---|---:|---:|---:|---|"]
    for key, target in PARITY_TARGETS.items():
        got   = parity.get(key, float("nan"))
        delta = got - target
        ok    = "✓" if abs(delta) <= tol else "✗"
        lines.append(f"| {key} | {got:.4f} | {target:.4f} | {delta:+.4f} | {ok} |")

    lines += [f"", f"5cm > 10cm > 20cm trend: {'✓' if parity.get('trend_ok') else '✗'}",
              f"", f"## Per-resolution mortality F1", f"",
              f"| Resolution | F1 |", f"|---|---:|"]
    for res in RESOLUTIONS:
        lines.append(f"| {res} | {parity.get(f'mortality_f1_{res}', float('nan')):.4f} |")

    lines += [f"", f"## Next step", f"",
              f"**G0 PASS → immediately start G1 (`scripts/dte_g1_diagnose.py`). "
              f"Predictions frozen at `dte_runs/g0_parity/predictions/`. "
              f"Do NOT run additional baselines before G1 visual diagnosis.**"]

    report_path = out_dir / "gate0.md"
    report_path.write_text("\n".join(lines))
    (out_dir / "gate0.json").write_text(json.dumps({**qc, **parity}, indent=2, default=str))
    log(f"  Report → {report_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("DTE G0 parity runner")
    p.add_argument("--bench",  required=True)
    p.add_argument("--meta",   required=True)
    p.add_argument("--ckpt",   required=True)
    p.add_argument("--cfg",    required=True)
    p.add_argument("--out",    default="dte_runs/g0_parity")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-inference", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    bench  = Path(args.bench)
    meta   = Path(args.meta)
    ckpt   = Path(args.ckpt)
    cfg    = Path(args.cfg)
    out    = Path(args.out)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)

    log(f"DTE G0  |  device={device}  |  out={out}")
    ckpt_sha = sha256(ckpt)
    log(f"  Checkpoint SHA-256: {ckpt_sha}")

    qc_result = qc_benchmark(bench, meta)
    if qc_result["status"] == "FAIL":
        log("QC FAILED. Fix issues before inference.")
        sys.exit(1)

    parquet_path = out / "per_tile_metrics.parquet"
    if args.skip_inference and parquet_path.exists():
        log(f"Skipping inference (existing {parquet_path})")
    else:
        model, _ = build_model(cfg, ckpt, device)
        parquet_path = run_inference(model, bench, meta, out, device)

    parity_result = parity_check(parquet_path)
    write_report(qc_result, parity_result, ckpt_sha, out)

    verdict = parity_result.get("verdict","FAIL")
    log(f"Done. Verdict: {verdict}")
    if verdict == "PASS":
        log("Next: python scripts/dte_g1_diagnose.py --bench ... --meta ... --preds dte_runs/g0_parity/predictions --out dte_runs/g1_diagnosis")
    sys.exit(0 if verdict == "PASS" else 2)


if __name__ == "__main__":
    main()
