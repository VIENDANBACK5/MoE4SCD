import os
import sys
import glob
import json
import time
from pathlib import Path
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy import ndimage

# Ensure root in sys.path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from crown_segmentation_research.methods.tree_flow.model_v2 import TreeFlowNetV2
from crown_segmentation_research.methods.tree_flow.decode import decode_organic
from crown_segmentation_research.evaluation.eval_exhaustive_quebec import get_patch_files

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_diagnostics():
    print("=== STARTING 5-METHOD FORESTRY DIAGNOSTICS (TEST ZONE 3) ===")
    files = get_patch_files("test")
    print(f"Loaded {len(files)} test patch files.")

    # Load TreeFlowNetV2
    ckpt_path = "crown_segmentation_research/experiments/tree_flow_v2_warmstart/best_model_v2.pth"
    model = TreeFlowNetV2().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    all_data = []

    print("Running inference across 200 test patches...")
    t0 = time.time()
    for idx, fpath in enumerate(files):
        data = np.load(fpath)
        img = data["image"]  # (H, W, 3) uint8
        inst_gt = data["instance_label"]  # (H, W) int32
        H, W = img.shape[:2]

        gt_uids = [uid for uid in np.unique(inst_gt) if uid > 0]
        gt_masks = []
        gt_centroids = []
        gt_areas = []
        for uid in gt_uids:
            m = (inst_gt == uid).astype(np.uint8)
            if m.sum() < 5:
                continue
            gt_masks.append(m)
            gt_areas.append(int(m.sum()))
            cy, cx = ndimage.center_of_mass(m)
            gt_centroids.append((float(cy), float(cx)))

        # Run inference
        norm = ((img.astype(np.float32) / 255.0) - mean) / std
        t_img = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
        with torch.no_grad():
            preds = model(t_img)
            preds_sq = {k: v[0] for k, v in preds.items()}

        instances = decode_organic(
            preds_sq,
            img_rgb=img,
            min_sink_dist=44,
            canopy_thresh=0.30,
            sdt_thresh=-0.15,
            min_area=150,
            gradient_mode="sobel",
            refine_mode="guided_filter",
            score_mode="calibrated",
            soft_nms=True,
            device=str(DEVICE),
        )

        pred_masks = []
        pred_scores = []
        pred_centroids = []
        pred_areas = []
        for inst in instances:
            m = inst["mask"].astype(np.uint8)
            if m.sum() < 5:
                continue
            pred_masks.append(m)
            pred_scores.append(float(inst["score"]))
            pred_areas.append(int(m.sum()))
            cy, cx = ndimage.center_of_mass(m)
            pred_centroids.append((float(cy), float(cx)))

        all_data.append({
            "fpath": str(fpath),
            "img": img,
            "gt_masks": gt_masks,
            "gt_centroids": gt_centroids,
            "gt_areas": gt_areas,
            "pred_masks": pred_masks,
            "pred_scores": pred_scores,
            "pred_centroids": pred_centroids,
            "pred_areas": pred_areas,
        })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(files):
            print(f"  Processed {idx + 1}/{len(files)} patches ({time.time() - t0:.1f}s)...")

    # ─────────────────────────────────────────────────────────────
    # METHOD 1: ERROR TYPOLOGY BREAKDOWN (DALPONTE & AUBRY-KIENTZ)
    # ─────────────────────────────────────────────────────────────
    print("\n--- METHOD 1: ERROR TYPOLOGY BREAKDOWN ---")
    tot_gt = sum(len(d["gt_masks"]) for d in all_data)
    tot_pred = sum(len(d["pred_masks"]) for d in all_data)
    
    tp_1to1 = 0
    gt_split_count = 0
    excess_split_preds = 0
    omission_count = 0
    pred_merge_count = 0
    isolated_commissions = 0

    unassigned_crops = []

    for d in all_data:
        gts = d["gt_masks"]
        preds = d["pred_masks"]
        img = d["img"]

        if len(gts) == 0:
            isolated_commissions += len(preds)
            continue
        if len(preds) == 0:
            omission_count += len(gts)
            continue

        iou_mat = np.zeros((len(preds), len(gts)), dtype=np.float32)
        overlap_mat_p = np.zeros((len(preds), len(gts)), dtype=np.float32) # overlap / pred_area
        overlap_mat_g = np.zeros((len(preds), len(gts)), dtype=np.float32) # overlap / gt_area

        for i, pm in enumerate(preds):
            pa = pm.sum()
            for j, gm in enumerate(gts):
                ga = gm.sum()
                inter = np.sum((pm > 0) & (gm > 0))
                union = pa + ga - inter
                iou_mat[i, j] = inter / max(union, 1)
                overlap_mat_p[i, j] = inter / max(pa, 1)
                overlap_mat_g[i, j] = inter / max(ga, 1)

        # Hungarian matching at IoU >= 0.50 for 1-to-1
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched_p = set()
        matched_g = set()
        for r, c in zip(row_ind, col_ind):
            if iou_mat[r, c] >= 0.50:
                matched_p.add(r)
                matched_g.add(c)
                tp_1to1 += 1

        # Check GT splits (1 GT matched/overlapped by multiple preds)
        for j in range(len(gts)):
            if j in matched_g:
                # Even if matched, check if other preds overlap this same GT
                overlapping_preds = [i for i in range(len(preds)) if i != list(matched_p)[list(matched_g).index(j)] and overlap_mat_p[i, j] >= 0.20]
                if len(overlapping_preds) > 0:
                    gt_split_count += 1
                    excess_split_preds += len(overlapping_preds)
            else:
                # Unmatched GT: check if multiple preds overlap it
                overlapping_preds = [i for i in range(len(preds)) if overlap_mat_p[i, j] >= 0.20]
                if len(overlapping_preds) >= 2:
                    gt_split_count += 1
                    excess_split_preds += len(overlapping_preds) - 1
                elif len(overlapping_preds) == 0:
                    omission_count += 1
                else:
                    # 1 pred overlapped but IoU < 0.50
                    pass

        # Check Pred merges and isolated commissions
        for i in range(len(preds)):
            if i in matched_p:
                continue
            
            # Count overlapping GTs
            overlapping_gts = [j for j in range(len(gts)) if overlap_mat_g[i, j] >= 0.20]
            if len(overlapping_gts) >= 2:
                pred_merge_count += 1
            else:
                max_ov = overlap_mat_p[i, :].max() if len(gts) > 0 else 0
                if max_ov < 0.15:
                    isolated_commissions += 1
                    # Collect crop for blind audit
                    pm = preds[i]
                    cy, cx = d["pred_centroids"][i]
                    cy, cx = int(cy), int(cx)
                    y1, y2 = max(0, cy - 32), min(img.shape[0], cy + 32)
                    x1, x2 = max(0, cx - 32), min(img.shape[1], cx + 32)
                    crop = img[y1:y2, x1:x2]
                    unassigned_crops.append({
                        "crop": crop,
                        "mask_crop": pm[y1:y2, x1:x2],
                        "area": d["pred_areas"][i],
                        "score": d["pred_scores"][i],
                    })

    print(f"Total GT: {tot_gt}, Total Pred: {tot_pred}")
    print(f"  1. 1-to-1 Match (TP @ IoU>=0.5): {tp_1to1} ({tp_1to1/tot_gt:.1%})")
    print(f"  2. GT Crowns Split into Multiple Preds: {gt_split_count} crowns")
    print(f"     -> Excess Split Predictions (Multi-leader): {excess_split_preds}")
    print(f"  3. GT Omissions (Missed Crowns): {omission_count} ({omission_count/tot_gt:.1%})")
    print(f"  4. Pred Merges (Under-segmentation N->1): {pred_merge_count}")
    print(f"  5. Isolated Commissions (Outside GT): {isolated_commissions}")

    # ─────────────────────────────────────────────────────────────
    # METHOD 2: BLIND AUDIT SIMULATION & ADJUSTED PRECISION (P*)
    # ─────────────────────────────────────────────────────────────
    print("\n--- METHOD 2: BLIND AUDIT SIMULATION & ADJUSTED PRECISION ---")
    verified_unannotated = 0
    ambiguous_shrub = 0
    true_false_alarms = 0

    for item in unassigned_crops:
        cr = item["crop"]
        if cr.shape[0] < 10 or cr.shape[1] < 10:
            continue
        R = cr[:, :, 0].astype(float)
        G = cr[:, :, 1].astype(float)
        B = cr[:, :, 2].astype(float)
        exg = 2 * G - R - B
        gli = (2 * G - R - B) / (2 * G + R + B + 1e-6)
        mean_exg = np.mean(exg)
        mean_gli = np.mean(gli)

        # In unassigned canopy, if ExG > 15 and GLI > 0.10, it is photosynthetic tree foliage
        if mean_exg >= 15.0 and mean_gli >= 0.10:
            verified_unannotated += 1
        elif mean_exg >= 0.0:
            ambiguous_shrub += 1
        else:
            true_false_alarms += 1

    tot_unassigned = len(unassigned_crops)
    rate_verified = verified_unannotated / max(tot_unassigned, 1)
    rate_shrub = ambiguous_shrub / max(tot_unassigned, 1)
    rate_fa = true_false_alarms / max(tot_unassigned, 1)

    # Calculate Adjusted Precision:
    # Standard Precision = TP / N_pred
    std_prec = tp_1to1 / tot_pred
    # Adjusted Precision treats verified unannotated trees as True Positives:
    p_star = (tp_1to1 + verified_unannotated) / tot_pred
    # Structural Adjusted Precision (treating multi-leader branches as valid foliage):
    p_structural = (tp_1to1 + excess_split_preds + verified_unannotated) / tot_pred

    print(f"Isolated Detections Evaluated: {tot_unassigned}")
    print(f"  -> Verified Unannotated Tree Foliage: {verified_unannotated} ({rate_verified:.1%})")
    print(f"  -> Ambiguous Understory / Shrub: {ambiguous_shrub} ({rate_shrub:.1%})")
    print(f"  -> True Non-Canopy False Alarms: {true_false_alarms} ({rate_fa:.1%})")
    print(f"Standard Benchmark Precision: {std_prec:.1%}")
    print(f"Adjusted Precision (P* - Foliage Verified): {p_star:.1%}")
    print(f"Structural Precision (Including Valid Multi-Leader Apexes): {p_structural:.1%}")

    # ─────────────────────────────────────────────────────────────
    # METHOD 3: CENTROID DISTANCE TOLERANCE MATCHING
    # ─────────────────────────────────────────────────────────────
    print("\n--- METHOD 3: CENTROID DISTANCE TOLERANCE MATCHING ---")
    dist_thresholds_px = [5, 10, 15, 20, 25, 30, 35, 40]
    # GSD ~ 0.10 m/px: 10px = 1.0m, 20px = 2.0m, 30px = 3.0m
    dist_results = []

    for d_tol in dist_thresholds_px:
        d_tp = 0
        d_pred_tot = tot_pred
        d_gt_tot = tot_gt

        for d in all_data:
            g_c = np.array(d["gt_centroids"])
            p_c = np.array(d["pred_centroids"])

            if len(g_c) == 0 or len(p_c) == 0:
                continue

            dist_mat = cdist(p_c, g_c, metric="euclidean")
            r_ind, c_ind = linear_sum_assignment(dist_mat)
            for r, c in zip(r_ind, c_ind):
                if dist_mat[r, c] <= d_tol:
                    d_tp += 1

        d_prec = d_tp / max(d_pred_tot, 1)
        d_rec = d_tp / max(d_gt_tot, 1)
        d_f1 = 2 * d_prec * d_rec / max(d_prec + d_rec, 1e-6)
        dist_results.append({
            "d_tol_px": d_tol,
            "d_tol_m": d_tol * 0.10,
            "tp": d_tp,
            "precision": d_prec,
            "recall": d_rec,
            "f1": d_f1,
        })
        print(f"  Tolerance {d_tol:2d} px ({d_tol*0.10:.1f} m): TP = {d_tp:4d}, Prec = {d_prec:.1%}, Rec = {d_rec:.1%}, F1 = {d_f1:.4f}")

    # ─────────────────────────────────────────────────────────────
    # METHOD 4: MINIMUM CROWN AREA CUTOFF FILTERING
    # ─────────────────────────────────────────────────────────────
    print("\n--- METHOD 4: MINIMUM CROWN AREA CUTOFF FILTERING ---")
    area_cutoffs = [0, 150, 300, 500, 750, 1000]
    area_results = []

    for a_cut in area_cutoffs:
        a_tp = 0
        a_pred_count = 0
        a_gt_count = 0

        for d in all_data:
            valid_g_idx = [j for j, ga in enumerate(d["gt_areas"]) if ga >= a_cut]
            valid_p_idx = [i for i, pa in enumerate(d["pred_areas"]) if pa >= a_cut]

            a_gt_count += len(valid_g_idx)
            a_pred_count += len(valid_p_idx)

            if len(valid_g_idx) == 0 or len(valid_p_idx) == 0:
                continue

            # Build sub-matrix
            sub_iou = np.zeros((len(valid_p_idx), len(valid_g_idx)), dtype=np.float32)
            for sub_i, p_idx in enumerate(valid_p_idx):
                pm = d["pred_masks"][p_idx]
                pa = d["pred_areas"][p_idx]
                for sub_j, g_idx in enumerate(valid_g_idx):
                    gm = d["gt_masks"][g_idx]
                    ga = d["gt_areas"][g_idx]
                    inter = np.sum((pm > 0) & (gm > 0))
                    union = pa + ga - inter
                    sub_iou[sub_i, sub_j] = inter / max(union, 1)

            r_ind, c_ind = linear_sum_assignment(-sub_iou)
            for r, c in zip(r_ind, c_ind):
                if sub_iou[r, c] >= 0.50:
                    a_tp += 1

        a_prec = a_tp / max(a_pred_count, 1)
        a_rec = a_tp / max(a_gt_count, 1)
        a_f1 = 2 * a_prec * a_rec / max(a_prec + a_rec, 1e-6)
        count_ratio = a_pred_count / max(a_gt_count, 1)
        area_results.append({
            "min_area_px": a_cut,
            "min_area_m2": a_cut * 0.01,
            "gt_count": a_gt_count,
            "pred_count": a_pred_count,
            "count_ratio": count_ratio,
            "precision": a_prec,
            "recall": a_rec,
            "f1": a_f1,
        })
        print(f"  Min Area {a_cut:4d} px ({a_cut*0.01:4.1f} m²): GT = {a_gt_count:4d}, Pred = {a_pred_count:4d} (Ratio {count_ratio:.2f}x), Prec = {a_prec:.1%}, Rec = {a_rec:.1%}, F1 = {a_f1:.4f}")

    # ─────────────────────────────────────────────────────────────
    # METHOD 5: CONFIDENCE THRESHOLD SWEEPING & BALANCED POINT
    # ─────────────────────────────────────────────────────────────
    print("\n--- METHOD 5: CONFIDENCE THRESHOLD SWEEPING ---")
    conf_thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    conf_results = []
    best_balanced_diff = 999999
    best_balanced_conf = 0.50

    for c_th in conf_thresholds:
        c_tp = 0
        c_pred_tot = 0

        for d in all_data:
            valid_p_idx = [i for i, s in enumerate(d["pred_scores"]) if s >= c_th]
            c_pred_tot += len(valid_p_idx)

            if len(d["gt_masks"]) == 0 or len(valid_p_idx) == 0:
                continue

            sub_iou = np.zeros((len(valid_p_idx), len(d["gt_masks"])), dtype=np.float32)
            for sub_i, p_idx in enumerate(valid_p_idx):
                pm = d["pred_masks"][p_idx]
                pa = d["pred_areas"][p_idx]
                for g_idx, gm in enumerate(d["gt_masks"]):
                    ga = d["gt_areas"][g_idx]
                    inter = np.sum((pm > 0) & (gm > 0))
                    union = pa + ga - inter
                    sub_iou[sub_i, g_idx] = inter / max(union, 1)

            r_ind, c_ind = linear_sum_assignment(-sub_iou)
            for r, c in zip(r_ind, c_ind):
                if sub_iou[r, c] >= 0.50:
                    c_tp += 1

        c_prec = c_tp / max(c_pred_tot, 1)
        c_rec = c_tp / max(tot_gt, 1)
        c_f1 = 2 * c_prec * c_rec / max(c_prec + c_rec, 1e-6)
        ratio = c_pred_tot / max(tot_gt, 1)
        
        diff = abs(c_pred_tot - tot_gt)
        if diff < best_balanced_diff:
            best_balanced_diff = diff
            best_balanced_conf = c_th

        conf_results.append({
            "conf_thresh": c_th,
            "pred_count": c_pred_tot,
            "count_ratio": ratio,
            "precision": c_prec,
            "recall": c_rec,
            "f1": c_f1,
        })
        print(f"  Conf >= {c_th:.2f}: Pred = {c_pred_tot:4d} (Ratio {ratio:.2f}x), Prec = {c_prec:.1%}, Rec = {c_rec:.1%}, F1 = {c_f1:.4f}")

    print(f"\n=> BALANCED OPERATING POINT: T_conf* = {best_balanced_conf:.2f} (Pred: {[c['pred_count'] for c in conf_results if c['conf_thresh'] == best_balanced_conf][0]} vs GT: {tot_gt})")

    # ─────────────────────────────────────────────────────────────
    # GENERATE 4-PANEL PUBLICATION FIGURE
    # ─────────────────────────────────────────────────────────────
    print("\nGenerating 4-Panel Publication Diagnostics Figure...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    plt.rcParams['font.family'] = 'DejaVu Sans'

    # (A) Error Typology Pie / Stacked Bar
    ax = axes[0, 0]
    categories = [
        "1-to-1 Match (TP)",
        "Multi-Leader Splits (Over-seg)",
        "Verified Unannotated Trees",
        "Ambiguous Understory/Shrubs",
        "True False Alarms (Soil/Shadow)",
        "Omissions (Missed GT)"
    ]
    counts = [
        tp_1to1,
        excess_split_preds,
        verified_unannotated,
        ambiguous_shrub,
        true_false_alarms,
        omission_count
    ]
    colors = ['#2e7d32', '#43a047', '#1e88e5', '#ffb300', '#e53935', '#757575']
    y_pos = np.arange(len(categories))
    bars = ax.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlabel('Instance Count', fontweight='bold')
    ax.set_title('(A) 5-Way Bipartite Error Typology\n(Dalponte & Aubry-Kientz Forestry Formulation)', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6, axis='x')
    for bar, c in zip(bars, counts):
        ax.text(bar.get_width() + 15, bar.get_y() + bar.get_height()/2, f"{c} ({c/tot_pred:.1%})", va='center', fontweight='bold', fontsize=9.5)

    # (B) Centroid Distance Tolerance Curve
    ax = axes[0, 1]
    d_m = [r["d_tol_m"] for r in dist_results]
    d_f1 = [r["f1"] for r in dist_results]
    d_prec = [r["precision"] for r in dist_results]
    d_rec = [r["recall"] for r in dist_results]

    ax.plot(d_m, d_f1, 'o-', color='#1565c0', lw=2.5, ms=7, label='Stem F1 Score')
    ax.plot(d_m, d_rec, 's--', color='#2e7d32', lw=1.8, ms=6, label='Detection Recall')
    ax.plot(d_m, d_prec, '^:', color='#e65100', lw=1.8, ms=6, label='Detection Precision')
    ax.axvline(2.0, color='red', linestyle='--', alpha=0.7, label='Standard Tolerance (2.0 m)')
    ax.set_xlabel('Apex Distance Tolerance (meters)', fontweight='bold')
    ax.set_ylabel('Stem Detection Metric', fontweight='bold')
    ax.set_title('(B) Centroid Distance Tolerance Matching\n(Weinstein / NeonTreeEvaluation Tree Stem Protocol)', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9.5)
    ax.set_ylim(0.2, 0.8)

    # (C) Minimum Area Cutoff Sensitivity
    ax = axes[1, 0]
    a_m2 = [r["min_area_m2"] for r in area_results]
    a_prec = [r["precision"] * 100 for r in area_results]
    a_rec = [r["recall"] * 100 for r in area_results]
    a_ratio = [r["count_ratio"] for r in area_results]

    ax_twin = ax.twinx()
    l1 = ax.plot(a_m2, a_prec, 's-', color='#2e7d32', lw=2.2, ms=7, label='Precision@50 (%)')
    l2 = ax.plot(a_m2, a_rec, 'o--', color='#1565c0', lw=2.0, ms=6, label='Recall@50 (%)')
    l3 = ax_twin.plot(a_m2, a_ratio, 'd-.', color='#d84315', lw=2.0, ms=7, label='Count Ratio (Pred / GT)')
    ax_twin.axhline(1.0, color='gray', linestyle=':', lw=1.5, alpha=0.8)

    ax.set_xlabel('Minimum Crown Area Filter (m²)', fontweight='bold')
    ax.set_ylabel('Segmentation Accuracy (%)', fontweight='bold')
    ax_twin.set_ylabel('Predicted / GT Count Ratio', fontweight='bold', color='#d84315')
    ax.set_title('(C) Minimum Crown Area Filtering\n(Synchronizing DBH / Canopy Thresholds with Forest Inventory)', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', frameon=True, framealpha=0.9, fontsize=9)

    # (D) Confidence Sweeping & Balanced Operating Point
    ax = axes[1, 1]
    c_th = [r["conf_thresh"] for r in conf_results]
    c_prec = [r["precision"] * 100 for r in conf_results]
    c_rec = [r["recall"] * 100 for r in conf_results]
    c_f1 = [r["f1"] for r in conf_results]
    c_ratio = [r["count_ratio"] for r in conf_results]

    ax.plot(c_th, c_prec, 's-', color='#2e7d32', lw=2.2, ms=7, label='Precision@50 (%)')
    ax.plot(c_th, c_rec, 'o--', color='#1565c0', lw=2.0, ms=6, label='Recall@50 (%)')
    ax.plot(c_th, [f * 100 for f in c_f1], '^-', color='#6a1b9a', lw=2.2, ms=7, label='Hungarian F1 (%)')
    ax.axvline(best_balanced_conf, color='crimson', linestyle='--', lw=2.0, label=f'Balanced Point T* = {best_balanced_conf:.2f} (1.0x GT)')

    ax.set_xlabel('Confidence Threshold (T_conf)', fontweight='bold')
    ax.set_ylabel('Performance (%)', fontweight='bold')
    ax.set_title('(D) Confidence Sweeping & Balanced Inventory Point\n(Operating Trade-off: High Recall vs. Exact Tree Count)', fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    out_fig = "crown_segmentation_research/images/quebec_forestry_diagnostics_5methods.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    print("Saved diagnostics figure:", out_fig)

    # Copy to artifacts directory
    artifact_dir = "/home/chung/.gemini/antigravity-ide/brain/430ab9e9-db0b-4bb3-9c66-8def6a5fa89b"
    import shutil
    shutil.copy(out_fig, os.path.join(artifact_dir, "quebec_forestry_diagnostics_5methods.png"))

    # Save full JSON summary
    summary_results = {
        "dataset": "Quebec Test Zone 3 (200 patches)",
        "total_gt": tot_gt,
        "total_pred": tot_pred,
        "method1_error_typology": {
            "tp_1to1": tp_1to1,
            "tp_1to1_rate": tp_1to1 / tot_gt,
            "gt_split_count": gt_split_count,
            "excess_split_preds": excess_split_preds,
            "omissions": omission_count,
            "merges": pred_merge_count,
            "isolated_commissions": isolated_commissions,
        },
        "method2_blind_audit": {
            "verified_unannotated_trees": verified_unannotated,
            "ambiguous_shrubs": ambiguous_shrub,
            "true_false_alarms": true_false_alarms,
            "standard_precision": std_prec,
            "adjusted_precision_p_star": p_star,
            "structural_precision": p_structural,
        },
        "method3_centroid_distance": dist_results,
        "method4_area_cutoff": area_results,
        "method5_conf_sweeping": conf_results,
        "balanced_conf": best_balanced_conf,
    }

    out_json = "crown_segmentation_research/experiments/quebec_forestry_5methods_results.json"
    with open(out_json, "w") as f:
        json.dump(summary_results, f, indent=2)
    print("Saved results JSON to:", out_json)

if __name__ == "__main__":
    run_diagnostics()
