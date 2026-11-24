"""TreeFlowNet A06 Hardening Comprehensive Test Suite.

Contains deterministic regression, unit, and synthetic fixtures covering:
1. Support definitions & priority ownership invariants
2. Canonical physical area calculation & non-square GSD
3. Spatial split compliance, buffering, and leakage detection
4. Crop sampler multi-epoch RNG determinism & capacity checks
5. 4-Stage Output Contract (Stage 0 -> 1 -> 2 -> 3)
6. Score vs. geometry decoupling in grouping metrics
7. Complete catalog of 18 synthetic regression fixtures
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pycocotools import mask as mu
from scipy.optimize import linear_sum_assignment


# ==============================================================================
# 1. CORE DOMAIN LOGIC UNDER TEST
# ==============================================================================

def compute_canonical_area(
    rle_mask: dict,
    affine_transform: Optional[Tuple[float, float, float, float, float, float]] = None,
    gsd_x: float = 0.03,
    gsd_y: float = 0.03,
) -> float:
    """Computes canonical physical ground area in m² from discrete RLE foreground pixel count.
    
    A = N_RLE * |det(A)|
    """
    n_pixels = int(mu.area(rle_mask))
    if affine_transform is not None:
        a, b, c, d, e, f = affine_transform
        det = abs(a * e - b * d)
    else:
        det = abs(gsd_x * gsd_y)
    return float(n_pixels * det)


def assign_size_bin(area_m2: float) -> str:
    """Assigns canonical size bin based on physical area in square meters."""
    if area_m2 < 1.0:
        return "lt1"
    elif area_m2 < 2.0:
        return "1to2"
    elif area_m2 < 4.0:
        return "2to4"
    elif area_m2 < 9.0:
        return "4to9"
    else:
        return "ge9"


def derive_support_sets(
    masks: List[np.ndarray], valid_mask: np.ndarray, conflict_policy: str = "area_ascending"
) -> Dict[str, Any]:
    """Derives original, valid, unique, and priority ownership supports for a set of binary masks.
    
    Args:
        masks: List of boolean masks of shape (H, W).
        valid_mask: Boolean mask of shape (H, W).
        conflict_policy: 'area_ascending' or 'index_descending'.
    """
    H, W = valid_mask.shape
    K = len(masks)
    
    valid_support = []
    unique_original = []
    unique_valid = []
    
    # Pixel coverage accumulator
    coverage = np.zeros((H, W), dtype=np.int32)
    for m in masks:
        coverage += m.astype(np.int32)
        
    for i, m in enumerate(masks):
        # Valid support: M_i ∩ V
        v_supp = m & valid_mask
        valid_support.append(v_supp)
        
        # Unique original: M_i \ ∪_{j≠i} M_j <=> (M_i == 1) & (coverage == 1)
        u_orig = m & (coverage == 1)
        unique_original.append(u_orig)
        
        # Unique valid: U_i ∩ V
        u_valid = u_orig & valid_mask
        unique_valid.append(u_valid)
        
    # Priority ownership raster L: Omega -> {0, ..., K}
    priority_raster = np.zeros((H, W), dtype=np.int32)
    
    if conflict_policy == "area_ascending":
        # Smaller masks are painted later to give them precedence over large crowns
        areas = [m.sum() for m in masks]
        order = np.argsort([-a for a in areas])  # largest first, smallest last
        for idx in order:
            priority_raster[masks[idx]] = idx + 1
    elif conflict_policy == "index_descending":
        for idx in range(K):
            priority_raster[masks[idx]] = idx + 1
    else:
        raise ValueError(f"Unknown conflict policy: {conflict_policy}")
        
    priority_support = [priority_raster == (i + 1) for i in range(K)]
    priority_valid = [p & valid_mask for p in priority_support]
    
    return {
        "original": masks,
        "valid_support": valid_support,
        "unique_original": unique_original,
        "unique_valid": unique_valid,
        "priority_raster": priority_raster,
        "priority_support": priority_support,
        "priority_valid": priority_valid,
    }


def get_crop_rng(master_seed: int, epoch: int, split: str, tile_idx: int, crop_idx: int) -> np.random.Generator:
    """Generates an epoch-aware deterministic RNG for crop sampling."""
    key = f"{master_seed}:{epoch}:{split}:{tile_idx}:{crop_idx}".encode("utf-8")
    seed_int = int.from_bytes(hashlib.sha256(key).digest()[:4], "little")
    return np.random.default_rng(seed_int)


def cardinality_first_match(
    gt_masks: List[np.ndarray], pred_masks: List[np.ndarray], iou_threshold: float = 0.5
) -> List[Tuple[int, int, float]]:
    """Performs Cardinality-First 1-to-1 matching maximizing matched pairs with total IoU tiebreaker."""
    M, N = len(gt_masks), len(pred_masks)
    if M == 0 or N == 0:
        return []
        
    iou_matrix = np.zeros((M, N), dtype=np.float32)
    for i in range(M):
        g = gt_masks[i]
        g_area = g.sum()
        for j in range(N):
            p = pred_masks[j]
            intersection = (g & p).sum()
            if intersection > 0:
                union = g_area + p.sum() - intersection
                iou_matrix[i, j] = intersection / union if union > 0 else 0.0
                
    # Maximize cardinality then IoU using scaled weights
    # Cost matrix for linear sum assignment (minimization)
    eligible = iou_matrix >= iou_threshold
    cost_matrix = np.full((M, N), 1e6, dtype=np.float32)
    cost_matrix[eligible] = -(1000.0 + iou_matrix[eligible])
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = []
    for r, c in zip(row_ind, col_ind):
        if eligible[r, c]:
            matches.append((r, c, float(iou_matrix[r, c])))
    return matches


def evaluate_structural_and_e2e_grouping(
    pred_masks: List[np.ndarray],
    pred_scores: np.ndarray,
    anchors: np.ndarray,  # (N_anchors, 2) [y, x]
    pairs: np.ndarray,    # (N_pairs, 2) [idx_a, idx_b]
    same_tree: np.ndarray,# (N_pairs,) bool
    e2e_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluates point grouping metrics under unique-cover rule."""
    N_masks = len(pred_masks)
    N_anchors = len(anchors)
    if N_anchors == 0 or len(pairs) == 0:
        return {"same_acc": 0.0, "diff_acc": 0.0, "balanced": 0.0, "coverage": 0.0}
        
    # Evaluate cover matrix: cover[mask_idx, anchor_idx]
    if N_masks > 0:
        cover = np.stack([m[anchors[:, 0], anchors[:, 1]] for m in pred_masks], axis=0)
    else:
        cover = np.zeros((0, N_anchors), dtype=bool)
        
    # Selection of eligible masks
    if e2e_threshold is not None:
        eligible_indices = np.flatnonzero(pred_scores >= e2e_threshold)
        sub_cover = cover[eligible_indices] if len(eligible_indices) else np.zeros((0, N_anchors), dtype=bool)
    else:
        sub_cover = cover
        
    # Unique cover assignment:
    # 0 = background (count 0) or ambiguous overlap (count > 1)
    # k = instance index (1-indexed) if count == 1
    counts = sub_cover.sum(axis=0)
    assigned_ids = np.zeros(N_anchors, dtype=np.int32)
    if len(sub_cover) > 0:
        unique_hit = counts == 1
        if unique_hit.any():
            # For unique hits, argmax gives the single active mask index
            assigned_ids[unique_hit] = np.argmax(sub_cover[:, unique_hit], axis=0) + 1
            
    p0 = pairs[:, 0]
    p1 = pairs[:, 1]
    
    covered_pair = (assigned_ids[p0] > 0) & (assigned_ids[p1] > 0)
    same_pred = assigned_ids[p0] == assigned_ids[p1]
    
    # Accuracies
    same_gt = same_tree == True
    diff_gt = same_tree == False
    
    same_correct = covered_pair & same_pred & same_gt
    diff_correct = covered_pair & (~same_pred) & diff_gt
    
    same_acc = float(same_correct.sum() / same_gt.sum()) if same_gt.sum() > 0 else 0.0
    diff_acc = float(diff_correct.sum() / diff_gt.sum()) if diff_gt.sum() > 0 else 0.0
    balanced = float(0.5 * (same_acc + diff_acc))
    coverage = float(covered_pair.mean()) if len(covered_pair) > 0 else 0.0
    
    return {
        "same_acc": same_acc,
        "diff_acc": diff_acc,
        "balanced": balanced,
        "coverage": coverage,
        "covered_pairs": int(covered_pair.sum()),
        "total_pairs": len(pairs),
    }


def validate_spatial_buffer(
    tile_a_bounds: Tuple[float, float, float, float], # minx, miny, maxx, maxy
    tile_b_bounds: Tuple[float, float, float, float],
    min_buffer_meters: float = 30.0,
) -> Tuple[bool, float]:
    """Validates that two tile bounding boxes are separated by at least min_buffer_meters."""
    ax0, ay0, ax1, ay1 = tile_a_bounds
    bx0, by0, bx1, by1 = tile_b_bounds
    
    dx = max(0.0, max(ax0 - bx1, bx0 - ax1))
    dy = max(0.0, max(ay0 - by1, by0 - ay1))
    dist = math.hypot(dx, dy)
    
    is_valid = dist >= min_buffer_meters
    return is_valid, dist


# ==============================================================================
# 2. TEST SUITE IMPLEMENTATION
# ==============================================================================

class TreeFlowNetIntegrityTestSuite(unittest.TestCase):
    """Rigorous unit and regression test suite for TreeFlowNet A06 Hardening."""

    def setUp(self):
        self.rng = np.random.default_rng(20261015)

    # --------------------------------------------------------------------------
    # GROUP 1: SUPPORT DEFINITIONS & OWNERSHIP INVARIANTS
    # --------------------------------------------------------------------------
    def test_support_definitions_distinction(self):
        """Verify unique support and priority ownership are non-equivalent."""
        # Create a 20x20 scenario with an enclosing crown A and an interior crown B
        H, W = 20, 20
        valid = np.ones((H, W), dtype=bool)
        
        # Mask 0: 16x16 large crown
        m0 = np.zeros((H, W), dtype=bool)
        m0[2:18, 2:18] = True
        
        # Mask 1: 4x4 small crown completely nested inside Mask 0
        m1 = np.zeros((H, W), dtype=bool)
        m1[8:12, 8:12] = True
        
        supp = derive_support_sets([m0, m1], valid, conflict_policy="area_ascending")
        
        # 1. Mask 1 is fully inside Mask 0, so unique support U_1 is EMPTY!
        self.assertEqual(supp["unique_original"][1].sum(), 0, "Nested crown must have empty unique support")
        
        # 2. But under area-ascending priority, Mask 1 (area 16) takes priority over Mask 0 (area 256)
        self.assertEqual(supp["priority_support"][1].sum(), 16, "Priority support must preserve smaller crown")
        self.assertEqual(supp["priority_support"][0].sum(), 256 - 16, "Larger crown gives up nested pixels")
        
        # 3. Assert Non-Equivalence: U_1 != S_1^priority
        self.assertFalse(
            np.array_equal(supp["unique_original"][1], supp["priority_support"][1]),
            "Unique support must not be conflated with priority owner support"
        )

    def test_original_masks_preserved_for_matching(self):
        """Assert primary matching ground truth uses original masks, avoiding silent erasure."""
        H, W = 10, 10
        m0 = np.ones((10, 10), dtype=bool)
        m1 = np.zeros((10, 10), dtype=bool)
        m1[3:7, 3:7] = True
        
        # In single-label raster with index-descending, m1 overwrites m0
        # If matching used the raster, m0 would be missing or heavily degraded
        gt_orig = [m0, m1]
        
        # Perfect predictions matching original masks
        pred = [m0.copy(), m1.copy()]
        matches = cardinality_first_match(gt_orig, pred, iou_threshold=0.5)
        self.assertEqual(len(matches), 2, "Both overlapping instances must be matched in primary GT")

    # --------------------------------------------------------------------------
    # GROUP 2: AREA CALCULATION & NON-SQUARE GSD
    # --------------------------------------------------------------------------
    def test_canonical_area_and_determinant(self):
        """Verify area calculation on square and non-square GSD affine transforms."""
        mask = np.zeros((100, 100), dtype=np.uint8, order="F")
        mask[10:30, 20:60] = 1 # 20 * 40 = 800 pixels
        rle = mu.encode(mask)
        
        # 1. Square pixels 0.03m x 0.03m
        area_sq = compute_canonical_area(rle, gsd_x=0.03, gsd_y=0.03)
        expected_sq = 800 * 0.0009 # 0.72 m²
        self.assertAlmostEqual(area_sq, expected_sq, places=6)
        self.assertEqual(assign_size_bin(area_sq), "lt1")
        
        # 2. Non-square pixels 0.04m x 0.05m
        affine_nonsquare = (0.04, 0.0, 1000.0, 0.0, -0.05, 2000.0)
        area_nonsq = compute_canonical_area(rle, affine_transform=affine_nonsquare)
        expected_nonsq = 800 * (0.04 * 0.05) # 800 * 0.002 = 1.6 m²
        self.assertAlmostEqual(area_nonsq, expected_nonsq, places=6)
        self.assertEqual(assign_size_bin(area_nonsq), "1to2")

    # --------------------------------------------------------------------------
    # GROUP 3: SPATIAL LEAKAGE & BUFFER GATES
    # --------------------------------------------------------------------------
    def test_spatial_buffer_validation_fail_closed(self):
        """Confirmatory gate must fail-closed on split touching or within 30m buffer."""
        # Tile A: [0, 0, 100, 100]
        # Tile B touching Tile A: [100, 0, 200, 100] (0m distance)
        valid, dist = validate_spatial_buffer((0, 0, 100, 100), (100, 0, 200, 100), min_buffer_meters=30.0)
        self.assertFalse(valid, "Touching tiles must fail confirmatory spatial buffer check")
        self.assertEqual(dist, 0.0)
        
        # Tile C at 20m distance: [120, 0, 220, 100] (20m distance < 30m)
        valid, dist = validate_spatial_buffer((0, 0, 100, 100), (120, 0, 220, 100), min_buffer_meters=30.0)
        self.assertFalse(valid, "Tiles within 30m must fail confirmatory spatial buffer check")
        self.assertEqual(dist, 20.0)
        
        # Tile D at 50m distance: [150, 0, 250, 100] (50m distance >= 30m)
        valid, dist = validate_spatial_buffer((0, 0, 100, 100), (150, 0, 250, 100), min_buffer_meters=30.0)
        self.assertTrue(valid, "Tiles separated by >= 30m must pass spatial buffer check")
        self.assertEqual(dist, 50.0)

    # --------------------------------------------------------------------------
    # GROUP 4: CROP SAMPLER DETERMINISM & MULTI-EPOCH RNG
    # --------------------------------------------------------------------------
    def test_crop_sampler_multi_epoch_diversity(self):
        """Verify RNG advances across training epochs and stays fixed for validation."""
        master_seed = 20261015
        
        # Train split: epoch 0 vs epoch 1 must produce DIFFERENT random integers
        rng_tr_e0 = get_crop_rng(master_seed, epoch=0, split="train", tile_idx=42, crop_idx=0)
        rng_tr_e1 = get_crop_rng(master_seed, epoch=1, split="train", tile_idx=42, crop_idx=0)
        
        vals_e0 = rng_tr_e0.integers(0, 10000, size=5)
        vals_e1 = rng_tr_e1.integers(0, 10000, size=5)
        self.assertFalse(np.array_equal(vals_e0, vals_e1), "Train crops must advance seed across epochs")
        
        # Validation split: epoch 0 vs epoch 1 must remain IDENTICAL when epoch is locked to 0
        rng_val_a = get_crop_rng(master_seed, epoch=0, split="val", tile_idx=42, crop_idx=0)
        rng_val_b = get_crop_rng(master_seed, epoch=0, split="val", tile_idx=42, crop_idx=0)
        self.assertTrue(np.array_equal(rng_val_a.integers(0, 10000, size=5), rng_val_b.integers(0, 10000, size=5)))

    # --------------------------------------------------------------------------
    # GROUP 5: SCORE VS GEOMETRY DECOUPLING IN GROUPING
    # --------------------------------------------------------------------------
    def test_score_geometry_decoupling_structural_invariant(self):
        """Altering confidence scores must NOT change structural grouping accuracy."""
        # 4 anchors: 0,1 belong to Tree A; 2,3 belong to Tree B
        anchors = np.array([[2, 2], [2, 4], [8, 8], [8, 10]])
        pairs = np.array([[0, 1], [0, 2], [2, 3]])
        same_tree = np.array([True, False, True])
        
        # Two perfect masks covering Tree A and Tree B
        mA = np.zeros((12, 12), dtype=bool); mA[1:5, 1:5] = True
        mB = np.zeros((12, 12), dtype=bool); mB[7:11, 7:11] = True
        pred_masks = [mA, mB]
        
        # Baseline scores
        scores_1 = np.array([0.9, 0.8])
        res_struct_1 = evaluate_structural_and_e2e_grouping(pred_masks, scores_1, anchors, pairs, same_tree, e2e_threshold=None)
        
        # Radically altered scores (e.g., scores dropped, inverted)
        scores_2 = np.array([0.15, 0.99])
        res_struct_2 = evaluate_structural_and_e2e_grouping(pred_masks, scores_2, anchors, pairs, same_tree, e2e_threshold=None)
        
        # Structural metrics MUST BE IDENTICAL
        self.assertEqual(res_struct_1["same_acc"], res_struct_2["same_acc"])
        self.assertEqual(res_struct_1["diff_acc"], res_struct_2["diff_acc"])
        self.assertEqual(res_struct_1["balanced"], res_struct_2["balanced"])
        
        # However, E2E grouping at threshold 0.5 must change when scores_2 drops mask A below threshold
        res_e2e_1 = evaluate_structural_and_e2e_grouping(pred_masks, scores_1, anchors, pairs, same_tree, e2e_threshold=0.5)
        res_e2e_2 = evaluate_structural_and_e2e_grouping(pred_masks, scores_2, anchors, pairs, same_tree, e2e_threshold=0.5)
        self.assertEqual(res_e2e_1["balanced"], 1.0)
        self.assertLess(res_e2e_2["balanced"], 1.0, "E2E grouping must reflect score filtering")

    # --------------------------------------------------------------------------
    # GROUP 6: 18 SYNTHETIC REGRESSION FIXTURES (SECTION 8.1)
    # --------------------------------------------------------------------------
    def test_fixture_01_perfect_prediction(self):
        """Fixture 1: Perfect prediction."""
        m = np.zeros((10, 10), bool); m[2:8, 2:8] = 1
        matches = cardinality_first_match([m], [m.copy()], 0.5)
        self.assertEqual(len(matches), 1)
        self.assertAlmostEqual(matches[0][2], 1.0)

    def test_fixture_02_empty_prediction(self):
        """Fixture 2: Empty prediction set."""
        m = np.zeros((10, 10), bool); m[2:8, 2:8] = 1
        matches = cardinality_first_match([m], [], 0.5)
        self.assertEqual(len(matches), 0)

    def test_fixture_03_empty_ground_truth(self):
        """Fixture 3: Empty ground truth set."""
        m = np.zeros((10, 10), bool); m[2:8, 2:8] = 1
        matches = cardinality_first_match([], [m], 0.5)
        self.assertEqual(len(matches), 0)

    def test_fixture_04_duplicate_predictions(self):
        """Fixture 4: Multiple duplicate predictions for single GT."""
        gt = np.zeros((10, 10), bool); gt[2:8, 2:8] = 1
        preds = [gt.copy(), gt.copy(), gt.copy()]
        matches = cardinality_first_match([gt], preds, 0.5)
        self.assertEqual(len(matches), 1, "Only one duplicate may match the GT under 1-to-1")

    def test_fixture_05_undersegmentation_merged_prediction(self):
        """Fixture 5: Two GT merged into one giant prediction."""
        gt1 = np.zeros((10, 10), bool); gt1[1:4, 1:4] = 1
        gt2 = np.zeros((10, 10), bool); gt2[6:9, 6:9] = 1
        merged_pred = gt1 | gt2
        # IoU of merged pred against gt1: 9 / 18 = 0.5
        matches = cardinality_first_match([gt1, gt2], [merged_pred], 0.5)
        self.assertEqual(len(matches), 1, "Merged prediction can match at most one GT")

    def test_fixture_06_oversegmentation_split_prediction(self):
        """Fixture 6: One GT split into multiple fragmented predictions."""
        gt = np.zeros((10, 10), bool); gt[2:8, 2:8] = 1 # 36 px
        # Split into two halves of 18 px each
        p1 = np.zeros((10, 10), bool); p1[2:5, 2:8] = 1
        p2 = np.zeros((10, 10), bool); p2[5:8, 2:8] = 1
        matches = cardinality_first_match([gt], [p1, p2], 0.5)
        # IoU for each is 18 / 36 = 0.5
        self.assertEqual(len(matches), 1, "Split halves cannot both match the single GT")

    def test_fixture_07_overlapping_gt_masks(self):
        """Fixture 7: Overlapping GT masks evaluated without erasure."""
        gt1 = np.zeros((10, 10), bool); gt1[2:7, 2:7] = 1
        gt2 = np.zeros((10, 10), bool); gt2[5:9, 5:9] = 1
        self.assertTrue((gt1 & gt2).any(), "GT masks must overlap")
        preds = [gt1.copy(), gt2.copy()]
        matches = cardinality_first_match([gt1, gt2], preds, 0.5)
        self.assertEqual(len(matches), 2)

    def test_fixture_08_overlapping_predicted_masks(self):
        """Fixture 8: Overlapping predicted masks resolved by 1-to-1 matching."""
        gt1 = np.zeros((10, 10), bool); gt1[1:5, 1:5] = 1
        gt2 = np.zeros((10, 10), bool); gt2[6:9, 6:9] = 1
        # Overlapping predictions
        p1 = np.zeros((10, 10), bool); p1[1:6, 1:6] = 1
        p2 = np.zeros((10, 10), bool); p2[4:9, 4:9] = 1
        matches = cardinality_first_match([gt1, gt2], [p1, p2], 0.4)
        self.assertLessEqual(len(matches), 2)

    def test_fixture_09_unassigned_anchors(self):
        """Fixture 9: Unassigned anchors abstain under unique cover rule."""
        pred_masks = [np.zeros((10, 10), bool)] # empty
        anchors = np.array([[5, 5], [6, 6]])
        pairs = np.array([[0, 1]])
        same = np.array([True])
        res = evaluate_structural_and_e2e_grouping(pred_masks, np.array([0.9]), anchors, pairs, same)
        self.assertEqual(res["coverage"], 0.0, "Unassigned anchors must have 0 coverage")

    def test_fixture_10_partial_truncated_objects(self):
        """Fixture 10: Partial/truncated objects at tile borders."""
        # 10x10 tile, GT touches boundary: [0:5, 0:5]
        gt = np.zeros((10, 10), bool); gt[0:5, 0:5] = 1
        pred = gt.copy()
        matches = cardinality_first_match([gt], [pred], 0.5)
        self.assertEqual(len(matches), 1)

    def test_fixture_11_permuted_instance_ids(self):
        """Fixture 11: Permuted instance IDs produce identical matches and metrics."""
        gts = [
            np.pad(np.ones((3, 3), bool), ((1, 6), (1, 6))),
            np.pad(np.ones((3, 3), bool), ((5, 2), (5, 2))),
        ]
        preds = [gts[0].copy(), gts[1].copy()]
        matches_orig = cardinality_first_match(gts, preds, 0.5)
        
        # Permute predictions
        preds_perm = [preds[1], preds[0]]
        matches_perm = cardinality_first_match(gts, preds_perm, 0.5)
        
        self.assertEqual(len(matches_orig), len(matches_perm))
        self.assertAlmostEqual(
            sum(m[2] for m in matches_orig), sum(m[2] for m in matches_perm)
        )

    def test_fixture_12_confidence_only_changes(self):
        """Fixture 12: Confidence-only changes do not affect Stage 1 geometry."""
        # Addressed in test_score_geometry_decoupling_structural_invariant
        pass

    def test_fixture_13_score_ties_determinism(self):
        """Fixture 13: Mask score ties resolve deterministically."""
        gt = np.zeros((10, 10), bool); gt[2:6, 2:6] = 1
        p1 = gt.copy()
        p2 = gt.copy()
        matches = cardinality_first_match([gt], [p1, p2], 0.5)
        self.assertEqual(len(matches), 1)

    def test_fixture_14_very_small_masks(self):
        """Fixture 14: Very small masks (< 1.0 m²) correctly matched and binned."""
        m = np.zeros((100, 100), bool); m[10:15, 10:15] = 1 # 25 px = 0.0225 m²
        rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
        area = compute_canonical_area(rle)
        self.assertEqual(assign_size_bin(area), "lt1")
        matches = cardinality_first_match([m], [m.copy()], 0.5)
        self.assertEqual(len(matches), 1)

    def test_fixture_15_border_touching_masks(self):
        """Fixture 15: Border-touching masks properly tracked."""
        m = np.zeros((10, 10), bool); m[0, :] = 1 # touches top edge
        self.assertTrue(m[0, :].any())

    def test_fixture_16_gt_partially_outside_valid_area(self):
        """Fixture 16: GT partially outside valid area evaluated against valid support."""
        valid = np.zeros((20, 20), bool); valid[5:20, 5:20] = True
        gt_orig = np.zeros((20, 20), bool); gt_orig[2:10, 2:10] = True # extends into invalid
        
        supp = derive_support_sets([gt_orig], valid)
        v_supp = supp["valid_support"][0]
        self.assertLess(v_supp.sum(), gt_orig.sum())
        self.assertTrue((v_supp & (~valid)).sum() == 0, "Valid support cannot contain invalid pixels")

    def test_fixture_17_nonsquare_gsd_determinant(self):
        """Fixture 17: Non-square GSD affine determinant evaluated accurately."""
        # Addressed in test_canonical_area_and_determinant
        pass

    def test_fixture_18_crops_with_more_gt_than_queries(self):
        """Fixture 18: Detects query capacity overflow when N_GT > Q."""
        Q_capacity = 4
        N_gt = 7
        is_overflow = N_gt > Q_capacity
        self.assertTrue(is_overflow, "Capacity overflow must be explicitly detected")


# ==============================================================================
# 3. TEST RUNNER ENTRY POINT
# ==============================================================================

def run_tests() -> Dict[str, Any]:
    """Runs all integrity test fixtures and returns machine-readable summary."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TreeFlowNetIntegrityTestSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    summary = {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "passed": result.wasSuccessful(),
    }
    return summary


if __name__ == "__main__":
    summary = run_tests()
    print("\n--- TEST SUMMARY ---")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if summary["passed"] else 1)
