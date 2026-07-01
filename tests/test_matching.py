"""
tests/test_matching.py
======================
Unit tests for Stage 3 token matching.

Run with:
    cd /home/chung/RS/phase1
    python -m pytest tests/test_matching.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# ── ensure parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import torch

from token_matching_utils import (
    centroid_distance_matrix,
    compute_mask_iou,
    compute_match_metrics,
    cosine_similarity_matrix,
    cross_attention_matching,
    detect_splits_merges,
    fused_similarity_matrix,
    normalize_embeddings,
    sinkhorn_normalize,
    soft_matrix_softmax,
    soft_matrix_sinkhorn,
    topk_pruned_cost_matrix,
)
from token_matching import MatchConfig, TokenMatcher


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    torch.manual_seed(42)
    return torch.Generator().manual_seed(42)


def _rand_tokens(N: int, D: int = 256) -> torch.Tensor:
    return torch.randn(N, D)


def _rand_centroids(N: int) -> torch.Tensor:
    return torch.rand(N, 2)


def _rand_areas(N: int) -> torch.Tensor:
    a = torch.rand(N)
    return a / a.sum()


# ─────────────────────────────────────────────────────────────────────────────
# 1. normalize_embeddings
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalizeEmbeddings:
    def test_unit_norm(self):
        tokens = _rand_tokens(20)
        normed = normalize_embeddings(tokens)
        norms  = normed.norm(dim=1)
        assert torch.allclose(norms, torch.ones(20), atol=1e-5), \
            "Expected unit-norm rows after normalize_embeddings"

    def test_zero_vector_safe(self):
        """Zero vector should not produce NaN."""
        tokens = torch.zeros(5, 8)
        normed = normalize_embeddings(tokens)
        assert not torch.isnan(normed).any(), "NaN produced from zero embedding"


# ─────────────────────────────────────────────────────────────────────────────
# 2. cosine_similarity_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineSimilarityMatrix:
    def test_shape(self):
        A = normalize_embeddings(_rand_tokens(10))
        B = normalize_embeddings(_rand_tokens(15))
        S = cosine_similarity_matrix(A, B)
        assert S.shape == (10, 15)

    def test_self_similarity_is_one(self):
        A = normalize_embeddings(_rand_tokens(8, D=64))
        S = cosine_similarity_matrix(A, A)
        diag = S.diagonal()
        assert torch.allclose(diag, torch.ones(8), atol=1e-5), \
            "Self-cosine should be 1"

    def test_range(self):
        A = normalize_embeddings(_rand_tokens(12))
        B = normalize_embeddings(_rand_tokens(7))
        S = cosine_similarity_matrix(A, B)
        assert S.min() >= -1.01 and S.max() <= 1.01


# ─────────────────────────────────────────────────────────────────────────────
# 3. centroid_distance_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestCentroidDistanceMatrix:
    def test_shape(self):
        C1 = _rand_centroids(10)
        C2 = _rand_centroids(14)
        D  = centroid_distance_matrix(C1, C2)
        assert D.shape == (10, 14)

    def test_self_distance_is_zero(self):
        C = _rand_centroids(8)
        D = centroid_distance_matrix(C, C)
        assert torch.allclose(D.diagonal(), torch.zeros(8), atol=1e-5), \
            "Self-distance should be 0"

    def test_non_negative(self):
        C1 = _rand_centroids(6)
        C2 = _rand_centroids(9)
        D  = centroid_distance_matrix(C1, C2)
        assert (D >= 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 4. topk_pruned_cost_matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestTopkPruning:
    def test_keeps_k_per_row(self):
        sim = torch.randn(10, 20)
        k   = 5
        pruned = topk_pruned_cost_matrix(sim, top_k=k, fill_val=-1e6)
        # Each row: exactly k entries should differ from fill_val
        non_fill = (pruned > -1e5).sum(dim=1)
        assert (non_fill == k).all(), f"Expected {k} non-fill per row, got {non_fill}"

    def test_shape_unchanged(self):
        sim    = torch.randn(8, 12)
        pruned = topk_pruned_cost_matrix(sim, top_k=4)
        assert pruned.shape == sim.shape

    def test_top_values_preserved(self):
        sim    = torch.randn(5, 10)
        k      = 3
        pruned = topk_pruned_cost_matrix(sim, top_k=k, fill_val=-1e6)
        for r in range(5):
            real_topk = sim[r].topk(k).values
            pruned_vals = pruned[r][pruned[r] > -1e5].sort(descending=True).values
            assert torch.allclose(real_topk.sort(descending=True).values,
                                  pruned_vals, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. soft_matrix_softmax
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftMatrixSoftmax:
    def test_shape(self):
        sim  = torch.randn(8, 12)
        soft = soft_matrix_softmax(sim, temperature=0.1)
        assert soft.shape == (8, 12)

    def test_rows_sum_to_one(self):
        sim  = torch.randn(10, 15)
        soft = soft_matrix_softmax(sim, temperature=0.1)
        row_sums = soft.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(10), atol=1e-5), \
            "Softmax rows must sum to 1"

    def test_non_negative(self):
        soft = soft_matrix_softmax(torch.randn(6, 9), temperature=0.2)
        assert (soft >= 0).all()

    def test_gated_minus_inf_to_zero(self):
        """Entries set to -inf (spatial gate) should become 0 after softmax."""
        sim = torch.randn(4, 6)
        sim[0, :] = float("-inf")
        sim[0, 2] = 1.0   # one valid entry
        soft = soft_matrix_softmax(sim, temperature=0.1)
        # Row 0: only idx 2 should be non-zero
        assert soft[0, 2] > 0.99


# ─────────────────────────────────────────────────────────────────────────────
# 6. sinkhorn_normalize
# ─────────────────────────────────────────────────────────────────────────────

class TestSinkhornNormalize:
    def test_doubly_stochastic(self):
        M = torch.rand(8, 8)
        S = sinkhorn_normalize(M, n_iters=50)
        row_sums = S.sum(dim=1)
        col_sums = S.sum(dim=0)
        assert torch.allclose(row_sums, torch.ones(8), atol=1e-3), \
            f"Row sums: {row_sums}"
        assert torch.allclose(col_sums, torch.ones(8), atol=1e-3), \
            f"Col sums: {col_sums}"

    def test_non_negative_output(self):
        M = torch.rand(6, 9)
        S = sinkhorn_normalize(M)
        assert (S >= 0).all()

    def test_handles_minus_inf(self):
        """Should not crash or produce NaN when -inf gating entries exist."""
        M = torch.randn(5, 7)
        M[0, :] = float("-inf")
        S = sinkhorn_normalize(M.exp())
        assert not torch.isnan(S).any()


# ─────────────────────────────────────────────────────────────────────────────
# 7. Hungarian matching (via TokenMatcher)
# ─────────────────────────────────────────────────────────────────────────────

class TestHungarianMatching:
    def _make_data(self, N, D=64):
        return {
            "tokens":    _rand_tokens(N, D),
            "centroids": _rand_centroids(N),
            "areas":     _rand_areas(N),
        }

    def test_unique_indices(self):
        cfg = MatchConfig(method="hungarian", device="cpu", top_k=10,
                          hungarian_threshold=-10.0, spatial_gate_dist=0.0)
        matcher = TokenMatcher(cfg)
        d1 = self._make_data(15)
        d2 = self._make_data(20)
        res = matcher.match(d1, d2)
        t1_idxs = [p[0] for p in res["pairs"]]
        t2_idxs = [p[1] for p in res["pairs"]]
        assert len(t1_idxs) == len(set(t1_idxs)), "T1 indices in pairs must be unique"
        assert len(t2_idxs) == len(set(t2_idxs)), "T2 indices in pairs must be unique"

    def test_output_keys(self):
        cfg = MatchConfig(method="hungarian", device="cpu", spatial_gate_dist=0.0)
        matcher = TokenMatcher(cfg)
        res = matcher.match(self._make_data(8), self._make_data(10))
        assert "pairs"       in res
        assert "soft_matrix" in res
        assert "unmatched_T1" in res
        assert "unmatched_T2" in res
        assert "metadata"    in res

    def test_threshold_filters_weak_matches(self):
        cfg = MatchConfig(method="hungarian", device="cpu",
                          hungarian_threshold=0.99, spatial_gate_dist=0.0,
                          top_k=10)
        matcher = TokenMatcher(cfg)
        d1 = self._make_data(10)
        d2 = self._make_data(10)
        res = matcher.match(d1, d2)
        for p in res["pairs"]:
            assert p[2] >= 0.99, f"Score {p[2]} below threshold 0.99"

    def test_soft_matrix_shape(self):
        cfg = MatchConfig(method="hungarian", device="cpu", spatial_gate_dist=0.0)
        matcher = TokenMatcher(cfg)
        d1 = self._make_data(7)
        d2 = self._make_data(11)
        res = matcher.match(d1, d2)
        assert res["soft_matrix"].shape == (7, 11)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Nearest-neighbour matching
# ─────────────────────────────────────────────────────────────────────────────

class TestNearestNeighborMatching:
    def test_at_most_topk_per_token(self):
        cfg = MatchConfig(method="nearest_neighbor", device="cpu",
                          top_k=3, hungarian_threshold=-10.0,
                          spatial_gate_dist=0.0)
        matcher = TokenMatcher(cfg)
        d1 = {"tokens": _rand_tokens(5), "centroids": _rand_centroids(5),
              "areas":  _rand_areas(5)}
        d2 = {"tokens": _rand_tokens(8), "centroids": _rand_centroids(8),
              "areas":  _rand_areas(8)}
        res = matcher.match(d1, d2)
        from collections import Counter
        t1_counts = Counter(p[0] for p in res["pairs"])
        assert all(v <= 3 for v in t1_counts.values()), "Each T1 token has at most top_k matches"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Soft matching method
# ─────────────────────────────────────────────────────────────────────────────

class TestSoftMatching:
    def _make_data(self, N, D=64):
        return {
            "tokens":    _rand_tokens(N, D),
            "centroids": _rand_centroids(N),
            "areas":     _rand_areas(N),
        }

    def test_softmax_output_shape(self):
        cfg = MatchConfig(method="soft", soft_sub_method="softmax",
                          device="cpu", spatial_gate_dist=0.0)
        matcher = TokenMatcher(cfg)
        res = matcher.match(self._make_data(6), self._make_data(9))
        assert res["soft_matrix"].shape == (6, 9)

    def test_sinkhorn_output_shape(self):
        cfg = MatchConfig(method="soft", soft_sub_method="sinkhorn",
                          device="cpu", spatial_gate_dist=0.0)
        matcher = TokenMatcher(cfg)
        res = matcher.match(self._make_data(7), self._make_data(7))
        assert res["soft_matrix"].shape == (7, 7)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Spatial gating
# ─────────────────────────────────────────────────────────────────────────────

class TestSpatialGating:
    def test_distant_pairs_rejected(self):
        """Tokens placed far apart should have very high gated_fraction."""
        c1 = torch.tensor([[0.0, 0.0], [0.05, 0.05]])
        c2 = torch.tensor([[0.9, 0.9], [0.95, 0.95]])
        tok1_n = normalize_embeddings(_rand_tokens(2, D=64))
        tok2_n = normalize_embeddings(_rand_tokens(2, D=64))
        _, _, stats = fused_similarity_matrix(
            tok1_n, tok2_n, c1, c2,
            alpha=1.0, beta=0.5, spatial_gate_dist=0.3
        )
        # All pairs are >0.3 away, so gated_fraction should be 1.0
        assert stats["gated_fraction"] == pytest.approx(1.0, abs=0.01)

    def test_nearby_pairs_not_gated(self):
        """Tokens at same location should have gated_fraction = 0."""
        c1 = torch.tensor([[0.3, 0.3]])
        c2 = torch.tensor([[0.31, 0.31]])
        tok1_n = normalize_embeddings(_rand_tokens(1, D=64))
        tok2_n = normalize_embeddings(_rand_tokens(1, D=64))
        _, _, stats = fused_similarity_matrix(
            tok1_n, tok2_n, c1, c2,
            alpha=1.0, beta=0.5, spatial_gate_dist=0.3
        )
        assert stats["gated_fraction"] == pytest.approx(0.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Split / merge detection
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitMergeDetection:
    def test_split_detected(self):
        """One T1 token matched to two T2 tokens with large combined area → split."""
        areas_t1 = torch.tensor([0.2, 0.1])
        areas_t2 = torch.tensor([0.12, 0.12, 0.05])
        # T1[0] → T2[0], T1[0] → T2[1]  (combined 0.24 > 0.6 * 0.2)
        pairs = [(0, 0, 0.9), (0, 1, 0.8), (1, 2, 0.7)]
        result = detect_splits_merges(pairs, areas_t1, areas_t2, area_ratio_threshold=0.6)
        assert 0 in result["splits"]

    def test_no_split_when_area_small(self):
        areas_t1 = torch.tensor([0.5])
        areas_t2 = torch.tensor([0.01, 0.01])
        pairs = [(0, 0, 0.9), (0, 1, 0.8)]
        result = detect_splits_merges(pairs, areas_t1, areas_t2, area_ratio_threshold=0.6)
        assert 0 not in result["splits"]

    def test_merge_detected(self):
        """Two T1 tokens both matching one T2 token → merge."""
        areas_t1 = torch.tensor([0.15, 0.15])
        areas_t2 = torch.tensor([0.3])
        pairs = [(0, 0, 0.9), (1, 0, 0.85)]
        result = detect_splits_merges(pairs, areas_t1, areas_t2, area_ratio_threshold=0.6)
        assert 0 in result["merges"]


# ─────────────────────────────────────────────────────────────────────────────
# 12. Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMatchMetrics:
    def test_perfect_match_with_gt(self):
        pairs    = [[0, 0, 1.0], [1, 1, 0.9], [2, 2, 0.8]]
        gt_pairs = [(0, 0), (1, 1), (2, 2)]
        m = compute_match_metrics(pairs, N1=3, N2=3, gt_pairs=gt_pairs)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"]    == pytest.approx(1.0)
        assert m["F1"]        == pytest.approx(1.0, abs=1e-4)

    def test_no_gt_returns_none(self):
        pairs = [[0, 1, 0.8], [1, 2, 0.7]]
        m = compute_match_metrics(pairs, N1=3, N2=4)
        assert m["precision"] is None

    def test_unmatched_count(self):
        pairs = [[0, 0, 0.9]]           # only 1 of 3 T1 matched
        m = compute_match_metrics(pairs, N1=3, N2=4)
        assert m["unmatched_t1"] == 2
        assert m["unmatched_t2"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# 13. IoU helper
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskIoU:
    def test_identical_masks(self):
        mask = np.ones((10, 10), dtype=bool)
        assert compute_mask_iou(mask, mask) == pytest.approx(1.0)

    def test_disjoint_masks(self):
        a = np.zeros((10, 10), dtype=bool)
        b = np.zeros((10, 10), dtype=bool)
        a[:5, :5] = True
        b[5:, 5:] = True
        assert compute_mask_iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.zeros((4, 4), dtype=bool)
        b = np.zeros((4, 4), dtype=bool)
        a[0:2, 0:2] = True   # 4 pixels
        b[1:3, 1:3] = True   # 4 pixels, 1 overlap
        iou = compute_mask_iou(a, b)
        # intersection=1, union=7
        assert iou == pytest.approx(1 / 7, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# 14. End-to-end CPU integration test
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEndCPU:
    """Full pipeline test: load synthetic data, match, check output contract."""

    def _make_data(self, N, D=256):
        return {
            "tokens":    _rand_tokens(N, D),
            "centroids": _rand_centroids(N),
            "areas":     _rand_areas(N),
        }

    @pytest.mark.parametrize("method", [
        "cosine_spatial", "nearest_neighbor", "hungarian", "soft", "cross_attention"
    ])
    def test_all_methods_cpu(self, method):
        cfg = MatchConfig(
            method=method, device="cpu",
            spatial_gate_dist=0.0,   # disable gate so all pairs are valid
            hungarian_threshold=-2.0,
            top_k=10,
        )
        matcher = TokenMatcher(cfg)
        d1 = self._make_data(12)
        d2 = self._make_data(15)
        res = matcher.match(d1, d2)

        # Contract checks
        assert "pairs"       in res
        assert "soft_matrix" in res
        assert "unmatched_T1" in res
        assert "unmatched_T2" in res
        assert "metadata"    in res

        assert res["soft_matrix"].shape == (12, 15)
        assert res["metadata"]["n_t1"] == 12
        assert res["metadata"]["n_t2"] == 15

        # No NaN in soft matrix
        assert not torch.isnan(res["soft_matrix"]).any(), \
            f"NaN in soft_matrix for method={method}"

        # Pair indices in valid range
        for p in res["pairs"]:
            assert 0 <= p[0] < 12, f"T1 idx {p[0]} out of range"
            assert 0 <= p[1] < 15, f"T2 idx {p[1]} out of range"

    def test_metadata_has_stats(self):
        cfg = MatchConfig(method="hungarian", device="cpu",
                          spatial_gate_dist=0.0)
        matcher = TokenMatcher(cfg)
        res = matcher.match(self._make_data(8), self._make_data(8))
        meta = res["metadata"]
        assert "mean_cosine"    in meta
        assert "std_cosine"     in meta
        assert "gated_fraction" in meta
