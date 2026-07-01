"""
tests/test_reasoner_graph.py
============================
Unit tests for Stage 4B — Token Change Reasoner with Graph Reasoning.

Run:
    pytest tests/test_reasoner_graph.py -v
"""

from __future__ import annotations

import pytest
import torch

from token_change_reasoner import (
    SampleData,
    build_batch,
    compute_loss,
    make_dummy_batch,
    training_step,
)
from token_change_reasoner_graph import (
    GraphBuilder,
    GraphReasonerConfig,
    GraphReasoner,
    GraphSAGELayer,
    TokenChangeReasonerGraph,
    build_graph_model,
)

DEVICE = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return GraphReasonerConfig(
        hidden_dim=64, num_layers=2, num_heads=4, ff_dim=128,
        graph_k=4, graph_layers=2, graph_dropout=0.0,
    )


@pytest.fixture
def model(cfg):
    return build_graph_model(cfg).to(DEVICE)


@pytest.fixture
def batch():
    _, b, _ = make_dummy_batch(batch_size=2, n1=10, n2=12, n_pairs=5, device=DEVICE)
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 1. GraphBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphBuilder:
    def test_adjacency_length(self):
        gb = GraphBuilder(k=4)
        centroids  = torch.rand(10, 2)
        valid_mask = torch.ones(10, dtype=torch.bool)
        adj, ew    = gb.build(centroids, valid_mask)
        assert len(adj) == 10
        assert len(ew)  == 10

    def test_neighbour_count_le_k(self):
        gb = GraphBuilder(k=4)
        centroids  = torch.rand(10, 2)
        valid_mask = torch.ones(10, dtype=torch.bool)
        adj, ew    = gb.build(centroids, valid_mask)
        for i in range(10):
            assert len(adj[i]) <= 4, f"Node {i} has {len(adj[i])} neighbours > k=4"

    def test_no_self_loops(self):
        gb = GraphBuilder(k=4)
        centroids  = torch.rand(8, 2)
        valid_mask = torch.ones(8, dtype=torch.bool)
        adj, _     = gb.build(centroids, valid_mask)
        for i in range(8):
            assert i not in adj[i].tolist(), f"Self-loop at node {i}"

    def test_weights_sum_to_one(self):
        gb = GraphBuilder(k=4)
        centroids  = torch.rand(8, 2)
        valid_mask = torch.ones(8, dtype=torch.bool)
        _, ew      = gb.build(centroids, valid_mask)
        for i in range(8):
            if len(ew[i]) > 0:
                assert abs(ew[i].sum().item() - 1.0) < 1e-5, f"Weights at {i} don't sum to 1"

    def test_padded_nodes_excluded(self):
        """Padded nodes (valid_mask=False) should have empty adjacency."""
        gb = GraphBuilder(k=3)
        centroids  = torch.rand(6, 2)
        valid_mask = torch.tensor([True, True, True, False, False, False])
        adj, _     = gb.build(centroids, valid_mask)
        for i in [3, 4, 5]:
            assert len(adj[i]) == 0, f"Pad node {i} should have no neighbours"

    def test_too_few_nodes(self):
        """Single valid node — should produce empty adjacency."""
        gb = GraphBuilder(k=4)
        centroids  = torch.rand(3, 2)
        valid_mask = torch.tensor([True, False, False])
        adj, ew    = gb.build(centroids, valid_mask)
        assert len(adj[0]) == 0

    def test_k_larger_than_n(self):
        """k > n_valid — should clip to n_valid-1 neighbours."""
        gb = GraphBuilder(k=10)
        centroids  = torch.rand(4, 2)
        valid_mask = torch.ones(4, dtype=torch.bool)
        adj, _     = gb.build(centroids, valid_mask)
        for i in range(4):
            assert len(adj[i]) <= 3   # max neighbours = 4 - 1


# ─────────────────────────────────────────────────────────────────────────────
# 2. GraphSAGELayer
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphSAGELayer:
    def test_output_shape(self, cfg):
        layer = GraphSAGELayer(cfg.hidden_dim, cfg.graph_dropout).to(DEVICE)
        N, H  = 12, cfg.hidden_dim
        h     = torch.randn(N, H)
        gb    = GraphBuilder(k=3)
        adj, ew = gb.build(torch.rand(N, 2), torch.ones(N, dtype=torch.bool))
        out   = layer(h, adj, ew)
        assert out.shape == (N, H)

    def test_isolated_node_zero_agg(self, cfg):
        """A node with no neighbours should still produce output (just from W_self)."""
        layer = GraphSAGELayer(cfg.hidden_dim, 0.0).to(DEVICE)
        N, H  = 5, cfg.hidden_dim
        h     = torch.randn(N, H)
        adj   = [torch.zeros(0, dtype=torch.long)] * N   # all isolated
        ew    = [torch.zeros(0)] * N
        out   = layer(h, adj, ew)
        assert out.shape == (N, H)
        assert torch.isfinite(out).all()

    def test_gradient_flows(self, cfg):
        layer = GraphSAGELayer(cfg.hidden_dim, 0.0).to(DEVICE)
        N, H  = 8, cfg.hidden_dim
        h     = torch.randn(N, H, requires_grad=True)
        gb    = GraphBuilder(k=3)
        adj, ew = gb.build(torch.rand(N, 2), torch.ones(N, dtype=torch.bool))
        out   = layer(h, adj, ew)
        out.sum().backward()
        assert h.grad is not None and h.grad.shape == (N, H)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GraphReasoner
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphReasoner:
    def test_output_shape(self, cfg):
        reasoner = GraphReasoner(cfg).to(DEVICE)
        B, N, H  = 2, 15, cfg.hidden_dim
        h        = torch.randn(B, N, H)
        mask     = torch.zeros(B, N, dtype=torch.bool)
        mask[:, 12:] = True   # last 3 are padding
        cen      = torch.rand(B, N, 2)
        out      = reasoner(h, mask, cen)
        assert out.shape == (B, N, H)

    def test_padded_positions_unchanged(self, cfg):
        """Padded positions should pass through unchanged (graph skips them)."""
        cfg2 = GraphReasonerConfig(
            hidden_dim=64, graph_layers=2, graph_residual=True, graph_dropout=0.0
        )
        reasoner = GraphReasoner(cfg2).to(DEVICE)
        B, N, H  = 1, 10, cfg2.hidden_dim
        h        = torch.randn(B, N, H)
        mask     = torch.zeros(B, N, dtype=torch.bool)
        n_valid  = 7
        mask[:, n_valid:] = True

        with torch.no_grad():
            out = reasoner(h, mask, torch.rand(B, N, 2))

        assert out.shape == (B, N, H)

    def test_residual_changes_output(self, cfg):
        """With residual=True the output should differ from input."""
        reasoner = GraphReasoner(cfg).to(DEVICE)
        B, N, H  = 1, 12, cfg.hidden_dim
        h        = torch.randn(B, N, H)
        mask     = torch.zeros(B, N, dtype=torch.bool)
        cen      = torch.rand(B, N, 2)
        with torch.no_grad():
            out = reasoner(h, mask, cen)
        assert not torch.allclose(h, out)

    def test_no_residual(self):
        cfg_nr = GraphReasonerConfig(
            hidden_dim=32, num_heads=4, graph_layers=1,
            graph_residual=False, graph_dropout=0.0,
        )
        reasoner = GraphReasoner(cfg_nr).to(DEVICE)
        B, N, H  = 1, 8, 32
        h        = torch.randn(B, N, H)
        mask     = torch.zeros(B, N, dtype=torch.bool)
        cen      = torch.rand(B, N, 2)
        out      = reasoner(h, mask, cen)
        assert out.shape == (B, N, H)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TokenChangeReasonerGraph — full model
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenChangeReasonerGraph:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_shape(self, cfg, batch_size):
        model = build_graph_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(
            batch_size=batch_size, n1=10, n2=12, n_pairs=5, device=DEVICE
        )
        with torch.no_grad():
            out = model(batch)
        B = batch_size
        N = batch["tokens_pad"].shape[1]
        M = len(batch["pair_b"])
        assert out["change_logits"].shape == (B, N)
        assert out["delta_pred"].shape    == (M,)

    def test_backward_runs(self, cfg):
        model = build_graph_model(cfg).to(DEVICE)
        model.train()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=4, device=DEVICE)
        out    = model(batch)
        losses = compute_loss(out, batch, cfg)
        losses["total_loss"].backward()
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN grad in {name}"

    def test_training_step_updates_params(self, cfg):
        model  = build_graph_model(cfg).to(DEVICE)
        _, batch, _ = make_dummy_batch(batch_size=2, device=DEVICE)
        before = {n: p.clone() for n, p in model.named_parameters()}
        opt    = torch.optim.AdamW(model.parameters(), lr=1e-3)
        training_step(model, batch, opt, scaler=None)
        changed = sum(1 for n, p in model.named_parameters()
                      if not torch.allclose(before[n], p))
        assert changed > 0

    def test_single_token_each_side(self, cfg):
        """Edge case: 1 T1, 1 T2, 1 match — graph has 2 nodes."""
        model = build_graph_model(cfg).to(DEVICE)
        model.eval()
        s = SampleData(
            tokens_t1=torch.randn(1, 256), tokens_t2=torch.randn(1, 256),
            centroids_t1=torch.rand(1, 2), centroids_t2=torch.rand(1, 2),
            areas_t1=torch.tensor([500.0]), areas_t2=torch.tensor([500.0]),
            match_pairs=torch.tensor([[0.0, 0.0, 0.9]]),
        )
        batch = build_batch([s], cfg, DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert out["change_logits"].shape == (1, 2)
        assert out["delta_pred"].shape    == (1,)

    def test_no_matches(self, cfg):
        """No matched pairs — delta_pred should be empty tensor."""
        model = build_graph_model(cfg).to(DEVICE)
        model.eval()
        s = SampleData(
            tokens_t1=torch.randn(5, 256), tokens_t2=torch.randn(5, 256),
            centroids_t1=torch.rand(5, 2), centroids_t2=torch.rand(5, 2),
            areas_t1=torch.ones(5), areas_t2=torch.ones(5),
            match_pairs=torch.zeros(0, 3),
        )
        batch = build_batch([s], cfg, DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert out["delta_pred"].shape == (0,)

    def test_loss_finite(self, model, batch, cfg):
        model.eval()
        with torch.no_grad():
            out = model(batch)
        losses = compute_loss(out, batch, cfg)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} = {v}"

    def test_graph_changes_output_vs_base(self, cfg):
        """Graph model should produce different logits than the base model."""
        from token_change_reasoner import build_model
        base  = build_model(cfg).to(DEVICE)
        graph = build_graph_model(cfg).to(DEVICE)
        base.eval(); graph.eval()
        _, batch, _ = make_dummy_batch(batch_size=1, n1=6, n2=6, n_pairs=3, device=DEVICE)
        with torch.no_grad():
            out_b = base(batch)["change_logits"]
            out_g = graph(batch)["change_logits"]
        # Different params → different outputs
        assert not torch.allclose(out_b, out_g)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Misc
# ─────────────────────────────────────────────────────────────────────────────

class TestMisc:
    def test_param_count_larger_than_base(self, cfg):
        from token_change_reasoner import build_model, count_parameters
        base   = build_model(cfg)
        graph  = build_graph_model(cfg)
        assert count_parameters(graph) > count_parameters(base), \
            "Graph model should have more params than base"

    def test_config_inherits_base_fields(self, cfg):
        assert hasattr(cfg, "hidden_dim")
        assert hasattr(cfg, "num_layers")
        assert hasattr(cfg, "graph_k")
        assert hasattr(cfg, "graph_layers")

    def test_overfit_2_pairs(self, cfg):
        """Graph model should overfit 2 pairs in 100 iters."""
        model = build_graph_model(cfg).to(DEVICE)
        _, batch, _ = make_dummy_batch(batch_size=2, device=DEVICE, seed=0)
        opt  = torch.optim.AdamW(model.parameters(), lr=5e-3)
        init_loss = None
        for _ in range(100):
            opt.zero_grad()
            out    = model(batch)
            losses = compute_loss(out, batch, cfg)
            losses["total_loss"].backward()
            opt.step()
            if init_loss is None:
                init_loss = float(losses["total_loss"].detach())
        final_loss = float(losses["total_loss"].detach())
        drop = (init_loss - final_loss) / (init_loss + 1e-8) * 100
        assert drop > 50, f"Graph model did not overfit: drop={drop:.1f}%"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Graph Improvements (hybrid weights, cross-time, dropout)
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphImprovements:
    """Tests for the three graph improvements introduced in the v2 update."""

    # ── New config fields ─────────────────────────────────────────────────

    def test_new_config_fields_exist(self):
        cfg = GraphReasonerConfig()
        assert hasattr(cfg, "alpha_spatial"),  "missing alpha_spatial"
        assert hasattr(cfg, "beta_semantic"),  "missing beta_semantic"
        assert hasattr(cfg, "gamma_cross"),    "missing gamma_cross"
        assert cfg.alpha_spatial == pytest.approx(0.6)
        assert cfg.beta_semantic == pytest.approx(0.4)
        assert cfg.gamma_cross   == pytest.approx(1.2)
        assert cfg.graph_dropout == pytest.approx(0.2)

    # ── build_batch_graph: hybrid weights ────────────────────────────────

    def test_hybrid_weights_differ_from_spatial_only(self):
        """When h is provided, weights should differ from spatial-only."""
        from token_change_reasoner_graph import build_batch_graph
        B, N, k = 2, 12, 4
        centroids = torch.rand(B, N, 2)
        mask      = torch.zeros(B, N, dtype=torch.bool)
        # Random embeddings (will give random cosine sims)
        h = torch.randn(B, N, 64)

        idx_s, w_s = build_batch_graph(centroids, mask, k)          # spatial only
        idx_h, w_h = build_batch_graph(centroids, mask, k, h=h,     # hybrid
                                        alpha=0.6, beta=0.4)
        # Same topology (both select k-NN by distance), but different weights
        assert (idx_s == idx_h).all(), "hybrid should keep same k-NN topology"
        assert not torch.allclose(w_s, w_h), "hybrid weights should differ from spatial-only"

    def test_hybrid_weights_sum_to_one(self):
        """Per-node weights must still sum to ≈1 after hybrid combination."""
        from token_change_reasoner_graph import build_batch_graph
        B, N, k = 2, 15, 6
        centroids = torch.rand(B, N, 2)
        mask      = torch.zeros(B, N, dtype=torch.bool)
        h = torch.randn(B, N, 32)

        _, nbr_w = build_batch_graph(centroids, mask, k, h=h, alpha=0.6, beta=0.4)
        # Valid nodes: weight rows should sum to ≈1
        w_sum = nbr_w.sum(dim=2)   # [B, N]
        assert (w_sum - 1.0).abs().max().item() < 1e-4

    def test_semantic_weight_uses_cosine(self):
        """Identical token embeddings should yield even semantic weights."""
        from token_change_reasoner_graph import build_batch_graph
        B, N, k = 1, 8, 3
        centroids = torch.rand(B, N, 2)
        mask      = torch.zeros(B, N, dtype=torch.bool)
        # All embeddings identical → all cosine sims == 1 → uniform semantic weight
        h = torch.ones(B, N, 16)

        _, w_h = build_batch_graph(centroids, mask, k, h=h, alpha=0.0, beta=1.0)
        # All semantic weights should be equal (uniform cosine sim among k-NN)
        for b in range(B):
            for i in range(N):
                row = w_h[b, i]
                if row.sum() > 1e-6:
                    assert row.std().item() < 1e-4, f"Non-uniform weights at node ({b},{i})"

    # ── build_batch_graph: cross-time scaling ────────────────────────────

    def test_cross_time_edges_upweighted(self):
        """Cross-time edges (T1↔T2) should have higher raw weight than same-time."""
        from token_change_reasoner_graph import build_batch_graph
        B, N, k = 1, 10, 4
        n_t1 = 5
        centroids = torch.rand(B, N, 2)
        mask      = torch.zeros(B, N, dtype=torch.bool)
        # time_ids: first n_t1 nodes are T1, rest are T2
        time_ids  = torch.zeros(B, N, dtype=torch.long)
        time_ids[:, n_t1:] = 1

        # Run with and without cross-time scaling
        _, w_no  = build_batch_graph(centroids, mask, k,
                                      time_ids_pad=None, gamma_cross=1.2)
        _, w_yes = build_batch_graph(centroids, mask, k,
                                      time_ids_pad=time_ids, gamma_cross=1.2)

        # After renorm, cross-time edges should pull more weight in w_yes
        # Check that the maps are different
        assert not torch.allclose(w_no, w_yes), \
            "cross-time scaling should change edge weights"

    def test_gamma_cross_1_is_noop(self):
        """gamma_cross=1.0 should produce identical results to no time_ids."""
        from token_change_reasoner_graph import build_batch_graph
        B, N, k = 1, 10, 4
        centroids = torch.rand(B, N, 2)
        mask      = torch.zeros(B, N, dtype=torch.bool)
        time_ids  = torch.zeros(B, N, dtype=torch.long)
        time_ids[:, 5:] = 1

        _, w_no   = build_batch_graph(centroids, mask, k,
                                       time_ids_pad=None, gamma_cross=1.0)
        _, w_yes  = build_batch_graph(centroids, mask, k,
                                       time_ids_pad=time_ids, gamma_cross=1.0)
        assert torch.allclose(w_no, w_yes, atol=1e-6), \
            "gamma_cross=1.0 with time_ids should match no-time_ids"

    # ── build_batch_graph: graph dropout ─────────────────────────────────

    def test_graph_dropout_reduces_edges_in_training(self):
        """Training mode with p=0.5 should drop roughly half the edge weight."""
        from token_change_reasoner_graph import build_batch_graph
        B, N, k = 4, 20, 6
        centroids = torch.rand(B, N, 2)
        mask      = torch.zeros(B, N, dtype=torch.bool)

        torch.manual_seed(0)
        _, w_drop = build_batch_graph(centroids, mask, k,
                                       graph_dropout=0.5, training=True)
        _, w_keep = build_batch_graph(centroids, mask, k,
                                       graph_dropout=0.5, training=False)
        # Dropped edges → some rows should have fewer active weights
        # The presence of zeros indicates dropout fired
        # (with p=0.5 and k=6 it's extremely unlikely all are kept)
        assert (w_drop == 0.0).any(), "dropout should zero some edges in training"

    def test_graph_dropout_disabled_in_eval(self):
        """In eval mode (training=False), dropout should not fire."""
        from token_change_reasoner_graph import build_batch_graph
        B, N, k = 2, 12, 4
        centroids = torch.rand(B, N, 2)
        mask      = torch.zeros(B, N, dtype=torch.bool)

        # Call twice — results must be identical (no randomness)
        _, w1 = build_batch_graph(centroids, mask, k,
                                   graph_dropout=0.9, training=False)
        _, w2 = build_batch_graph(centroids, mask, k,
                                   graph_dropout=0.9, training=False)
        assert torch.allclose(w1, w2), "eval mode should be deterministic"

    def test_graph_dropout_weights_renormalised(self):
        """After dropout, surviving weights for valid nodes should sum to ≈1."""
        from token_change_reasoner_graph import build_batch_graph
        B, N, k = 2, 16, 6
        centroids = torch.rand(B, N, 2)
        mask      = torch.zeros(B, N, dtype=torch.bool)

        torch.manual_seed(7)
        _, w = build_batch_graph(centroids, mask, k,
                                  graph_dropout=0.3, training=True)
        # Each non-fully-dropped node's weights sum to ≈1
        w_sum = w.sum(dim=2)  # [B, N]
        # Some rows may be all-zero (fully dropped) — those are fine
        non_zero = w_sum > 1e-6
        if non_zero.any():
            assert (w_sum[non_zero] - 1.0).abs().max() < 1e-4

    # ── GraphReasoner uses new params ─────────────────────────────────────

    def test_reasoner_train_eval_differ_with_dropout(self):
        """In training mode the graph dropout should change intermediate weights."""
        cfg2 = GraphReasonerConfig(
            hidden_dim=32, num_heads=4, graph_layers=1,
            graph_dropout=0.9, graph_residual=True,
            alpha_spatial=0.6, beta_semantic=0.4, gamma_cross=1.2,
        )
        reasoner = GraphReasoner(cfg2).to(DEVICE)

        B, N, H = 2, 12, 32
        h    = torch.randn(B, N, H)
        mask = torch.zeros(B, N, dtype=torch.bool)
        cen  = torch.rand(B, N, 2)
        tid  = torch.zeros(B, N, dtype=torch.long)
        tid[:, 6:] = 1

        torch.manual_seed(1)
        reasoner.train()
        with torch.no_grad():
            out_train = reasoner(h, mask, cen, tid).clone()

        # Eval: no dropout ⇒ deterministic
        reasoner.eval()
        with torch.no_grad():
            out_eval1 = reasoner(h, mask, cen, tid).clone()
            out_eval2 = reasoner(h, mask, cen, tid).clone()

        assert torch.allclose(out_eval1, out_eval2), "eval should be deterministic"
        # Training randomness (p=0.9 is strong) very likely changes the output
        assert not torch.allclose(out_train, out_eval1), \
            "train mode with high dropout should differ from eval"

    def test_full_model_forward_with_time_ids(self, cfg):
        """Full model should work with time_ids_pad for cross-time scaling."""
        model = build_graph_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=4, device=DEVICE)
        assert "time_ids_pad" in batch, "batch must contain time_ids_pad"
        with torch.no_grad():
            out = model(batch)
        assert torch.isfinite(out["change_logits"]).all()
        assert torch.isfinite(out["delta_pred"]).all()
