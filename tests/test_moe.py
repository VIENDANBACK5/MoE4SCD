"""
tests/test_moe.py
=================
Unit tests for Stage 4C — Mixture-of-Experts module.

Tests cover:
  1. MoEConfig       — fields, defaults, inheritance
  2. Expert          — shape, gradient
  3. MoELayer        — output shape, residual, aux losses, expert dispatch
  4. TokenChangeReasonerMoE — full forward, backward, parameter count
  5. compute_moe_loss        — total, aux loss weighting
  6. Router behaviours       — collapse detection, determinism in eval
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from token_change_reasoner_moe import (
    MoEConfig,
    Expert,
    MoELayer,
    TokenChangeReasonerMoE,
    build_moe_model,
    compute_moe_loss,
)
from token_change_reasoner import make_dummy_batch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return MoEConfig(
        hidden_dim=32, num_heads=4, num_layers=2, ff_dim=64,
        graph_k=4, graph_layers=2,
        moe_num_experts=4, moe_expert_dim=64,
        lambda_balance=0.01, lambda_entropy=0.001,
    )

@pytest.fixture
def moe_layer(cfg):
    return MoELayer(cfg).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MoEConfig
# ─────────────────────────────────────────────────────────────────────────────

class TestMoEConfig:

    def test_default_fields(self):
        cfg = MoEConfig()
        assert cfg.moe_num_experts == 4
        assert cfg.moe_expert_dim  == 512
        assert cfg.lambda_balance  == pytest.approx(0.01)
        assert cfg.lambda_entropy  == pytest.approx(0.001)

    def test_inherits_graph_fields(self):
        cfg = MoEConfig()
        assert hasattr(cfg, "graph_k")
        assert hasattr(cfg, "alpha_spatial")
        assert hasattr(cfg, "beta_semantic")
        assert hasattr(cfg, "gamma_cross")
        assert hasattr(cfg, "graph_dropout")

    def test_custom_values(self):
        cfg = MoEConfig(moe_num_experts=8, moe_expert_dim=1024, lambda_balance=0.1)
        assert cfg.moe_num_experts == 8
        assert cfg.moe_expert_dim  == 1024
        assert cfg.lambda_balance  == pytest.approx(0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Expert
# ─────────────────────────────────────────────────────────────────────────────

class TestExpert:

    def test_output_shape(self):
        H, D = 32, 64
        exp = Expert(H, D).to(DEVICE)
        x   = torch.randn(7, H, device=DEVICE)
        out = exp(x)
        assert out.shape == (7, H), f"expected ({7},{H}), got {out.shape}"

    def test_gradient_flows(self):
        H, D = 32, 64
        exp = Expert(H, D).to(DEVICE)
        x   = torch.randn(5, H, device=DEVICE, requires_grad=True)
        out = exp(x).sum()
        out.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_has_two_linears(self):
        exp = Expert(32, 64)
        linears = [m for m in exp.modules() if isinstance(m, nn.Linear)]
        assert len(linears) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. MoELayer
# ─────────────────────────────────────────────────────────────────────────────

class TestMoELayer:

    def test_output_shape(self, moe_layer, cfg):
        B, N, H = 2, 20, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)
        out, bal, ent, tpe = moe_layer(x)
        assert out.shape == (B, N, H)
        assert bal.dim() == 0, "balance_loss should be a scalar"
        assert ent.dim() == 0, "entropy_loss should be a scalar"
        assert tpe.shape == (cfg.moe_num_experts,)

    def test_residual_connection(self, moe_layer, cfg):
        """With zero-initialised expert weights the output equals x."""
        # Zero out all expert parameters
        with torch.no_grad():
            for exp in moe_layer.experts:
                for p in exp.parameters():
                    p.zero_()
        B, N, H = 1, 10, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)
        out, _, _, _ = moe_layer(x)
        assert torch.allclose(out, x, atol=1e-6), "residual should = x when experts output 0"

    def test_tokens_per_expert_sums_to_total(self, moe_layer, cfg):
        B, N, H = 3, 24, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)
        _, _, _, tpe = moe_layer(x)
        assert tpe.sum().item() == B * N

    def test_balance_loss_positive(self, moe_layer, cfg):
        B, N, H = 2, 16, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)
        _, bal, _, _ = moe_layer(x)
        assert float(bal) > 0.0, "balance_loss should be positive"

    def test_entropy_loss_positive(self, moe_layer, cfg):
        B, N, H = 2, 16, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)
        _, _, ent, _ = moe_layer(x)
        assert float(ent) > 0.0, "entropy_loss should be positive"

    def test_balance_loss_uniform_is_1(self, moe_layer, cfg):
        """
        When all tokens go to experts with probability 1/E,
        p_i = 1/E  →  L_balance = E * E * (1/E)² = 1.
        """
        E = cfg.moe_num_experts
        # Patch router so all logits are equal → uniform softmax
        with torch.no_grad():
            moe_layer.router.weight.zero_()
            moe_layer.router.bias.zero_()
        B, N, H = 2, 20, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)
        _, bal, _, _ = moe_layer(x)
        # L_balance = E * E*(1/E)² = E*(1/E) = 1  ← exactly 1
        assert abs(float(bal) - 1.0) < 1e-4, f"uniform router balance_loss should be 1, got {float(bal):.4f}"

    def test_gradient_flows_through_moe(self, moe_layer, cfg):
        B, N, H = 2, 12, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE, requires_grad=True)
        out, bal, ent, _ = moe_layer(x)
        loss = out.sum() + bal + ent
        loss.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()

    def test_different_inputs_different_outputs(self, moe_layer, cfg):
        B, N, H = 2, 10, cfg.hidden_dim
        x1 = torch.randn(B, N, H, device=DEVICE)
        x2 = torch.randn(B, N, H, device=DEVICE)
        out1, _, _, _ = moe_layer(x1)
        out2, _, _, _ = moe_layer(x2)
        assert not torch.allclose(out1, out2)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TokenChangeReasonerMoE — full model
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenChangeReasonerMoE:

    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_shape(self, cfg, batch_size):
        model = build_moe_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(
            batch_size=batch_size, n1=8, n2=10, n_pairs=4, device=DEVICE
        )
        with torch.no_grad():
            out = model(batch)

        B = batch["tokens_pad"].shape[0]
        N = batch["tokens_pad"].shape[1]
        M = len(batch["pair_b"])

        assert out["change_logits"].shape == (B, N)
        assert out["delta_pred"].shape    == (M,)
        assert out["balance_loss"].dim()  == 0
        assert out["entropy_loss"].dim()  == 0
        assert out["tokens_per_expert"].shape == (cfg.moe_num_experts,)

    def test_forward_all_finite(self, cfg):
        model = build_moe_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=10, n2=12, n_pairs=6, device=DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert torch.isfinite(out["change_logits"]).all()
        assert torch.isfinite(out["delta_pred"]).all()
        assert torch.isfinite(out["balance_loss"])
        assert torch.isfinite(out["entropy_loss"])

    def test_backward_runs(self, cfg):
        """Full backward pass should not raise."""
        model = build_moe_model(cfg).to(DEVICE).train()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=4, device=DEVICE)
        out    = model(batch)
        losses = compute_moe_loss(out, batch, cfg)
        losses["total_loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_tokens_per_expert_sums_to_BN(self, cfg):
        model = build_moe_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=3, n1=10, n2=10, n_pairs=5, device=DEVICE)
        with torch.no_grad():
            out = model(batch)
        B = batch["tokens_pad"].shape[0]
        N = batch["tokens_pad"].shape[1]
        assert out["tokens_per_expert"].sum().item() == B * N

    def test_param_count_larger_than_graph(self, cfg):
        from token_change_reasoner_graph import build_graph_model, GraphReasonerConfig
        from token_change_reasoner        import count_parameters
        # Build graph model with identical architecture (same hidden_dim, layers, etc.)
        g_cfg = GraphReasonerConfig(
            hidden_dim   = cfg.hidden_dim,
            num_heads    = cfg.num_heads,
            num_layers   = cfg.num_layers,
            ff_dim       = cfg.ff_dim,
            graph_k      = cfg.graph_k,
            graph_layers = cfg.graph_layers,
        )
        moe_params   = count_parameters(build_moe_model(cfg))
        graph_params = count_parameters(build_graph_model(g_cfg))
        assert moe_params > graph_params, \
            f"MoE ({moe_params}) should have more params than graph ({graph_params})"

    def test_training_step_updates_params(self, cfg):
        model = build_moe_model(cfg).to(DEVICE).train()
        opt   = torch.optim.AdamW(model.parameters(), lr=1e-3)
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=4, device=DEVICE)

        params_before = {n: p.clone() for n, p in model.named_parameters()}

        out    = model(batch)
        losses = compute_moe_loss(out, batch, cfg)
        losses["total_loss"].backward()
        opt.step()

        changed = sum(
            1 for n, p in model.named_parameters()
            if not torch.allclose(p, params_before[n])
        )
        assert changed > 0, "No parameters updated after training step"

    def test_no_match_pairs(self, cfg):
        """Model should handle a batch with zero match pairs gracefully."""
        model = build_moe_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=0, device=DEVICE)
        with torch.no_grad():
            out = model(batch)
        assert out["delta_pred"].shape == (0,)
        assert out["change_logits"].shape[0] == 2


# ─────────────────────────────────────────────────────────────────────────────
# 5. compute_moe_loss
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMoELoss:

    def test_loss_keys_present(self, cfg):
        model = build_moe_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=4, device=DEVICE)
        with torch.no_grad():
            out = model(batch)
        losses = compute_moe_loss(out, batch, cfg)
        for key in ["total_loss", "change_loss", "delta_loss", "balance_loss", "entropy_loss"]:
            assert key in losses, f"Missing loss key: {key}"

    def test_total_is_sum_of_components(self, cfg):
        model = build_moe_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=4, device=DEVICE)
        with torch.no_grad():
            out = model(batch)
        losses = compute_moe_loss(out, batch, cfg)

        # Recompute expected total
        from token_change_reasoner import compute_loss
        base = compute_loss(out, batch, cfg)
        expected = (base["total_loss"]
                    + cfg.lambda_balance * out["balance_loss"]
                    + cfg.lambda_entropy * out["entropy_loss"])
        assert torch.allclose(
            losses["total_loss"], expected, atol=1e-5
        ), f"total mismatch: {float(losses['total_loss']):.6f} vs {float(expected):.6f}"

    def test_all_losses_finite(self, cfg):
        model = build_moe_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=4, device=DEVICE)
        with torch.no_grad():
            out = model(batch)
        losses = compute_moe_loss(out, batch, cfg)
        for k, v in losses.items():
            assert torch.isfinite(v), f"Loss {k} is not finite: {v}"

    def test_lambda_zero_reverts_to_base(self, cfg):
        """With λ_balance = λ_entropy = 0, total should equal base total."""
        cfg0 = MoEConfig(
            hidden_dim   = cfg.hidden_dim,
            num_heads    = cfg.num_heads,
            num_layers   = cfg.num_layers,
            graph_k      = cfg.graph_k,
            graph_layers = cfg.graph_layers,
            moe_num_experts = cfg.moe_num_experts,
            moe_expert_dim  = cfg.moe_expert_dim,
            lambda_balance  = 0.0,
            lambda_entropy  = 0.0,
        )
        model = build_moe_model(cfg0).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10, n_pairs=4, device=DEVICE)
        with torch.no_grad():
            out = model(batch)

        from token_change_reasoner import compute_loss
        base   = compute_loss(out, batch, cfg0)
        losses = compute_moe_loss(out, batch, cfg0)
        assert torch.allclose(losses["total_loss"], base["total_loss"], atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Router behaviours
# ─────────────────────────────────────────────────────────────────────────────

class TestRouter:

    def test_router_output_is_prob_distribution(self, moe_layer, cfg):
        B, N, H = 2, 16, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)
        # Access router probs directly
        x_flat = x.reshape(B * N, H)
        with torch.no_grad():
            logits = moe_layer.router(x_flat)
            probs  = torch.softmax(logits, dim=-1)
        assert (probs >= 0).all()
        assert torch.allclose(probs.sum(dim=-1), torch.ones(B * N, device=DEVICE), atol=1e-5)

    def test_top1_each_token_one_expert(self, moe_layer, cfg):
        B, N, H = 2, 20, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)
        x_flat = x.reshape(B * N, H)
        with torch.no_grad():
            logits = moe_layer.router(x_flat)
            probs  = torch.softmax(logits, dim=-1)
            idx    = probs.argmax(dim=-1)
        # Each of B*N tokens should map to exactly one expert (0 ≤ idx < E)
        assert (idx >= 0).all()
        assert (idx < cfg.moe_num_experts).all()

    def test_deterministic_in_eval(self, moe_layer, cfg):
        moe_layer.eval()
        B, N, H = 2, 16, cfg.hidden_dim
        torch.manual_seed(0)
        x = torch.randn(B, N, H, device=DEVICE)
        with torch.no_grad():
            out1, _, _, _ = moe_layer(x)
            out2, _, _, _ = moe_layer(x)
        assert torch.allclose(out1, out2), "MoELayer should be deterministic in eval"

    def test_balance_loss_decreases_on_uniform_input(self, cfg):
        """
        With uniform inputs, after one gradient step toward lower balance loss,
        we should see lower balance loss (sanity check gradient direction).
        """
        layer = MoELayer(cfg).to(DEVICE)
        opt   = torch.optim.SGD(layer.router.parameters(), lr=0.1)
        B, N, H = 4, 32, cfg.hidden_dim
        x = torch.randn(B, N, H, device=DEVICE)

        with torch.no_grad():
            _, bal0, _, _ = layer(x)

        # Multiple steps
        for _ in range(10):
            opt.zero_grad()
            _, bal, _, _ = layer(x)
            bal.backward()
            opt.step()

        with torch.no_grad():
            _, bal1, _, _ = layer(x)

        # Gradient descent on the balance loss should reduce it
        assert float(bal1) <= float(bal0) + 1e-4, \
            f"balance_loss should not increase after gradient steps: {float(bal0):.4f} → {float(bal1):.4f}"
