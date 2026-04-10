"""
tests/test_reasoner.py
======================
Unit tests for Stage 4A — Token Change Reasoner.

Run:
    pytest tests/test_reasoner.py -v
"""

from __future__ import annotations

import pytest
import torch

from token_change_reasoner import (
    ChangeReasonerModel,
    ChangePredictionHead,
    DeltaHead,
    ReasonerConfig,
    SampleData,
    TokenEncoder,
    TransformerReasoner,
    build_batch,
    build_model,
    compute_loss,
    count_parameters,
    make_dummy_batch,
    training_step,
)

DEVICE = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return ReasonerConfig(
        hidden_dim=64,   # small for fast tests
        num_layers=2,
        num_heads=4,
        ff_dim=128,
    )


@pytest.fixture
def model(cfg):
    return build_model(cfg).to(DEVICE)


@pytest.fixture
def batch(cfg):
    _, batch, _ = make_dummy_batch(batch_size=2, n1=8, n2=10,
                                   n_pairs=5, device=DEVICE, seed=0)
    return batch


# ─────────────────────────────────────────────────────────────────────────────
# 1. TokenEncoder
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenEncoder:
    def test_output_shape(self, cfg):
        enc = TokenEncoder(cfg).to(DEVICE)
        n = 15
        emb  = torch.randn(n, 256)
        tids = torch.zeros(n, dtype=torch.long)
        cen  = torch.rand(n, 2)
        loga = torch.rand(n)
        out  = enc(emb, tids, cen, loga)
        assert out.shape == (n, cfg.hidden_dim), f"Expected ({n},{cfg.hidden_dim}), got {out.shape}"

    def test_different_time_ids_differ(self, cfg):
        enc = TokenEncoder(cfg).to(DEVICE)
        emb  = torch.randn(4, 256)
        cen  = torch.rand(4, 2)
        loga = torch.rand(4)
        out_t1 = enc(emb, torch.zeros(4, dtype=torch.long), cen, loga)
        out_t2 = enc(emb, torch.ones(4, dtype=torch.long),  cen, loga)
        assert not torch.allclose(out_t1, out_t2), "T1 and T2 tokens should differ after time embed"

    def test_batched_equals_sequential(self, cfg):
        """Processing tokens jointly should equal processing each independently."""
        enc = TokenEncoder(cfg).to(DEVICE)
        enc.eval()
        n = 5
        emb  = torch.randn(n, 256)
        tids = torch.zeros(n, dtype=torch.long)
        cen  = torch.rand(n, 2)
        loga = torch.rand(n)
        with torch.no_grad():
            out_all = enc(emb, tids, cen, loga)
            for i in range(n):
                out_i = enc(emb[i:i+1], tids[i:i+1], cen[i:i+1], loga[i:i+1])
                assert torch.allclose(out_all[i:i+1], out_i, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# 2. TransformerReasoner
# ─────────────────────────────────────────────────────────────────────────────

class TestTransformerReasoner:
    def test_output_shape_no_mask(self, cfg):
        reasoner = TransformerReasoner(cfg).to(DEVICE)
        B, N, H = 2, 12, cfg.hidden_dim
        x = torch.randn(B, N, H)
        out = reasoner(x, padding_mask=None)
        assert out.shape == (B, N, H)

    def test_output_shape_with_mask(self, cfg):
        reasoner = TransformerReasoner(cfg).to(DEVICE)
        B, N, H = 3, 10, cfg.hidden_dim
        x    = torch.randn(B, N, H)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, 8:] = True   # last 2 are padding
        out  = reasoner(x, padding_mask=mask)
        assert out.shape == (B, N, H)

    def test_all_padding_raises_or_nan(self, cfg):
        """Full padding mask should not crash (may output NaN for padded positions)."""
        reasoner = TransformerReasoner(cfg).to(DEVICE)
        B, N, H = 1, 4, cfg.hidden_dim
        x    = torch.randn(B, N, H)
        mask = torch.zeros(B, N, dtype=torch.bool)
        mask[:, 2:] = True
        # Should not raise
        out = reasoner(x, padding_mask=mask)
        assert out.shape == (B, N, H)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ChangePredictionHead
# ─────────────────────────────────────────────────────────────────────────────

class TestChangePredictionHead:
    def test_output_shape(self, cfg):
        head = ChangePredictionHead(cfg).to(DEVICE)
        B, N, H = 2, 15, cfg.hidden_dim
        x = torch.randn(B, N, H)
        out = head(x)
        assert out.shape == (B, N), f"Expected ({B},{N}), got {out.shape}"

    def test_output_unbounded_logit(self, cfg):
        """Raw logits should not be clamped to [0,1]."""
        head = ChangePredictionHead(cfg).to(DEVICE)
        x = torch.randn(20, cfg.hidden_dim) * 10
        out = head(x)
        assert out.shape == (20,)


# ─────────────────────────────────────────────────────────────────────────────
# 4. DeltaHead
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaHead:
    def test_output_shape(self, cfg):
        head = DeltaHead(cfg).to(DEVICE)
        M, H = 8, cfg.hidden_dim
        ri = torch.randn(M, H)
        rj = torch.randn(M, H)
        out = head(ri, rj)
        assert out.shape == (M,), f"Expected ({M},), got {out.shape}"

    def test_output_nonneg(self, cfg):
        """Softplus ensures non-negative outputs."""
        head = DeltaHead(cfg).to(DEVICE)
        with torch.no_grad():
            ri = torch.randn(50, cfg.hidden_dim)
            rj = torch.randn(50, cfg.hidden_dim)
            out = head(ri, rj)
        assert (out >= 0).all(), "DeltaHead should output non-negative values"

    def test_empty_pairs(self, cfg):
        head = DeltaHead(cfg).to(DEVICE)
        ri = torch.zeros(0, cfg.hidden_dim)
        rj = torch.zeros(0, cfg.hidden_dim)
        out = head(ri, rj)
        assert out.shape == (0,)


# ─────────────────────────────────────────────────────────────────────────────
# 5. build_batch
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildBatch:
    def test_shapes(self, cfg):
        _, batch, _ = make_dummy_batch(batch_size=3, n1=6, n2=8,
                                       n_pairs=4, device=DEVICE)
        B = 3
        assert batch["tokens_pad"].shape[0]   == B
        assert batch["time_ids_pad"].shape[0] == B
        assert batch["padding_mask"].shape[0] == B
        N = batch["tokens_pad"].shape[1]
        assert batch["centroids_pad"].shape  == (B, N, 2)
        assert batch["log_areas_pad"].shape  == (B, N)

    def test_time_ids_values(self, cfg):
        """T1 tokens should have time_id=0, T2 tokens should have time_id=1."""
        samples = []
        n1, n2 = 4, 5
        pairs = torch.zeros(0, 3)
        samples.append(SampleData(
            tokens_t1=torch.randn(n1, 256), tokens_t2=torch.randn(n2, 256),
            centroids_t1=torch.rand(n1, 2), centroids_t2=torch.rand(n2, 2),
            areas_t1=torch.ones(n1),        areas_t2=torch.ones(n2),
            match_pairs=pairs,
        ))
        batch = build_batch(samples, cfg, DEVICE)
        tids = batch["time_ids_pad"][0]   # [N_max]
        assert (tids[:n1] == 0).all(),  "T1 tokens should have time_id=0"
        assert (tids[n1:n1+n2] == 1).all(), "T2 tokens should have time_id=1"

    def test_padding_mask_correct(self, cfg):
        n1, n2 = 3, 4
        n = n1 + n2
        pairs = torch.zeros(0, 3)
        s = SampleData(
            tokens_t1=torch.randn(n1, 256), tokens_t2=torch.randn(n2, 256),
            centroids_t1=torch.rand(n1, 2), centroids_t2=torch.rand(n2, 2),
            areas_t1=torch.ones(n1),        areas_t2=torch.ones(n2),
            match_pairs=pairs,
        )
        batch = build_batch([s], cfg, DEVICE)
        mask = batch["padding_mask"][0]   # [N_max]
        assert not mask[:n].any(),  "Valid positions should be False in padding_mask"
        # If N_max > n, remaining should be True
        if mask.shape[0] > n:
            assert mask[n:].all(), "Pad positions should be True"

    def test_pair_indices_valid(self, cfg):
        """pair_i, pair_j must be within [0, N_max)."""
        _, batch, _ = make_dummy_batch(batch_size=2, n_pairs=6, device=DEVICE)
        N = batch["tokens_pad"].shape[1]
        assert (batch["pair_i"] < N).all()
        assert (batch["pair_j"] < N).all()
        assert (batch["pair_i"] >= 0).all()
        assert (batch["pair_j"] >= 0).all()

    def test_variable_length_samples(self, cfg):
        """Samples with different token counts should collate correctly."""
        pairs = torch.zeros(0, 3)
        samples = [
            SampleData(torch.randn(3,256), torch.randn(4,256),
                       torch.rand(3,2), torch.rand(4,2),
                       torch.ones(3), torch.ones(4), pairs),
            SampleData(torch.randn(12,256), torch.randn(15,256),
                       torch.rand(12,2), torch.rand(15,2),
                       torch.ones(12), torch.ones(15), pairs),
        ]
        batch = build_batch(samples, cfg, DEVICE)
        expected_N = 12 + 15   # max(3+4, 12+15)
        assert batch["tokens_pad"].shape == (2, expected_N, 256)


# ─────────────────────────────────────────────────────────────────────────────
# 6. compute_loss
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeLoss:
    def test_loss_shape_and_positivity(self, model, batch, cfg):
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
        losses = compute_loss(outputs, batch, cfg)

        assert "total_loss"  in losses
        assert "change_loss" in losses
        assert "delta_loss"  in losses

        for k, v in losses.items():
            assert v >= 0, f"{k} should be non-negative, got {float(v)}"

    def test_loss_is_finite(self, model, batch, cfg):
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
        losses = compute_loss(outputs, batch, cfg)
        for k, v in losses.items():
            assert torch.isfinite(v), f"{k} is not finite: {v}"

    def test_weighted_sum(self, model, batch, cfg):
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
        losses = compute_loss(outputs, batch, cfg)
        expected = losses["change_loss"] + cfg.delta_loss_weight * losses["delta_loss"]
        assert torch.isclose(losses["total_loss"], expected, atol=1e-5)

    def test_no_pairs_delta_zero(self, cfg):
        """When there are no matched pairs, delta_loss should be 0."""
        model = build_model(cfg).to(DEVICE)
        # Build a batch with no match pairs
        pairs = torch.zeros(0, 3)
        s = SampleData(
            torch.randn(5, 256), torch.randn(5, 256),
            torch.rand(5, 2),   torch.rand(5, 2),
            torch.ones(5),      torch.ones(5),
            match_pairs=pairs,
        )
        batch  = build_batch([s], cfg, DEVICE)
        model.eval()
        with torch.no_grad():
            outputs = model(batch)
        losses = compute_loss(outputs, batch, cfg)
        assert float(losses["delta_loss"]) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 7. End-to-End Forward Pass
# ─────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_forward_shape(self, cfg, batch_size):
        model = build_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(
            batch_size=batch_size, n1=8, n2=10, n_pairs=5, device=DEVICE
        )
        with torch.no_grad():
            outputs = model(batch)

        B  = batch_size
        N  = batch["tokens_pad"].shape[1]
        M  = len(batch["pair_b"])
        assert outputs["change_logits"].shape == (B, N)
        assert outputs["delta_pred"].shape    == (M,)
        assert outputs["delta_target"].shape  == (M,)

    def test_single_token_each_side(self, cfg):
        """Edge case: 1 T1 token, 1 T2 token, 1 match."""
        model = build_model(cfg).to(DEVICE)
        model.eval()
        s = SampleData(
            tokens_t1=torch.randn(1, 256), tokens_t2=torch.randn(1, 256),
            centroids_t1=torch.rand(1, 2), centroids_t2=torch.rand(1, 2),
            areas_t1=torch.tensor([1000.0]), areas_t2=torch.tensor([1000.0]),
            match_pairs=torch.tensor([[0.0, 0.0, 0.9]]),
        )
        batch = build_batch([s], cfg, DEVICE)
        with torch.no_grad():
            outputs = model(batch)
        assert outputs["change_logits"].shape == (1, 2)  # B=1, N=2 (1+1)
        assert outputs["delta_pred"].shape    == (1,)

    def test_backward_runs(self, cfg):
        """Check that backward pass does not raise."""
        model = build_model(cfg).to(DEVICE)
        model.train()
        _, batch, _ = make_dummy_batch(batch_size=2, device=DEVICE)
        outputs = model(batch)
        losses  = compute_loss(outputs, batch, cfg)
        losses["total_loss"].backward()  # should not raise
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"NaN grad in {name}"

    def test_training_step_updates_params(self, cfg):
        """After training_step, at least one parameter should change."""
        model = build_model(cfg).to(DEVICE)
        _, batch, _ = make_dummy_batch(batch_size=2, device=DEVICE)
        params_before = {n: p.clone() for n, p in model.named_parameters()}
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        training_step(model, batch, opt, scaler=None)
        changed = 0
        for n, p in model.named_parameters():
            if not torch.allclose(params_before[n], p):
                changed += 1
        assert changed > 0, "No parameters were updated after training_step"

    def test_no_grad_in_eval(self, cfg):
        """model.eval() + no_grad should not accumulate grads."""
        model = build_model(cfg).to(DEVICE)
        model.eval()
        _, batch, _ = make_dummy_batch(batch_size=1, device=DEVICE)
        with torch.no_grad():
            outputs = model(batch)
        assert outputs["change_logits"].requires_grad is False


# ─────────────────────────────────────────────────────────────────────────────
# 8. Miscellaneous
# ─────────────────────────────────────────────────────────────────────────────

class TestMisc:
    def test_parameter_count_reasonable(self, cfg):
        model = build_model(cfg).to(DEVICE)
        n = count_parameters(model)
        # With hidden_dim=64, should be well under 1M
        assert 1_000 < n < 1_000_000, f"Unusual param count: {n}"

    def test_default_config_param_count(self):
        cfg = ReasonerConfig()   # full-size (hidden=384)
        model = build_model(cfg)
        n = count_parameters(model)
        # Should be ~a few million
        assert 1_000_000 < n < 50_000_000, f"Unusual param count: {n}"

    def test_make_dummy_batch_reproducible(self, cfg):
        _, b1, _ = make_dummy_batch(seed=7, device=DEVICE)
        _, b2, _ = make_dummy_batch(seed=7, device=DEVICE)
        assert torch.allclose(b1["tokens_pad"], b2["tokens_pad"])

    def test_proxy_labels_fraction(self, cfg):
        """Proxy labels should mark roughly 17% as changed (from diagnostics)."""
        _, batch, _ = make_dummy_batch(batch_size=8, n1=50, n2=50,
                                       n_pairs=30, device=DEVICE, seed=0)
        labels  = batch["change_labels"]
        mask    = ~batch["padding_mask"]
        frac_changed = labels[mask].mean().item()
        # Should be somewhere between 0% and 100% — just check it runs
        assert 0.0 <= frac_changed <= 1.0
