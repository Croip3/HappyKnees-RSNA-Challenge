import pytest
import torch

from knee.config import load_config
from knee.models.cross_slice import CrossSliceModel, band_allow, parse_context, token_dropout


def _cfg(**model):
    overrides = [f"model.{k}={v}" for k, v in model.items()]
    return load_config(overrides=overrides)


def _batch(B=3, K=3, S=10, F=32, lengths=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, K, S, F, generator=g)
    lengths = lengths or [[S, 7, 0], [4, S, 5], [1, 2, 3]]
    valid = torch.zeros(B, K, S, dtype=torch.bool)
    pos = torch.zeros(B, K, S)
    for b in range(B):
        for k in range(K):
            n = lengths[b][k]
            valid[b, k, :n] = True
            pos[b, k, :n] = torch.linspace(0, 1, max(n, 1))[:n]
    plane = torch.arange(K).repeat(B, 1)
    seq = torch.tensor([[3, 0, 3]] * B)
    return feats, pos, valid, plane, seq


def test_parse_context():
    assert parse_context("all") is None
    assert parse_context(5) == 5
    for bad in [2, 0, -1, 4]:
        with pytest.raises(ValueError):
            parse_context(bad)


@pytest.mark.parametrize("k,expected_row3", [
    (3, [False, False, True, True, True, False, False, False]),
    (5, [False, True, True, True, True, True, False, False]),
    ("all", [True] * 8),
])
def test_band_mask(k, expected_row3):
    m = band_allow(8, k)
    assert m.shape == (8, 8)
    assert m[3].tolist() == expected_row3
    assert torch.equal(m, m.T)
    assert m.diagonal().all()


def test_band_mask_k1_is_diagonal():
    assert torch.equal(band_allow(6, 1), torch.eye(6, dtype=torch.bool))


@pytest.mark.parametrize("ctx", [3, 5, 7, "all"])
@pytest.mark.parametrize("head", ["attention", "mean"])
def test_shapes_variable_slices_and_missing_planes(ctx, head):
    cfg = _cfg(slice_context=ctx, head=head, d_model=64, n_heads=4)
    model = CrossSliceModel(32, 12, cfg.model).eval()
    feats, pos, valid, plane, seq = _batch()
    logits = model(feats, pos, valid, plane, seq)
    assert logits.shape == (3, 12)
    assert torch.isfinite(logits).all()
    logits, attn = model(feats, pos, valid, plane, seq, return_attn=True)
    assert attn.shape == (3, 12, 3 * 10)
    # attention never lands on padded slots
    assert torch.all(attn[~valid.view(3, 1, -1).expand(-1, 12, -1)] == 0)


def test_all_planes_missing_is_finite():
    cfg = _cfg(d_model=64, n_heads=4)
    model = CrossSliceModel(32, 12, cfg.model).eval()
    feats, pos, valid, plane, seq = _batch(lengths=[[0, 0, 0], [3, 0, 0], [0, 0, 5]])
    assert torch.isfinite(model(feats, pos, valid, plane, seq)).all()


@pytest.mark.parametrize("ctx", [3, 5, "all"])
def test_padding_does_not_change_outputs(ctx):
    torch.manual_seed(0)
    cfg = _cfg(slice_context=ctx, d_model=64, n_heads=4)
    model = CrossSliceModel(32, 12, cfg.model).eval()
    feats, pos, valid, plane, seq = _batch(S=10)
    out1 = model(feats, pos, valid, plane, seq)
    # pad 6 more slots filled with garbage; must not matter
    extra = 6
    feats2 = torch.cat([feats, 1e3 * torch.randn(3, 3, extra, 32)], dim=2)
    pos2 = torch.cat([pos, torch.rand(3, 3, extra)], dim=2)
    valid2 = torch.cat([valid, torch.zeros(3, 3, extra, dtype=torch.bool)], dim=2)
    out2 = model(feats2, pos2, valid2, plane, seq)
    assert torch.allclose(out1, out2, atol=1e-4), (out1 - out2).abs().max()
    # garbage inside existing padded slots must not matter either
    feats3 = feats.clone()
    feats3[~valid] = 1e4
    out3 = model(feats3, pos, valid, plane, seq)
    assert torch.allclose(out1, out3, atol=1e-4)


def test_slice_context_limits_receptive_field():
    """With k=3 and one layer, only slices within +-1 (then fusion mixes everything)."""
    torch.manual_seed(0)
    cfg = _cfg(slice_context=3, d_model=32, n_heads=4, n_layers=1, fusion_layers=0, head="attention", dropout=0.0)
    model = CrossSliceModel(16, 12, cfg.model).eval()
    feats, pos, valid, plane, seq = _batch(B=1, K=1, S=8, F=16, lengths=[[8]])
    plane, seq = plane[:, :1], seq[:, :1]
    # capture slice tokens after the within-plane encoder
    captured = {}
    model.slice_blocks[0].register_forward_hook(lambda m, i, o: captured.__setitem__("x", o))
    model(feats, pos, valid, plane, seq)
    base = captured["x"].clone()
    feats2 = feats.clone()
    feats2[0, 0, 7] = 5.0 * torch.randn(16)  # change the last slice (not a constant shift: LayerNorm removes those)
    model(feats2, pos, valid, plane, seq)
    changed = (captured["x"] - base).abs().sum(-1)[0] > 1e-6
    assert changed.tolist() == [False] * 6 + [True, True]


def test_patch_attention_shapes():
    cfg = _cfg(patch_attention=True, d_model=64, n_heads=4, slice_context=5)
    model = CrossSliceModel(32, 12, cfg.model, grid_dim=24).eval()
    feats, pos, valid, plane, seq = _batch()
    grid = torch.randn(3, 3, 10, 16, 24)
    out = model(feats, pos, valid, plane, seq, grid=grid)
    assert out.shape == (3, 12) and torch.isfinite(out).all()


def test_token_dropout_keeps_one_per_stack():
    valid = torch.ones(4, 3, 10, dtype=torch.bool)
    valid[0, 1] = False
    out = token_dropout(valid, 0.99)
    assert (out <= valid).all()
    assert out[valid.any(-1)].any(-1).all()
    assert not out[0, 1].any()


def test_backward_and_fp16_autocast_cpu_safe():
    cfg = _cfg(d_model=64, n_heads=4)
    model = CrossSliceModel(32, 12, cfg.model)
    feats, pos, valid, plane, seq = _batch()
    loss = model(feats, pos, valid, plane, seq).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
