"""Correctness tests. Run with:  pytest -q"""

import math

import pytest
import torch
import torch.nn as nn

from transformer.data import build_splits
from transformer.masks import make_causal_mask, make_src_mask, make_tgt_mask
from transformer.model import (LayerNorm, MultiHeadAttention, PositionalEncoding,
                               Transformer, count_parameters,
                               scaled_dot_product_attention)
from transformer.train import NoamScheduler

torch.manual_seed(0)


def small_model(**kw):
    cfg = dict(src_vocab_size=20, tgt_vocab_size=20, d_model=32, n_heads=4,
               num_encoder_layers=2, num_decoder_layers=2, d_ff=64, dropout=0.0)
    cfg.update(kw)
    return Transformer(**cfg).eval()


# ---------------------------------------------------------------- masking
def test_decoder_is_causal():
    """Changing FUTURE target tokens must not change outputs at earlier positions."""
    m = small_model()
    src = torch.randint(3, 20, (2, 7))
    tgt = torch.randint(3, 20, (2, 6))
    tgt2 = tgt.clone()
    tgt2[:, 3:] = torch.randint(3, 20, (2, 3))
    with torch.no_grad():
        a = m(src, tgt, make_src_mask(src), make_tgt_mask(tgt))
        b = m(src, tgt2, make_src_mask(src), make_tgt_mask(tgt2))
    assert torch.allclose(a[:, :3], b[:, :3], atol=1e-5)
    assert not torch.allclose(a[:, 3:], b[:, 3:], atol=1e-3)


def test_source_padding_is_ignored():
    """Appending PAD tokens to the source must not change the output.
    This failed in the original notebook (mask became 5-D)."""
    m = small_model()
    src = torch.randint(3, 20, (1, 5))
    src_padded = torch.cat([src, torch.zeros(1, 4, dtype=torch.long)], dim=1)
    tgt = torch.randint(3, 20, (1, 4))
    with torch.no_grad():
        a = m(src, tgt, make_src_mask(src), make_tgt_mask(tgt))
        b = m(src_padded, tgt, make_src_mask(src_padded), make_tgt_mask(tgt))
    assert torch.allclose(a, b, atol=1e-5)


def test_mask_shapes_and_dtype():
    tgt = torch.tensor([[1, 5, 6, 0, 0]])
    m = make_tgt_mask(tgt)
    assert m.dtype == torch.bool and m.shape == (1, 1, 5, 5)
    assert not m[0, 0, 4, 3]           # padding key masked
    assert not m[0, 0, 0, 1]           # future masked
    assert m[0, 0, 2, 1]               # past visible
    assert make_causal_mask(4).shape == (1, 1, 4, 4)


def test_attention_rows_sum_to_one_and_respect_mask():
    q = k = v = torch.randn(2, 4, 6, 8)
    mask = make_causal_mask(6)
    _, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.allclose(w.sum(-1), torch.ones(2, 4, 6), atol=1e-6)
    assert (w.masked_select(~mask.expand_as(w)) == 0).all()


def test_mha_padding_mask_broadcasts():
    mha = MultiHeadAttention(32, 4, dropout=0.0)
    x = torch.randn(3, 5, 32)
    src = torch.tensor([[4, 5, 6, 0, 0]] * 3)
    out, w = mha(x, x, x, make_src_mask(src))
    assert out.shape == (3, 5, 32) and w.shape == (3, 4, 5, 5)
    assert torch.all(w[..., 3:] == 0)


# ---------------------------------------------------------------- components
def test_layernorm_matches_pytorch():
    x = torch.randn(4, 7, 64) * 3 + 2
    ours, ref = LayerNorm(64), nn.LayerNorm(64)
    assert torch.allclose(ours(x), ref(x), atol=1e-5)


def test_positional_encoding_frequencies():
    """Low dims oscillate FAST, high dims SLOWLY (the original notebook
    stated the opposite)."""
    pe = PositionalEncoding(128, max_len=200, dropout=0.0).pe[0]
    rate = lambda col: (col[1:] - col[:-1]).abs().mean().item()  # mean |change| per step
    assert rate(pe[:, 0]) > rate(pe[:, 32]) > rate(pe[:, 64]) > rate(pe[:, 126])
    assert pe.abs().max() <= 1.0


# ---------------------------------------------------------------- scale
def _closed_form_params(V, d, h, N, dff, tied):
    attn = 4 * (d * d + d)
    ffn = d * dff + dff + dff * d + d
    ln = 2 * d
    enc = attn + ffn + 2 * ln
    dec = 2 * attn + ffn + 3 * ln
    emb = V * d
    out_bias = V
    if tied:
        return N * enc + N * dec + emb + out_bias
    return N * enc + N * dec + 3 * emb + out_bias


@pytest.mark.parametrize("tied", [False, True])
def test_base_parameter_count(tied):
    V = 37000  # paper's shared BPE vocabulary (EN-DE)
    m = Transformer(V, V, tie_embeddings=tied)
    n = count_parameters(m)
    assert n == _closed_form_params(V, 512, 8, 6, 2048, tied)
    if tied:
        # Paper reports 65M for base; we get ~63.1M (paper's exact vocab /
        # bias choices are not fully specified).
        assert 60e6 < n < 66e6


# ---------------------------------------------------------------- data / optim
@pytest.mark.parametrize("task", ["copy", "reverse", "addition"])
def test_splits_are_disjoint(task):
    s = build_splits(task, 500, 100, 100, 50, seed=0)
    keys = {k: {p[0] for p in v.pairs} for k, v in s.items()}
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["val"]
    assert not keys["train"] & keys["ood"]


def test_addition_targets_are_correct():
    from transformer.data import decode_tokens
    s = build_splits("addition", 200, 10, 10, 10, seed=1)
    for src, tgt in s["train"].pairs:
        a, b = decode_tokens(src, "addition").split("+")
        assert int(decode_tokens(tgt, "addition")) == int(a) + int(b)


def test_noam_schedule_peaks_at_warmup():
    opt = torch.optim.Adam([nn.Parameter(torch.zeros(1))], lr=0.0)
    s = NoamScheduler(opt, d_model=128, warmup_steps=400)
    lrs = [s.lr_at(t) for t in range(1, 2000)]
    assert max(range(len(lrs)), key=lrs.__getitem__) + 1 == 400
    assert math.isclose(max(lrs), 128**-0.5 * 400**-0.5, rel_tol=1e-9)
    assert s.lr_at(0) > 0  # no division by zero before the first step


def test_greedy_decode_shapes():
    m = small_model()
    src = torch.randint(3, 20, (4, 6))
    out = m.greedy_decode(src, make_src_mask(src), 1, 2, 0, max_len=8)
    assert out.shape[0] == 4 and out.shape[1] <= 8
    assert (out != 2).all()  # EOS stripped
