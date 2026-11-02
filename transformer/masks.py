"""
Attention masks. All masks are BOOLEAN with True = "may attend" and are
returned already 4-D so they broadcast against (batch, heads, len_q, len_k).

Bugs fixed relative to the original notebook:
  * torch.tril(torch.ones(...)) produced a FLOAT mask, so
    `look_ahead & padding` raised "bitwise_and not implemented for Float".
  * MultiHeadAttention unsqueezed the mask a second time, turning the
    (b,1,1,k) padding mask into a 5-D tensor.
"""

import torch


def make_causal_mask(size, device=None):
    """(1, 1, size, size); position i may attend to j <= i."""
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).view(
        1, 1, size, size
    )


def make_pad_mask(seq, pad_idx=0):
    """(batch, 1, 1, len_k); False where the KEY is padding."""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_src_mask(src, pad_idx=0):
    return make_pad_mask(src, pad_idx)


def make_tgt_mask(tgt, pad_idx=0):
    """Causal AND key-padding: (batch, 1, len, len)."""
    return make_pad_mask(tgt, pad_idx) & make_causal_mask(tgt.size(1), tgt.device)
