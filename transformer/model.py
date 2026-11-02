"""
Encoder-decoder Transformer (Vaswani et al., 2017), implemented from scratch.

Conventions used throughout:
  * Tensors are batch-first: (batch, seq_len, d_model).
  * Masks are BOOLEAN, True = "may attend", and must broadcast to
    (batch, n_heads, len_q, len_k). See transformer/masks.py.
  * Post-LN residual blocks, as in the original paper:
        x = LayerNorm(x + Dropout(Sublayer(x)))
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Fixed sinusoidal encoding (Sec. 3.5).

    PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Low dimension indices have the HIGHEST frequency (wavelength 2*pi at i=0),
    high indices the lowest (wavelength up to 10000*2*pi).
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"seq_len {x.size(1)} exceeds max_len {self.pe.size(1)}")
        return self.dropout(x + self.pe[:, : x.size(1)])


class TokenEmbedding(nn.Module):
    """Token lookup scaled by sqrt(d_model) (Sec. 3.4)."""

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

    def forward(self, x):
        return self.positional_encoding(self.token_embedding(x))


# --------------------------------------------------------------------------
# Attention
# --------------------------------------------------------------------------
def scaled_dot_product_attention(query, key, value, mask=None, dropout=None):
    """softmax(Q K^T / sqrt(d_k)) V   (Eq. 1).

    query: (..., len_q, d_k)   key/value: (..., len_k, d_k)
    mask:  bool, broadcastable to (..., len_q, len_k); True = keep.
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # dtype-aware fill value: -1e9 overflows to -inf in fp16.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attention_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    return torch.matmul(attention_weights, value), attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # Most recent attention map (detached), for analysis / visualisation.
        self.last_attention = None

    def split_heads(self, x):
        b, t, _ = x.size()
        return x.view(b, t, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        b, _, t, _ = x.size()
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))

        # mask is already 4-D (b|1, 1, len_q|1, len_k): NO extra unsqueeze here.
        # (The original notebook unsqueezed again, which turned a padding mask
        #  of shape (b,1,1,k) into 5-D and silently broke broadcasting.)
        out, attn = scaled_dot_product_attention(Q, K, V, mask, self.dropout)
        self.last_attention = attn.detach()
        return self.W_o(self.combine_heads(out)), attn


# --------------------------------------------------------------------------
# Position-wise FFN, LayerNorm, residual
# --------------------------------------------------------------------------
class FeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2   (Eq. 2)."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNorm(nn.Module):
    """LayerNorm (Ba et al., 2016) written out by hand.

    Uses the BIASED variance and puts eps inside the sqrt, which makes it
    numerically identical to torch.nn.LayerNorm (verified in tests/).
    """

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class ResidualConnection(nn.Module):
    """Post-LN residual, as in the paper: LayerNorm(x + Dropout(Sublayer(x)))."""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))


# --------------------------------------------------------------------------
# Encoder / decoder
# --------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask=None):
        x = self.residual1(x, lambda y: self.self_attention(y, y, y, src_mask)[0])
        return self.residual2(x, self.feed_forward)


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.residual1 = ResidualConnection(d_model, dropout)
        self.residual2 = ResidualConnection(d_model, dropout)
        self.residual3 = ResidualConnection(d_model, dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.residual1(x, lambda y: self.masked_self_attention(y, y, y, tgt_mask)[0])
        x = self.residual2(x, lambda y: self.cross_attention(y, memory, memory, src_mask)[0])
        return self.residual3(x, self.feed_forward)


class Encoder(nn.Module):
    # No extra final LayerNorm: with Post-LN every block already ends in a
    # LayerNorm. (A final norm is a Pre-LN convention; the notebook had both.)
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, src_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1,
        tie_embeddings=False,
    ):
        """tie_embeddings=True shares src embedding, tgt embedding and the
        pre-softmax projection (Sec. 3.4). Requires src_vocab == tgt_vocab.
        This is what the paper's 65M 'base' parameter count assumes."""
        super().__init__()
        self.d_model = d_model
        self.src_embedding = TransformerEmbedding(src_vocab_size, d_model, max_seq_len, dropout)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, d_model, max_seq_len, dropout)
        self.encoder = Encoder(num_encoder_layers, d_model, n_heads, d_ff, dropout)
        self.decoder = Decoder(num_decoder_layers, d_model, n_heads, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        self._init_parameters()

        if tie_embeddings:
            assert src_vocab_size == tgt_vocab_size, "tying needs a shared vocabulary"
            shared = self.src_embedding.token_embedding.embedding.weight
            self.tgt_embedding.token_embedding.embedding.weight = shared
            self.fc_out.weight = shared

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embedding(src), src_mask)

    def decode(self, tgt, memory, src_mask=None, tgt_mask=None):
        return self.decoder(self.tgt_embedding(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encode(src, src_mask)
        return self.fc_out(self.decode(tgt, memory, src_mask, tgt_mask))

    # ----------------------------------------------------------------------
    @torch.no_grad()
    def greedy_decode(self, src, src_mask, sos_idx, eos_idx, pad_idx, max_len):
        """Batched autoregressive greedy decoding.

        Returns (batch, <=max_len) token ids, EOS excluded, PAD after EOS.
        """
        from .masks import make_tgt_mask

        self.eval()
        memory = self.encode(src, src_mask)
        b = src.size(0)
        ys = torch.full((b, 1), sos_idx, dtype=torch.long, device=src.device)
        finished = torch.zeros(b, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            out = self.decode(ys, memory, src_mask, make_tgt_mask(ys, pad_idx))
            next_tok = self.fc_out(out[:, -1]).argmax(-1)
            next_tok = next_tok.masked_fill(finished, pad_idx)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            finished |= next_tok == eos_idx
            if finished.all():
                break

        ys = ys[:, 1:]
        # Replace EOS and anything after it with PAD.
        after_eos = (ys == eos_idx).cumsum(dim=1) > 0
        return ys.masked_fill(after_eos, pad_idx)


def count_parameters(model):
    """Unique trainable parameters (tied weights counted once)."""
    seen, total = set(), 0
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total
