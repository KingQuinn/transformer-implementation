"""
Synthetic sequence-to-sequence tasks of increasing difficulty.

  copy      x1..xn            -> x1..xn
  reverse   x1..xn            -> xn..x1
  addition  "a+b" (characters) -> digits of a+b

Every task:
  * uses variable-length sequences, so padding masks are actually exercised;
  * builds train / val / test from a single pool of UNIQUE examples, so the
    splits are disjoint (no example in test was ever seen in training);
  * offers an out-of-distribution (OOD) test set with longer inputs than any
    training example, to probe length generalisation.
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2
N_SPECIAL = 3

# addition vocabulary
DIGIT_OFFSET = N_SPECIAL  # '0' -> 3 ... '9' -> 12
PLUS_IDX = DIGIT_OFFSET + 10  # 13
ADDITION_VOCAB = PLUS_IDX + 1  # 14


@dataclass
class TaskSpec:
    name: str
    vocab_size: int
    train_range: tuple  # (min, max) length or digit count, inclusive
    ood_range: tuple


TASKS = {
    "copy": TaskSpec("copy", vocab_size=20, train_range=(5, 12), ood_range=(13, 16)),
    "reverse": TaskSpec("reverse", vocab_size=20, train_range=(5, 12), ood_range=(13, 16)),
    # ranges are number of digits per operand
    "addition": TaskSpec("addition", vocab_size=ADDITION_VOCAB, train_range=(1, 3), ood_range=(4, 4)),
}


# --------------------------------------------------------------------------
def _encode_number(n):
    return [DIGIT_OFFSET + int(c) for c in str(n)]


def decode_tokens(ids, task):
    """Human-readable string for a list of token ids (special tokens dropped)."""
    ids = [i for i in ids if i >= N_SPECIAL]
    if task == "addition":
        return "".join("+" if i == PLUS_IDX else str(i - DIGIT_OFFSET) for i in ids)
    return " ".join(str(i) for i in ids)


def _sample_symbol_pair(rng, task, vocab_size, lo, hi):
    n = int(rng.integers(lo, hi + 1))
    seq = tuple(int(x) for x in rng.integers(N_SPECIAL, vocab_size, size=n))
    tgt = seq if task == "copy" else seq[::-1]
    return seq, tgt


def _sample_addition_pair(rng, lo, hi):
    def num():
        d = int(rng.integers(lo, hi + 1))
        low = 0 if d == 1 else 10 ** (d - 1)
        return int(rng.integers(low, 10**d))

    a, b = num(), num()
    src = tuple(_encode_number(a) + [PLUS_IDX] + _encode_number(b))
    return src, tuple(_encode_number(a + b))


def generate_pairs(task, n, length_range, rng, exclude=frozenset()):
    spec = TASKS[task]
    lo, hi = length_range
    seen, pairs = set(exclude), []
    attempts = 0
    while len(pairs) < n:
        attempts += 1
        if attempts > 50 * n:
            raise RuntimeError(f"could not generate {n} unique {task} examples")
        if task == "addition":
            src, tgt = _sample_addition_pair(rng, lo, hi)
        else:
            src, tgt = _sample_symbol_pair(rng, task, spec.vocab_size, lo, hi)
        if src in seen:
            continue
        seen.add(src)
        pairs.append((src, tgt))
    return pairs


def build_splits(task, n_train, n_val, n_test, n_ood, seed=0):
    """Disjoint train/val/test (in-distribution) + an OOD-length test set."""
    rng = np.random.default_rng(seed)
    spec = TASKS[task]
    pool = generate_pairs(task, n_train + n_val + n_test, spec.train_range, rng)
    train = pool[:n_train]
    val = pool[n_train : n_train + n_val]
    test = pool[n_train + n_val :]
    ood = generate_pairs(task, n_ood, spec.ood_range, rng, exclude={s for s, _ in pool})
    return {k: Seq2SeqDataset(v) for k, v in
            dict(train=train, val=val, test=test, ood=ood).items()}


# --------------------------------------------------------------------------
class Seq2SeqDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src = torch.tensor(src, dtype=torch.long)
        tgt = torch.tensor(tgt, dtype=torch.long)
        tgt_in = torch.cat([torch.tensor([SOS_IDX]), tgt])   # teacher-forcing input
        tgt_out = torch.cat([tgt, torch.tensor([EOS_IDX])])  # shifted target
        return src, tgt_in, tgt_out


def collate(batch):
    src, tgt_in, tgt_out = zip(*batch)
    pad = lambda xs: pad_sequence(xs, batch_first=True, padding_value=PAD_IDX)
    return pad(src), pad(tgt_in), pad(tgt_out)
