"""
Produce every figure and table in the README from saved checkpoints/results.

    python scripts/analyze.py --tasks copy reverse addition --seeds 0 1 2
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformer.data import (EOS_IDX, PAD_IDX, SOS_IDX, TASKS, build_splits,  # noqa: E402
                              collate, decode_tokens)
from transformer.masks import make_src_mask, make_tgt_mask  # noqa: E402
from transformer.model import FeedForward, PositionalEncoding, ResidualConnection  # noqa: E402
from transformer.train import Config, build_model  # noqa: E402

FIG = "results/figures"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(task, seed):
    path = f"checkpoints/{task}_s{seed}/best.pt"
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = Config(**ck["config"])
    m = build_model(cfg, TASKS[task].vocab_size)
    m.load_state_dict(ck["model"])
    return m.eval(), cfg


# --------------------------------------------------------------------------
def summary_table(tasks, seeds):
    rows = []
    for t in tasks:
        R = [json.load(open(f"results/{t}_s{s}.json")) for s in seeds
             if os.path.exists(f"results/{t}_s{s}.json")]
        if not R:
            continue
        get = lambda split, k: np.array([r["final"][split][k] for r in R]) * 100
        rows.append({
            "task": t, "n_seeds": len(R), "params": R[0]["params"],
            "epochs": [r["epochs_trained"] for r in R],
            "test_tf_tok": (get("test", "token_acc_teacher_forced").mean(), get("test", "token_acc_teacher_forced").std()),
            "test_em": (get("test", "exact_match_greedy").mean(), get("test", "exact_match_greedy").std()),
            "ood_em": (get("ood", "exact_match_greedy").mean(), get("ood", "exact_match_greedy").std()),
            "per_seed_test_em": get("test", "exact_match_greedy").round(2).tolist(),
            "per_seed_ood_em": get("ood", "exact_match_greedy").round(2).tolist(),
        })
    json.dump(rows, open("results/summary.json", "w"), indent=2)
    print("\n| Task | Params | Test token acc (TF) | Test exact match (greedy) | OOD exact match (greedy) |")
    print("|---|---|---|---|---|")
    for r in rows:
        f = lambda x: f"{x[0]:.2f} ± {x[1]:.2f}"
        print(f"| {r['task']} | {r['params']:,} | {f(r['test_tf_tok'])} | {f(r['test_em'])} | {f(r['ood_em'])} |")
    return rows


def training_curves(tasks, seeds):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    colors = dict(copy="#2a78b5", reverse="#d1495b", addition="#3f9b5a")
    for t in tasks:
        for s in seeds:
            p = f"results/{t}_s{s}.json"
            if not os.path.exists(p):
                continue
            h = json.load(open(p))["history"]
            e = [r["epoch"] for r in h]
            kw = dict(color=colors[t], alpha=0.85, lw=1.4, label=t if s == seeds[0] else None)
            ax[0].plot(e, [r["train_loss"] for r in h], **kw)
            ax[1].plot(e, [100 * r["val_exact_match_greedy"] for r in h], **kw)
    ax[0].set(yscale="log", xlabel="epoch", ylabel="train cross-entropy", title="Training loss")
    ax[1].set(xlabel="epoch", ylabel="val exact match, greedy (%)", title="Validation accuracy (autoregressive)", ylim=(0, 101))
    ax[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/training_curves.png")
    plt.close(fig)


@torch.no_grad()
def accuracy_by_length(tasks, seed):
    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 3.2), squeeze=False)
    out = {}
    for ax, t in zip(axes[0], tasks):
        m, cfg = load(t, seed)
        s = build_splits(t, cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_ood, seed=cfg.seed)
        lens, ok = [], []
        for split in ("test", "ood"):
            ds = s[split]
            for i in range(0, len(ds), 256):
                src, _, tgt_out = collate([ds[j] for j in range(i, min(i + 256, len(ds)))])
                tgt = tgt_out.masked_fill(tgt_out == EOS_IDX, PAD_IDX)
                pred = m.greedy_decode(src, make_src_mask(src), SOS_IDX, EOS_IDX, PAD_IDX, tgt.size(1) + 5)
                L = max(pred.size(1), tgt.size(1))
                pred = nn.functional.pad(pred, (0, L - pred.size(1)))
                tgt = nn.functional.pad(tgt, (0, L - tgt.size(1)))
                ok += (pred == tgt).all(1).tolist()
                lens += (src != PAD_IDX).sum(1).tolist()
        lens, ok = np.array(lens), np.array(ok)
        xs = sorted(set(lens))
        acc = [100 * ok[lens == x].mean() for x in xs]
        lo, hi = TASKS[t].train_range
        max_train_len = hi if t != "addition" else 2 * hi + 1
        ax.axvspan(min(xs) - 0.5, max_train_len + 0.5, color="#eee", zorder=0, label="training lengths")
        ax.plot(xs, acc, "o-", color="#333", ms=4)
        ax.set(title=t, xlabel="source length (tokens)", ylabel="exact match (%)", ylim=(-3, 103))
        out[t] = dict(zip([int(x) for x in xs], [round(a, 2) for a in acc]))
    axes[0][0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{FIG}/accuracy_by_length.png")
    plt.close(fig)
    json.dump(out, open("results/accuracy_by_length.json", "w"), indent=2)


@torch.no_grad()
def cross_attention_maps(tasks, seed, n_alignment=500):
    """Cross-attention of the last decoder layer on a test example, plus a
    QUANTITATIVE alignment score for copy/reverse: how often the head-averaged
    attention argmax for output i lands on the source token it must copy."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 3.8), squeeze=False)
    align = {}
    for ax, t in zip(axes[0], tasks):
        m, cfg = load(t, seed)
        s = build_splits(t, cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_ood, seed=cfg.seed)
        ds = s["test"]

        def attn_for(idx):
            src, tgt_in, _ = collate([ds[idx]])
            m(src, tgt_in, make_src_mask(src), make_tgt_mask(tgt_in))
            return src[0], m.decoder.layers[-1].cross_attention.last_attention[0]  # (H, T, S)

        if t in ("copy", "reverse"):
            hits = tot = 0
            for i in range(min(n_alignment, len(ds))):
                src, A = attn_for(i)
                n = len(src)
                arg = A.mean(0)[:n].argmax(-1)  # output steps 0..n-1 (exclude EOS step)
                expect = torch.arange(n) if t == "copy" else torch.arange(n - 1, -1, -1)
                hits += (arg == expect).sum().item()
                tot += n
            align[t] = round(100 * hits / tot, 2)

        ex = max(range(min(200, len(ds))), key=lambda i: len(ds.pairs[i][0]))  # longest of first 200
        src, A = attn_for(ex)
        tgt = list(ds.pairs[ex][1])
        A = A.mean(0).numpy()  # (T, S)
        ax.imshow(A, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        sl = decode_tokens(src.tolist(), t).split() if t != "addition" else list(decode_tokens(src.tolist(), t))
        tl = (decode_tokens(tgt, t).split() if t != "addition" else list(decode_tokens(tgt, t))) + ["<eos>"]
        ax.set_xticks(range(len(sl)), sl, fontsize=8)
        ax.set_yticks(range(len(tl)), tl, fontsize=8)
        ax.set(xlabel="source", ylabel="output", title=f"{t}: last-layer cross-attn (head mean)")
    fig.tight_layout()
    fig.savefig(f"{FIG}/cross_attention.png")
    plt.close(fig)
    json.dump(align, open("results/attention_alignment.json", "w"), indent=2)
    print("attention alignment (% argmax on correct source token):", align)


def positional_encoding_figure():
    pe = PositionalEncoding(128, max_len=100, dropout=0.0).pe[0].numpy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
    im = ax[0].imshow(pe.T, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    ax[0].set(xlabel="position", ylabel="embedding dimension", title="Sinusoidal positional encoding (d=128)")
    fig.colorbar(im, ax=ax[0])
    for d in (0, 16, 64, 126):
        ax[1].plot(pe[:, d], label=f"dim {d}")
    ax[1].set(xlabel="position", ylabel="value", title="Low dims oscillate fast; high dims slowly")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/positional_encoding.png")
    plt.close(fig)


def residual_gradient_experiment(d=64, depths=(2, 4, 8, 16, 32), trials=5):
    """Input-gradient norm vs depth, with and without residual connections.

    The notebook's version used loss = output.sum(). After a LayerNorm with
    gamma=1, beta=0 each row sums to exactly 0, so that gradient is
    identically zero -- the experiment measured nothing. Here the loss is a
    projection onto a fixed random direction instead."""
    torch.manual_seed(0)
    res = {"plain": [], "residual": []}
    for L in depths:
        for kind in res:
            vals = []
            for _ in range(trials):
                ffns = nn.ModuleList([FeedForward(d, 4 * d, 0.0) for _ in range(L)])
                norms = nn.ModuleList([ResidualConnection(d, 0.0) for _ in range(L)])
                x = torch.randn(8, 10, d, requires_grad=True)
                h = x
                for f, r in zip(ffns, norms):
                    h = r(h, f) if kind == "residual" else r.norm(f(h))  # same LN, no skip
                (h * torch.randn_like(h)).sum().backward()
                vals.append(x.grad.norm().item())
            res[kind].append(float(np.mean(vals)))
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(depths, res["plain"], "o-", label="no residual (FFN → LN)")
    ax.plot(depths, res["residual"], "s-", label="residual (Post-LN)")
    ax.set(xscale="log", yscale="log", xlabel="depth (layers)", ylabel="‖∂loss/∂input‖",
           title="Gradient reaching the input")
    ax.set_xticks(depths, [str(x) for x in depths])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIG}/residual_gradients.png")
    plt.close(fig)
    json.dump({"depths": list(depths), **res}, open("results/residual_gradients.json", "w"), indent=2)
    print("residual experiment:", res)


def write_results_markdown(rows):
    """Render results/RESULTS.md and splice it into README.md between markers,
    so the numbers in the README always come from the result files."""
    f = lambda x: f"{x[0]:.1f} ± {x[1]:.1f}"
    align = json.load(open("results/attention_alignment.json")) if os.path.exists("results/attention_alignment.json") else {}
    bylen = json.load(open("results/accuracy_by_length.json")) if os.path.exists("results/accuracy_by_length.json") else {}
    L = ["| Task | Params | Seeds | Epochs | Test token acc, teacher-forced (%) | **Test exact match, greedy (%)** | OOD-length exact match, greedy (%) |",
         "|---|---:|---:|---|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['task']} | {r['params']:,} | {r['n_seeds']} | {', '.join(map(str, r['epochs']))} | "
                 f"{f(r['test_tf_tok'])} | **{f(r['test_em'])}** | {f(r['ood_em'])} |")
    L += ["", "Mean ± std over seeds. Per-seed test exact match: " +
          "; ".join(f"{r['task']} {r['per_seed_test_em']}" for r in rows) + "."]
    if align:
        L += ["", "**Attention alignment** (seed 0, 500 test examples): fraction of output steps whose "
              "head-averaged last-layer cross-attention argmax lands on the source token being emitted: " +
              ", ".join(f"{k} **{v:.1f}%**" for k, v in align.items()) + "."]
    if bylen:
        L += ["", "**Exact match by source length** (seed 0):", ""]
        for t, d in bylen.items():
            L.append(f"- {t}: " + ", ".join(f"len {k}: {v:.0f}%" for k, v in d.items()))
    md = "\n".join(L) + "\n"
    open("results/RESULTS.md", "w").write(md)
    if os.path.exists("README.md"):
        s = open("README.md").read()
        a, b = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
        if a in s and b in s:
            s = s[: s.index(a) + len(a)] + "\n" + md + s[s.index(b):]
            open("README.md", "w").write(s)
            print("README.md results block updated")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["copy", "reverse", "addition"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = p.parse_args()
    os.makedirs(FIG, exist_ok=True)
    rows = summary_table(a.tasks, a.seeds)
    training_curves(a.tasks, a.seeds)
    accuracy_by_length(a.tasks, a.seeds[0])
    cross_attention_maps(a.tasks, a.seeds[0])
    positional_encoding_figure()
    residual_gradient_experiment()
    write_results_markdown(rows)
