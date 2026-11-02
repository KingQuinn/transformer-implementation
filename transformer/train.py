"""
Training / evaluation pipeline.

    python -m transformer.train --task reverse --seed 0

Writes
    checkpoints/<task>_s<seed>/{last.pt,best.pt}   (resumable with --resume)
    results/<task>_s<seed>.json                   (config, history, final metrics)

Model selection uses AUTOREGRESSIVE (greedy) exact-match on the validation
set; final numbers are reported on a held-out test set that is never used
for any decision, plus an out-of-distribution (longer) test set.
"""

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import EOS_IDX, PAD_IDX, SOS_IDX, TASKS, build_splits, collate
from .masks import make_src_mask, make_tgt_mask
from .model import Transformer, count_parameters


@dataclass
class Config:
    task: str = "copy"
    seed: int = 0
    # data
    n_train: int = 20000
    n_val: int = 1000
    n_test: int = 2000
    n_ood: int = 1000
    # model (scaled-down "base")
    d_model: int = 128
    n_heads: int = 4
    num_layers: int = 2
    d_ff: int = 512
    dropout: float = 0.1
    # optimisation (paper: Adam(0.9, 0.98, 1e-9) + inverse-sqrt warmup)
    batch_size: int = 128
    epochs: int = 30
    warmup_steps: int = 400
    lr_factor: float = 1.0
    label_smoothing: float = 0.0
    grad_clip: float = 1.0
    patience: int = 5
    out_dir: str = "."


class NoamScheduler:
    """lr = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)   (Eq. 3)."""

    def __init__(self, optimizer, d_model, warmup_steps, factor=1.0):
        self.optimizer, self.d_model = optimizer, d_model
        self.warmup_steps, self.factor = warmup_steps, factor
        self.step_num = 0
        self._set(self.lr_at(1))  # never start at lr=0 / divide by zero

    def lr_at(self, step):
        step = max(step, 1)
        return self.factor * self.d_model**-0.5 * min(step**-0.5, step * self.warmup_steps**-1.5)

    def _set(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def step(self):
        self.step_num += 1
        self._set(self.lr_at(self.step_num))

    def get_last_lr(self):
        return self.lr_at(self.step_num)

    def state_dict(self):
        return {"step_num": self.step_num}

    def load_state_dict(self, d):
        self.step_num = d["step_num"]
        self._set(self.lr_at(self.step_num))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, vocab_size):
    return Transformer(
        src_vocab_size=vocab_size, tgt_vocab_size=vocab_size,
        d_model=cfg.d_model, n_heads=cfg.n_heads,
        num_encoder_layers=cfg.num_layers, num_decoder_layers=cfg.num_layers,
        d_ff=cfg.d_ff, max_seq_len=128, dropout=cfg.dropout,
    )


# --------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, scheduler, device, grad_clip):
    model.train()
    total, n_tok = 0.0, 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        logits = model(src, tgt_in, make_src_mask(src, PAD_IDX), make_tgt_mask(tgt_in, PAD_IDX))
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        k = (tgt_out != PAD_IDX).sum().item()
        total += loss.item() * k
        n_tok += k
    return total / n_tok


@torch.no_grad()
def evaluate(model, dataset, criterion, device, batch_size=256):
    """Teacher-forced loss + token accuracy, and greedy (autoregressive)
    exact-match accuracy -- the metric that reflects real use."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    loss_sum = tok_correct = n_tok = seq_correct = n_seq = 0
    for src, tgt_in, tgt_out in loader:
        src, tgt_in, tgt_out = src.to(device), tgt_in.to(device), tgt_out.to(device)
        src_mask = make_src_mask(src, PAD_IDX)

        logits = model(src, tgt_in, src_mask, make_tgt_mask(tgt_in, PAD_IDX))
        nonpad = tgt_out != PAD_IDX
        loss_sum += criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1)).item() * nonpad.sum().item()
        tok_correct += ((logits.argmax(-1) == tgt_out) & nonpad).sum().item()
        n_tok += nonpad.sum().item()

        # Greedy decoding with NO access to the target.
        target = tgt_out.masked_fill(tgt_out == EOS_IDX, PAD_IDX)
        pred = model.greedy_decode(src, src_mask, SOS_IDX, EOS_IDX, PAD_IDX,
                                   max_len=target.size(1) + 5)
        L = max(pred.size(1), target.size(1))
        pred = nn.functional.pad(pred, (0, L - pred.size(1)), value=PAD_IDX)
        target = nn.functional.pad(target, (0, L - target.size(1)), value=PAD_IDX)
        seq_correct += (pred == target).all(dim=1).sum().item()
        n_seq += src.size(0)

    return {
        "loss": loss_sum / n_tok,
        "token_acc_teacher_forced": tok_correct / n_tok,
        "exact_match_greedy": seq_correct / n_seq,
    }


# --------------------------------------------------------------------------
def run(cfg, resume=False):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{cfg.task}_s{cfg.seed}"
    ckpt_dir = os.path.join(cfg.out_dir, "checkpoints", run_name)
    res_dir = os.path.join(cfg.out_dir, "results")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    splits = build_splits(cfg.task, cfg.n_train, cfg.n_val, cfg.n_test, cfg.n_ood, seed=cfg.seed)
    vocab = TASKS[cfg.task].vocab_size
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(splits["train"], batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, generator=g)

    model = build_model(cfg, vocab).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = NoamScheduler(optimizer, cfg.d_model, cfg.warmup_steps, cfg.lr_factor)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=cfg.label_smoothing)

    history, start_epoch, best_em, bad_epochs = [], 0, -1.0, 0
    last_path = os.path.join(ckpt_dir, "last.pt")
    best_path = os.path.join(ckpt_dir, "best.pt")

    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        torch.set_rng_state(ck["torch_rng"])
        g.set_state(ck["loader_rng"])
        history, start_epoch = ck["history"], ck["epoch"] + 1
        best_em, bad_epochs = ck["best_em"], ck["bad_epochs"]
        print(f"resumed from epoch {start_epoch}")

    n_params = count_parameters(model)
    print(f"[{run_name}] params={n_params:,} device={device} "
          f"train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])} ood={len(splits['ood'])}")

    for epoch in range(start_epoch, cfg.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, cfg.grad_clip)
        val = evaluate(model, splits["val"], criterion, device)
        row = {"epoch": epoch + 1, "train_loss": train_loss, "lr": scheduler.get_last_lr(),
               "seconds": time.time() - t0, **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        print(f"ep {epoch+1:3d} | train {train_loss:.4f} | val loss {val['loss']:.4f} "
              f"tok {val['token_acc_teacher_forced']:.4f} EM {val['exact_match_greedy']:.4f} | {row['seconds']:.1f}s",
              flush=True)

        improved = val["exact_match_greedy"] > best_em
        if improved:
            best_em, bad_epochs = val["exact_match_greedy"], 0
        else:
            bad_epochs += 1

        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "config": asdict(cfg), "epoch": epoch,
                 "history": history, "best_em": best_em, "bad_epochs": bad_epochs,
                 "torch_rng": torch.get_rng_state(), "loader_rng": g.get_state()}
        torch.save(state, last_path)
        if improved:
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "epoch": epoch}, best_path)

        if best_em == 1.0 or bad_epochs >= cfg.patience:
            break

    # Final evaluation of the BEST checkpoint on untouched splits.
    model.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=False)["model"])
    final = {k: evaluate(model, splits[k], criterion, device) for k in ("val", "test", "ood")}
    out = {"run": run_name, "params": n_params, "config": asdict(cfg),
           "epochs_trained": len(history), "best_epoch": int(np.argmax([h["val_exact_match_greedy"] for h in history]) + 1),
           "history": history, "final": final}
    with open(os.path.join(res_dir, f"{run_name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[{run_name}] TEST EM={final['test']['exact_match_greedy']:.4f}  OOD EM={final['ood']['exact_match_greedy']:.4f}")
    return out


def main():
    p = argparse.ArgumentParser()
    for k, v in asdict(Config()).items():
        p.add_argument(f"--{k}", type=type(v), default=v)
    p.add_argument("--resume", action="store_true")
    args = vars(p.parse_args())
    resume = args.pop("resume")
    run(Config(**args), resume=resume)


if __name__ == "__main__":
    main()
