"""
train.py -- train the 2.5D scatter-prediction U-Net.

Defaults reflect the settled first-run choices: L2 loss, NON-split (correlated)
labels, window_k=5. Flip --loss poisson or --split to run the ablations. The
train/val/test split is BY RUN (leakage guard).

Usage (from a machine with the data + torch + mcgpu-pet-wrapper installed):
    python -m scatter_ml.train --run_root ../data-gen/data/runs --epochs 20
    python -m scatter_ml.train --run_root ... --loss poisson --split   # ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import mcgpu_pet_wrapper as mpw

from .dataset import (SinogramWindowDataset, RunBatchSampler,
                      split_runs_by_recipe)
from .model import UNet2p5D
from .losses import LOSSES


def _loader(run_dirs, cfg, args, shuffle_runs):
    ds = SinogramWindowDataset(run_dirs, cfg, window_k=args.window_k,
                               split=args.split, split_p=args.split_p,
                               cache_capacity=args.cache_capacity, seed=args.seed)
    sampler = RunBatchSampler(ds.run_of, args.batch_size, seed=args.seed,
                              drop_last=False)
    loader = DataLoader(ds, batch_sampler=sampler, num_workers=args.num_workers,
                        pin_memory=True)
    return ds, sampler, loader


def _run_epoch(model, loader, loss_fn, device, opt=None):
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    torch.set_grad_enabled(train)
    loader = tqdm(loader, desc="train" if train else "val", leave=False)
    for window, d_norm, label, _scale in loader:
        window = window.to(device, non_blocking=True)
        d_norm = d_norm.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        pred = model(window, d_norm)
        loss = loss_fn(pred, label)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        total += loss.item() * window.shape[0]
        n += window.shape[0]
        loader.set_postfix(loss=loss.item())   # live loss per batch
    return total / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--config", default=None,
                    help="config.json; default: first run's config")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window_k", type=int, default=5)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--loss", choices=list(LOSSES), default="mse")
    ap.add_argument("--split", action="store_true",
                    help="independent-label Poisson split (default off)")
    ap.add_argument("--split_p", type=float, default=0.5)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--cache_capacity", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--limit_train_runs", type=int, default=0,
                    help="use only the first N train runs (pilot/learning-curve)")
    ap.add_argument("--limit_val_runs", type=int, default=0)
    ap.add_argument("--limit_test_runs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    train_runs, val_runs, test_runs = split_runs_by_recipe(
        args.run_root, args.val_frac, args.test_frac, args.seed)
    if args.limit_train_runs:
        train_runs = train_runs[:args.limit_train_runs]
    (out / "split.json").write_text(json.dumps(
        {"train": [str(r) for r in train_runs],
         "val": [str(r) for r in val_runs],
         "test": [str(r) for r in test_runs]}, indent=2))
    print(f"runs: train={len(train_runs)} val={len(val_runs)} test={len(test_runs)}")

    cfg = mpw.load_config(args.config) if args.config \
        else mpw.load_config(train_runs[0] / "config.json")

    _, tr_sampler, tr_loader = _loader(train_runs, cfg, args, shuffle_runs=True)
    _, _, va_loader = _loader(val_runs, cfg, args, shuffle_runs=False)

    model = UNet2p5D(window_k=args.window_k, base=args.base).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = LOSSES[args.loss]

    best = float("inf")
    for e in range(args.epochs):
        tr_sampler.set_epoch(e)
        tr = _run_epoch(model, tr_loader, loss_fn, device, opt)
        va = _run_epoch(model, va_loader, loss_fn, device, opt=None)
        print(f"epoch {e+1:3d}/{args.epochs}  train={tr:.5g}  val={va:.5g}")
        torch.save({"model": model.state_dict(), "args": vars(args),
                    "epoch": e, "val": va}, out / "last.pt")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": e, "val": va}, out / "best.pt")
    print(f"done. best val = {best:.5g}  ->  {out/'best.pt'}")


if __name__ == "__main__":
    main()