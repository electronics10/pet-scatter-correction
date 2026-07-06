"""
engine.py -- the training loop, in ONE place.

Two functions, one nested inside the other's job:
  run_epoch(...) -- one pass over a loader (called once per epoch).
  fit(...)       -- the whole training job: build model, loop epochs, checkpoint
                    (called once per experiment). Named `fit` in the scikit-learn
                    / Keras sense of "run the complete training loop", NOT one
                    step.

The library owns the machinery; the caller owns policy. So fit() takes an
explicit `hparams` dict and pre-selected train/val run lists -- it reads no
module-level constants and does no run selection. That keeps scatter_ml
importable and side-effect-free; the *which runs / what ratio* decisions live in
the launch script.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import SinogramWindowDataset, RunBatchSampler
from .model import UNet2p5D
from .losses import LOSSES


def make_loader(run_dirs, cfg, hp):
    """Build a (sampler, loader) pair from run dirs + an hparams dict.

    The sampler is returned alongside the loader because callers must invoke
    sampler.set_epoch(e) each epoch for correct per-epoch reshuffling.
    """
    ds = SinogramWindowDataset(
        run_dirs, cfg,
        window_k=hp["window_k"], split=hp["split"], split_p=hp["split_p"],
        cache_capacity=hp["cache_capacity"], seed=hp["seed"],
    )
    sampler = RunBatchSampler(ds.run_of, hp["batch_size"], seed=hp["seed"],
                              drop_last=False)
    loader = DataLoader(ds, batch_sampler=sampler,
                        num_workers=hp["num_workers"], pin_memory=True)
    return sampler, loader


def run_epoch(model, loader, loss_fn, device, opt=None):
    """One pass over `loader`. Trains if `opt` is given, else evaluates.
    Returns the sample-weighted mean loss."""
    train = opt is not None
    model.train(train)
    total, n = 0.0, 0
    torch.set_grad_enabled(train)
    bar = tqdm(loader, desc="train" if train else "val", leave=False)
    for window, d_norm, label, _scale in bar:
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
        bar.set_postfix(loss=loss.item())
    return total / max(n, 1)


def fit(train_runs, val_runs, cfg, hp, out_dir):
    """Run a full training job. Returns the best validation loss.

    train_runs, val_runs : pre-selected lists of run dirs (disjoint by run).
    cfg                   : the wrapper config (geometry source of truth).
    hp                    : hparams dict (see run_pilot.py for the keys).
    out_dir               : checkpoints written here (best.pt, last.pt).
    """
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr_sampler, tr_loader = make_loader(train_runs, cfg, hp)
    _, va_loader = make_loader(val_runs, cfg, hp)

    model = UNet2p5D(window_k=hp["window_k"], base=hp["base"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    loss_fn = LOSSES[hp["loss"]]
    meta = {"window_k": hp["window_k"], "base": hp["base"]}

    best = float("inf")
    for e in range(hp["epochs"]):
        tr_sampler.set_epoch(e)
        tr = run_epoch(model, tr_loader, loss_fn, device, opt)
        va = run_epoch(model, va_loader, loss_fn, device, opt=None)
        print(f"epoch {e+1:3d}/{hp['epochs']}  train={tr:.5g}  val={va:.5g}",
              flush=True)
        torch.save({"model": model.state_dict(), "args": meta,
                    "epoch": e, "val": va}, out / "last.pt")
        if va < best:
            best = va
            torch.save({"model": model.state_dict(), "args": meta,
                        "epoch": e, "val": va}, out / "best.pt")
    print(f"done. best val = {best:.5g} -> {out/'best.pt'}")
    return best