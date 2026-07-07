## trial1 (20260706)

```Python
"""
run_pilot.py -- flag-free pilot training. Edit the CONSTANTS, then:

    python run_pilot.py

This script owns POLICY only: which runs, what split ratio, which hyper-params.
The training loop itself lives in scatter_ml.engine.fit (imported below), so
there is exactly one copy of it. select_and_split stays here because "which runs
/ what ratio" is an experiment decision, not reusable machinery.

Logic: take N_RUNS completed runs, split THOSE by ratio into train/val/test (so
val/test are automatically small, being fractions of N_RUNS), then fit. The split
is frozen to split.json and reloaded on rerun, so the pilot trains on an identical
partition even if more runs finish meanwhile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import mcgpu_pet_wrapper as mpw
from scatter_ml.engine import fit

# ------------------------- CONSTANTS (edit these) -------------------------
RUN_ROOT = "../MCGPU_data/runs"
OUT      = "checkpoints_pilot"
N_RUNS   = 70            # total runs to use, split by ratio below
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED     = 0

HP = {                   # hyper-parameters handed to engine.fit
    "epochs": 15,
    "batch_size": 16,
    "lr": 1e-3,
    "window_k": 5,
    "base": 32,
    "loss": "mse",       # "mse" or "poisson"
    "split": False,      # independent-label Poisson split (off first)
    "split_p": 0.5,
    "cache_capacity": 8, # runs held in RAM; higher = less re-read per epoch
    "num_workers": 4,
    "seed": SEED,
}
# --------------------------------------------------------------------------


def select_and_split(run_root, n_runs, val_frac, test_frac, seed, out_dir):
    """Pick n_runs completed runs and split THEM by ratio. Frozen to split.json:
    if it exists, reuse it verbatim (stable across reruns / new completed runs)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sp = out_dir / "split.json"
    if sp.exists():
        d = json.loads(sp.read_text())
        print(f"reusing frozen split: {sp}")
        return ([Path(p) for p in d["train"]], [Path(p) for p in d["val"]],
                [Path(p) for p in d["test"]])

    runs = sorted(p for p in Path(run_root).glob("run_*") if (p / "DONE").exists())
    if len(runs) < n_runs:
        raise ValueError(f"asked for {n_runs} runs but only {len(runs)} completed.")
    rng = np.random.default_rng(seed)
    rng.shuffle(runs)
    runs = runs[:n_runs]                      # subset FIRST, then split these
    n_test = int(n_runs * test_frac)
    n_val = int(n_runs * val_frac)
    test, val, train = (runs[:n_test], runs[n_test:n_test + n_val],
                        runs[n_test + n_val:])
    sp.write_text(json.dumps({"train": [str(r) for r in train],
                              "val": [str(r) for r in val],
                              "test": [str(r) for r in test]}, indent=2))
    print(f"froze split -> {sp}")
    return train, val, test


def main():
    train_runs, val_runs, test_runs = select_and_split(
        RUN_ROOT, N_RUNS, VAL_FRAC, TEST_FRAC, SEED, OUT)
    print(f"runs: train={len(train_runs)} val={len(val_runs)} test={len(test_runs)}")
    cfg = mpw.load_config(train_runs[0] / "config.json")
    fit(train_runs, val_runs, cfg, HP, OUT)


if __name__ == "__main__":
    main()
```

**Results**

reusing frozen split: checkpoints_pilot/split.json
runs: train=50 val=10 test=10
epoch   1/15  train=227.27  val=560.92
epoch   2/15  train=80.837  val=25.809
epoch   3/15  train=18.841  val=14.351
epoch   4/15  train=8.6357  val=7.9255
epoch   5/15  train=11.022  val=773.53 
epoch   6/15  train=7.4485  val=27.6          
epoch   7/15  train=9.0871  val=1.2877   
epoch   8/15  train=7.3249  val=552.82 
epoch   9/15  train=7.5565  val=575.51
epoch  10/15  train=7.8141  val=711.96  
epoch  11/15  train=11.023  val=81.894 
epoch  12/15  train=6.2506  val=147.68
epoch  13/15  train=7.5305  val=613.03 
epoch  14/15  train=6.7544  val=63.233 
epoch  15/15  train=8.0414  val=140.77
done. best val = 1.2877 -> checkpoints_pilot/best.pt

**Analysis**

```bash
window mean (raw counts) percentiles:
     0.0%  1e-06
     0.1%  1e-06
     1.0%  1e-06
     5.0%  5.217e-05
    50.0%  4.735
    95.0%  28.49
   100.0%  33.41

samples below candidate floors (as fraction of median window mean):
  < 0.001*median (=0.004735): 82  (8.63%)
  < 0.01*median (=0.04735): 145  (15.26%)
  < 0.05*median (=0.2367): 197  (20.74%)
  < 0.1*median (=0.4735): 223  (23.47%)

scaled-label max (label/scale) percentiles -- the exploding quantity:
    50.0%  1.635
    95.0%  224.7
    99.0%  4956
    99.9%  4.895e+04
   100.0%  4.895e+04

ratio  max / median  of scaled-label-max = 2.994e+04   (huge => outlier-driven val loss)
```

The diagnosis is confirmed, and the numbers give exact thresholds. Reading them against the prediction:

- **Window-mean low tail:** the 0th–1st percentile is `1e-6` — i.e. exactly `eps`, meaning those windows are *literally empty* (`window.mean() = 0`, so `scale = eps`). At least 1% of samples are empty planes. Prediction confirmed.
- **Scaled-label max:** median `1.6`, but 99th percentile `4956` and max `4.9e4` — the `max/median` ratio is `3e4`. The loss is squared, so those samples contribute ~`(4.9e4)² ≈ 2.4e9` versus a median contribution of ~`2.7`. A single empty plane outweighs a *billion* normal samples. That is the entire val-loss thrashing. Prediction confirmed, decisively.

**One thing the numbers add that I underweighted:** this isn't a rare handful. **8.6% of samples fall below 0.1% of the median**, and 23% below 10%. Empty/near-empty planes are a *large* fraction — expected, since high-`d` (very oblique) segments have few rings and the axial-edge planes there carry almost no counts. So the fix isn't just "clip a few outliers"; it's "a quarter of your samples are low-signal, and the scaling makes the emptiest ones explode."

This reframes the two fixes — both are now clearly justified, and they do *different* jobs:

**Fix 1 — scale floor (stops the explosion).** Replace `scale = window.mean() + eps` with `scale = max(window.mean(), FLOOR)`. From the percentiles, the median window mean is `4.7`; a floor around `0.05`–`0.24` (1–5% of median) caps the denominator so an empty plane's label can't inflate past ~(true counts)/FLOOR. This keeps empty planes in the dataset but harmless.

**Fix 2 — drop near-empty planes (removes wasted samples).** Skip samples whose window mean is below a threshold at index-build time. Justification from the data: a plane with `<1%` of median counts carries almost no scatter to learn, so it's noise injection, not signal. Dropping the `~15%` below `0.01×median` also shrinks every epoch by ~15% — free speed.

They're complementary, not redundant: Fix 2 removes the *truly* empty planes (no signal); Fix 1 protects against the *semi*-empty ones that survive the cut but still have small denominators. Stress-test of "just floor it": flooring alone leaves 15–23% of training samples as near-empty low-signal planes that the model wastes capacity on and that still dominate gradients occasionally; dropping alone still lets a plane just above the drop threshold explode. Under the hypothesis that scatter is only worth learning where there are counts to predict, you want both — a hard cut for the empty, a soft floor for the rest.

Here are the two edits to `dataset.py`. First, the scale floor in `__getitem__`:

```python
# in SinogramWindowDataset.__init__, add a param:
def __init__(self, run_dirs, config, window_k=5, split=False, split_p=0.5,
             cache_capacity=2, eps=1e-6, scale_floor=0.05, min_window_mean=0.05,
             seed=0):
    ...
    self.scale_floor = scale_floor
    self.min_window_mean = min_window_mean
```

```python
# in __getitem__, replace the scale line:
scale = max(float(window.mean()), self.scale_floor)   # was: window.mean() + eps
```

Second, the empty-plane skip. This is subtler because the sample index is built in `__init__` *before* any data is loaded, so you can't test window means there without loading runs. Two options:

- **Cheap/approximate:** skip by geometry — drop the axial-edge positions of high-`d` segments, which are the empty ones, without loading data. But that hard-codes an assumption about *where* empties are.
- **Correct/data-driven:** filter at first access. Keep the full index, but in `__getitem__` return a sentinel for sub-threshold samples and filter in a custom `collate_fn` — messy with the batch sampler.

Given the batch sampler, the clean approach is a **one-time pre-scan** that records each sample's window mean (the diagnostic already computes this), then drops sub-threshold entries from `self.index`. It costs one pass but makes every epoch faster and cleaner.

Before I write the pre-scan version: is a one-time pre-scan at dataset construction acceptable (adds ~30 s at startup, given the speed we just saw), or do you prefer the zero-cost geometric skip that assumes empties are the high-`d` axial edges? The data-driven pre-scan is safer; the geometric one is instant but assumption-laden. Which do you want?

## trial2 (20260707)

```Python
RUN_ROOT = "../MCGPU_data/runs"
OUT      = "checkpoints_pilot"
N_RUNS   = 70            # total runs to use, split by ratio below
VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED     = 0

HP = {                   # hyper-parameters handed to engine.fit
    "epochs": 25,
    "batch_size": 16,
    "lr": 5e-4,
    "window_k": 7,
    "base": 48,
    "loss": "poisson",       # "mse" or "poisson"
    "split": True,      # independent-label Poisson split (off first)
    "split_p": 0.5,
    "cache_capacity": 8, # runs held in RAM; higher = less re-read per epoch
    "num_workers": 4,
    "seed": SEED,
}
```
An aggressive modification. If it works I won't know why, and if it fails I won't know which knob to blame