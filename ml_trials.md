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

Results

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