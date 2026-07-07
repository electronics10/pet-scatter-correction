"""
dataset.py -- 2.5D training samples from merged (zbar, d) sinograms.

One SAMPLE = one merged plane (the target) plus a window of k neighbours along
zbar at the same d (the input channels). The model predicts the target's scatter.

Three design points, each isolated:

1. Per-sample scaling. Total counts span >1 order of magnitude across runs
   (~4e7 to ~1e9). That overall brightness is the DEGENERATE axis (noise level,
   not scatter pattern), so we divide each sample by its input-window mean and
   predict scatter in that scaled space. predict.py multiplies the scale back.
   This makes the network brightness-invariant by construction.

2. Poisson-split API (independent labels), OFF by default.
     split=False (default): input = prompts (trues+scatter), label = the SAME
         scatter realisation. Label noise is CONTAINED in the input -- the
         "independence trap". Fine for a first "does it learn" run.
     split=True: thin trues and scatter to fraction p for the input
         (t_in, sc_in ~ Binom(., p)); build the label from the COMPLEMENTARY
         scatter events (sc - sc_in), rescaled by p/(1-p) to the input's count
         level. sc-sc_in is independent of both sc_in and t_in (Poisson
         splitting), so label noise is exactly independent of the input. Enables
         the clean Noise2Noise argument. Train the split vs non-split ablation by
         flipping this flag.

3. Run-local caching. Merged arrays are large (~350 MB each) and there are
   thousands of runs, so we do NOT cache all merged data to disk. Instead an LRU
   holds a few runs' merged (trues, scatter) in RAM, and RunBatchSampler emits
   each batch from a single run, so a loaded run serves many samples before
   eviction. Merge is recomputed per run per epoch (cheap: one add.at pass).
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

import mcgpu_pet_wrapper as mpw

from .representation import build_geometry


class MergedRunCache:
    """LRU cache of per-run merged (trues, scatter) arrays, keyed by run_dir."""

    def __init__(self, geom, capacity=2):
        self.geom = geom
        self.capacity = capacity
        self._store = OrderedDict()

    def get(self, run_dir, cfg):
        key = str(run_dir)
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        t, _, _ = mpw.read_sinogram_ring_pairs(run_dir, cfg, scatter=False)
        s, _, _ = mpw.read_sinogram_ring_pairs(run_dir, cfg, scatter=True)
        t_m = self.geom.merge(t)                        # (M, A, R)
        s_m = self.geom.merge(s)
        self._store[key] = (t_m, s_m)
        self._store.move_to_end(key)
        while len(self._store) > self.capacity:
            self._store.popitem(last=False)
        return t_m, s_m


class SinogramWindowDataset(Dataset):
    """2.5D windows over a list of runs sharing one config/geometry."""

    def __init__(self, run_dirs, config, window_k=5, split=False, split_p=0.5,
                 cache_capacity=2, eps=1e-6, scale_floor=0.05, min_window_mean=0.05,
                 seed=0):
        assert window_k % 2 == 1, "window_k must be odd"
        self.run_dirs = [Path(r) for r in run_dirs]
        self.config = config
        self.k = window_k
        self.h = window_k // 2
        self.split = split
        self.p = split_p
        self.eps = eps
        self.geom = build_geometry(config)
        self.cache = MergedRunCache(self.geom, capacity=cache_capacity)
        self._rng = np.random.default_rng(seed)
        self.scale_floor = scale_floor
        self.min_window_mean = min_window_mean

        # sample index: (run_id, d, j) where j is position within segment d
        self.index = []
        self.run_of = []       # run_id per sample (for RunBatchSampler grouping)
        for run_id in range(len(self.run_dirs)):
            for d, seg in self.geom.segments.items():
                for j in range(len(seg)):
                    self.index.append((run_id, d, j))
                    self.run_of.append(run_id)
        self.run_of = np.array(self.run_of)

    def __len__(self):
        return len(self.index)

    def _window_ids(self, d, j):
        """Merged plane indices for the window centred at position j in segment d,
        replicate-padded at segment ends."""
        seg = self.geom.segments[d]
        js = np.clip(np.arange(j - self.h, j + self.h + 1), 0, len(seg) - 1)
        return seg[js], seg[j]

    def _binom(self, counts, p):
        return self._rng.binomial(counts.astype(np.int64), p).astype(np.float32)

    def __getitem__(self, i):
        run_id, d, j = self.index[i]
        t_m, s_m = self.cache.get(self.run_dirs[run_id], self.config)
        win_ids, center = self._window_ids(d, j)

        t_win = t_m[win_ids]                            # (k, A, R)
        s_win = s_m[win_ids]

        if self.split:
            t_in = self._binom(t_win, self.p)
            s_in = self._binom(s_win, self.p)
            window = t_in + s_in                        # thinned prompts input
            s_center = s_m[center]
            s_in_center = s_in[self.h]                  # center row of the window
            label = (s_center - s_in_center) * (self.p / (1.0 - self.p))
        else:
            window = t_win + s_win                      # full prompts input
            label = s_m[center]

        scale = max(float(window.mean()), self.scale_floor)   # was: window.mean() + eps
        window = window / scale
        label = label / scale

        d_norm = d / max(self.geom.n_rings - 1, 1)
        return (torch.from_numpy(window.astype(np.float32)),
                torch.tensor(d_norm, dtype=torch.float32),
                torch.from_numpy(label.astype(np.float32))[None],   # (1, A, R)
                torch.tensor(scale, dtype=torch.float32))


class RunBatchSampler(Sampler):
    """Yield batches whose samples all come from ONE run, so the LRU stays warm.

    Run order and within-run order are reshuffled every epoch, so the only lost
    randomness is cross-run mixing WITHIN a batch -- an acceptable, common
    trade for volume/patch sampling. Raise cache_capacity + interleave runs if
    you want more mixing.
    """

    def __init__(self, run_of, batch_size, seed=0, drop_last=False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.by_run = {}
        for idx, r in enumerate(run_of):
            self.by_run.setdefault(int(r), []).append(idx)

    def set_epoch(self, e):
        self.epoch = e

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        run_ids = list(self.by_run.keys())
        rng.shuffle(run_ids)
        for r in run_ids:
            idxs = np.array(self.by_run[r])
            rng.shuffle(idxs)
            for b in range(0, len(idxs), self.batch_size):
                batch = idxs[b:b + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                yield batch.tolist()

    def __len__(self):
        n = 0
        for idxs in self.by_run.values():
            if self.drop_last:
                n += len(idxs) // self.batch_size
            else:
                n += (len(idxs) + self.batch_size - 1) // self.batch_size
        return n


def split_runs_by_recipe(run_root, val_frac=0.15, test_frac=0.15, seed=0):
    """Split COMPLETED runs into train/val/test BY RUN (never by plane).

    Splitting by plane would leak planes of one phantom across sets and inflate
    scores; splitting by run is the leakage guard. Returns (train, val, test)
    lists of run_dir paths.
    """
    run_root = Path(run_root)
    runs = sorted(d for d in run_root.glob("run_*") if (d / "DONE").exists())
    rng = np.random.default_rng(seed)
    rng.shuffle(runs)
    n = len(runs)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = runs[:n_test]
    val = runs[n_test:n_test + n_val]
    train = runs[n_test + n_val:]
    return train, val, test