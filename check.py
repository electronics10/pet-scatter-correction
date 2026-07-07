"""
check_window_means.py -- confirm (or refute) the empty-plane outlier diagnosis.

Diagnosis under test: a few samples have near-zero window means, so per-sample
scaling (divide by window.mean()+eps) inflates their labels and explodes the L2
loss. If true, we expect a heavy LOW tail in the window-mean distribution and a
few samples whose scaled-label max is orders of magnitude above the median.

Prints, over the VAL runs (same split the pilot froze):
  - percentiles of per-sample window mean (raw counts)
  - how many samples fall below candidate floors
  - percentiles of the scaled-label MAX (label/scale), the quantity that explodes
"""
import json
from pathlib import Path
import numpy as np
import mcgpu_pet_wrapper as mpw
from scatter_ml.dataset import SinogramWindowDataset

OUT = "checkpoints_pilot/trial1_202600706"
val_runs = [Path(p) for p in json.loads(Path(OUT, "split.json").read_text())["val"]]
cfg = mpw.load_config(val_runs[0] / "config.json")

ds = SinogramWindowDataset(val_runs, cfg, window_k=5, split=False,
                           cache_capacity=4, seed=0)

means, smax = [], []
for i in range(len(ds)):
    window, d_norm, label, scale = ds[i]
    means.append(float(scale))            # scale = window.mean()+eps
    smax.append(float(label.max()))       # label already divided by scale
means = np.array(means); smax = np.array(smax)

def pct(a, ps): return {p: float(np.percentile(a, p)) for p in ps}

print("n samples:", len(means))
print("\nwindow mean (raw counts) percentiles:")
for p, v in pct(means, [0, 0.1, 1, 5, 50, 95, 100]).items():
    print(f"  {p:6.1f}%  {v:.4g}")

med = np.median(means)
print("\nsamples below candidate floors (as fraction of median window mean):")
for frac in [0.001, 0.01, 0.05, 0.1]:
    thr = frac * med
    print(f"  < {frac:.3g}*median (={thr:.4g}): "
          f"{int((means < thr).sum())}  ({100*(means<thr).mean():.2f}%)")

print("\nscaled-label max (label/scale) percentiles -- the exploding quantity:")
for p, v in pct(smax, [50, 95, 99, 99.9, 100]).items():
    print(f"  {p:6.1f}%  {v:.4g}")
print(f"\nratio  max / median  of scaled-label-max = "
      f"{smax.max()/np.median(smax):.4g}   (huge => outlier-driven val loss)")