"""
eval_harness.py -- the "measuring stick" for scatter correction.

For one run it reconstructs the SAME measured sinogram (prompts = trues+scatter)
several ways, differing ONLY in the scatter estimate fed to MLEM's contamination
term, then scores each reconstruction against the known painted activity map.

The two reference arms make any score interpretable:
  - floor  : contamination = None            (no scatter correction; worst case)
  - oracle : contamination = TRUE scatter    (exact scatter; best any method can do)
A model's score is then read as a FRACTION of the floor->oracle gap it closed.
Add arms (a traditional estimate, a model prediction) by passing more entries in
`scatter_arms`; nothing else changes.

Assumptions (all confirmed for this dataset):
  - span == 1  (read_sinogram_ring_pairs is invertible).
  - ground truth = the painted .vox activity (read_vox), same (Nz,Ny,Nx) grid as
    the reconstruction, so they align voxel-for-voxel.
  - per-object regions come from recipe.json (exact shapes/positions).

Metrics (why these three): scatter adds a smooth positive background, which
inflates low-activity regions and washes out contrast. So we measure
  1. contrast recovery (CRC)      -- scale-free; the quantity SC exists to fix.
  2. background non-uniformity CoV -- scale-free; over/under-subtraction shows here.
  3. ROI activity bias            -- needs one global scale (MLEM is correct only
                                     up to a constant); we fit it by least squares
                                     over the scored regions, the same way for
                                     every arm, so comparisons stay fair.

ROI scheme (a defensible STARTING POINT, documented so it can be revised):
  - each object is voxelized to a "core" (shape shrunk linearly by CORE_SHRINK) to
    drop partial-volume edge voxels; ownership follows paint order (largest first,
    smaller overwrite), matching the simulation.
  - the largest-volume object is the background/reference region.
  - objects whose TRUE mean activity exceeds HOT_RATIO x the reference mean are
    "hot"; CRC is measured for each hot object against the reference.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import mcgpu_pet_wrapper as mpw
from mcgpu_pet_wrapper.config import voxel_space_shape_zyx, grid_size_mm

# reconstruction lives in mcgpu-recon
from mcgpu_recon import from_run, mlem

# ---- array backend -------------------------------------------------------
# Projection is the heavy, embarrassingly-parallel work -> run it on the GPU.
# Metrics (masks, means, least-squares) are light + irregular -> run in numpy on
# the host. The two meet at exactly one hand-off: _to_numpy(), right after mlem.
#
# GPU (recommended -- everything stays on the device):
import array_api_compat.cupy as xp
XP_KW = dict(xp=xp, plane_chunk=256)
# CPU fallback (no cupy/CUDA): comment the two lines above and uncomment these.
# import numpy as xp
# XP_KW = dict(xp=xp, plane_chunk=256, num_chunks=8)


def _to_numpy(a):
    """Move a reconstruction to a host numpy array (identity if already numpy)."""
    if xp is np:
        return np.asarray(a)
    return xp.asnumpy(a) if hasattr(xp, "asnumpy") else np.asarray(a.get())


CORE_SHRINK = 0.6   # linear shrink of each object to its edge-free core
HOT_RATIO = 1.2     # object is "hot" if true mean > HOT_RATIO * reference mean


# ---------------------------------------------------------------------------
# 1. reconstruction arms
# ---------------------------------------------------------------------------
def _load_input(run_dir, cfg):
    """Return (prompts, true_scatter, A) as ON-DEVICE arrays in one plane order.

    prompts = trues + scatter is the measured sinogram every arm reconstructs;
    true_scatter is the oracle's contamination. Both from_run calls share the
    same read_sinogram_ring_pairs plane order, and A is built from the trues call
    so its bins match by construction. XP_KW puts A and the sinograms on the GPU.
    """
    y_t, A = from_run(run_dir, cfg, **XP_KW)                 # trues + projector
    y_s, _ = from_run(run_dir, cfg, scatter=True, **XP_KW)   # scatter, same order
    prompts = xp.asarray(y_t + y_s)
    return prompts, xp.asarray(y_s), A


def reconstruct_arms(run_dir, cfg, extra_arms=None, n_iter=20):
    """Reconstruct the floor and oracle arms (plus any extra scatter estimates).

    extra_arms : optional dict name -> scatter_sinogram (same shape as prompts),
        e.g. {"model": predicted_scatter, "conv": conv_subtraction_estimate}.
    Returns dict name -> reconstructed image (Nz,Ny,Nx) float32.
    """
    prompts, true_scatter, A = _load_input(run_dir, cfg)
    arms = {"floor": None, "oracle": true_scatter}
    if extra_arms:
        # a passed-in estimate (e.g. a model prediction) may be a host array;
        # move it onto the same device as the projector before use.
        arms.update({k: (None if v is None else A.xp.asarray(v))
                     for k, v in extra_arms.items()})
    recons = {}
    for name, contam in arms.items():
        x = mlem(A, prompts, n_iter=n_iter, contamination=contam, verbose=True)
        recons[name] = _to_numpy(x)          # single GPU->host hand-off
    return recons


# ---------------------------------------------------------------------------
# 2. ground truth + per-object regions
# ---------------------------------------------------------------------------
def _grid(cfg):
    nz, ny, nx = voxel_space_shape_zyx(cfg)
    dx, dy, dz = grid_size_mm(cfg)
    # voxel-frame centers (first octant), matching the builder/.vox convention
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dy
    z = (np.arange(nz) + 0.5) * dz
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    return (nz, ny, nx), (X, Y, Z)


def _ellipsoid_mask(X, Y, Z, c, semi, shrink):
    cx, cy, cz = c
    ax, ay, az = (s * shrink for s in semi)
    return ((X - cx) / ax) ** 2 + ((Y - cy) / ay) ** 2 + ((Z - cz) / az) ** 2 <= 1.0


def _cylinder_mask(X, Y, Z, c, rx, ry, height, axis, theta_deg, shrink):
    cx, cy, cz = c
    rx, ry, half = rx * shrink, ry * shrink, (height / 2.0) * shrink
    t = math.radians(theta_deg or 0.0)
    ct, st = math.cos(t), math.sin(t)
    if axis == "z":
        u, v, w = X - cx, Y - cy, Z - cz
    elif axis == "x":
        u, v, w = Y - cy, Z - cz, X - cx
    else:  # y
        u, v, w = X - cx, Z - cz, Y - cy
    ur = u * ct + v * st
    vr = -u * st + v * ct
    return ((ur / rx) ** 2 + (vr / ry) ** 2 <= 1.0) & (np.abs(w) <= half)


def object_labels(run_dir, cfg, shrink=CORE_SHRINK):
    """Label volume (Nz,Ny,Nx): 0 = background, k = object k's core.

    Objects painted largest-to-smallest so smaller overwrite (matches the
    simulation's paint order and thus the .vox). Returns (labels, volumes) where
    volumes[k] is object k's approximate volume (for picking the reference).
    """
    recipe = json.loads((Path(run_dir) / "recipe.json").read_text())
    insts = list(recipe["instructions"])
    insts.sort(key=lambda i: i.get("approx_volume_mm3", 0.0), reverse=True)

    shape, (X, Y, Z) = _grid(cfg)
    labels = np.zeros(shape, dtype=np.int32)
    volumes = {}
    for k, inst in enumerate(insts, start=1):
        c = tuple(inst["center_mm"])
        if inst["kind"] == "ellipsoid":
            m = _ellipsoid_mask(X, Y, Z, c, inst["semi_axes_mm"], shrink)
        else:
            m = _cylinder_mask(X, Y, Z, c, inst["rx_mm"], inst["ry_mm"],
                               inst["height_mm"], inst["axis"],
                               inst.get("theta_deg", 0.0), shrink)
        labels[m] = k
        volumes[k] = inst.get("approx_volume_mm3", int(m.sum()))
    return labels, volumes


# ---------------------------------------------------------------------------
# 3. metrics
# ---------------------------------------------------------------------------
def _global_scale(recon, truth, mask):
    """Least-squares c minimizing ||truth - c*recon|| over mask (MLEM is correct
    only up to a global constant; fit the same way for every arm)."""
    r = recon[mask].astype(np.float64)
    t = truth[mask].astype(np.float64)
    denom = float((r * r).sum())
    return float((r * t).sum() / denom) if denom > 0 else 0.0


def score(recon, truth, labels, volumes):
    """Score one reconstruction. Returns a flat dict of metrics."""
    if not volumes:
        return {"note": "no objects"}

    ref = max(volumes, key=volumes.get)          # largest object = reference/bg
    ref_mask = labels == ref
    scored_mask = labels > 0                      # all cores, for the scale fit
    c = _global_scale(recon, truth, scored_mask)
    xr = c * recon

    true_ref = float(truth[ref_mask].mean())
    rec_ref = float(xr[ref_mask].mean())

    # hot objects: true mean well above the reference
    hot = [k for k in volumes
           if k != ref and float(truth[labels == k].mean()) > HOT_RATIO * true_ref]

    crc = []
    for k in hot:
        m = labels == k
        true_ratio = float(truth[m].mean()) / true_ref
        rec_ratio = float(xr[m].mean()) / rec_ref if rec_ref > 0 else 0.0
        if true_ratio > 0:
            crc.append(rec_ratio / true_ratio)    # 1.0 = perfect; scale-free

    # ROI bias over every core (post global scale)
    biases = []
    for k in volumes:
        m = labels == k
        tt = float(truth[m].mean())
        if tt > 0:
            biases.append((float(xr[m].mean()) - tt) / tt)

    bg = xr[ref_mask]
    return {
        "scale_c": c,
        "n_hot": len(hot),
        "crc_mean": float(np.mean(crc)) if crc else float("nan"),
        "abs_bias_mean": float(np.mean(np.abs(biases))) if biases else float("nan"),
        "bg_cov": float(bg.std() / bg.mean()) if bg.mean() > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# 4. top-level: evaluate one run across all arms
# ---------------------------------------------------------------------------
def evaluate(run_dir, cfg=None, extra_arms=None, n_iter=20):
    """Reconstruct floor + oracle (+ extras) for one run and score each.

    Returns dict arm_name -> metrics dict. The floor/oracle pair brackets the
    achievable range; a later arm's 'fraction of gap closed' on any metric is
    (floor - arm) / (floor - oracle).
    """
    run_dir = Path(run_dir)
    if cfg is None:
        cfg = mpw.load_config(run_dir / "config.json")
    truth = mpw.read_vox(run_dir, cfg).activity          # (Nz,Ny,Nx) Bq/voxel
    labels, volumes = object_labels(run_dir, cfg)
    recons = reconstruct_arms(run_dir, cfg, extra_arms=extra_arms, n_iter=n_iter)
    return {name: score(x, truth, labels, volumes) for name, x in recons.items()}


def gap_closed(results, metric="crc_mean"):
    """Fraction of the floor->oracle gap an arm closed on `metric`.
    1.0 = matched the oracle; 0.0 = no better than the floor. For CRC the target
    is 1.0 (perfect recovery), so we score closeness to the oracle's value."""
    floor = results["floor"][metric]
    oracle = results["oracle"][metric]
    span = oracle - floor
    out = {}
    for name, m in results.items():
        if name in ("floor", "oracle"):
            continue
        out[name] = (m[metric] - floor) / span if span != 0 else float("nan")
    return out


if __name__ == "__main__":
    import sys

    run_dirs = sys.argv[1:] or ["../MCGPU_data/runs/run_00009"]
    for rd in run_dirs:
        print(f"\n=== {rd} ===")
        res = evaluate(rd)
        for arm, m in res.items():
            print(f"  {arm:8s} "
                  f"CRC={m.get('crc_mean', float('nan')):.3f}  "
                  f"|bias|={m.get('abs_bias_mean', float('nan')):.3f}  "
                  f"bg_CoV={m.get('bg_cov', float('nan')):.3f}  "
                  f"(c={m.get('scale_c', float('nan')):.3g}, "
                  f"n_hot={m.get('n_hot', 0)})")