"""
predict.py -- the recon API: run -> predicted scatter -> reconstruction.

predict_scatter() maps a run to a scatter sinogram in ORDERED (A.out_shape) plane
order, ready to hand to mlem(..., contamination=...). It closes the loop:

    prompts (merged, per-sample scaled)
        -> UNet2p5D  -> scatter in scaled space
        -> * scale   -> scatter in merged count space
        -> unmerge   -> scatter in ordered span=1 planes   (the inverse seam)

Because the ordered order equals read_sinogram_ring_pairs / from_run's A order,
the result plugs directly into the eval harness:

    from eval_harness import evaluate
    from scatter_ml.predict import predict_scatter
    pred = predict_scatter(run_dir, cfg, model)          # numpy (P, A, R)
    res = evaluate(run_dir, cfg, extra_arms={"model": pred})

which reconstructs floor / oracle / model on the same footing and scores them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import mcgpu_pet_wrapper as mpw

from .representation import build_geometry
from .model import UNet2p5D


def load_model(ckpt_path, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    a = ck.get("args", {})
    model = UNet2p5D(window_k=a.get("window_k", 5), base=a.get("base", 32))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, device, a


@torch.no_grad()
def predict_scatter(run_dir, config, model, window_k=5, device=None, eps=1e-6,
                    batch_planes=64):
    """Predict the scatter sinogram for one run.

    Returns ordered scatter (P, A, R) float32 in A.out_shape plane order, i.e.
    exactly what mlem expects as `contamination`. Mirrors dataset.py's per-sample
    scaling and windowing so train and inference are consistent.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    geom = build_geometry(config)

    # merged prompts (the model sees prompts = trues + scatter, as at train time
    # with split=False)
    t, _, _ = mpw.read_sinogram_ring_pairs(run_dir, config, scatter=False)
    s, _, _ = mpw.read_sinogram_ring_pairs(run_dir, config, scatter=True)
    prompts_m = geom.merge(t) + geom.merge(s)           # (M, A, R)

    A, R = prompts_m.shape[1:]
    h = window_k // 2
    pred_m = np.zeros((geom.M, A, R), dtype=np.float32)

    for d, seg in geom.segments.items():
        d_norm = d / max(geom.n_rings - 1, 1)
        # build all windows for this segment, then run in plane-batches
        windows, scales, centers = [], [], []
        for j in range(len(seg)):
            js = np.clip(np.arange(j - h, j + h + 1), 0, len(seg) - 1)
            win = prompts_m[seg[js]]                     # (k, A, R)
            scale = float(win.mean()) + eps
            windows.append(win / scale)
            scales.append(scale)
            centers.append(seg[j])
        windows = np.asarray(windows, dtype=np.float32)  # (n_d, k, A, R)

        for b in range(0, len(windows), batch_planes):
            wb = torch.from_numpy(windows[b:b + batch_planes]).to(device)
            db = torch.full((wb.shape[0],), d_norm, dtype=torch.float32,
                            device=device)
            out = model(wb, db).cpu().numpy()[:, 0]      # (nb, A, R), scaled
            for local, gi in enumerate(range(b, min(b + batch_planes, len(windows)))):
                pred_m[centers[gi]] = out[local] * scales[gi]   # un-scale

    return geom.unmerge(pred_m)                          # ordered (P, A, R)


def predict_scatter_from_ckpt(run_dir, ckpt_path, config=None):
    """Convenience: load config + checkpoint and predict in one call."""
    run_dir = Path(run_dir)
    if config is None:
        config = mpw.load_config(run_dir / "config.json")
    model, device, a = load_model(ckpt_path)
    return predict_scatter(run_dir, config, model,
                           window_k=a.get("window_k", 5), device=device)