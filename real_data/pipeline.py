"""
pipeline.py -- end-to-end "test the trained scatter model on a DICOM" for
one real-data run (option A: scatter-free forward projection).

Steps, each isolated in a function so you can drop into a REPL and inspect
intermediate arrays:

  1. load_activity(dicom_path)          -> Volume (Bq/mL)
  2. build_inputs(vol, cfg, mu_mode)    -> (activity_zyx, mu_per_mm_zyx)
  3. synthesize_prompt(A, activity, mu, target_counts, seed)
                                        -> (y_prompts_ordered, af, y_clean)
  4. predict_scatter_ordered(y_prompts, cfg, ckpt) -> s_hat_ordered
  5. reconstruct_pair(A, y, af, s_hat, n_iter)     -> (x_floor, x_model, c)
  6. save_outputs(...)                             -> npy + PNG files

GPU is expected (parallelproj + cupy). On CPU the projector still works but
each MLEM iteration takes minutes; use small `n_iter` for a sanity run.

Option B (add SSS to the prompt) is a two-line change in synthesize_prompt,
gated behind an argument. See `synthesize_prompt(add_sss=True)` for how to
wire it once the pilot A run has been reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import sys

import numpy as np

# The heavy deps (torch, parallelproj, mcgpu_recon) are imported lazily inside
# the functions that need them so a bare `python -c "import pipeline"` on a
# machine without CUDA still succeeds and lets you inspect signatures.


# ---------------- 1. load activity --------------------------------------

def load_activity(dicom_path: str | Path):
    """Return a dicom_loader.Volume in Bq/mL for a PET DICOM."""
    from dicom_loader import read_volume, describe
    vol = read_volume(dicom_path)
    if vol.metadata.get("modality") != "PT":
        raise ValueError(f"expected modality PT, got "
                         f"{vol.metadata.get('modality')!r} for {dicom_path}")
    print(describe(vol))
    return vol


# ---------------- 2. build inputs on the trained grid --------------------

def build_inputs(vol, cfg, *, mu_mode: str = "body_mask",
                 verbose: bool = True):
    """Resample activity to trained grid; build mu-map on same grid.

    Returns
    -------
    activity_zyx : (Nz, Ny, Nx) float32 in Bq/mL
    mu_per_mm    : (Nz, Ny, Nx) float32 in 1/mm at 511 keV
    """
    from resample import target_grid_from_config, resample_to_grid
    from mu_map import build_mu_map
    tg = target_grid_from_config(cfg)
    activity = resample_to_grid(vol, tg, order=1, verbose=verbose)
    mu = build_mu_map(activity, mode=mu_mode, verbose=verbose)
    return activity.astype(np.float32), mu.astype(np.float32)


# ---------------- 3. synthesize a pseudo-prompt sinogram ------------------

def synthesize_prompt(activity_zyx, mu_per_mm, cfg, *,
                      target_counts: float = 5e8, seed: int = 0,
                      add_sss: bool = False, sss_scale: Optional[float] = None,
                      xp_name: str = "auto", verbose: bool = True):
    """Forward-project resampled activity through the trained geometry to a
    prompt sinogram in ordered ring-pair layout (P, A, R).

    Option A (default): y = Poisson(alpha * af * A@x). No scatter added; the
    model sees a scatter-free input. The reported CNR delta is the model's
    behaviour on such an input.

    Option B (add_sss=True): after building y_true = alpha * af * A@x, add a
    scatter contribution `sss_scale * SSS(x, mu)` before drawing Poisson noise.
    Requires mcgpu_sss to be importable; kept opt-in so the A-only path has
    no SSS dependency.

    Returns
    -------
    y_prompts : (P, A, R) numpy float32 (Poisson counts in ordered order)
    af        : (P, A, R) numpy float32 (per-bin attenuation factors) so MLEM
                 can be called with mult=af without recomputing
    y_clean   : (P, A, R) numpy float32 (noise-free expected prompts) -- kept
                 for diagnostics
    A_proj    : the MCGPUProjector built for cfg (returned so the same one
                 feeds MLEM and predict; building it is O(seconds) so we
                 avoid duplicating)
    """
    xp, using_gpu = _select_xp(xp_name)
    if verbose:
        print(f"[synthesize] using array namespace: {xp.__name__} "
              f"(GPU={using_gpu})")

    # ---- projector, restricted to ordered ring-pair layout ----
    from mcgpu_pet_wrapper.config import plane_ring_pairs
    from mcgpu_recon import MCGPUProjector, attenuation_factors

    pairs = plane_ring_pairs(cfg)
    filled = [i for i, p in enumerate(pairs) if p]
    ring1 = np.asarray([pairs[i][0][0] for i in filled], dtype=np.int64)
    ring2 = np.asarray([pairs[i][0][1] for i in filled], dtype=np.int64)
    A_proj = MCGPUProjector(cfg, ring1, ring2, xp=xp)

    # ---- forward project activity + attenuation ----
    act = xp.asarray(activity_zyx, dtype=xp.float32)
    mu  = xp.asarray(mu_per_mm,   dtype=xp.float32)
    af  = attenuation_factors(A_proj, mu)                 # (P, A, R), xp
    line_int = A_proj(act)                                # (P, A, R), xp, in Bq/mL * mm
    y_trues = line_int * af                               # unit-agnostic "expected trues" up to scale

    if add_sss:
        s = _sss_scatter(cfg, act, mu, ring1, ring2, xp=xp, verbose=verbose)
        if sss_scale is not None:
            s = s * float(sss_scale)
        y_expected = y_trues + s
    else:
        y_expected = y_trues

    # ---- scale to target_counts and draw Poisson ----
    total = float(y_expected.sum())
    if total <= 0:
        raise RuntimeError(f"forward projection produced non-positive total "
                           f"({total}); check the resampled activity")
    alpha = float(target_counts) / total
    y_clean = y_expected * alpha
    # Poisson always on CPU (numpy) -- cupy also has it but this keeps the seeded
    # generator behaviour identical across the two backends.
    rng = np.random.default_rng(seed)
    y_clean_np = _to_numpy(y_clean)
    y_prompts_np = rng.poisson(y_clean_np).astype(np.float32)

    if verbose:
        print(f"[synthesize] scaled by alpha={alpha:.4g} so sum(y_clean)="
              f"{y_clean_np.sum():.3g} ~ target_counts={target_counts:.3g}")
        print(f"             sum(y_prompts) = {y_prompts_np.sum():.3g}")

    # keep af/y_clean in xp so downstream MLEM avoids a copy
    return y_prompts_np, af, y_clean, A_proj


# ---------------- 4. predict scatter with the trained model --------------

def predict_scatter_ordered(y_prompts, cfg, ckpt_path, verbose: bool = True):
    """Run the pilot model on the pseudo-prompt; return scatter in ordered
    plane order ready for mlem(..., contamination=...)."""
    # scatter_ml sits at the sibling repo root; keep the import local so a
    # partial-repo checkout without torch still lets the rest import.
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scatter_ml.predict import load_model, predict_scatter_from_prompts

    if verbose:
        print(f"[predict] loading model checkpoint: {ckpt_path}")
    model, device, a = load_model(ckpt_path)
    if verbose:
        print(f"          device={device}  window_k={a.get('window_k')}  "
              f"base={a.get('base')}  loss={a.get('loss')}  split={a.get('split')}")

    s_hat = predict_scatter_from_prompts(
        y_prompts, cfg, model,
        window_k=a.get("window_k", 5), device=device,
    )
    if verbose:
        print(f"[predict] s_hat shape={s_hat.shape} sum={s_hat.sum():.3g} "
              f"(vs prompts sum={y_prompts.sum():.3g})")
    return s_hat


# ---------------- 5. reconstruct twice, scale-match ----------------------

def reconstruct_pair(A_proj, y_prompts, af, s_hat, *,
                     n_iter: int = 20, sens_floor_frac: float = 0.025,
                     verbose: bool = True):
    """MLEM twice with identical settings: floor (no scatter correction)
    and model (contamination=s_hat). Returns numpy arrays.
    """
    from mcgpu_recon import mlem, scale_match
    xp = A_proj.xp
    y  = xp.asarray(y_prompts, dtype=xp.float32)
    af = xp.asarray(af,       dtype=xp.float32)
    s  = xp.asarray(s_hat,    dtype=xp.float32)

    if verbose: print(f"[recon] floor  MLEM ({n_iter} iter, no contamination)")
    x_floor = mlem(A_proj, y, n_iter=n_iter, mult=af,
                   sens_floor_frac=sens_floor_frac, verbose=verbose)
    if verbose: print(f"[recon] model  MLEM ({n_iter} iter, contamination=s_hat)")
    x_model = mlem(A_proj, y, n_iter=n_iter, mult=af, contamination=s,
                   sens_floor_frac=sens_floor_frac, verbose=verbose)

    # scale-match model to floor so a subsequent difference is not confused by
    # a global constant (MLEM has an unfitted overall gain; see mcgpu_recon
    # docstring on scale_match).
    x_model_matched, c = scale_match(x_floor, x_model)
    if verbose: print(f"[recon] scale_match(model -> floor) c={c:.4g}")

    return _to_numpy(x_floor), _to_numpy(x_model_matched), float(c)


# ---------------- 6. persistence & one-shot driver -----------------------

def save_outputs(out_dir, *, activity, mu, y_prompts, af, s_hat, x_floor,
                 x_model, cfg, meta: dict):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "activity_resampled.npy", activity)
    np.save(out_dir / "mu_per_mm.npy",           mu)
    np.save(out_dir / "y_prompts_ordered.npy",   y_prompts)
    np.save(out_dir / "attenuation_factors.npy", _to_numpy(af))
    np.save(out_dir / "s_hat_ordered.npy",       s_hat)
    np.save(out_dir / "x_floor.npy",             x_floor)
    np.save(out_dir / "x_model.npy",             x_model)
    (out_dir / "meta.txt").write_text("\n".join(f"{k}: {v}"
                                                 for k, v in meta.items()))
    import json
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[save] wrote all arrays + meta to {out_dir}")


def run_one(dicom_path, ckpt_path, out_dir, cfg=None, *,
            mu_mode: str = "body_mask", target_counts: float = 5e8,
            n_iter: int = 20, seed: int = 0, add_sss: bool = False):
    """The whole option-A pipeline for one PET DICOM. Call from a driver.

    Returns a dict of the primary arrays for interactive inspection.
    """
    if cfg is None:
        import mcgpu_pet_wrapper as mpw
        cfg = mpw.default_config()

    vol = load_activity(dicom_path)
    activity, mu = build_inputs(vol, cfg, mu_mode=mu_mode)
    y_prompts, af, y_clean, A_proj = synthesize_prompt(
        activity, mu, cfg,
        target_counts=target_counts, seed=seed, add_sss=add_sss,
    )
    s_hat = predict_scatter_ordered(y_prompts, cfg, ckpt_path)
    x_floor, x_model, scale_c = reconstruct_pair(
        A_proj, y_prompts, af, s_hat, n_iter=n_iter,
    )
    meta = dict(dicom_path=str(dicom_path), ckpt_path=str(ckpt_path),
                mu_mode=mu_mode, target_counts=target_counts, seed=seed,
                add_sss=add_sss, n_iter=n_iter, scale_c=scale_c)
    save_outputs(out_dir, activity=activity, mu=mu, y_prompts=y_prompts,
                 af=af, s_hat=s_hat, x_floor=x_floor, x_model=x_model,
                 cfg=cfg, meta=meta)
    return dict(activity=activity, mu=mu, y_prompts=y_prompts, af=_to_numpy(af),
                s_hat=s_hat, x_floor=x_floor, x_model=x_model, scale_c=scale_c)


# ---------------------- xp / device helpers ------------------------------

def _select_xp(xp_name: str):
    """Return (xp module, using_gpu). 'auto' prefers cupy if available."""
    if xp_name == "auto":
        try:
            import array_api_compat.cupy as xp
            return xp, True
        except Exception:
            import numpy as xp
            return xp, False
    if xp_name in ("cupy", "gpu"):
        import array_api_compat.cupy as xp
        return xp, True
    if xp_name in ("numpy", "cpu"):
        import numpy as xp
        return xp, False
    raise ValueError(f"unknown xp_name={xp_name!r}")


def _to_numpy(a):
    """cupy or numpy -> numpy."""
    if hasattr(a, "get"):    # cupy ndarray
        return a.get()
    return np.asarray(a)


def _sss_scatter(cfg, act_xp, mu_per_mm_xp, ring1, ring2, xp, verbose):
    """Option-B helper: build an SSS estimate for the resampled activity.

    Left as a thin adapter around mcgpu_sss.sss_estimate. mcgpu_sss wants a
    VoxelGrid so it can pull the material files; we synthesize a minimal
    'water' VoxelGrid whose density map == mu_per_mm / mu_water. Deferred
    until option A is reported.
    """
    raise NotImplementedError(
        "add_sss=True (option B) is deferred; leaving as a hook until the "
        "option-A pilot has been reviewed. See real_data/pipeline.py notes.")
