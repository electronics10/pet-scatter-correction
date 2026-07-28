"""
mu_map.py -- build a linear attenuation map at 511 keV on the trained grid.

The map goes straight into mcgpu_recon.attenuation_factors(A, mu_per_mm),
which returns exp(-integral of mu along each LOR) -- the per-bin factors MLEM
uses in its `mult` argument.

Three modes; only the first two are implemented in this pilot:

  "body_mask"  (default; deployable without external inputs)
      Threshold the resampled PET activity to a body-vs-air mask, morph-clean
      it, fill body with water's mu at 511 keV (0.0096 /mm), leave air = 0.
      Correct to ~1% for soft tissue since Compton dominates at 511 keV and
      mu(511 keV) is nearly proportional to electron density = rho * (Z/A) * N_A,
      which for water and typical soft tissue tracks to a few percent (see
      mcgpu-pet-wrapper README §2.3, "materials"). For a PMMA-walled phantom
      like the NEMA IQ, water is a close enough surrogate for the walls too
      (PMMA rho = 1.19 gives mu ~= 0.0114, but the walls are ~2 mm thick and
      the resulting AF error is ~2%).

  "none"       (unity-attenuation fallback)
      mu = 0 everywhere -> attenuation_factors returns 1 everywhere. Both
      reconstructions in the pilot then share the SAME af=1, so the with-vs-
      without-scatter CNR comparison remains internally consistent -- the
      images will show mild attenuation cupping in both, but the delta the
      supervisor cares about (CNR change) is unaffected. Use this as a safety
      net if the body mask misbehaves.

  "mri"        (placeholder; not implemented)
      Would require a co-registered MR volume placed on the trained grid,
      thresholded to a body mask. Deferred per project decision -- the two
      DICOM MRs inspected had unusable IPP, and the phantom pilot doesn't
      need MR-derived accuracy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening


MU_WATER_PER_MM_AT_511 = 0.0096   # linear attenuation coeff of water at 511 keV, 1/mm


def _body_mask_from_pet(pet_resampled: np.ndarray,
                        threshold_frac: float = 0.02,
                        close_iter: int = 2,
                        open_iter: int = 1) -> np.ndarray:
    """Simple threshold + morphology on a resampled PET activity volume.

    Steps:
      1. Threshold at `threshold_frac` * max intensity (defaults to 2%).
      2. Binary closing to bridge small activity dropouts inside the body
         (e.g. cold inserts, low-activity walls).
      3. Fill holes in the resulting mask (2D per axial slab is enough; we
         use 3D fill_holes for simplicity).
      4. Binary opening to shave off single-voxel specks outside the body.

    This is deliberately dumb -- no Otsu, no per-slice logic. For a
    reasonably filled phantom or animal it recovers the outer boundary
    within a couple of voxels. If the input has extreme dynamic range
    (e.g. very hot rod + very dim background), the 2% threshold may cut
    the low-activity background; the caller can lower it.
    """
    if pet_resampled.max() <= 0:
        return np.zeros_like(pet_resampled, dtype=bool)
    mask = pet_resampled > (threshold_frac * float(pet_resampled.max()))
    if close_iter > 0:
        mask = binary_closing(mask, iterations=close_iter)
    mask = binary_fill_holes(mask)
    if open_iter > 0:
        mask = binary_opening(mask, iterations=open_iter)
    return mask


def build_mu_map(pet_resampled: np.ndarray, *,
                 mode: str = "body_mask",
                 mu_water_per_mm: float = MU_WATER_PER_MM_AT_511,
                 threshold_frac: float = 0.02,
                 verbose: bool = True) -> np.ndarray:
    """Return a (Nz, Ny, Nx) float32 mu-map in 1/mm on the trained grid.

    Parameters
    ----------
    pet_resampled : (Nz, Ny, Nx) float32
        The PET activity already placed on the trained grid by
        resample.resample_to_grid. Only its shape and (for `body_mask`) its
        intensity distribution are used -- values are not consumed as physics.
    mode : {"body_mask", "none", "mri"}
    mu_water_per_mm : float
        Water's linear attenuation at 511 keV; overridable for experiments.
    threshold_frac : float
        Fractional-of-max threshold for the body mask (body_mask mode only).
    verbose : bool
        Print body-fraction diagnostic.

    Returns
    -------
    mu_per_mm : (Nz, Ny, Nx) float32, ready for attenuation_factors(A, mu).
    """
    if mode == "none":
        if verbose:
            print("[mu_map] mode='none': mu ≡ 0 (attenuation_factors will be 1)")
        return np.zeros(pet_resampled.shape, dtype=np.float32)

    if mode == "mri":
        raise NotImplementedError(
            "mode='mri' is a placeholder. The MRIs in this dataset have "
            "unreliable IPP, so an MR-based mu-map was deferred. Use "
            "mode='body_mask' or mode='none' for the pilot.")

    if mode != "body_mask":
        raise ValueError(f"unknown mode {mode!r}; expected one of "
                         "{'body_mask', 'none', 'mri'}")

    mask = _body_mask_from_pet(pet_resampled, threshold_frac=threshold_frac)
    mu = np.zeros(pet_resampled.shape, dtype=np.float32)
    mu[mask] = float(mu_water_per_mm)
    if verbose:
        frac = mask.mean() * 100.0
        print(f"[mu_map] mode='body_mask' at {threshold_frac*100:.1f}% threshold: "
              f"body occupies {mask.sum()} voxels ({frac:.2f}% of FOV); "
              f"mu_water_per_mm={mu_water_per_mm}")
    return mu
