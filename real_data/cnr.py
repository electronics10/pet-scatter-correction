"""
cnr.py -- draw ROIs and compute CNR on the reconstructed images.

CNR here follows the standard scatter-correction definition:

    CNR = (mean_hot - mean_bg) / std_bg

with:
    hot_roi  -- a small volume inside the uniform hot region
    bg_roi   -- a small volume inside the cold water insert (or air insert)

Report CNR for BOTH reconstructions (floor = no scatter correction; model =
network scatter subtracted). The interesting quantity for the supervisor is
`delta = CNR_model - CNR_floor`. A positive delta means the network is doing
something useful; a null or negative delta is also a valid report ("the model
does not help / hurts on this input").

Two ROI-selection modes:

  (1) explicit_boxes: caller supplies (k_slice, j0, j1, i0, i1) rectangles.
      Deterministic and reviewable. This is what the driver script uses; you
      pick the boxes by looking at the resampled activity (or a saved axial
      montage) and hard-code them.

  (2) auto_iq: heuristic ROI picking for the specific stacked-phantom
      layout found in 20260316/IQ_Phantom (a large uniform slab at low z and
      a NEMA-IQ-like circular phantom with hot rods + two cold inserts at
      high z). Uses connected-component analysis to find the largest dim
      region inside the circular phantom (=cold water insert) and takes a
      hot ROI in the middle of the uniform hot region. Reviewable by
      inspecting the returned masks. NOT robust for other datasets.

Both modes return `RoiSpec` objects so the same CNR routine consumes either.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import numpy as np


@dataclass
class RoiSpec:
    """A named 3D binary mask on the (Nz, Ny, Nx) reconstruction grid."""
    name: str
    mask: np.ndarray            # bool, shape (Nz, Ny, Nx)

    def stats(self, image: np.ndarray) -> dict:
        vals = image[self.mask]
        return dict(name=self.name,
                    n_voxels=int(self.mask.sum()),
                    mean=float(vals.mean()) if vals.size else float("nan"),
                    std=float(vals.std()) if vals.size else float("nan"),
                    min=float(vals.min()) if vals.size else float("nan"),
                    max=float(vals.max()) if vals.size else float("nan"))


def roi_from_box(name: str, shape_zyx, k_range, j_range, i_range) -> RoiSpec:
    """Rectangular ROI, half-open interval convention: k_range=(k0,k1) means
    voxels k0..k1-1 inclusive."""
    Nz, Ny, Nx = shape_zyx
    m = np.zeros(shape_zyx, dtype=bool)
    m[k_range[0]:k_range[1], j_range[0]:j_range[1], i_range[0]:i_range[1]] = True
    if not m.any():
        raise ValueError(f"ROI {name!r} is empty; ranges {k_range}, {j_range}, "
                         f"{i_range} vs shape {shape_zyx}")
    return RoiSpec(name=name, mask=m)


def cnr(image: np.ndarray, hot: RoiSpec, bg: RoiSpec) -> dict:
    """Compute CNR = (mean_hot - mean_bg) / std_bg on `image`. Returns both
    ROI stats and the CNR so the caller sees the ingredients, not just the
    scalar output."""
    hs = hot.stats(image)
    bs = bg.stats(image)
    if bs["std"] == 0:
        cnr_val = float("nan")
    else:
        cnr_val = (hs["mean"] - bs["mean"]) / bs["std"]
    return dict(hot=hs, bg=bs, cnr=cnr_val)


def compare_cnr(x_floor: np.ndarray, x_model: np.ndarray,
                hot: RoiSpec, bg: RoiSpec) -> dict:
    """CNR for both arms; return a printable summary dict.

    Note: x_floor and x_model should already be scale-matched
    (pipeline.reconstruct_pair does this) so mean values are comparable.
    Contrast and noise scale together, so CNR is scale-invariant per-arm --
    the scale-match is only needed if you also want to look at absolute means.
    """
    f = cnr(x_floor, hot, bg)
    m = cnr(x_model, hot, bg)
    return dict(
        floor=f,
        model=m,
        delta_cnr=m["cnr"] - f["cnr"],
        rel_change=(m["cnr"] - f["cnr"]) / f["cnr"] if f["cnr"] != 0 else float("nan"),
    )


def print_cnr_report(summary: dict) -> None:
    """Pretty-print what compare_cnr returns."""
    def _fmt(s):
        return (f"mean={s['mean']:.4g}  std={s['std']:.4g}  "
                f"n={s['n_voxels']}  [{s['min']:.4g}, {s['max']:.4g}]")
    print(f"  hot ROI ({summary['floor']['hot']['name']}):")
    print(f"     floor: {_fmt(summary['floor']['hot'])}")
    print(f"     model: {_fmt(summary['model']['hot'])}")
    print(f"  bg  ROI ({summary['floor']['bg']['name']}):")
    print(f"     floor: {_fmt(summary['floor']['bg'])}")
    print(f"     model: {_fmt(summary['model']['bg'])}")
    print(f"  CNR floor = {summary['floor']['cnr']:.4g}")
    print(f"  CNR model = {summary['model']['cnr']:.4g}")
    print(f"  delta     = {summary['delta_cnr']:+.4g}  "
          f"({summary['rel_change']*100:+.2f}% relative)")


# ------------------- IQ-phantom-specific auto-ROIs (heuristic) -------------

def auto_iq_rois(activity: np.ndarray, *,
                 hot_slab_k=(85, 95), cold_slab_k=(98, 105),
                 hot_radius_vox: int = 3, cold_radius_vox: int = 2,
                 verbose: bool = True) -> tuple:
    """Heuristic ROIs for the stacked IQ phantom in 20260316/IQ_Phantom.

    Two assumptions specific to that dataset (see the axial montage in
    iq_resample_mu_check.png):
      * Uniform hot region sits around k in `hot_slab_k`. Take a small
        cubic ROI near the centroid of that slab, avoiding rods/inserts.
      * Cold water insert sits around k in `cold_slab_k`. It appears as a
        dim disk inside the bright ring. Threshold the slab at half-max,
        take the darkest connected component with area > 5 voxels and pick
        its centroid as the ROI center.

    Numbers above assume the trained grid (Nz=149, dz=1mm, phantom placed
    near scanner origin) -- edit the k-ranges if you use a different dataset.
    """
    Nz, Ny, Nx = activity.shape
    assert 0 <= hot_slab_k[0] < hot_slab_k[1] <= Nz
    assert 0 <= cold_slab_k[0] < cold_slab_k[1] <= Nz

    # ---- hot ROI: center-of-mass of the hot slab, take a small cube ----
    hot_slab = activity[hot_slab_k[0]:hot_slab_k[1]]
    # weight by intensity to find where the hot ring is; centroid of the top
    # 30% intensity voxels is a good stand-in for "middle of hot ring".
    thr = np.quantile(hot_slab[hot_slab > 0], 0.7) if (hot_slab > 0).any() else 0
    hot_bin = hot_slab > thr
    if not hot_bin.any():
        raise RuntimeError("auto_iq_rois: could not find hot voxels in slab")
    cz, cj, ci = _centroid(hot_bin)
    cz += hot_slab_k[0]                        # back to global k
    hot_roi = roi_from_box(
        "hot_center",
        activity.shape,
        (int(cz) - 1, int(cz) + 2),
        (int(cj) - hot_radius_vox, int(cj) + hot_radius_vox + 1),
        (int(ci) - hot_radius_vox, int(ci) + hot_radius_vox + 1),
    )

    # ---- cold ROI: holes inside the bright body of the cold slab ----
    # A cold insert is a dim region inside a bright ring. Detect it by
    # filling holes in the body mask and subtracting: (filled - body) are
    # the enclosed dim regions.
    from scipy.ndimage import binary_fill_holes
    cold_slab = activity[cold_slab_k[0]:cold_slab_k[1]]
    body_thr = (np.quantile(cold_slab[cold_slab > 0], 0.3)
                if (cold_slab > 0).any() else 0)
    body = cold_slab > body_thr
    # fill holes slice-by-slice so we detect the ring shape per axial slab
    filled = np.stack([binary_fill_holes(body[k]) for k in range(body.shape[0])])
    dark = filled & (~body)
    if not dark.any():
        raise RuntimeError("auto_iq_rois: could not find cold insert; "
                           "supply an explicit box via roi_from_box instead")
    cz, cj, ci = _centroid(dark)
    cz += cold_slab_k[0]
    cold_roi = roi_from_box(
        "cold_water",
        activity.shape,
        (int(cz) - 1, int(cz) + 2),
        (int(cj) - cold_radius_vox, int(cj) + cold_radius_vox + 1),
        (int(ci) - cold_radius_vox, int(ci) + cold_radius_vox + 1),
    )

    if verbose:
        print(f"[auto_iq_rois] hot  ROI centered at (k,j,i)="
              f"({int(cz - cold_slab_k[0]) + hot_slab_k[0] - 2}, "
              f"{hot_roi.mask.any(axis=(0,2)).nonzero()[0][0]}, "
              f"{hot_roi.mask.any(axis=(0,1)).nonzero()[0][0]})")
        print(f"              cold ROI centered at (k,j,i)="
              f"({int(cz)}, "
              f"{cold_roi.mask.any(axis=(0,2)).nonzero()[0][0]}, "
              f"{cold_roi.mask.any(axis=(0,1)).nonzero()[0][0]})")
    return hot_roi, cold_roi


def _centroid(mask: np.ndarray) -> tuple:
    """Integer centroid of a boolean mask."""
    idx = np.argwhere(mask)
    return tuple(idx.mean(axis=0))
