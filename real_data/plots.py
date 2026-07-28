"""
plots.py -- diagnostic figures for the real-data pilot.

Two figures the driver script generates:

  roi_overlay(activity, hot_roi, cold_roi, out_path):
      Axial montage of the activity volume through the hot and cold slabs,
      with ROI boxes drawn on top. Sanity-check where auto_iq_rois landed.

  recon_comparison(x_floor, x_model, hot_roi, cold_roi, out_path):
      Side-by-side floor / model reconstructions at the same axial slices,
      plus the difference image and the CNR values as text. This is the
      figure you show your supervisor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _roi_extent(roi):
    """Return (k_center, (j0, j1), (i0, i1)) of a rectangular ROI mask."""
    k_any = roi.mask.any(axis=(1, 2))
    j_any = roi.mask.any(axis=(0, 2))
    i_any = roi.mask.any(axis=(0, 1))
    k_idx = np.flatnonzero(k_any)
    j_idx = np.flatnonzero(j_any)
    i_idx = np.flatnonzero(i_any)
    k_ctr = (k_idx[0] + k_idx[-1]) // 2
    return k_ctr, (j_idx[0], j_idx[-1] + 1), (i_idx[0], i_idx[-1] + 1)


def _draw_box(ax, j_range, i_range, color, label=None):
    j0, j1 = j_range
    i0, i1 = i_range
    ax.add_patch(Rectangle((i0 - 0.5, j0 - 0.5), i1 - i0, j1 - j0,
                           fill=False, edgecolor=color, linewidth=2))
    if label:
        ax.text(i1, j0, label, color=color, fontsize=8,
                ha="left", va="bottom")


def roi_overlay(activity: np.ndarray, hot, cold, out_path,
                title: str = "ROI placement"):
    """Save an axial montage with ROI boxes drawn on top. Grayscale + colorbar."""
    k_hot, jh, ih = _roi_extent(hot)
    k_cold, jc, ic = _roi_extent(cold)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    im0 = axes[0].imshow(activity[k_hot], origin="lower", cmap="gray")
    axes[0].set_title(f"axial k={k_hot} (hot slab)")
    _draw_box(axes[0], jh, ih, "cyan", "hot")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(activity[k_cold], origin="lower", cmap="gray")
    axes[1].set_title(f"axial k={k_cold} (cold slab)")
    _draw_box(axes[1], jc, ic, "cyan", "cold")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def recon_comparison(x_floor: np.ndarray, x_model: np.ndarray,
                     hot, cold, out_path,
                     cnr_floor: float = None, cnr_model: float = None,
                     title: str = "Reconstruction: floor vs model"):
    """Side-by-side floor / model reconstructions at hot and cold slabs.

    Bottom row shows the model - floor difference on a symmetric diverging
    scale so over-/under-correction is visible.
    """
    k_hot, jh, ih = _roi_extent(hot)
    k_cold, jc, ic = _roi_extent(cold)

    diff = x_model - x_floor
    vmax_diff = float(np.abs(diff).max()) or 1.0
    vmax_hot = float(max(x_floor[k_hot].max(), x_model[k_hot].max()))
    vmax_cold = float(max(x_floor[k_cold].max(), x_model[k_cold].max()))

    fig, axes = plt.subplots(3, 2, figsize=(10, 13))
    for row, k, jr, ir, vmax, lbl in [
        (0, k_hot, jh, ih, vmax_hot, "hot"),
        (1, k_cold, jc, ic, vmax_cold, "cold"),
    ]:
        # floor + model share the same colorbar per row so they are directly
        # comparable by eye. Colorbar attached to the RIGHT panel to save space.
        axes[row, 0].imshow(x_floor[k], origin="lower", cmap="gray",
                            vmin=0, vmax=vmax)
        axes[row, 0].set_title(f"floor  k={k} ({lbl})")
        _draw_box(axes[row, 0], jr, ir, "cyan", lbl)
        im = axes[row, 1].imshow(x_model[k], origin="lower", cmap="gray",
                                 vmin=0, vmax=vmax)
        axes[row, 1].set_title(f"model  k={k} ({lbl})")
        _draw_box(axes[row, 1], jr, ir, "cyan", lbl)
        fig.colorbar(im, ax=axes[row, :].ravel().tolist(),
                     fraction=0.03, pad=0.02, label="activity (a.u.)")

    im2a = axes[2, 0].imshow(diff[k_hot], origin="lower", cmap="bwr",
                             vmin=-vmax_diff, vmax=vmax_diff)
    axes[2, 0].set_title(f"model - floor  k={k_hot}")
    im2b = axes[2, 1].imshow(diff[k_cold], origin="lower", cmap="bwr",
                             vmin=-vmax_diff, vmax=vmax_diff)
    axes[2, 1].set_title(f"model - floor  k={k_cold}")
    fig.colorbar(im2b, ax=axes[2, :].ravel().tolist(),
                 fraction=0.03, pad=0.02, label="model - floor")

    subtitle = title
    if cnr_floor is not None and cnr_model is not None:
        subtitle += (f"\nCNR floor = {cnr_floor:.3g},  "
                     f"CNR model = {cnr_model:.3g},  "
                     f"delta = {cnr_model - cnr_floor:+.3g}")
    fig.suptitle(subtitle)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
