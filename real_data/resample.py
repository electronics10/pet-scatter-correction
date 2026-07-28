"""
resample.py -- place a DICOM Volume onto the trained scanner's voxel grid.

The trained scanner grid is a fixed target: shape (Nz, Ny, Nx) at 1.0 mm
isotropic (from config), centered on the scanner origin in the "array-axis
order" MCGPUProjector expects. Concretely, voxel [k, j, i] center sits at
  z = (k - (Nz-1)/2) * dz
  y = (j - (Ny-1)/2) * dy
  x = (i - (Nx-1)/2) * dx
which matches mcgpu_recon.MCGPUProjector's img_origin convention.

The source is a DICOM Volume (see dicom_loader.py) whose axis order is
(frame, row, col), with an orientation matrix mapping array axes to the
patient/scanner frame. So the map from a target scanner-frame point (z, y, x)
to a source (frame, row, col) index is:

    p = (z, y, x)                            # scanner-frame point
    delta = p - origin_src                   # displacement from source[0,0,0]
    (dframe, drow, dcol) = orient_src @ delta / spacing_src
    value = source[dframe, drow, dcol]       # trilinear

We build every target voxel's (dframe, drow, dcol) coordinate once, then let
scipy.ndimage.map_coordinates do the trilinear interpolation. Voxels that
fall outside the source's bounding box get zero. This is a pure geometry
operation -- no unit conversion, no smoothing, no thresholding.

The frame we work in ("scanner frame") is DICOM's LPS patient frame with the
scanner assumed to sit at the origin. That matches how MCGPU-PET's negative-
radius convention places its detector: at the center of the voxel-space bbox.
For the PMOD exports here, the PET bbox is centered on the origin already
(within a fraction of a mm), so no extra "recentering" is needed. For any
input where the phantom is offset from the DICOM origin, this resampler puts
it wherever the DICOM says -- if you want to recenter, do that upstream by
adjusting the target grid's origin or by preprocessing the input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from dicom_loader import Volume


@dataclass
class TargetGrid:
    """The trained scanner's voxel grid, in (z, y, x) array-axis order.

    Built from a config via `target_grid_from_config`. Kept as a value object
    so it's obvious what geometry is being used, without importing the wrapper
    everywhere resample is called.
    """
    shape_zyx: tuple             # (Nz, Ny, Nx)
    voxsize_zyx: tuple           # (dz, dy, dx) in mm
    origin_zyx: tuple            # position of voxel [0,0,0] center in scanner frame (z,y,x)

    def voxel_centers(self):
        """Grid of every voxel's (z, y, x) center, shape (Nz, Ny, Nx, 3)."""
        Nz, Ny, Nx = self.shape_zyx
        dz, dy, dx = self.voxsize_zyx
        oz, oy, ox = self.origin_zyx
        z = oz + np.arange(Nz, dtype=np.float64) * dz
        y = oy + np.arange(Ny, dtype=np.float64) * dy
        x = ox + np.arange(Nx, dtype=np.float64) * dx
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        return np.stack([Z, Y, X], axis=-1)


def target_grid_from_config(cfg) -> TargetGrid:
    """Build the trained scanner's TargetGrid from a wrapper config dict.

    Matches mcgpu_recon.MCGPUProjector's image geometry exactly:
      shape       = voxel_space_shape_zyx(cfg)
      voxsize     = (dz, dy, dx) with (dx, dy, dz) = grid_size_mm(cfg)
      img_origin  = -(N-1)/2 * voxsize per axis
    """
    from mcgpu_pet_wrapper.config import voxel_space_shape_zyx, grid_size_mm
    Nz, Ny, Nx = voxel_space_shape_zyx(cfg)
    dx, dy, dz = grid_size_mm(cfg)
    origin = (-(Nz - 1) / 2.0 * dz,
              -(Ny - 1) / 2.0 * dy,
              -(Nx - 1) / 2.0 * dx)
    return TargetGrid(shape_zyx=(Nz, Ny, Nx),
                      voxsize_zyx=(dz, dy, dx),
                      origin_zyx=origin)


def resample_to_grid(vol: Volume, target: TargetGrid,
                     order: int = 1, cval: float = 0.0,
                     verbose: bool = True) -> np.ndarray:
    """Resample a DICOM Volume onto a TargetGrid.

    Parameters
    ----------
    vol : Volume
        Source, from dicom_loader.read_volume.
    target : TargetGrid
        Destination geometry, from target_grid_from_config.
    order : int
        Spline interpolation order for map_coordinates. 1 = trilinear
        (default; safe and preserves nonnegativity), 3 = cubic (sharper,
        can undershoot to negative -- fine for MR, wrong for activity).
    cval : float
        Fill value for target voxels that land outside the source bbox.
    verbose : bool
        Print a one-line summary of the mapping.

    Returns
    -------
    out : (Nz, Ny, Nx) float32 in the source's own units (Bq/mL, MR signal, ...).
    """
    # Coordinate-frame bookkeeping. Both frames use DICOM LPS axis directions:
    #   patient x (Left), y (Posterior), z (Superior, = scanner axial).
    # Our TargetGrid labels its axes (z, y, x) in array-axis order to match
    # MCGPUProjector; that maps to patient (LPS_z, LPS_y, LPS_x) 1:1. DICOM
    # stores origin_mm as raw IPP = patient (LPS_x, LPS_y, LPS_z), and
    # vol.orient's rows are unit vectors in the same patient LPS frame. So we
    # reorder target points to (x, y, z) before comparing with origin_mm.
    pts_zyx = target.voxel_centers().reshape(-1, 3)         # (N, 3): (z, y, x)
    pts_xyz = pts_zyx[:, [2, 1, 0]]                          # (N, 3): (x, y, z)

    origin_xyz = np.asarray(vol.origin_mm, dtype=np.float64)
    delta = pts_xyz - origin_xyz[None, :]                    # patient-frame displacement

    # Project onto source's array axes.
    # vol.orient rows are (frame_normal, row_dir, col_dir) in patient (x,y,z),
    # so `delta @ orient.T` gives the components of delta along each of those
    # unit vectors -- i.e. displacement in mm along (frame, row, col).
    src_idx = delta @ vol.orient.T                            # (N, 3): mm along (frame, row, col)
    dspc = np.asarray(vol.spacing_mm, dtype=np.float64)       # (dframe, drow, dcol)
    src_idx = src_idx / dspc[None, :]                         # (N, 3): fractional indices

    if verbose:
        Nk, Nj, Ni = vol.data.shape
        inside = ((src_idx[:, 0] >= 0) & (src_idx[:, 0] <= Nk - 1) &
                  (src_idx[:, 1] >= 0) & (src_idx[:, 1] <= Nj - 1) &
                  (src_idx[:, 2] >= 0) & (src_idx[:, 2] <= Ni - 1))
        print(f"[resample] target {target.shape_zyx} vs source {vol.data.shape}: "
              f"{inside.sum()}/{inside.size} target voxels ({100*inside.mean():.1f}%) "
              f"fall inside source bbox")

    # map_coordinates expects coords with shape (ndim, N) -- (frame, row, col)
    coords = src_idx.T                                    # (3, N)
    out = map_coordinates(vol.data, coords, order=order, mode="constant",
                          cval=cval, prefilter=(order > 1))
    return out.reshape(target.shape_zyx).astype(np.float32)


def _describe_grid(target: TargetGrid) -> str:
    Nz, Ny, Nx = target.shape_zyx
    dz, dy, dx = target.voxsize_zyx
    oz, oy, ox = target.origin_zyx
    return (f"TargetGrid: shape (Nz,Ny,Nx)=({Nz},{Ny},{Nx}) at "
            f"({dz},{dy},{dx}) mm iso; z-range [{oz:.2f}, {oz + (Nz-1)*dz:.2f}] mm, "
            f"y-range [{oy:.2f}, {oy + (Ny-1)*dy:.2f}] mm, "
            f"x-range [{ox:.2f}, {ox + (Nx-1)*dx:.2f}] mm")
