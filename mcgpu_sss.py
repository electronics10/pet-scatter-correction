"""
Single Scatter Simulation (SSS, Watson-style) for MCGPU-PET span=1 sinograms.

Produces a model-based estimate of the scatter sinogram in EXACTLY the bin
order of mcgpu_recon.from_run / read_sinogram_ring_pairs:
(n_planes, NANGLES, NRAD), mirror-symmetrized, ready to pass as
mlem(..., contamination=c * s_sss).

Scanner-agnostic by construction: every geometric and spectral constant is
read from the run's config.json (via mcgpu_pet_wrapper.lors and the config
accessors) or from the run's own PENELOPE material files (.mcgpu.gz), which
the Runner symlinks into every run directory. Nothing scanner- or
phantom-specific is hardcoded.

CPU/GPU: pass xp=numpy (default) or xp=array_api_compat.cupy, exactly as with
MCGPUProjector. The heavy stages (ray tables, pair assembly, interpolation)
all run in the chosen namespace; only material-file parsing and the small
coarse-detector layout stay on the host.

Caching: sss_estimate persists the COARSE pair matrix (nD x nD, a few MB),
not the ~700 MB full sinogram. The full sinogram is a cheap deterministic
function of the coarse matrix (interpolate_full), so re-expanding on load
costs seconds and removes any chance of a stale full-resolution array
outliving the coarse one it came from. The cache key hashes every input that
can change the result (config, voxel grid, lambda map, knobs).

Physics model (single scatter, story form)
------------------------------------------
For detectors A, B and scatter point S, two stories produce a single-scatter
coincidence on bin (A, B):

  story(A->S->B): annihilation on segment [A, S]; one photon reaches A
      unscattered (511 keV); the twin reaches S, Compton-scatters by angle
      theta into B, arriving with energy E' = E0 / (2 - cos(theta)).
  story(B->S->A): the mirror.

  s_AB += n_e(S) * dsigma_KN/dOmega(theta) * W(E') * G(A,B,S)
          * [ I(A,S) T(A,S;E0) T(B,S;E') + I(B,S) T(B,S;E0) T(A,S;E') ] * dV

  I(d,S)    = emission line integral of lambda over segment [d, S]
  T(d,S;E)  = exp(-int_[d,S] mu(E) dl)   (mu from the material files)
  W(E')     = probability the Gaussian-blurred detected energy falls in the
              acquisition window [E_low, E_high]  (intrinsic efficiency is
              constant in MCGPU-PET -- no crystal is simulated -- so W is the
              entire energy-dependent efficiency)
  G         = |cos a_A| |cos a_B| / (R_AS^2 R_BS^2), cos a_d = incidence
              cosine on the (radial-normal) cylinder surface at detector d
  n_e(S)    = electron density, derived from the material file's Compton MFP
              at E0 (so it is consistent with what MCGPU actually simulates)

Scattering-angle convention (the classic sign trap): with u_d the unit vector
from S TOWARD detector d, the pre-scatter photon in story(A->S->B) travels
along -u_A (away from A, through S), so

  cos(theta) = (-u_A) . u_B  =  -(u_A . u_B).

Check: A, B diametrically opposite through S gives u_B = -u_A, cos(theta)=+1,
theta=0 (forward, unscattered limit). Correct.

Absolute scale is NOT trusted (multiple scatter, discretization, dropped
constants): fit it with fit_scale_oracle (against MC scatter, isolates shape
error) and/or fit_scale_tail (deployment-realistic, from object-missing LORs).

Assumptions to verify once against your MCGPU-PET build (both are single
greps in the kernel source; see VERIFY notes inline):
  1. material-file table layout: energy | MFP_Rayleigh | MFP_Compton |
     MFP_photoelectric | MFP_total, MFPs in cm at the nominal density
     (parse_mcgpu_material prints a summary; water mu/rho at 511 keV should
     come out ~0.0958 cm^2/g).
  2. energy-resolution model: sigma(E) = res * E / 2.3548 ("linear", default
     here) vs res * sqrt(E0 * E) / 2.3548 ("sqrt"). Grep E_RESOL in the .cu.

Typical use: see example_sss.py.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import parallelproj

from mcgpu_pet_wrapper import lors
from mcgpu_pet_wrapper.config import voxel_space_shape_zyx, grid_size_mm

E0_EV = 511.0e3          # annihilation photon energy used by MCGPU-PET
_FWHM2SIG = 1.0 / 2.354820045
_INV_SQRT2 = 1.0 / math.sqrt(2.0)


# ===========================================================================
# 0. Array-namespace helpers  (xp is numpy or array_api_compat.cupy)
# ===========================================================================

def to_host(a) -> np.ndarray:
    """Coerce any (numpy | cupy | array-api) array to a host numpy array."""
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


def _erf(xp, z):
    """erf on either namespace: numpy has none (route to scipy), cupy has
    cupyx.scipy.special.erf. Both accept their own native array type."""
    if xp is np:
        from scipy.special import erf as _e
        return _e(z)
    try:
        from cupyx.scipy.special import erf as _e
        return _e(z)
    except ImportError:                          # pragma: no cover
        from scipy.special import erf as _e
        return xp.asarray(_e(to_host(z)))


# ===========================================================================
# 1. Material files -> mu(E) and electron density   (host-side, small)
# ===========================================================================

@dataclass
class MaterialTable:
    """Parsed PENELOPE/MCGPU material cross-section table.

    energy_eV        : (N,) ascending grid
    mfp_cm           : dict of (N,) mean free paths at NOMINAL density:
                       keys 'rayleigh', 'compton', 'photoelectric', 'total'
    nominal_density  : g/cm^3 the MFPs refer to
    name             : best-effort material name from the header
    """
    energy_eV: np.ndarray
    mfp_cm: dict
    nominal_density: float
    name: str

    def mu_per_cm(self, E_eV: float, density: float) -> float:
        """Total linear attenuation (1/cm) at energy E for a voxel of the
        given mass density. mu = (rho / rho_nominal) / MFP_total(E)."""
        inv = 1.0 / np.interp(E_eV, self.energy_eV, self.mfp_cm["total"])
        return float(inv) * (density / self.nominal_density)

    def inv_mfp_compton_at_E0(self) -> float:
        """1/MFP_Compton (1/cm) at E0 and nominal density. Proportional to
        the electron density n_e (the KN total cross-section constant cancels
        in the fitted global scale), and consistent with the cross sections
        MCGPU actually samples."""
        return float(1.0 / np.interp(E0_EV, self.energy_eV,
                                     self.mfp_cm["compton"]))


def parse_mcgpu_material(path, verbose=True) -> MaterialTable:
    """Parse a .mcgpu(.gz) material file.

    Strategy: tolerant, structural. The nominal density is the first positive
    float found on/after a header line containing 'DENSITY'. The MFP table is
    the first run of >= 20 consecutive lines with >= 5 numeric columns whose
    first column ascends; columns are read as
    energy | Rayleigh | Compton | photoelectric | total.   # VERIFY (note 1)
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", errors="replace") as f:
        lines = f.read().splitlines()

    name, density = path.stem, None
    for i, ln in enumerate(lines[:60]):
        up = ln.upper()
        if "MATERIAL NAME" in up:
            for nxt in lines[i + 1:i + 4]:
                t = nxt.lstrip("#").strip()
                if t:
                    name = t
                    break
        if "DENSITY" in up and density is None:
            for nxt in ([ln] + lines[i + 1:i + 4]):
                for tok in nxt.replace("#", " ").split():
                    try:
                        v = float(tok)
                    except ValueError:
                        continue
                    if v > 0:
                        density = v
                        break
                if density is not None:
                    break
    if density is None:
        raise ValueError(f"{path}: could not find nominal density in header. "
                         "Inspect the file (zcat | head -40) and fix the "
                         "parser or construct MaterialTable manually.")

    rows, run = [], []
    for ln in lines:
        toks = [] if ln.lstrip().startswith("#") else ln.split()
        vals, ok = [], len(toks) >= 5
        if ok:
            try:
                vals = [float(t) for t in toks[:5]]
            except ValueError:
                ok = False
        if ok and (not run or vals[0] > run[-1][0]):
            run.append(vals)
        else:
            if len(run) >= 20:
                rows = run
                break
            run = [vals] if ok else []
    if not rows and len(run) >= 20:
        rows = run
    if not rows:
        raise ValueError(f"{path}: no MFP table found (>=20 ascending rows "
                         "with >=5 numeric columns).")

    arr = np.asarray(rows, dtype=np.float64)
    tab = MaterialTable(
        energy_eV=arr[:, 0],
        mfp_cm={"rayleigh": arr[:, 1], "compton": arr[:, 2],
                "photoelectric": arr[:, 3], "total": arr[:, 4]},
        nominal_density=density, name=name,
    )
    if verbose:
        # sanity: water should give mu/rho(511 keV) ~ 0.0958 cm^2/g
        mu_rho = tab.mu_per_cm(E0_EV, density) / density
        print(f"[material] {name}: rho_nom={density:g} g/cm^3, "
              f"E=[{arr[0,0]:.3g}, {arr[-1,0]:.3g}] eV, "
              f"mu/rho(511keV)={mu_rho:.4f} cm^2/g")
    return tab


def load_run_materials(run_dir, config, verbose=True) -> list[MaterialTable]:
    """Parse the exact material files this run used (paths from config,
    resolved inside run_dir where Runner symlinked materials/)."""
    run_dir = Path(run_dir)
    return [parse_mcgpu_material(run_dir / p, verbose)
            for p in config["mcgpu"]["materials"]]


def mu_map_per_mm(vg, tables: list[MaterialTable], E_eV: float) -> np.ndarray:
    """(Nz,Ny,Nx) float32 linear attenuation in 1/mm at energy E, from the
    voxel grid's material ids + densities and the run's own cross sections.
    Material id k (1-based) uses tables[k-1]. Host array."""
    mat = np.asarray(vg.material_id)
    rho = np.asarray(vg.density, dtype=np.float64)
    mu = np.zeros_like(rho)
    for k, tab in enumerate(tables, start=1):
        sel = mat == k
        if sel.any():
            inv_mfp = 1.0 / np.interp(E_eV, tab.energy_eV, tab.mfp_cm["total"])
            mu[sel] = inv_mfp * rho[sel] / tab.nominal_density
    return (mu / 10.0).astype(np.float32)          # 1/cm -> 1/mm


def electron_density_map(vg, tables: list[MaterialTable]) -> np.ndarray:
    """(Nz,Ny,Nx) float32 RELATIVE electron density (arbitrary units; the
    absolute constant is absorbed by the fitted scale). Derived from the
    Compton MFP at 511 keV: n_e proportional to (rho/rho_nom)/MFP_Co(E0)."""
    mat = np.asarray(vg.material_id)
    rho = np.asarray(vg.density, dtype=np.float64)
    ne = np.zeros_like(rho)
    for k, tab in enumerate(tables, start=1):
        sel = mat == k
        if sel.any():
            ne[sel] = tab.inv_mfp_compton_at_E0() * rho[sel] / tab.nominal_density
    return ne.astype(np.float32)


# ===========================================================================
# 2. Physics kernel pieces (namespace-generic; operate on xp arrays)
# ===========================================================================

def klein_nishina(xp, cos_theta):
    """dsigma/dOmega per electron, up to the constant r_e^2/2 (absorbed by
    the fitted scale). k = E'/E0 = 1/(2 - cos_theta) for E0 = 511 keV."""
    k = 1.0 / (2.0 - cos_theta)
    return k * k * (k + 1.0 / k - (1.0 - cos_theta * cos_theta))


def window_acceptance(xp, E_eV, config, sigma_model="linear"):
    """P(detected energy in [E_low, E_high]) for a photon arriving with true
    energy E, under MCGPU-PET's Gaussian blur with constant intrinsic
    efficiency.  sigma_model: 'linear' -> sigma = res*E*_FWHM2SIG;
    'sqrt' -> sigma = res*sqrt(E0*E)*_FWHM2SIG.   # VERIFY (note 2)
    """
    m = config["mcgpu"]
    res = float(m["energy_resolution"])
    lo, hi = float(m["energy_window_low_eV"]), float(m["energy_window_high_eV"])
    if res <= 0:
        return xp.astype((E_eV >= lo) & (E_eV <= hi), E_eV.dtype) \
            if hasattr(xp, "astype") else ((E_eV >= lo) & (E_eV <= hi)) * 1.0
    if sigma_model == "linear":
        sig = res * E_eV * _FWHM2SIG
    elif sigma_model == "sqrt":
        sig = res * xp.sqrt(E0_EV * E_eV) * _FWHM2SIG
    else:
        raise ValueError(f"unknown sigma_model {sigma_model!r}")
    inv = _INV_SQRT2 / sig
    return 0.5 * (_erf(xp, (hi - E_eV) * inv) - _erf(xp, (lo - E_eV) * inv))


# ===========================================================================
# 3. Geometry: image frame, coarse detectors, scatter points  (host-side)
# ===========================================================================

def image_frame(config):
    """(img_origin, voxsize) in array-axis (z,y,x) mm, scanner-centered --
    identical to MCGPUProjector's convention. Host float32."""
    nz, ny, nx = voxel_space_shape_zyx(config)
    dx, dy, dz = grid_size_mm(config)
    voxsize = np.asarray([dz, dy, dx], dtype=np.float32)
    origin = np.asarray([-(nz - 1) / 2 * dz, -(ny - 1) / 2 * dy,
                         -(nx - 1) / 2 * dx], dtype=np.float32)
    return origin, voxsize


def coarse_detectors(config, crystal_stride=8, n_ring_samples=15):
    """Subsampled detector set. Returns (pos_zyx, normal_yx, cs, rs):
      pos_zyx   (nD,3) float32 positions, D indexed d = ci*nR + ri;
      normal_yx (nD,2) outward radial normal (y,x) at each detector;
      cs (nC,)  coarse crystal indices (uniform stride; requires
                ncrystals % crystal_stride == 0 for a clean periodic wrap);
      rs (nR,)  coarse ring indices, ENDPOINTS INCLUDED so ring interpolation
                never extrapolates.
    """
    sc = config["scanner"]
    ncryst, nrings, R = (sc["num_detectors_per_ring"], sc["num_rings"],
                         sc["radius_mm"])
    if ncryst % crystal_stride:
        raise ValueError(f"crystal_stride {crystal_stride} must divide "
                         f"num_detectors_per_ring {ncryst} (periodic wrap).")
    cs = np.arange(0, ncryst, crystal_stride)
    rs = np.unique(np.round(np.linspace(0, nrings - 1,
                                        n_ring_samples)).astype(int))
    phi = lors.crystal_angles_rad(config)[cs]
    z = lors.ring_z_positions_mm(config)[rs]

    nC, nR = len(cs), len(rs)
    sin_p, cos_p = np.sin(phi), np.cos(phi)
    pos = np.empty((nC * nR, 3), dtype=np.float32)      # (z, y, x)
    nrm = np.empty((nC * nR, 2), dtype=np.float32)      # (y, x)
    for ci in range(nC):
        sl = slice(ci * nR, (ci + 1) * nR)
        pos[sl, 0] = z
        pos[sl, 1] = R * sin_p[ci]
        pos[sl, 2] = R * cos_p[ci]
        nrm[sl, 0] = sin_p[ci]
        nrm[sl, 1] = cos_p[ci]
    return pos, nrm, cs, rs


def select_scatter_points(mu511_per_mm, config, stride=4, mu_thresh=5e-4):
    """Scatter-point lattice inside the attenuating object.

    mu_thresh (1/mm) excludes air (water at 511 keV is ~9.6e-3/mm, air
    ~1.2e-5/mm, so any threshold between works). Returns
      pts (nS,3) float32 scanner-centered mm, in (z,y,x)
      idx (nS,3) int voxel indices (for sampling n_e at the same points)
      dV  float, mm^3 represented by each point (voxel volume * stride^3)
    """
    origin, voxsize = image_frame(config)
    mask = np.asarray(to_host(mu511_per_mm)) > mu_thresh
    sub = np.zeros_like(mask)
    sub[::stride, ::stride, ::stride] = True
    idx = np.argwhere(mask & sub)                       # (nS,3) in (z,y,x)
    if len(idx) == 0:
        raise ValueError("no scatter points selected; check mu map / mu_thresh")
    pts = origin[None, :] + idx.astype(np.float32) * voxsize[None, :]
    dV = float(np.prod(voxsize)) * stride ** 3
    return pts.astype(np.float32), idx, dV


# ===========================================================================
# 4. Ray-integral tables (the O(nD * nS) precomputation)
# ===========================================================================

def _line_integrals(xp, vol, det_pos, pts, origin, voxsize,
                    ray_chunk=2_000_000):
    """Joseph line integrals of vol over every (detector, point) segment.
    Returns (nD, nS) xp float32. Positions in (z,y,x) mm, matching vol's
    (Nz,Ny,Nx) axes. Chunked over detectors so device memory stays bounded."""
    nD, nS = len(det_pos), len(pts)
    v3 = xp.asarray(np.ascontiguousarray(to_host(vol), dtype=np.float32))
    org = xp.asarray(origin, dtype=xp.float32)
    vsz = xp.asarray(voxsize, dtype=xp.float32)
    dp = xp.asarray(det_pos, dtype=xp.float32)
    pp = xp.asarray(pts, dtype=xp.float32)

    out = xp.empty((nD, nS), dtype=xp.float32)
    per = max(1, ray_chunk // max(nS, 1))       # whole detectors per chunk
    for lo in range(0, nD, per):
        hi = min(lo + per, nD)
        m = hi - lo
        xs = xp.repeat(dp[lo:hi], nS, axis=0)            # (m*nS, 3)
        xe = xp.tile(pp, (m, 1))
        val = parallelproj.joseph3d_fwd(xs, xe, v3, org, vsz)
        out[lo:hi] = xp.reshape(xp.asarray(val, dtype=xp.float32), (m, nS))
    return out


def precompute_tables(xp, lambda_map, mu_maps_by_E, det_pos, pts, config,
                      ray_chunk=2_000_000, verbose=True):
    """All per-(detector, scatter-point) quantities, as xp arrays.

    mu_maps_by_E : dict {E_eV: mu_map_per_mm}; must contain E0_EV exactly.
    Returns dict with
      I      (nD,nS) emission integrals of lambda over [det, S]
      T0     (nD,nS) exp(-int mu(E0))
      TE     (nE,nD,nS) exp(-int mu(E_k)) on the ascending grid E_grid
      E_grid (nE,) xp float32, ascending
    """
    origin, voxsize = image_frame(config)
    I = _line_integrals(xp, lambda_map, det_pos, pts, origin, voxsize, ray_chunk)
    E_host = np.asarray(sorted(mu_maps_by_E.keys()), dtype=np.float64)
    TE = xp.empty((len(E_host), len(det_pos), len(pts)), dtype=xp.float32)
    for k, E in enumerate(E_host):
        TE[k] = xp.exp(-_line_integrals(xp, mu_maps_by_E[E], det_pos, pts,
                                        origin, voxsize, ray_chunk))
        if verbose:
            print(f"  [sss] attenuation table {k+1}/{len(E_host)} "
                  f"(E={E/1e3:.0f} keV)")
    k0 = int(np.searchsorted(E_host, E0_EV))
    if k0 >= len(E_host) or abs(E_host[k0] - E0_EV) > 1.0:
        raise ValueError("mu_maps_by_E must contain E0_EV exactly")
    return {"I": I, "T0": TE[k0], "TE": TE,
            "E_grid": xp.asarray(E_host, dtype=xp.float32)}


# ===========================================================================
# 5. Assembly: coarse scatter estimate over detector pairs
# ===========================================================================

def assemble_coarse(xp, tables, det_pos, det_nrm, pts, ne_vals, dV, config,
                    sigma_model="linear", include_incidence=True,
                    verbose=True):
    """Sum single-scatter contributions of every scatter point over every
    coarse detector pair. Returns s_pairs (nD,nD) xp float32, symmetric,
    zero diagonal. Vectorized over pairs; python loop over scatter points.

    ne_vals : (nS,) relative electron density AT the scatter points.
    dV      : mm^3 per scatter point (the quadrature weight).
    """
    I, T0, TE, E_grid = (tables["I"], tables["T0"], tables["TE"],
                         tables["E_grid"])
    nD, nS = I.shape
    nE = int(E_grid.shape[0])

    dp = xp.asarray(det_pos, dtype=xp.float32)
    dn = xp.asarray(det_nrm, dtype=xp.float32)
    pp = xp.asarray(pts, dtype=xp.float32)
    ne = xp.asarray(ne_vals, dtype=xp.float32)
    cols = xp.broadcast_to(xp.arange(nD)[None, :], (nD, nD))

    s = xp.zeros((nD, nD), dtype=xp.float32)
    step = max(1, nS // 10)
    for si in range(nS):
        v = dp - pp[si][None, :]                 # (nD,3) S->det, in (z,y,x)
        R2 = xp.sum(v * v, axis=1)
        u = v / xp.sqrt(R2)[:, None]

        cos_th = xp.clip(-(u @ u.T), -1.0, 1.0)  # the sign trap; see docstring
        Eprime = E0_EV / (2.0 - cos_th)

        kn = klein_nishina(xp, cos_th)
        acc = window_acceptance(xp, Eprime, config, sigma_model)

        # scattered-leg attenuation, linear-interpolated over the energy grid.
        # TpB[a,b] = T(det b, S; E'(a,b));  TpA = TpB.T since E' is symmetric.
        j = xp.clip(xp.reshape(xp.searchsorted(E_grid, xp.reshape(Eprime, (-1,))),
                               (nD, nD)), 1, nE - 1)
        e0, e1 = E_grid[j - 1], E_grid[j]
        w = xp.clip((Eprime - e0) / (e1 - e0), 0.0, 1.0)
        TEs = TE[:, :, si]                       # (nE, nD)
        TpB = TEs[j - 1, cols] * (1.0 - w) + TEs[j, cols] * w

        IT = I[:, si] * T0[:, si]                # (nD,)
        term = IT[:, None] * TpB + IT[None, :] * TpB.T

        geom = 1.0 / (R2[:, None] * R2[None, :])
        if include_incidence:
            ca = xp.abs(xp.sum(u[:, 1:] * dn, axis=1))
            geom = geom * (ca[:, None] * ca[None, :])

        s = s + ne[si] * kn * acc * geom * term

        if verbose and (si + 1) % step == 0:
            print(f"  [sss] scatter points {si+1}/{nS}")

    s = s * dV
    idx = xp.arange(nD)
    s[idx, idx] = 0.0
    return s


# ===========================================================================
# 6. Interpolation up to the full (n_planes, NANGLES, NRAD) sinogram
# ===========================================================================

def _bracket(xp, grid, q):
    """For ascending `grid` (nG,) and query `q` (...,), return (i0, w) with
    i0 the lower corner index in [0, nG-2] and w in [0,1] the fractional
    offset. Clamps outside the grid (no extrapolation)."""
    nG = int(grid.shape[0])
    i0 = xp.clip(xp.searchsorted(grid, q) - 1, 0, nG - 2)
    g0, g1 = grid[i0], grid[i0 + 1]
    return i0, xp.clip((q - g0) / (g1 - g0), 0.0, 1.0)


def interpolate_full(xp, s_pairs, cs, rs, ring1, ring2, config,
                     plane_chunk=256, verbose=True):
    """Coarse pair matrix -> full sinogram in read_sinogram_ring_pairs order.

    Quadrilinear gather on the grid s4[c1, r1, c2, r2] (coarse crystals x
    coarse rings, squared). The crystal axes are periodic, so the grid is
    padded with a wrapped copy at index cs[0]+ncryst; the ring axis includes
    both endpoints, so no extrapolation occurs on either axis.

    Each full bin (plane p, ith, ir) maps to crystals (ix1, ix2) via
    lors.transverse_crystal_pairs and rings (ring1[p], ring2[p]); the bin
    value is the MEAN of the two mirror orientations, matching
    MCGPUProjector's forward model exactly.

    Interpolation happens in detector-INDEX space (not physical space): the
    scatter surface is smoothest there, and the coarse grid is regular there.
    """
    ncryst = config["scanner"]["num_detectors_per_ring"]
    nC, nR = len(cs), len(rs)
    s4 = xp.reshape(xp.asarray(s_pairs, dtype=xp.float32), (nC, nR, nC, nR))
    # periodic pad on both crystal axes -> (nC+1, nR, nC+1, nR)
    s4 = xp.concat([s4, s4[:1]], axis=0) if hasattr(xp, "concat") \
        else xp.concatenate([s4, s4[:1]], axis=0)
    s4 = xp.concat([s4, s4[:, :, :1]], axis=2) if hasattr(xp, "concat") \
        else xp.concatenate([s4, s4[:, :, :1]], axis=2)

    cg = xp.asarray(np.concatenate([cs, [cs[0] + ncryst]]).astype(np.float32))
    rg = xp.asarray(np.asarray(rs, dtype=np.float32))
    nCg = nC + 1

    ix1_t, ix2_t, hit = lors.transverse_crystal_pairs(config)
    nang, nrad = hit.shape
    hflat = hit.ravel()
    hit_idx = xp.asarray(np.flatnonzero(hflat))
    ix1 = xp.asarray(ix1_t.ravel()[hflat].astype(np.float32))
    ix2 = xp.asarray(ix2_t.ravel()[hflat].astype(np.float32))
    nhit = int(hflat.sum())

    # crystal brackets are plane-independent: hoist them out of the loop
    a0, wa = _bracket(xp, cg, ix1)          # (nhit,)
    b0, wb = _bracket(xp, cg, ix2)
    flat = xp.reshape(s4, (-1,))   # linear index ((a*nR + c)*nCg + b)*nR + d

    def gather(ra, rb):
        """Quadrilinear value at (ix1, ra, ix2, rb) for all hit bins."""
        c0, wc = _bracket(xp, rg, xp.full((nhit,), ra, dtype=xp.float32))
        d0, wd = _bracket(xp, rg, xp.full((nhit,), rb, dtype=xp.float32))
        acc = xp.zeros((nhit,), dtype=xp.float32)
        for da, fa in ((0, 1.0 - wa), (1, wa)):
            ia = a0 + da
            for dc, fc in ((0, 1.0 - wc), (1, wc)):
                ic = c0 + dc
                base_ac = (ia * nR + ic) * nCg
                for db, fb in ((0, 1.0 - wb), (1, wb)):
                    ib = b0 + db
                    base_acb = (base_ac + ib) * nR
                    for dd, fd in ((0, 1.0 - wd), (1, wd)):
                        acc = acc + flat[base_acb + d0 + dd] * (fa * fc * fb * fd)
        return acc

    n_planes = len(ring1)
    out = xp.zeros((n_planes, nang * nrad), dtype=xp.float32)
    for lo in range(0, n_planes, plane_chunk):
        hi = min(lo + plane_chunk, n_planes)
        block = xp.empty((hi - lo, nhit), dtype=xp.float32)
        for i, p in enumerate(range(lo, hi)):
            ra, rb = float(ring1[p]), float(ring2[p])
            block[i] = 0.5 * (gather(ra, rb) + gather(rb, ra))   # mirror mean
        out[lo:hi, hit_idx] = block
        if verbose:
            print(f"  [sss] planes {hi}/{n_planes}")
    out = xp.maximum(out, 0.0)              # kill fp undershoot
    return xp.reshape(out, (n_planes, nang, nrad))


# ===========================================================================
# 7. Scale fitting  (host-side; cheap)
# ===========================================================================

def fit_scale_oracle(s_sss, y_scatter, mask=None) -> float:
    """Global least-squares c minimizing ||y_scatter - c*s_sss|| (optionally
    on a mask). Uses MC truth: isolates SSS *shape* error. Returns c."""
    a = to_host(s_sss).astype(np.float64)
    b = to_host(y_scatter).astype(np.float64)
    if mask is not None:
        m = to_host(mask)
        a, b = a[m], b[m]
    den = float((a * a).sum())
    return float((a * b).sum() / den) if den > 0 else 0.0


def fit_scale_tail(s_sss, y_total, mu_line, mu_thresh=1e-3, min_bins=1000):
    """Deployment-realistic scale from bins whose LOR misses the object
    (mu line integral < mu_thresh, dimensionless): there trues ~ 0 and
    prompts ~ scatter. Returns (c, n_tail_bins). Returns c = nan when the
    tail is too thin to fit -- report that, it is itself a finding for a
    small-bore preclinical scanner."""
    tail = to_host(mu_line) < mu_thresh
    n = int(tail.sum())
    if n < min_bins:
        return float("nan"), n
    return fit_scale_oracle(s_sss, y_total, mask=tail), n


# ===========================================================================
# 8. Caching (coarse level) + pipeline glue
# ===========================================================================

def _cache_key(config, vg, lambda_map, knobs) -> str:
    """SHA1 over everything that can change s_pairs. lambda_map is hashed by
    its float32 bytes, so re-running the prelim MLEM with different settings
    correctly MISSES the cache rather than silently reusing it."""
    h = hashlib.sha1()
    h.update(json.dumps(config, sort_keys=True).encode())
    h.update(json.dumps(knobs, sort_keys=True).encode())
    for a in (np.ascontiguousarray(np.asarray(vg.material_id)),
              np.ascontiguousarray(np.asarray(vg.density, dtype=np.float32)),
              np.ascontiguousarray(to_host(lambda_map).astype(np.float32))):
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def save_coarse(path, s_pairs, cs, rs, key, knobs) -> Path:
    """Persist the coarse pair matrix (a few MB) + provenance. The full
    sinogram is deliberately NOT stored: it is a deterministic function of
    this, and re-expanding costs seconds (see module docstring)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path,
                        s_pairs=to_host(s_pairs).astype(np.float32),
                        cs=np.asarray(cs), rs=np.asarray(rs),
                        key=np.asarray(key), knobs=np.asarray(json.dumps(knobs)))
    return path


def load_coarse(path, key=None):
    """Load a coarse cache. If `key` is given and does not match the stored
    one, return None (stale cache -> recompute), never a silent wrong answer.
    Returns (s_pairs, cs, rs) or None."""
    path = Path(path)
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    if key is not None and str(z["key"]) != str(key):
        return None
    return z["s_pairs"], z["cs"], z["rs"]


def sss_estimate(run_dir, config, vg, lambda_map, ring1, ring2,
                 xp=np,
                 crystal_stride=8, n_ring_samples=15,
                 point_stride=4, mu_thresh=5e-4,
                 E_grid_keV=(300, 350, 400, 450, 511),
                 sigma_model="linear", include_incidence=True,
                 plane_chunk=256, ray_chunk=2_000_000,
                 cache_path=None, use_cache=True, verbose=True):
    """End-to-end SSS: run dir + first-pass activity -> full scatter sinogram,
    UNSCALED (fit the scale afterwards with fit_scale_oracle/fit_scale_tail).

    Parameters of note
    ------------------
    vg         : VoxelGrid of the run (mpw.read_vox) -- materials + densities.
    lambda_map : (Nz,Ny,Nx) first-pass activity estimate (e.g. MLEM with
                 attenuation, no scatter correction). Units arbitrary.
    ring1/2    : plane ring labels from read_sinogram_ring_pairs -- passing
                 them guarantees plane order matches the data (the same
                 contract MCGPUProjector uses).
    xp         : numpy (default) or array_api_compat.cupy, as MCGPUProjector.
    cache_path : where to persist/reuse the COARSE pair matrix. Default
                 run_dir/'sss_coarse.npz'. Stale caches are detected by hash
                 and recomputed, never reused.

    Returns
    -------
    s_sss : xp float32, shape (n_planes, NANGLES, NRAD) == A.out_shape.
    """
    knobs = dict(crystal_stride=crystal_stride, n_ring_samples=n_ring_samples,
                 point_stride=point_stride, mu_thresh=mu_thresh,
                 E_grid_keV=list(E_grid_keV), sigma_model=sigma_model,
                 include_incidence=include_incidence)
    cache_path = Path(cache_path) if cache_path is not None \
        else Path(run_dir) / "sss_coarse.npz"
    key = _cache_key(config, vg, lambda_map, knobs)

    hit = load_coarse(cache_path, key) if use_cache else None
    if hit is not None:
        s_pairs, cs, rs = hit
        if verbose:
            print(f"[sss] coarse cache hit: {cache_path} (key {key})")
    else:
        mats = load_run_materials(run_dir, config, verbose)
        mu_by_E = {float(k) * 1e3: mu_map_per_mm(vg, mats, float(k) * 1e3)
                   for k in E_grid_keV}
        if E0_EV not in mu_by_E:
            mu_by_E[E0_EV] = mu_map_per_mm(vg, mats, E0_EV)
        ne_map = electron_density_map(vg, mats)

        det_pos, det_nrm, cs, rs = coarse_detectors(config, crystal_stride,
                                                    n_ring_samples)
        pts, idx, dV = select_scatter_points(mu_by_E[E0_EV], config,
                                             point_stride, mu_thresh)
        ne_vals = ne_map[idx[:, 0], idx[:, 1], idx[:, 2]]
        if verbose:
            print(f"[sss] {len(det_pos)} coarse detectors "
                  f"({len(cs)} crystals x {len(rs)} rings), "
                  f"{len(pts)} scatter points (dV={dV:.1f} mm^3), "
                  f"E grid {sorted(mu_by_E)} eV, xp={xp.__name__}")

        tabs = precompute_tables(xp, lambda_map, mu_by_E, det_pos, pts,
                                 config, ray_chunk, verbose)
        s_pairs = assemble_coarse(xp, tabs, det_pos, det_nrm, pts, ne_vals,
                                  dV, config, sigma_model, include_incidence,
                                  verbose)
        if use_cache:
            save_coarse(cache_path, s_pairs, cs, rs, key, knobs)
            if verbose:
                print(f"[sss] coarse cache written: {cache_path} (key {key})")

    return interpolate_full(xp, s_pairs, cs, rs, ring1, ring2, config,
                            plane_chunk, verbose)