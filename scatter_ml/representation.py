"""
representation.py -- mirror-merge, (zbar, d) coordinates, and the inverse map.

The seam between DATA and MODEL. Three facts it encodes:

1. Mirror-merge (EXACT, Poisson-additive). Each oblique ring pair is stored as
   two ordered planes (a,b) and (b,a) that hold the same physical LORs (the
   kernel's orientation isotropy). Summing them is lossless: sum of two Poisson
   counts with the same mean is Poisson. 5625 ordered pairs -> 2850 unordered
   planes, counts per plane doubled.

2. (zbar, d) coordinates. Index each merged plane by
       d    = |r1 - r2|   (ring difference / obliqueness)
       s    = r1 + r2      (zbar = s/2, axial position)
   At fixed d the planes form one axial stack ordered by s -- the "segment".
   This is a RELABELING of the plane axis, not a data transform.

3. Inverse map (the reconstruction seam). MLEM consumes span=1 ORDERED planes as
   its `contamination` term, in read_sinogram_ring_pairs order. A model predicts
   MERGED scatter, so we must undo the merge:
       ordered[(a,b)] = 0.5 * merged[{a,b}]   for oblique (split back to mirrors)
       ordered[(a,a)] =       merged[{a,a}]   for direct
   Get the 0.5 or the ordering wrong and the scatter scale entering MLEM is
   silently off -- so this lives in one audited place.

Geometry depends only on the config (ring pairing), NOT on any run's data, so it
is built once and reused for every run. The ordered-plane order here is IDENTICAL
to read_sinogram_ring_pairs / from_run's A.out_shape (both derive from
plane_ring_pairs' filled order), so merged<->ordered round-trips line up with the
projector by construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mcgpu_pet_wrapper.config import plane_ring_pairs


@dataclass
class MergeGeometry:
    inv: np.ndarray        # (P,) ordered plane -> merged plane index
    w: np.ndarray          # (P,) unmerge weight: 0.5 oblique, 1.0 direct
    m_rlo: np.ndarray      # (M,) merged plane's low ring
    m_rhi: np.ndarray      # (M,) merged plane's high ring
    d: np.ndarray          # (M,) ring difference  = m_rhi - m_rlo
    s: np.ndarray          # (M,) ring sum         = m_rhi + m_rlo
    segments: dict         # d -> (n_d,) merged indices, sorted by s ascending
    P: int                 # number of ordered planes (== A.out_shape[0])
    M: int                 # number of merged planes
    n_rings: int

    def merge(self, ordered: np.ndarray) -> np.ndarray:
        """(P, A, R) ordered counts -> (M, A, R) merged counts (summed mirrors)."""
        A, R = ordered.shape[1:]
        merged = np.zeros((self.M, A, R), dtype=np.float32)
        np.add.at(merged, self.inv, ordered.astype(np.float32))
        return merged

    def unmerge(self, merged: np.ndarray) -> np.ndarray:
        """(M, A, R) merged -> (P, A, R) ordered, split back across mirrors.

        This is the reconstruction seam: the returned array is in A.out_shape
        plane order and can be passed straight to mlem(..., contamination=...).
        """
        ordered = merged[self.inv]                      # (P, A, R)
        return ordered * self.w[:, None, None]


def build_geometry(config) -> MergeGeometry:
    """Construct the merge geometry from the config alone (no data read)."""
    pairs = plane_ring_pairs(config)
    filled = [i for i, p in enumerate(pairs) if p]
    # ordered plane ring labels, in read_sinogram_ring_pairs / A order
    ring1 = np.array([pairs[i][0][0] for i in filled], dtype=np.int64)
    ring2 = np.array([pairs[i][0][1] for i in filled], dtype=np.int64)
    P = len(filled)
    n_rings = int(config["scanner"]["num_rings"])

    rlo = np.minimum(ring1, ring2)
    rhi = np.maximum(ring1, ring2)
    key = rlo * n_rings + rhi                            # unique per unordered pair
    uniq, inv = np.unique(key, return_inverse=True)      # inv: ordered -> merged
    M = len(uniq)
    m_rlo = uniq // n_rings
    m_rhi = uniq % n_rings
    d = m_rhi - m_rlo
    s = m_rhi + m_rlo

    # unmerge weight: direct planes have one ordering (w=1), oblique two (w=0.5)
    w_merged = np.where(m_rlo == m_rhi, 1.0, 0.5).astype(np.float32)
    w = w_merged[inv]

    segments = {}
    for dd in np.unique(d):
        idx = np.flatnonzero(d == dd)
        segments[int(dd)] = idx[np.argsort(s[idx])]     # sort by axial position

    return MergeGeometry(inv=inv.astype(np.int64), w=w, m_rlo=m_rlo, m_rhi=m_rhi,
                         d=d, s=s, segments=segments, P=P, M=M, n_rings=n_rings)