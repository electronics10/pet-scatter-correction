"""
dicom_loader.py -- read an Enhanced PT / MR multi-frame DICOM into a plain 3D
array plus the affine that places its voxels in the scanner (patient) frame.

We deliberately do NOT hide the coordinate math behind a black box: the caller
sees the axis order, the voxel spacing, the row/column/slice direction cosines,
and the position of voxel [0,0,0]. resample.py uses those to place any two
volumes in a common frame.

Enhanced PT DICOM (SOP 1.2.840.10008.5.1.4.1.1.130) stores the geometry per
frame under SharedFunctionalGroupsSequence / PerFrameFunctionalGroupsSequence
rather than at the top level, which is why the naive `Rows / Columns /
PixelSpacing` scrape returns blanks. This module reads the functional groups.

Notes on the specific PMOD exports in this project
--------------------------------------------------
- Units are given by RealWorldValueMapping (LUT Explanation = "Bq/ml" for PET,
  "1" for MR). RescaleSlope/Intercept in the per-frame group encodes
  `physical_value = pixel * slope + intercept`.
- Frames are stacked along the slice-normal direction. We infer the slice
  spacing from consecutive frames' ImagePositionPatient, not from
  SliceThickness (which is the physical slice thickness, not the between-slice
  spacing -- they usually agree for reconstructions but not always).
- The `attention_corrected` / `scatter_corrected` flags in these files were
  observed to be unreliable at the folder level; we return them for
  bookkeeping but do not gate anything on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pydicom


@dataclass
class Volume:
    """A 3D DICOM volume plus enough geometry to place it in the scanner frame.

    Attributes
    ----------
    data : (Nf, Nr, Nc) float32
        The pixel values with RescaleSlope/Intercept applied, in the units
        reported by RealWorldValueMapping (see `units`).
    axis_order : tuple of str
        Always ('frame', 'row', 'col') for the DICOM array. This is a reminder,
        not a knob.
    spacing_mm : (float, float, float)
        Voxel spacing along (frame, row, col), in mm. Frame spacing comes from
        successive IPPs; row/col from PixelSpacing.
    origin_mm : (float, float, float)
        Position of voxel [0, 0, 0] in the patient/scanner frame, in mm
        (DICOM's LPS convention). This is IPP of frame 0.
    orient : (3, 3) float
        Direction cosines from array axes to patient frame. Row 0 is the
        frame-axis normal (unit vector), row 1 the row-axis direction, row 2
        the col-axis direction. Assembled from IOP + cross-product.
    units : str
        e.g. "Bq/ml", "g/ml{SUVbw}", "1".
    metadata : dict
        Everything else worth remembering: modality, patient position code,
        AC/SC flags (unreliable but informative), reconstruction method,
        energy window if present.
    """
    data: np.ndarray
    axis_order: tuple
    spacing_mm: tuple
    origin_mm: tuple
    orient: np.ndarray
    units: str
    metadata: dict

    def voxel_center_mm(self, k: int, j: int, i: int) -> np.ndarray:
        """Position of voxel [k, j, i] in the patient frame."""
        o = np.asarray(self.origin_mm, dtype=np.float64)
        d = np.asarray(self.spacing_mm, dtype=np.float64)
        return o + (k * d[0]) * self.orient[0] \
                 + (j * d[1]) * self.orient[1] \
                 + (i * d[2]) * self.orient[2]

    def physical_extent_mm(self) -> dict:
        """The bounding box's corner positions in the patient frame."""
        Nk, Nj, Ni = self.data.shape
        corners = [self.voxel_center_mm(k, j, i)
                   for k in (0, Nk - 1) for j in (0, Nj - 1) for i in (0, Ni - 1)]
        c = np.asarray(corners)
        return dict(min=c.min(0), max=c.max(0), span=c.max(0) - c.min(0))


# ------------------------- internal helpers --------------------------------

def _sq_walk(seq):
    """Yield every dataset element inside a nested SQ tree."""
    for item in seq:
        for el in item:
            yield el
            if el.VR == "SQ":
                for sub in _sq_walk(el.value):
                    yield sub


def _from_functional_groups(ds):
    """Pull the geometry / rescale / units fields from
    SharedFunctionalGroupsSequence and PerFrameFunctionalGroupsSequence.

    Returns (spacing_rc, slice_thk, iop, ipp_per_frame, slope, intercept, units).
    """
    ps = None
    slice_thk = None
    iop = None
    ipp = []
    slope, intercept = None, None
    units = None
    sf = getattr(ds, "SharedFunctionalGroupsSequence", None)
    pf = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if sf:
        for el in _sq_walk(sf):
            if el.tag == (0x0028, 0x0030): ps = list(el.value)
            elif el.tag == (0x0018, 0x0050): slice_thk = float(el.value)
            elif el.tag == (0x0020, 0x0037): iop = list(el.value)
    if pf:
        for i, item in enumerate(pf):
            # first frame carries geometry we want; grab all IPPs for spacing
            for el in _sq_walk([item]):
                if el.tag == (0x0020, 0x0032): ipp.append(list(el.value))
                elif el.tag == (0x0020, 0x0037) and iop is None:
                    iop = list(el.value)
                elif el.tag == (0x0028, 0x0030) and ps is None:
                    ps = list(el.value)
                elif el.tag == (0x0018, 0x0050) and slice_thk is None:
                    slice_thk = float(el.value)
                elif el.tag == (0x0028, 0x1053) and slope is None:
                    slope = float(el.value)
                elif el.tag == (0x0028, 0x1052) and intercept is None:
                    intercept = float(el.value)
                elif el.tag == (0x0028, 0x3003) and units is None:
                    units = str(el.value)
                elif el.tag == (0x0040, 0x9210) and units is None:
                    units = str(el.value)
    return ps, slice_thk, iop, ipp, slope, intercept, units


def _orient_from_iop(iop) -> np.ndarray:
    """Build a 3x3 orientation matrix (frame_normal, row_dir, col_dir) from the
    DICOM IOP row/col direction cosines. Frame normal = row_dir x col_dir."""
    row = np.asarray(iop[:3], dtype=np.float64)
    col = np.asarray(iop[3:], dtype=np.float64)
    normal = np.cross(row, col)
    n = np.linalg.norm(normal)
    if n < 1e-9:
        raise ValueError(f"degenerate IOP (row x col = 0): {iop}")
    normal = normal / n
    return np.vstack([normal, row, col])


def _spacing_from_ipp(ipp_list, orient, fallback_thk) -> float:
    """Slice spacing = signed distance between consecutive frames along the
    slice normal. Falls back to SliceThickness if only one frame is present."""
    if len(ipp_list) < 2:
        if fallback_thk is None:
            raise ValueError("cannot infer frame spacing from a single frame "
                             "and no SliceThickness")
        return float(fallback_thk)
    p0 = np.asarray(ipp_list[0], dtype=np.float64)
    p1 = np.asarray(ipp_list[1], dtype=np.float64)
    normal = orient[0]
    dz = float(np.dot(p1 - p0, normal))
    if dz == 0.0:
        raise ValueError("consecutive frame IPPs project to zero along the "
                         "slice normal -- unusual layout")
    # Return a POSITIVE spacing; the sign is absorbed into orient[0] if needed.
    if dz < 0:
        orient[0] = -orient[0]
        dz = -dz
    return dz


# ------------------------------ public API ---------------------------------

def read_volume(path: str | Path) -> Volume:
    """Read one Enhanced PT / MR multi-frame DICOM into a Volume.

    Does NOT do any resampling; that lives in resample.py. Does NOT interpret
    the values (Bq/mL, SUV, arbitrary MR signal); that is left to `units` on
    the returned Volume.
    """
    path = Path(path)
    ds = pydicom.dcmread(str(path))

    ps, slice_thk, iop, ipp, slope, intercept, units = _from_functional_groups(ds)
    if iop is None or ps is None or not ipp:
        raise ValueError(f"{path} lacks per-frame geometry (IOP/PixelSpacing/IPP)")

    orient = _orient_from_iop(iop)
    dz = _spacing_from_ipp(ipp, orient, slice_thk)   # may flip orient[0]
    dy, dx = float(ps[0]), float(ps[1])
    origin = tuple(float(v) for v in ipp[0])

    pix = ds.pixel_array.astype(np.float32)
    if slope is not None or intercept is not None:
        pix = pix * float(slope or 1.0) + float(intercept or 0.0)

    modality = str(getattr(ds, "Modality", ""))
    meta = dict(
        modality=modality,
        manufacturer=str(getattr(ds, "Manufacturer", "")),
        model=str(getattr(ds, "ManufacturerModelName", "")),
        series_description=str(getattr(ds, "SeriesDescription", "")),
        patient_position=str(getattr(ds, "PatientPosition", "")),
        anatomical_orientation_type=str(getattr(ds, "AnatomicalOrientationType", "")),
        attenuation_corrected=(str(ds[0x0018, 0x9759].value)
                               if (0x0018, 0x9759) in ds else ""),
        scatter_corrected=(str(ds[0x0018, 0x9760].value)
                           if (0x0018, 0x9760) in ds else ""),
        source_path=str(path),
    )
    # Energy window (PET only)
    ew = getattr(ds, "EnergyWindowRangeSequence", None)
    if ew:
        try:
            item = ew[0]
            meta["energy_window_low_keV"] = float(item[0x0054, 0x0014].value)
            meta["energy_window_high_keV"] = float(item[0x0054, 0x0015].value)
        except Exception:
            pass

    return Volume(
        data=pix,
        axis_order=("frame", "row", "col"),
        spacing_mm=(dz, dy, dx),
        origin_mm=origin,
        orient=orient,
        units=units or "",
        metadata=meta,
    )


def describe(vol: Volume) -> str:
    """Human-readable one-block summary. Handy for logs."""
    ext = vol.physical_extent_mm()
    lines = [
        f"Volume from {vol.metadata.get('source_path')}",
        f"  modality={vol.metadata.get('modality')} units={vol.units!r}",
        f"  shape (frame,row,col) = {vol.data.shape}",
        f"  spacing (mm)          = {vol.spacing_mm}",
        f"  origin (mm)           = {vol.origin_mm}",
        f"  orient  (rows: frame_normal, row_dir, col_dir)",
        f"    frame_normal = {vol.orient[0]}",
        f"    row_dir      = {vol.orient[1]}",
        f"    col_dir      = {vol.orient[2]}",
        f"  patient-frame bbox: min={ext['min']} max={ext['max']} span={ext['span']}",
        f"  value stats: min={vol.data.min():.4g} max={vol.data.max():.4g} "
        f"mean={vol.data.mean():.4g} median={np.median(vol.data):.4g}",
        f"  AC={vol.metadata.get('attenuation_corrected')!r} "
        f"SC={vol.metadata.get('scatter_corrected')!r}",
    ]
    return "\n".join(lines)
