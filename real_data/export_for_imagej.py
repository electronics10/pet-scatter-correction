"""
export_for_imagej.py -- convert .npy volumes from a run to TIFF stacks for
ImageJ / Fiji.

Usage:

    python real_data/export_for_imagej.py [out_dir]

If out_dir is omitted, defaults to real_data/out/iq_phantom_optA (matching
run_iq_phantom.py's OUT_DIR).

Writes to <out_dir>/imagej/ one 32-bit float TIFF stack per volume:

    activity_resampled.tif   -- resampled DICOM activity (Bq/mL)
    mu_per_mm.tif            -- linear attenuation coefficient (1/mm)
    x_floor.tif              -- MLEM reconstruction, no scatter correction
    x_model.tif              -- MLEM with model scatter, scale-matched to floor
    diff_model_minus_floor.tif -- (model - floor) for the difference view

Voxel spacing is 1.0 mm isotropic per config; ImageJ's Image → Properties
lets you enter it if the file doesn't carry the metadata (TIFF does not
preserve mm units natively). On the trained grid used here, the axial
direction is dimension 0 (Z), rows are Y, columns are X.

Recommended ImageJ workflow for CNR:
  1. Open a stack (File → Open → x_floor.tif).
  2. Image → Properties: set voxel width/height/depth to 1.0 mm.
  3. Analyze → Set Measurements: check Mean, Std Dev, Min & Max.
  4. Navigate to the slice you want, draw an oval or rectangle ROI.
  5. Analyze → Measure. Repeat for the second ROI.
  6. Repeat everything for x_model.tif with the SAME ROIs (via ROI Manager:
     add each ROI, save the ROI set, load it on the model stack, Measure).
  7. Compute CNR = (mean_hot - mean_bg) / std_bg for each stack.
  8. If you want to run compare_cnr programmatically instead, use
     rescore.py with the (k, j, i) ranges you settled on.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _save_tiff(arr: np.ndarray, path: Path) -> None:
    """Write a 3D array as a multi-frame 32-bit float TIFF stack.

    Prefers `tifffile` (proper multi-page TIFF with 32-bit float); falls back
    to a naive per-slice PNG dump if tifffile is not available, but that path
    is a last resort and prints a warning.
    """
    try:
        import tifffile
    except ImportError:
        raise ImportError(
            "tifffile is required for TIFF export. Install with:\n"
            "    pixi run pip install tifffile\n"
            "or pipx equivalents. Alternatively rewrite this function to use "
            "another writer of your choice.")
    tifffile.imwrite(str(path), arr.astype(np.float32),
                     photometric="minisblack", metadata={
                         "axes": "ZYX", "unit": "mm", "spacing": 1.0,
                     })


def export(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise NotADirectoryError(f"{out_dir} does not exist")

    dst = out_dir / "imagej"
    dst.mkdir(parents=True, exist_ok=True)

    named = {
        "activity_resampled":       "activity_resampled.npy",
        "mu_per_mm":                "mu_per_mm.npy",
        "x_floor":                  "x_floor.npy",
        "x_model":                  "x_model.npy",
    }
    volumes = {}
    for name, fname in named.items():
        p = out_dir / fname
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        v = np.load(p)
        volumes[name] = v
        _save_tiff(v, dst / f"{name}.tif")
        print(f"[ok] wrote {dst / (name + '.tif')}  shape={v.shape} "
              f"dtype={v.dtype}  min={v.min():.4g} max={v.max():.4g}")

    if "x_floor" in volumes and "x_model" in volumes:
        diff = volumes["x_model"] - volumes["x_floor"]
        _save_tiff(diff, dst / "diff_model_minus_floor.tif")
        print(f"[ok] wrote {dst / 'diff_model_minus_floor.tif'}  "
              f"shape={diff.shape}  |min|={np.abs(diff).min():.4g} "
              f"max={diff.max():.4g}  min={diff.min():.4g}")

    (dst / "README_imagej.txt").write_text(
        "Voxel size: 1.0 mm isotropic (Nz=149, Ny=80, Nx=80).\n"
        "Set in ImageJ: Image -> Properties -> Voxel width/height/depth = 1.\n"
        "\n"
        "Suggested ROI regions (approximate; verify visually):\n"
        "  hot  ROI: k in [85, 91], within the uniform bright body of the\n"
        "            circular phantom, avoiding rods (k<=80) and inserts (k>=96).\n"
        "  cold ROI: k in [99, 103], INSIDE one of the two dark circles\n"
        "            (cold water/air inserts).\n"
        "\n"
        "CNR = (mean_hot - mean_bg) / std_bg. Use the SAME ROIs on x_floor\n"
        "and x_model (ROI Manager -> Save Selection).\n"
    )
    print(f"[ok] wrote {dst / 'README_imagej.txt'}")


def main():
    default = Path(__file__).resolve().parent / "out" / "iq_phantom_optA"
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    print(f"[export_for_imagej] out_dir={arg}")
    export(arg)


if __name__ == "__main__":
    main()
