"""
rescore.py -- recompute CNR with user-specified ROI boxes.

Doesn't rerun MLEM. Loads x_floor.npy and x_model.npy from an existing
out_dir, applies rectangular ROIs supplied at the top of the file, and
rewrites recon_comparison.png + cnr_report.txt in place.

Edit HOT_K/J/I and COLD_K/J/I below to match ROIs you drew in ImageJ (or
picked visually from roi_overlay.png). Ranges are half-open: (a, b) means
voxels a .. b-1 inclusive, matching numpy slicing.

To keep the previous auto-picked results, back up recon_comparison.png and
cnr_report.txt before running this script.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np


# ------------------------ EDIT THESE ROI BOXES -----------------------------
# Coordinates use trained-grid axis order (k = axial, j = row = y, i = col = x).
# Half-open intervals. Sanity: on the 149 x 80 x 80 grid, the phantom's
# circular part sits at roughly k in [75, 105], j in [25, 65], i in [15, 55].
#
# Standard NEMA NU 4-2008 CNR: hot ROI inside the uniform hot region;
# cold ROI inside the water insert (one of the two dark circles at k=99..103).

HOT_K  = (85, 92)     # 7 axial slices in the uniform-hot middle of the phantom
HOT_J  = (37, 44)     # 7-voxel window around phantom centroid (edit)
HOT_I  = (32, 39)     # 7-voxel window around phantom centroid (edit)

COLD_K = (99, 104)    # 5 axial slices inside the cold-insert end
COLD_J = (42, 48)     # inside one of the two dark circles at k=99..103 (edit)
COLD_I = (22, 28)     # inside one of the two dark circles at k=99..103 (edit)

# ------------------------- run configuration -------------------------------

DEFAULT_OUT_DIR = HERE / "out" / "iq_phantom_optA"


def main():
    import cnr as cnr_mod
    import plots

    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT_DIR
    print(f"[rescore] out_dir = {out_dir}")
    x_floor = np.load(out_dir / "x_floor.npy")
    x_model = np.load(out_dir / "x_model.npy")
    activity = np.load(out_dir / "activity_resampled.npy")
    print(f"[rescore] loaded x_floor {x_floor.shape}, x_model {x_model.shape}, "
          f"activity {activity.shape}")

    hot = cnr_mod.roi_from_box("hot_manual", x_floor.shape,
                               HOT_K, HOT_J, HOT_I)
    cold = cnr_mod.roi_from_box("cold_manual", x_floor.shape,
                                COLD_K, COLD_J, COLD_I)
    print(f"[rescore] hot  ROI voxels = {hot.mask.sum()}  "
          f"(k={HOT_K}, j={HOT_J}, i={HOT_I})")
    print(f"[rescore] cold ROI voxels = {cold.mask.sum()}  "
          f"(k={COLD_K}, j={COLD_J}, i={COLD_I})")

    plots.roi_overlay(activity, hot, cold,
                      out_dir / "roi_overlay_manual.png",
                      title="Manual ROIs on resampled activity")
    print(f"[plots] saved {out_dir / 'roi_overlay_manual.png'}")

    print("\n" + "-" * 70)
    print("CNR comparison: floor vs model (manual ROIs)")
    print("-" * 70)
    summary = cnr_mod.compare_cnr(x_floor, x_model, hot, cold)
    cnr_mod.print_cnr_report(summary)

    plots.recon_comparison(
        x_floor, x_model, hot, cold,
        out_dir / "recon_comparison_manual.png",
        cnr_floor=summary["floor"]["cnr"],
        cnr_model=summary["model"]["cnr"],
        title=f"IQ_Phantom pilot (manual ROIs; k_hot={HOT_K}, k_cold={COLD_K})",
    )
    print(f"[plots] saved {out_dir / 'recon_comparison_manual.png'}")

    report = {
        "floor": summary["floor"],
        "model": summary["model"],
        "delta_cnr": summary["delta_cnr"],
        "rel_change": summary["rel_change"],
        "hot_roi":  {"k": HOT_K,  "j": HOT_J,  "i": HOT_I,
                     "n_voxels": int(hot.mask.sum())},
        "cold_roi": {"k": COLD_K, "j": COLD_J, "i": COLD_I,
                     "n_voxels": int(cold.mask.sum())},
    }
    (out_dir / "cnr_report_manual.txt").write_text(
        json.dumps(report, indent=2, default=lambda o: str(o)))
    print(f"[report] wrote {out_dir / 'cnr_report_manual.txt'}")


if __name__ == "__main__":
    main()
