"""
run_iq_phantom.py -- pilot driver for the 20260316/IQ_Phantom dataset.

One thin script: hard-code the paths, call pipeline.run_one, then run the
CNR machinery on the returned floor/model reconstructions and save the
diagnostic figures.

Run on the Linux/GPU box:

    cd pet-scatter-correction
    pixi run python real_data/run_iq_phantom.py

Outputs go to `real_data/out/iq_phantom_optA/` by default:
    activity_resampled.npy, mu_per_mm.npy,
    y_prompts_ordered.npy, attenuation_factors.npy, s_hat_ordered.npy,
    x_floor.npy, x_model.npy,
    roi_overlay.png, recon_comparison.png,
    meta.txt, config.json, cnr_report.txt.

The ROIs are picked automatically by cnr.auto_iq_rois; open roi_overlay.png
before trusting the CNR numbers, and override with cnr.roi_from_box(...) if
they land in a bad place.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np


# ------------------- CONFIGURATION (edit these) ---------------------------

# Which DICOM to test. The MLEM_AC and MLEM_NAC folders have essentially
# identical pixel data (see dicom investigation); pick either.
DICOM = ("/sessions/cool-beautiful-tesla/mnt/Claude/dicom_data/"
         "20260316/IQ_Phantom/MLEM_AC/PET.dcm")

# Trained model checkpoint.
CKPT = str(HERE.parent / "checkpoints_pilot" / "trial2" / "best.pt")

# Where to write outputs.
OUT_DIR = HERE / "out" / "iq_phantom_optA"

# Attenuation strategy: "body_mask" (water inside PET-thresholded body) or
# "none" (unity af). Both reconstructions in the pilot share the same af so
# the CNR delta is internally consistent either way; "body_mask" reduces
# cupping and makes the images look more reasonable.
MU_MODE = "body_mask"

# Total counts for the pseudo-prompt. Training set spans ~4e7 to ~1e9;
# 5e8 is mid-range and matches typical mouse-scan counts.
TARGET_COUNTS = 5.0e8

# MLEM iteration count. K is a regularizer (§3.5.5 in the draft) -- more
# iterations = sharper, noisier. 20 is a reasonable clinical-ish number.
N_ITER = 20

# Reproducibility for the Poisson noise draw.
SEED = 0

# --------------------------------------------------------------------------


def main():
    import pipeline
    import cnr as cnr_mod
    import plots

    # ---- 1. run the pipeline (returns arrays + saves .npy files) ------
    print("=" * 70)
    print("PILOT: IQ_Phantom, option A (scatter-free forward projection)")
    print("=" * 70)
    result = pipeline.run_one(
        dicom_path=DICOM,
        ckpt_path=CKPT,
        out_dir=OUT_DIR,
        mu_mode=MU_MODE,
        target_counts=TARGET_COUNTS,
        n_iter=N_ITER,
        seed=SEED,
        add_sss=False,
    )

    activity = result["activity"]
    x_floor = result["x_floor"]
    x_model = result["x_model"]

    # ---- 2. pick ROIs on the resampled activity (ground-truth-like) ---
    # Rationale: pick ROIs from the resampled activity, not from x_floor,
    # because the reconstruction can shift/blur features and we want a
    # stable ROI definition shared between the two reconstructions.
    print("\n" + "-" * 70)
    print("Picking ROIs from resampled activity")
    print("-" * 70)
    hot, cold = cnr_mod.auto_iq_rois(activity, verbose=True)

    plots.roi_overlay(
        activity, hot, cold, OUT_DIR / "roi_overlay.png",
        title="Auto-picked ROIs on resampled activity",
    )
    print(f"[plots] saved {OUT_DIR / 'roi_overlay.png'} -- inspect BEFORE "
          "trusting CNR numbers")

    # ---- 3. compute CNR on both reconstructions -----------------------
    print("\n" + "-" * 70)
    print("CNR comparison: floor vs model")
    print("-" * 70)
    summary = cnr_mod.compare_cnr(x_floor, x_model, hot, cold)
    cnr_mod.print_cnr_report(summary)

    # ---- 4. save the comparison figure and a text report --------------
    plots.recon_comparison(
        x_floor, x_model, hot, cold, OUT_DIR / "recon_comparison.png",
        cnr_floor=summary["floor"]["cnr"],
        cnr_model=summary["model"]["cnr"],
        title=f"IQ_Phantom pilot (option A, {N_ITER}-iter MLEM, "
              f"mu={MU_MODE}, counts={TARGET_COUNTS:.0e})",
    )
    print(f"[plots] saved {OUT_DIR / 'recon_comparison.png'}")

    (OUT_DIR / "cnr_report.txt").write_text(json.dumps(
        {"floor": summary["floor"],
         "model": summary["model"],
         "delta_cnr": summary["delta_cnr"],
         "rel_change": summary["rel_change"],
         "dicom": DICOM,
         "ckpt":  CKPT,
         "mu_mode": MU_MODE,
         "target_counts": TARGET_COUNTS,
         "n_iter": N_ITER,
         "seed": SEED},
        indent=2, default=lambda o: str(o)))
    print(f"[report] wrote {OUT_DIR / 'cnr_report.txt'}")


if __name__ == "__main__":
    main()
