from pathlib import Path
import numpy as np
import array_api_compat.cupy as xp

import mcgpu_pet_wrapper as mpw
from mcgpu_recon import (from_run, mlem, attenuation_factors,
                         attenuation_map_from_vox, scale_match)
from mcgpu_recon.metrics import rois_from_activity, object_bbox, evaluate_recon
from mcgpu_recon.draw_tools import plot3Dimage
from scatter_ml.predict import predict_scatter_from_ckpt

XP_KW = dict(xp=xp, plane_chunk=256)
num = "00149"
run_dir = Path("../MCGPU_data/runs/run_" + num)
ckpt = "checkpoints_pilot/trial1_20260706/best.pt"
cfg = mpw.load_config(run_dir / "config.json")
NIT, FLOOR = 23, 0.07

# --- attenuation map from the simulation's own voxel grid ---------------------
vg = mpw.read_vox(run_dir, cfg)
MU_RHO = {1: 0.0870, 2: 0.0958, 3: 0.0937, 4: 0.0929}  # cm^2/g @511keV per material
mu_per_mm = attenuation_map_from_vox(vg, MU_RHO)

# --- data + attenuation factors -----------------------------------------------
y,   A = from_run(run_dir, cfg, **XP_KW)
y_s, _ = from_run(run_dir, cfg, scatter=True, **XP_KW)
y, y_s = xp.asarray(y), xp.asarray(y_s)
af = attenuation_factors(A, xp.asarray(mu_per_mm))
sf = float(y_s.sum() / (y_s.sum() + y.sum()))
print(f"SF = {sf*100:.2f}%")

plot3Dimage(mpw.read_emission_image(run_dir, cfg), "recon_img/emission.png", f"Emission ({num})")

# --- reference (trues) + three arms, identical recon settings -----------------
x       = mlem(A, y,       n_iter=NIT, mult=af, sens_floor_frac=FLOOR, verbose=True)  # target
x_tot   = mlem(A, y + y_s, n_iter=NIT, mult=af, sens_floor_frac=FLOOR, verbose=True)  # floor
x_sc    = mlem(A, y + y_s, n_iter=NIT, mult=af, contamination=y_s,
               sens_floor_frac=FLOOR, verbose=True)                                   # oracle
pred_s  = xp.asarray(predict_scatter_from_ckpt(run_dir, ckpt, cfg))
x_model = mlem(A, y + y_s, n_iter=NIT, mult=af, contamination=pred_s,
               sens_floor_frac=FLOOR)                                                 # model

plot3Dimage(xp.asnumpy(x),       "recon_img/mlem_trues.png",    f"MLEM trues ({num})")
plot3Dimage(xp.asnumpy(x_tot),   "recon_img/mlem_total.png",    f"MLEM total, SF={sf*100:.2f}% ({num})")
plot3Dimage(xp.asnumpy(x_sc),    "recon_img/mlem_sc.png",       f"MLEM SC (MCGPU-PET) ({num})")
plot3Dimage(xp.asnumpy(x_model), "recon_img/mlem_sc_model.png", f"MLEM SC (model) ({num})")

# --- residual images (scale-matched) ------------------------------------------
denom = xp.linalg.norm(x)
for name, arm, fname in [("total", x_tot,   "residual_total"),
                         ("SC",    x_sc,    "residual_sc"),
                         ("model", x_model, "residual_sc_model")]:
    arm_m, c = scale_match(x, arm)
    res = arm_m - x
    df = float(xp.linalg.norm(res) / denom)
    print(f"{name:6s} scale c={c:.4g}  residual={df*100:.2f}%")
    plot3Dimage(xp.asnumpy(res), f"recon_img/{fname}.png",
                f"{name} - trues (scale-matched), diff={df*100:.2f}% ({num})")

# --- metrics: PSNR / SSIM / CRC / CNR / SNR vs trues reference -----------------
# CRC = accuracy (bias): floor worst, oracle best.
# CNR/SNR = noise-sensitive: oracle may be FLAT/WORSE than floor because
#           correction subtracts counts and raises variance (bias-variance
#           tradeoff) -- expected, not a failure.
hot, warm = rois_from_activity(np.asarray(vg.activity))
bbox = object_bbox(np.asarray(vg.activity) > 0)
act  = np.asarray(vg.activity)
ref  = xp.asnumpy(x)

print(f"\n{'arm':8s} {'PSNR':>7s} {'SSIM':>7s} {'CRC':>7s} {'CNR':>7s} {'SNR':>8s}")
for name, arm in [("floor", x_tot), ("oracle", x_sc), ("model", x_model)]:
    arm_m, _ = scale_match(x, arm)
    m = evaluate_recon(xp.asnumpy(arm_m), ref, act, hot, warm, bbox=bbox)
    print(f"{name:8s} {m['psnr']:7.2f} {m['ssim']:7.3f} {m['crc']:7.3f} "
          f"{m['cnr']:7.3f} {m['snr']:8.3f}")