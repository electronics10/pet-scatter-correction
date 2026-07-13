from pathlib import Path
import numpy as np
import array_api_compat.cupy as xp # GPU; use `import numpy as xp` for CPU

import mcgpu_pet_wrapper as mpw
from mcgpu_recon import (from_run, mlem, attenuation_factors,
                         attenuation_map_from_vox, scale_match)
from mcgpu_recon.metrics import object_bbox, evaluate_recon
from mcgpu_recon.draw_tools import plot3Dimage
from scatter_ml.predict import predict_scatter_from_ckpt
import mcgpu_sss as sss
from mcgpu_sss import to_host


ckpt = "checkpoints_pilot/trial2/best.pt"
XP_KW = dict(xp=xp, plane_chunk=256)
run_dir = Path("../MCGPU_data/runs/run_00149")
cfg = mpw.load_config(run_dir / "config.json")
plot3Dimage(mpw.read_emission_image(run_dir, cfg), run_dir/"recon_img/emission.png", "Emission")

# --- attenuation map straight from the simulation's own voxel grid ---------
# mass attenuation coefficients at 511 keV (cm^2/g) per material id; look these
# up for YOUR material list (NIST XCOM). ~0.096 for soft tissue is a fine start.
MU_RHO = {1: 0.087, 2: 0.096, 3: 0.094, 4: 0.093}   # air, water, adipose, spongiosa
vg = mpw.read_vox(run_dir, cfg)
mu_per_mm = attenuation_map_from_vox(vg, MU_RHO)

# --- measured data + attenuation factors -----------------------------------
y, A, r1, r2 = from_run(run_dir, cfg, **XP_KW)                # trues
y_s, _, _, _ = from_run(run_dir, cfg, scatter=True, **XP_KW)  # true scatter
y, y_s = xp.asarray(y), xp.asarray(y_s)
y_tot = y + y_s
sf = float(y_s.sum() / (y_tot.sum()))
af = attenuation_factors(A, xp.asarray(mu_per_mm))   # pass as mlem(mult=...)

pred_s  = xp.asarray(predict_scatter_from_ckpt(run_dir, ckpt, cfg))

# -------- reconstruction -------------------------------
NIT, FLOOR = 23, 0.07

# # ------ SSS ------------
mats = sss.load_run_materials(run_dir, cfg)
mu511 = sss.mu_map_per_mm(vg, mats, sss.E0_EV) 
lam = mlem(A, y_tot, n_iter=10, mult=af, sens_floor_frac=FLOOR, verbose=True)
s_sss = sss.sss_estimate(run_dir, cfg, vg, to_host(lam), r1, r2,
                         xp=xp,
                         crystal_stride=8, n_ring_samples=15,
                         point_stride=4)
mu_line = A(xp.asarray(mu511))
c_tail, n_tail = sss.fit_scale_tail(s_sss, y_tot, mu_line)
s_sss = xp.asarray(c_tail * to_host(s_sss), dtype=xp.float32)


x       = mlem(A, y,       n_iter=NIT, mult=af, sens_floor_frac=FLOOR, verbose=True)  # trues ref
x_tot   = mlem(A, y_tot, n_iter=NIT, mult=af, sens_floor_frac=FLOOR, verbose=True)    # no correction
x_gt    = mlem(A, y_tot, n_iter=NIT, mult=af, sens_floor_frac=FLOOR, verbose=True, 
              contamination=y_s)                                                      # exact scatter
x_ml    = mlem(A, y_tot, n_iter=NIT, mult=af, sens_floor_frac=FLOOR, verbose=True, 
              contamination=pred_s)
x_sss   = mlem(A, y_tot, n_iter=NIT, mult=af, sens_floor_frac=FLOOR, verbose=True, 
              contamination=s_sss)                                                    # model scatter

plot3Dimage(xp.asnumpy(x),     run_dir/"recon_img/mlem_tonly.png", "MLEM trues only")
plot3Dimage(xp.asnumpy(x_tot), run_dir/"recon_img/mlem_tot.png",  f"MLEM total, SF={sf*100:.2f}% (uncorrected)")
plot3Dimage(xp.asnumpy(x_gt),  run_dir/"recon_img/mlem_gt.png",    "MLEM ground truth (corrected with known scatter)")
plot3Dimage(xp.asnumpy(x_ml),  run_dir/"recon_img/mlem_ml.png",    "MLEM model (corrected with model estimated scatter)")
plot3Dimage(xp.asnumpy(x_sss),  run_dir/"recon_img/mlem_sss.png",   "MLEM model (corrected with SSS estimated scatter)")

# --- residual images ------------------------------------------
x_gt_m, _ = scale_match(x, x_gt)
df = float(xp.linalg.norm(x_gt_m - x) / xp.linalg.norm(x))
plot3Dimage(xp.asnumpy(x_gt_m - x), run_dir/"recon_img/residual_gt.png", f"gt - trues only (scale-matched), diff={df*100:.2f}%")

df = float(xp.linalg.norm(x_tot - x_gt) / xp.linalg.norm(x_gt))
plot3Dimage(xp.asnumpy(x_tot - x_gt), run_dir/"recon_img/residual_tot.png", f"tot - gt, diff={df*100:.2f}%")

df = float(xp.linalg.norm(x_ml - x_gt) / xp.linalg.norm(x_gt))
plot3Dimage(xp.asnumpy(x_ml - x_gt), run_dir/"recon_img/residual_ml.png", f"ml - gt, diff={df*100:.2f}%")

df = float(xp.linalg.norm(x_sss - x_gt) / xp.linalg.norm(x_gt))
plot3Dimage(xp.asnumpy(x_sss - x_gt), run_dir/"recon_img/residual_sss.png", f"sss - gt, diff={df*100:.2f}%")

# --- quantitative metrics --------------------------------------
bbox = object_bbox(np.asarray(vg.activity) > 0)

print(f"{'arm':8s} {'PSNR':>7s} {'SSIM':>7s}")
m = evaluate_recon(xp.asnumpy(x_tot), xp.asnumpy(x_gt), bbox=bbox)
print(f"{"floor":8s} {m['psnr']:7.2f} {m['ssim']:7.3f}")
m = evaluate_recon(xp.asnumpy(x_ml), xp.asnumpy(x_gt), bbox=bbox)
print(f"{"model":8s} {m['psnr']:7.2f} {m['ssim']:7.3f}")
m = evaluate_recon(xp.asnumpy(x_sss), xp.asnumpy(x_gt), bbox=bbox)
print(f"{"sss":8s} {m['psnr']:7.2f} {m['ssim']:7.3f}")