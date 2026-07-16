from pathlib import Path
import mcgpu_pet_wrapper as mpw
import mcgpu_pet_wrapper.rebinning as rb

import matplotlib.pyplot as plt
import numpy as np

from mcgpu_recon.draw_tools import plot3Dimage
from skimage.transform import iradon, resize

run_dir = "data/run_00149"
run_dir = Path(run_dir)
cfg = mpw.load_config(run_dir / "config.json")

sino = mpw.read_sinogram_segments(run_dir,cfg)
sino = rb.ssrb(sino, cfg, arc_correction=False) # (nplanes, nangles, nradials)
plot3Dimage(sino, "sinogram_na.png", "Simulated (MC) sinogram (SSRB; trues only)")

def fbp_3d(sino: np.ndarray, output_size = None) -> np.ndarray:
    n_angles = sino.shape[1]
    theta = np.linspace(90.0, -90.0, n_angles, endpoint=False)
    n_slices = sino.shape[0]
    recons = []
    for z in range(n_slices):
        sino_2d = sino[z, :, :].T
        recons.append(iradon(sino_2d, theta=theta,
                             filter_name='ramp',
                             output_size=output_size))
    return np.stack(recons)

recon = fbp_3d(sino)
# recon: (n_slices, 193, 193) from your original fbp_3d without output_size
recon = resize(recon, (recon.shape[0], 80, 80), order=1, anti_aliasing=True, preserve_range=True)
plot3Dimage(recon, "fbp_na.png", "FBP reconstruction (not arc corrected; trues only)")