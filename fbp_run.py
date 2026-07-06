from pathlib import Path
import mcgpu_pet_wrapper as mpw
import mcgpu_pet_wrapper.rebinning as rb

import matplotlib.pyplot as plt
import numpy as np

from skimage.transform import iradon, radon
from scipy.ndimage import zoom

run_dir = "../MCGPU_data/runs/run_00009"
run_dir = Path(run_dir)
cfg = mpw.load_config(run_dir / "config.json")


## ---- plot image ------
fig, axes = plt.subplots(2, 2, figsize=(14,14))

# load simulation data ---------
# emission image
img = mpw.read_emission_image(run_dir, cfg)
# img = np.sum(img, axis=0)
img = img[74]
axes[0][0].imshow(img, origin="lower")
axes[0][0].set_title("mcgpu_image")

# sinogram
trues = mpw.read_sinogram_segments(run_dir,cfg)
scatter = mpw.read_sinogram_segments(run_dir,cfg, scatter=True)
trues = rb.ssrb(trues, cfg)
scatter = rb.ssrb(scatter, cfg)
total = trues + scatter
# trues = np.sum(trues, axis=0)
# scatter = np.sum(scatter, axis=0)
# total = np.sum(total, axis=0)
trues = trues[74]
scatter = scatter[74]
total = total[74]

# Reconstruction
def fbp(sino) -> np.ndarray:
    theta = np.linspace(90.0, -90.0, sino.shape[1], endpoint=False)
    recon = iradon(sino, theta=theta, filter_name='ramp')
    return recon

scatter = fbp(scatter.T) # swap angular and radial
axes[0][1].imshow(scatter, origin="lower")
axes[0][1].set_title("scatter")

total = fbp(total.T) # swap angular and radial
axes[1][0].imshow(total, origin="lower")
axes[1][0].set_title("total")

trues = fbp(trues.T) # swap angular and radial
axes[1][1].imshow(trues, origin="lower")
axes[1][1].set_title("trues")

plt.tight_layout()
plt.savefig("recon_img/comparison.png")
plt.close()
print("Figure save to", "recon_img/comparison.png")