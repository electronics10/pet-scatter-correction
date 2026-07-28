# real_data — pilot test of the trained scatter model on DICOM data

Test whether the pilot scatter model (`checkpoints_pilot/trial2/best.pt`)
improves CNR when applied to a real reconstructed DICOM, forward-projected
back into the trained scanner geometry.

**What this is**: a supervisor-facing demo. We do not have raw sinograms
from the real scanner, so we forward-project a reconstructed DICOM to
synthesize a pseudo-prompt in our trained (discrete-crystal) geometry, run
the model, reconstruct twice (with and without the model's scatter
estimate), and compare CNR on a hot-vs-cold ROI pair.

**What this is not**: a rigorous validation. The pseudo-prompt is not a
real acquisition, we do not have ground-truth scatter, and the mu-map is a
heuristic body mask. See "Caveats" at the bottom.

---

## Layout

```
real_data/
  dicom_loader.py      # read Enhanced PT / MR DICOM -> Volume(...)
  resample.py          # place a Volume onto the trained voxel grid
  mu_map.py            # build a mu-map at 511 keV on the trained grid
  pipeline.py          # forward-project -> predict scatter -> MLEM twice
  cnr.py               # ROIs + CNR = (mean_hot - mean_bg) / std_bg
  plots.py             # ROI overlay + reconstruction-comparison figures
  run_iq_phantom.py    # driver script for the IQ_Phantom dataset
  README.md            # this file
  out/                 # created on first run
```

**Sibling modification**: `scatter_ml/predict.py` grew a new function
`predict_scatter_from_prompts(prompts_ordered, cfg, model, ...)`. The
existing `predict_scatter` (reads trues + scatter from a run dir) is
unchanged.

---

## Run it (Linux/GPU box)

```bash
cd pet-scatter-correction
pixi run python real_data/run_iq_phantom.py
```

Outputs go to `real_data/out/iq_phantom_optA/`. Look at `roi_overlay.png`
first — if the auto-picked ROIs land in obviously wrong places, override
them (see "Overriding ROIs" below) and rerun.

Expected runtime on a Titan RTX: ~30 s for forward projection + attenuation
factors, ~10 s for the model, ~1 minute per MLEM arm × 2 arms. Total < 3
minutes.

## Overriding ROIs

In `run_iq_phantom.py`, replace the `auto_iq_rois(...)` call with explicit
boxes:

```python
from cnr import roi_from_box

hot = roi_from_box("hot_manual", activity.shape,
                    k_range=(85, 92), j_range=(35, 45), i_range=(30, 40))
cold = roi_from_box("cold_manual", activity.shape,
                    k_range=(100, 104), j_range=(40, 48), i_range=(40, 48))
```

Ranges are half-open (numpy slicing). Pick them by opening
`activity_resampled.npy` in a viewer, or by re-saving `roi_overlay.png`
with different boxes.

---

## How the pipeline works

**Step 1: DICOM → Volume**  (`dicom_loader.read_volume`)
Reads the multi-frame Enhanced PT DICOM, pulls the per-frame geometry
(`ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`,
`SliceThickness`, `RescaleSlope`/`RescaleIntercept`, `RealWorldValueMapping`)
and returns a plain 3D array plus its patient-frame affine.

**Step 2: Volume → trained voxel grid**  (`resample.resample_to_grid`)
The trained scanner uses 149 × 80 × 80 voxels at 1 mm iso, scanner-centered.
We build the affine that maps each target voxel to a fractional index in the
source volume, then `scipy.ndimage.map_coordinates` handles trilinear
interpolation. Verified against a synthetic single-hot-voxel case.

**Step 3: Build mu-map on the trained grid**  (`mu_map.build_mu_map`)
Two modes:
- `body_mask`: threshold PET at 2% of max, morphologically clean, fill with
  water (0.0096 /mm) inside. Air outside.
- `none`: mu ≡ 0 everywhere → af = 1 everywhere.

Both reconstructions in the CNR comparison use the same af, so the delta is
internally consistent regardless of mode.

**Step 4: Forward-project → pseudo-prompt**  (`pipeline.synthesize_prompt`)
```
y_expected = A(activity) * af
alpha = target_counts / y_expected.sum()
y_prompts = Poisson(alpha * y_expected)
```
Option A (default): no scatter added. The model sees a scatter-free input.
Option B (not implemented in this pilot): add SSS-based scatter for a more
in-distribution input; hooks in place but the function raises
`NotImplementedError`.

**Step 5: Predict scatter**  (`pipeline.predict_scatter_ordered` →
`scatter_ml.predict.predict_scatter_from_prompts`)
Merges the ordered prompt to (M, A, R), builds 2.5D windows along z̄ per
segment d, per-sample brightness-scales, runs the U-Net, un-scales,
un-merges back to ordered (P, A, R) ready for MLEM's `contamination` arg.

**Step 6: Reconstruct twice, scale-match**  (`pipeline.reconstruct_pair`)
Identical MLEM settings (K=20 iterations, sens_floor_frac=0.025, mult=af)
run once with `contamination=None` (floor) and once with
`contamination=s_hat` (model). scale_match the model to the floor so a
subsequent difference or absolute-mean comparison is not confused by MLEM's
overall-scale freedom. CNR itself is scale-invariant per arm.

**Step 7: CNR**  (`cnr.compare_cnr`)
`CNR = (mean_hot - mean_bg) / std_bg`, computed on both arms. Report both
values and the delta.

---

## Caveats — what this demo does NOT prove

1. **No ground-truth scatter.** We cannot compute a floor/oracle/model
   bracket the way `plan.md` §3 prescribes. The reported quantity is a
   CNR delta between two arms of the same pseudo-acquisition, not a
   fraction of achievable gain.

2. **Pseudo-prompt, not real acquisition.** We forward-project a
   scatter-corrected DICOM reconstruction and add Poisson noise. Real
   acquisitions have effects the forward model does not capture (crystal
   dead time, continuous-crystal light distribution decoding, DOI, etc.).
   The trained model is exposed to none of those in training either, so
   the comparison at least is on the same footing.

3. **Option A is a scatter-free input.** The model was trained on
   trues+scatter prompts. Feeding it just trues (option A) puts it
   out-of-distribution in a specific way: the shape of the input is
   right, but the total-count budget assigned to scatter in training is
   zero here. The model's output on this input is not what a real-scanner
   prompt would elicit. Option B (add SSS to the prompt) is the
   in-distribution version; it is stubbed out until option A is reported.

4. **mu-map is heuristic.** Water inside a PET-thresholded body mask is
   a first-cut approximation. For the phantoms in this dataset it should
   be within a few percent of correct in the body region. For animals
   with bone/lung it would be substantially off. Setting `MU_MODE="none"`
   removes attenuation modeling entirely; both arms still get the same
   treatment so the CNR delta is unchanged in principle.

5. **DICOM AC/NAC folder labels are unreliable.** Both `MLEM_AC` and
   `MLEM_NAC` folders in `20260316/IQ_Phantom` report
   `AttenuationCorrected = NO` in the DICOM tags and have essentially
   identical pixel data. We use whichever, and treat it as our best
   available estimate of activity.

6. **Model checkpoint provenance.** `trial2/best.pt` was trained with
   `loss=poisson, split=True, window_k=7, base=48` per `run_pilot.py`.
   `checkpoints_pilot/trial1` is deprecated per user note.

The CNR delta on a demo like this is directional evidence only — it
answers "does the model change the image" and (weakly) "in which
direction." It does not measure how *close* the correction is to what a
real scanner would need, and it cannot rule out silent-failure modes
(§4.5.5 of the draft).

---

## Files created per run

Inside `out/iq_phantom_optA/` (or wherever `OUT_DIR` is set in the driver):

| file                        | contents                                          |
|-----------------------------|---------------------------------------------------|
| `activity_resampled.npy`    | resampled Bq/mL on the trained grid               |
| `mu_per_mm.npy`             | mu-map at 511 keV (1/mm)                           |
| `y_prompts_ordered.npy`     | pseudo-prompt sinogram in ring-pair order         |
| `attenuation_factors.npy`   | per-bin exp(-∫mu dℓ) factors                       |
| `s_hat_ordered.npy`         | model's scatter prediction                        |
| `x_floor.npy`               | MLEM reconstruction, no scatter correction        |
| `x_model.npy`               | MLEM reconstruction with model scatter, scale-matched to floor |
| `roi_overlay.png`           | ROI placement diagnostic                          |
| `recon_comparison.png`      | floor vs model side-by-side + difference          |
| `cnr_report.txt`            | JSON of ROI stats + CNR values + settings         |
| `config.json`               | trained-scanner config used for the run           |
| `meta.txt`                  | provenance (paths, hyperparams, scale constants)  |

Everything is a numpy array — feed them straight into a Jupyter notebook
if you want to look at other slices, other ROIs, or difference statistics
that the driver does not compute.
