# Sinogram evaluation: caveats

## The core issue

MC scatter sinograms are **sparse Poisson realizations of a smooth rate**.
Typical numbers for the NEMA IQ run:

    ~174M bins, ~3.3M total counts  ->  mean = 0.019 counts/bin.

Almost every bin holds 0 or 1 count. Bin-by-bin comparison against ANY smooth
estimate (SSS, ML, or the true mean itself) is dominated by Poisson noise, not
by estimation error.

Concretely: if your estimator equals the true mean m_i exactly, the expected
relative bin-wise error is

    sqrt( sum(m_i) / sum(m_i^2 + m_i) )

For this run: sqrt(3.26e6 / 3.51e6) = **96.5% noise floor**. Any metric that
reads ~96% is measuring noise, not quality.

## Diagnostic signature (when a raw-bin metric is lying)

- global c and per-plane c agree to within 1-2 %,
- per-plane totals match well (axial-profile err < 10 %),
- the reconstruction looks correct,
- but raw ||c*s - y|| / ||y|| ~= 90-100 %.

All four together = you are at the noise floor. Do NOT interpret the raw
number as shape error.

## The two fixes

### A. Aggregation (report this)

Sum bins into blocks large enough that mean counts per block >~ 20, then
compare directly. No debiasing formula needed.

    def agg(a, pa=16, pr=8):
        n_p, n_a, n_r = a.shape
        a = a[:n_p//pa*pa, :, :n_r//pr*pr]
        return a.reshape(n_p//pa, pa, 1, n_a, n_r//pr, pr).sum(axis=(1,3,5))

    ya, sa = agg(y_scatter), agg(s_est)
    c   = (sa*ya).sum() / (sa*sa).sum()
    err = np.linalg.norm(c*sa - ya) / np.linalg.norm(ya)
    floor = np.sqrt(ya.sum() / (ya**2).sum())
    # report err AND floor as a pair

Choose block sizes so mean(ya) >~ 20 and floor << err. If floor is comparable
to err, aggregate more.

### B. Noise-debiased error (secondary sanity check)

    signal_power  = (y**2).sum() - y.sum()
    residual_power = ((c*s - y)**2).sum() - y.sum()
    shape_err = sqrt(max(residual_power, 0) / max(signal_power, 1))

Correct in expectation but subtracts two large near-equal numbers -> noisy
for near-binary sinograms. Use it to double-check A, not as the primary
number.

## Always-report checklist

For any sinogram-domain comparison (SSS vs MC, ML vs MC, SSS vs ML):

1. **Aggregated shape error, with its Poisson floor next to it.**
2. **Per-plane total (axial-profile) error** - aggregates ~all bins per plane,
   nearly noise-free, most trustworthy single number.
3. **Global scale ratio c_oracle / c_tail** - checks absolute normalization.
4. **Image-domain PSNR / SSIM** vs the oracle-corrected reconstruction, in an
   object bbox. Only measure inside the phantom; edge voxels are noise.

Never report:

- raw ||s - y|| / ||y|| on unaggregated sparse sinograms,
- raw MSE-per-bin on unaggregated sparse sinograms,
- per-bin RMSE / mean(y) without a noise floor next to it.

## Consequence for the SSS-vs-ML head-to-head

An ML model trained with MSE loss learns the Poisson mean by construction, so
it is not disadvantaged by the noise itself. But whatever metric you reported
during ML evaluation (per-bin MSE, per-bin RMSE, PSNR on the sinogram) sits
against the same ~96% floor as SSS. Re-evaluate the ML model with the
aggregated protocol, or the comparison is noise vs. noise and neither method
can win.

The metrics that WILL distinguish them, in order of informativeness:

1. Image-domain PSNR / SSIM vs oracle recon, on an OOD phantom (a class not
   in the 3000-sample training set). SSS's physics generalizes; the ML prior
   may not.
2. Aggregated shape error, per plane.
3. Robustness to a wrong mu map (perturb densities +-10 %, re-evaluate).
4. Runtime and memory. Different regimes; both worth reporting.

## Related traps (not this run, but likely to matter later)

- **Log-scale plots hide the floor.** A "beautiful" sinogram plot on log
  color scale can mask a factor-of-2 shape error. Always plot on linear scale
  when comparing.
- **Masking by object bbox** in the image domain is essential (already in
  metrics.object_bbox). The FOV background is dominated by noise + edge
  artifacts; including it makes every method look ~equally good.
- **Scale-match before differencing** (scale_match in mcgpu_recon). MLEM is
  correct up to a global constant; a raw difference conflates level with
  shape.