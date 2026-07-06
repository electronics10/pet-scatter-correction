# scatter_ml — 2.5D sinogram scatter prediction

End-to-end: merged `(z̄, d)` representation → 2.5D U-Net → predicted scatter →
`mlem` reconstruction, scored by the eval harness. Depends on
`mcgpu-pet-wrapper`, `mcgpu-recon`, `torch`, and (for the harness) the GPU
`parallelproj`/`cupy` stack.

## Modules

- `representation.py` — mirror-merge (5625→2850 planes, exact/Poisson-additive),
  `(z̄, d)` coordinates, and the **inverse map** back to span=1 ordered planes.
  Geometry is config-only, so ordered↔merged round-trips align with `A.out_shape`
  by construction. This is the reconstruction seam; its math is unit-tested.
- `dataset.py` — 2.5D windows (k neighbours along z̄ at fixed d), **per-sample
  brightness scaling**, the **Poisson-split API** (independent labels, off by
  default), run-local LRU cache + `RunBatchSampler`, and **split-by-run**.
- `model.py` — small 3-level `UNet2p5D`; input = k planes + a d channel; softplus
  output (non-negative).
- `losses.py` — `mse` (default) and `poisson_nll` (both Bregman → conditional-mean
  minimizer).
- `train.py` — training CLI; run-level split; checkpoints `best.pt`/`last.pt`.
- `predict.py` — recon API: run → predicted scatter in mlem-ready ordered order.

## First run (settled defaults: L2, non-split, k=5)

```bash
# pilot / learning-curve point: limit train runs, cheap
python -m scatter_ml.train --run_root ../MCGPU_data/runs \
    --epochs 20 --limit_train_runs 200

# ablations (one switch each)
python -m scatter_ml.train --run_root ... --loss poisson
python -m scatter_ml.train --run_root ... --split          # independent labels
```

## Evaluate a trained model against floor / oracle

```python
import mcgpu_pet_wrapper as mpw
from eval_harness import evaluate            # scoring fix still pending, see note
from scatter_ml.predict import predict_scatter_from_ckpt

run_dir = "../MCGPU_data/runs/run_00003"
cfg = mpw.load_config(run_dir + "/config.json")
pred = predict_scatter_from_ckpt(run_dir, "checkpoints/best.pt", cfg)  # (P,A,R)
res = evaluate(run_dir, cfg, extra_arms={"model": pred})
# res -> {floor, oracle, model}; gap_closed(res) -> fraction of oracle gap closed
```

## Design notes (why, briefly)

- **Merge is free** (Poisson additivity); it halves planes and doubles counts.
  The inverse splits a merged oblique plane 50/50 back to its two mirrors — the
  0.5 that keeps the scatter scale correct into MLEM.
- **Per-sample scaling** removes overall brightness, which is the *degenerate*
  axis (noise level, not scatter pattern), making the net brightness-invariant.
- **Split off by default**: first answer "does it learn?" with correlated labels,
  then flip `--split` to test whether the independence trap matters in practice.
- **Split by run, never by plane** — otherwise planes of one phantom leak across
  train/test and inflate every score.

## Known caveats

- The **eval harness scoring** still needs its pending fix (reference-region
  calibration instead of global least-squares; drop CoV; add background bias).
  The reconstruction path and the `extra_arms` plumbing here are unaffected.
- **First runs use full (very high) counts**, so inputs are cleaner than
  deployment. Use `--split` (which thins) or add a `thin_to` option when you want
  realistic-noise inputs; the count ceiling in the data is high enough to thin
  down.
- **Run-batched sampling** trades some cross-run mixing for cache locality; raise
  `--cache_capacity` if you want more mixing.