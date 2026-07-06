# ML Scatter Correction for MCGPU-PET — Project Plan

**Goal.** Test whether machine learning can perform PET scatter correction, and
whether it can surpass the shortcomings of traditional methods (convolution
subtraction, single-scatter simulation). The pipeline is three repos:
`mcgpu-pet-wrapper` (simulation), `pet-sim-gen` (dataset generation),
`mcgpu-recon` (reconstruction). All are revisable; the plan fixes the parts
that are *expensive to get wrong* and defers the rest to cheap experiments.

---

## 0. Design principle

Separate **generation-time invariants** (baked into every run, costly to
change — the span=11 lesson) from **downstream views** (recomputable from
stored outputs at zero simulation cost). Commit carefully to the former; keep
the latter fluid.

**Stored per run (already produced):** trues sinogram, scatter sinogram,
activity (emission) image, density map (μ-source), config, recipe. Every
approach below — image/sinogram domain, 2D/2.5D/3D, with/without μ-map,
predict-scatter/direct-correction — is a *view* of these. So generation need
not wait on the approach choice, **provided the invariants are right**.

---

## 1. Generation-time invariants (get right the first run)

| Invariant | Decision | Reason |
|---|---|---|
| **span** | **span=1 at tally time** | Only span=1 preserves the one-plane-per-ring-pair LOR map that reconstruction needs. Compression at tally is *not invertible*. Rebinning to higher span *afterward* is fine (it's a downstream view); simulating at high span is the mistake that killed the 2000-sample set if those were tallied at span=11. **Check which happened before discarding them.** |
| **Scanner geometry** | Match the scanner whose DICOMs may later be touched | Geometry is fixed at simulation time; no bounds change fixes a wrong scanner. The default template is *preclinical* (radius 52.5 mm, ~80 mm FOV — rodent scale). If deployment DICOMs are human/clinical, re-template to that ring count / radius / FOV first. |
| **Phantom distribution** | **Add an explicit warm background** (one large tissue ellipsoid with nonzero activity; sample lesions *inside* it) | Scatter is dominated by the bulk active+attenuating medium. The current sampler (air background, disjoint inserts) produces "activity floating in air" — a scatter regime the deployment target does not share. This is the single highest-leverage generation change for scatter work. |
| **Count level** | Simulate at the **top** of the count range | Poisson thinning/splitting can only go *down* in counts, not up. Set the ceiling here; derive lower-count realizations downstream for free (§4). |
| **Acquisition time / activity scale** | Leave time fixed; do **not** add a time bound | Counts ≈ activity × time × sensitivity; with time and geometry fixed, counts ∝ activity, and expected scatter pattern + SF are invariant to that scale (SF is a ratio, scale cancels). Time and activity-scale are one effective parameter. The noise axis is produced downstream, not by simulating more. |

**Nuance — activity does double duty.** Each object's `activity_Bq_per_mL` is
drawn independently, so one variable sets both (i) inter-object *contrast*
(label-relevant) and (ii) global *scale* → counts → noise (label-irrelevant in
expectation). Because they're drawn together, contrast and count-level are
correlated in the current sampler. Decorrelating them (same contrast at many
count levels) is what the downstream count axis buys.

---

## 2. Dataset size — decide by measurement, not guess

1. Generate a **pilot (~300 runs)**, stratified on the existing `sf_proxy` key.
2. Train the first model; plot **validation error vs. training-set size** on
   nested subsets; extrapolate to choose the full batch.
3. Converts "how many runs" from a guess into a learning-curve read. 300 runs
   is ~40 min at your throughput, cheap against a wrong 10k-run batch.

**Leverage note.** One run yields many training samples in 2D/2.5D: after
mirror-merge there are **2850 planes/run** (§6), so a 300-run pilot is ~850k
plane-samples. Per-plane data is not the bottleneck; *phantom diversity* is.

---

## 3. Evaluation harness — build BEFORE training

"Surpass traditional correction" is only meaningful against a bracket. Monte
Carlo gives the exact scatter separation real data never provides, so the
bracket is nearly free. Reconstruct (your `mlem`, `contamination=` term) three
ways:

- **Floor** — `contamination=None` on trues+scatter (no correction).
- **Oracle ceiling** — `contamination = true scatter sinogram`. No method can
  beat this; it *defines* 100% of achievable gain.
- **Method** — `contamination = predicted scatter`.
- *(Optional traditional point inside the bracket)* — convolution-subtraction
  first; full SSS only if needed.

**Metrics on the RECONSTRUCTION, not the sinogram:** ROI bias, contrast
recovery, background uniformity. Rationale: sinogram MSE is misleading —
low-count-bin errors are cheap in MSE but amplified by MLEM's ratio update, so
a sinogram-MSE-optimal prediction can still reconstruct badly.

*This is the highest-priority item: until the bracket exists, no training
result is interpretable (you can't state what fraction of oracle gain you
captured).*

---

## 4. The noise axis and the independence trap

**The trap.** The scatter label is itself a Poisson realization whose counts
are *literally contained in* the trues+scatter input. So the standard
Noise2Noise assumption (label noise independent of input) fails — a network can
partially fit label noise because it is visible in the input.

**Fix — Poisson splitting (exact theorem).** If `N ~ Poisson(λ)` and each event
goes to stream A with prob `p` else B, then `N_A ~ Poisson(pλ)`,
`N_B ~ Poisson((1−p)λ)`, and **`N_A ⊥ N_B` exactly**. Binned form: per bin draw
`N_A ~ Binomial(N, p)`, set `N_B = N − N_A`. No listmode needed, zero extra GPU.

Use it two ways:
- **Independent label:** input = split-A of (trues+scatter); label = split-B of
  scatter, rescaled by `p/(1−p)` to be an unbiased estimate of the input's
  scatter mean. Label noise is now independent → conditional-mean target is
  clean.
- **Count (noise) axis:** thin the high-count run to any lower `p` to produce
  realistic low-count inputs at fixed contrast/geometry.

**Hypothesis under which this holds:** counts Poisson, events independent — true
here (no randoms, no dead time in MCGPU trues/scatter tallies). It would break
only under rate-nonlinear detector effects, which are absent.

**Ablation (now nearly free):** correlated labels (naive thinning) vs.
independent labels (split), one switch in the data loader — measures whether the
trap bites in practice. "Run the simulation twice for independent scatter" is
the *right idea but the expensive implementation*: splitting one run is
theorem-equivalent at zero cost.

---

## 5. Loss function — theory is robust

**Theorem (Banerjee et al. 2005):** for **any Bregman divergence** loss (L2 and
Poisson NLL both belong), the minimizer over predictions of expected loss is the
**conditional mean** `E[label | input]`.

- Poisson NLL `ℓ(ŷ,y)=ŷ − y·log ŷ` is *linear in y* → `E_y[ℓ] = ŷ − E[y]·log ŷ`,
  minimized at `ŷ = E[y]`. So the conditional-mean guarantee (and the N2N
  argument) transfers unchanged; **independence is still the load-bearing
  hypothesis**, not the loss.
- Nuance: after `p/(1−p)` rescaling the label isn't Poisson, so NLL is no longer
  a *proper likelihood* — but the minimizer proof used only linearity in `y`, so
  the target is still correct. You lose only the likelihood interpretation
  (calibrated uncertainty), not the regression target.
- Stress-test: equivalence is about the *minimizer*, not the *optimization
  path*. NLL penalizes *relative* error (low-count bins weigh more); L2 penalizes
  absolute. Finite data/finite nets land differently → loss ablation still worth
  running.

---

## 6. Sinogram geometry and the 2.5D representation

**Coordinate change first.** Mirror-merge each unordered ring pair into one
plane (justified by the kernel's orientation-isotropy finding; Poisson-additive,
so free — halves planes, doubles counts/plane). Natural coordinates:
`z̄ = (r1+r2)/2` (axial position), `d = |r2−r1|` (ring difference / obliqueness).
At fixed `d` a segment is a 1-D stack of `N−d` planes along `z̄` — variable
length across segments (the michelogram problem).

**Default-config plane accounting (75 rings, span=1, MRD=74):**
- Allocated block `read_sinogram` returns: **NSINOS = 11101** planes,
  shape `(11101, 160, 193)`. (Segment 0 = 149 = `num_axial_planes`; total
  `= 149 + 2·Σ_{d=1}^{74}(149−2d)`.)
- Filled (physical) planes: **5625 = 75²** — every ordered ring pair, what
  `read_sinogram_ring_pairs` returns; the other 5476 are structural zeros.
- Identity: `11101 = 75² + 74²` (filled + interleaved empty slots).
- After mirror-merge: **2850 = 75·76/2** planes (75 direct + 2775 oblique).

**Representation options, ordered by structure exploited:**

- **(a) Per-segment 2.5D windows — the first-model baseline.** Each plane is a
  sample; channels = a window of `k` neighbors *along `z̄` within the same `d`*
  (replicate-pad at segment ends); feed `d` as a scalar embedding/extra channel.
  One shared network → variable segment length becomes irrelevant (samples are
  windows, not whole segments). Replaces the old SSRB-to-2D route, whose high
  MRD was the problem, not the 2D-ness.
- **(b) Predict in a compressed `(z̄, d)` domain, interpolate up — biggest
  compute saver, second experiment.** Correction to the span=11 lesson: what
  broke was compressing the *data* (recon input must stay span=1). Compressing
  the *scatter estimate* is fine because scatter is smooth in `(z̄, d)`. Predict
  on a coarse grid, interpolate to every span-1 plane, feed as `contamination`.
  **Cross-field connection:** this is exactly how clinical single-scatter
  simulation runs — scatter computed on sparse planes/angles and interpolated,
  precisely because smoothness makes it lossless *for the estimate*.
- **(c) Full 4-D `(z̄, d, θ, r)` tensor with support masking** — most
  expressive, most expensive, no evidence yet it's needed. Ablation only.

**Stress-test of "scatter is smooth in `(z̄, d)`"** (option b leans on it): holds
when single scatter in a bulky medium dominates; weakens near sharp axial density
boundaries (object ends) and at axial-FOV edges (out-of-FOV scatter changes
character). Option (a) doesn't assume it; validate (b) *against* (a).

---

## 7. First model and ablation matrix

**First model:** sinogram-domain, **predict scatter** (not corrected sinogram),
**2.5D** via representation (a).

Reasons: (i) predicting scatter plugs into MLEM's additive term and *preserves
the Poisson model* — direct subtraction makes negatives and breaks the
likelihood; (ii) scatter is smooth/low-frequency (why traditional convolution
works at all) → easiest regression target; (iii) 2.5D captures most axial
correlation at a fraction of 3D cost.

**Ablations (each one switch against this baseline):**
- correlated vs. independent labels (§4)
- L2 vs. Poisson NLL (§5)
- representation (a) vs. compressed-interpolated (b) (§6)
- with vs. without μ-map as an extra input channel
- sinogram-domain vs. image-domain
- 2.5D vs. 3D

**Note on the μ-map ablation and real-data ambition.** A map-free model must
infer the medium from the trues+scatter sinogram itself (object outline, count
deficits, scatter tails outside the object). Learnable, but only if training
shows enough medium diversity (→ the warm-background + geometry invariants of
§1) to force inference rather than memorization. The μ-map channel is the
upper-bound reference for how much that costs.

---

## 8. Priority order

1. **Confirm the 2000-sample set's span** (tallied at 1, or only rebinned?) —
   decides salvage vs. regenerate.
2. **Warm-background sampler change** + **confirm scanner geometry** (§1).
3. **Pilot generation ~300 runs**, stratified on `sf_proxy` (§2).
4. **Evaluation harness** with floor/oracle/method bracket (§3) — *interpretation
   of everything downstream depends on this.*
5. **2.5D sinogram model**, representation (a), predict-scatter (§7).
6. **Learning curve** → choose full dataset scale (§2).
7. **Ablations** (§7), independent-label split first (cheapest, tests the trap).

---

## Appendix — accumulated nuances / gotchas

- span=1 at tally is non-negotiable for reconstruction; higher span only ever as
  a downstream view.
- Mirror-merge is free (Poisson-additive) and simplifies geometry; direct planes
  (`d=0`) need no special handling.
- Poisson splitting is *exact*, not approximate; binomial per-bin form needs no
  listmode.
- Metrics live in image space; sinogram MSE misranks because MLEM amplifies
  low-count-bin error.
- Conditional-mean target is loss-agnostic (Bregman); independence of label
  noise is the real requirement.
- Compressing the *estimate* ≠ compressing the *data*; only the latter was the
  span=11 failure.
- Absolute-scale calibration of any reconstruction is unidentifiable from
  geometry alone; anchor to a known ROI in the input phantom (transpose `xyz`↔
  `(z,y,x)` first).
