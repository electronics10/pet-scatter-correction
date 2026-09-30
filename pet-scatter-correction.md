# Deep Learning-based Scatter Correction for Preclinical PET

---

## How to read this draft

This is a **structural rewrite**, not a final draft. Three kinds of annotation appear throughout:

| Marker    | Meaning                                                                                                                                                                                                                                                 |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[YOURS]` | A passage you must write or verify. The surrounding prose is mine; the mathematical or design content is yours to own.                                                                                                                                  |
| `[SPEC]`  | A pseudocode placeholder. Interface (inputs / outputs / invariants) is sketched; **internal logic is deliberately blank.** You fill it from memory of your code, _before_ looking at the code. That is what makes the later code-audit diagnostic work. |
| `[TK]`    | A missing number, figure, or fact only you have.                                                                                                                                                                                                        |

**What was cut, and why** — see Appendix B. Short version: everything mathematical that did not earn its keep downstream is gone. What remains is a chain of eight steps you can defend end-to-end (Appendix C).

**Provenance note for future-you.** Organizational prose, transitions, and framing in this draft were drafted with Claude. All _mathematical_ content is either (a) already yours from the previous draft, (b) a transcription of a calculation you confirmed you understand, or (c) marked `[YOURS]`. No new theorems were introduced.

---

## Scope note: what this work does and does not model

Stated once, here, so that later chapters can be read without ambiguity.

- **Data are Monte-Carlo simulated** (MCGPU-PET), not acquired on a physical scanner.
- **Randoms are not modelled.** The simulator produces separated _trues_ and _scatter_ sinograms; there is no randoms population. Throughout this report the randoms term $r$ appears in the general forward model of §3.3 because it is standard PET theory and because the Fisher-information argument of §3.6 holds for _any_ additive background — but **in every experiment reported here, $r = 0$.** Where an equation would otherwise carry a dead term, it is dropped and the omission noted.
- **No time-of-flight.** The scanner model is non-TOF. This matters in §4.5 (it removes one route by which activity and attenuation could be disentangled) and is the reason certain claims there are stated as physical intuition rather than as appeals to the joint activity–attenuation literature.
- **Single scanner geometry.** A small-animal ring geometry, held fixed across all runs. Generalization _across scanners_ is out of scope; generalization _across objects_ is the subject of the work.

---

# Chapter 1: Introduction

## 1.1 What PET measures

Cameras are the imaging device everyone knows: they map the visible-light reflectance of surfaces onto a 2D sensor. But visible light does not penetrate the body, and reflectance says little about what tissue is made of or what it is doing. Medical imaging extends the idea of forming a spatial map to signals that do penetrate — or originate from within — the body. Modalities are distinguished by the physical signal used and by whether that signal is transmitted through, emitted after external excitation, or emitted from an internally distributed source. Ultrasound uses acoustic waves and is fast, cheap, and non-ionizing. Magnetic resonance imaging (MRI) uses nuclear spin resonance in a strong magnetic field and gives excellent soft-tissue contrast. Computed tomography (CT) reconstructs X-ray attenuation and provides fast, high-resolution anatomical images, especially of bone and dense structures. Positron emission tomography (PET), the focus of this report, images the spatial distribution of an injected radiotracer, which — depending on the tracer — reports on metabolism, perfusion, or receptor binding.

The tracer decays by positron ($\beta^+$) emission, releasing a positron that thermalizes over a short distance (sub-millimetre to several millimetres, depending on the isotope and the medium) and then annihilates with a nearby electron. The annihilation produces two 511 keV gamma photons travelling in nearly opposite directions — a direct consequence of energy and momentum conservation for an almost-at-rest electron–positron pair. A ring of scintillation detectors surrounding the target (patient, animal, or phantom) records photon pairs that arrive within a short coincidence timing window, typically a few nanoseconds. Each such coincidence defines a **line of response** (LOR): the line joining the two detector elements that fired, along which the annihilation is presumed to have occurred. The reconstruction task is to recover the activity distribution $f(\mathbf{x})$, in positron emissions per unit volume per unit time (Bq/cm³), from the measured set of coincidences.

The physics underlying this scheme was established well before the imaging technology caught up: Dirac predicted the existence of the positron in 1928, later confirmed by Anderson in 1932; artificial positron-emitting radionuclides were produced by the Joliot-Curies in 1934, and the back-to-back annihilation photons were characterized soon after.

## 1.2 From coincidence counting to tomography

The idea of using annihilation coincidence to localize a positron emitter dates to the late 1940s, when Wrenn et al. built opposed-detector devices for brain-tumour localization. These early instruments were essentially one-dimensional and non-tomographic. The decisive step came in the 1970s, once Hounsfield's CT had shown that a tomographic slice could be reconstructed from its projections by filtered backprojection (FBP), and that a volume could be built up as a stack of such slices. Applied to coincidence data — where each LOR within a detector ring is naturally a line integral of activity in that transaxial plane — the same mathematics turned a projection imager into a true tomograph. In 1973–1975, Ter-Pogossian et al. built PETT (Positron Emission Transaxial Tomograph), the direct ancestor of every modern PET scanner. Shortly after, in 1976, Wolf and Fowler synthesized ¹⁸F-FDG, a glucose analogue labelled with fluorine-18. FDG made PET clinically useful — most tumours are hypermetabolic and take up glucose avidly — and it remains the dominant tracer half a century later.

Coincidence events are organized either into a **sinogram**, where LORs are indexed by angle and radial offset, or stored in **list-mode**, event by event with timestamps. Early PET scanners used inter-ring tungsten septa to reject oblique LORs, so acquisition was effectively 2D: each detector ring produced an independent transaxial sinogram, reconstructed slice by slice with FBP and stacked into a volume.

FBP is fast and analytically exact in the noiseless limit, but PET data are neither noiseless nor Gaussian — photon counts per LOR are low and the noise is Poisson — so FBP images suffer from streak artifacts and a poor bias–variance tradeoff at clinical dose levels. Statistical iterative reconstruction addressed this directly. Shepp and Vardi's maximum-likelihood expectation-maximization (MLEM) algorithm, introduced in 1982, models the measurement as a Poisson process and iteratively updates $f$ to maximize the likelihood of the observed counts. MLEM is slow, but Hudson and Larkin's ordered-subset variant OSEM (1994) accelerated it by roughly one to two orders of magnitude and made iterative reconstruction clinically routine. Iterative methods also opened a broader modelling window: attenuation, scatter, randoms, detector response, and normalization can all be incorporated as terms in the forward model, so that a single reconstruction produces a quantitative activity map rather than a picture requiring separate correction.

A separate development in the 1990s was the transition to fully 3D acquisition, achieved by removing the septa: sensitivity increased roughly fivefold, but the scanner now recorded oblique LORs connecting detectors in different rings, and the data no longer decomposed into independent 2D slices. Kinahan and Rogers' 3DRP (1989) provided the first practical fully-3D analytic algorithm, using reprojection to fill in the LORs missing from the scanner's finite axial acceptance before applying a 3D filtered backprojection; it was accurate but computationally expensive. Defrise's FORE (Fourier rebinning, 1997) sidestepped this cost by rebinning the oblique data back onto 2D sinograms in the frequency domain, after which any 2D algorithm — FBP or, more usefully, 2D OSEM — could be applied. FORE is an approximation, derived from the frequency–distance relation under a small-oblique-angle assumption; it degrades for LORs with large ring differences and for activity far from the scanner axis, which limited its use in scanners with long axial fields of view. Within these limits, FORE+OSEM became the clinical standard through the late 1990s and 2000s, until computing power made fully 3D MLEM/OSEM directly on the oblique data routine.

**Fully 3D acquisition also let far more scattered photons into the data, turning scatter from a modest correction into a dominant background.** That sentence is the reason this report exists.

## 1.3 Four modern axes

From the early 2000s onward, PET has developed along four largely independent axes.

**Hybrid imaging.** PET/CT, introduced clinically in 2001, pairs each PET scan with a co-registered CT, which provides anatomical context and — critically — the attenuation map needed for quantitative reconstruction. PET/CT displaced standalone PET within a few years and made whole-body FDG oncology imaging the workhorse application. PET/MR followed around 2011, trading CT's speed for MR's soft-tissue contrast, and is used mainly in neurological and paediatric settings.

**Time-of-flight (TOF).** An idea from the 1980s revived by fast scintillators (LYSO, LSO) and fast electronics: by measuring the sub-nanosecond difference in the two photons' arrival times, the annihilation can be localized to a segment of the LOR (typically a few centimetres), which improves signal-to-noise ratio and reduces sensitivity to inaccuracies in the attenuation and scatter estimates. _(This work is non-TOF; see the scope note.)_

**Digital detectors.** Silicon photomultipliers (SiPMs) have replaced photomultiplier tubes in new scanners, improving timing resolution and enabling denser detector packaging.

**Total-body PET.** Exemplified by the uEXPLORER (~2 m axial field of view, first human images in 2018), which increases sensitivity by roughly 40× over a conventional scanner and enables ultra-low-dose, delayed, and whole-body dynamic imaging.

Overlaid on all four axes is the rapid entry of machine learning into the reconstruction pipeline — for image denoising, attenuation-map synthesis, low-dose image recovery, and, of direct relevance here, scatter estimation.

## 1.4 What breaks the idealized picture

The idealized model — each coincidence defines an LOR through the annihilation point, and the count along each LOR is a line integral of $f$ — is broken by several physical effects.

**Attenuation** is the largest: as the two 511 keV photons traverse tissue, each has an independent probability of being absorbed or scattered out of the LOR, so the probability that _both_ reach the detector depends on the total attenuation along the entire LOR through the body.

**Scatter** occurs when one or both photons undergo Compton scattering in the body before detection; the annihilation point no longer lies on the recorded LOR, producing mispositioned events that manifest as a smooth background.

**Random coincidences** arise when two photons from unrelated annihilations happen to fall within the coincidence timing window. Their spatial distribution is smooth and estimable from single-detector count rates. _(Not modelled here; see the scope note.)_

Beyond these, two effects set a resolution floor independent of the detectors. **Positron range** is the distance the positron travels before annihilating — sub-millimetre for ¹⁸F, several millimetres for higher-energy emitters such as ⁸²Rb — displacing the reconstructed point from the true decay site. **Photon non-collinearity** arises because the annihilating pair is not exactly at rest: residual momentum tilts the two photons away from a strict 180° by about 0.25° FWHM, blurring the LOR by roughly 2 mm across a typical 80 cm bore. Additional detector-level effects — finite crystal size, inter-crystal scatter and penetration, dead time, and gain variations — further degrade the ideal model and are handled through detector-response modelling and normalization scans.

## 1.5 Why scatter is the hard one

Quantitative PET requires each effect above to be estimated and either subtracted from the data or — more commonly in modern practice — incorporated into the forward model of an iterative reconstruction. Attenuation correction uses a map of linear attenuation coefficients at 511 keV, from CT or synthesized from MR. Randoms correction uses a delayed coincidence window or singles rates. Normalization uses a uniform source scan. Dead-time and decay corrections come from the scanner's electronics and the known half-life. Detector response modelling embeds crystal geometry into the system matrix.

**Scatter correction is the remaining major term and the most difficult to estimate accurately.** In fully 3D PET, scattered coincidences typically account for 30–60% of all recorded events in whole-body imaging, making scatter the largest correctable background. A scattered photon loses energy in proportion to its scattering angle, so a narrow energy window around 511 keV rejects some scatter — but not enough. Detector energy resolution (~10–15% FWHM for LYSO) overlaps substantially with small-angle scatter, which is the majority.

Several classes of estimation method are used in practice, and Chapter 4 treats them systematically. In brief: analytic model-based methods, most prominently the **single-scatter simulation** (SSS) of Watson and Ollinger, integrate the Klein–Nishina cross-section over the emission and attenuation maps; SSS is fast and clinically standard, but models only single scatter, so multiple-scatter contributions must be absorbed into a tail-fit scaling. **Monte Carlo** scatter estimation simulates photon transport directly and captures multiple scatter and out-of-field activity accurately, at the cost of computation time. **Energy-window methods** estimate scatter empirically from off-peak windows; simple but noisy.

Scatter correction is particularly hard when the attenuation map is inaccurate (a common issue in PET/MR, where bone is difficult to image), when activity lies outside the field of view, or when TOF information is not available to constrain the estimate spatially. These limitations, together with the general trend of applying data-driven methods across the PET pipeline, motivate the deep-learning approach developed here: a network trained to predict the scatter distribution from the measured emission data alone.

## 1.6 Structure of this report

The argument runs in a single chain.

- **Chapter 2** builds the minimum mathematical and statistical vocabulary: inverse problems, ill-posedness, regularization, likelihood, and the bridge between them (regularization _is_ MAP estimation with a prior).
- **Chapter 3** specializes to PET: the Radon transform and why FBP fails; the Poisson forward model; MLEM as its inversion; why scatter must enter _additively_; and the **Fisher-information ceiling** that bounds what any scatter correction can achieve.
- **Chapter 4** surveys how $\hat s$ is actually produced — convolution methods, SSS, Monte Carlo, and learned estimators — and states the specific bet this work makes.
- **Chapter 5** describes the method: data generation, sinogram representation, Poisson splitting, loss, architecture, training, and evaluation protocol.
- **Chapter 6** reports preliminary results.
- **Chapter 7** discusses where each family of method wins, and what remains open.

---

# Chapter 2: Mathematical and Statistical Preliminaries

This chapter assembles exactly the tools used later, and no more. Each section states where it is used downstream; a reader who already knows this material can skip to Chapter 3 and refer back through the cross-references.

## 2.1 Background facts

### 2.1.1 Fourier transform (unnormalized convention)

For $f: \mathbb{R}^d \to \mathbb{C}$ suitably decaying,

$$\hat f(\mathbf{k}) = \int_{\mathbb{R}^d} f(\mathbf{x}), e^{-i \mathbf{k} \cdot \mathbf{x}} , d\mathbf{x}, \qquad f(\mathbf{x}) = \frac{1}{(2\pi)^d} \int_{\mathbb{R}^d} \hat f(\mathbf{k}), e^{i \mathbf{k} \cdot \mathbf{x}} , d\mathbf{k}.$$

Two facts are used: (i) the Fourier transform of $f * g$ is $\hat f \cdot \hat g$; (ii) integrating a Dirac delta $\delta(\mathbf{x} \cdot \hat{\mathbf{n}} - t)$ against $e^{-i \sigma t}$ collapses the $t$-integral, leaving $e^{-i \sigma (\mathbf{x} \cdot \hat{\mathbf{n}})}$.

_Used in:_ §3.2 (Fourier slice theorem, filtered backprojection).

### 2.1.2 Adjoint of a linear operator

Given a linear map $T: \mathcal{H}_1 \to \mathcal{H}_2$ between inner-product spaces, its **adjoint** $T^\top: \mathcal{H}_2 \to \mathcal{H}_1$ is the unique operator satisfying

$$\langle T f, g \rangle_{\mathcal{H}_2} = \langle f, T^\top g \rangle_{\mathcal{H}_1} \qquad \forall f, g.$$

For matrices with the standard inner product, the adjoint is the transpose. For integral operators, the adjoint is obtained by swapping the roles of input and output variables in the kernel. The point that matters: **the adjoint maps from the codomain back to the domain** — it goes the right way, even though it is not an inverse.

_Used in:_ §3.2 (backprojection is $R^\top$), §3.5 (MLEM's update backprojects a ratio).

### 2.1.3 Poisson distribution

A random variable $Y$ is Poisson with mean $\lambda \geq 0$, written $Y \sim \mathrm{Poisson}(\lambda)$, if

$$\Pr(Y = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

Three facts, all used repeatedly:

1. $\mathbb{E}[Y] = \mathrm{Var}(Y) = \lambda$. **Mean equals variance** — this single identity drives the Fisher ceiling of §3.6.
2. **Additivity**: independent Poisson variables sum to a Poisson variable, $Y_1 + Y_2 \sim \mathrm{Poisson}(\lambda_1 + \lambda_2)$.
3. Photon counts under weak sources are Poisson to excellent approximation.

_Used in:_ §3.3 (forward model), §3.5 (MLEM derivation), §3.6 (Fisher ceiling), §5.4 (Poisson splitting).

## 2.2 Inverse problems

### 2.2.1 Forward model and inverse problem

Many problems in applied mathematics share a structure. A system is described by unknown parameters $x$ in a space $X$. A **forward model** $A: X \to Y$ sends parameters to observations $y = A(x)$. The **inverse problem** asks: given a (possibly noisy) $y$, recover $x$.

Canonical examples:

- **Tomographic imaging**: $x$ is a tissue property or radioactivity, $A$ is a line-integral operator, $y$ is measured counts.
- **Electromagnetic and quantum scattering**: $x$ is a scatterer's shape or potential, $A$ maps it to a scattering amplitude, $y$ is the far-field pattern.
- **Geophysical inversion**: $x$ is a subsurface property, $A$ solves a wave or diffusion equation, $y$ is measured at the surface.

The forward direction is a computation. The inverse direction is an inference — and it is much harder, for reasons the next sections make precise.

### 2.2.2 Operators, adjoints, and why inversion is unstable

The forward map $A$ is, in most physical inverse problems, linear. Studying linear maps between (possibly infinite-dimensional) function spaces is the province of **operator theory**. The facts below are accepted without proof.

**Linear operators between function spaces.** In finite dimensions, a linear map is a matrix $A \in \mathbb{R}^{m \times n}$. In infinite dimensions, vectors become functions (typically in $L^2$) and matrices become operators. The Radon transform $R$ of §3.2 is a linear operator $L^2(\mathbb{R}^2) \to L^2([0,\pi) \times \mathbb{R})$, sending an image to a sinogram. It plays the role of a matrix with infinitely many rows and columns.

**Adjoints are not inverses.** Every linear operator $A$ has an adjoint $A^\top$ (§2.1.2). Applying $A^\top$ in place of $A^{-1}$ is a common move — $A^\top$ is bounded, stable, and often geometrically meaningful — but **it does not invert $A$**. The next paragraph says what it does instead.

**Least squares and the normal equations.** When $A$ is not square, or when $y$ contains noise, the equation $Ax = y$ typically has no exact solution. The standard remedy is least squares:

$$\hat x = \arg\min_{x} ; |Ax - y|^2.$$

The solution has a clean geometric characterization. As $x$ ranges over $\mathbb{R}^n$, the vector $Ax$ ranges over the column space $\mathrm{col}(A) \subset \mathbb{R}^m$. Minimizing $|Ax - y|$ asks: which point of $\mathrm{col}(A)$ is closest to $y$?

**Orthogonality principle.** For a subspace $V$ of an inner-product space, the closest point $v^\star \in V$ to a given $y$ is characterized by

$$v^\star = \arg\min_{v \in V} |y - v| \quad \Longleftrightarrow \quad v^\star \in V ; \text{and} ; (y - v^\star) \perp V.$$

Distance minimization is equivalent to the residual being orthogonal to the subspace. Applied here: $Ax - y$ must be orthogonal to every column of $A$, which is exactly

$$A^\top (Ax - y) = 0 \quad \Longleftrightarrow \quad A^\top A , \hat x = A^\top y.$$

These are the **normal equations**, and $A^\top A$ is the **normal operator**. It is square, lives on the input space, is self-adjoint and positive semi-definite ($\langle A^\top A x, x\rangle = |Ax|^2 \geq 0$), and its eigenvalues are the squared singular values of $A$.

> **[STITCH — editorial correction]** The previous draft claimed that $R^\top R$ is "the object whose inversion gives filtered backprojection." That is not right: FBP inverts $R$ directly via the Fourier slice theorem and a ramp filter (§3.2.4); it never forms or inverts $R^\top R$. The normal-operator picture is retained here because it explains _why small singular values are dangerous_ — which is reused in §3.2.6 and §3.5.5 — not because FBP uses it.

**Singular values and the source of instability.** Every matrix admits a singular value decomposition $A = U \Sigma V^\top$ with singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$. The condition number $\kappa(A) = \sigma_1 / \sigma_{\min}$ controls noise amplification: small singular values of $A$ become tiny eigenvalues of $A^\top A$, and inverting them amplifies whatever noise sits in the corresponding directions.

Operators between function spaces admit an analogous decomposition, with an infinite sequence of singular values. An operator is **compact** if it maps bounded sets to relatively compact ones; integral operators with smooth kernels — including the Radon transform — are compact. The fact that matters:

> If $A$ is compact on an infinite-dimensional space, its singular values satisfy $\sigma_k \to 0$ as $k \to \infty$. Hence $A^{-1}$, if it exists, is **unbounded**.

So the "condition number" of a compact operator is effectively infinite.

> **Slogan.** _Compact operators smooth; their inverses de-smooth; regularization is how we de-smooth safely._

This is the abstract reason inverse problems in imaging, scattering, and PDE-based inference are hard. The finite-dimensional counterpart — a matrix whose small singular values leak noise through — is the numerical shadow of the same phenomenon.

### 2.2.3 Hadamard well-posedness

Hadamard called an inverse problem **well-posed** if

1. a solution exists for every admissible $y$,
2. the solution is unique, and
3. the solution depends continuously on $y$.

If any condition fails, the problem is **ill-posed**. In imaging, failure of (3) — instability — is the operational difficulty: when $A$ is compact, $A^{-1}$ is unbounded, so an arbitrarily small perturbation of $y$ can produce an arbitrarily large perturbation in $x$. Measurement noise then dominates the reconstruction.

### 2.2.4 Regularization

The normal equations are unstable when $A^\top A$ has near-zero eigenvalues. The most direct fix is to lift those eigenvalues away from zero by adding a small multiple of the identity:

$$(A^\top A + \lambda I) , \hat x = A^\top y, \qquad \lambda > 0.$$

Equivalently, $\hat x$ minimizes a penalized least-squares objective:

$$\hat x = \arg\min_x ; |Ax - y|^2 + \lambda |x|^2.$$

This is **Tikhonov regularization**. The mechanism is transparent in the singular value basis: if $A = U \Sigma V^\top$, then

$$\hat x = V , \mathrm{diag}!\left( \frac{\sigma_k}{\sigma_k^2 + \lambda} \right) U^\top y.$$

Compare with the naive pseudoinverse, which uses $\sigma_k / \sigma_k^2 = 1/\sigma_k$ — blowing up when $\sigma_k$ is small. The Tikhonov filter $\sigma_k / (\sigma_k^2 + \lambda)$ behaves like $1/\sigma_k$ when $\sigma_k \gg \sqrt{\lambda}$ (large singular values barely affected) and like $\sigma_k / \lambda \to 0$ when $\sigma_k \ll \sqrt{\lambda}$ (small singular values damped, not inverted). The parameter $\lambda$ sets the floor: components of $y$ along directions with $\sigma_k^2 \lesssim \lambda$ are discarded rather than amplified.

**That is the entire mechanism of regularization.**

Tikhonov chose $|x|^2$, which says "solutions of small magnitude are preferred." One can substitute any non-negative penalty $\Phi(x)$ encoding prior beliefs:

$$\hat x = \arg\min_x ; |Ax - y|^2 + \lambda \Phi(x).$$

The choice of $\Phi$ reflects domain knowledge, but the mechanism is unchanged: every reasonable $\Phi$ suppresses the solution's growth along directions where the data is uninformative. §2.3.5 shows that this choice is _exactly_ a choice of prior.

## 2.3 Statistical vocabulary

Everything in this section is standard. It is included because Chapters 3 and 5 use it constantly, and because the bridge in §2.3.5 is what lets the deterministic picture of §2.2 and the statistical picture of §3.3 be the same picture.

### 2.3.1 Models, estimators, likelihood

**Statistical model.** A family $\mathcal{P} = {p(y \mid \theta) : \theta \in \Theta}$ of probability distributions on the data space, indexed by a parameter $\theta$. The data $y$ is random; $\theta$ is the unknown.

**Estimator.** Any function $\hat\theta : y \mapsto \hat\theta(y) \in \Theta$. It is itself a random variable, through $y$.

**Likelihood.** Given observed data $\bar y$, the likelihood is $L(\theta) := p(y = \bar y \mid \theta)$, viewed as a function of $\theta$ with the data held fixed. **It is not a probability distribution over $\theta$.**

**Log-likelihood and NLL.** Because models factor over independent measurements, one works with $\ell(\theta) := \log L(\theta)$ — sums differentiate more easily than products, and underflow is avoided. Optimization convention is to minimize, so one defines the **negative log-likelihood** $\mathrm{NLL}(\theta) := -\ell(\theta)$. "The loss is the NLL" means this.

### 2.3.2 Maximum likelihood

$$\hat\theta_{\mathrm{MLE}} \in \arg\max_{\theta \in \Theta} L(\theta) = \arg\min_{\theta \in \Theta} \mathrm{NLL}(\theta).$$

**Consistency (informal, standard).** Under regularity — identifiability, smoothness, correct model specification — $\hat\theta_{\mathrm{MLE}} \to \theta^{\mathrm{true}}$ in probability as sample size grows.

_Hypothesis worth noting:_ this fails silently when the model is misspecified, or when $\dim\Theta$ grows with sample size — exactly the regime of imaging, where $\dim x = n$ voxels is huge and fixed while the counts are finite. This is the statistical reason regularization is not optional in tomography.

### 2.3.3 Worked case: Gaussian noise gives least squares

Let $y = A\theta + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$. Then

$$p(y \mid \theta) = (2\pi\sigma^2)^{-m/2} \exp\Big(-\tfrac{1}{2\sigma^2}|y - A\theta|_2^2\Big),$$

so $\mathrm{NLL}(\theta) = \tfrac{1}{2\sigma^2}|y - A\theta|_2^2 + \text{const}$, and **MLE = least squares**.

This is the pivot that matters. The squared $\ell_2$ loss of §2.2.4 is not a modelling choice pulled from a hat; it is the NLL of Gaussian noise. Which means: **if the noise is not Gaussian, the data-fit term should not be squared error.** PET's noise is Poisson.

### 2.3.4 Worked case: Poisson noise

Let $y_i \sim \mathrm{Poisson}(\bar y_i(x))$ independently. Then

$$\ell(x) = \sum_i \big[y_i \log \bar y_i(x) - \bar y_i(x) - \log(y_i!)\big].$$

Dropping the $\log(y_i!)$ term (constant in $x$),

$$\boxed{;\mathrm{NLL}(x) = \sum_i \big[\bar y_i(x) - y_i \log \bar y_i(x)\big].;}$$

This replaces $|Ax - y|^2$ as the data-fit term everywhere in this report.

### 2.3.5 The bridge: regularization is MAP estimation

MLE treats $\theta$ as a fixed unknown constant. Bayesian inference treats it as a random variable with its own distribution.

**Prior.** A distribution $p(\theta)$ on $\Theta$, encoding beliefs before seeing data.

**Posterior.** By Bayes' rule (a theorem of probability, not of statistics),

$$p(\theta \mid y) = \frac{p(y \mid \theta), p(\theta)}{p(y)} \propto L(\theta), p(\theta).$$

**Maximum a posteriori (MAP).** The mode of the posterior:

$$\hat\theta_{\mathrm{MAP}} \in \arg\max_\theta \big[L(\theta), p(\theta)\big] = \arg\min_\theta \big[\mathrm{NLL}(\theta) - \log p(\theta)\big].$$

So **MAP = MLE + a log-prior penalty**. When $p(\theta)$ is uniform, MAP reduces to MLE.

**The bridge.** Writing a regularized objective

$$\hat x = \arg\min_x\ \mathrm{NLL}(x) + \lambda \Phi(x)$$

is _exactly_ MAP estimation with prior $p(x) \propto e^{-\lambda \Phi(x)}$. Common correspondences:

|Penalty $\Phi(x)$|Prior|Effect|
|---|---|---|
|$\tfrac{1}{2}\|x\|_2^2$ (Tikhonov / ridge)|Gaussian on voxels|shrinks magnitude|
|$\|x\|_1$ (LASSO)|Laplace|promotes sparsity|
|$\mathrm{TV}(x)$ (total variation)|Laplace on image _gradients_|promotes piecewise-constant images|

The weight $\lambda$ controls prior strength; choosing it is itself a statistical problem (cross-validation, L-curve, discrepancy principle).

> **Caveat to have ready, not to belabour.** The identification $p(x) \propto e^{-\lambda\Phi(x)}$ is _formal_: for some $\Phi$ the right-hand side is not normalizable, i.e. the "prior" is improper. This is standard practice and harmless for point estimation, but the correspondence is a dictionary, not a theorem about well-defined probability measures.
> 
> A second caveat, worth knowing: MAP is **not invariant under reparametrization** $\theta \mapsto \phi(\theta)$ — the Jacobian shifts the mode. Posterior _means_ are reparametrization-covariant; posterior _modes_ are not. This is a genuine limitation of MAP, not a presentational quibble.

**Where this is used.** §3.5.5 (MLEM's implicit and explicit regularizers), and as the conceptual frame for why early stopping and non-negativity "count" as regularization at all.

---

# Chapter 3: The PET Inverse Problem and Reconstruction

PET reconstruction is ill-posed for two separate reasons, and the chapter addresses them in turn.

- **Analytic ill-posedness.** The forward operator smooths; the singular values of the continuous Radon transform decay to zero, so its inverse is unbounded (§2.2.2). §3.2 develops the analytical model and its inversion, filtered backprojection, and shows exactly how it fails.
- **Statistical ill-posedness.** Even a well-conditioned forward operator would leave us estimating $x$ from Poisson counts — a statistical problem, not an algebraic one. Low count rates and scatter make it worse. §3.3 develops the statistical model, §3.5 its inversion (MLEM/OSEM).

§3.4 then establishes the one structural fact about scatter that governs everything downstream: it must enter the model **additively**, not subtractively. §3.6 closes the chapter with the **Fisher-information ceiling** — the statement that even a _perfect_ scatter estimate cannot recover the information that scatter destroyed. That ceiling is what the evaluation protocol of §5.8 measures against.

## 3.1 Notation

|Object|Symbol|Notes|
|---|---|---|
|Continuous activity image|$f(\mathbf{x})$, $\mathbf{x} \in \mathbb{R}^2$|§3.2 only|
|Discrete activity image|$x \in \mathbb{R}^n_{\geq 0}$|§3.3 onward|
|Continuous forward operator (Radon)|$R$|§3.2|
|Discrete forward operator (geometric projector)|$A \in \mathbb{R}^{m \times n}_{\geq 0}$|§3.3+|
|Adjoint / backprojection|$R^\top$, $A^\top$||
|Sinogram (continuous)|$g(\theta, t)$|radial coordinate is $t$|
|Measured counts (discrete)|$y \in \mathbb{Z}^m_{\geq 0}$|"prompts"|
|Expected counts|$\bar y \in \mathbb{R}^m_{\geq 0}$|$\bar y = D_a A x + s + r$|
|Attenuation factors (diagonal)|$D_a = \mathrm{diag}(a)$, $a_i \in (0, 1]$||
|Scatter|$s \in \mathbb{R}^m_{\geq 0}$||
|Randoms|$r \in \mathbb{R}^m_{\geq 0}$|**$r = 0$ throughout this work**|
|Trues|$t \in \mathbb{Z}^m_{\geq 0}$|realized counts; $\bar t = D_a A x$|
|Regularization penalty|$\Phi(x)$||
|2D spatial frequency|$\mathbf{k} \in \mathbb{R}^2$||
|1D projection frequency|$\sigma \in \mathbb{R}$|conjugate to $t$|
|Unit direction|$\hat{\mathbf{n}}_\theta = (\cos\theta, \sin\theta)$||

## 3.2 The analytical model: Radon transform and filtered backprojection

We specialize to the idealized 2D setup: a continuous activity distribution $f: \mathbb{R}^2 \to \mathbb{R}_{\geq 0}$ observed through noise-free line integrals. This is not what PET measures — real data is noisy Poisson counts on a discrete detector — but the analytical model isolates the _geometric_ part of the inverse problem cleanly, introduces operators reused later, and exhibits the first form of PET's ill-posedness.

### 3.2.1 The Radon transform

For each angle $\theta \in [0, \pi)$ and signed radial offset $t \in \mathbb{R}$, let $\hat{\mathbf{n}}_\theta = (\cos\theta, \sin\theta)$ and let

$$L_{\theta, t} = {\mathbf{x} \in \mathbb{R}^2 :; \mathbf{x} \cdot \hat{\mathbf{n}}_\theta = t}$$

be the line perpendicular to $\hat{\mathbf{n}}_\theta$ at signed distance $t$ from the origin. The **Radon transform** $R$ maps $f$ to the collection of its line integrals:

$$(Rf)(\theta, t) = \int_{L_{\theta, t}} f , d\ell = \iint_{\mathbb{R}^2} f(\mathbf{x}) , \delta(\mathbf{x} \cdot \hat{\mathbf{n}}_\theta - t) , d\mathbf{x} =: g(\theta, t).$$

Fixing $\theta$ yields a 1D function $p_\theta(t) := (Rf)(\theta, t)$, the **projection** at angle $\theta$. Geometrically, $Rf$ is a "wheel" of 1D projections, one per viewing direction; $g(\theta, t)$ is the **sinogram**.

In the language of §2.2.2, $R$ is a compact linear operator $L^2(\mathbb{R}^2) \to L^2([0, \pi) \times \mathbb{R})$ — its singular values decay to zero, so its inverse is unbounded. **This is the analytic ill-posedness.**

### 3.2.2 Backprojection is the adjoint

The natural operator to pair with $R$ when trying to solve $g = Rf$ is its adjoint $R^\top$, characterized by $\langle Rf, g \rangle = \langle f, R^{\top} g \rangle$ for all $f, g$.

> **Derivation.** Expanding the left-hand side using the delta representation, $$\langle Rf, g \rangle = \int_0^\pi \int_\mathbb{R} \left[\iint f(\mathbf{x}), \delta(\mathbf{x} \cdot \hat{\mathbf{n}}_\theta - t), d\mathbf{x}\right] g(\theta, t) , dt , d\theta.$$ Swapping the order of integration and using the delta to collapse the $t$-integral to $t = \mathbf{x} \cdot \hat{\mathbf{n}}_\theta$, $$\langle Rf, g \rangle = \iint f(\mathbf{x}) \left[\int_0^\pi g(\theta, \mathbf{x} \cdot \hat{\mathbf{n}}_\theta) , d\theta\right] d\mathbf{x} = \langle f, R^\top g \rangle,$$ so $$(R^\top g)(\mathbf{x}) = \int_0^\pi g(\theta, \mathbf{x} \cdot \hat{\mathbf{n}}_\theta) , d\theta. \qquad \square$$

To evaluate $R^\top g$ at a point $\mathbf{x}$: for each angle $\theta$, look up the projection value on the line through $\mathbf{x}$, then integrate over angles. This is exactly **backprojection** — smear each projection back along its lines of origin, and sum over viewing directions.

$R^\top$ is not $R^{-1}$. Applied to $Rf$, it produces a _blurred_ version of $f$. The next two subsections make the blur precise and cancel it.

### 3.2.3 The Fourier slice theorem

**Lemma (Fourier slice).** For every $\theta \in [0, \pi)$ and $\sigma \in \mathbb{R}$,

$$\widehat{p_\theta}(\sigma) = \hat f(\sigma \hat{\mathbf{n}}_\theta).$$

That is: the 1D Fourier transform of the projection at angle $\theta$ equals the 2D Fourier transform of $f$ restricted to the line through the origin in direction $\hat{\mathbf{n}}_\theta$.

> **Derivation.** From the delta representation of $p_\theta$, $$\widehat{p_\theta}(\sigma) = \int p_\theta(t), e^{-i \sigma t}, dt = \iint f(\mathbf{x}) \left[ \int \delta(\mathbf{x} \cdot \hat{\mathbf{n}}_\theta - t), e^{-i \sigma t} , dt \right] d\mathbf{x} = \iint f(\mathbf{x}), e^{-i (\sigma \hat{\mathbf{n}}_\theta) \cdot \mathbf{x}}, d\mathbf{x} = \hat f(\sigma \hat{\mathbf{n}}_\theta). \qquad \square$$

Collecting projections at all angles fills in $\hat f$ along a family of lines through the origin, covering the 2D frequency plane.

**Remark (direct Fourier reconstruction).** One could invert the 2D Fourier transform at this point. That is exact, but requires interpolating $\hat f$ from polar to Cartesian sampling — numerically delicate. FBP avoids the interpolation by handling the polar-to-Cartesian conversion analytically.

### 3.2.4 Filtered backprojection

**Theorem (FBP).** Define the **ramp filter** $\Lambda$, acting on each projection independently in the radial variable, by $\widehat{\Lambda g}(\theta, \sigma) = |\sigma| , \hat g(\theta, \sigma)$. Then for sufficiently regular $f$ with $g = Rf$,

$$\boxed{;f = \frac{1}{2\pi} R^\top \Lambda g.;}$$

> **Proof.** Start from 2D Fourier inversion: $$f(\mathbf{x}) = \frac{1}{(2\pi)^2} \iint_{\mathbb{R}^2} \hat f(\mathbf{k}) , e^{i \mathbf{k} \cdot \mathbf{x}} , d\mathbf{k}.$$ Switch to polar coordinates $\mathbf{k} = \sigma \hat{\mathbf{n}}_\theta$ with $\sigma \geq 0$, $\theta \in [0, 2\pi)$, Jacobian $\sigma$: $$f(\mathbf{x}) = \frac{1}{(2\pi)^2} \int_0^{2\pi} \int_0^\infty \hat f(\sigma \hat{\mathbf{n}}_\theta) , e^{i \sigma (\hat{\mathbf{n}}_\theta \cdot \mathbf{x})} , \sigma , d\sigma , d\theta.$$ Use $\hat{\mathbf{n}}_{\theta + \pi} = -\hat{\mathbf{n}}_\theta$ to fold $\theta$ down to $[0, \pi)$, letting $\sigma$ range over $\mathbb{R}$; the Jacobian $\sigma$ becomes $|\sigma|$: $$f(\mathbf{x}) = \frac{1}{(2\pi)^2} \int_0^\pi \int_{-\infty}^\infty \hat f(\sigma \hat{\mathbf{n}}_\theta) , e^{i \sigma (\hat{\mathbf{n}}_\theta \cdot \mathbf{x})} , |\sigma| , d\sigma , d\theta.$$ Apply the Fourier slice theorem and recognize $|\sigma| \hat g(\theta, \sigma) = \widehat{\Lambda g}(\theta, \sigma)$: $$f(\mathbf{x}) = \frac{1}{2\pi} \int_0^\pi \left[\frac{1}{2\pi} \int_{-\infty}^\infty \widehat{\Lambda g}(\theta, \sigma) , e^{i \sigma (\hat{\mathbf{n}}_\theta \cdot \mathbf{x})} , d\sigma\right] d\theta.$$ The bracketed integral is the 1D inverse Fourier transform of $\Lambda g$ evaluated at $t = \mathbf{x} \cdot \hat{\mathbf{n}}_\theta$. So $$f(\mathbf{x}) = \frac{1}{2\pi} \int_0^\pi (\Lambda g)(\theta, \mathbf{x} \cdot \hat{\mathbf{n}}_\theta) , d\theta = \frac{1}{2\pi} (R^\top \Lambda g)(\mathbf{x}). \qquad \square$$

**Where the ramp comes from.** The $|\sigma|$ is the polar Jacobian. Backprojection implicitly reconstructs $f$ from a polar sampling of $\hat f$ but omits this Jacobian; $\Lambda$ puts it back. Equivalently: every projection direction contributes to $\hat f(\mathbf{0})$, but only a vanishing angular fraction contributes to any high-frequency point, so backprojection alone underweights high frequencies by exactly $1/|\mathbf{k}|$ — and the ramp filter, which boosts by $|\sigma| = |\mathbf{k}|$, cancels this precisely.

### 3.2.5 FBP as an algorithm

**General steps.**

1. For each angle $\theta$: take the 1D Fourier transform of the projection, $\widehat{p_\theta}(\sigma)$.
2. Multiply by the ramp filter $|\sigma|$ (in practice, by an apodized ramp; see §3.2.6).
3. Inverse Fourier transform to obtain the filtered projection $(\Lambda g)(\theta, \cdot)$.
4. Backproject: integrate the filtered projections over all $\theta$.

Each step is $O(N \log N)$ per projection. The formula is analytically exact for the idealized noise-free continuous problem — and only for that problem.

### 3.2.6 Why FBP fails on PET data

**Noise amplification.** The ramp filter's amplification of high frequencies is precisely what recovers sharp features in the noise-free setting — and precisely what amplifies noise catastrophically in the real setting. This is the "small singular values kill you" story of §2.2.2 in Fourier disguise: large $|\mathbf{k}|$ corresponds to small singular values of $R$, and the ramp filter is the naive $1/\sigma_k$ inversion, unregularized. Practical FBP softens the ramp with an apodizing window (Shepp–Logan, Hann, Hamming) that rolls off high frequencies — an implicit regularizer trading resolution for noise stability. This helps but does not resolve the underlying instability.

**No natural place for background.** FBP inverts $f \mapsto Rf$. But PET measurements include scatter, so the measured sinogram is, in expectation, $D_a R f + s$, not $Rf$. One could subtract an estimate $\hat s$ before FBP — but that changes the noise statistics: the corrected data is no longer Poisson, can be negative, and has inflated variance. **FBP has no mechanism to model $s$ — only to subtract it and hope.** §3.4 makes this precise and is the reason the rest of the report uses MLEM.

## 3.3 The statistical model: Poisson data in PET

§3.2 treated PET data as a deterministic function. Real PET data is nothing of the kind. We now shift from deterministic to statistical; discretization enters at the end as a computational step.

### 3.3.1 The physics of coincidence counting

Two physical facts drive the model:

- **Photon emissions are independent and rare on any given LOR.** Under weak-source conditions (which hold in clinical and preclinical PET), independent rare events accumulating over a fixed observation window are Poisson-distributed.
- **Different LORs are independent.** Coincidences on distinct LORs come from distinct annihilations and distinct detector-pair firings.

Together:

> On each LOR $L$, the observed coincidence count $y(L)$ is a Poisson random variable, and counts on distinct LORs are independent.

### 3.3.2 The Poisson forward model

In the simplest (attenuation-, scatter-free) case, the expected count is proportional to the line integral of activity:

$$y(L) \sim \mathrm{Poisson}(\bar y(L)), \quad \bar y(L) = (Rf)(L),$$

with counts on distinct LORs independent. This is the same operator $R$ as §3.2 — the geometry has not changed — but the observation is now a random function of $f$.

The log-likelihood factorizes across LORs:

$$\log p(y \mid f) = \sum_L \left[ y(L) \log \bar y(L) - \bar y(L) \right] + \text{const}.$$

By §2.3.4, this is the object that replaces $|Rf - g|^2$ as the data-fit term.

### 3.3.3 Physical corrections: attenuation and scatter

**Attenuation.** As the two photons travel outward to their detectors, each has a probability $\exp(-\int \mu , d\ell)$ of surviving, where $\mu(\mathbf{x})$ is the linear attenuation coefficient at 511 keV. Because the _product_ of the two half-line exponentials equals the exponential of the integral along the _entire_ LOR,

$$P(\text{both photons detected} \mid \text{annihilation on } L) = \exp\left(-\int_L \mu , d\ell\right) =: a(L).$$

This single factor $a(L) \in (0, 1]$ multiplies the line integral. Note the structure: attenuation depends on the _whole LOR_, not on where along it the annihilation happened. That is a peculiarity of PET (CT does not have it) and it is what makes attenuation correction tractable.

**Scatter.** A photon can Compton-scatter in the object, arriving at a detector along a direction different from its original emission. The resulting coincidence is registered on the **wrong** LOR — not the one containing the annihilation.  

**Randoms.** Two photons from unrelated annihilations arriving within the coincidence window produce a false coincidence, adding $r(L)$. _(Not present in this work — see the scope note. The term is carried through the general model because the Fisher argument of §3.6 applies to any additive background.)_

**Poisson additivity closes the model.** Both $s$ and $r$ are additive contributions to the _mean_, from photon populations independent of the trues on $L$. By §2.1.3(2), the total is still Poisson:

$$y(L) \sim \mathrm{Poisson}\left(a(L) \cdot (Rf)(L) + s(L) + r(L)\right).$$

### 3.3.4 Discretization

The image is represented on a voxel grid: $f(\mathbf{x}) \approx \sum_j x_j \phi_j(\mathbf{x})$, with $\phi_j$ an indicator (or smooth basis function) for voxel $j$, and $x \in \mathbb{R}^n_{\geq 0}$ the vector of voxel activities. LORs are indexed by detector-pair-and-ring combinations, giving $m$ bins.

The discrete counterpart of $R$ is the **system matrix** $A \in \mathbb{R}^{m \times n}_{\geq 0}$, with

$$A_{ij} = \text{length of LOR } i \text{ inside voxel } j$$

(possibly weighted by a smoother interpolation kernel). In implementation, the Joseph projector — used by `parallelproj` and by the code accompanying this report — evaluates $Ax$ by tracing each LOR through the voxel grid and accumulating length-weighted contributions.

Writing $D_a = \mathrm{diag}(a)$, the discrete forward model  is

$$\boxed{; \bar y = D_a A x + s + r, \qquad y_i \sim \mathrm{Poisson}(\bar y_i), \quad i = 1, \ldots, m. ;}$$

This factored form — geometry ($A$), attenuation ($D_a$), background ($s + r$) — is the one used in the code, and it separates concerns cleanly for §3.5.

### 3.3.5 Why analytical inversion fails, restated

Even before noise, FBP is inapplicable:

- FBP inverts $R$, not $D_a R$; the ramp filter knows nothing about attenuation.
- $s$ sits outside the range of the linear model FBP inverts; subtracting it corrupts the noise statistics (§3.2.6, §3.4).

And once noise enters, the ramp amplifies it precisely where PET data has the least signal. What is needed is an inversion that (i) takes $D_a$ and $s$ as **inputs to the forward model** rather than corrections to the data; (ii) uses the Poisson likelihood as data-fit; (iii) enforces $x \geq 0$. That procedure is MLEM.

## 3.4 Scatter must enter additively

This section is short and load-bearing. It justifies the entire architecture of the method: the network predicts $\hat s$, which is handed to MLEM as a **term in the mean**, never subtracted from the data.

### 3.4.1 The two conventions

**Additive.** Include $\hat s$ in the forward mean:

$$\bar y_i = (D_a A x)_i + \hat s_i, \qquad y_i \sim \mathrm{Poisson}(\bar y_i).$$

The likelihood is exactly the Poisson likelihood of §3.3, with $\hat s$ in the denominator of the MLEM ratio.

**Subtractive.** Precorrect the data and hope:

$$\tilde y_i = y_i - \hat s_i, \qquad \tilde y_i ; \overset{?}{\sim} ; \mathrm{Poisson}((D_a A x)_i).$$

### 3.4.2 Why subtractive is wrong

Three independent failures:

1. **$\tilde y_i$ can be negative.** Subtracting a positive estimate from a Poisson variable produces a variable with negative support. No Poisson distribution has that.
2. **The variance is wrong.** $\mathrm{Var}(\tilde y_i) = \mathrm{Var}(y_i) + \mathrm{Var}(\hat s_i) > \bar y_i$, whereas a genuine Poisson variable has variance equal to its mean. The subtraction _adds_ noise while pretending to remove signal.
3. **The likelihood is misspecified.** Feeding $\tilde y$ to a Poisson likelihood biases the estimator; the ML property is lost.

The additive model has none of these. It treats $\hat s$ as a **known constant added to the model's mean** — which is exactly what it physically is, conditional on the estimate being accurate. The Poisson likelihood remains valid, the MLEM derivation of §3.5 goes through unchanged, and $x \geq 0$ is preserved.

> **Consequence for this work.** The learned estimator's output is a _sinogram of expected scatter counts_, to be added to the mean. This is why the output layer is softplus-constrained (§5.5.3) and why the evaluation protocol reconstructs with $\hat s$ rather than subtracting it (§5.8).

## 3.5 Statistical inversion: MLEM and OSEM

### 3.5.1 The maximum likelihood principle

Given the model of §3.3.4, the natural estimator of $x$ maximizes the probability of the observed data:

$$\hat x_{\mathrm{ML}} = \arg\max_{x \geq 0} ; \ell(x), \qquad \ell(x) = \sum_{i=1}^{m} \left[ y_i \log \bar y_i - \bar y_i \right],$$

with $\bar y_i = (D_a A x)_i + s_i + r_i$ and $x$-independent constants dropped.

Two structural properties matter:

- $\ell(x)$ is **concave** in $x$: $\log$ is concave, $\bar y$ is affine in $x$, so the composition is concave. **No spurious local maxima.**
- $\ell(x)$ has **no closed-form maximizer**, so we iterate.

### 3.5.2 MLEM via a latent-variable argument

> **[STITCH]** The previous draft derived MLEM twice — once from a general EM template with an ELBO/variational-inference framing, and once directly. The general EM machinery is gone (Appendix B, cut #3); it was used exactly once and its removal costs nothing. What follows is the direct derivation, which is self-contained and rests on one Poisson fact.

The idea: introduce latent variables that would, if observed, make the problem trivial; take their conditional expectation; maximize. The whole derivation turns on **Poisson thinning** — conditioning a sum of independent Poissons on its total gives a multinomial, so each observed count can be _attributed_ to its possible sources in proportion to their expected contributions.

> **▸ Derivation.**
> 
> **Complete-data model.** By Poisson additivity (§2.1.3), decompose the count on LOR $i$ into per-voxel and background pieces. Let $z_{ij}$ be the count on LOR $i$ originating from voxel $j$, and $z_i^s, z_i^r$ the scatter and randoms contributions. Under the forward model, $$z_{ij} \sim \mathrm{Poisson}(a_i A_{ij} x_j), \qquad z_i^s \sim \mathrm{Poisson}(s_i), \qquad z_i^r \sim \mathrm{Poisson}(r_i),$$ all independent, with $y_i = \sum_j z_{ij} + z_i^s + z_i^r$. If we could observe the $z_{ij}$, ML estimation would decouple across voxels — but we cannot, so we work with their conditional expectations.
> 
> **Attribution step.** Given a current estimate $x^{(k)}$, Poisson thinning gives the proportional attribution $$\bar z_{ij}^{(k)} := \mathbb{E}[z_{ij} \mid y_i, x^{(k)}] = y_i \cdot \frac{a_i A_{ij} x_j^{(k)}}{(D_a A x^{(k)})_i + s_i + r_i} = y_i \cdot \frac{a_i A_{ij} x_j^{(k)}}{\bar y_i^{(k)}}.$$ _Intuition:_ LOR $i$'s $y_i$ counts are attributed to voxels and background **in proportion to their contributions to the current expected count**. Note that the background terms $s_i, r_i$ sit in the denominator and therefore _absorb_ their share of the counts — this is exactly the additive model of §3.4 doing its work.
> 
> **Maximization step.** With $\bar z_{ij}$ treated as observed, the complete-data log-likelihood decouples into per-voxel Poisson likelihoods. Maximizing over $x_j \geq 0$ gives $$x_j^{(k+1)} = \frac{\sum_i \bar z_{ij}^{(k)}}{\sum_i a_i A_{ij}} = \frac{x_j^{(k)}}{\sum_i a_i A_{ij}} \sum_i a_i A_{ij} \cdot \frac{y_i}{\bar y_i^{(k)}}.$$
> 
> **◂**

**The MLEM update in operator form:**

$$\boxed{x^{(k+1)} = \frac{x^{(k)}}{A^\top a} \odot A^\top\left( a \odot \frac{y}{D_a A x^{(k)} + s + r} \right) }$$

where $\odot$ is elementwise multiplication and division is elementwise.

Reading the update piece by piece:

- **Multiplicative form.** Each voxel is scaled by a correction factor, so $x^{(k)} \geq 0$ is preserved automatically, with no projection step. This is why MLEM historically dominated additive gradient methods.
- **Sensitivity denominator.** $A^\top a$ is the vector of **sensitivities**: entry $j$ is the total attenuation-weighted detection probability for voxel $j$. Voxels near the FOV edge, or behind dense tissue, have small sensitivities.
- **Ratio correction.** $y_i / \bar y_i^{(k)}$ compares measured to predicted counts on each LOR. The update backprojects _how wrong the current model is on each LOR_, then normalizes by sensitivity.
- **Guarantee.** $\ell(x^{(k+1)}) \geq \ell(x^{(k)})$: MLEM never decreases the Poisson log-likelihood.

> **Stress-test the slogan.** _"MLEM is the algorithm you get when the latent variable is: which voxel did each detected photon come from?"_ This holds because of Poisson thinning specifically. Under Gaussian noise the same latent-variable trick does **not** produce MLEM — you get iteratively reweighted least squares. **The multiplicative form is a genuine consequence of Poisson structure, not a universal feature of latent-variable methods.**

### 3.5.3 MLEM as an algorithm

**General steps.**

1. **Precompute sensitivity** $\text{sens} = A^\top a$, once. This is a backprojection of the attenuation factors.
2. **Identify low-sensitivity voxels**: those with $\text{sens}_j$ below a fixed fraction of the peak. Hold them at zero for the entire run (see the note below).
3. **Initialize** $x^{(0)}$ uniform over the supported voxels.
4. **Iterate** $K$ times: a. Forward project and add background: $\bar y = a \odot (A x) + s + r$. b. Form the ratio $y / \bar y$ elementwise. c. Backproject the attenuation-weighted ratio: $c = A^\top(a \odot (y/\bar y))$. d. Update: $x \leftarrow x \odot c ,/, \text{sens}$, holding floored voxels at zero.
5. **Return** $x^{(K)}$.

```
[SPEC — YOU FILL]  mlem_reconstruct

Inputs
  y        : measured prompt sinogram, shape (m,)          [physical, fixed]
  s_hat    : scatter estimate, shape (m,)                  [physical; = 0 for the "floor" run]
  a        : attenuation factors, shape (m,), in (0,1]     [physical, fixed]
  A        : forward projector (callable + adjoint)        [geometry, fixed]
  K              : iteration count                          <-- KNOB (regularizer! see 3.5.5)
  sens_floor_frac: sensitivity threshold as frac. of peak   <-- KNOB
  x0             : initial image                            <-- KNOB (default: uniform)

Output
  x        : reconstructed activity, shape (n,), x >= 0

Invariants that must hold (check these against your code)
  - x >= 0 at every iteration, with no explicit clipping
  - floored voxels remain exactly 0 for all K iterations
  - s_hat enters ONLY in the denominator of the ratio, never subtracted from y
  - r is absent / zero in this pipeline

  <internal logic: fill from memory, then diff against code>
```

**Two notes tying this to the code.**

- **Sensitivity floor.** Voxels with sensitivity below a fraction of the peak (e.g. 2.5%) are excluded and held at zero. This is _implicit regularization_, not a numerical hack: the update divides by `sens`, so near-zero sensitivity would amplify backprojected noise into hot boundary artifacts. §3.5.5 explains why it counts as regularization.
- **Cost per iteration.** Dominated by one forward projection and one backprojection, both $O(mn)$ worst-case but exploiting sparsity of $A$ in practice.

### 3.5.4 OSEM

MLEM's monotone ascent is slow. **Ordered-Subset EM** (Hudson & Larkin, 1994) partitions the LOR indices into $B$ disjoint subsets $S_1, \ldots, S_B$ and performs one MLEM-style update _per subset_, cycling through all subsets to complete one "OSEM iteration":

$$x^{(k, b+1)} = \frac{x^{(k, b)}}{A_{S_b}^\top a_{S_b}} \odot A_{S_b}^\top!\left( a_{S_b} \odot \frac{y_{S_b}}{(D_a A x^{(k, b)})_{S_b} + s_{S_b} + r_{S_b}} \right).$$

Each subset gives a noisier but cheaper estimate of the full gradient, so early progress is roughly $B$ times faster.

**Trade-off.** OSEM does **not** converge to the ML estimate; it enters a limit cycle among the subset solutions. In practice this is tolerated because one stops early anyway (typically 2–4 iterations with $B \approx 20$–30 subsets), well before the cycle becomes visible. Early stopping is doing real work here — see next.

### 3.5.5 Regularization in MLEM

By §2.3.5, regularization is a prior. But MLEM as stated has **no explicit penalty** — so in what sense is it regularized at all? The answer is that four _implicit_ regularizers are already operating, each restricting the solution to a set where small singular values cannot do damage (§2.2.4).

**Non-negativity.** The multiplicative update enforces $x^{(k)} \geq 0$ automatically. This excludes negative-valued directions in image space where the null space of $A$ typically lives, restoring a form of uniqueness the raw problem lacks.

**The Poisson likelihood as data-adaptive weighting.** Gaussian least squares weights all LORs uniformly. The Poisson log-likelihood effectively weights each LOR by roughly $1/\bar y_i$ — low-count LORs contribute less to the gradient. Noisy measurements have diminished influence.

**Early stopping.** MLEM iterates start smooth and become progressively noisier as they climb toward the ML estimate — which is itself noise-dominated (§2.3.2: $\dim\Theta$ does not shrink as counts accumulate). Stopping after a few iterations is the standard clinical regularizer, and its mechanism is exactly §2.2.4's: the low-singular-value directions are the _last_ ones the iterate learns, so stopping early leaves them untouched. **The iteration count $K$ is a regularization parameter, not a convergence tolerance.**

**Sensitivity flooring.** Voxels with $(A^\top a)_j$ below threshold are unidentifiable — their update is numerical noise divided by numerical noise. Pinning them to zero removes ill-defined degrees of freedom. This is a restriction of the solution set, i.e. regularization by §2.2.4's mechanism, with no explicit $\Phi$.

**Explicit penalization (not used here).** For more control one adds a penalty:

$$\hat x = \arg\max_{x \geq 0} ; \ell(x) - \lambda \Phi(x),$$

which is **penalized-likelihood** or **MAP** reconstruction — exactly §2.3.5 with Poisson data-fit. Choices of $\Phi$ include quadratic smoothness, total variation, or anatomical priors from co-registered CT/MRI. Algorithms include OSL-EM, BSREM, and preconditioned gradient methods. **This report does not use explicit penalization**; it is mentioned because it is the natural next axis and because it makes the implicit/explicit distinction above concrete.

### 3.5.6 Convergence, stopping, scale

- **Convergence.** Under mild conditions MLEM converges to the (generally non-unique) set of ML maximizers. Convergence is slow and has a _bulk-then-detail_ character: large-scale intensity is recovered in the first few iterations; edges and peaks sharpen over tens.
- **Choice of $K$.** In the absence of an explicit criterion, $K$ _is_ the regularization parameter (§3.5.5).
- **Global scale.** Without a solid-angle/efficiency model, MLEM recovers $x$ only up to a global multiplicative constant. Quantitative comparison across reconstructions therefore requires an explicit scale-matching step (`scale_match` in the code). **This matters for §5.8**: the floor/oracle/method bracket compares three reconstructions, and all three must be on a common scale before their errors mean anything.

## 3.6 The Fisher-information ceiling

This section establishes the result that the entire evaluation protocol is read against. The claim is narrow and provable in half a page: **additive background destroys information about $x$ that no amount of clever correction can recover.**

### 3.6.1 What Fisher information is

For a statistical model $p(y \mid x)$ with smooth dependence on $x$, the **Fisher information matrix** at $x$ is

$$I(x) = \mathbb{E}!\left[ \nabla_x \log p(y \mid x) , \nabla_x \log p(y \mid x)^\top \right].$$

Read this literally: take the log-likelihood, differentiate with respect to $x$ (this gives the _score_, a random vector because $y$ is random), and take its covariance. That is the whole definition.

**Cramér–Rao bound (classical; stated as a fact, not proved here).** For any _unbiased_ estimator $\hat x$,

$$\mathrm{Cov}(\hat x) \succeq I(x)^{-1}$$

in the Loewner (positive-semidefinite) order. Larger Fisher information means tighter achievable estimation.

> **Stress-test.** The CRLB is for **unbiased** estimators. Regularized reconstructions — MAP, penalized-likelihood, early-stopped MLEM — are _deliberately biased_, and routinely beat the CRLB in mean-squared error. That is the entire point of regularization. So the ceiling below is a statement about the _information in the measurement_, not a hard floor on any particular estimator's MSE. It is used here in the first sense.

### 3.6.2 The ceiling

> **▸ Derivation.**
> 
> For the Poisson model $y_i \sim \mathrm{Poisson}(\bar y_i)$ with $\bar y_i = (D_a A x)_i + s_i + r_i$, the log-likelihood is $$\log p(y \mid x) = \sum_i \left[ y_i \log \bar y_i - \bar y_i \right] + \text{const}.$$ Differentiating with respect to $x_j$, using $\partial \bar y_i / \partial x_j = a_i A_{ij}$: $$\frac{\partial \log p}{\partial x_j} = \sum_i a_i A_{ij} \left( \frac{y_i}{\bar y_i} - 1 \right).$$ Since $\mathbb{E}[y_i] = \bar y_i$ and $\mathrm{Var}(y_i) = \bar y_i$, and the LORs are independent, $$I(x)_{jk} = \sum_i a_i A_{ij} \cdot a_i A_{ik} \cdot \frac{\mathrm{Var}(y_i)}{\bar y_i^2} = \sum_i \frac{a_i^2 A_{ij} A_{ik}}{\bar y_i}.$$ In matrix form, $$I(x) = (D_a A)^\top D_{1/\bar y} (D_a A), \qquad D_{1/\bar y} = \mathrm{diag}(1/\bar y_i).$$
> 
> **◂**

Compare with the scatter-free case, where $\bar y_i^{\text{clean}} = (D_a A x)_i$:

$$I_{\text{clean}}(x) = (D_a A)^\top D_{1/\bar y^{\text{clean}}} (D_a A).$$

Since $\bar y_i = \bar y_i^{\text{clean}} + s_i + r_i \geq \bar y_i^{\text{clean}} > 0$, we have $1/\bar y_i \leq 1/\bar y_i^{\text{clean}}$ entrywise, so $D_{1/\bar y} \preceq D_{1/\bar y^{\text{clean}}}$, and therefore

$$\boxed{; I(x) \preceq I_{\text{clean}}(x), ;}$$

with equality iff $s = r = 0$.

### 3.6.3 What it means

The mechanism in one sentence: **each measurement's signal-to-noise ratio for estimating $x$ is $(\partial \bar y_i / \partial x) / \sqrt{\bar y_i}$; adding scatter to the mean increases the denominator (Poisson variance = mean) without increasing the numerator (scatter does not depend on $x$, once $\hat s$ is treated as known). So each measurement carries less information.**

Two consequences, both used later.

1. **Perfect scatter correction does not restore scatter-free performance.** Setting $\hat s = s$ exactly uses the available information _optimally_ — it does not create information. The gap between a scatter-corrected reconstruction and a scatter-free reference is bounded below by the difference between $I(x)^{-1}$ and $I_{\text{clean}}(x)^{-1}$. Any reconstruction achieving that gap is optimal in the Fisher sense.
    
2. **This defines an operational ceiling.** The best any scatter estimator can do is match the reconstruction obtained with the true $s$. That reconstruction — MLEM run with $\hat s = s^{\text{MC}}$ — is the **oracle** of the evaluation bracket in §5.8. The Fisher inequality is what guarantees the oracle is a genuine ceiling and not merely a strong baseline.
    

> **Caveat on how far this goes.** The inequality is exact and assumption-light. But it bounds _information_, not any specific estimator's error, and it says nothing about _how close_ a given method gets. It cannot, by itself, tell you whether a 3% gap to oracle is good. That is why §5.8's gain metric is a _fraction of oracle gain captured_, not a distance to the Fisher bound: the former is measurable, the latter is not without knowing the estimator's bias.

---

# Chapter 4: Scatter Estimation Methods

## 4.1 What a scatter estimator does

Chapter 3 fixed the shape of the problem: the forward model is $\bar y = D_a A x + s + r$, and MLEM inverts it using $\hat s$ as a known additive constant in the denominator of its update (§3.4, §3.5). This chapter is about how $\hat s$ gets produced.

Formally, a **scatter estimator** is a map $\mathcal{E}$ taking some subset of the available inputs and returning a nonnegative sinogram $\hat s \in \mathbb{R}^m_{\geq 0}$:

$$\hat s = \mathcal{E}\bigl(\underbrace{y}_{\text{prompts}}, ; \underbrace{\mu}_{\text{atten. map}}, ; \underbrace{\hat f}_{\text{first-pass activity}}, ; \underbrace{\text{geom}}_{\text{scanner}}\bigr).$$

Not every estimator uses every input, and **the choice of input set is the first distinguishing feature**. A convolution method may use only $y$ and a kernel. SSS uses $\mu$, $\hat f$, and geometry. Monte Carlo uses all four. A learned method may use any subset the designer chooses — and the choice made here (prompts only) is the sharpest claim of this work.

The second feature is **physics content**. Convolution methods encode almost none — a kernel is a shape, not a mechanism. SSS encodes single Compton scatter analytically. Monte Carlo encodes everything simulable. Learned methods encode whatever their training data implicitly contains, which for MC-supervised networks can in principle include everything MC does.

The third is the **speed–generalization tradeoff**. An estimator built on explicit physics generalizes _by construction_ — it applies to any object for which its inputs are meaningful. A learned estimator is fast at inference but generalizes only over its training distribution. This tradeoff is the axis along which the field currently negotiates, and §4.5.5 argues it is a _qualitative_ distinction, not a quantitative one.

The families covered here span:

||**Deterministic**|**Sampled / Learned**|
|---|---|---|
|**Approximate physics**|Convolution methods (§4.3.1)|—|
|**Full physics (in principle)**|SSS (§4.3.2)|Monte Carlo (§4.4); learned (§4.5)|

The empty cell reflects a real fact: no one has built a "sampled convolution method," because there would be no reason to.

> **Slogan.** _Every scatter estimator is a choice about which physics you can afford to compute explicitly, and which you're willing to trust to a shortcut._
> 
> _Stress-test:_ does this hold for pure statistical methods (energy-window subtraction)? Only partially — those are shortcuts with **no** physics content, occupying a column further left than the table shows. They are mentioned in §4.3.1 but do not merit standalone treatment here.

## 4.2 Physical ingredients

Every scatter estimator knows or approximates a small collection of physical quantities. This section develops them once, so §4.3–§4.5 can refer back rather than re-derive.

### 4.2.1 Compton kinematics

**(Standard, textbook.)** A photon of energy $E_0$ scattering off a free electron at rest emerges with reduced energy $E'$ and deflected direction. Conservation of energy and momentum gives

$$E'(\theta) = \frac{E_0}{1 + (E_0 / m_e c^2)(1 - \cos\theta)}$$

where $\theta$ is the scattering angle and $m_e c^2 = 511$ keV is the electron rest energy.

For annihilation photons, $E_0 = m_e c^2$ **exactly**, so the ratio is 1 and the formula collapses to

$$\boxed{E'(\theta) = \frac{511 \text{ keV}}{2 - \cos\theta}.}$$

Three reference angles: $\theta = 0°$ gives $E' = 511$ keV (grazing, no loss); $\theta = 90°$ gives $E' \approx 256$ keV; $\theta = 180°$ gives $E' \approx 170$ keV (backscatter, maximum loss).

**A singly-scattered 511 keV photon can never emerge below ~170 keV.** This bound matters for any estimator modelling energy-window rejection.

_(For 511 keV photons in soft tissue, Compton dominates all other photon interactions by roughly two orders of magnitude — so "scatter" in PET essentially **means** Compton.)_

### 4.2.2 Klein–Nishina differential cross-section

Kinematics says _what happens_ if scattering occurs; it does not say _how likely_ each angle is. That is the Klein–Nishina cross-section.

**(Standard; derived from QED.)** For a photon of energy $E$ scattering off a free electron,

$$\frac{d\sigma_{KN}}{d\Omega}(\theta, E) = \frac{r_e^2}{2} \left(\frac{E'}{E}\right)^2 \left(\frac{E}{E'} + \frac{E'}{E} - \sin^2\theta\right),$$

where $r_e = 2.82 \times 10^{-15}$ m is the classical electron radius and $E'/E = [1 + \varepsilon(1-\cos\theta)]^{-1}$ with $\varepsilon = E/m_ec^2$.

**Two limits.**

- **Thomson ($E \to 0$, $\varepsilon \to 0$):** $E'/E \to 1$ for all angles, and the formula reduces to $(r_e^2/2)(1 + \cos^2\theta)$ — forward–backward symmetric under $\theta \leftrightarrow \pi - \theta$. Physically: the photon barely recoils the electron, so the pattern is classical dipole radiation.
- **511 keV ($\varepsilon = 1$):** now $E'/E$ depends strongly on angle — forward keeps $E'/E \approx 1$, backscatter gives $E'/E = 1/3$. The $(E'/E)^2$ prefactor equals 1 at $\theta = 0$ and drops to $1/9$ at $\theta = \pi$, **breaking the symmetry toward the forward direction.** The suppression grows with energy.

The result is **forward-peaked scattering**, with two consequences used throughout:

1. Most singly-scattered photons carry near-511-keV energy and are consequently hard to reject by energy discrimination.
2. Multiple-scatter tails, formed by chains of small-angle events, are smoother and broader than single-scatter events — a fact that both SSS's tail-fit and the learned method's smoothness prior exploit.

### 4.2.3 Energy window and detector resolution

PET scanners accept coincidences in which both detected photons have measured energy in $[E_\text{lo}, E_\text{hi}]$. Typical LYSO scanners use $E_\text{lo} \approx 400$–450 keV, $E_\text{hi} \approx 650$ keV.

**Angular consequence.** Setting $E'(\theta_\text{max}) = E_\text{lo}$ and inverting §4.2.1:

$$\cos\theta_\text{max} = 2 - \frac{511}{E_\text{lo}}.$$

For $E_\text{lo} = 350$ keV, $\theta_\text{max} \approx 57°$; for $E_\text{lo} = 450$ keV, $\theta_\text{max} \approx 41°$. Photons scattered beyond $\theta_\text{max}$ are rejected outright. **The surviving scatter is therefore small-angle, low-energy-loss scatter** — which is why the scatter sinogram is smooth (§4.5.4, Bet 1).

**Detector resolution.** The measured energy is $E'$ blurred by the detector's Gaussian energy resolution $\sigma(E)$. The probability that a photon of true energy $E'$ is measured inside the window is

$$W(E') = \Phi!\left(\frac{E_\text{hi} - E'}{\sigma(E')}\right) - \Phi!\left(\frac{E_\text{lo} - E'}{\sigma(E')}\right),$$

with $\Phi$ the standard normal CDF. This softens the angular cutoff by a few degrees.

> **Implication — a named failure mode.** Any estimator that predicts scatter shape must know the window and the resolution model. **A network trained at one setting will mispredict at another**, and this is an out-of-distribution failure with a specific, identifiable physical cause. In this work the window and resolution are held fixed across every run (§5.2.3, Invariant 4), which is what makes the training distribution well-defined — and also what bounds the claims.

### 4.2.4 Electron density and attenuation

The rate at which photons of energy $E$ interact per unit path length is the **linear attenuation coefficient**:

$$\mu(E) = n_e , \sigma(E), \qquad n_e = \rho \cdot (Z/A) \cdot N_A,$$

with $\rho$ the mass density, $Z/A$ the atomic-number-to-mass-number ratio (≈ 1/2 for light tissue elements), and $N_A$ Avogadro's number. Here $\sigma(E)$ is the _total_ Klein–Nishina cross-section, obtained by integrating §4.2.2 over solid angle.

Its reciprocal $1/\mu$ is the **mean free path**. For 511 keV photons in soft tissue, $1/\mu \approx 10$ cm — comparable to the body itself. **That single number is why scatter is unavoidable in PET.**

> **A note on terminology.** In "good geometry" experiments (narrow beam, small detector, only unscattered photons counted), $\mu$ is called _attenuation_ because scattered photons are lost to the beam. In PET this framing is misleading — the "lost" photons often reappear on other LORs as scatter. The physical quantity is the same; the interpretation is not. Throughout this report, $\mu(E_0)$ denotes the **interaction rate**, whether or not the interaction removes the photon from the data.

**Two roles in scatter estimation.**

1. **Survival probability** $\exp(-\int \mu , d\ell)$ along photon paths — the fraction reaching a detector unscattered.
2. **Electron density proxy**: at 511 keV Compton dominates, so $\mu(E_0) \propto n_e$ to good approximation. The rate at which scatters happen inside a voxel is proportional to $n_e$.

Both roles appear in Watson's SSS formula (§4.3.2). **In a learned method that does not receive $\mu$, both must be inferred implicitly from the prompt sinogram** — this is the crux of §4.5.4.

### 4.2.5 Which method uses what

|Ingredient|Convolution|SSS|Monte Carlo|Learned|
|---|---|---|---|---|
|Compton kinematics|implicit (kernel)|analytical|sampled|implicit|
|Klein–Nishina|implicit|analytical|sampled|implicit|
|Energy window $W(E')$|not modelled|analytical|applied per photon|implicit|
|Attenuation $\mu(E_0)$|not modelled|required input|required input|★ _inferred from_ $y$|
|Electron density $n_e$|not modelled|derived from $\mu$|derived from $\mu$|★ _inferred from_ $y$|
|Multiple scatter|absorbed in kernel|absorbed in tail-fit|native|learnable from MC|

**The two starred entries are this work's specific bet:** a prompt-only learned method must recover the object's attenuation structure from the prompt sinogram alone.

## 4.3 Analytical methods

### 4.3.1 Convolution methods

**Physical picture.** If scatter is dominated by small-angle events and multiple-scattering paths are gently curved, the scatter contribution to a given LOR is approximately a smooth blur of the trues on nearby LORs:

$$s_i \approx (k * D_a A x)_i,$$

where $k$ is a scatter kernel — usually mono- or bi-exponential in radial offset — and $*$ denotes convolution along the sinogram's radial axis.

**History.** Bergstrom et al. (1983) introduced the fixed-kernel form. Bailey and Meikle (1994) extended it with _position-dependent_ kernels, parameterized by local object thickness inferred from the attenuation scan. Later refinements added energy-window dependence and iterated the convolution with the activity estimate.

**Why it works, when it works.** Convolution methods succeed for objects resembling the calibration phantom: roughly circular cross-sections, uniform attenuation, single-scatter-dominated regimes. They fail in three specific ways:

1. Non-uniform attenuation (lung/soft-tissue boundaries) violates the assumption that one kernel describes the local scatter physics.
2. Out-of-FOV activity contributes scatter from an unknown direction — the convolution model has no vocabulary for this.
3. Small-object regimes (small-animal PET, i.e. **this setting**) have short scatter tails and non-negligible edge effects, precisely where the kernel's asymptotic exponential shape is least appropriate.

Convolution methods have largely been displaced by SSS. They remain interesting as the _simplest_ physics-based model and as a baseline any learned method must beat with room to spare.

### 4.3.2 Single scatter simulation

Watson (1996) and Ollinger (1996) independently developed what became the clinical standard for two decades: an analytical estimate of the _singly-scattered_ coincidence rate on every detector pair, integrated over scatter points in the object. This section derives Watson's formula so that the physical content of every factor is visible — the alternative, quoting the formula, would leave the sign trap of step (iii) invisible.

#### Physical setup

Fix two detectors, $A$ and $B$. A **singly-scattered coincidence** on the pair $(A, B)$ is a detected pair of photons from a single annihilation in which one photon was detected without scattering (the _unscattered leg_) and the other underwent exactly one Compton scattering at some point $S$ (the _scattered leg_).

There are two mutually exclusive ways this happens:

- **Story 1:** annihilation on the line $[A, S]$; one photon travels to $A$ unscattered; the twin travels to $S$ and Compton-scatters into $B$.
- **Story 2:** the mirror — annihilation on $[B, S]$; the twin scatters at $S$ into $A$.

Both stories put a coincidence on the _detected_ LOR $(A, B)$, **even though neither annihilation point lies on that line.** This mispositioning is the entire reason scatter must be corrected. A true coincidence lies on the straight line joining the two detectors; a singly-scattered one follows a bent "dog-leg" path.

#### Building the integrand

For a scatter point $S$ in a volume element $dV$, assemble the Story-1 contribution factor by factor.

**(i) Emission along the unscattered leg.** With $\lambda(\mathbf{x})$ the activity density, the expected number of annihilations per unit time on $[A, S]$, each producing one photon toward $A$, is proportional to

$$I_{A,S} = \int_{[A, S]} \lambda(\mathbf{x}) , d\ell.$$

**(ii) Survival of the unscattered leg at 511 keV.**

$$T_{A,S}(E_0) = \exp!\left(-\int_{[A, S]} \mu(\mathbf{x}, E_0) , d\ell\right).$$

_(This uses the standard Watson simplification of replacing the point-dependent survival by survival along the full segment; the approximation is tight because activity in a thin segment sees a near-constant surviving fraction to $A$.)_

**(iii) Compton probability at $S$.** The probability per unit volume per unit solid angle that a photon arriving at $S$ scatters into $d\Omega$ toward $B$ is

$$n_e(S) \cdot \frac{d\sigma_{KN}}{d\Omega}(\theta_{ASB}, E_0) \cdot dV.$$

Write $\hat u_A$ for the unit vector from $S$ toward $A$, and $\hat u_B$ from $S$ toward $B$. The pre-scatter photon travels along $-\hat u_A$ (through $S$, away from the annihilation on $[A,S]$); the post-scatter photon travels along $+\hat u_B$. Hence

$$\cos\theta_{ASB} = (-\hat u_A) \cdot \hat u_B = -,\hat u_A \cdot \hat u_B.$$

> **The sign trap.** The minus sign is easy to miss. Check it against the straight-through configuration: $\hat u_B = -\hat u_A$ gives $\cos\theta = +1$, i.e. $\theta = 0$ — correct, no scatter needed. **A missing minus sign silently flips forward-peaked scatter into back-peaked**, poisoning every downstream result while producing a plausible-looking sinogram. The code tests this configuration explicitly (§4.3.3).

**(iv) Survival of the scattered leg at $E'$.** After scattering, the photon has energy $E' = E'(\theta_{ASB})$ from §4.2.1 and must survive $[S, B]$:

$$T_{B,S}(E') = \exp!\left(-\int_{[S, B]} \mu(\mathbf{x}, E') , d\ell\right).$$

Note $\mu$ is evaluated at $E'$, **not** $E_0$: the survival cross-section depends on the photon's _current_ energy.

**(v) Energy-window acceptance at $E'$.** The factor $W(E')$ from §4.2.3.

**(vi) Detection geometry.** Absorbing detector areas into a scanner-dependent constant (fitted afterward):

$$G_{AB}(S) = \frac{\cos\alpha_A \cos\alpha_B}{R_{AS}^2 , R_{BS}^2},$$

with $\alpha_d$ the incidence angle at detector $d$ and $R_{dS}$ the $S$-to-$d$ distance.

**Story 1 assembled.**

$$dS_{AB}^{(1)} = n_e(S) \cdot \frac{d\sigma_{KN}}{d\Omega} \cdot W(E') \cdot G_{AB}(S) \cdot I_{A,S} \cdot T_{A,S}(E_0) \cdot T_{B,S}(E') \cdot dV.$$

**Story 2** is the mirror ($A \leftrightarrow B$).

**Watson's formula.** Summing stories and integrating over scatter points:

$$\boxed{s_{AB} = \int_{V_S} n_e(S)\, \frac{d\sigma_{KN}}{d\Omega}(\theta_{ASB})\, W(E')\, G_{AB}(S) \, \bigl[I_{A,S}\, T_{A,S}(E_0)\, T_{B,S}(E') + I_{B,S}\, T_{B,S}(E_0)\, T_{A,S}(E')\bigr] \, dV.}$$

#### Practical issues

**Absolute scale.** The derivation dropped several constants (the $r_e^2/2$ from K–N, detector efficiencies, solid-angle constants folded into $G_{AB}$). In practice the _shape_ of the SSS estimate is trusted but its magnitude is not; a global scalar $c$ is fitted afterward: $\hat s_{AB} = c \cdot s_{AB}^\text{SSS}$.

Two fitting strategies:

- **Tail-fitting.** Identify LORs whose lines miss the object entirely (the _tails_). There, trues $\approx 0$ and prompts $\approx$ pure scatter, so $c$ is fit by least squares on the tail region. Deployment-realistic.
- **Oracle fitting.** If MC ground truth is available, fit $c$ against it directly. This isolates _shape_ error from _scale_ error, and is used here as a **diagnostic**, not as a deployable method.

> **A failure mode specific to this setting.** Tail-fitting breaks down in **small-animal PET**: the bore is small, the object fills most of it, and few LORs miss the object. When the tail is thin, the fit is noisy. The code detects this and reports it rather than silently returning a bad scale. This is why §5.8's optional SSS baseline requires a per-experiment decision between tail-fit and oracle-scaled SSS.

**Iteration with the activity estimate.** SSS needs $\hat\lambda$ (for $I_{A,S}$, $I_{B,S}$), but $\hat\lambda$ needs scatter correction. One iterates: reconstruct without correction → coarse $\hat\lambda_0$; run SSS → $\hat s_1$; reconstruct with $\hat s_1$ → $\hat\lambda_1$; repeat. Convergence is fast (2–3 iterations). This is a fixed-point iteration for the coupled activity–scatter problem.

**μ-map dependency.** SSS requires $\mu$ at 511 keV _and_ at each $E'$ in the scattering range. In PET/CT $\mu$ comes from the CT; in PET/MR it must be synthesized, itself an open problem (Hofmann et al. 2008 onward). **When $\mu$ is inaccurate, SSS is systematically biased in ways hard to detect without ground truth** — a direct motivation for methods that do not require $\mu$ at all (§4.5).

**Multiple scatter.** Watson's formula is single-scatter by construction. Multiple-scatter events contribute a smoother, broader background absorbed into $c$. This is a controlled approximation (multiple scatter is 15–30% of total scatter in clinical PET, less in preclinical), but it means **SSS's absolute-scale story is "single scatter plus a fudge factor for the rest."**

### 4.3.3 SSS implementation

The SSS implementation used here is a scanner-agnostic Python module `[TK: appendix reference]`. This section describes its design at the level of ideas; the code is the specification of record.

**General steps** — six stages, matching the physical derivation:

1. **Material tables.** Parse PENELOPE-format cross-section files (the same files MCGPU-PET uses), producing $\mu(E)$ and electron density per material (§4.2.4).
2. **Geometry.** Build a coarse subsampled detector array (every 8th crystal, ~15 rings) and a lattice of candidate scatter points $S$ inside the object.
3. **Ray-integral tables.** Precompute, for every (detector, scatter-point) pair, the emission integral $I$, the 511-keV survival $T_0$, and the survival at each $E'$ on a small energy grid.
4. **Assembly.** Iterate over scatter points, evaluate the Watson integrand, accumulate $s_{AB}$ over coarse detector pairs.
5. **Interpolation.** Upsample the coarse pair matrix to the full sinogram, in the exact bin order the scanner's data uses.
6. **Scale fitting.** Fit $c$ from MC truth (oracle) or from object-missing tails.

```
[SPEC — YOU FILL]  sss_estimate

Inputs
  activity_map   : first-pass activity estimate            [required]
  mu_map         : attenuation map at E_0                  [required]
  material_table : PENELOPE cross-sections -> mu(E), n_e   [required]
  geometry       : scanner config                          [fixed]
  detector_stride: coarse subsample factor (~8)             <-- KNOB
  n_rings_coarse : coarse ring count (~15)                  <-- KNOB
  scatter_pt_grid: lattice spacing for S                    <-- KNOB
  energy_grid    : E' sample points for survival tables     <-- KNOB
  scale_mode     : {tail_fit, oracle}                       <-- KNOB (diagnostic choice)

Output
  s_hat : scatter sinogram, full resolution, bin order matching the scanner

Invariants that must hold (check these against your code)
  - cos(theta_ASB) = -dot(u_A, u_B)          <-- the sign trap; test straight-through gives theta=0
  - mu evaluated at E' on the scattered leg, at E_0 on the unscattered leg
  - coarse pair matrix is cached; full sinogram is NOT cached
  - tail_fit reports a diagnostic when the tail region is too thin to fit reliably

  <internal logic: fill from memory, then diff against code>
```

**Two structural decisions worth highlighting.**

_Coarse detectors, then interpolate._ Computing $s_{AB}$ on every detector pair costs $O(n_D^2)$ per scatter point — prohibitive for $n_D \sim 10^4$. But the scatter surface is smooth in detector-index space (small-angle dominated, §4.2.3), so subsampling detectors by ~8 and interpolating up introduces error much smaller than the model's other approximations. The full sinogram is then a _deterministic function_ of the coarse pair matrix — which is exactly what the caching layer exploits.

_Array-namespace-generic._ The module dispatches through an `xp` argument (`numpy` or `array_api_compat.cupy`), so the same code runs on CPU and GPU unmodified. On small-animal geometries the CPU path takes ~2 minutes; the GPU path ~5 seconds. Zero maintenance cost, portable across hardware.

**Caching strategy.** The coarse pair matrix (~few MB) is cached to disk, keyed by a hash of config + voxel grid + activity map + knobs. The _full_ sinogram (~700 MB) is deliberately **not** cached — it is cheap to re-expand, and caching only the coarse level eliminates the risk of a stale full-resolution array outliving its coarse source.

## 4.4 Monte Carlo: the reference

Monte Carlo simulates photon transport directly — tracking each annihilation from birth to detection and applying every interaction (Compton, Rayleigh, photoelectric, positron range, energy blurring) with correct probabilities. Given enough samples, MC produces the true expected scatter sinogram with only statistical error, capturing **all orders of scatter, all detector effects, and out-of-FOV activity** — everything SSS approximates or ignores.

**Why MC is not the standard clinical estimator.** Cost. A clinical scan has ~$10^9$ coincidences; sufficient-sample-density transport historically took hours per scan on CPUs. SSS became standard because it is minutes-to-seconds fast at comparable accuracy on the singly-scattered majority.

**GPU MC as enabling technology.** Photon transport is embarrassingly parallel — each photon is independent — and modern GPU implementations (MCGPU-PET, Badal & Sempau; GATE-RTion; ~2015 onward) close the speed gap. MCGPU-PET produces separated trues and scatter sinograms in tens of seconds on a single consumer GPU for a small-animal geometry.

> **This is what makes MC-supervised deep learning tractable at all.** Without GPU MC, generating a training set of thousands of phantoms with per-event scatter labels would be a multi-CPU-year problem. The method of Chapter 5 is not a clever idea that happened to become possible; it is a _consequence_ of MC becoming cheap.

**MC's two roles here.** MC is **not** compared against SSS or the learned method as an estimator in its own right. It serves as:

- **Ground-truth label generator.** For each simulated phantom, MCGPU-PET emits the separated scatter sinogram $s^\text{MC}$ — the training target.
- **Reference for scatter-corrected reconstruction.** The empirical ceiling is MLEM run with $\hat s = s^\text{MC}$: the reconstruction one would obtain with a perfect scatter estimate. **This is §3.6's Fisher ceiling in operational form.**

Details of the MCGPU-PET setup appear in Chapter 5.

## 4.5 Learned scatter estimation

The idea that a neural network could estimate PET scatter directly from the prompt sinogram entered the literature around 2018. The proposition was straightforward: scatter is a smooth, low-frequency sinogram; smooth functions of images are what convolutional networks approximate cheaply; the labels can come from MC. What was less obvious — and remains contested — is **under what conditions such a network can substitute for physical modelling in a _quantitative_ imaging context.** That is the question this report is about.

### 4.5.1 A brief history

**Berker et al. (2018)** gave the earliest well-cited demonstration: a 2D U-Net trained on MC-simulated brain PET data to predict scatter sinograms from prompts. RMSE against MC ground truth was competitive with SSS; inference was seconds per scan.

**Yang et al. (2019, 2020)** extended to whole-body PET/CT and studied the role of $\mu$ as auxiliary input. Adding $\mu$ improved accuracy, but **less than expected** — the prompt sinogram alone already captured much of the object structure. This is the main empirical anchor for the prompt-only design pursued here.

**Qian et al. (2020, 2022)** introduced transformer architectures and multi-scale supervision, arguing that long-range dependencies in the sinogram benefit from attention rather than pure convolution.

**Häggström et al. (2021)** and others (see Reader et al. 2021 for a review) explored image-domain networks operating on a first-pass reconstruction, arguing that image-domain features are more anatomically interpretable.

**Recent directions (2023 onward)** include unrolled reconstruction networks folding scatter estimation into MLEM iterates, and self-supervised approaches avoiding MC labels via consistency with the acquired data.

The field is roughly five years old, methodologically diverse, and — despite competitive benchmarks — **has not displaced SSS in clinical software.** The gap between benchmark performance and clinical adoption is itself informative; §4.5.5 argues it is not an accident.

### 4.5.2 A taxonomy

Learned scatter estimators differ along three axes.

**Input domain.**

- _Sinogram-domain_: input is the prompt sinogram (± auxiliary inputs), output a scatter sinogram. Estimation happens in the same coordinate system as the physics; the network's job is essentially nonlinear smoothing.
- _Image-domain_: input is a first-pass reconstruction; output is a corrected image or a scatter estimate that is then forward-projected. Anatomically intuitive, but couples estimation to reconstruction.
- _Hybrid_: one domain in, the other out.

**Supervision signal.**

- _MC-supervised_: labels from Monte Carlo; ceiling is MC accuracy.
- _Physics-informed / SSS-supervised_: labels from a physical model; ceiling is that model's accuracy — usually worse than MC.
- _Self-supervised_: no labels; consistency-based training.

**Integration.**

- _Standalone estimator_: network produces $\hat s$; downstream reconstruction unchanged.
- _Unrolled reconstruction_: scatter estimation and reconstruction share weights, trained jointly.

The resulting grid has ~8 cells, each with published examples.

### 4.5.3 Where this work sits

This work is **sinogram-domain, MC-supervised, standalone**, with the **prompt sinogram as sole input** — no $\mu$-map, no first-pass activity estimate. Each choice is load-bearing:

1. **Sinogram domain**, because scatter is _defined_ in sinogram space; a sinogram-in-sinogram-out network avoids the forward-projection ambiguity image-domain methods must resolve.
2. **MC supervision**, because the objective is to match physics, not to match a physical _approximation_. The ceiling is set by MC (§4.4), and §3.6 confirms that ceiling is meaningful.
3. **Standalone**, because the goal is a drop-in replacement for SSS in existing MLEM pipelines. Practical deployability requires interchangeability.
4. **Prompt-only input**, because — and this is the sharpest claim — the prompt sinogram already implicitly encodes the attenuation and electron density that SSS receives explicitly.

Point 4 needs defending.

### 4.5.4 The three bets

**Bet 1: The target is easy to represent.** Compton scattering at 511 keV is forward-peaked (§4.2.2); surviving scatter after energy-window rejection is small-angle-dominated (§4.2.3); the resulting scatter sinogram is smooth and low-frequency. Convolutional networks approximate smooth functions of images cheaply, so the target sits where modest architectures suffice. **The actual information content of the scatter sinogram is much less than its pixel count** — which is why a U-Net with a few million parameters can compete with SSS at all.

> _Stress-test._ Does this fail when multiple scatter is large? Yes — multiple scatter is smoother still, but its _magnitude_ depends nonlinearly on object size in ways single-scatter reasoning does not capture. A network trained on small-animal-scale objects will underpredict multiple scatter on large objects. **This is a training-distribution question, not an architecture question.**

**Bet 2: The input already carries the object information SSS gets from $\mu$.**

> **[STITCH]** The previous draft justified this by analogy to MLAA (maximum-likelihood activity–attenuation). That analogy is removed (Appendix B, cut #5): MLAA is a specific literature with known ill-posedness results, most of which assume TOF — which this work does not have. Invoking it invites an objection the work cannot answer. The physical statement below is what was actually meant, and it stands on its own.

The trues sinogram is $D_a A x$: the geometric projection of activity, attenuated by $\exp(-\int \mu , d\ell)$ along each LOR. So the prompt — trues plus scatter — encodes both $x$ and the **path-integrated attenuation** $\int \mu , d\ell$, implicitly, along every line. The network's task is to exploit that joint encoding.

This is plausible _specifically because_ **scatter depends mainly on path-integrated electron density, not on fine voxel detail** (§4.2.4) — and path integrals are exactly what a sinogram preserves. The network does not need to _recover_ $\mu$ as an image; it needs only enough of $\mu$'s structure to compute a smooth functional of it.

> _Stress-test._ Does this fail when activity and attenuation are decoupled? Yes. The argument leans on the activity distribution illuminating the attenuating material — a decay must occur _behind_ material for that material's $\int\mu,d\ell$ to register in the prompt. A network trained on tracers that correlate with anatomy (as FDG does with soft tissue) will misestimate scatter for tracers with unusual distributions. Again: a training-distribution question. **And note this is exactly where TOF would help and is unavailable here** — TOF localizes the annihilation along the LOR, which is precisely the information that would break the activity–attenuation entanglement.

**Bet 3: MC labels dominate SSS on physics SSS cannot model.** Multiple scatter, out-of-FOV activity, detector energy blurring, positron range — all present in MC, none in SSS. A network trained on MC labels can, in principle, exceed SSS's accuracy for objects where these effects matter, at one forward pass of inference cost instead of an iterative integral.

> _Stress-test._ Does this fail when the MC labels are themselves biased? Yes — if the simulator's cross-sections, geometry, or detector model deviate from the real scanner, **the network inherits those biases invisibly.** This is why the choice of MC (MCGPU-PET) and its cross-section files matters as much as the network architecture. It also bounds the whole work: this is a claim about matching _MCGPU-PET's physics_, not about matching _nature_.

### 4.5.5 The cost of the bet

Every stress-test above shares a structural feature: **the failure mode is training-distribution-dependent.** SSS generalizes by physical construction — the Klein–Nishina cross-section is the Klein–Nishina cross-section on any object. A learned method generalizes only over its training distribution. **This distinction is qualitative, not quantitative.**

**Silent failure.** When SSS fails, it fails _detectably_: over- or under-correction with known signatures, large tail-fit residuals, negative-valued sinograms. When a learned method fails on out-of-distribution input, it may return a plausible-looking scatter sinogram that is **silently wrong**. There is no in-band diagnostic. This asymmetry, more than any benchmark number, is why SSS still ships in clinical software.

**Load-bearing training data.** In this framing, the training-data pipeline of Chapter 5 is not an implementation detail. **It _is_ the method's operational definition of physics.** Every choice — phantom library, count levels, tracer distributions, μ-heterogeneity — determines what the model considers reasonable, and therefore what it will silently misestimate. This is why Chapter 5 spends as much space on phantom generation as on network architecture.

**Where the cost falls.** The Fisher analysis of §3.6 provides a partial guardrail: no model can beat the ceiling, so gross overfitting produces visibly worse-than-SSS performance. But **between the ceiling and outright failure lies a large regime in which the model is _plausibly wrong_** — and this is where honest reporting of test distributions matters most.

> **Slogan.** _SSS computes scatter from the object; the learned network recognizes scatter from its shadow in the data._
> 
> _Stress-test:_ this holds when (i) scatter is smooth and low-frequency, and (ii) the prompt sinogram statistically determines the object well enough that $\mu$ need not be given. It fails where SSS also struggles, but for the _opposite_ reason: very large or dense objects (multiple scatter is no longer a smooth perturbation), or out-of-distribution activity distributions where the learned prior is simply wrong. **SSS would still be roughly right in these regimes; the network would not.**

### 4.5.6 The research question

Chapters 3 and 4 converge on one testable claim:

> **Can a sinogram-domain, MC-supervised convolutional network taking only the prompt sinogram as input match or exceed SSS as a scatter estimator on small-animal PET data, and how does its performance behave on held-out phantoms outside the training distribution?**

The yes/no answer is almost certainly _yes, in-distribution_: sinogram-domain DL scatter estimators have reached SSS-competitive accuracy in every published benchmark since Berker (2018). **The scientific contribution is the distributional clause** — over what range of objects, count levels, and physics regimes does the method hold? A clean answer to that is what separates a technical demonstration from a reproducible piece of physics.

Chapter 5 describes the pipeline. Chapter 6 gives the current empirical answer, which is preliminary.

---

# Chapter 5: A Prompt-Only Deep Scatter Estimator

## 5.1 Overview

Chapter 4 argued that a sinogram-domain, MC-supervised, prompt-only convolutional network is the right instrument. This chapter describes how it was built, along four axes:

1. **Data** — what phantoms exist, how their scatter fractions are distributed, how count levels are managed (§5.2).
2. **Representation** — how raw MCGPU-PET sinograms are re-coordinatized before entering the network (§5.3).
3. **Loss and supervision** — what the network minimizes, and how labels are constructed (§5.4, §5.5).
4. **Architecture and optimization** — the network and the mechanics of training it (§5.6, §5.7).

One section carries the chapter's theoretical weight: **§5.4, Poisson splitting.** It is the analogue of §3.6's Fisher ceiling and §4.3.2's SSS derivation — the place where a single mathematical fact does real work. Everything else is engineering, but engineering that defends itself against a stated alternative.

The pipeline:

$$\text{phantom recipe} \xrightarrow{\text{MCGPU-PET}} (\text{trues}, \text{scatter}) \xrightarrow{\text{merge + split}} (\text{input}, \text{label}) \xrightarrow{\text{U-Net}_{2.5\text{D}}} \hat s^{\text{merged}} \xrightarrow{\text{unmerge}} \hat s^{\text{ordered}} \xrightarrow{\text{MLEM}} \hat x.$$

Each arrow is justified below.

## 5.2 Data generation

### 5.2.1 Phantom library

Training phantoms are procedurally generated from a **bounds policy**: a dictionary specifying, per material, density and activity ranges, and per object, size, aspect ratio, and count. Two primitive shapes are used: **ellipsoids** and **elliptic cylinders**. Spheres and circular cylinders are degenerate cases (equal semi-axes), so they appear automatically without a separate type; aspect-ratio sampling covers the interpolation between them.

Materials come from the MCGPU-PET cross-section library (§4.4), typically air + water, with PMMA at 1.19 g/cm³ optionally added for phantom-realistic runs. Because Compton dominates at 511 keV and $\mu(E_0) \propto n_e \propto \rho$ for light elements ($Z/A \approx 0.5$; §4.2.4), **water at a chosen density is a valid surrogate for soft tissue to ~1%.** The 4–5% error from using water for cortical bone is confined to bone voxels and is small compared to MCGPU-PET's own ~10% agreement with GATE.

The bounds dict is deliberately **task-agnostic**: it describes what phantoms _can exist_, not what supervision task they serve. The same generated dataset supports scatter estimation, attenuation-map prediction, or reconstruction training — a separation of concerns that lets the expensive part (simulation) run once.

### 5.2.2 Stratification by scatter fraction

A uniform-in-parameters sampler produces a heavily peaked distribution of downstream quantities. Scatter fraction (SF) piles up in a middle band and starves the extremes — **exactly where a network is weakest.** Stratification steers generation so that a chosen scalar is _uniformly covered_ rather than uniformly sampled.

This cannot be achieved by editing the bounds dict, because flattening a nonlinear function of the parameters requires accept/reject at generation time. It is therefore passed at the call site: the user supplies a key function $\text{key_fn}: \text{Recipe} \to \mathbb{R}$ and a target $(\min, \max, n_\text{bins})$; the generator computes the key per candidate recipe, bins it, and admits recipes to keep bin counts balanced.

**The SF proxy.** The key used here is an activity-weighted mean escape optical depth. Two facts about the setup fix its form:

1. The energy window and detector resolution — dominant contributors to SF (§4.2.3) — are **held constant across every run.** So the _acquisition_ contribution to SF is a constant across the dataset; only the _phantom_ contribution varies.
2. At 511 keV, Compton dominates and scatter rate per unit volume is proportional to electron density, which for soft tissue is proportional to mass density. So "amount of scattering material" is simply **mass**, with no per-material correction.

Given a fixed window, the residual SF spread is driven by _how much material a coincidence must traverse_, weighted by _where the activity sits_. A decay deep inside a large dense body produces photons crossing more material — and scattering more often — than the same decay near a surface.

For each emitting voxel $v$, define the mean escape optical depth

$$\tau(v) = \frac{1}{n_{\text{dir}}} \sum_{\hat{\mathbf{n}}} \int_{[v,, \text{exit}(v, \hat{\mathbf{n}})]} \mu(\mathbf{x}, E_0) , d\ell,$$

with $\hat{\mathbf{n}}$ ranging over a coarse sample of escape directions (six axis directions in the implementation, for cheapness). The proxy is the activity-weighted mean of this depth, squashed into $[0, 1)$:

$$\boxed{\text{SFproxy}(\text{recipe}) = 1 - \exp\left(-\sum_v w(v) , \tau(v)\right), \qquad w(v) = \frac{\text{activity}(v)}{\sum_{v'} \text{activity}(v')}.}$$

> **[YOURS — FLAG: the prose currently claims more than the formula delivers.]**
> 
> The $1 - e^{-(\cdot)}$ wrapping is presented above as though it _derives_ an SF estimate from physics. It does not. $\sum_v w(v)\tau(v)$ is an activity-weighted mean optical depth — a sensible scalar. But wrapping it in $1 - e^{-(\cdot)}$ treats a _direction-averaged, activity-averaged_ depth as if it were the exponent of a _single-photon survival probability_, and then reads $1 - \text{survival}$ as a scatter fraction. Those are not the same quantity: a coincidence needs _both_ photons to escape; the average is over directions not actually taken along any LOR; and SF is not $1 - P(\text{escape})$.
> 
> **What I believe is actually true** (and what the $\rho \approx 0.90$ validation supports): this is a **monotone squashing function** applied to a physically-motivated ranking statistic. Since stratification only needs _rank order_, the squash is harmless — it maps a $[0,\infty)$ statistic onto a bounded interval so that bin edges can be specified conveniently. That is a legitimate and honest justification. It is not the justification currently written.
> 
> **Decide and rewrite this paragraph yourself.** Either (a) you had a specific reason for the exponential form, in which case state it; or (b) it is a convenience squash, in which case say so plainly and let the rank-correlation validation carry the weight. Do not leave it as-is — this is the one place in the chapter where the prose would not survive a pointed question.

Two design decisions distinguish this from a naive "size × density" alternative:

**(D1) Weight by activity _fraction_, not total activity.** SF is a _ratio_ — both numerator (scatter) and denominator (scatter + trues) scale linearly with total activity, so activity magnitude cancels. What survives is _where_ the activity is relative to the mass. Using total activity injects a confound with near-zero rank correlation to true SF.

**(D2) Aggregate the _dominant path_, not a sum over inserts.** SF is governed by the deepest material a coincidence traverses, not the count of separate objects. Summing per-insert optical depths conflates "one deep body" with "many shallow ones" and ranks poorly. The activity-weighted depth captures the dominant path automatically.

**Validation.** On a 20-sample Monte Carlo battery, `sf_proxy` achieves Spearman rank correlation $\rho \approx 0.90$ against true post-simulation SF. Naive alternatives — sum of per-insert optical depths ($\rho \approx 0.21$–$0.63$) and total activity ($\rho \approx 0.18$–$0.38$) — are unusable. Total mass alone reaches $\rho \approx 0.67$: useful, but geometry-blind.

> **Slogan.** _SF is a ratio, so activity cancels; what matters is where activity sits relative to mass._
> 
> _Stress-test:_ this holds in the **optically-thin** regime, where every added scatterer proportionally adds scatter events. In the optically-thick limit, multiple-scatter events leave the energy window _and_ trues also attenuate, so SF **saturates** — the "more material → more scatter" heuristic bends over. **The proxy is therefore weakest exactly where SF is highest**; stratification targets should not extend into the saturation regime.

**A limitation to own.** `sf_proxy` _ranks_ but does not _calibrate_: an absolute proxy value is not a predicted SF. Stratification targets are set in proxy units, not true-SF units. Coverage is also bounded by what the sampler can build; if the top bin exceeds the reachable regime, it fills with the highest-available phantoms.

### 5.2.3 Generation-time invariants

Some choices are cheap to change downstream; others are expensive because they are baked into the simulation. Being explicit about which is which prevents wasted compute.

**Invariant 1: span = 1 at tally time.** Only span = 1 preserves the one-plane-per-ring-pair LOR map that reconstruction needs. Compression at tally time is **not invertible** — the michelogram averaging destroys the axial-index structure MLEM operates on. Rebinning to higher span _after the fact_ is fine as a downstream view of a span-1 dataset. **Simulating at span > 1 is the single most consequential mistake possible in dataset generation.**

**Invariant 2: warm background.** Scatter is dominated by the bulk active + attenuating medium, not by disjoint inserts in air. A sampler producing "activity floating in air" simulates a scatter regime the deployment target does not share. All training phantoms therefore have an explicit warm background object (a large tissue-like ellipsoid with nonzero activity), with lesions and structural inserts sampled _inside_ it.

**Invariant 3: top-count acquisition.** Poisson thinning (§5.4) produces _lower_-count realizations from a higher-count original — the operation is one-way. Setting the simulated count level at the **top** of the intended range allows every lower level to be derived downstream at zero simulation cost. **The noise axis is not produced by re-simulating at different acquisition times; it is produced by binomial thinning of a single top-count simulation.**

**Invariant 4: scanner geometry.** MCGPU-PET's discrete-crystal scanner stands in for whatever real scanner the model would eventually target. The physically real fields — `radius_mm`, `axial_fov_mm`, `transaxial_fov_mm`, `energy_window_low_eV`, `energy_window_high_eV`, `energy_resolution` — must match the deployment scanner (§4.2.3). The discrete stand-in fields (`num_rings`, `num_detectors_per_ring`, `span`, `max_ring_difference`) may be chosen freely, subject to being fine enough not to be the bottleneck.

### 5.2.4 Dataset size by learning curve, not fiat

A common failure mode is guessing the training-set size, generating that many samples, and discovering afterward that the model either plateaued at a tenth of the data or was still improving at the full amount. Neither is a good outcome.

**General steps.**

1. Generate a **pilot** (~300 runs, sf_proxy-stratified).
2. Train the model on progressively larger _nested_ subsets.
3. Plot validation error against training-set size.
4. Extrapolate to choose the full batch.

This converts "how many runs" from a guess into a measurement.

**Note on what counts as a sample.** One run yields many training samples. After mirror-merge (§5.3.1), each phantom produces $\binom{N+1}{2}$ merged planes for $N$ rings — 2850 planes per run at the default 75-ring geometry — so a 300-run pilot is ~$8.5 \times 10^5$ plane-samples. **Per-plane data is not the bottleneck; phantom diversity is.** This is the concrete meaning of "the training distribution _is_ the model's physics" (§4.5.5).

## 5.3 Sinogram geometry and the 2.5D representation

### 5.3.1 Mirror-merge and $(\bar z, d)$ coordinates

The raw MCGPU-PET sinogram is a stack of planes indexed by (ring 1, ring 2), with additional structure from the michelogram — a segment-based layout packing oblique LORs into variable-length axial columns. For learning, this parameterization is doubly awkward: it has a redundancy (mirror pairs) and a variable segment length.

Both resolve with one coordinate change.

**Mirror-merge.** Each oblique ring pair is stored as two ordered planes $(a, b)$ and $(b, a)$ holding the _same_ physical LORs, by the kernel's orientation isotropy. Summing them is lossless.

> **Proposition (mirror-merge).** Let $y_{(a,b), i}$ and $y_{(b,a), i}$ be the counts on LOR $i$ in the two orderings. Under the MCGPU-PET forward model these are independent, each $\sim \text{Poisson}(\bar y_i)$. Then by Poisson additivity (§2.1.3), $$y_i^{\text{merged}} := y_{(a,b), i} + y_{(b,a), i} \sim \text{Poisson}(2 \bar y_i),$$ and $y^{\text{merged}}$ is a **sufficient statistic** for $\bar y_i$ — the ordered pair carries no information beyond the sum. $\square$

Merging halves the plane count and doubles counts per plane. At 75 rings, 5625 ordered pairs become 2850 unordered planes.

**Coordinate change.** Each merged plane is naturally indexed by

$$\bar z = \tfrac{1}{2}(r_1 + r_2) \quad \text{(axial position)}, \qquad d = |r_1 - r_2| \quad \text{(ring difference / obliqueness)}.$$

At fixed $d$, planes form a 1D stack ordered by $\bar z$ — a **segment**. This is a _relabeling_ of the plane axis: no interpolation, no averaging.

**Why these coordinates.** In $(\bar z, d)$, scatter is smooth in _both_ axes — along $\bar z$ within a segment, and along $d$ across segments. That smoothness is what makes the 2.5D representation viable (§5.3.2) and what would make compressed-and-interpolated estimation possible (§5.9, Ablation 5).

**The unmerge inverse.** To hand the model's output back to MLEM, ordered planes must be recovered:

$$s_{(a,b)}^{\text{ordered}} = \begin{cases} s^{\text{merged}}_{{a,a}} & \text{if } a = b \text{ (direct plane)} \[4pt] \tfrac{1}{2}, s^{\text{merged}}_{{a,b}} & \text{if } a \neq b \text{ (oblique; split between mirrors)}. \end{cases}$$

> **Getting the factor of $\tfrac12$ or the ordering wrong silently rescales the scatter entering MLEM.** There is no error message; the reconstruction is simply wrong by a factor. This inversion therefore lives in exactly one audited place (`representation.py::unmerge`), and merge/unmerge round-tripping should be a unit test.

```
[SPEC — YOU FILL]  merge / unmerge

merge:   ordered planes (n_rings, n_rings, A, R) -> merged planes (n_merged, A, R)
unmerge: merged planes (n_merged, A, R)          -> ordered planes (n_rings, n_rings, A, R)

Invariants that must hold (check these against your code)
  - merge is a pure sum over mirror pairs; no averaging, no interpolation
  - unmerge(merge(x)) == x  for the EXPECTED counts (not for a single noisy realization)
  - direct planes (a == b) are NOT halved on unmerge
  - oblique planes ARE halved on unmerge
  - bin order out of unmerge matches exactly what MLEM's projector expects

  <internal logic: fill from memory, then diff against code>
```

### 5.3.2 2.5D windowing

Segments differ in length: at ring difference $d$, a segment has $N - d$ planes. A network consuming whole segments would need variable-length inputs. But scatter varies slowly along $\bar z$, so per-plane prediction with a **window of neighbours** captures most of the axial coherence.

- **Sample** = one merged plane at position $(d, j)$ — segment index, position within segment.
- **Input** = the merged prompt sinograms of the $k$ planes at positions $j-h, \ldots, j+h$ within segment $d$ (where $h = (k-1)/2$), each as one channel, plus $d/(N-1)$ as a scalar broadcast to an extra channel.
- **Label** = the merged scatter sinogram at $(d, j)$.
- **Segment ends** use **replicate padding**: positions $< 0$ clamp to 0; positions beyond the end clamp to the last. Zero padding would create false discontinuities at segment ends.

This is what "2.5D" means: three-dimensional axial context is exploited via the $k$-plane window, but the convolutions operate in 2D on the $(A, R)$ angular–radial plane. Memory cost is $k\times$ a 2D network; the accuracy cost against full 3D is bounded by the smoothness of scatter along $\bar z$, which is high (§4.2.3).

**Choice of $k$.** In the pilot, $k = 7$. The optimum trades axial context against _effective_ training-set size — larger $k$ means fewer effectively independent samples per segment. Planned ablation (§5.9).

**Why the $d$ channel.** Ring difference is a _physical_ parameter (obliqueness), not merely an index. Feeding it as a broadcast scalar lets one set of weights specialize across obliquenesses. The alternative — separate networks per segment — would fragment the training set and discard the smoothness prior along $d$.

### 5.3.3 Per-sample brightness normalization

Prompt sinograms in the training set span more than an order of magnitude in total counts (~$4 \times 10^7$ to ~$10^9$), driven by the range of activity levels in the bounds policy. **This brightness is a degenerate axis**: it changes the noise level of input and label, but not the _scatter pattern_ the network must learn.

Feeding raw counts would waste capacity on a brightness invariance the model would have to learn. Instead each sample is scaled by its input-window mean:

$$\text{window}_{\text{scaled}} = \frac{\text{window}}{\overline{\text{window}}}, \qquad \text{label}_{\text{scaled}} = \frac{\text{label}}{\overline{\text{window}}}.$$

The network predicts in this scaled space; at inference the scale is multiplied back to recover physical counts.

**What this decouples.** $\overline{\text{window}}$ carries count-level information — label-irrelevant, noise-axis. Scatter _pattern_ (shape, angular structure, radial falloff) is preserved by the normalization. The network sees a distribution of shapes at common brightness, while the noise axis remains available separately via Poisson splitting (§5.4).

> **Slogan.** _Brightness is a noise axis; scatter pattern is a signal axis; decouple them at the sample level._
> 
> _Stress-test:_ this holds when scatter _shape_ is truly brightness-invariant, i.e. when the physics is linear in activity. **For MCGPU-PET's forward model this is exact.** For real scanners with pile-up or dead-time nonlinearities it would be an approximation — not a concern here, but a concern for deployment.

## 5.4 Poisson splitting: the noise axis and the independence trap

This section develops the most important theoretical tool in the pipeline. One operation fixes a subtle statistical bug **and** provides the noise axis for free.

### 5.4.1 The splitting theorem

**Theorem (Poisson splitting; standard, e.g. Kingman 1993 §2.3).** Let $N \sim \text{Poisson}(\lambda)$, and independently assign each of the $N$ events to stream $A$ with probability $p \in (0,1)$ or to stream $B$ with probability $1-p$. Then

$$N_A \sim \text{Poisson}(p\lambda), \qquad N_B \sim \text{Poisson}((1-p)\lambda), \qquad N_A \perp N_B.$$

Equivalently, conditional on $N = n$, $N_A \sim \text{Binomial}(n, p)$ and $N_B = N - N_A$.

> **Proof.** By the splitting mechanism, $P(N_A = a, N_B = b \mid N = a+b) = \binom{a+b}{a} p^a (1-p)^b$. Combining with $P(N = a+b) = \frac{\lambda^{a+b} e^{-\lambda}}{(a+b)!}$: $$P(N_A = a, N_B = b) = \frac{\lambda^{a+b} e^{-\lambda}}{(a+b)!} \binom{a+b}{a} p^a (1-p)^b = \frac{(p\lambda)^a e^{-p\lambda}}{a!} \cdot \frac{((1-p)\lambda)^b e^{-(1-p)\lambda}}{b!},$$ which factorizes as a product of the two marginals — proving independence and the marginal distributions simultaneously. $\square$

**The independence is the surprising part.** $N_A$ and $N_B$ sum to $N$, so intuition says they must be negatively correlated. They are not — because $N$ itself is random, and Poisson randomness is exactly the amount that makes the constraint non-binding. This is special to Poisson; the same construction on a fixed $N$ gives a binomial pair that _is_ dependent.

**Why this matters here.** MCGPU-PET's forward model gives, for each LOR $i$, independent Poisson counts of trues and scatter:

$$t_i \sim \text{Poisson}(\bar t_i), \qquad s_i \sim \text{Poisson}(\bar s_i), \qquad t_i \perp s_i.$$

The observed prompt is $y_i = t_i + s_i$. Applying the theorem _separately_ to $t_i$ and $s_i$ with the same $p$ produces four mutually independent Poisson variables:

$$t_i^A \sim \text{Poisson}(p \bar t_i), \quad s_i^A \sim \text{Poisson}(p \bar s_i), \quad t_i^B \sim \text{Poisson}((1-p) \bar t_i), \quad s_i^B \sim \text{Poisson}((1-p) \bar s_i),$$

with $t_i^A + t_i^B = t_i$ and $s_i^A + s_i^B = s_i$. The implementation is one line: `binomial(count, p)` per bin. **No listmode required. No additional simulation.**

### 5.4.2 The independence trap

Naive scatter training uses:

- **Input** = $t_i + s_i$ (measured prompts — what a scanner sees)
- **Label** = $s_i$ (the MC scatter realization)

This looks innocuous. It is not: **the label $s_i$ is literally contained in the input $t_i + s_i$.** The label's Poisson noise is not independent of the input's — they share a realization.

Why it matters: the Noise2Noise argument (Lehtinen et al. 2018) shows that a network trained to predict a noisy target from a noisy input recovers the conditional mean $\mathbb{E}[\text{label} \mid \text{input}]$ — **provided the two noises are independent.** When they are not, the network can partially fit the label's noise: it sees the noise in the input and reproduces it in the output. This inflates training scores and produces an estimate that has memorized _this particular MC realization_.

> **Slogan.** _If the label's noise is visible in the input, the network will memorize it._
> 
> _Stress-test:_ the trap bites in proportion to how much of the label variance is contained in the input — here, $\text{Var}(s_i)/\text{Var}(t_i + s_i) = \bar s_i/(\bar t_i + \bar s_i) = \text{SF}$. **So the trap is worst exactly where scatter matters most.**

### 5.4.3 Independent labels via splitting

The fix follows directly:

- **Input** = $t_i^A + s_i^A$ (split-$A$ prompts)
- **Label** = $\dfrac{p}{1-p} \cdot s_i^B$ (rescaled complementary scatter)

**Independence.** By §5.4.1, ${t_i^A, s_i^A} \perp {t_i^B, s_i^B}$. The input is built from split-$A$ components; the label from $s_i^B$. **The Noise2Noise assumption holds exactly.**

**Correct mean.** $\mathbb{E}[s_i^A] = p\bar s_i$ is the scatter expected count at the input's count level. Since $\mathbb{E}[s_i^B] = (1-p)\bar s_i$, rescaling by $p/(1-p)$ gives $\mathbb{E}[\text{label}] = p \bar s_i = \mathbb{E}[s_i^A]$. The network is trained to predict the scatter mean _at the input's count level_, with independent label noise.

**Cost.** One binomial draw per bin per sample per epoch. Zero additional GPU time. Zero additional simulation.

> **[YOURS — the variance cost of the rescaling.]**
> 
> The rescaling is unbiased, but it **inflates the label variance by $(p/(1-p))^2$**. Concretely: $\text{Var}(\tfrac{p}{1-p} s_i^B) = \tfrac{p^2}{(1-p)^2}\cdot(1-p)\bar s_i = \tfrac{p^2}{1-p}\bar s_i$, versus $\text{Var}(s_i^A) = p\bar s_i$ — a ratio of $p/(1-p)$.
> 
> At $p = 0.5$: ratio 1, no inflation, the symmetric choice. At $p = 0.9$: ratio 9 — the label is nine times noisier than the input's own scatter component. **So $p$ is a bias/variance knob, not a free parameter**, and the "correct" $p$ depends on where you want the input's count level to sit relative to the simulated top count.
> 
> **You should state which $p$ the pilot used and why.** If it is 0.5, say that the symmetric split is variance-optimal and that count level is controlled separately by §5.4.4's thinning. If it is not 0.5, justify it.

**Loss of Poissonicity.** The rescaled label $\tfrac{p}{1-p} s_i^B$ is **not** Poisson: it takes non-integer values, and its variance-to-mean ratio is $p/(1-p) \neq 1$ in general. This has consequences for the loss function — addressed in §5.5.

### 5.4.4 Splitting as the noise axis

The same operation, with a different probability, produces the noise axis.

Let the simulation be run at some fixed top count level. To obtain a realization at a _lower_ level, thin **both** trues and scatter by a common factor $\alpha \in (0, 1]$:

$$t_i^{\alpha} = \text{Binomial}(t_i, \alpha), \qquad s_i^{\alpha} = \text{Binomial}(s_i, \alpha).$$

> **[STITCH — notation cleaned up.]** The previous draft wrote this as a ratio $p_\text{eff}/p_\text{sim}$, where $p_\text{sim}$ was doing double duty as both a physical rate and a normalization, which made the algebra hard to check. The single factor $\alpha$ says the same thing without the ambiguity: **simulate at rate $\lambda$, thin by $\alpha$, get $\text{Poisson}(\alpha\lambda)$.** No ratios.

By §5.4.1, $t_i^{\alpha} \sim \text{Poisson}(\alpha \bar t_i)$ and $s_i^{\alpha} \sim \text{Poisson}(\alpha \bar s_i)$ — a **valid Poisson realization at the reduced count level**, not an approximation to one.

**Consequences.**

- **Scatter fraction is preserved**: both $\bar t$ and $\bar s$ scale by $\alpha$, so the ratio is invariant.
- **Object geometry is preserved**: thinning is per-bin and does not touch the phantom.
- **Poisson noise is realistic at every $\alpha$**, by the theorem, not by approximation.

So one high-count simulation supports an entire noise axis, decoupling _count level_ (label-irrelevant) from _SF and geometry_ (label-relevant). **This is why Invariant 3 (§5.2.3) mandates simulating at the top of the range.**

> **Slogan.** _Thinning goes down for free; simulating goes up for real GPU time. Simulate at the top; thin to any count level._
> 
> _Stress-test:_ this holds under Poisson statistics with independent events across LORs — MCGPU-PET's exact model. It would **fail** under rate-nonlinear detector effects (pile-up, dead time), which real scanners have and the simulator does not. So the training-time noise axis is Poisson-correct; deployment to a real scanner with pile-up would need to model that separately, or restrict to low-rate regimes where pile-up is negligible.

```
[SPEC — YOU FILL]  build_training_sample

Inputs
  trues_merged   : (A, R) merged trues for one plane      [from simulation, top count]
  scatter_merged : (A, R) merged scatter for one plane    [from simulation, top count]
  alpha          : count-level thinning factor in (0,1]    <-- KNOB (noise axis)
  p              : split probability in (0,1)              <-- KNOB (see the variance note above)
  rng            : random state                            <-- reseeded per epoch?  [YOURS: confirm]

Outputs
  input_window : (k+1, A, R)   -- k neighbour planes + broadcast d-channel, brightness-normalized
  label        : (A, R)        -- rescaled complementary scatter, same normalization

Invariants that must hold (check these against your code)
  - thinning applied to BOTH trues and scatter with the SAME alpha  (else SF is corrupted)
  - split applied SEPARATELY to trues and scatter with the SAME p
  - input uses ONLY split-A components; label uses ONLY split-B scatter
  - label rescaled by p/(1-p)
  - brightness normalization divides input AND label by the same window mean
  - resampled every epoch, not cached  [YOURS: confirm this is what the code does]

  <internal logic: fill from memory, then diff against code>
```

## 5.5 Loss function

### 5.5.1 Both candidate losses target the conditional mean

> **[STITCH]** The previous draft justified this via the Bregman-divergence conditional-mean theorem (Banerjee et al. 2005). That machinery is removed (Appendix B, cut #4): it is a general framework, and the specific fact needed here is two lines of calculus. What follows is that calculation. If asked "isn't this a Bregman-divergence property?", the honest answer is "yes, that is the general framework; the fact I need is the calculation below."

Two loss candidates dominate count-data regression: squared error and Poisson NLL. **Both are minimized, in expectation, at the mean of $Y$.**

_Squared error:_ $$\frac{d}{dc},\mathbb{E}\big[(Y - c)^2\big] = -2\big(\mathbb{E}[Y] - c\big) = 0 \iff c = \mathbb{E}[Y].$$

_Poisson NLL:_ $$\frac{d}{dc},\mathbb{E}\big[c - Y \log c\big] = 1 - \frac{\mathbb{E}[Y]}{c} = 0 \iff c = \mathbb{E}[Y].$$

**Consequence.** For a network predicting $\hat y(x)$ from input $x$, both losses have the same risk-minimizing solution: the conditional mean $\mathbb{E}[Y \mid X = x]$. **The choice of loss does not change the regression target.** It changes the optimization path and the finite-sample behaviour — nothing more.

This is exactly what licenses the §5.4.3 construction. The rescaled label is not Poisson, so Poisson NLL is no longer a _likelihood_ for it — but the calculation above never assumed the label was Poisson. It only assumed $\mathbb{E}[Y]$ exists. **So Poisson NLL still targets the right thing.**

### 5.5.2 Poisson NLL is the choice

The pilot uses

$$\ell_{\text{Poisson}}(\hat y, y) = \hat y - y \log \hat y \quad (+ \text{const in } \hat y).$$

Two reasons.

**(a) It matches the noise model where the noise model applies.** For genuinely Poisson labels, Poisson NLL _is_ the likelihood, with the statistical efficiency that implies. For the rescaled labels of §5.4.3, the strict likelihood interpretation is lost — but §5.5.1 shows the minimizer is unchanged, so nothing breaks.

**(b) It weights _relative_ error, not absolute error.** The gradient of $\ell_\text{Poisson}$ with respect to $\hat y$ is $1 - y/\hat y$, which vanishes at $\hat y = y$ and scales with the _ratio_. Contrast L2, whose gradient $\hat y - y$ scales with the absolute residual. **In sinograms with large dynamic range, L2 is dominated by the peaks; Poisson NLL treats low-count bins on relative-error terms** — matching the physics that low-count bins are Poisson-noise-dominated, and matching §3.5.5's observation that MLEM itself weights LORs by roughly $1/\bar y_i$.

**Where they might differ empirically.** Finite data and finite networks do not reach the theoretical optimum; they land at a compromise governed by the loss's optimization path. If the pipeline emphasizes low-count bins (e.g. object-missing tails, where scatter dominates), Poisson NLL should win. If it emphasizes peak bins, L2 might. **This is an empirical question, and §5.9 Ablation 2 is designed to answer it** — not a question §5.5.1 settles.

### 5.5.3 Softplus output

Poisson NLL requires $\hat y > 0$ (the $\log$ diverges at 0). Scatter counts are physically nonnegative, and §3.4 requires $\hat s \geq 0$ for MLEM. Both constraints are enforced by passing the raw output through softplus:

$$\hat y = \log(1 + e^{z}), \quad z = \text{last linear output},$$

which is smooth, strictly positive, and asymptotically linear for large $z$ (so it does not compress large predictions).

Softplus is preferred over ReLU because ReLU's hard zero would produce **infinite loss** at Poisson NLL's zero point, and the dead-ReLU pathology would make training unstable at exactly the bins that matter.

## 5.6 Architecture: UNet2p5D

The network is a standard 3-level U-Net operating on 2D planes with an extended input channel dimension. **The design goal is to be minimally opinionated** — a first model that works — while respecting the constraints of §5.4 and §5.5. Architecture is deliberately not where this work's contribution lies.

**Input.** $(k+1) \times A \times R$, with $A = 168$ angular bins, $R = 147$ radial bins, $k = 7$ the axial window, and the "+1" the broadcast $d$-channel.

**Padding for divisibility.** Three levels of 2× downsampling require spatial dimensions divisible by 8 (the code uses 16 to be safe). $R = 147$ is padded to 192 before the network and cropped back to 147 at output. $A = 168$ is cropped to 160 at input, with the final column restored by a one-column pad after the output layer. Both operations affect only sinogram edges, where physically nothing important happens (tail LORs missing the object).

> **[YOURS: confirm this is benign.]** The claim "nothing important happens at the edges" is doing real work here — §4.3.2 notes that tails are precisely where scatter dominates and where tail-fitting operates. **Cropping 8 angular bins is probably harmless; padding 45 radial bins is a bigger intervention.** Confirm that the padded region is genuinely outside the object's radial support for every phantom in the bounds policy, or note it as a limitation.

**Encoder.** Three levels of (3×3 conv, BN, ReLU) × 2 blocks, base 48 channels, doubling at each downsampling: $48 \to 96 \to 192 \to 384$ (bottleneck).

**Decoder.** Three levels of transposed-convolution upsampling, each followed by concatenation with the corresponding encoder skip connection and a double-conv block. Channels halve at each step.

**Output.** A 1×1 conv to one channel, then softplus (§5.5.3).

**Parameter count.** ~5–10M depending on $k$ and base width. Comfortably small: trainable on a single consumer GPU, and inference on a full run (2850 merged planes) takes seconds. **The inference-cost comparison against SSS (§4.3.3: ~5 s GPU) is therefore roughly a wash on this geometry** — the DL speed advantage is a claim about clinical-scale problems, not about this one.

**Deferred to ablation.** Depth (3 vs 4 levels); base channels (32/48/64); BatchNorm vs GroupNorm vs none. BatchNorm is used pragmatically; small-batch training in memory-constrained settings sometimes prefers GroupNorm.

## 5.7 Training procedure

**Optimizer.** Adam, learning rate $5 \times 10^{-4}$. No warmup, no schedule in the pilot; both are candidates for refinement.

**Batch size.** 16 planes. Bounded by GPU memory for $(k+1) \times A \times R = 8 \times 160 \times 192$ inputs at U-Net's footprint.

**Epochs.** 25 in the pilot. Validation loss monitored per epoch; best checkpoint retained.

**Data split.** Train/val/test = 70/15/15 **by run, not by plane.** Splitting by plane would leak neighbouring planes from the same phantom into val and test, inflating scores — planes of one phantom are strongly correlated, and the 2.5D window makes adjacent planes _share input data outright_. The split is frozen to `split.json` after first invocation, so reruns use an identical partition even as more runs finish in the background.

**Batch composition (`RunBatchSampler`).** Each batch's samples come from a _single run_, keeping the LRU cache warm — one run's merged sinograms (~350 MB) load once and serve many batches before eviction. Run order and within-run order reshuffle every epoch, so **the only lost randomness is cross-run mixing within a batch.** This is an accepted trade for ~10× reduction in disk I/O.

> **[YOURS: is this trade actually benign?]** Single-run batches mean every sample in a batch shares a phantom, hence shares SF, geometry, and count level. With BatchNorm, batch statistics are then computed over a _correlated_ sample — which is exactly the regime where BatchNorm is known to misbehave. This may interact with the BatchNorm-vs-GroupNorm choice in §5.6 in a way the ablation list does not currently capture. **Worth a sentence acknowledging it, or an added ablation.**

**Cache.** LRU with capacity 8 runs, holding recently-used merged (trues, scatter) arrays in RAM. Merge is recomputed per run on cache miss (cheap: one accumulate pass).

**Hardware.** Single NVIDIA GPU (compute capability ≥ 7.5). Wall-clock per epoch: `[TK: X]` minutes on 50 training runs at pilot scale.

**Checkpoints.** Two files per run: `last.pt` and `best.pt`, each carrying model state dict, hyperparameter dict, and validation loss.

## 5.8 Evaluation protocol

### 5.8.1 The floor/oracle/method bracket

A learned scatter estimate is not interpretable in isolation. The comparison places it between two extremes:

- **Floor**: MLEM with $\hat s = 0$ — no scatter correction. Worst-case reference.
- **Oracle ceiling**: MLEM with $\hat s = s^{\text{MC}}$ — the true MC scatter. **No estimator can do better** — this is §3.6's Fisher ceiling in operational form.
- **Method**: MLEM with $\hat s = $ network output.

The fraction of oracle gain captured is

$$\text{gain} = \frac{\text{error}(\text{floor}) - \text{error}(\text{method})}{\text{error}(\text{floor}) - \text{error}(\text{oracle})} \in [0, 1].$$

**This is the primary reported quantity.** It normalizes for phantom difficulty: an easy phantom (low SF) has a small floor-to-ceiling gap, so _any_ method looks close to oracle in raw error. The gain metric renormalizes so that the interesting axis — how much of the _available_ gain is captured — is visible.

> **A caution on the metric.** When the floor-to-oracle gap is small, `gain` is a ratio of two small numbers and becomes numerically unstable — the denominator approaches the noise level of the reconstruction itself. **Report the raw floor and oracle errors alongside the gain**, always, so a reader can see when the denominator is too small to trust. A gain of 0.95 on a phantom where floor and oracle differ by 1% means very little.

**Optional traditional baseline.** SSS with tail-fit scaling (§4.3.2) can be inserted as an additional data point. In small-animal PET this is complicated by the thin-tail failure mode; the choice between tail-fit SSS and oracle-scaled SSS is a diagnostic decision made per experiment, and **must be reported** — oracle-scaled SSS is not a deployable baseline and comparing against it flatters the learned method.

**General steps.**

1. Hold out a test run (phantom + MC trues/scatter at top count).
2. Thin to the target count level with a fixed $\alpha$ and a **recorded seed** (§5.4.4).
3. Form prompts $y = t^\alpha + s^\alpha$.
4. Run the network on $y$ (2.5D windowed, brightness-normalized), unnormalize, unmerge → $\hat s^{\text{ordered}}$.
5. Reconstruct three times with identical MLEM settings: $\hat s = 0$; $\hat s = s^{\text{MC},\alpha}$; $\hat s = \hat s^{\text{network}}$.
6. Scale-match all three against ground truth (§3.5.6).
7. Compute image-space metrics (§5.8.2) and the gain ratio.

```
[SPEC — YOU FILL]  evaluate_bracket

Inputs
  test_run       : phantom + MC (trues, scatter) at top count
  model          : trained checkpoint
  alpha          : count level                              <-- KNOB
  mlem_settings  : K, sens_floor_frac, x0                   <-- KNOB (identical across all 3 arms)
  roi_masks      : ground-truth regions for metrics         [from phantom recipe]

Output
  {floor, oracle, method} x {ROI bias, contrast recovery, uniformity}, plus gain ratio

Invariants that must hold (check these against your code)
  - all three reconstructions use IDENTICAL MLEM settings (K especially -- it is a regularizer)
  - all three scale-matched by the SAME procedure before any metric is computed
  - the same thinning realization feeds all three arms  (else you are comparing noise, not method)
  - raw floor/oracle errors reported alongside gain

  <internal logic: fill from memory, then diff against code>
```

### 5.8.2 Metrics in image space, not sinogram space

**The instinct to evaluate a sinogram estimator with sinogram MSE is wrong.** MLEM's update multiplies by $y/\bar y$ per LOR (§3.5.2); the ratio is sensitive to low-count bins, and a bin-wise error at low $\bar y$ is _amplified_ in the reconstructed image.

Concretely: an estimator undershooting scatter by 10% on peak bins produces a mild negative bias in the image. An estimator undershooting by 10% on _tail_ bins can produce dramatic streaking, because those bins' $y/\bar y$ ratios are driven up disproportionately. **Sinogram MSE ranks these two errors as comparable. The image tells the truth.**

Metrics are therefore all in image space, on the reconstructed activity:

- **ROI bias**: relative error in mean activity of a physically-meaningful region (background, hot rod, cold insert).
- **Contrast recovery**: standardized comparison of a hot region against a warm background.
- **Background uniformity**: coefficient of variation in a nominally uniform region.

These are inspired by the NEMA NU 4-2008 protocol, adapted for the arbitrary phantom shapes in the training/test sets.

> **Slogan.** _Sinogram MSE ranks; image ROI bias tells the truth._
> 
> _Stress-test:_ this holds because MLEM's amplification is nonlinear in $\bar y$. It would fail for a **linear** reconstruction (FBP), where sinogram MSE and image MSE would be more comparable. For the statistical inversion actually used, image-space evaluation is the honest one.

## 5.9 Planned ablations

The pilot is a second training run. The following ablations are **planned but not completed**. Listing them makes the incomplete state legible rather than hidden.

|#|Ablation|Tests|
|---|---|---|
|1|Correlated vs independent labels (`split=False` vs `True`)|Does §5.4.2's trap bite in practice? Expect inflated train-vs-val gap and worse test performance in the correlated arm.|
|2|L2 vs Poisson NLL (with `split=True`)|§5.5.1 guarantees the _target_ is identical; differences are finite-data optimization paths.|
|3|Axial context $k \in {1,3,5,7,9}$|$k=1$ is pure 2D; larger $k$ trades context for fewer effectively independent samples.|
|4|With/without μ-map input channel|**Direct test of §4.5.4 Bet 2.** Small marginal gain ⟹ Bet 2 empirically confirmed for this training distribution.|
|5|Compressed prediction + interpolation|Predict on a coarse $(\bar z, d)$ grid, interpolate up — mirrors SSS's coarse-detector trick (§4.3.3). Tests the smoothness prior aggressively.|
|6|Sinogram vs image domain|Retrain in image domain (input: first-pass MLEM; output: scatter image). Natural axis from §4.5.2's taxonomy.|
|7|2.5D vs full 3D|Replace the 2D U-Net with a 3D one consuming full segment stacks. Memory-heavy end of the representation axis.|
|8|SF-generalization sweep|Evaluate across SF bands inside _and outside_ the training range. **Directly tests §4.5.5's silent-failure warning** — the closest thing to a measurement of the thesis's actual claim.|
|9|Multi-realization ceiling analysis|Rerun MC with different Poisson seeds on the same test phantom to establish the Fisher-ceiling gap **statistically**. Upgrades the current single-realization "at-ceiling" result from an observation to a claim.|

> **Priority note.** Ablations 8 and 9 are not equal in value to the others. **Ablation 9 is required before any "at-ceiling" claim can be made at all** — a single realization cannot distinguish "at the ceiling" from "one draw that happened to land near it." **Ablation 8 is the thesis's actual contribution** (§4.5.6's distributional clause). Ablations 1–7 are engineering hygiene. If time is limited, 9 then 8, then the rest.

---

# Chapter 6: Preliminary Results

> **[YOURS — this chapter does not exist yet.]**
> 
> Below is a skeleton with the claims the rest of the report has set up, so you can see what shape the evidence needs to take. Fill each with what you actually have; **delete any subsection you cannot support.** An empty section is better than a padded one.

## 6.1 What was run

`[TK]` Pilot dataset size, count levels, SF range covered, training-set/val/test split sizes, hardware, wall-clock.

## 6.2 The bracket on the test phantom

`[TK]` Floor / oracle / method, with raw errors _and_ gain (per the caution in §5.8.1).

> **Framing that the rest of the report has earned, and the limit of it:** §3.6 establishes that the oracle is a genuine ceiling. If the method lands at it, that is meaningful. But **on one phantom and one noise realization, "at ceiling" is an observation, not a result** — it cannot distinguish a method that matches the ceiling from a draw that landed near it. State it that way. Ablation 9 (§5.9) is what converts it.

## 6.3 SSS baseline

`[TK]` If run: tail-fit or oracle-scaled? Report which (§5.8.1) and report the tail-fit diagnostic (§4.3.2).

## 6.4 What is not yet known

`[TK]` Explicitly: the distributional clause of §4.5.6 is untested (Ablation 8); the ceiling claim is single-realization (Ablation 9); the μ-map bet is unconfirmed (Ablation 4).

---

# Chapter 7: Discussion

> **[YOURS — skeleton only.]** The structure below follows from what Chapters 3–5 established. The content is yours.

## 7.1 Where deep learning competes

`[TK]` Speed at MC-quality labels. Note the honest caveat from §5.6: on _this_ geometry, SSS is already ~5 s on GPU, so the speed argument is a claim about clinical-scale problems, not this one. What generalizes: the ability to learn physics SSS cannot model (§4.5.4 Bet 3).

## 7.2 Where classical methods still win

`[TK]` The argument is already made in §4.5.5 and should be summarized, not re-argued: generalization by construction; detectable rather than silent failure; no training data required; interpretability of each factor in Watson's formula.

## 7.3 The distributional question

`[TK]` This is the contribution (§4.5.6). What would a satisfying answer look like? Probably: a map of gain-vs-SF with the training range marked, showing where degradation begins.

## 7.4 Open frontier

`[TK]` Joint reconstruction (unrolled networks folding scatter estimation into MLEM iterates, §4.5.2); self-supervised training without MC labels; transfer across scanner geometries (the invariant-4 problem, §5.2.3).

---

# Appendix A: Notation

See §3.1 for the reconstruction notation table. Additional symbols introduced later:

|Object|Symbol|Section|
|---|---|---|
|Scatter angle at point $S$ between detectors $A$, $B$|$\theta_{ASB}$|§4.3.2|
|Klein–Nishina differential cross-section|$d\sigma_{KN}/d\Omega$|§4.2.2|
|Energy-window acceptance|$W(E')$|§4.2.3|
|Electron density|$n_e$|§4.2.4|
|Split probability|$p$|§5.4.3|
|Count-level thinning factor|$\alpha$|§5.4.4|
|Axial window size|$k$|§5.3.2|
|Mean axial position, ring difference|$\bar z$, $d$|§5.3.1|

---

# Appendix B: Change log — what was cut from the previous draft, and why

Each cut was made on one criterion: **does this earn its keep downstream?** Content that was true and interesting but unused was removed, because prose that cannot be defended and is not load-bearing is pure downside.

|#|Cut|Rationale|
|---|---|---|
|1|**ML-loss dictionary** (cross-entropy ↔ Bernoulli NLL, weight decay ↔ Gaussian prior on weights, categorical NLL)|Never used. The _specific_ Gaussian-NLL ↔ least-squares pivot survives as §2.3.3, because it is the bridge to Poisson NLL.|
|2|**Standalone Fisher/CRLB section** (score has mean zero, negative-expected-Hessian identity, Loewner order in generality, Van Trees, Barankin)|Fisher information is used **once**, in §3.6, and that derivation is self-contained. Definition + CRLB-as-fact now live inline where used. Two pages → half a page.|
|3|**General EM algorithm, ELBO, variational-inference/VAE connection**|EM was used exactly once, to derive MLEM. The general template added a layer of abstraction over a derivation that is clearer stated directly (§3.5.2). The MLEM derivation also appeared **twice** in the previous draft; one copy retained.|
|4|**Bregman-divergence conditional-mean theorem** (Banerjee et al. 2005)|Replaced by the two-line calculation in §5.5.1, which delivers the identical conclusion for the two losses actually considered. The general framework is namechecked, not relied on.|
|5|**MLAA analogy** in Bet 2|MLAA is a specific literature with known ill-posedness results, mostly assuming TOF — which this work lacks. The analogy invited an objection the work cannot answer. Replaced with the physical statement that was actually meant (§4.5.4).|
|6|**Duplicate Chapter 3** (an earlier Compton/SSS/ML-principle draft)|Fully superseded by the current Chapter 4.|
|7|**E-step "orthogonality principle" callback**|Correct (conditional expectation is $L^2$ projection onto the σ-algebra of the observation) but connected to §2.2.2's finite-dimensional least-squares orthogonality only by analogy, not by the same theorem. As a logic-chain aid it slightly misleads. Illuminating, not load-bearing.|
|8|**Sobolev / TV examples** in the regularization section; **"$R^\top R$ is what FBP inverts"**|The examples are unused. The $R^\top R$ claim is simply wrong — FBP inverts $R$ via the Fourier slice theorem and never forms the normal operator. Corrected in place (§2.2.2).|
|9|**Randoms $r$** as a live term|Not in the pipeline. Retained in the _general_ forward model (§3.3.3) because it is standard theory and because §3.6's argument covers any additive background; dropped everywhere it would be a dead symbol. Declared once in the scope note.|

**Kept, against an earlier recommendation to cut:**

- **Operator theory** (§2.2.2), trimmed by roughly half. Retained because the singular-value/ill-posedness intuition is genuinely reused (§3.2.6, §3.5.5) and because you find it clarifying. The parts that served nothing — Sobolev/TV, the normal-operator-as-central-object framing — are gone.
- **Regularization ↔ MAP bridge** (§2.3.5). Retained because §3.5.5 depends on it to explain why non-negativity, early stopping, and sensitivity flooring count as regularization at all.

---

# Appendix C: The defensible spine

Eight steps. If you can defend each, you can defend the mathematical content of this report.

1. **Radon transform; ill-posedness via decaying singular values** → FBP with a ramp filter is the unregularized inverse. (§2.2.2, §3.2)
2. **Tikhonov as an SVD floor** → penalized least squares → **regularization = MAP with a prior.** (§2.2.4, §2.3.5)
3. **Gaussian NLL = least squares** → therefore, for count data, use **Poisson NLL.** (§2.3.3, §2.3.4)
4. **MLEM from Poisson thinning**: the latent is "which voxel did this photon come from?"; the multiplicative form is a consequence of Poisson structure specifically. (§3.5.2)
5. **MLEM's regularizers are implicit**: non-negativity, likelihood weighting, early stopping ($K$ is the regularization parameter), sensitivity flooring. (§3.5.5)
6. **Fisher ceiling**: $I(x) \preceq I_{\text{clean}}(x)$ — additive background inflates variance without adding information, so perfect scatter correction cannot restore scatter-free performance. This makes the oracle a genuine ceiling. (§3.6)
7. **Poisson splitting**: fixes the independence trap (label noise must not be visible in the input) and supplies the noise axis for free. Independence of the two streams is the non-obvious part, and it is Poisson-specific. (§5.4)
8. **Both L2 and Poisson NLL target $\mathbb{E}[Y \mid X]$** — two lines of calculus, no Bregman machinery. So the loss choice is about optimization path, not about the target. (§5.5.1)

Everything else in Chapters 2–5 is scaffolding, and should be judged by whether it supports one of these eight.