# stoch Stage 2 Roadmap

## Context

Compared against TFP Python (v0.25, 115+ distributions, 85+ bijectors), Pyro (v1.9, 105+ distributions, 19 autoguides), PyMC (v5.27, 100+ distributions, GP module, ArviZ), and Stan (via CmdStanPy, auto-constraint transforms). All inspected from conda env `prob`.

## Current inventory (Stage 1 — complete)

### Distributions (17) — `src/distributions/`
- [x] Normal, Bernoulli, Uniform, Gamma, Beta, Exponential, Poisson, Categorical
- [x] Independent, MultivariateNormalDiag, LogNormal, StudentT, Dirichlet
- [x] MixtureSameFamily, TransformedDistribution
- [x] JointDistributionNamed, JointDistributionSequential
- [x] KL: Normal-Normal, Bernoulli-Bernoulli (2 pairs)

### Bijectors (9) — `src/bijectors/`
- [x] Identity, Exp, Log, Softplus, Sigmoid, Shift, Scale, Chain

### MCMC — `src/mcmc/`
- [x] HamiltonianMonteCarlo, NoUTurnSampler, RandomWalkMetropolis
- [x] TransformedTransitionKernel, DualAveragingStepSizeAdaptation
- [x] sampleChain, leapfrogIntegrate
- [x] effectiveSampleSize, potentialScaleReduction
- [x] stateToArray, arrayToState, computeGrads, valueAndGrads

### VI — `src/vi/`
- [x] trainableNormal, buildMeanFieldPosterior
- [x] computeElbo, fitSurrogatePosterior

### Math — `src/math/`
- [x] Constants: LOG_PI, LOG_2, LOG_2PI, LOG_SQRT_2PI, SQRT_2, etc.
- [x] Special: logGamma, digamma, logBeta, ndtr, logNdtr, ndtri
- [x] Generic: log1mexp, logAddExp, softplusInverse
- [x] Linalg: cholesky (differentiable), triangularSolve

### Internal — `src/internal/`
- [x] assertPositive, assertNonNegative, assertInRange, assertOneOf
- [x] commonDtype, isFloatDtype, isIntDtype
- [x] toTensor, broadcastShapes, broadcastShapesMultiple

---

## Gap summary

| Category | stoch | TFP Python | Pyro | PyMC | Stan |
|---|---|---|---|---|---|
| Distributions | 17 | 115+ | 105+ | 100+ | 50+ |
| Bijectors | 9 | 85+ | 21 | implicit | implicit |
| MCMC kernels | 3+2 | 6+3 | 3+SVI | NUTS+MH+SMC | HMC+NUTS |
| VI/Guides | 1 type | 28 fns | 19 autoguides | ADVI+FullRank | variational |
| KL pairs | 2 | 40+ | implicit | implicit | N/A |
| GPs | none | 5 dists | GPyTorch | full+HSGP | functions |
| Time series | none | STS(28) | MCMC | AR,GARCH,GRW | ODE |
| BNN layers | none | 41 | TyXe | none | none |
| Diagnostics | ESS,Rhat | full stats | traces | ArviZ suite | diagnose |
| Model comparison | none | none | none | LOO,WAIC | loo |

---

## Stage 2 — Workstream A: Distributions (+23)

### A0. Math prerequisites
- [ ] `incompleteBeta(a, b, x)` — needed by Binomial CDF, Beta CDF
- [ ] `regularizedIncompleteGamma(a, x)` — needed by Chi2, Gamma, Poisson CDF
- [ ] `besselI0(x)`, `besselI1(x)` — needed by VonMises
- [ ] `logChoose(n, k)` — needed by Binomial, NegBinomial logProb
- File: `src/math/special.js`

### A1. Simple continuous (7)
- [ ] Cauchy(loc, scale) — `src/distributions/cauchy.js`
- [ ] Laplace(loc, scale) — `src/distributions/laplace.js`
- [ ] Logistic(loc, scale) — `src/distributions/logistic.js`
- [ ] Gumbel(loc, scale) — `src/distributions/gumbel.js`
- [ ] HalfNormal(scale) — `src/distributions/half_normal.js`
- [ ] HalfCauchy(scale) — `src/distributions/half_cauchy.js`
- [ ] Chi2(df) — `src/distributions/chi2.js`

### A2. Simple discrete (4)
- [ ] Binomial(totalCount, probs) — `src/distributions/binomial.js`
- [ ] NegativeBinomial(totalCount, probs) — `src/distributions/negative_binomial.js`
- [ ] Geometric(probs) — `src/distributions/geometric.js`
- [ ] Multinomial(totalCount, probs) — `src/distributions/multinomial.js`

### A3. Parameterized continuous (5)
- [ ] InverseGamma(concentration, scale) — `src/distributions/inverse_gamma.js`
- [ ] Pareto(concentration, scale) — `src/distributions/pareto.js`
- [ ] Weibull(concentration, scale) — `src/distributions/weibull.js`
- [ ] VonMises(loc, concentration) — `src/distributions/von_mises.js`
- [ ] TruncatedNormal(loc, scale, low, high) — `src/distributions/truncated_normal.js`

### A4. Multivariate & matrix (3)
- [ ] MultivariateNormalTriL(loc, scaleTril) — `src/distributions/mvn_tril.js`
- [ ] Wishart(df, scaleTril) — `src/distributions/wishart.js`
- [ ] LKJCholesky(dimension, concentration) — `src/distributions/lkj_cholesky.js`

### A5. Differentiable discrete & zero-inflated (4)
- [ ] OneHotCategorical(probs/logits) — `src/distributions/one_hot_categorical.js`
- [ ] RelaxedBernoulli(temperature, probs) — `src/distributions/relaxed_bernoulli.js`
- [ ] RelaxedOneHotCategorical(temperature, probs) — `src/distributions/relaxed_one_hot_categorical.js`
- [ ] ZeroInflatedPoisson(rate, gate) — `src/distributions/zero_inflated_poisson.js`

### A6. KL divergence pairs (+6)
- [ ] Gamma-Gamma
- [ ] Beta-Beta
- [ ] Exponential-Exponential
- [ ] Dirichlet-Dirichlet
- [ ] Categorical-Categorical
- [ ] Laplace-Laplace
- File: `src/distributions/kullback_leibler.js`

---

## Stage 2 — Workstream B: Bijectors (+8)

- [ ] Tanh — `src/bijectors/tanh.js`
- [ ] Invert (wraps any bijector) — `src/bijectors/invert.js`
- [ ] Power — `src/bijectors/power.js`
- [ ] SoftmaxCentered — `src/bijectors/softmax_centered.js`
- [ ] FillTriangular — `src/bijectors/fill_triangular.js`
- [ ] CorrelationCholesky — `src/bijectors/correlation_cholesky.js`
- [ ] Ascending — `src/bijectors/ascending.js`
- [ ] AffineScalar — `src/bijectors/affine_scalar.js`

---

## Stage 2 — Workstream C: High-level API & Stats

### C1. Stats module (new `src/stats/`)
- [ ] `summary(chains)` → mean, std, HDI, ESS, R-hat, MCSE — `src/stats/summary.js`
- [ ] `hdi(samples, prob)` → [low, high] — `src/stats/hdi.js`
- [ ] `mcse(samples)` → Monte Carlo standard error — `src/stats/mcse.js`
- [ ] Module index — `src/stats/index.js`
- [ ] Export from `src/index.js`

### C2. High-level `sample()`
- [ ] Auto-NUTS + DualAveraging + multi-chain + convergence — `src/mcmc/sample.js`
- [ ] Returns `{ samples, summary, diagnostics }`

### C3. Predictive utilities
- [ ] `posteriorPredictive({ samples, predictFn })` — `src/mcmc/posterior_predictive.js`
- [ ] `priorPredictive({ priorFn, numSamples })` — `src/mcmc/prior_predictive.js`

### C4. NUTS improvements
- [ ] Divergence tracking — modify `src/mcmc/nuts.js`

---

## Stage 2 — Workstream D: Gaussian Processes (new `src/gp/`)

### D1. Kernels
- [ ] SquaredExponential (RBF) — `src/gp/kernels/squared_exponential.js`
- [ ] Matern (nu=1/2, 3/2, 5/2) — `src/gp/kernels/matern.js`
- [ ] Linear — `src/gp/kernels/linear.js`
- [ ] Periodic — `src/gp/kernels/periodic.js`
- [ ] White — `src/gp/kernels/white.js`
- [ ] Combinators (Add, Product, Scale) — `src/gp/kernels/combinators.js`

### D2. GP distributions
- [ ] GaussianProcess(meanFn, kernel) — `src/gp/gaussian_process.js`
- [ ] GaussianProcessRegressionModel — `src/gp/gp_regression.js`
- [ ] Module index — `src/gp/index.js`
- [ ] Export from `src/index.js`

---

## Deferred to Stage 3+

| Feature | Stage |
|---|---|
| Normalizing flows (RealNVP, MAF) | 3 |
| STS time series module | 3 |
| BNN layers (DenseVariational) | 3 |
| Auto-guides (full-rank, flow) | 3 |
| LOO/WAIC model comparison | 3 |
| SMC / importance sampling | 3 |
| Slice sampler, parallel tempering | 3 |
| Effect handlers (sample/plate) | 4 |
| ODE solvers | 4 |

---

## Implementation order

1. Math: incompleteBeta, incompleteGamma, besselI, logChoose
2. A1: Cauchy, Laplace, Logistic, Gumbel, HalfNormal, HalfCauchy, Chi2
3. A2: Binomial, NegativeBinomial, Geometric, Multinomial
4. A3: InverseGamma, Pareto, Weibull, VonMises, TruncatedNormal
5. B: Tanh, Invert, FillTriangular, CorrelationCholesky, SoftmaxCentered, Ascending, Power, AffineScalar
6. A4: MultivariateNormalTriL, Wishart, LKJCholesky
7. A5: OneHotCategorical, RelaxedBernoulli, RelaxedOneHotCategorical, ZeroInflatedPoisson
8. A6: KL pairs
9. C1: Stats module (summary, HDI, MCSE)
10. C2: High-level sample()
11. C3+C4: Predictive utilities + NUTS divergence tracking
12. D1: GP kernels
13. D2: GP distributions

## Verification

Per distribution: generate reference data via `conda run -n prob python`, unit tests vs scipy/TFP, `npm run test:unit`

Per bijector: forward/inverse roundtrip, logDetJacobian vs numerical

High-level API: recover known posterior, R-hat<1.1, ESS>100

GPs: posterior mean interpolates observations, kernel matrix matches sklearn

End-to-end: `npm run build`

## Docs updates

- README.md: distribution table, GP section, high-level API
- CHANGELOG.md: Stage 2 entries
- AGENTS.md: repo map with new modules
