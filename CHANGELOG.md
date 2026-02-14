# Changelog

## 0.1.0 (unreleased)

### Added

#### Project infrastructure
- Project scaffolding: package.json, webpack config, jest config, babel config
- Directory structure: src/, tests/, scripts/, benchmarks/, examples/, docs/
- ESM source with UMD/CJS Webpack output (~88KB dev bundles)
- TensorFlow.js ^4.x as externalized dependency
- Python reference data generator (`scripts/generate-reference-data.py`) using scipy.stats
- Test helpers: `expectClose`, `expectTensorClose`, `autoTolerance`, `sampleStats`

#### Internal utilities
- `assert-util`: assertPositive, assertNonNegative, assertInRange, assertOneOf
- `dtype-util`: commonDtype, isFloatDtype, isIntDtype
- `tensor-util`: toTensor, broadcastShapes, broadcastShapesMultiple, shapeSize

#### Math utilities (`stoch.math`)
- Constants: LOG_PI, LOG_2, LOG_2PI, LOG_SQRT_2PI, SQRT_2, EULER_MASCHERONI
- Generic: log1mexp, logAddExp, softplusInverse
- Special functions: logGamma (Lanczos), digamma, logBeta, ndtr, logNdtr
- Inverse normal CDF: ndtri (Acklam's algorithm)
- Linear algebra: differentiable cholesky decomposition (Iain Murray 2016 backward pass via tf.customGrad), triangularSolve for lower/upper systems with adjoint support

#### Distribution base class
- Abstract base class with public/internal method split (logProb → _logProb)
- Automatic tf.tidy() wrapping on all public methods
- dispose() for parameter tensor cleanup
- Parameter validation (validateArgs, global toggle via setValidateArgs)
- Broadcasting support: batchShape computed from parameter shapes
- Default implementations: prob from exp(logProb), stddev from sqrt(variance), logCdf from log(cdf)

#### Distributions (`stoch.distributions`)
- Normal(loc, scale): sample, logProb, prob, cdf, logCdf, mean, variance, entropy, mode
- Bernoulli(probs|logits): sample, logProb, prob, mean, variance, entropy, mode
- Uniform(low, high): sample, logProb, prob, cdf, logCdf, mean, variance, entropy
- Gamma(concentration, rate): sample, logProb, prob, cdf, mean, variance, entropy, mode
- Beta(concentration1, concentration0): sample, logProb, prob, mean, variance, entropy, mode
- Exponential(rate): sample, logProb, prob, cdf, mean, variance, entropy
- Poisson(rate): sample, logProb, prob, cdf, mean, variance, mode
- Categorical(probs|logits): sample, logProb, prob, mean, variance, mode

#### KL divergence
- Registry-based KL divergence system (registerKL, klDivergence)
- Normal-Normal KL divergence
- Bernoulli-Bernoulli KL divergence
- Gamma-Gamma KL divergence
- Beta-Beta KL divergence
- Exponential-Exponential KL divergence
- Dirichlet-Dirichlet KL divergence
- Categorical-Categorical KL divergence
- Laplace-Laplace KL divergence

#### Stats (`stoch.stats`)
- `hdi(samples, prob)`: Highest Density Interval — narrowest interval containing given probability mass
- `mcse(samples)`: Monte Carlo Standard Error — sd / sqrt(ESS), accounts for autocorrelation
- `summary(chains, options)`: ArviZ-style summary — mean, sd, HDI, ESS, R-hat, MCSE per parameter

#### Benchmarks
- Benchmark infrastructure (`benchmarks/run.js`) using Benchmark.js
- Comparison against WebPPL (the only other JS probabilistic programming library)
- Results across 3 backends (JS CPU, native CPU via tfjs-node, GPU via tfjs-node-gpu):
  - logProb: up to **52x faster** (native CPU) and **26x faster** (GPU) vs WebPPL
  - Sampling: competitive on CPU, stoch wins on Beta and Exponential with native/GPU backends
- Memory leak test: zero tensor leaks confirmed across 100 create/sample/dispose cycles
- Batched operation benchmarks (stoch only — WebPPL has no batching support)

#### Bijectors (`stoch.bijectors`)
- Abstract Bijector base class (forward, inverse, forwardLogDetJacobian, inverseLogDetJacobian)
- Element-wise bijectors: Identity, Exp, Log, Softplus, Sigmoid
- Affine bijectors: Shift, Scale
- Composition: Chain (applies bijectors right-to-left)
- TransformedDistribution: Distribution + Bijector → new distribution (e.g., LogNormal = Normal + Exp)

#### Extended distributions
- Independent: reinterprets batch dims as event dims (needed for multivariate logProb)
- MultivariateNormalDiag: diagonal-covariance MVN via Independent(Normal)
- LogNormal: via TransformedDistribution(Normal, Exp)
- StudentT: heavy-tailed alternative to Normal, uses logGamma
- Dirichlet: simplex prior, Ahrens-Dieter sampling for concentration < 1
- MixtureSameFamily: finite mixture models with logSumExp logProb
- JointDistributionNamed: named dependency graph with topological sort (explicit deps + arg parsing)
- JointDistributionSequential: list-based model specification

#### MCMC inference (`stoch.mcmc`)
- State utilities: state↔array marshaling for tf.grads compatibility
- TransitionKernel: abstract base class (oneStep, bootstrapResults)
- Leapfrog integrator: symplectic (Stormer-Verlet) with tf.tidy memory management
- HamiltonianMonteCarlo: HMC with Metropolis acceptance criterion
- sampleChain: main sampling loop with burn-in, thinning, trace collection, memory management
- TransformedTransitionKernel: bijector↔MCMC bridge for constrained parameters
- DualAveragingStepSizeAdaptation: Nesterov dual averaging for auto-tuning step size
- RandomWalkMetropolis: gradient-free MCMC with configurable proposals
- NoUTurnSampler (NUTS): classic Algorithm 3 (Hoffman & Gelman 2014) with adaptive trajectory length
- Diagnostics: effectiveSampleSize (Geyer 1992 initial positive sequence), potentialScaleReduction (R-hat)
- High-level `sample()`: auto-NUTS/HMC with step size adaptation, multi-chain support, convergence diagnostics
- NUTS divergence tracking: numDivergent, numMaxDepth, meanLeapfrogs in diagnostics
- `posteriorPredictive({ samples, predictFn })`: generate predictions from posterior samples
- `priorPredictive({ priorFn, predictFn })`: generate predictions from prior draws

#### Gaussian Processes (`stoch.gp`)
- Kernel base class with matrix() and apply() methods
- SquaredExponential (RBF) kernel with amplitude and lengthScale
- Matern kernel (nu=0.5, 1.5, 2.5) with amplitude and lengthScale
- Linear kernel with variance and bias
- Periodic (ExpSinSquared) kernel with period parameter
- White noise kernel
- Kernel combinators: Add, Product, Scale
- GaussianProcess: GP prior with sample, logProb, and posterior methods
- GaussianProcessRegressionModel: GP conditioned on data with predict, sample, logMarginalLikelihood

#### Variational inference (`stoch.vi`)
- trainableNormal: Normal distribution with tf.variable() parameters and softplus constraint
- buildMeanFieldPosterior: independent Normal posteriors for multi-parameter models
- computeElbo: ELBO estimation with Monte Carlo and reparameterization trick
- fitSurrogatePosterior: optimization loop with tf.train.* optimizers (Adam, SGD)

#### Examples
- `examples/linear_regression.html`: Bayesian linear regression (y = 4x + 2) with HMC, posterior scatter/histograms, summary stats
- `examples/nuts_explorer.html`: fullscreen animated NUTS sampler exploring 2D distributions (Rosenbrock, bimodal, donut) with heatmap, sample trails, live diagnostics

#### Testing
- 1063 unit tests across 83 test suites
- Reference data tests validated against scipy.stats (Python)
- Statistical sampling tests with auto-tolerance
- MCMC posterior recovery tests (N(0,1) mean and variance)
- Memory leak tests for MCMC pipeline
