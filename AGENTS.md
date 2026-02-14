# Project briefing for awesome agents

You are the most talented and experienced agent in the world, and you have been tasked with leading the development of `stoch`, a probabilistic programming library based on TensorFlow.js. This library will enable users to define and work with probabilistic models directly in the browser, leveraging the power of TensorFlow.js for efficient computation.

## Philosophy

`stoch` abstracts away the complexities of probabilistic programming, allowing users to focus on their core logic and data. It provides a simple interface for defining probabilistic models, performing inference, and analyzing results, and takes care of the rest. By leveraging modern web technologies like TensorFlow.js, `stoch` ensures that computations are efficient and scalable.


## Style

Use **2-space indentation** and **semicolon-free** syntax. Use **single quotes** for strings. Preserve the existing style in the codebase, and avoid introducing new formatting styles or conventions.

## Quick start

```bash
npm install          # install dependencies
npm run build-dev    # fast dev build (no tests, no minification)
npm run build        # production build + full test suite
npm run test:unit    # run unit tests (fast, no browser needed)
```

## Repo map

```
src/
  index.js                          # main entry, exports {distributions, bijectors, mcmc, vi, math}
  distributions/
    index.js                        # re-exports all distributions + klDivergence
    distribution.js                 # abstract base class (tf.tidy wrapping, dispose, validation)
    normal.js                       # Normal(loc, scale)
    bernoulli.js                    # Bernoulli(probs|logits)
    uniform.js                      # Uniform(low, high)
    gamma.js                        # Gamma(concentration, rate)
    beta.js                         # Beta(concentration1, concentration0)
    exponential.js                  # Exponential(rate)
    poisson.js                      # Poisson(rate)
    categorical.js                  # Categorical(probs|logits)
    independent.js                  # Independent (batch → event dims)
    mvn_diag.js                     # MultivariateNormalDiag
    log_normal.js                   # LogNormal (via TransformedDistribution)
    student_t.js                    # StudentT(df, loc, scale)
    dirichlet.js                    # Dirichlet(concentration)
    mixture_same_family.js          # MixtureSameFamily
    joint_distribution_named.js     # JointDistributionNamed (model specification)
    joint_distribution_sequential.js # JointDistributionSequential
    transformed_distribution.js     # TransformedDistribution (Distribution + Bijector)
    kullback_leibler.js             # KL divergence registry + Normal-Normal, Bernoulli-Bernoulli
  bijectors/
    index.js                        # re-exports all bijectors
    bijector.js                     # abstract Bijector base class
    identity.js, exp.js, log.js     # element-wise bijectors
    softplus.js, sigmoid.js         # numerically stable bijectors
    shift.js, scale.js              # affine bijectors
    chain.js                        # composition (right-to-left)
  mcmc/
    index.js                        # re-exports all MCMC modules
    state_util.js                   # state↔array marshaling for tf.grads
    kernel.js                       # TransitionKernel base class
    leapfrog.js                     # symplectic integrator
    hmc.js                          # HamiltonianMonteCarlo
    nuts_util.js                    # NUTS tree-building utilities
    nuts.js                         # NoUTurnSampler (classic Algorithm 3)
    random_walk_metropolis.js       # gradient-free MCMC
    transformed_kernel.js           # bijector↔MCMC bridge
    sample_chain.js                 # main sampling loop with memory management
    dual_averaging.js               # step size adaptation
    diagnostics.js                  # ESS (Geyer 1992), R-hat
    sample.js                       # high-level sample() with auto-NUTS, multi-chain, diagnostics
    posterior_predictive.js          # posteriorPredictive, priorPredictive
  vi/
    index.js                        # re-exports all VI modules
    trainable.js                    # trainableNormal, buildMeanFieldPosterior
    elbo.js                         # ELBO computation
    fit_surrogate_posterior.js      # optimization loop
  stats/
    index.js                        # re-exports all stats modules
    hdi.js                          # Highest Density Interval
    mcse.js                         # Monte Carlo Standard Error
    summary.js                      # ArviZ-style summary (mean, sd, HDI, ESS, R-hat, MCSE)
  math/
    index.js                        # re-exports all math utilities
    numeric.js                      # constants (LOG_2PI, LOG_SQRT_2PI, etc.)
    generic.js                      # log1mexp, logAddExp, softplusInverse
    special.js                      # logGamma, digamma, logBeta, ndtr, logNdtr
    ndtri.js                        # inverse normal CDF (Acklam's algorithm)
    linalg.js                       # differentiable cholesky (tf.customGrad, Murray 2016 backward)
    triangularSolve.js              # triangular linear system solver (L·X=B, Lᵀ·X=B)
  gp/
    index.js                        # re-exports all GP modules
    kernels/
      kernel.js                     # abstract Kernel base class
      squared_exponential.js        # RBF/SE kernel
      matern.js                     # Matern (0.5, 1.5, 2.5)
      linear.js                     # Linear kernel
      periodic.js                   # Periodic (ExpSinSquared) kernel
      white.js                      # White noise kernel
      combinators.js                # Add, Product, Scale kernel combinators
      index.js                      # re-exports all kernels
    gaussian_process.js             # GaussianProcess (prior: sample, logProb, posterior)
    gp_regression.js                # GaussianProcessRegressionModel (predict, sample, logMarginalLikelihood)
  internal/
    index.js                        # re-exports internal utilities
    assert-util.js                  # assertPositive, assertNonNegative, assertInRange, assertOneOf
    dtype-util.js                   # commonDtype, isFloatDtype, isIntDtype
    tensor-util.js                  # toTensor, broadcastShapes, broadcastShapesMultiple
tests/
  unit/                             # unit tests (npm run test:unit)
    helpers.test.js                 # tolerance helper tests
    internal.test.js                # internal utility tests
    math/                           # math utility tests
    distributions/                  # distribution tests
    bijectors/                      # bijector tests
    mcmc/                           # MCMC tests (HMC, NUTS, diagnostics)
    vi/                             # VI tests (trainable, ELBO, fitting)
  helpers/
    tolerance.js                    # expectClose, expectTensorClose, autoTolerance, sampleStats
  reference-data/                   # generated JSON from scipy.stats (gitignored)
scripts/
  generate-reference-data.py        # generates reference data from scipy.stats
benchmarks/
  run.js                            # benchmark runner: stoch vs WebPPL (npm run bench)
examples/
  linear_regression.html            # Bayesian linear regression with HMC (browser)
  nuts_explorer.html                # fullscreen animated NUTS sampler visualization (browser)
docs/                               # documentation (planned)
reference/                          # reference implementations (gitignored, ~484MB)
```

## Definition of done
- Run: `npm run build` (or `npm run build-dev` during iteration)
- Run unit tests: `npm run test:unit` (fast, no browser needed)

## Constraints
- Don’t add new production dependencies without asking.
- Update the `README.md`, `AGENTS.md`, `CHANGELOG.md` with any user-facing changes or new features.
- Commit messages should be clear and descriptive, following the format: `feature|fix|test|docs: short description` (e.g., `feature: add new column type classification`). No multiline and EOF!
- Each change: for non-trivial changes, add comments, update docs (new features that can be used by users should be represented in `README.md` and `CHANGELOG.md`), run tests, commit with a clear message. For trivial changes (e.g., fixing typos), you can skip some steps but still ensure the change is well-documented in the commit message.
- If a change is breaking, stop and ask for help. If you need to make a breaking change, update the version in `package.json` and clearly document the change in `CHANGELOG.md`.
- Commit changes one at a time, with clear messages. Avoid large commits that combine multiple changes.
- Follow TensorFlow Probability’s style and conventions as closely as possible, while adapting to the JavaScript ecosystem where necessary. This includes code structure, naming conventions, and documentation style.
- Support both Node.js and browser environments, ensuring that the library can be used in a wide range of applications. CommonJS modules should be used for Node.js compatibility, and the library should be bundled appropriately for browser usage (e.g., using Webpack)
- Always check commit size in case some data or large files are accidentally included.
- Update `README.md` when running benchmarks to reflect the latest performance comparisons against WebPPL and different backends

## Testing and benchmarking
- For each feature, case compare against a reference implementation (e.g., TensorFlow Probability in Python) to ensure correctness. Use python scripts to generate test data and expected outputs, and include these in the test suite.
- For performance-sensitive features benchmark against WebPPL as the only other probabilistic programming library in JavaScript. Document performance comparisons in the `CHANGELOG.md` and `README.md` to highlight the advantages of `stoch` (parallelization, GPU acceleration, etc.).
- Use synthetic datasets for testing inference (e.g. Friedman datasets) to ensure that the library can recover known parameters and distributions when performing Baysian Inference. Include these datasets in the test suite and document the results in the `CHANGELOG.md` and `README.md`.
- Don't commit datasets, commit data-generation scripts instead, and include instructions in the `README.md` for how to generate the datasets locally.

## Conventions
- Formatting: no formatter is configured; preserve existing style (2-space indentation, semicolon-free, single quotes)
- Types: plain JavaScript with runtime type checks (`typeof`, `isNaN`) and counters
- Error handling: prefer tolerant stream processing (skip malformed lines, classify invalid/empty values as `missing`, degrade expensive stats for unsuitable columns instead of hard-failing)
