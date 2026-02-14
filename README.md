# stoch

Probabilistic programming in JavaScript powered by TensorFlow.js, inspired by TensorFlow Probability and Stan.

40 distributions, 16 bijectors, MCMC (HMC + NUTS), variational inference, Gaussian processes, and convergence diagnostics — browser and Node.js, with GPU acceleration.

**[GitHub](https://github.com/statsim/stoch)** · **[npm](https://www.npmjs.com/package/stoch)**

## Install

```bash
npm install stoch @tensorflow/tfjs
```

| Backend | Package | Best for |
|---|---|---|
| CPU (JS) | `@tensorflow/tfjs` | Browser, quick prototyping |
| CPU (native) | `@tensorflow/tfjs-node` | Node.js production |
| GPU (CUDA) | `@tensorflow/tfjs-node-gpu` | Large models, GPU inference |

## Usage

```js
import * as tf from '@tensorflow/tfjs'
import stoch from 'stoch'
```

All parameters accept scalars, arrays, or tensors. Arrays/tensors create batched distributions that vectorize all operations.

## Module overview

```
stoch.distributions   40 probability distributions + KL divergence
stoch.bijectors       16 differentiable invertible transforms
stoch.mcmc            HMC, NUTS, Random Walk Metropolis, diagnostics
stoch.vi              Variational inference (ELBO, mean-field)
stoch.math            Special functions, constants, differentiable linear algebra
stoch.stats           HDI, MCSE, ArviZ-style summary
stoch.gp              Gaussian processes and kernels
```

```js
stoch.setValidateArgs(false)  // disable runtime argument validation (faster)
stoch.getValidateArgs()       // check current setting (default: true)
```

---

## Distributions

All distributions extend a common base class:

```js
const dist = new stoch.distributions.Normal({ loc: 0, scale: 1 })

dist.sample([1000])   // shape [1000]
dist.logProb(0.5)     // scalar tensor
dist.prob(0.5)        // exp(logProb(x))
dist.cdf(0.5)         // cumulative distribution function
dist.logCdf(0.5)      // log CDF (numerically stable)
dist.mean()           // distribution mean
dist.variance()       // distribution variance
dist.stddev()         // sqrt(variance())
dist.entropy()        // Shannon entropy
dist.mode()           // mode (where implemented)
dist.dispose()        // free parameter tensors
```

Batching:

```js
const dists = new stoch.distributions.Normal({ loc: [0, 1, 2], scale: 1 })
dists.sample([100])   // shape [100, 3]
dists.logProb(0.5)    // shape [3]
```

### Continuous

| Distribution | Constructor |
|---|---|
| `Normal` | `{ loc, scale }` |
| `LogNormal` | `{ loc, scale }` |
| `StudentT` | `{ df, loc, scale }` |
| `Uniform` | `{ low, high }` |
| `Beta` | `{ concentration1, concentration0 }` |
| `Gamma` | `{ concentration, rate }` |
| `Exponential` | `{ rate }` |
| `InverseGamma` | `{ concentration, scale }` |
| `Chi2` | `{ df }` |
| `Cauchy` | `{ loc, scale }` |
| `Laplace` | `{ loc, scale }` |
| `Logistic` | `{ loc, scale }` |
| `Gumbel` | `{ loc, scale }` |
| `HalfNormal` | `{ scale }` |
| `HalfCauchy` | `{ scale }` |
| `Pareto` | `{ concentration, scale }` |
| `Weibull` | `{ concentration, scale }` |
| `VonMises` | `{ loc, concentration }` |
| `TruncatedNormal` | `{ loc, scale, low, high }` |

### Discrete

| Distribution | Constructor |
|---|---|
| `Bernoulli` | `{ probs }` or `{ logits }` |
| `Categorical` | `{ probs }` or `{ logits }` |
| `Binomial` | `{ totalCount, probs }` or `{ totalCount, logits }` |
| `Poisson` | `{ rate }` |
| `Geometric` | `{ probs }` or `{ logits }` |
| `NegativeBinomial` | `{ totalCount, probs }` or `{ totalCount, logits }` |
| `Multinomial` | `{ totalCount, probs }` or `{ totalCount, logits }` |
| `OneHotCategorical` | `{ probs }` or `{ logits }` |
| `ZeroInflatedPoisson` | `{ rate, gate }` |

### Relaxed (differentiable approximations)

| Distribution | Constructor |
|---|---|
| `RelaxedBernoulli` | `{ temperature, probs }` or `{ temperature, logits }` |
| `RelaxedOneHotCategorical` | `{ temperature, probs }` or `{ temperature, logits }` |

### Multivariate

| Distribution | Constructor |
|---|---|
| `MultivariateNormalDiag` | `{ loc, scaleDiag }` |
| `MultivariateNormalTriL` | `{ loc, scaleTril }` |
| `Dirichlet` | `{ concentration }` |
| `Wishart` | `{ df, scaleTril }` |
| `LKJCholesky` | `{ dimension, concentration }` |

### Compound

| Distribution | Constructor |
|---|---|
| `Independent` | `{ distribution, reinterpretedBatchNdims }` |
| `MixtureSameFamily` | `{ mixtureDist, componentDist }` |
| `TransformedDistribution` | `{ distribution, bijector }` |

### KL divergence

```js
const p = new stoch.distributions.Normal({ loc: 0, scale: 1 })
const q = new stoch.distributions.Normal({ loc: 1, scale: 2 })
const kl = stoch.distributions.klDivergence(p, q)  // KL(p || q)
```

Built-in same-family pairs: Normal, Bernoulli, Gamma, Beta, Exponential, Dirichlet, Categorical, Laplace.

Register custom:

```js
stoch.distributions.registerKL(DistP, DistQ, (p, q) => { /* return tf.Tensor */ })
```

### Joint models

Named model with explicit deps (safe under minification):

```js
const model = new stoch.distributions.JointDistributionNamed({
  mu:    { deps: [], fn: () => new stoch.distributions.Normal({ loc: 0, scale: 10 }) },
  sigma: { deps: [], fn: () => new stoch.distributions.LogNormal({ loc: 0, scale: 1 }) },
  y:     { deps: ['mu', 'sigma'], fn: ({ mu, sigma }) =>
    new stoch.distributions.Normal({ loc: mu, scale: sigma }) }
})

model.sample()              // { mu: Tensor, sigma: Tensor, y: Tensor }
model.sample([100])         // 100 joint draws
model.logProb(values)       // scalar — joint log probability
model.logProbParts(values)  // per-component log probabilities
model.variableNames         // ['mu', 'sigma', 'y'] (topological order)
```

Shorthand (arg-name parsing, breaks under minification):

```js
const model = new stoch.distributions.JointDistributionNamed({
  mu: () => new stoch.distributions.Normal({ loc: 0, scale: 10 }),
  y:  ({ mu }) => new stoch.distributions.Normal({ loc: mu, scale: 1 })
})
```

Sequential model (positional deps, most recent first):

```js
const model = new stoch.distributions.JointDistributionSequential([
  () => new stoch.distributions.Normal({ loc: 0, scale: 1 }),
  (x0) => new stoch.distributions.Normal({ loc: x0, scale: 0.1 })
])

model.sample()              // [Tensor, Tensor]
model.logProb([x0, x1])     // scalar
```

---

## Bijectors

Differentiable invertible transforms for constrained-parameter inference and building transformed distributions.

```js
const bij = new stoch.bijectors.Exp()
bij.forward(tf.scalar(-1))               // exp(-1) ≈ 0.368
bij.inverse(tf.scalar(2))                // log(2) ≈ 0.693
bij.forwardLogDetJacobian(tf.scalar(0))  // log|det(df/dx)|
bij.inverseLogDetJacobian(tf.scalar(2))  // log|det(df⁻¹/dy)|
```

### Available bijectors

| Bijector | Transform | Use case |
|---|---|---|
| `Identity` | x | No-op |
| `Exp` | exp(x) | R → R+ |
| `Log` | log(x) | R+ → R |
| `Softplus` | log(1 + exp(x)) | Smooth R → R+ |
| `Sigmoid` | sigmoid(x) | R → (0, 1) |
| `Tanh` | tanh(x) | R → (-1, 1) |
| `Shift({ shift })` | x + shift | Location shift |
| `Scale({ scale })` | x × scale | Scaling |
| `AffineScalar({ shift, scale })` | shift + scale × x | Affine transform |
| `Power({ power })` | x^power | Power transform |
| `Invert({ bijector })` | Swaps forward/inverse | Reverse any bijector |
| `Chain({ bijectors })` | Compose right-to-left | Build pipelines |
| `Ascending` | R^d → sorted R^d | Ordered constraints |
| `SoftmaxCentered` | R^(d-1) → simplex(d) | Probability simplex |
| `FillTriangular` | R^(n(n+1)/2) → lower triangular | Matrix structure |
| `CorrelationCholesky` | R^(d(d-1)/2) → correlation Cholesky | Correlation matrices |

### Composed transforms

```js
// LogNormal = Normal + Exp
const logNormal = new stoch.distributions.TransformedDistribution({
  distribution: new stoch.distributions.Normal({ loc: 0, scale: 1 }),
  bijector: new stoch.bijectors.Exp()
})

// Compose multiple bijectors (applied right-to-left)
const chain = new stoch.bijectors.Chain({
  bijectors: [new stoch.bijectors.Exp(), new stoch.bijectors.Scale({ scale: 2 })]
})
// chain.forward(x) = exp(2 * x)
```

---

## MCMC

### High-level API — `stoch.mcmc.sample()`

Auto-configures NUTS with step-size adaptation:

```js
const { samples, diagnostics } = stoch.mcmc.sample({
  targetLogProbFn: (x) => tf.mul(-0.5, tf.square(x)),
  initialState: tf.scalar(0),
  numResults: 1000,
  numBurninSteps: 500,
  stepSize: 0.1
})
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `targetLogProbFn` | Function | **required** | `(state) => tf.Tensor` scalar log-density |
| `initialState` | Tensor/Object | **required** | Starting point. Object for multi-parameter models |
| `numResults` | number | 1000 | Samples to collect per chain |
| `numBurninSteps` | number | 500 | Warmup steps (discarded) |
| `numChains` | number | 1 | Independent chains (>=2 enables R-hat) |
| `stepSize` | number | 0.1 | Initial leapfrog step size |
| `kernel` | string | `'nuts'` | `'nuts'` or `'hmc'` |
| `maxTreeDepth` | number | 10 | NUTS max tree depth |
| `numLeapfrogSteps` | number | 10 | HMC leapfrog steps (ignored for NUTS) |
| `bijectors` | Object | — | `{ paramName: Bijector }` for constrained params |
| `numAdaptationSteps` | number | `numBurninSteps` | Step-size adaptation steps |
| `targetAcceptProb` | number | 0.8 | Target acceptance rate |
| `numStepsBetweenResults` | number | 0 | Thinning interval |
| `traceFn` | Function | — | `(state, kernelResults) => any` |

Returns `{ samples, diagnostics, trace }`. Diagnostics include `ess`, `rhat`, `numDivergent`, `numMaxDepth`, `meanLeapfrogs`.

Multi-parameter with constraints:

```js
const { samples, diagnostics } = stoch.mcmc.sample({
  targetLogProbFn: ({ mu, logSigma }) => {
    const sigma = tf.exp(logSigma)
    return tf.add(
      tf.mul(-0.5, tf.square(tf.div(mu, sigma))),
      tf.neg(logSigma)
    )
  },
  initialState: { mu: tf.scalar(0), logSigma: tf.scalar(0) },
  numResults: 1000,
  numBurninSteps: 500,
  numChains: 2,
  stepSize: 0.1,
  targetAcceptProb: 0.8
})
```

### Low-level API

Full control over kernel composition:

```js
const kernel = new stoch.mcmc.DualAveragingStepSizeAdaptation({
  innerKernel: new stoch.mcmc.TransformedTransitionKernel({
    innerKernel: new stoch.mcmc.NoUTurnSampler({
      targetLogProbFn: targetLogProb,
      stepSize: 0.1,
      maxTreeDepth: 10
    }),
    bijectors: { sigma: new stoch.bijectors.Exp() }
  }),
  numAdaptationSteps: 400,
  targetAcceptProb: 0.75
})

const { samples, trace } = stoch.mcmc.sampleChain({
  numResults: 1000,
  numBurninSteps: 500,
  currentState: { mu: tf.scalar(0), sigma: tf.scalar(1) },
  kernel,
  numStepsBetweenResults: 0,
  traceFn: (state, kr) => ({ accepted: kr.isAccepted.dataSync()[0] })
})
```

#### Kernels

| Kernel | Constructor |
|---|---|
| `NoUTurnSampler` | `{ targetLogProbFn, stepSize, maxTreeDepth, maxEnergyDiff }` |
| `HamiltonianMonteCarlo` | `{ targetLogProbFn, stepSize, numLeapfrogSteps }` |
| `RandomWalkMetropolis` | `{ targetLogProbFn, newStateProposalFn, proposalScale }` |

#### Wrappers

| Wrapper | Constructor |
|---|---|
| `TransformedTransitionKernel` | `{ innerKernel, bijectors }` |
| `DualAveragingStepSizeAdaptation` | `{ innerKernel, numAdaptationSteps, targetAcceptProb }` |

### Diagnostics

Operate on plain JS arrays (use `tensor.dataSync()`):

```js
const ess = stoch.mcmc.effectiveSampleSize(chain.dataSync())       // Geyer 1992
const rhat = stoch.mcmc.potentialScaleReduction([chain1, chain2])   // Gelman-Rubin (>=2 chains)
```

### Predictive checks

```js
// Posterior predictive: one prediction per posterior draw
const yPred = stoch.mcmc.posteriorPredictive({
  samples: posteriorSamples,    // stacked tensor [n, ...] or { param: tensor }
  predictFn: ({ slope, intercept }) => tf.add(tf.mul(slope, xNew), intercept),
  numSamples: 200               // optional, defaults to all
})

// Prior predictive
const yPrior = stoch.mcmc.priorPredictive({
  priorFn: () => ({ slope: tf.randomNormal([]), intercept: tf.randomNormal([]) }),
  predictFn: ({ slope, intercept }) => tf.add(tf.mul(slope, xNew), intercept),
  numSamples: 100               // default: 100
})
```

---

## Variational inference

### `trainableNormal({ loc, scale, name })`

Normal distribution with `tf.variable()` parameters optimized via gradient descent. Scale is parameterized internally via softplus to stay positive.

```js
const q = stoch.vi.trainableNormal({ loc: 0, scale: 1 })

q.sample()              // reparameterized: μ + σ * ε
q.sample([10])          // shape [10]
q.logProb(value)        // log N(value; μ, σ)
q.getParameters()       // { loc: number, scale: number }
q.trainableVariables    // [locVar, unconstrainedScaleVar]
q.dispose()
```

### `buildMeanFieldPosterior(initialState, { initialScale })`

One independent `trainableNormal` per parameter:

```js
const q = stoch.vi.buildMeanFieldPosterior(
  { mu: 0, sigma: 1 },
  { initialScale: 1.0 }
)

q.sample()           // { mu: Tensor, sigma: Tensor }
q.logProb(values)    // scalar — sum of independent log-probs
q.getParameters()    // { mu: { loc, scale }, sigma: { loc, scale } }
q.trainableVariables // all tf.variables
q.dispose()
```

### `computeElbo({ targetLogProbFn, surrogatePosterior, numSamples })`

ELBO = E_q[ log p(z) - log q(z) ]. Returns scalar tensor (higher is better).

```js
const elbo = stoch.vi.computeElbo({
  targetLogProbFn: (z) => tf.mul(-0.5, tf.square(z)),
  surrogatePosterior: q,
  numSamples: 10       // default: 1
})
```

### `fitSurrogatePosterior({ ... })`

Optimization loop minimizing -ELBO:

```js
const { surrogatePosterior, losses } = stoch.vi.fitSurrogatePosterior({
  targetLogProbFn: (z) => tf.mul(-0.5, tf.square(z)),
  surrogatePosterior: q,
  optimizer: tf.train.adam(0.01),
  numSteps: 1000,
  numElboSamples: 1,                          // default: 1
  convergenceFn: (step, loss) => loss < 0.01,  // optional early stop
  traceLogProbFn: (step, loss) => { ... }      // optional logging
})
// losses: number[] — loss at each step
```

---

## Stats

Summary statistics for MCMC output. All functions operate on plain JS arrays (use `tensor.dataSync()`).

```js
const [low, high] = stoch.stats.hdi(samples, 0.94)   // Highest Density Interval
const se = stoch.stats.mcse(samples)                  // Monte Carlo Standard Error

const result = stoch.stats.summary({
  mu: [chain1_mu, chain2_mu],   // multiple chains → computes R-hat
  sigma: chain1_sigma           // single chain → R-hat = NaN
}, { hdiProb: 0.94 })
// result.mu = { mean, sd, hdiLow, hdiHigh, ess, rhat, mcse }
```

---

## Gaussian processes

### Kernels

All kernels implement `matrix(x1, x2)` → kernel matrix `[n, m]`.

| Kernel | Constructor |
|---|---|
| `SquaredExponential` | `{ amplitude, lengthScale }` |
| `Matern` | `{ nu, amplitude, lengthScale }` — nu: 0.5, 1.5, or 2.5 |
| `Linear` | `{ variance, bias }` |
| `Periodic` | `{ amplitude, lengthScale, period }` |
| `White` | `{ variance }` |

Combinators: `Add(k1, k2)`, `Product(k1, k2)`, `Scale(kernel, scale)`.

```js
const kernel = new stoch.gp.Add(
  new stoch.gp.SquaredExponential({ lengthScale: 1 }),
  new stoch.gp.White({ variance: 0.1 })
)
```

### `GaussianProcess({ kernel, meanFn, observationNoiseVariance })`

GP prior over functions:

```js
const gpPrior = new stoch.gp.GaussianProcess({
  kernel: new stoch.gp.SquaredExponential({ lengthScale: 1 }),
  meanFn: (x) => tf.zeros([x.shape[0]]),   // optional, default: zero
  observationNoiseVariance: 0.01            // optional, default: 0
})

const x = tf.tensor2d([[0], [1], [2], [3], [4]])
gpPrior.sample(x, [5])                // 5 function draws, shape [5, 5]
gpPrior.logProb(x, observations)       // marginal log-likelihood
gpPrior.posterior(x, observations)     // { mean, covariance }
```

### `GaussianProcessRegressionModel({ ... })`

GP conditioned on observed data:

```js
const gprm = new stoch.gp.GaussianProcessRegressionModel({
  kernel: new stoch.gp.SquaredExponential({ amplitude: 1, lengthScale: 0.5 }),
  indexPoints: xTrain,             // [n, d] training inputs
  observations: yTrain,            // [n] training targets
  observationNoiseVariance: 0.01,  // optional, default: 1e-6
  predictiveNoiseVariance: 0,      // optional, adds noise to predictions
  predictiveIndexPoints: xTest,    // optional default test points
  meanFn: null                     // optional prior mean function
})

const { mean, covariance } = gprm.predict(xTest)
const fSamples = gprm.sample(xTest, [10])    // [10, m] posterior draws
const logML = gprm.logMarginalLikelihood()     // model selection
```

---

## Math

### Special functions

All operate on tensors (scalars auto-converted):

| Function | Description |
|---|---|
| `logGamma(x)` | Log Gamma function (Lanczos) |
| `digamma(x)` | Psi function d/dx log Gamma |
| `logBeta(a, b)` | Log Beta function |
| `ndtr(x)` | Normal CDF Phi(x) |
| `logNdtr(x)` | Numerically stable log Phi(x) |
| `ndtri(p)` | Inverse normal CDF Phi⁻¹(p) |
| `logChoose(n, k)` | Log binomial coefficient |
| `incompleteGamma(a, x)` | Returns `{ lower, upper }` |
| `incompleteBeta(a, b, x)` | Regularized incomplete beta I_x(a,b) |
| `besselI0(x)` | Modified Bessel I₀ |
| `besselI1(x)` | Modified Bessel I₁ |
| `logBesselI0(x)` | Stable log I₀ for large x |

### Numerically stable operations

| Function | Description |
|---|---|
| `log1mexp(x)` | log(1 - exp(x)) for x < 0 |
| `logAddExp(a, b)` | log(exp(a) + exp(b)) |
| `softplusInverse(x)` | log(exp(x) - 1) |

### Constants

| Constant | Value |
|---|---|
| `LOG_PI` | log(π) |
| `LOG_2` | log(2) |
| `LOG_2PI` | log(2π) |
| `LOG_SQRT_2PI` | 0.5 × log(2π) |
| `SQRT_2` | √2 |
| `SQRT_2_OVER_PI` | √(2/π) |
| `EULER_MASCHERONI` | 0.5772... |

### Differentiable linear algebra

```js
// Cholesky decomposition with custom gradient (Murray 2016)
const L = stoch.math.cholesky(A)   // L where A = LLᵀ — supports tf.grad

// Triangular linear system solver
stoch.math.triangularSolve(L, b)                         // L·X = B (default: lower=true)
stoch.math.triangularSolve(L, b, { adjoint: true })      // Lᵀ·X = B
stoch.math.triangularSolve(U, b, { lower: false })       // U·X = B
```

---

## Memory management

Distributions allocate parameter tensors. Always dispose when done:

```js
const dist = new stoch.distributions.Normal({ loc: 0, scale: 1 })
// ... use dist ...
dist.dispose()
```

Or use `tf.tidy()` for automatic cleanup of intermediates:

```js
const result = tf.tidy(() => {
  const dist = new stoch.distributions.Normal({ loc: 0, scale: 1 })
  const lp = dist.logProb(0.5)
  dist.dispose()
  return lp  // survives tf.tidy
})
```

`sampleChain` manages internal tensor lifecycle automatically. Dispose returned sample tensors when done.

---

## Performance

Benchmarked on Node.js v19.8.1, AMD Ryzen 7 5800HS, RTX 3060. WebPPL is the only other JS probabilistic programming library.

| Task | tfjs | tfjs-node | tfjs-node-gpu | WebPPL |
|---|---|---|---|---|
| Normal.logProb (100K) | 131 (1.8x) | **3,517 (52x)** | 1,808 (26x) | 71 |
| Gamma.logProb (100K) | 122 (2.0x) | **1,176 (21x)** | 405 (7x) | 60 |
| Beta.logProb (100K) | 101 (3.3x) | **502 (17x)** | 158 (6x) | 31 |
| Normal.sample (100K) | 171 | 300 | 272 | **348** |
| Exponential.sample (100K) | 230 | **1,083** | 924 | 471 |

ops/s, higher is better. **Bold** = fastest. Speedup vs WebPPL in parentheses.

Log-prob is up to **52x faster** with native backend. GPU shines on larger tensors and gradient-heavy workloads.

```bash
npm run bench          # JS CPU
npm run bench:native   # native CPU (tfjs-node)
npm run bench:gpu      # GPU (tfjs-node-gpu, requires CUDA)
```

---

## Examples

Build, then open in browser:

```bash
npm run build-dev
# open examples/*.html
```

| Example | Description |
|---|---|
| [linear_regression.html](examples/linear_regression.html) | Bayesian linear regression with HMC |
| [nuts_explorer.html](examples/nuts_explorer.html) | Animated NUTS sampler on 2D distributions |
| [visual_tests.html](examples/visual_tests.html) | 10 interactive visual tests with live controls |

---

## Development

```bash
npm install          # install dependencies
npm run build-dev    # fast dev build (no tests, no minification)
npm run build        # production build + full test suite
npm run test:unit    # 1063 tests across 83 suites
npm run bench        # benchmarks vs WebPPL
```

Reference data for distribution tests:

```bash
python3 scripts/generate-reference-data.py   # requires scipy, numpy
```

## License

Apache-2.0
