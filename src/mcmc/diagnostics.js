/**
 * MCMC convergence diagnostics.
 *
 * - effectiveSampleSize: accounts for autocorrelation using the
 *   initial positive sequence estimator (Geyer 1992).
 * - potentialScaleReduction: Gelman-Rubin R-hat across multiple chains.
 *
 * Both functions operate on plain JS arrays (from tensor.dataSync()).
 *
 * References:
 *   [1] Geyer, C. J. (1992). "Practical Markov Chain Monte Carlo."
 *   [2] Gelman, Rubin (1992). "Inference from Iterative Simulation
 *       Using Multiple Sequences."
 *   [3] Vehtari et al. (2021). "Rank-normalization, folding, and
 *       localization: An improved R-hat."
 */

/**
 * Estimate effective sample size using the initial positive
 * sequence estimator (Geyer 1992).
 *
 * For a chain of length N with autocorrelation R[k]:
 *   ESS = N / (1 + 2 * Σ R[k])
 *
 * The sum is truncated at the first negative pair of consecutive
 * autocorrelations, which is robust for reversible MCMC chains.
 *
 * @param {number[]|Float32Array} samples - 1D array of MCMC samples
 * @returns {number} estimated effective sample size
 */
export function effectiveSampleSize(samples) {
  const n = samples.length
  if (n < 4) return n

  // Compute sample mean and variance
  let sum = 0
  for (let i = 0; i < n; i++) sum += samples[i]
  const mean = sum / n

  let varSum = 0
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mean
    varSum += d * d
  }
  const variance = varSum / n

  if (variance < 1e-30) return n // constant chain

  // Compute autocorrelation and use initial positive sequence estimator
  // Sum consecutive pairs of autocorrelations and stop when a pair is negative
  let tauSum = 1.0 // Start with R[0] = 1
  const maxLag = Math.floor(n / 2)

  for (let lag = 1; lag < maxLag; lag += 2) {
    // Compute R[lag] and R[lag+1]
    const r1 = autoCorrelation(samples, mean, variance, lag)
    const r2 = lag + 1 < n ? autoCorrelation(samples, mean, variance, lag + 1) : 0

    const pairSum = r1 + r2

    // Initial positive sequence: stop at first non-positive pair
    if (pairSum <= 0) break

    tauSum += 2 * pairSum
  }

  // ESS = N / tau, clamp to [1, N]
  const ess = n / tauSum
  return Math.max(1, Math.min(n, ess))
}

/**
 * Compute autocorrelation at a given lag.
 * R[k] = (1/N) * Σ_{t=1}^{N-k} (x_t - mean)(x_{t+k} - mean) / variance
 */
function autoCorrelation(samples, mean, variance, lag) {
  const n = samples.length
  let sum = 0
  for (let i = 0; i < n - lag; i++) {
    sum += (samples[i] - mean) * (samples[i + lag] - mean)
  }
  return (sum / n) / variance
}

/**
 * Compute Gelman-Rubin potential scale reduction factor (R-hat).
 *
 * Measures convergence across multiple chains by comparing within-chain
 * variance to between-chain variance. R-hat ≈ 1 indicates convergence.
 * Values > 1.1 suggest insufficient convergence.
 *
 * @param {number[][]} chains - array of chains, each a 1D array of samples.
 *   All chains must have the same length.
 * @returns {number} R-hat statistic
 */
export function potentialScaleReduction(chains) {
  const numChains = chains.length
  if (numChains < 2) {
    throw new Error('potentialScaleReduction requires at least 2 chains')
  }

  const n = chains[0].length
  if (n < 2) {
    throw new Error('Each chain must have at least 2 samples')
  }

  // Compute chain means and variances
  const chainMeans = new Array(numChains)
  const chainVars = new Array(numChains)

  for (let c = 0; c < numChains; c++) {
    const chain = chains[c]
    let sum = 0
    for (let i = 0; i < n; i++) sum += chain[i]
    chainMeans[c] = sum / n

    let varSum = 0
    for (let i = 0; i < n; i++) {
      const d = chain[i] - chainMeans[c]
      varSum += d * d
    }
    chainVars[c] = varSum / (n - 1) // unbiased within-chain variance
  }

  // W: mean of within-chain variances
  let w = 0
  for (let c = 0; c < numChains; c++) w += chainVars[c]
  w /= numChains

  // B/n: variance of chain means (between-chain variance / n)
  let overallMean = 0
  for (let c = 0; c < numChains; c++) overallMean += chainMeans[c]
  overallMean /= numChains

  let bDivN = 0
  for (let c = 0; c < numChains; c++) {
    const d = chainMeans[c] - overallMean
    bDivN += d * d
  }
  bDivN /= (numChains - 1)

  // Estimated variance: σ²_+ = (n-1)/n * W + B/n
  const sigma2Plus = ((n - 1) / n) * w + bDivN

  // R-hat = sqrt( ((m+1)/m * σ²_+ / W) - (n-1)/(m*n) )
  const rhat = ((numChains + 1) / numChains) * sigma2Plus / w
    - (n - 1) / (numChains * n)

  return Math.sqrt(Math.max(rhat, 0))
}
