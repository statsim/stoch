/**
 * Monte Carlo Standard Error (MCSE).
 *
 * Estimates the standard error of the posterior mean estimate,
 * accounting for autocorrelation via the effective sample size.
 *
 * MCSE = sd(samples) / sqrt(ESS)
 *
 * References:
 *   [1] Flegal, J. M. et al. (2008). "Markov chain Monte Carlo:
 *       Can we trust the third significant figure?"
 *   [2] ArviZ: arviz.mcse
 */

import { effectiveSampleSize } from '../mcmc/diagnostics'

/**
 * Compute Monte Carlo Standard Error for a 1D sample.
 *
 * @param {number[]|Float32Array} samples - 1D array of MCMC samples
 * @returns {number} MCSE estimate
 */
export function mcse(samples) {
  const n = samples.length
  if (n < 2) {
    throw new Error(`Need at least 2 samples, got ${n}`)
  }

  // Compute standard deviation
  let sum = 0
  for (let i = 0; i < n; i++) sum += samples[i]
  const mean = sum / n

  let varSum = 0
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mean
    varSum += d * d
  }
  const sd = Math.sqrt(varSum / (n - 1))

  // ESS accounts for autocorrelation
  const ess = effectiveSampleSize(samples)

  return sd / Math.sqrt(ess)
}
