/**
 * Summary statistics for MCMC chains.
 *
 * Produces an ArviZ-style summary table with mean, std, HDI,
 * ESS, R-hat, and MCSE for each parameter.
 *
 * References:
 *   [1] ArviZ: arviz.summary
 */

import { effectiveSampleSize, potentialScaleReduction } from '../mcmc/diagnostics'
import { hdi } from './hdi'
import { mcse } from './mcse'

/**
 * Compute summary statistics for MCMC samples.
 *
 * @param {Object} chains - Object mapping parameter names to arrays of chains.
 *   Each value is either:
 *   - A single chain: number[] or Float32Array
 *   - Multiple chains: number[][] (array of arrays)
 * @param {Object} [options]
 * @param {number} [options.hdiProb=0.94] - HDI probability mass
 * @returns {Object} Map of parameter name → { mean, sd, hdiLow, hdiHigh, ess, rhat, mcse }
 */
export function summary(chains, { hdiProb = 0.94 } = {}) {
  const result = {}

  for (const [name, value] of Object.entries(chains)) {
    // Normalize to array of chains
    const isMultiChain = Array.isArray(value[0])
    const chainArrays = isMultiChain ? value : [value]

    // Combine all chains for point estimates
    const allSamples = []
    for (const chain of chainArrays) {
      for (let i = 0; i < chain.length; i++) {
        allSamples.push(chain[i])
      }
    }

    const n = allSamples.length
    if (n < 2) {
      result[name] = { mean: allSamples[0] || NaN, sd: NaN, hdiLow: NaN, hdiHigh: NaN, ess: n, rhat: NaN, mcse: NaN }
      continue
    }

    // Mean
    let sum = 0
    for (let i = 0; i < n; i++) sum += allSamples[i]
    const mean = sum / n

    // Std dev
    let varSum = 0
    for (let i = 0; i < n; i++) {
      const d = allSamples[i] - mean
      varSum += d * d
    }
    const sd = Math.sqrt(varSum / (n - 1))

    // HDI
    const [hdiLow, hdiHigh] = hdi(allSamples, hdiProb)

    // ESS (from first chain or combined)
    const ess = effectiveSampleSize(allSamples)

    // R-hat (only if multiple chains)
    let rhat = NaN
    if (isMultiChain && chainArrays.length >= 2) {
      rhat = potentialScaleReduction(chainArrays)
    }

    // MCSE
    const mcseVal = mcse(allSamples)

    result[name] = { mean, sd, hdiLow, hdiHigh, ess, rhat, mcse: mcseVal }
  }

  return result
}
