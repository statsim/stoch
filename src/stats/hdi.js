/**
 * Highest Density Interval (HDI).
 *
 * Computes the narrowest interval containing a given probability mass.
 * This is the Bayesian credible interval analog.
 *
 * References:
 *   [1] Kruschke, J. K. (2014). "Doing Bayesian Data Analysis."
 *   [2] ArviZ: arviz.hdi
 */

/**
 * Compute the Highest Density Interval for a 1D sample.
 *
 * Algorithm: sort samples, then slide a window of size ceil(n * prob)
 * across the sorted array and pick the window with the smallest width.
 *
 * @param {number[]|Float32Array} samples - 1D array of samples
 * @param {number} [prob=0.94] - probability mass to include (0 < prob < 1)
 * @returns {[number, number]} [low, high] bounds of the HDI
 */
export function hdi(samples, prob = 0.94) {
  if (prob <= 0 || prob >= 1) {
    throw new Error(`prob must be in (0, 1), got ${prob}`)
  }

  const n = samples.length
  if (n < 2) {
    throw new Error(`Need at least 2 samples, got ${n}`)
  }

  // Sort a copy
  const sorted = Array.from(samples).sort((a, b) => a - b)

  // Window size: number of samples in the HDI
  const windowSize = Math.ceil(n * prob)

  if (windowSize >= n) {
    return [sorted[0], sorted[n - 1]]
  }

  // Slide window and find narrowest
  let bestWidth = Infinity
  let bestLow = sorted[0]
  let bestHigh = sorted[windowSize - 1]

  for (let i = 0; i <= n - windowSize; i++) {
    const width = sorted[i + windowSize - 1] - sorted[i]
    if (width < bestWidth) {
      bestWidth = width
      bestLow = sorted[i]
      bestHigh = sorted[i + windowSize - 1]
    }
  }

  return [bestLow, bestHigh]
}
