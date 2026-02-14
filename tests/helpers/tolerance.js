import * as tf from '@tensorflow/tfjs'

/**
 * Assert that actual ≈ expected within combined absolute and relative tolerance.
 * Mirrors numpy/tf.debugging.assert_near: |actual - expected| <= atol + rtol * |expected|
 */
export function expectClose(actual, expected, { rtol = 1e-5, atol = 1e-6 } = {}) {
  const diff = Math.abs(actual - expected)
  const tolerance = atol + rtol * Math.abs(expected)
  if (diff > tolerance) {
    throw new Error(
      `Expected ${expected}, got ${actual} (diff=${diff}, tolerance=${tolerance}, rtol=${rtol}, atol=${atol})`
    )
  }
}

/**
 * Assert element-wise tensor closeness.
 */
export function expectTensorClose(actual, expected, { rtol = 1e-5, atol = 1e-6 } = {}) {
  const actualData = actual instanceof tf.Tensor ? actual.dataSync() : actual
  const expectedData = expected instanceof tf.Tensor ? expected.dataSync() : expected
  expect(actualData.length).toBe(expectedData.length)
  for (let i = 0; i < actualData.length; i++) {
    expectClose(actualData[i], expectedData[i], { rtol, atol })
  }
}

/**
 * Compute automatic tolerance for sample statistics based on asymptotic distributions.
 * Adapted from WebPPL test-samplers.js lines 96-132.
 *
 * Uses generous multiples (8σ for mean/variance, 100σ for higher moments) to give
 * ~1e-15 failure probability for mean/variance and account for the fact that
 * skew/kurtosis sampling distributions are only asymptotically normal.
 *
 * @param {string} statName - one of 'mean', 'variance', 'skew', 'kurtosis'
 * @param {number} n - sample size
 * @param {number} populationVariance - σ² of the distribution
 * @param {number} [moment4] - E[(X-μ)⁴], needed for variance tolerance
 * @returns {number} tolerance
 */
export function autoTolerance(statName, n, populationVariance, moment4) {
  let samplingDistVariance

  if (statName === 'mean') {
    samplingDistVariance = populationVariance / n
  } else if (statName === 'variance') {
    // Sample variance is asymptotically normal
    // http://stats.stackexchange.com/a/105338/71884
    const sigma4 = populationVariance * populationVariance
    samplingDistVariance = (moment4 || 0) / n - sigma4 * (n - 3) / (n * (n - 1))
  } else if (statName === 'skew') {
    // Assumes asymptotically normal (van der Vaart, Asymptotic Statistics p29)
    samplingDistVariance = 6 * n * (n - 1) / ((n - 2) * (n + 1) * (n + 3))
  } else if (statName === 'kurtosis') {
    samplingDistVariance = 24 * n * (n - 1) * (n - 1) /
      ((n - 3) * (n - 2) * (n + 3) * (n + 5))
  } else {
    throw new Error(`Unknown stat name: ${statName}`)
  }

  const multiple = {
    mean: 8,
    variance: 8,
    skew: 100,
    kurtosis: 100
  }

  return multiple[statName] * Math.sqrt(Math.abs(samplingDistVariance))
}

/**
 * Compute sample statistics from a Float32Array or Array of samples.
 */
export function sampleStats(samples) {
  const n = samples.length
  let sum = 0
  for (let i = 0; i < n; i++) sum += samples[i]
  const mean = sum / n

  let m2 = 0, m3 = 0, m4 = 0
  for (let i = 0; i < n; i++) {
    const d = samples[i] - mean
    const d2 = d * d
    m2 += d2
    m3 += d2 * d
    m4 += d2 * d2
  }

  const variance = m2 / (n - 1)
  const sd = Math.sqrt(variance)
  const skew = sd > 0 ? (m3 / n) / (sd * sd * sd) : 0
  const kurtosis = sd > 0 ? (m4 / n) / (variance * variance) : 0

  return { mean, variance, sd, skew, kurtosis }
}
