import * as tf from '@tensorflow/tfjs'

/**
 * Compute the Evidence Lower Bound (ELBO).
 *
 * ELBO = E_q[ log p(z) - log q(z) ]
 *
 * Where:
 *   p(z) = target (unnormalized log posterior)
 *   q(z) = surrogate posterior
 *
 * Uses Monte Carlo estimation with the reparameterization trick
 * for low-variance gradients.
 *
 * @param {Object} params
 * @param {Function} params.targetLogProbFn - (state) → scalar Tensor
 *   The unnormalized log posterior log p(z).
 * @param {Object} params.surrogatePosterior - must have sample() and logProb()
 * @param {number} [params.numSamples=1] - MC samples for gradient estimation
 * @returns {tf.Tensor} scalar ELBO estimate (higher is better)
 */
export function computeElbo({ targetLogProbFn, surrogatePosterior, numSamples = 1 }) {
  let elboSum = tf.scalar(0)

  for (let i = 0; i < numSamples; i++) {
    // Sample z ~ q(z) using reparameterization trick
    const z = surrogatePosterior.sample()

    // log p(z)
    const logP = targetLogProbFn(z)

    // log q(z)
    const logQ = surrogatePosterior.logProb(z)

    // ELBO contribution: log p(z) - log q(z)
    const contribution = tf.sub(logP, logQ)
    const newSum = tf.add(elboSum, contribution)

    // Clean up intermediates
    elboSum.dispose()
    contribution.dispose()
    logP.dispose()
    logQ.dispose()
    if (z instanceof tf.Tensor) {
      z.dispose()
    } else {
      for (const v of Object.values(z)) {
        if (v instanceof tf.Tensor) v.dispose()
      }
    }

    elboSum = newSum
  }

  // Average over samples
  if (numSamples > 1) {
    const avg = tf.div(elboSum, numSamples)
    elboSum.dispose()
    return avg
  }

  return elboSum
}
