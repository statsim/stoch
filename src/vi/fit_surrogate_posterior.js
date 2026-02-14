import * as tf from '@tensorflow/tfjs'

/**
 * Fit a surrogate posterior by maximizing the ELBO.
 *
 * Uses a tf.train.* optimizer (e.g., Adam) to minimize -ELBO,
 * which is equivalent to minimizing KL(q || p).
 *
 * @param {Object} params
 * @param {Function} params.targetLogProbFn - (state) → scalar Tensor (log posterior)
 * @param {Object} params.surrogatePosterior - must have sample(), logProb(),
 *   and trainableVariables array
 * @param {Object} params.optimizer - tf.train.* optimizer (e.g., tf.train.adam(0.01))
 * @param {number} params.numSteps - number of optimization steps
 * @param {number} [params.numElboSamples=1] - MC samples per ELBO estimate
 * @param {Function} [params.convergenceFn] - optional (step, loss) → bool callback;
 *   return true to stop early
 * @param {Function} [params.traceLogProbFn] - optional (step, loss) callback
 * @returns {{ surrogatePosterior, losses: number[] }}
 */
export function fitSurrogatePosterior({
  targetLogProbFn,
  surrogatePosterior,
  optimizer,
  numSteps,
  numElboSamples = 1,
  convergenceFn,
  traceLogProbFn
}) {
  const losses = []
  const trainableVars = surrogatePosterior.trainableVariables

  for (let step = 0; step < numSteps; step++) {
    // Minimize -ELBO (= maximize ELBO)
    const loss = optimizer.minimize(() => {
      let elboSum = tf.scalar(0)

      for (let i = 0; i < numElboSamples; i++) {
        const z = surrogatePosterior.sample()
        const logP = targetLogProbFn(z)
        const logQ = surrogatePosterior.logProb(z)
        const contribution = tf.sub(logP, logQ)

        const newSum = tf.add(elboSum, contribution)
        elboSum = newSum
      }

      // Average and negate (minimize -ELBO)
      const elbo = numElboSamples > 1
        ? tf.div(elboSum, numElboSamples)
        : elboSum

      return tf.neg(elbo)
    }, true, trainableVars)

    const lossVal = loss.dataSync()[0]
    losses.push(lossVal)
    loss.dispose()

    if (traceLogProbFn) {
      traceLogProbFn(step, lossVal)
    }

    if (convergenceFn && convergenceFn(step, lossVal)) {
      break
    }
  }

  return { surrogatePosterior, losses }
}
