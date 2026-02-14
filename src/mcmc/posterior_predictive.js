import * as tf from '@tensorflow/tfjs'

/**
 * Generate posterior predictive samples.
 *
 * Given posterior samples and a prediction function, generates
 * one predictive sample per posterior draw.
 *
 * @param {Object} params
 * @param {Object|tf.Tensor} params.samples - posterior samples
 *   (stacked tensor [numSamples, ...] or object of stacked tensors)
 * @param {Function} params.predictFn - (singleSample) => tf.Tensor
 *   Called once per posterior sample with the unstacked sample value
 * @param {number} [params.numSamples] - number of samples to use (default: all)
 * @returns {tf.Tensor} stacked predictive samples [numSamples, ...predictShape]
 */
export function posteriorPredictive({ samples, predictFn, numSamples }) {
  const isTensor = samples instanceof tf.Tensor
  const n = isTensor ? samples.shape[0] : samples[Object.keys(samples)[0]].shape[0]
  const count = numSamples ? Math.min(numSamples, n) : n

  const predictions = []

  for (let i = 0; i < count; i++) {
    const pred = tf.tidy(() => {
      let sample
      if (isTensor) {
        sample = samples.slice(i, 1).squeeze([0])
      } else {
        sample = {}
        for (const [key, val] of Object.entries(samples)) {
          sample[key] = val.slice(i, 1).squeeze([0])
        }
      }
      return predictFn(sample)
    })
    predictions.push(pred)
  }

  const stacked = tf.stack(predictions)
  predictions.forEach(p => p.dispose())
  return stacked
}

/**
 * Generate prior predictive samples.
 *
 * Draws samples from the prior and generates predictions.
 *
 * @param {Object} params
 * @param {Function} params.priorFn - () => sample from the prior (tensor or object)
 * @param {Function} params.predictFn - (priorSample) => tf.Tensor prediction
 * @param {number} [params.numSamples=100] - number of prior draws
 * @returns {tf.Tensor} stacked predictive samples [numSamples, ...predictShape]
 */
export function priorPredictive({ priorFn, predictFn, numSamples = 100 }) {
  const predictions = []

  for (let i = 0; i < numSamples; i++) {
    const pred = tf.tidy(() => {
      const sample = priorFn()
      return predictFn(sample)
    })
    predictions.push(pred)
  }

  const stacked = tf.stack(predictions)
  predictions.forEach(p => p.dispose())
  return stacked
}
