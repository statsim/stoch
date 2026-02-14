import * as tf from '@tensorflow/tfjs'
import { cholesky } from '../math/linalg'
import { GaussianProcess } from './gaussian_process'

/**
 * Gaussian Process Regression Model.
 *
 * A GP conditioned on observed data, providing the posterior predictive
 * distribution at new test points.
 *
 * This is a convenience wrapper around GaussianProcess.posterior() that
 * stores the training data and provides sample/logProb methods at
 * prediction points.
 *
 * @param {Object} params
 * @param {Object} params.kernel - GP kernel
 * @param {tf.Tensor} params.indexPoints - training inputs, shape [n, d]
 * @param {tf.Tensor} params.observations - training targets, shape [n]
 * @param {tf.Tensor} [params.predictiveIndexPoints] - test inputs, shape [m, d]
 * @param {Function} [params.meanFn] - prior mean function
 * @param {number} [params.observationNoiseVariance=1e-6]
 * @param {number} [params.predictiveNoiseVariance=0]
 */
export class GaussianProcessRegressionModel {
  constructor({
    kernel,
    indexPoints,
    observations,
    predictiveIndexPoints,
    meanFn,
    observationNoiseVariance = 1e-6,
    predictiveNoiseVariance = 0
  }) {
    this._gp = new GaussianProcess({
      kernel,
      meanFn,
      observationNoiseVariance
    })
    this._indexPoints = indexPoints
    this._observations = observations
    this._predictiveIndexPoints = predictiveIndexPoints
    this._predictiveNoiseVariance = predictiveNoiseVariance
  }

  get kernel() { return this._gp.kernel }

  /**
   * Compute posterior mean and covariance at predictive points.
   *
   * @param {tf.Tensor} [predictPoints] - override predictiveIndexPoints
   * @returns {{ mean: tf.Tensor, covariance: tf.Tensor }}
   */
  predict(predictPoints) {
    const pts = predictPoints || this._predictiveIndexPoints
    if (!pts) throw new Error('No predictive index points provided')

    const result = this._gp.posterior(
      this._indexPoints, this._observations, pts
    )

    // Add predictive noise if specified
    if (this._predictiveNoiseVariance > 0) {
      const m = pts.shape[0]
      const noisyCov = tf.tidy(() =>
        tf.add(result.covariance, tf.mul(this._predictiveNoiseVariance, tf.eye(m)))
      )
      result.covariance.dispose()
      result.covariance = noisyCov
    }

    return result
  }

  /**
   * Sample from posterior predictive.
   *
   * @param {tf.Tensor} [predictPoints] - override predictiveIndexPoints
   * @param {number[]} [sampleShape=[]]
   * @returns {tf.Tensor}
   */
  sample(predictPoints, sampleShape = []) {
    const pts = predictPoints || this._predictiveIndexPoints
    if (!pts) throw new Error('No predictive index points provided')

    return tf.tidy(() => {
      const { mean, covariance } = this.predict(pts)
      const m = pts.shape[0]

      // Add jitter for numerical stability
      const covJitter = tf.add(covariance, tf.mul(1e-6, tf.eye(m)))
      const L = cholesky(covJitter)

      const numSamples = sampleShape.reduce((a, b) => a * b, 1)
      if (numSamples === 0 || sampleShape.length === 0) {
        const z = tf.randomNormal([m, 1])
        return tf.add(mean, tf.matMul(L, z).squeeze())
      }

      const z = tf.randomNormal([m, numSamples])
      const samples = tf.add(
        mean.expandDims(0),
        tf.transpose(tf.matMul(L, z))
      )
      return samples.reshape([...sampleShape, m])
    })
  }

  /**
   * Marginal log-likelihood of training data.
   *
   * @returns {tf.Tensor} scalar
   */
  logMarginalLikelihood() {
    return this._gp.logProb(this._indexPoints, this._observations)
  }

  dispose() {
    // Nothing to dispose (tensors are owned by caller)
  }
}
