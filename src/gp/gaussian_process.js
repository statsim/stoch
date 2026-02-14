import * as tf from '@tensorflow/tfjs'
import { cholesky } from '../math/linalg'
import { triangularSolve } from '../math/triangularSolve'

/**
 * Gaussian Process prior distribution.
 *
 * GP(meanFn, kernel) defines a distribution over functions.
 * Given index points X, the marginal distribution is:
 *   f(X) ~ N(m(X), K(X, X))
 *
 * @param {Object} params
 * @param {Object} params.kernel - GP kernel (covariance function)
 * @param {Function} [params.meanFn] - mean function m(x) → tensor, default zero
 * @param {number} [params.observationNoiseVariance=0] - σ² added to diagonal
 */
export class GaussianProcess {
  constructor({ kernel, meanFn, observationNoiseVariance = 0 }) {
    this._kernel = kernel
    this._meanFn = meanFn || ((x) => tf.zeros([x.shape[0]]))
    this._noiseVar = observationNoiseVariance
  }

  get kernel() { return this._kernel }
  get observationNoiseVariance() { return this._noiseVar }

  /**
   * Compute the marginal log probability of observations at given index points.
   *
   * log p(y | X) = -0.5 * (yᵀ K⁻¹ y + log|K| + n*log(2π))
   *
   * @param {tf.Tensor} indexPoints - shape [n, d]
   * @param {tf.Tensor} observations - shape [n]
   * @returns {tf.Tensor} scalar log probability
   */
  logProb(indexPoints, observations) {
    return tf.tidy(() => {
      const n = indexPoints.shape[0]
      const mean = this._meanFn(indexPoints)
      const y = tf.sub(observations, mean) // centered

      // Kernel matrix + noise
      let K = this._kernel.matrix(indexPoints, indexPoints)
      if (this._noiseVar > 0) {
        K = tf.add(K, tf.mul(this._noiseVar, tf.eye(n)))
      }

      // Cholesky decomposition
      const L = cholesky(K)

      // Solve L * alpha_half = y => alpha_half
      // Then L^T * alpha = alpha_half => alpha = K^{-1} y
      // Use forward substitution via triangularSolve
      const yCol = y.reshape([n, 1])
      const alphaHalf = triangularSolve(L, yCol, { lower: true })
      const alpha = triangularSolve(L, alphaHalf, { lower: true, adjoint: true })

      // y^T K^{-1} y = alpha^T y
      const quadForm = tf.sum(tf.mul(alpha.squeeze(), y))

      // log|K| = 2 * sum(log(diag(L)))
      const diagL = []
      const Ldata = L.dataSync()
      for (let i = 0; i < n; i++) {
        diagL.push(Math.log(Math.abs(Ldata[i * n + i])))
      }
      const logDetVal = 2 * diagL.reduce((a, b) => a + b, 0)

      const logProb = tf.scalar(
        -0.5 * (quadForm.dataSync()[0] + logDetVal + n * Math.log(2 * Math.PI))
      )
      return logProb
    })
  }

  /**
   * Sample function values at index points.
   *
   * @param {tf.Tensor} indexPoints - shape [n, d]
   * @param {number[]} [sampleShape=[]] - number of samples
   * @returns {tf.Tensor} shape [...sampleShape, n]
   */
  sample(indexPoints, sampleShape = []) {
    return tf.tidy(() => {
      const n = indexPoints.shape[0]
      const mean = this._meanFn(indexPoints)

      let K = this._kernel.matrix(indexPoints, indexPoints)
      if (this._noiseVar > 0) {
        K = tf.add(K, tf.mul(this._noiseVar, tf.eye(n)))
      }

      // Add small jitter for numerical stability
      K = tf.add(K, tf.mul(1e-6, tf.eye(n)))

      const L = cholesky(K)

      const numSamples = sampleShape.reduce((a, b) => a * b, 1)
      if (numSamples === 0 || sampleShape.length === 0) {
        // Single sample
        const z = tf.randomNormal([n, 1])
        const sample = tf.add(mean, tf.matMul(L, z).squeeze())
        return sample
      }

      // Multiple samples
      const z = tf.randomNormal([n, numSamples])
      const samples = tf.add(
        mean.expandDims(0),
        tf.transpose(tf.matMul(L, z))
      ) // [numSamples, n]
      return samples.reshape([...sampleShape, n])
    })
  }

  /**
   * Compute posterior mean and covariance given observations.
   *
   * @param {tf.Tensor} indexPoints - training points, shape [n, d]
   * @param {tf.Tensor} observations - training observations, shape [n]
   * @param {tf.Tensor} predictPoints - test points, shape [m, d]
   * @returns {{ mean: tf.Tensor, covariance: tf.Tensor }}
   *   mean: shape [m], covariance: shape [m, m]
   */
  posterior(indexPoints, observations, predictPoints) {
    return tf.tidy(() => {
      const n = indexPoints.shape[0]
      const mean = this._meanFn(indexPoints)
      const y = tf.sub(observations, mean)

      // K(X, X) + σ²I
      let Kxx = this._kernel.matrix(indexPoints, indexPoints)
      if (this._noiseVar > 0) {
        Kxx = tf.add(Kxx, tf.mul(this._noiseVar, tf.eye(n)))
      }
      Kxx = tf.add(Kxx, tf.mul(1e-6, tf.eye(n)))

      // K(X*, X), K(X*, X*)
      const Ksx = this._kernel.matrix(predictPoints, indexPoints) // [m, n]
      const Kss = this._kernel.matrix(predictPoints, predictPoints) // [m, m]

      // L = cholesky(Kxx)
      const L = cholesky(Kxx)

      // alpha = Kxx^{-1} y
      const yCol = y.reshape([n, 1])
      const alphaHalf = triangularSolve(L, yCol, { lower: true })
      const alpha = triangularSolve(L, alphaHalf, { lower: true, adjoint: true })

      // Posterior mean: m* + K(X*, X) @ alpha
      const predMean = this._meanFn(predictPoints)
      const posteriorMean = tf.add(predMean, tf.matMul(Ksx, alpha).squeeze())

      // v = L^{-1} K(X, X*)
      const KxsT = tf.transpose(Ksx) // [n, m]
      const v = triangularSolve(L, KxsT, { lower: true })

      // Posterior covariance: K(X*, X*) - v^T v
      const posteriorCov = tf.sub(Kss, tf.matMul(tf.transpose(v), v))

      return { mean: posteriorMean, covariance: posteriorCov }
    })
  }
}
