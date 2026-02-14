import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma } from '../math/special'
import { assertPositive } from '../internal/assert-util'

/**
 * Dirichlet distribution.
 *
 * Prior over simplexes (vectors that sum to 1 with all elements > 0).
 * Essential for mixture weights, topic models, and Bayesian categorical priors.
 *
 * pdf(x; α) = [Π x_k^(α_k-1)] / B(α)
 * where B(α) = Π Γ(α_k) / Γ(Σ α_k)
 *
 * Parameterized by concentration (vector of positive values).
 *
 * eventShape: [K] where K = concentration.length
 */
export class Dirichlet extends Distribution {
  constructor({ concentration, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Dirichlet'
    })

    this._concentration = this._addParameter('concentration', concentration)

    if (this._validateArgs) {
      assertPositive(this._concentration, 'concentration')
    }
  }

  get concentration() { return this._concentration }

  _eventShape() {
    // Last dim of concentration is the event dimension
    const shape = this._concentration.shape
    return [shape[shape.length - 1]]
  }

  get batchShape() {
    const shape = this._concentration.shape
    return shape.slice(0, shape.length - 1)
  }

  _sampleN(n) {
    // Sample by normalizing K independent Gamma(alpha_k, 1) samples.
    // For alpha < 1, use Ahrens-Dieter trick: Gamma(alpha+1) * U^(1/alpha)
    const concData = this._concentration.dataSync()
    const K = this._concentration.shape[this._concentration.shape.length - 1]
    const batchShape = this.batchShape
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1

    // Generate gamma samples for each component
    const sampleShape = [n, ...batchShape, K]
    const totalSamples = n * batchSize

    // Build gamma samples per component
    const gammaArrays = []
    for (let k = 0; k < K; k++) {
      const alpha = concData[k]
      let samples
      if (alpha < 1) {
        // Ahrens-Dieter: sample Gamma(alpha+1) * U^(1/alpha)
        const shape = [totalSamples]
        const g = tf.randomGamma(shape, alpha + 1, 1)
        const u = tf.randomUniform(shape)
        samples = tf.mul(g, tf.pow(u, 1 / alpha))
        g.dispose()
        u.dispose()
      } else {
        samples = tf.randomGamma([totalSamples], alpha, 1)
      }
      gammaArrays.push(samples)
    }

    // Stack and normalize
    const stacked = tf.stack(gammaArrays, 1) // [totalSamples, K]
    for (const g of gammaArrays) g.dispose()

    const sums = tf.sum(stacked, -1, true) // [totalSamples, 1]
    const normalized = tf.div(stacked, sums)
    stacked.dispose()
    sums.dispose()

    return normalized.reshape(sampleShape)
  }

  _logProb(value) {
    // logpdf = Σ (α_k - 1) * log(x_k) - logB(α)
    // logB(α) = Σ logΓ(α_k) - logΓ(Σ α_k)
    const logX = tf.log(value)
    const kernel = tf.sum(tf.mul(tf.sub(this._concentration, 1), logX), -1)
    const logBeta = this._logMultinomialBeta()
    return tf.sub(kernel, logBeta)
  }

  _entropy() {
    // H = logB(α) + (α₀ - K) * ψ(α₀) - Σ (α_k - 1) * ψ(α_k)
    const { digamma } = require('../math/special')
    const K = this._concentration.shape[this._concentration.shape.length - 1]
    const alpha0 = tf.sum(this._concentration, -1)
    const logBeta = this._logMultinomialBeta()
    const digammaAlpha0 = digamma(alpha0)
    const digammaAlpha = digamma(this._concentration)

    return tf.add(
      logBeta,
      tf.sub(
        tf.mul(tf.sub(alpha0, K), digammaAlpha0),
        tf.sum(tf.mul(tf.sub(this._concentration, 1), digammaAlpha), -1)
      )
    )
  }

  _mean() {
    // E[X_k] = α_k / α₀
    const alpha0 = tf.sum(this._concentration, -1, true)
    return tf.div(this._concentration, alpha0)
  }

  _variance() {
    // Var[X_k] = α_k * (α₀ - α_k) / (α₀² * (α₀ + 1))
    const alpha0 = tf.sum(this._concentration, -1, true)
    return tf.div(
      tf.mul(this._concentration, tf.sub(alpha0, this._concentration)),
      tf.mul(tf.square(alpha0), tf.add(alpha0, 1))
    )
  }

  _mode() {
    // mode_k = (α_k - 1) / (α₀ - K) for all α_k > 1
    const K = this._concentration.shape[this._concentration.shape.length - 1]
    const alpha0 = tf.sum(this._concentration, -1, true)
    return tf.div(
      tf.sub(this._concentration, 1),
      tf.sub(alpha0, K)
    )
  }

  /**
   * Log of the multivariate Beta function.
   * logB(α) = Σ logΓ(α_k) - logΓ(Σ α_k)
   */
  _logMultinomialBeta() {
    const sumLogGamma = tf.sum(logGamma(this._concentration), -1)
    const alpha0 = tf.sum(this._concentration, -1)
    return tf.sub(sumLogGamma, logGamma(alpha0))
  }
}
