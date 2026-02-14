import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { LOG_2PI } from '../math/numeric'
import { ndtr, logNdtr } from '../math/special'
import { assertPositive } from '../internal/assert-util'

/**
 * Normal (Gaussian) distribution.
 *
 * The probability density function:
 *   pdf(x; μ, σ) = exp(-0.5 * ((x - μ) / σ)²) / (σ * sqrt(2π))
 *
 * Parameterized by loc (mean μ) and scale (standard deviation σ > 0).
 *
 * Mirrors TFP Python's Normal distribution.
 */
export class Normal extends Distribution {
  constructor({ loc = 0, scale = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Normal'
    })

    this._loc = this._addParameter('loc', loc)
    this._scale = this._addParameter('scale', scale)

    if (this._validateArgs) {
      assertPositive(this._scale, 'scale')
    }
  }

  get loc() { return this._loc }
  get scale() { return this._scale }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    return tf.add(
      tf.mul(tf.randomStandardNormal(shape, 'float32'), this._scale),
      this._loc
    )
  }

  _logProb(value) {
    const z = this._standardize(value)
    return tf.sub(
      tf.mul(-0.5, tf.add(tf.square(z), LOG_2PI)),
      tf.log(this._scale)
    )
  }

  _cdf(value) {
    return ndtr(this._standardize(value))
  }

  _logCdf(value) {
    return logNdtr(this._standardize(value))
  }

  _entropy() {
    // H = 0.5 * log(2πe * σ²) = 0.5 * (LOG_2PI + 1) + log(σ)
    return tf.add(0.5 * (LOG_2PI + 1), tf.log(this._scale))
  }

  _mean() {
    return tf.clone(this._loc)
  }

  _variance() {
    return tf.square(this._scale)
  }

  _stddev() {
    return tf.clone(this._scale)
  }

  _mode() {
    return tf.clone(this._loc)
  }

  /**
   * Standardize: z = (x - μ) / σ
   */
  _standardize(value) {
    return tf.div(tf.sub(value, this._loc), this._scale)
  }
}
