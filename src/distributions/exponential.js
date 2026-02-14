import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertPositive } from '../internal/assert-util'

/**
 * Exponential distribution.
 *
 * pdf(x; λ) = λ * exp(-λx) for x >= 0
 *
 * Parameterized by rate (λ > 0).
 * Special case of Gamma(concentration=1, rate).
 *
 * Mirrors TFP Python's Exponential distribution.
 */
export class Exponential extends Distribution {
  constructor({ rate, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Exponential'
    })

    this._rate = this._addParameter('rate', rate)

    if (this._validateArgs) {
      assertPositive(this._rate, 'rate')
    }
  }

  get rate() { return this._rate }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    // Inverse CDF method: -log(U) / rate
    return tf.div(
      tf.neg(tf.log(tf.randomUniform(shape))),
      this._rate
    )
  }

  _logProb(value) {
    // log pdf = log(rate) - rate * x
    return tf.sub(tf.log(this._rate), tf.mul(this._rate, value))
  }

  _cdf(value) {
    // CDF = 1 - exp(-rate * x)
    return tf.sub(1, tf.exp(tf.neg(tf.mul(this._rate, value))))
  }

  _entropy() {
    // H = 1 - log(rate)
    return tf.sub(1, tf.log(this._rate))
  }

  _mean() {
    return tf.reciprocal(this._rate)
  }

  _variance() {
    return tf.reciprocal(tf.square(this._rate))
  }

  _mode() {
    return tf.zerosLike(this._rate)
  }
}
