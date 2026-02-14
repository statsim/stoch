import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma } from '../math/special'
import { assertPositive } from '../internal/assert-util'

/**
 * Poisson distribution.
 *
 * P(X = k) = rate^k * exp(-rate) / k!
 *
 * Parameterized by rate (λ > 0).
 *
 * Mirrors TFP Python's Poisson distribution.
 */
export class Poisson extends Distribution {
  constructor({ rate, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Poisson'
    })

    this._rate = this._addParameter('rate', rate)

    if (this._validateArgs) {
      assertPositive(this._rate, 'rate')
    }
  }

  get rate() { return this._rate }

  _sampleN(n) {
    // Knuth's algorithm for Poisson sampling
    // For small rates, this is efficient. For large rates, a normal
    // approximation would be better but this works for initial implementation.
    const shape = [n, ...this.batchShape]
    const rate = this._rate.dataSync()[0]
    const L = Math.exp(-rate)
    const size = shape.reduce((a, b) => a * b, 1)
    const result = new Float32Array(size)

    for (let i = 0; i < size; i++) {
      let k = 0
      let p = 1
      do {
        k++
        p *= Math.random()
      } while (p > L)
      result[i] = k - 1
    }

    return tf.tensor(result, shape)
  }

  _logProb(value) {
    // log P(k) = k * log(rate) - rate - logΓ(k + 1)
    const v = value.cast('float32')
    return tf.sub(
      tf.mul(v, tf.log(this._rate)),
      tf.add(this._rate, logGamma(tf.add(v, 1)))
    )
  }

  _cdf(value) {
    // CDF for Poisson is the regularized upper incomplete gamma function
    // For simplicity, use a cumulative sum approach for moderate values
    // This is a simplified version — exact CDF is complex
    const v = value.cast('float32')
    const rate = this._rate
    // Approximate using the normal CDF for large rates
    const { ndtr } = require('../math/special')
    return ndtr(tf.div(tf.sub(tf.add(v, 0.5), rate), tf.sqrt(rate)))
  }

  _mean() {
    return tf.clone(this._rate)
  }

  _variance() {
    return tf.clone(this._rate)
  }

  _mode() {
    return tf.floor(this._rate)
  }
}
