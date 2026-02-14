import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertPositive } from '../internal/assert-util'
import { EULER_MASCHERONI } from '../math/numeric'

/**
 * Gumbel distribution (right-skewed, type I extreme value).
 *
 * pdf(x; loc, scale) = exp(-(z + exp(-z))) / scale
 * where z = (x - loc) / scale
 */
export class Gumbel extends Distribution {
  constructor({ loc = 0, scale = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Gumbel'
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
    const u = tf.randomUniform(shape, 1e-7, 1 - 1e-7, 'float32')
    // Inverse CDF: loc - scale * log(-log(u))
    return tf.sub(this._loc, tf.mul(this._scale, tf.log(tf.neg(tf.log(u)))))
  }

  _logProb(value) {
    const z = tf.div(tf.sub(value, this._loc), this._scale)
    // -z - exp(-z) - log(scale)
    return tf.sub(tf.sub(tf.neg(z), tf.exp(tf.neg(z))), tf.log(this._scale))
  }

  _cdf(value) {
    const z = tf.div(tf.sub(value, this._loc), this._scale)
    return tf.exp(tf.neg(tf.exp(tf.neg(z))))
  }

  _entropy() {
    // H = log(scale) + 1 + γ (Euler-Mascheroni)
    return tf.add(tf.log(this._scale), 1 + EULER_MASCHERONI)
  }

  _mean() {
    // μ + scale * γ
    return tf.add(this._loc, tf.mul(this._scale, EULER_MASCHERONI))
  }

  _variance() {
    // (π * scale)² / 6
    return tf.mul(Math.PI * Math.PI / 6, tf.square(this._scale))
  }

  _mode() {
    return tf.clone(this._loc)
  }
}
