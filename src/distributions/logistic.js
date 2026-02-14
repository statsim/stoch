import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertPositive } from '../internal/assert-util'

/**
 * Logistic distribution.
 *
 * pdf(x; loc, scale) = exp(-z) / (scale * (1 + exp(-z))²)
 * where z = (x - loc) / scale
 */
export class Logistic extends Distribution {
  constructor({ loc = 0, scale = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Logistic'
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
    // Inverse CDF: loc + scale * log(u / (1 - u))
    return tf.add(this._loc, tf.mul(this._scale, tf.log(tf.div(u, tf.sub(1, u)))))
  }

  _logProb(value) {
    const z = tf.div(tf.sub(value, this._loc), this._scale)
    // -z - log(scale) - 2 * log(1 + exp(-z))
    return tf.sub(
      tf.sub(tf.neg(z), tf.log(this._scale)),
      tf.mul(2, tf.softplus(tf.neg(z)))
    )
  }

  _cdf(value) {
    const z = tf.div(tf.sub(value, this._loc), this._scale)
    return tf.sigmoid(z)
  }

  _entropy() {
    // H = log(scale) + 2
    return tf.add(tf.log(this._scale), 2)
  }

  _mean() {
    return tf.clone(this._loc)
  }

  _variance() {
    // Var = (π * scale)² / 3
    return tf.mul(Math.PI * Math.PI / 3, tf.square(this._scale))
  }

  _mode() {
    return tf.clone(this._loc)
  }
}
