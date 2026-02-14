import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertPositive } from '../internal/assert-util'

/**
 * Cauchy distribution.
 *
 * pdf(x; loc, scale) = 1 / (π * scale * (1 + ((x - loc) / scale)²))
 *
 * Has undefined mean and variance (heavy tails).
 */
export class Cauchy extends Distribution {
  constructor({ loc = 0, scale = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Cauchy'
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
    const u = tf.randomUniform(shape, 0, 1, 'float32')
    // Inverse CDF: loc + scale * tan(π * (u - 0.5))
    return tf.add(this._loc, tf.mul(this._scale, tf.tan(tf.mul(Math.PI, tf.sub(u, 0.5)))))
  }

  _logProb(value) {
    const z = tf.div(tf.sub(value, this._loc), this._scale)
    // -log(π) - log(scale) - log(1 + z²)
    return tf.sub(
      tf.sub(tf.scalar(-Math.log(Math.PI)), tf.log(this._scale)),
      tf.log(tf.add(1, tf.square(z)))
    )
  }

  _cdf(value) {
    const z = tf.div(tf.sub(value, this._loc), this._scale)
    return tf.add(0.5, tf.div(tf.atan(z), Math.PI))
  }

  _entropy() {
    // H = log(4π * scale)
    return tf.add(Math.log(4 * Math.PI), tf.log(this._scale))
  }

  _mean() {
    return tf.fill(this.batchShape, NaN)
  }

  _variance() {
    return tf.fill(this.batchShape, NaN)
  }

  _mode() {
    return tf.clone(this._loc)
  }
}
