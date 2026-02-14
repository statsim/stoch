import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertPositive } from '../internal/assert-util'

/**
 * Laplace distribution.
 *
 * pdf(x; loc, scale) = exp(-|x - loc| / scale) / (2 * scale)
 */
export class Laplace extends Distribution {
  constructor({ loc = 0, scale = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Laplace'
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
    const u = tf.sub(tf.randomUniform(shape, 0, 1, 'float32'), 0.5)
    // Inverse CDF: loc - scale * sign(u) * log(1 - 2|u|)
    return tf.sub(
      this._loc,
      tf.mul(this._scale, tf.mul(tf.sign(u), tf.log(tf.sub(1, tf.mul(2, tf.abs(u))))))
    )
  }

  _logProb(value) {
    // -|x - loc|/scale - log(2*scale)
    const z = tf.abs(tf.sub(value, this._loc))
    return tf.sub(
      tf.neg(tf.div(z, this._scale)),
      tf.add(Math.log(2), tf.log(this._scale))
    )
  }

  _cdf(value) {
    const z = tf.div(tf.sub(value, this._loc), this._scale)
    // 0.5 + 0.5 * sign(z) * (1 - exp(-|z|))
    return tf.add(0.5, tf.mul(0.5, tf.mul(tf.sign(z), tf.sub(1, tf.exp(tf.neg(tf.abs(z)))))))
  }

  _entropy() {
    // H = 1 + log(2 * scale)
    return tf.add(1, tf.add(Math.log(2), tf.log(this._scale)))
  }

  _mean() {
    return tf.clone(this._loc)
  }

  _variance() {
    return tf.mul(2, tf.square(this._scale))
  }

  _mode() {
    return tf.clone(this._loc)
  }
}
