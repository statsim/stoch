import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertPositive } from '../internal/assert-util'

/**
 * Half-Cauchy distribution (folded Cauchy, x >= 0).
 *
 * pdf(x; scale) = 2 / (π * scale * (1 + (x/scale)²))  for x >= 0
 */
export class HalfCauchy extends Distribution {
  constructor({ scale = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'HalfCauchy'
    })
    this._scale = this._addParameter('scale', scale)
    if (this._validateArgs) {
      assertPositive(this._scale, 'scale')
    }
  }

  get scale() { return this._scale }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    const u = tf.randomUniform(shape, 0, 1, 'float32')
    // Inverse CDF: scale * tan(π/2 * u)
    return tf.mul(this._scale, tf.tan(tf.mul(Math.PI / 2, u)))
  }

  _logProb(value) {
    const z = tf.div(value, this._scale)
    // log(2/π) - log(scale) - log(1 + z²)
    const logProb = tf.sub(
      tf.sub(tf.scalar(Math.log(2 / Math.PI)), tf.log(this._scale)),
      tf.log(tf.add(1, tf.square(z)))
    )
    return tf.where(tf.greaterEqual(value, 0), logProb, tf.fill(logProb.shape, -Infinity))
  }

  _cdf(value) {
    const z = tf.div(value, this._scale)
    const cdfVal = tf.mul(2 / Math.PI, tf.atan(z))
    return tf.where(tf.greaterEqual(value, 0), cdfVal, tf.zerosLike(value))
  }

  _entropy() {
    // H = log(2π * scale)
    return tf.add(Math.log(2 * Math.PI), tf.log(this._scale))
  }

  _mean() {
    return tf.fill(this.batchShape, NaN)
  }

  _variance() {
    return tf.fill(this.batchShape, NaN)
  }

  _mode() {
    return tf.zerosLike(this._scale)
  }
}
