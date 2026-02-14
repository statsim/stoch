import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertPositive } from '../internal/assert-util'

/**
 * Half-Normal distribution (folded normal, x >= 0).
 *
 * pdf(x; scale) = sqrt(2/π) / scale * exp(-x² / (2*scale²))  for x >= 0
 */
export class HalfNormal extends Distribution {
  constructor({ scale = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'HalfNormal'
    })
    this._scale = this._addParameter('scale', scale)
    if (this._validateArgs) {
      assertPositive(this._scale, 'scale')
    }
  }

  get scale() { return this._scale }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    return tf.abs(tf.mul(tf.randomStandardNormal(shape, 'float32'), this._scale))
  }

  _logProb(value) {
    // log(sqrt(2/π)) - log(scale) - x²/(2*scale²)
    // = 0.5*log(2/π) - log(scale) - 0.5*(x/scale)²
    const z = tf.div(value, this._scale)
    const logProb = tf.sub(
      tf.sub(tf.scalar(0.5 * Math.log(2 / Math.PI)), tf.log(this._scale)),
      tf.mul(0.5, tf.square(z))
    )
    // Return -Infinity for x < 0
    return tf.where(tf.greaterEqual(value, 0), logProb, tf.fill(logProb.shape, -Infinity))
  }

  _cdf(value) {
    const z = tf.div(value, tf.mul(this._scale, Math.SQRT2))
    const cdfVal = tf.erf(z)
    return tf.where(tf.greaterEqual(value, 0), cdfVal, tf.zerosLike(value))
  }

  _entropy() {
    // H = 0.5*log(π*scale²/2) + 0.5 = 0.5*log(π/2) + log(scale) + 0.5
    return tf.add(0.5 * Math.log(Math.PI / 2) + 0.5, tf.log(this._scale))
  }

  _mean() {
    // scale * sqrt(2/π)
    return tf.mul(this._scale, Math.sqrt(2 / Math.PI))
  }

  _variance() {
    // scale² * (1 - 2/π)
    return tf.mul(tf.square(this._scale), 1 - 2 / Math.PI)
  }

  _mode() {
    return tf.zerosLike(this._scale)
  }
}
