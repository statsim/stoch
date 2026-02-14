import * as tf from '@tensorflow/tfjs'
import { Kernel } from './kernel'

/**
 * Linear kernel.
 *
 * k(x1, x2) = variance * (x1 - bias)ᵀ(x2 - bias)
 *
 * @param {Object} params
 * @param {number} [params.variance=1]
 * @param {number} [params.bias=0]
 */
export class Linear extends Kernel {
  constructor({ variance = 1, bias = 0 } = {}) {
    super({ name: 'Linear' })
    this._variance = variance
    this._bias = bias
  }

  get variance() { return this._variance }
  get bias() { return this._bias }

  _matrix(x1, x2) {
    const x1c = tf.sub(x1, this._bias)
    const x2c = tf.sub(x2, this._bias)
    return tf.mul(this._variance, tf.matMul(x1c, x2c, false, true))
  }

  _apply(x1, x2) {
    const x1c = tf.sub(x1, this._bias)
    const x2c = tf.sub(x2, this._bias)
    return tf.mul(this._variance, tf.sum(tf.mul(x1c, x2c), -1))
  }
}
