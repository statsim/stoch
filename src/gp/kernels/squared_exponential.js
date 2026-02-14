import * as tf from '@tensorflow/tfjs'
import { Kernel } from './kernel'

/**
 * Squared Exponential (RBF) kernel.
 *
 * k(x1, x2) = amplitude² * exp(-||x1 - x2||² / (2 * lengthScale²))
 *
 * @param {Object} params
 * @param {number} [params.amplitude=1] - signal amplitude
 * @param {number} [params.lengthScale=1] - characteristic length scale
 */
export class SquaredExponential extends Kernel {
  constructor({ amplitude = 1, lengthScale = 1 } = {}) {
    super({ name: 'SquaredExponential' })
    this._amplitude = amplitude
    this._lengthScale = lengthScale
  }

  get amplitude() { return this._amplitude }
  get lengthScale() { return this._lengthScale }

  _matrix(x1, x2) {
    // x1: [n1, d], x2: [n2, d]
    const x1e = tf.expandDims(x1, 1) // [n1, 1, d]
    const x2e = tf.expandDims(x2, 0) // [1, n2, d]
    const diff = tf.sub(x1e, x2e)     // [n1, n2, d]
    const sqDist = tf.sum(tf.square(diff), -1) // [n1, n2]
    const a2 = this._amplitude * this._amplitude
    const l2 = this._lengthScale * this._lengthScale
    return tf.mul(a2, tf.exp(tf.div(sqDist, -2 * l2)))
  }

  _apply(x1, x2) {
    const diff = tf.sub(x1, x2)
    const sqDist = tf.sum(tf.square(diff), -1)
    const a2 = this._amplitude * this._amplitude
    const l2 = this._lengthScale * this._lengthScale
    return tf.mul(a2, tf.exp(tf.div(sqDist, -2 * l2)))
  }
}
