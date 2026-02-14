import * as tf from '@tensorflow/tfjs'
import { Kernel } from './kernel'

/**
 * Periodic (ExpSinSquared) kernel.
 *
 * k(x1, x2) = a² * exp(-2 * sin²(π * ||x1 - x2|| / period) / l²)
 *
 * @param {Object} params
 * @param {number} [params.amplitude=1]
 * @param {number} [params.lengthScale=1]
 * @param {number} [params.period=1]
 */
export class Periodic extends Kernel {
  constructor({ amplitude = 1, lengthScale = 1, period = 1 } = {}) {
    super({ name: 'Periodic' })
    this._amplitude = amplitude
    this._lengthScale = lengthScale
    this._period = period
  }

  get amplitude() { return this._amplitude }
  get lengthScale() { return this._lengthScale }
  get period() { return this._period }

  _computeFromDist(dist) {
    const a2 = this._amplitude * this._amplitude
    const l2 = this._lengthScale * this._lengthScale
    const sinArg = tf.mul(Math.PI / this._period, dist)
    const sin2 = tf.square(tf.sin(sinArg))
    return tf.mul(a2, tf.exp(tf.div(tf.mul(-2, sin2), l2)))
  }

  _matrix(x1, x2) {
    const x1e = tf.expandDims(x1, 1)
    const x2e = tf.expandDims(x2, 0)
    const diff = tf.sub(x1e, x2e)
    const dist = tf.sqrt(tf.add(tf.sum(tf.square(diff), -1), 1e-12))
    return this._computeFromDist(dist)
  }

  _apply(x1, x2) {
    const diff = tf.sub(x1, x2)
    const dist = tf.sqrt(tf.add(tf.sum(tf.square(diff), -1), 1e-12))
    return this._computeFromDist(dist)
  }
}
