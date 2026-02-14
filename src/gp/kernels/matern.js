import * as tf from '@tensorflow/tfjs'
import { Kernel } from './kernel'

/**
 * Matern kernel family.
 *
 * Supports nu = 0.5 (exponential), 1.5, and 2.5.
 *
 * k_{1/2}(r) = a² * exp(-r/l)
 * k_{3/2}(r) = a² * (1 + √3*r/l) * exp(-√3*r/l)
 * k_{5/2}(r) = a² * (1 + √5*r/l + 5r²/(3l²)) * exp(-√5*r/l)
 *
 * where r = ||x1 - x2||
 *
 * @param {Object} params
 * @param {number} [params.nu=2.5] - smoothness (0.5, 1.5, or 2.5)
 * @param {number} [params.amplitude=1]
 * @param {number} [params.lengthScale=1]
 */
export class Matern extends Kernel {
  constructor({ nu = 2.5, amplitude = 1, lengthScale = 1 } = {}) {
    super({ name: `Matern${nu}` })
    if (![0.5, 1.5, 2.5].includes(nu)) {
      throw new Error(`Matern nu must be 0.5, 1.5, or 2.5, got ${nu}`)
    }
    this._nu = nu
    this._amplitude = amplitude
    this._lengthScale = lengthScale
  }

  get nu() { return this._nu }
  get amplitude() { return this._amplitude }
  get lengthScale() { return this._lengthScale }

  _computeFromDist(sqDist) {
    const a2 = this._amplitude * this._amplitude
    const l = this._lengthScale
    const r = tf.sqrt(tf.add(sqDist, 1e-12)) // add eps to avoid sqrt(0) gradient issues

    if (this._nu === 0.5) {
      return tf.mul(a2, tf.exp(tf.div(tf.neg(r), l)))
    }

    if (this._nu === 1.5) {
      const s3r = tf.mul(Math.sqrt(3), tf.div(r, l))
      return tf.mul(a2, tf.mul(tf.add(1, s3r), tf.exp(tf.neg(s3r))))
    }

    // nu = 2.5
    const s5r = tf.mul(Math.sqrt(5), tf.div(r, l))
    const r2l2 = tf.div(sqDist, l * l)
    return tf.mul(a2, tf.mul(
      tf.add(tf.add(1, s5r), tf.mul(5 / 3, r2l2)),
      tf.exp(tf.neg(s5r))
    ))
  }

  _matrix(x1, x2) {
    const x1e = tf.expandDims(x1, 1)
    const x2e = tf.expandDims(x2, 0)
    const sqDist = tf.sum(tf.square(tf.sub(x1e, x2e)), -1)
    return this._computeFromDist(sqDist)
  }

  _apply(x1, x2) {
    const sqDist = tf.sum(tf.square(tf.sub(x1, x2)), -1)
    return this._computeFromDist(sqDist)
  }
}
