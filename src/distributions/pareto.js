import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertPositive } from '../internal/assert-util'

/**
 * Pareto distribution.
 *
 * pdf(x; α, xm) = α * xm^α / x^(α+1)  for x >= xm
 *
 * Parameterized by concentration (shape α > 0) and scale (minimum value xm > 0).
 *
 * Mirrors TFP Python's Pareto distribution.
 */
export class Pareto extends Distribution {
  constructor({ concentration, scale, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Pareto'
    })

    this._concentration = this._addParameter('concentration', concentration)
    this._scale = this._addParameter('scale', scale)

    if (this._validateArgs) {
      assertPositive(this._concentration, 'concentration')
      assertPositive(this._scale, 'scale')
    }
  }

  get concentration() { return this._concentration }
  get scale() { return this._scale }

  _sampleN(n) {
    // Sample: scale / U^(1/alpha) where U ~ Uniform(0, 1)
    const shape = [n, ...this.batchShape]
    const u = tf.randomUniform(shape)
    const invAlpha = tf.reciprocal(this._concentration)
    return tf.div(this._scale, tf.pow(u, invAlpha))
  }

  _logProb(value) {
    // log pdf = log(α) + α*log(xm) - (α+1)*log(x), -Infinity for x < xm
    const a = this._concentration
    const xm = this._scale
    const logP = tf.sub(
      tf.add(tf.log(a), tf.mul(a, tf.log(xm))),
      tf.mul(tf.add(a, 1), tf.log(value))
    )
    const inSupport = value.greaterEqual(xm)
    return tf.where(inSupport, logP, tf.fill(value.shape, -Infinity))
  }

  _cdf(value) {
    // CDF = 1 - (xm/x)^α for x >= xm, 0 otherwise
    const a = this._concentration
    const xm = this._scale
    const raw = tf.sub(1, tf.pow(tf.div(xm, value), a))
    const inSupport = value.greaterEqual(xm)
    return tf.where(inSupport, raw, tf.zerosLike(value))
  }

  _entropy() {
    // H = log(xm / α) + 1 + 1/α
    const a = this._concentration
    const xm = this._scale
    return tf.add(
      tf.log(tf.div(xm, a)),
      tf.add(1, tf.reciprocal(a))
    )
  }

  _mean() {
    // mean = α * xm / (α - 1) for α > 1, Infinity for α <= 1
    const a = this._concentration
    const xm = this._scale
    const raw = tf.div(tf.mul(a, xm), tf.sub(a, 1))
    if (this._allowNanStats) {
      return tf.where(
        a.greater(1),
        raw,
        tf.fill(raw.shape, Infinity)
      )
    }
    return raw
  }

  _variance() {
    // variance = xm² * α / ((α-1)² * (α-2)) for α > 2
    const a = this._concentration
    const xm = this._scale
    const am1 = tf.sub(a, 1)
    const am2 = tf.sub(a, 2)
    const raw = tf.div(
      tf.mul(tf.square(xm), a),
      tf.mul(tf.square(am1), am2)
    )
    if (this._allowNanStats) {
      return tf.where(
        a.greater(2),
        raw,
        tf.fill(raw.shape, NaN)
      )
    }
    return raw
  }

  _mode() {
    // mode = xm
    return tf.clone(this._scale)
  }
}
