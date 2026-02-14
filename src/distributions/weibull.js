import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma } from '../math/special'
import { assertPositive } from '../internal/assert-util'

/**
 * Weibull distribution.
 *
 * pdf(x; k, λ) = (k/λ) * (x/λ)^(k-1) * exp(-(x/λ)^k)  for x >= 0
 *
 * Parameterized by concentration (shape k > 0) and scale (λ > 0).
 *
 * Mirrors TFP Python's Weibull distribution.
 */
export class Weibull extends Distribution {
  constructor({ concentration, scale, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Weibull'
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
    // Sample: λ * (-log(U))^(1/k) where U ~ Uniform(0, 1)
    const shape = [n, ...this.batchShape]
    const u = tf.randomUniform(shape)
    const invK = tf.reciprocal(this._concentration)
    return tf.mul(this._scale, tf.pow(tf.neg(tf.log(u)), invK))
  }

  _logProb(value) {
    // log pdf = log(k) - k*log(λ) + (k-1)*log(x) - (x/λ)^k, -Infinity for x < 0
    const k = this._concentration
    const lam = this._scale
    const logP = tf.sub(
      tf.add(
        tf.sub(tf.log(k), tf.mul(k, tf.log(lam))),
        tf.mul(tf.sub(k, 1), tf.log(value))
      ),
      tf.pow(tf.div(value, lam), k)
    )
    const inSupport = value.greaterEqual(0)
    return tf.where(inSupport, logP, tf.fill(value.shape, -Infinity))
  }

  _cdf(value) {
    // CDF = 1 - exp(-(x/λ)^k) for x >= 0, 0 otherwise
    const k = this._concentration
    const lam = this._scale
    const raw = tf.sub(1, tf.exp(tf.neg(tf.pow(tf.div(value, lam), k))))
    const inSupport = value.greaterEqual(0)
    return tf.where(inSupport, raw, tf.zerosLike(value))
  }

  _entropy() {
    // H = γ*(1 - 1/k) + log(λ/k) + 1
    // where γ is the Euler-Mascheroni constant
    const { EULER_MASCHERONI } = require('../math/numeric')
    const k = this._concentration
    const lam = this._scale
    return tf.add(
      tf.mul(EULER_MASCHERONI, tf.sub(1, tf.reciprocal(k))),
      tf.add(tf.log(tf.div(lam, k)), 1)
    )
  }

  _mean() {
    // mean = λ * Γ(1 + 1/k)
    const k = this._concentration
    const lam = this._scale
    const gammaTerm = tf.exp(logGamma(tf.add(1, tf.reciprocal(k))))
    return tf.mul(lam, gammaTerm)
  }

  _variance() {
    // variance = λ² * [Γ(1 + 2/k) - Γ(1 + 1/k)²]
    const k = this._concentration
    const lam = this._scale
    const g1 = tf.exp(logGamma(tf.add(1, tf.reciprocal(k))))
    const g2 = tf.exp(logGamma(tf.add(1, tf.div(2, k))))
    return tf.mul(tf.square(lam), tf.sub(g2, tf.square(g1)))
  }

  _mode() {
    // mode = λ * ((k-1)/k)^(1/k) for k > 1, 0 otherwise
    const k = this._concentration
    const lam = this._scale
    const raw = tf.mul(lam, tf.pow(tf.div(tf.sub(k, 1), k), tf.reciprocal(k)))
    return tf.where(
      k.greater(1),
      raw,
      tf.zerosLike(raw)
    )
  }
}
