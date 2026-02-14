import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma, incompleteGamma } from '../math/special'
import { assertPositive } from '../internal/assert-util'

/**
 * Gamma distribution.
 *
 * pdf(x; α, β) = β^α * x^(α-1) * exp(-βx) / Γ(α)
 *
 * Parameterized by concentration (shape α > 0) and rate (inverse scale β > 0).
 *
 * Mirrors TFP Python's Gamma distribution.
 */
export class Gamma extends Distribution {
  constructor({ concentration, rate, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Gamma'
    })

    this._concentration = this._addParameter('concentration', concentration)
    this._rate = this._addParameter('rate', rate)

    if (this._validateArgs) {
      assertPositive(this._concentration, 'concentration')
      assertPositive(this._rate, 'rate')
    }
  }

  get concentration() { return this._concentration }
  get rate() { return this._rate }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    // tf.randomGamma expects shape parameter alpha and optional beta
    // It parameterizes as Gamma(alpha, beta) where beta = 1/rate (scale)
    const alpha = this._concentration.dataSync()[0]
    const beta = 1 / this._rate.dataSync()[0]
    return tf.randomGamma(shape, alpha, beta)
  }

  _logProb(value) {
    // log pdf = α*log(β) + (α-1)*log(x) - β*x - logΓ(α)
    return tf.sub(
      tf.add(
        tf.mul(this._concentration, tf.log(this._rate)),
        tf.mul(tf.sub(this._concentration, 1), tf.log(value))
      ),
      tf.add(
        tf.mul(this._rate, value),
        logGamma(this._concentration)
      )
    )
  }

  _cdf(value) {
    // CDF = P(α, β*x) where P is the regularized lower incomplete gamma
    const x = tf.mul(this._rate, value).dataSync()
    const a = this._concentration.dataSync()
    const result = new Float32Array(x.length)
    for (let i = 0; i < x.length; i++) {
      const ai = a.length === 1 ? a[0] : a[i]
      if (x[i] <= 0) {
        result[i] = 0
      } else {
        result[i] = incompleteGamma(ai, x[i]).lower
      }
    }
    return tf.tensor(result, value.shape)
  }

  _entropy() {
    // H = α - log(β) + logΓ(α) + (1-α)*ψ(α)
    const { digamma } = require('../math/special')
    return tf.add(
      tf.sub(this._concentration, tf.log(this._rate)),
      tf.add(
        logGamma(this._concentration),
        tf.mul(tf.sub(1, this._concentration), digamma(this._concentration))
      )
    )
  }

  _mean() {
    return tf.div(this._concentration, this._rate)
  }

  _variance() {
    return tf.div(this._concentration, tf.square(this._rate))
  }

  _mode() {
    // mode = (α - 1) / β for α >= 1, 0 otherwise
    const raw = tf.div(tf.sub(this._concentration, 1), this._rate)
    return tf.where(
      this._concentration.greaterEqual(1),
      raw,
      tf.zerosLike(raw)
    )
  }
}
