import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma, digamma, incompleteGamma } from '../math/special'
import { assertPositive } from '../internal/assert-util'

/**
 * Inverse Gamma distribution.
 *
 * If X ~ Gamma(a, b) then 1/X ~ InverseGamma(a, 1/b), or equivalently
 * scale/X ~ InverseGamma(a, scale*b).
 *
 * pdf(x; α, β) = β^α / Γ(α) * x^(-α-1) * exp(-β/x)
 *
 * Parameterized by concentration (shape α > 0) and scale (β > 0).
 *
 * Mirrors TFP Python's InverseGamma distribution.
 */
export class InverseGamma extends Distribution {
  constructor({ concentration, scale, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'InverseGamma'
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
    // Sample gamma ~ Gamma(concentration, 1) then return scale / gamma
    const shape = [n, ...this.batchShape]
    const alpha = this._concentration.dataSync()[0]
    const gammas = tf.randomGamma(shape, alpha, 1)
    return tf.div(this._scale, gammas)
  }

  _logProb(value) {
    // log pdf = α*log(β) - logΓ(α) - (α+1)*log(x) - β/x
    const a = this._concentration
    const b = this._scale
    return tf.sub(
      tf.sub(
        tf.mul(a, tf.log(b)),
        logGamma(a)
      ),
      tf.add(
        tf.mul(tf.add(a, 1), tf.log(value)),
        tf.div(b, value)
      )
    )
  }

  _cdf(value) {
    // CDF = Q(α, β/x) = 1 - P(α, β/x) where P is regularized lower incomplete gamma
    const a = this._concentration.dataSync()
    const b = this._scale.dataSync()
    const x = value.dataSync()
    const result = new Float32Array(x.length)
    for (let i = 0; i < x.length; i++) {
      const ai = a.length === 1 ? a[0] : a[i]
      const bi = b.length === 1 ? b[0] : b[i]
      if (x[i] <= 0) {
        result[i] = 0
      } else {
        result[i] = incompleteGamma(ai, bi / x[i]).upper
      }
    }
    return tf.tensor(result, value.shape)
  }

  _entropy() {
    // H = α + log(β * Γ(α)) - (1+α) * ψ(α)
    const a = this._concentration
    const b = this._scale
    return tf.add(
      tf.add(a, tf.add(tf.log(b), logGamma(a))),
      tf.mul(tf.neg(tf.add(1, a)), digamma(a))
    )
  }

  _mean() {
    // mean = β / (α - 1) for α > 1
    const a = this._concentration
    const b = this._scale
    const raw = tf.div(b, tf.sub(a, 1))
    if (this._allowNanStats) {
      return tf.where(
        a.greater(1),
        raw,
        tf.fill(raw.shape, NaN)
      )
    }
    return raw
  }

  _variance() {
    // variance = β² / ((α-1)² * (α-2)) for α > 2
    const a = this._concentration
    const b = this._scale
    const am1 = tf.sub(a, 1)
    const am2 = tf.sub(a, 2)
    const raw = tf.div(tf.square(b), tf.mul(tf.square(am1), am2))
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
    // mode = β / (α + 1)
    return tf.div(this._scale, tf.add(this._concentration, 1))
  }
}
