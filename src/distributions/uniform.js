import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'

/**
 * Uniform distribution on [low, high].
 *
 * pdf(x; a, b) = 1 / (b - a)  for a <= x <= b, 0 otherwise.
 *
 * Mirrors TFP Python's Uniform distribution.
 */
export class Uniform extends Distribution {
  constructor({ low = 0, high = 1, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Uniform'
    })

    this._low = this._addParameter('low', low)
    this._high = this._addParameter('high', high)

    if (this._validateArgs) {
      const lo = this._low.dataSync()[0]
      const hi = this._high.dataSync()[0]
      if (lo >= hi) {
        throw new Error(`low must be less than high, got low=${lo}, high=${hi}`)
      }
    }
  }

  get low() { return this._low }
  get high() { return this._high }

  _range() {
    return tf.sub(this._high, this._low)
  }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    return tf.add(
      tf.mul(tf.randomUniform(shape), this._range()),
      this._low
    )
  }

  _logProb(value) {
    const range = this._range()
    const logP = tf.neg(tf.log(range))
    // -Infinity outside [low, high]
    const inSupport = tf.logicalAnd(
      value.greaterEqual(this._low),
      value.lessEqual(this._high)
    )
    return tf.where(inSupport, tf.broadcastTo(logP, value.shape), tf.fill(value.shape, -Infinity))
  }

  _cdf(value) {
    const range = this._range()
    const raw = tf.div(tf.sub(value, this._low), range)
    return tf.clipByValue(raw, 0, 1)
  }

  _entropy() {
    return tf.log(this._range())
  }

  _mean() {
    return tf.div(tf.add(this._low, this._high), 2)
  }

  _variance() {
    return tf.div(tf.square(this._range()), 12)
  }

  _mode() {
    // Uniform has no unique mode; return midpoint by convention
    return this._mean()
  }
}
