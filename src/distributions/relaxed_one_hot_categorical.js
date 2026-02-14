import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertOneOf, assertPositive } from '../internal/assert-util'

/**
 * RelaxedOneHotCategorical distribution (Concrete / Gumbel-Softmax).
 *
 * A continuous relaxation of the OneHotCategorical distribution.
 * Samples are on the simplex (sum to 1, all positive) and approach
 * one-hot vectors as temperature → 0.
 *
 * Uses the Gumbel-Softmax trick:
 *   g_i ~ Gumbel(0, 1)
 *   y_i = softmax((logits + g_i) / temperature)
 *
 * Jang et al. (2016), Maddison et al. (2016).
 */
export class RelaxedOneHotCategorical extends Distribution {
  constructor({ temperature, probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'RelaxedOneHotCategorical'
    })

    assertOneOf(probs, logits, 'probs', 'logits')

    this._temperature = this._addParameter('temperature', temperature)

    if (this._validateArgs) {
      assertPositive(this._temperature, 'temperature')
    }

    if (probs != null) {
      this._probs = this._addParameter('probs', probs, 'float32')
      this._logits = null
    } else {
      this._logits = this._addParameter('logits', logits, 'float32')
      this._probs = null
    }
  }

  get temperature() { return this._temperature }

  get probs() {
    if (this._probs) return this._probs
    return tf.tidy(() => tf.softmax(this._logits))
  }

  get logits() {
    if (this._logits) return this._logits
    return tf.tidy(() => tf.log(this._probs))
  }

  get numCategories() {
    const p = this._probs || this._logits
    return p.shape[p.shape.length - 1]
  }

  _eventShape() {
    return [this.numCategories]
  }

  _computeBatchShape() {
    const p = this._probs || this._logits
    return p.shape.slice(0, -1)
  }

  _sampleN(n) {
    const logits = this._logits || tf.log(this._probs)
    const k = this.numCategories
    const shape = [n, ...this.batchShape, k]

    // Gumbel noise: -log(-log(U))
    const u = tf.randomUniform(shape, 1e-7, 1 - 1e-7)
    const gumbel = tf.neg(tf.log(tf.neg(tf.log(u))))

    // y = softmax((logits + gumbel) / temperature)
    const scaled = tf.div(tf.add(logits, gumbel), this._temperature)
    return tf.softmax(scaled, -1)
  }

  _logProb(value) {
    // Concrete distribution log density:
    // log p(y) = log(Γ(k)) + (k-1)*log(T) + sum_i (log(α_i) - T*log(y_i))
    //            - k * log(sum_i α_i * y_i^(-T))
    // where α_i = exp(logits_i)
    const logits = this._logits || tf.log(this._probs)
    const T = this._temperature
    const k = this.numCategories

    // log(Γ(k)) = log((k-1)!)
    let logFactK = 0
    for (let i = 2; i < k; i++) logFactK += Math.log(i)

    const logAlpha = logits // logits = log(probs) or raw logits
    const negTlogY = tf.mul(tf.neg(T), tf.log(value))

    // sum_i (log(α_i) - T*log(y_i))
    const perElement = tf.add(logAlpha, negTlogY)
    const sumTerm = tf.sum(perElement, -1)

    // log(sum_i α_i * y_i^(-T)) = log(sum_i exp(logAlpha_i - T*log(y_i)))
    // = logSumExp(logAlpha - T*log(y_i))
    const logSumExpTerm = tf.log(tf.sum(tf.exp(perElement), -1))

    return tf.sub(
      tf.add(logFactK + (k - 1) * Math.log(T.dataSync()[0]), sumTerm),
      tf.mul(k, logSumExpTerm)
    )
  }

  _mean() {
    // As T → 0, mean → one-hot(argmax). For finite T, mean ≈ softmax(logits/T)
    // but the exact mean of the Concrete distribution = probs (same as Categorical)
    return this._probs || tf.softmax(this._logits)
  }
}
