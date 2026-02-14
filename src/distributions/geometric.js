import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertOneOf, assertInRange } from '../internal/assert-util'

/**
 * Geometric distribution (0-indexed).
 *
 * P(X = k) = p * (1-p)^k  for k = 0, 1, 2, ...
 *
 * Counts the number of failures before the first success.
 *
 * Parameterized by either probs (p in (0,1]) or logits (log-odds),
 * but not both.
 *
 * Mirrors TFP Python's Geometric distribution.
 */
export class Geometric extends Distribution {
  constructor({ probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Geometric'
    })

    assertOneOf(probs, logits, 'probs', 'logits')

    if (probs != null) {
      this._probs = this._addParameter('probs', probs, 'float32')
      this._logits = null
      if (this._validateArgs) {
        assertInRange(this._probs, 0, 1, 'probs')
      }
    } else {
      this._logits = this._addParameter('logits', logits, 'float32')
      this._probs = null
    }
  }

  get probs() {
    if (this._probs) return this._probs
    return tf.tidy(() => tf.sigmoid(this._logits))
  }

  get logits() {
    if (this._logits) return this._logits
    return tf.tidy(() => tf.log(tf.div(this._probs, tf.sub(1, this._probs))))
  }

  _sampleN(n) {
    // k = floor(log(U) / log(1 - p)) where U ~ Uniform(0, 1)
    const p = this._probs || tf.sigmoid(this._logits)
    const shape = [n, ...this.batchShape]
    const u = tf.randomUniform(shape)
    const log1mP = tf.log(tf.sub(1, p))
    return tf.floor(tf.div(tf.log(u), log1mP))
  }

  _logProb(value) {
    // log P(k) = log(p) + k * log(1 - p)
    const v = value.cast('float32')
    const p = this._probs || tf.sigmoid(this._logits)
    const logP = tf.log(p)
    const log1mP = tf.log(tf.sub(1, p))
    return tf.add(logP, tf.mul(v, log1mP))
  }

  _cdf(value) {
    // CDF(k) = 1 - (1-p)^(floor(k)+1) for k >= 0
    const v = tf.floor(value.cast('float32'))
    const p = this._probs || tf.sigmoid(this._logits)
    const one = tf.onesLike(p)
    const raw = tf.sub(one, tf.pow(tf.sub(one, p), tf.add(v, 1)))
    // CDF is 0 for k < 0
    return tf.where(v.greaterEqual(0), raw, tf.zerosLike(raw))
  }

  _entropy() {
    // H = [-(1-p)*log(1-p) - p*log(p)] / p
    const p = this._probs || tf.sigmoid(this._logits)
    const q = tf.sub(1, p)
    return tf.div(
      tf.neg(tf.add(tf.mul(q, tf.log(q)), tf.mul(p, tf.log(p)))),
      p
    )
  }

  _mean() {
    // E[X] = (1-p) / p
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.div(tf.sub(1, p), p)
  }

  _variance() {
    // Var[X] = (1-p) / p^2
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.div(tf.sub(1, p), tf.square(p))
  }

  _mode() {
    // Mode is always 0 for the geometric distribution
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.zerosLike(p)
  }
}
