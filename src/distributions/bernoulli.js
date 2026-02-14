import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertOneOf, assertInRange } from '../internal/assert-util'

/**
 * Bernoulli distribution.
 *
 * P(X = 1) = p, P(X = 0) = 1 - p
 *
 * Parameterized by either probs (probability of 1) or logits (log-odds),
 * but not both. Mirrors TFP Python's Bernoulli distribution.
 */
export class Bernoulli extends Distribution {
  constructor({ probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'int32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Bernoulli'
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
    // logit(p) = log(p / (1 - p))
    return tf.tidy(() => tf.log(tf.div(this._probs, tf.sub(1, this._probs))))
  }

  _eventShape() { return [] }

  _sampleN(n) {
    const p = this._probs || tf.sigmoid(this._logits)
    const shape = [n, ...this.batchShape]
    return tf.less(tf.randomUniform(shape), p).cast('float32')
  }

  _logProb(value) {
    // For Bernoulli: logP(x) = x * log(p) + (1-x) * log(1-p)
    // Using logits form for numerical stability:
    // logP(x) = x * logits - softplus(logits)
    const logits = this._logits || tf.log(tf.div(this._probs, tf.sub(1, this._probs)))
    const v = value.cast('float32')
    return tf.sub(tf.mul(v, logits), tf.softplus(logits))
  }

  _prob(value) {
    const p = this._probs || tf.sigmoid(this._logits)
    const v = value.cast('float32')
    return tf.add(tf.mul(v, p), tf.mul(tf.sub(1, v), tf.sub(1, p)))
  }

  _cdf(value) {
    const p = this._probs || tf.sigmoid(this._logits)
    const v = value.cast('float32')
    // CDF: 0 for x < 0, 1-p for 0 <= x < 1, 1 for x >= 1
    const isGe0 = v.greaterEqual(0).cast('float32')
    const isGe1 = v.greaterEqual(1).cast('float32')
    return tf.add(tf.mul(isGe0, tf.sub(1, p)), tf.mul(isGe1, p))
  }

  _entropy() {
    const logits = this._logits || tf.log(tf.div(this._probs, tf.sub(1, this._probs)))
    // H = softplus(logits) - logits * sigmoid(logits)
    return tf.sub(tf.softplus(logits), tf.mul(logits, tf.sigmoid(logits)))
  }

  _mean() {
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.clone(p)
  }

  _variance() {
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.mul(p, tf.sub(1, p))
  }

  _mode() {
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.where(tf.greater(p, 0.5), tf.onesLike(p), tf.zerosLike(p))
  }
}
