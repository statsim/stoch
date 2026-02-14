import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertOneOf } from '../internal/assert-util'

/**
 * Categorical distribution over k categories.
 *
 * P(X = i) = probs[i]
 *
 * Parameterized by either probs (probability vector) or logits (log-odds vector),
 * but not both.
 *
 * Mirrors TFP Python's Categorical distribution.
 */
export class Categorical extends Distribution {
  constructor({ probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'int32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Categorical'
    })

    assertOneOf(probs, logits, 'probs', 'logits')

    if (probs != null) {
      this._probs = this._addParameter('probs', probs, 'float32')
      this._logits = null
    } else {
      this._logits = this._addParameter('logits', logits, 'float32')
      this._probs = null
    }
  }

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

  _eventShape() { return [] }

  _computeBatchShape() {
    // Batch shape is everything except the last dimension (which is num_categories)
    const p = this._probs || this._logits
    return p.shape.slice(0, -1)
  }

  _sampleN(n) {
    const logits = this._logits || tf.log(this._probs)
    // tf.multinomial expects 2D logits: [batch, numCategories]
    const flatLogits = logits.rank === 1
      ? logits.reshape([1, -1])
      : logits

    const samples = tf.multinomial(flatLogits, n)
    // samples shape: [batch, n], need [n, ...batch]
    if (logits.rank === 1) {
      return samples.reshape([n]).cast('float32')
    }
    return tf.transpose(samples).cast('float32')
  }

  _logProb(value) {
    const logProbs = this._logits
      ? tf.logSoftmax(this._logits)
      : tf.log(this._probs)

    const v = value.cast('int32')
    return tf.gatherND(logProbs, v.reshape([-1, 1])).reshape(v.shape)
  }

  _prob(value) {
    const p = this._probs || tf.softmax(this._logits)
    const v = value.cast('int32')
    return tf.gatherND(p, v.reshape([-1, 1])).reshape(v.shape)
  }

  _entropy() {
    const p = this._probs || tf.softmax(this._logits)
    const logP = this._logits ? tf.logSoftmax(this._logits) : tf.log(p)
    // H = -Σ p_i * log(p_i)
    return tf.neg(tf.sum(tf.mul(p, logP), -1))
  }

  _mean() {
    // E[X] = Σ i * p_i
    const p = this._probs || tf.softmax(this._logits)
    const k = this.numCategories
    const indices = tf.range(0, k, 1, 'float32')
    return tf.sum(tf.mul(p, indices), -1)
  }

  _variance() {
    const p = this._probs || tf.softmax(this._logits)
    const k = this.numCategories
    const indices = tf.range(0, k, 1, 'float32')
    const mean = tf.sum(tf.mul(p, indices), -1)
    // Var = Σ p_i * (i - mean)² = Σ p_i * i² - mean²
    return tf.sub(
      tf.sum(tf.mul(p, tf.square(indices)), -1),
      tf.square(mean)
    )
  }

  _mode() {
    const p = this._probs || tf.softmax(this._logits)
    return tf.argMax(p, -1).cast('float32')
  }
}
