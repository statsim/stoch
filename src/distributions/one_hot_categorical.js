import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { assertOneOf } from '../internal/assert-util'

/**
 * OneHotCategorical distribution.
 *
 * Samples are one-hot vectors of length k. The event shape is [k].
 * Equivalent to Categorical but with one-hot encoded output.
 *
 * P(X = e_i) = probs[i]
 */
export class OneHotCategorical extends Distribution {
  constructor({ probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'OneHotCategorical'
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

  _eventShape() {
    return [this.numCategories]
  }

  _computeBatchShape() {
    const p = this._probs || this._logits
    return p.shape.slice(0, -1)
  }

  _sampleN(n) {
    const logits = this._logits || tf.log(this._probs)
    const flatLogits = logits.rank === 1 ? logits.reshape([1, -1]) : logits
    const indices = tf.multinomial(flatLogits, n) // [batch, n]
    const k = this.numCategories

    if (logits.rank === 1) {
      // indices shape [1, n] → [n]
      const flat = indices.reshape([n])
      return tf.oneHot(flat, k).cast('float32')
    }
    // indices shape [batch, n] → transpose to [n, batch]
    const transposed = tf.transpose(indices)
    // Flatten, one-hot, reshape
    const flat = transposed.reshape([-1])
    const oneHot = tf.oneHot(flat, k).cast('float32')
    return oneHot.reshape([n, ...this.batchShape, k])
  }

  _logProb(value) {
    // value is one-hot: [..., k]
    // logP = sum(value * logProbs, -1)
    const logProbs = this._logits
      ? tf.logSoftmax(this._logits)
      : tf.log(this._probs)
    return tf.sum(tf.mul(value, logProbs), -1)
  }

  _entropy() {
    const p = this._probs || tf.softmax(this._logits)
    const logP = this._logits ? tf.logSoftmax(this._logits) : tf.log(p)
    return tf.neg(tf.sum(tf.mul(p, logP), -1))
  }

  _mean() {
    return this._probs || tf.softmax(this._logits)
  }

  _mode() {
    const p = this._probs || tf.softmax(this._logits)
    const k = this.numCategories
    const argmax = tf.argMax(p, -1)
    return tf.oneHot(argmax, k).cast('float32')
  }
}
