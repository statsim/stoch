import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma } from '../math/special'
import { assertOneOf, assertNonNegative } from '../internal/assert-util'

/**
 * Multinomial distribution.
 *
 * P(k1, ..., kd) = n! / (k1! * ... * kd!) * p1^k1 * ... * pd^kd
 *
 * where sum(ki) = n = totalCount.
 *
 * Parameterized by totalCount (n >= 0) and either probs (probability vector
 * summing to 1) or logits (unnormalized log-probabilities), but not both.
 *
 * eventShape: [d] where d = number of categories.
 *
 * Mirrors TFP Python's Multinomial distribution.
 */
export class Multinomial extends Distribution {
  constructor({ totalCount, probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Multinomial'
    })

    assertOneOf(probs, logits, 'probs', 'logits')

    this._totalCount = this._addParameter('totalCount', totalCount, 'float32')

    if (probs != null) {
      this._probs = this._addParameter('probs', probs, 'float32')
      this._logits = null
    } else {
      this._logits = this._addParameter('logits', logits, 'float32')
      this._probs = null
    }

    if (this._validateArgs) {
      assertNonNegative(this._totalCount, 'totalCount')
    }
  }

  get totalCount() { return this._totalCount }

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
    // Batch shape for probs/logits is everything except the last dimension
    const p = this._probs || this._logits
    const probBatch = p.shape.slice(0, -1)
    // totalCount shape is pure batch
    const tcShape = this._totalCount.shape
    // Broadcast the two batch shapes
    if (probBatch.length === 0 && tcShape.length === 0) return []
    if (probBatch.length === 0) return tcShape
    if (tcShape.length === 0) return probBatch
    // Simple broadcast: take the larger
    const maxLen = Math.max(probBatch.length, tcShape.length)
    const result = new Array(maxLen)
    for (let i = 0; i < maxLen; i++) {
      const a = i < probBatch.length ? probBatch[probBatch.length - 1 - i] : 1
      const b = i < tcShape.length ? tcShape[tcShape.length - 1 - i] : 1
      if (a !== 1 && b !== 1 && a !== b) {
        throw new Error(`Incompatible batch shapes: probs ${probBatch}, totalCount ${tcShape}`)
      }
      result[maxLen - 1 - i] = Math.max(a, b)
    }
    return result
  }

  _sampleN(n) {
    // Use tf.multinomial to draw totalCount categorical samples, then count
    const logits = this._logits || tf.log(this._probs)
    const totalCount = Math.round(this._totalCount.dataSync()[0])
    const K = this.numCategories
    const batchShape = this.batchShape
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1

    // tf.multinomial expects 2D logits: [batch, K]
    const flatLogits = logits.rank === 1
      ? logits.reshape([1, -1])
      : logits.reshape([-1, K])

    const numRows = flatLogits.shape[0]
    const sampleShape = [n, ...batchShape, K]
    const totalSamples = n * batchSize

    // Draw totalCount samples for each of (n * batchSize) instances
    // Each row of flatLogits gets totalSamples / numRows draws
    const samplesPerRow = totalSamples / numRows

    const resultData = new Float32Array(totalSamples * K)

    for (let row = 0; row < numRows; row++) {
      const rowLogits = flatLogits.slice([row, 0], [1, K])
      // Draw totalCount * samplesPerRow categorical samples
      const draws = tf.multinomial(rowLogits, totalCount * samplesPerRow)
      const drawData = draws.dataSync()
      draws.dispose()
      rowLogits.dispose()

      // Count occurrences for each sample
      for (let s = 0; s < samplesPerRow; s++) {
        const offset = (row * samplesPerRow + s) * K
        for (let t = 0; t < totalCount; t++) {
          const category = drawData[s * totalCount + t]
          resultData[offset + category]++
        }
      }
    }

    return tf.tensor(resultData, sampleShape)
  }

  _logProb(value) {
    // log P(k) = logΓ(n+1) - Σ logΓ(k_i + 1) + Σ k_i * log(p_i)
    const v = value.cast('float32')
    const n = this._totalCount
    const p = this._probs || tf.softmax(this._logits)

    const logN1 = logGamma(tf.add(n, 1))
    const sumLogGammaK1 = tf.sum(logGamma(tf.add(v, 1)), -1)
    const sumKLogP = tf.sum(tf.mul(v, tf.log(p)), -1)

    return tf.add(tf.sub(logN1, sumLogGammaK1), sumKLogP)
  }

  _mean() {
    // E[X_k] = n * p_k
    const p = this._probs || tf.softmax(this._logits)
    return tf.mul(this._totalCount.expandDims(-1), p)
  }

  _variance() {
    // Var[X_k] = n * p_k * (1 - p_k)
    const p = this._probs || tf.softmax(this._logits)
    const n = this._totalCount.expandDims(-1)
    return tf.mul(n, tf.mul(p, tf.sub(1, p)))
  }

  _mode() {
    // Approximate mode: round(n * p_k) adjusted so sum = n
    // Simpler: floor(n * p_k) as a reasonable approximation
    const p = this._probs || tf.softmax(this._logits)
    return tf.floor(tf.mul(this._totalCount.expandDims(-1), p))
  }
}
