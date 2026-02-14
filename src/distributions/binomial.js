import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma, incompleteBeta } from '../math/special'
import { assertOneOf, assertNonNegative, assertInRange } from '../internal/assert-util'

/**
 * Binomial distribution.
 *
 * P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
 *
 * Parameterized by totalCount (n >= 0, integer) and either probs (p in [0,1])
 * or logits (log-odds), but not both.
 *
 * Mirrors TFP Python's Binomial distribution.
 */
export class Binomial extends Distribution {
  constructor({ totalCount, probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Binomial'
    })

    assertOneOf(probs, logits, 'probs', 'logits')

    this._totalCount = this._addParameter('totalCount', totalCount, 'float32')

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

    if (this._validateArgs) {
      assertNonNegative(this._totalCount, 'totalCount')
    }
  }

  get totalCount() { return this._totalCount }

  get probs() {
    if (this._probs) return this._probs
    return tf.tidy(() => tf.sigmoid(this._logits))
  }

  get logits() {
    if (this._logits) return this._logits
    return tf.tidy(() => tf.log(tf.div(this._probs, tf.sub(1, this._probs))))
  }

  _sampleN(n) {
    // Sample by summing totalCount independent Bernoulli trials
    const p = this._probs || tf.sigmoid(this._logits)
    const totalCount = this._totalCount.dataSync()[0]
    const shape = [n, ...this.batchShape]
    const batchSize = shape.reduce((a, b) => a * b, 1)

    // For each sample, sum totalCount Bernoulli(p) draws
    const pVal = p.dataSync()
    const result = new Float32Array(batchSize)
    for (let i = 0; i < batchSize; i++) {
      const pi = pVal.length === 1 ? pVal[0] : pVal[i % pVal.length]
      let count = 0
      for (let t = 0; t < totalCount; t++) {
        if (Math.random() < pi) count++
      }
      result[i] = count
    }

    return tf.tensor(result, shape)
  }

  _logProb(value) {
    // log P(k) = logΓ(n+1) - logΓ(k+1) - logΓ(n-k+1) + k*log(p) + (n-k)*log(1-p)
    const v = value.cast('float32')
    const n = this._totalCount
    const p = this._probs || tf.sigmoid(this._logits)

    const logCombinatorial = tf.sub(
      logGamma(tf.add(n, 1)),
      tf.add(logGamma(tf.add(v, 1)), logGamma(tf.add(tf.sub(n, v), 1)))
    )

    const logP = tf.log(p)
    const log1mP = tf.log(tf.sub(1, p))

    return tf.add(
      logCombinatorial,
      tf.add(tf.mul(v, logP), tf.mul(tf.sub(n, v), log1mP))
    )
  }

  _cdf(value) {
    // CDF(k) = 1 - I_p(k+1, n-k) for integer k
    // where I_p is the regularized incomplete beta function
    const v = tf.floor(value.cast('float32'))
    const n = this._totalCount
    const p = this._probs || tf.sigmoid(this._logits)

    const vData = v.dataSync()
    const nData = n.dataSync()
    const pData = p.dataSync()
    const result = new Float32Array(vData.length)

    for (let i = 0; i < vData.length; i++) {
      const ki = vData[i]
      const ni = nData.length === 1 ? nData[0] : nData[i]
      const pi = pData.length === 1 ? pData[0] : pData[i]

      if (ki < 0) {
        result[i] = 0
      } else if (ki >= ni) {
        result[i] = 1
      } else {
        // CDF = 1 - I_p(k+1, n-k)
        result[i] = 1 - incompleteBeta(ki + 1, ni - ki, pi)
      }
    }

    return tf.tensor(result, value.shape)
  }

  _mean() {
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.mul(this._totalCount, p)
  }

  _variance() {
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.mul(tf.mul(this._totalCount, p), tf.sub(1, p))
  }

  _mode() {
    // mode = floor((n+1)*p)
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.floor(tf.mul(tf.add(this._totalCount, 1), p))
  }
}
