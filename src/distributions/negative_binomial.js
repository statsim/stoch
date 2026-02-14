import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma } from '../math/special'
import { assertOneOf, assertPositive, assertInRange } from '../internal/assert-util'

/**
 * Negative Binomial distribution.
 *
 * Counts the number of failures before totalCount successes.
 *
 * P(X = k) = C(k + r - 1, k) * p^r * (1-p)^k
 *
 * where r = totalCount (target successes), p = probs (success probability),
 * k = number of failures = 0, 1, 2, ...
 *
 * Parameterized by totalCount (r > 0) and either probs (p in (0,1])
 * or logits (log-odds), but not both.
 *
 * Mirrors TFP Python's NegativeBinomial distribution.
 */
export class NegativeBinomial extends Distribution {
  constructor({ totalCount, probs, logits, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'NegativeBinomial'
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
      assertPositive(this._totalCount, 'totalCount')
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
    // Gamma-Poisson mixture: λ ~ Gamma(r, (1-p)/p), then k ~ Poisson(λ)
    const shape = [n, ...this.batchShape]
    const r = this._totalCount.dataSync()[0]
    const p = this._probs
      ? this._probs.dataSync()[0]
      : 1 / (1 + Math.exp(-this._logits.dataSync()[0]))

    const scale = (1 - p) / p
    const size = shape.reduce((a, b) => a * b, 1)
    const result = new Float32Array(size)

    for (let i = 0; i < size; i++) {
      const lambda = sampleGamma(r, scale)
      result[i] = samplePoisson(lambda)
    }

    return tf.tensor(result, shape)
  }

  _logProb(value) {
    // log P(k) = logΓ(k+r) - logΓ(k+1) - logΓ(r) + r*log(p) + k*log(1-p)
    const v = value.cast('float32')
    const r = this._totalCount
    const p = this._probs || tf.sigmoid(this._logits)

    const logCombinatorial = tf.sub(
      logGamma(tf.add(v, r)),
      tf.add(logGamma(tf.add(v, 1)), logGamma(r))
    )

    const logP = tf.log(p)
    const log1mP = tf.log(tf.sub(1, p))

    return tf.add(
      logCombinatorial,
      tf.add(tf.mul(r, logP), tf.mul(v, log1mP))
    )
  }

  _mean() {
    // E[X] = r * (1-p) / p
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.div(tf.mul(this._totalCount, tf.sub(1, p)), p)
  }

  _variance() {
    // Var[X] = r * (1-p) / p^2
    const p = this._probs || tf.sigmoid(this._logits)
    return tf.div(
      tf.mul(this._totalCount, tf.sub(1, p)),
      tf.square(p)
    )
  }

  _mode() {
    // mode = floor((r-1)*(1-p)/p) when r > 1, else 0
    const p = this._probs || tf.sigmoid(this._logits)
    const raw = tf.floor(tf.div(
      tf.mul(tf.sub(this._totalCount, 1), tf.sub(1, p)),
      p
    ))
    return tf.where(
      this._totalCount.greater(1),
      raw,
      tf.zerosLike(raw)
    )
  }
}

/**
 * Sample from Gamma(alpha, scale) using Marsaglia and Tsang's method.
 */
function sampleGamma(alpha, scale) {
  if (alpha < 1) {
    return sampleGamma(alpha + 1, scale) * Math.pow(Math.random(), 1 / alpha)
  }

  const d = alpha - 1 / 3
  const c = 1 / Math.sqrt(9 * d)

  while (true) {
    let x, v
    do {
      x = randn()
      v = 1 + c * x
    } while (v <= 0)

    v = v * v * v
    const u = Math.random()
    const x2 = x * x

    if (u < 1 - 0.0331 * x2 * x2) return d * v * scale
    if (Math.log(u) < 0.5 * x2 + d * (1 - v + Math.log(v))) return d * v * scale
  }
}

function randn() {
  const u1 = Math.random()
  const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

function samplePoisson(lambda) {
  if (lambda === 0) return 0
  const L = Math.exp(-lambda)
  let k = 0
  let p = 1
  do {
    k++
    p *= Math.random()
  } while (p > L)
  return k - 1
}
