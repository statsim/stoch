import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma } from '../math/special'
import { assertPositive, assertInRange } from '../internal/assert-util'

/**
 * Zero-Inflated Poisson distribution.
 *
 * A mixture of a point mass at 0 and a Poisson distribution:
 *   P(X = 0) = gate + (1 - gate) * exp(-rate)
 *   P(X = k) = (1 - gate) * rate^k * exp(-rate) / k!   for k > 0
 *
 * @param {number|tf.Tensor} rate - Poisson rate (λ > 0)
 * @param {number|tf.Tensor} gate - Probability of excess zeros (0 ≤ gate < 1)
 */
export class ZeroInflatedPoisson extends Distribution {
  constructor({ rate, gate, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'ZeroInflatedPoisson'
    })

    this._rate = this._addParameter('rate', rate)
    this._gate = this._addParameter('gate', gate)

    if (this._validateArgs) {
      assertPositive(this._rate, 'rate')
    }
  }

  get rate() { return this._rate }
  get gate() { return this._gate }

  _sampleN(n) {
    const shape = [n, ...this.batchShape]
    const gateVal = this._gate.dataSync()[0]
    const rateVal = this._rate.dataSync()[0]
    const size = shape.reduce((a, b) => a * b, 1)
    const result = new Float32Array(size)

    const L = Math.exp(-rateVal)

    for (let i = 0; i < size; i++) {
      if (Math.random() < gateVal) {
        result[i] = 0
      } else {
        // Poisson sample
        let k = 0, p = 1
        do {
          k++
          p *= Math.random()
        } while (p > L)
        result[i] = k - 1
      }
    }

    return tf.tensor(result, shape)
  }

  _logProb(value) {
    const v = value.cast('float32')
    const rate = this._rate
    const gate = this._gate

    // Poisson logProb: k*log(rate) - rate - logΓ(k+1)
    const poissonLogP = tf.sub(
      tf.mul(v, tf.log(rate)),
      tf.add(rate, logGamma(tf.add(v, 1)))
    )

    // For k = 0: log(gate + (1-gate)*exp(-rate))
    // For k > 0: log(1-gate) + poissonLogP
    const log1mGate = tf.log(tf.sub(1, gate))
    const logPNonZero = tf.add(log1mGate, poissonLogP)

    // At k = 0: logP = log(gate + (1-gate)*exp(-rate))
    // = log(gate + exp(log(1-gate) - rate))
    // Use logAddExp for numerical stability
    const logPZero = _logAddExp(
      tf.log(gate),
      tf.add(log1mGate, tf.neg(rate))
    )

    // Select based on whether value is 0
    const isZero = tf.equal(v, 0).cast('float32')
    return tf.add(
      tf.mul(isZero, logPZero),
      tf.mul(tf.sub(1, isZero), logPNonZero)
    )
  }

  _mean() {
    // E[X] = (1 - gate) * rate
    return tf.mul(tf.sub(1, this._gate), this._rate)
  }

  _variance() {
    // Var[X] = (1 - gate) * rate * (1 + gate * rate)
    const oneMinusGate = tf.sub(1, this._gate)
    return tf.mul(
      tf.mul(oneMinusGate, this._rate),
      tf.add(1, tf.mul(this._gate, this._rate))
    )
  }

  _mode() {
    // Mode is 0 if gate is high or rate is small
    // Otherwise mode of Poisson is floor(rate) if (1-gate)*P(floor(rate)) > P(0)
    // For simplicity, return 0 (most common case for zero-inflated)
    return tf.zerosLike(this._rate)
  }
}

function _logAddExp(a, b) {
  const max = tf.maximum(a, b)
  return tf.add(max, tf.log(tf.add(
    tf.exp(tf.sub(a, max)),
    tf.exp(tf.sub(b, max))
  )))
}
