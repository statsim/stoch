import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logBeta, digamma } from '../math/special'
import { assertPositive } from '../internal/assert-util'

/**
 * Beta distribution on [0, 1].
 *
 * pdf(x; α, β) = x^(α-1) * (1-x)^(β-1) / B(α, β)
 *
 * Parameterized by concentration1 (α > 0) and concentration0 (β > 0).
 * Uses TFP naming convention (not a/b).
 *
 * Mirrors TFP Python's Beta distribution.
 */
export class Beta extends Distribution {
  constructor({ concentration1, concentration0, validateArgs, name } = {}) {
    super({
      dtype: 'float32',
      validateArgs: validateArgs != null ? validateArgs : true,
      name: name || 'Beta'
    })

    this._concentration1 = this._addParameter('concentration1', concentration1)
    this._concentration0 = this._addParameter('concentration0', concentration0)

    if (this._validateArgs) {
      assertPositive(this._concentration1, 'concentration1')
      assertPositive(this._concentration0, 'concentration0')
    }
  }

  get concentration1() { return this._concentration1 }
  get concentration0() { return this._concentration0 }

  _sampleN(n) {
    // Beta samples via ratio of Gamma samples:
    // X ~ Gamma(α), Y ~ Gamma(β), then X/(X+Y) ~ Beta(α, β)
    const shape = [n, ...this.batchShape]
    const a = this._concentration1.dataSync()[0]
    const b = this._concentration0.dataSync()[0]
    const x = tf.randomGamma(shape, a)
    const y = tf.randomGamma(shape, b)
    return tf.div(x, tf.add(x, y))
  }

  _logProb(value) {
    // log pdf = (α-1)*log(x) + (β-1)*log(1-x) - logBeta(α, β)
    return tf.sub(
      tf.add(
        tf.mul(tf.sub(this._concentration1, 1), tf.log(value)),
        tf.mul(tf.sub(this._concentration0, 1), tf.log(tf.sub(1, value)))
      ),
      logBeta(this._concentration1, this._concentration0)
    )
  }

  _entropy() {
    const a = this._concentration1
    const b = this._concentration0
    const ab = tf.add(a, b)
    // H = logBeta(α, β) - (α-1)*ψ(α) - (β-1)*ψ(β) + (α+β-2)*ψ(α+β)
    return tf.add(
      logBeta(a, b),
      tf.add(
        tf.neg(tf.mul(tf.sub(a, 1), digamma(a))),
        tf.add(
          tf.neg(tf.mul(tf.sub(b, 1), digamma(b))),
          tf.mul(tf.sub(ab, 2), digamma(ab))
        )
      )
    )
  }

  _mean() {
    return tf.div(this._concentration1, tf.add(this._concentration1, this._concentration0))
  }

  _variance() {
    const a = this._concentration1
    const b = this._concentration0
    const ab = tf.add(a, b)
    return tf.div(tf.mul(a, b), tf.mul(tf.square(ab), tf.add(ab, 1)))
  }

  _mode() {
    const a = this._concentration1
    const b = this._concentration0
    // mode = (α - 1) / (α + β - 2) for α > 1 and β > 1
    return tf.div(tf.sub(a, 1), tf.sub(tf.add(a, b), 2))
  }
}
