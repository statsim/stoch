import * as tf from '@tensorflow/tfjs'
import { Kernel } from './kernel'

/**
 * White noise kernel.
 *
 * k(x1, x2) = variance * δ(x1, x2)
 *
 * For matrix computation, adds variance on the diagonal only
 * when x1 === x2 (same reference) or when shapes match.
 *
 * @param {Object} params
 * @param {number} [params.variance=1]
 */
export class White extends Kernel {
  constructor({ variance = 1 } = {}) {
    super({ name: 'White' })
    this._variance = variance
  }

  get variance() { return this._variance }

  _matrix(x1, x2) {
    const n1 = x1.shape[0]
    const n2 = x2.shape[0]
    if (n1 === n2) {
      // Assume x1 === x2 (self-covariance): return variance * I
      return tf.mul(this._variance, tf.eye(n1))
    }
    // Cross-covariance: return zeros
    return tf.zeros([n1, n2])
  }

  _apply(x1, x2) {
    // Pointwise: always returns variance (assumed same points)
    return tf.fill([x1.shape[0]], this._variance)
  }
}
