import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * Ascending bijector: maps unconstrained R^d to ordered (sorted ascending) R^d.
 *
 * Forward:  y[0] = x[0], y[i] = y[i-1] + softplus(x[i]) for i > 0
 * Inverse:  x[0] = y[0], x[i] = softplus_inverse(y[i] - y[i-1]) for i > 0
 * FLDJ:     sum_{i>0} log(sigmoid(x[i])) = sum_{i>0} (x[i] - softplus(x[i]))
 *
 * Event shape: [d] → [d], forwardMinEventNdims = 1
 */
export class Ascending extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      forwardMinEventNdims: 1,
      inverseMinEventNdims: 1,
      validateArgs,
      name: name || 'Ascending'
    })
  }

  _forward(x) {
    // y[0] = x[0], y[i] = y[i-1] + softplus(x[i])
    // Equivalent to: y[0] = x[0], cumulative sum of [x[0], softplus(x[1]), ...]
    const rank = x.shape.length
    const d = x.shape[rank - 1]
    if (d <= 1) return x

    // Split into first element and rest along last axis
    const first = tf.slice(x, Array(rank).fill(0), [...x.shape.slice(0, -1), 1])
    const rest = tf.slice(x, [...Array(rank - 1).fill(0), 1], [...x.shape.slice(0, -1), d - 1])

    // Apply softplus to rest, then concatenate and cumsum
    const spRest = tf.softplus(rest)
    const diffs = tf.concat([first, spRest], rank - 1)
    return tf.cumsum(diffs, rank - 1)
  }

  _inverse(y) {
    const rank = y.shape.length
    const d = y.shape[rank - 1]
    if (d <= 1) return y

    const first = tf.slice(y, Array(rank).fill(0), [...y.shape.slice(0, -1), 1])
    const yPrev = tf.slice(y, Array(rank).fill(0), [...y.shape.slice(0, -1), d - 1])
    const yCurr = tf.slice(y, [...Array(rank - 1).fill(0), 1], [...y.shape.slice(0, -1), d - 1])

    // diffs = y[i] - y[i-1], apply softplusInverse
    const diffs = tf.sub(yCurr, yPrev)
    // softplus_inverse(z) = log(exp(z) - 1) = z + log(1 - exp(-z))
    const spInv = tf.add(diffs, tf.log(tf.sub(1, tf.exp(tf.neg(diffs)))))

    return tf.concat([first, spInv], rank - 1)
  }

  _forwardLogDetJacobian(x) {
    // FLDJ = sum_{i>0} log(sigmoid(x[i])) = sum_{i>0} (x[i] - softplus(x[i]))
    const rank = x.shape.length
    const d = x.shape[rank - 1]
    if (d <= 1) return tf.zeros(x.shape)

    const rest = tf.slice(x, [...Array(rank - 1).fill(0), 1], [...x.shape.slice(0, -1), d - 1])
    // log(sigmoid(z)) = z - softplus(z)
    return tf.sub(rest, tf.softplus(rest))
  }

  _inverseLogDetJacobian(y) {
    // ILDJ = -FLDJ(inverse(y)) = sum_{i>0} softplus(x[i]) - x[i]
    // where x = inverse(y) and softplus(x[i]) = y[i] - y[i-1]
    const rank = y.shape.length
    const d = y.shape[rank - 1]
    if (d <= 1) return tf.zeros(y.shape)

    const yPrev = tf.slice(y, Array(rank).fill(0), [...y.shape.slice(0, -1), d - 1])
    const yCurr = tf.slice(y, [...Array(rank - 1).fill(0), 1], [...y.shape.slice(0, -1), d - 1])
    const diffs = tf.sub(yCurr, yPrev)

    // softplusInverse(diffs)
    const spInv = tf.add(diffs, tf.log(tf.sub(1, tf.exp(tf.neg(diffs)))))

    // ILDJ per element = softplus(x[i]) - x[i] = diffs - spInv
    return tf.sub(diffs, spInv)
  }
}
