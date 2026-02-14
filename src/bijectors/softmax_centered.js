import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * SoftmaxCentered bijector: maps R^(d-1) → simplex(d).
 *
 * Forward:  y = softmax(concat([x, 0]))
 * Inverse:  x = log(y[0:d-1]) - log(y[d-1])
 *
 * The last component acts as the reference (pinned to 0 in log-space).
 *
 * FLDJ = log-determinant of the Jacobian of the softmax mapping, which is:
 *   sum(log(y)) + log(d) ... but we use the standard formula:
 *   sum_{i=0}^{d-1} log(y_i) - (d/2)*log(d) ... actually:
 *
 * For the centered softmax with d-1 → d mapping:
 *   FLDJ = sum(log(y_i) for all i) where y = softmax(concat([x, 0]))
 *
 * Event ndims: forward maps [..., d-1] → [..., d]
 */
export class SoftmaxCentered extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      forwardMinEventNdims: 1,
      inverseMinEventNdims: 1,
      validateArgs,
      name: name || 'SoftmaxCentered'
    })
  }

  _forward(x) {
    const rank = x.shape.length
    // Append a zero along the last axis: [x, 0]
    const zeroShape = [...x.shape.slice(0, -1), 1]
    const padded = tf.concat([x, tf.zeros(zeroShape)], rank - 1)
    return tf.softmax(padded, rank - 1)
  }

  _inverse(y) {
    const rank = y.shape.length
    const d = y.shape[rank - 1]
    // x_i = log(y_i) - log(y_{d-1})
    const logY = tf.log(y)
    const logYLast = tf.slice(logY, [...Array(rank - 1).fill(0), d - 1], [...y.shape.slice(0, -1), 1])
    const logYFirst = tf.slice(logY, Array(rank).fill(0), [...y.shape.slice(0, -1), d - 1])
    return tf.sub(logYFirst, logYLast)
  }

  _forwardLogDetJacobian(x) {
    // For the centered softmax bijector R^(d-1) → Δ^d:
    // FLDJ = sum_i log(y_i) where y = softmax([x, 0])
    const rank = x.shape.length
    const zeroShape = [...x.shape.slice(0, -1), 1]
    const padded = tf.concat([x, tf.zeros(zeroShape)], rank - 1)
    const y = tf.softmax(padded, rank - 1)
    return tf.sum(tf.log(y), rank - 1)
  }

  _inverseLogDetJacobian(y) {
    // ILDJ = -FLDJ = -sum_i log(y_i)
    const rank = y.shape.length
    return tf.neg(tf.sum(tf.log(y), rank - 1))
  }
}
