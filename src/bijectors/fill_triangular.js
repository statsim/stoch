import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * FillTriangular bijector: maps R^(n*(n+1)/2) → lower triangular n×n matrix.
 *
 * Forward:  vec of length n*(n+1)/2 → n×n lower triangular matrix
 * Inverse:  n×n lower triangular matrix → vec of length n*(n+1)/2
 *
 * Elements are filled row by row in the lower triangle.
 * This is a volume-preserving (permutation-like) bijector, so log-det-Jacobian = 0.
 *
 * Event ndims: forward 1 → 2, inverse 2 → 1
 */
export class FillTriangular extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      forwardMinEventNdims: 1,
      inverseMinEventNdims: 2,
      isConstantJacobian: true,
      validateArgs,
      name: name || 'FillTriangular'
    })
  }

  _forward(x) {
    const rank = x.shape.length
    const k = x.shape[rank - 1]
    // k = n*(n+1)/2 → n = (-1 + sqrt(1 + 8k)) / 2
    const n = Math.round((-1 + Math.sqrt(1 + 8 * k)) / 2)

    const batchShape = x.shape.slice(0, rank - 1)
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1
    const flatData = x.dataSync()
    const result = new Float32Array(batchSize * n * n)

    for (let b = 0; b < batchSize; b++) {
      const bOffset = b * k
      const rOffset = b * n * n
      let idx = 0
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
          result[rOffset + i * n + j] = flatData[bOffset + idx]
          idx++
        }
      }
    }

    return tf.tensor(result, [...batchShape, n, n])
  }

  _inverse(y) {
    const rank = y.shape.length
    const n = y.shape[rank - 1]
    const k = n * (n + 1) / 2

    const batchShape = y.shape.slice(0, rank - 2)
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1
    const flatData = y.dataSync()
    const result = new Float32Array(batchSize * k)

    for (let b = 0; b < batchSize; b++) {
      const yOffset = b * n * n
      const rOffset = b * k
      let idx = 0
      for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
          result[rOffset + idx] = flatData[yOffset + i * n + j]
          idx++
        }
      }
    }

    return tf.tensor(result, [...batchShape, k])
  }

  _forwardLogDetJacobian(x) {
    // Pure permutation → log-det-Jacobian = 0
    const batchShape = x.shape.slice(0, x.shape.length - 1)
    return batchShape.length === 0 ? tf.scalar(0) : tf.zeros(batchShape)
  }

  _inverseLogDetJacobian(y) {
    const batchShape = y.shape.slice(0, y.shape.length - 2)
    return batchShape.length === 0 ? tf.scalar(0) : tf.zeros(batchShape)
  }
}
