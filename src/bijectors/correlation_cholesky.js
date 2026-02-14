import * as tf from '@tensorflow/tfjs'
import { Bijector } from './bijector'

/**
 * CorrelationCholesky bijector: maps R^(d*(d-1)/2) → Cholesky factor of
 * a d×d correlation matrix.
 *
 * Uses the stick-breaking parameterization:
 *   1. Map unconstrained values through tanh to get partial correlations in (-1,1)
 *   2. Build the Cholesky factor L row by row:
 *      L[0,0] = 1
 *      L[i,0] = z[i,0]  (partial correlation)
 *      L[i,j] = z[i,j] * prod_{k<j} sqrt(1 - z[i,k]²)   for j > 0
 *      L[i,i] = prod_{k<i} sqrt(1 - z[i,k]²)
 *
 * This ensures each row of L has unit norm (rows sum-of-squares = 1),
 * so L·Lᵀ is a valid correlation matrix.
 *
 * Event ndims: forward 1 → 2
 */
export class CorrelationCholesky extends Bijector {
  constructor({ validateArgs, name } = {}) {
    super({
      forwardMinEventNdims: 1,
      inverseMinEventNdims: 2,
      validateArgs,
      name: name || 'CorrelationCholesky'
    })
  }

  _forward(x) {
    const rank = x.shape.length
    const k = x.shape[rank - 1]
    // k = d*(d-1)/2 → d = (1 + sqrt(1 + 8k)) / 2
    const d = Math.round((1 + Math.sqrt(1 + 8 * k)) / 2)

    const batchShape = x.shape.slice(0, rank - 1)
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1
    const flatData = x.dataSync()
    const result = new Float32Array(batchSize * d * d)

    for (let b = 0; b < batchSize; b++) {
      const xOffset = b * k
      const rOffset = b * d * d

      // Map to partial correlations via tanh
      let idx = 0
      const z = []
      for (let i = 1; i < d; i++) {
        z[i] = []
        for (let j = 0; j < i; j++) {
          z[i][j] = Math.tanh(flatData[xOffset + idx])
          idx++
        }
      }

      // Build L row by row
      result[rOffset] = 1 // L[0,0] = 1

      for (let i = 1; i < d; i++) {
        let remainingNorm = 1
        for (let j = 0; j < i; j++) {
          result[rOffset + i * d + j] = z[i][j] * Math.sqrt(remainingNorm)
          remainingNorm *= (1 - z[i][j] * z[i][j])
        }
        // Diagonal: remaining norm
        result[rOffset + i * d + i] = Math.sqrt(remainingNorm)
      }
    }

    return tf.tensor(result, [...batchShape, d, d])
  }

  _inverse(y) {
    const rank = y.shape.length
    const d = y.shape[rank - 1]
    const k = d * (d - 1) / 2

    const batchShape = y.shape.slice(0, rank - 2)
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1
    const flatData = y.dataSync()
    const result = new Float32Array(batchSize * k)

    for (let b = 0; b < batchSize; b++) {
      const yOffset = b * d * d
      const rOffset = b * k

      let idx = 0
      for (let i = 1; i < d; i++) {
        let remainingNorm = 1
        for (let j = 0; j < i; j++) {
          const lij = flatData[yOffset + i * d + j]
          const sqrtRem = Math.sqrt(Math.max(remainingNorm, 1e-12))
          const z = lij / sqrtRem
          // atanh(z) = 0.5 * log((1+z)/(1-z))
          const zClamped = Math.max(-1 + 1e-7, Math.min(1 - 1e-7, z))
          result[rOffset + idx] = 0.5 * Math.log((1 + zClamped) / (1 - zClamped))
          remainingNorm *= (1 - z * z)
          idx++
        }
      }
    }

    return tf.tensor(result, [...batchShape, k])
  }

  _forwardLogDetJacobian(x) {
    // The Jacobian combines:
    // 1. dtanh/dx = 1 - tanh²(x) for each element
    // 2. The stick-breaking Jacobian: product of sqrt(remainingNorm) terms
    // Total per-element: sum of log(1-tanh²(x_i)) + log-Jacobian of stick-breaking
    const rank = x.shape.length
    const k = x.shape[rank - 1]
    const d = Math.round((1 + Math.sqrt(1 + 8 * k)) / 2)

    const batchShape = x.shape.slice(0, rank - 1)
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1
    const flatData = x.dataSync()
    const result = new Float32Array(batchSize)

    for (let b = 0; b < batchSize; b++) {
      const xOffset = b * k
      let logDet = 0

      let idx = 0
      for (let i = 1; i < d; i++) {
        let remainingNorm = 1
        for (let j = 0; j < i; j++) {
          const t = Math.tanh(flatData[xOffset + idx])
          // log|d(tanh)/dx| = log(1 - t²)
          logDet += Math.log(Math.max(1 - t * t, 1e-30))
          // log(sqrt(remainingNorm)) from stick-breaking
          logDet += 0.5 * Math.log(Math.max(remainingNorm, 1e-30))
          remainingNorm *= (1 - t * t)
          idx++
        }
      }

      result[b] = logDet
    }

    return batchShape.length === 0 ? tf.scalar(result[0]) : tf.tensor(result, batchShape)
  }

  _inverseLogDetJacobian(y) {
    // Compute forward x, then negate FLDJ
    const x = this._inverse(y)
    const fldj = this._forwardLogDetJacobian(x)
    return tf.neg(fldj)
  }
}
