import * as tf from '@tensorflow/tfjs'
import { triangularSolve } from './triangularSolve'

/**
 * Cholesky decomposition with custom gradient support.
 *
 * Computes lower-triangular L where A = L · Lᵀ for symmetric positive-definite A.
 * The backward pass uses the Iain Murray (2016) formula so gradients flow
 * through to the input matrix — required for GP kernel hyperparameter learning.
 *
 * @param {tf.Tensor|Array} matrix  [..., N, N] symmetric positive-definite matrix
 * @param {Object} opts
 * @param {number} opts.jitter  Small value added to diagonal for numerical stability (default 0)
 * @returns {tf.Tensor} L with shape [..., N, N], lower-triangular
 */
export function cholesky(matrix, { jitter = 0 } = {}) {
  const a = matrix instanceof tf.Tensor ? matrix : tf.tensor(matrix)

  const f = tf.customGrad((a, save) => {
    const L = choleskyForward(a, jitter)
    save([L])
    return {
      value: L,
      gradFunc: (dL, saved) => [choleskyBackward(saved[0], dL)]
    }
  })

  return f(a)
}

/**
 * Forward Cholesky: pure JS via dataSync for efficiency on GP-sized matrices.
 * Computation done in float64 (JS number), output as float32 tensor.
 *
 * IMPORTANT: This function must NOT create any tracked tf ops on the input
 * tensor `a` (no reshape, slice, etc.) because it runs inside tf.customGrad's
 * forward body. Creating tracked ops would confuse the gradient tape.
 * Use dataSync() to pull raw data instead.
 */
function choleskyForward(a, jitter) {
  const shape = a.shape
  const rank = shape.length
  if (rank < 2) {
    throw new Error(`cholesky: requires at least a 2D tensor, got shape [${shape}]`)
  }
  const n = shape[rank - 1]
  const m = shape[rank - 2]
  if (n !== m) {
    throw new Error(`cholesky: requires square matrices, got [..., ${m}, ${n}]`)
  }

  const batchShape = shape.slice(0, rank - 2)
  const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1

  // Pull flat data without creating tracked tensors
  const flatData = a.dataSync()

  const results = new Float32Array(batchSize * n * n)

  for (let b = 0; b < batchSize; b++) {
    const offset = b * n * n

    // Build 2D JS array from flat data for this batch element
    const mat = []
    for (let i = 0; i < n; i++) {
      mat[i] = []
      for (let j = 0; j < n; j++) {
        mat[i][j] = flatData[offset + i * n + j]
      }
    }

    // Symmetrize — defensive against float32 rounding in kernel matrices
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < i; j++) {
        const avg = (mat[i][j] + mat[j][i]) / 2
        mat[i][j] = avg
        mat[j][i] = avg
      }
    }

    // Add jitter to diagonal if requested
    if (jitter > 0) {
      for (let i = 0; i < n; i++) {
        mat[i][i] += jitter
      }
    }

    _choleskyInPlace(mat, n, results, offset)
  }

  return tf.tensor(results, [...batchShape, n, n])
}

/**
 * Column-by-column (outer-product form) Cholesky decomposition.
 * Input mat is a 2D JS array; output is written into Float32Array `out` at `offset`.
 */
function _choleskyInPlace(mat, n, out, offset) {
  for (let j = 0; j < n; j++) {
    // L[j][j] = sqrt(A[j][j] - Σ_{k<j} L[j][k]²)
    let sumDiag = 0
    for (let k = 0; k < j; k++) {
      const ljk = out[offset + j * n + k]
      sumDiag += ljk * ljk
    }
    const diag = mat[j][j] - sumDiag
    if (diag <= 0) {
      throw new Error(
        `cholesky: matrix is not positive definite (diagonal element ${j} = ${diag})`
      )
    }
    const ljj = Math.sqrt(diag)
    out[offset + j * n + j] = ljj

    // L[i][j] = (A[i][j] - Σ_{k<j} L[i][k] * L[j][k]) / L[j][j]  for i > j
    for (let i = j + 1; i < n; i++) {
      let sumOff = 0
      for (let k = 0; k < j; k++) {
        sumOff += out[offset + i * n + k] * out[offset + j * n + k]
      }
      out[offset + i * n + j] = (mat[i][j] - sumOff) / ljj
    }
    // Upper triangle stays 0 (Float32Array is zero-initialized)
  }
}

/**
 * Backward pass for Cholesky: Iain Murray (2016) formula.
 *
 * Given upstream gradient dL (same shape as L), computes dA:
 *   dA = L⁻ᵀ · Φ(Lᵀ · dL) · L⁻¹,  symmetrized
 * where Φ extracts the lower triangle and halves the diagonal.
 *
 * NOTE: dL is NOT guaranteed to be lower-triangular. For example,
 * cholesky(A).sum() yields all-ones dL. The formula handles this correctly —
 * Lᵀ · dL mixes all elements, then Φ projects to the lower triangle.
 * Do NOT zero the upper triangle of dL before this function.
 */
function choleskyBackward(L, dL) {
  return tf.tidy(() => {
    const rank = L.shape.length
    // Permutation to swap last two axes: [..., i, j] → [..., j, i]
    const perm = []
    for (let i = 0; i < rank - 2; i++) perm.push(i)
    perm.push(rank - 1, rank - 2)

    // P = Lᵀ · dL
    const Lt = tf.transpose(L, perm)
    const P = tf.matMul(Lt, dL)

    // Φ(P) = lower_triangle(P) with halved diagonal
    // bandPart(-1, 0) = full lower triangle (including diagonal)
    // bandPart(0, 0)  = diagonal only
    const lower = tf.linalg.bandPart(P, -1, 0)
    const Phi = tf.sub(lower, tf.mul(tf.linalg.bandPart(P, 0, 0), 0.5))

    // S = L⁻ᵀ · Phi  (solve Lᵀ · S = Phi)
    const S = triangularSolve(L, Phi, { lower: true, adjoint: true })

    // dA_unsym = (L⁻ᵀ · Phi · L⁻¹) = solve(Lᵀ, Sᵀ)ᵀ
    const St = tf.transpose(S, perm)
    const dA_unsym = tf.transpose(
      triangularSolve(L, St, { lower: true, adjoint: true }),
      perm
    )

    // Symmetrize — gradient of a function of symmetric matrices must be symmetric
    return tf.div(tf.add(dA_unsym, tf.transpose(dA_unsym, perm)), 2)
  })
}
