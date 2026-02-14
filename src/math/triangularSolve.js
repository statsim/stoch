import * as tf from '@tensorflow/tfjs'

/**
 * Solve a triangular linear system L · X = B (or Lᵀ · X = B).
 *
 * @param {tf.Tensor|Array} matrix  [..., N, N] lower- or upper-triangular matrix
 * @param {tf.Tensor|Array} rhs     [..., N, K] right-hand side (K columns)
 * @param {Object} opts
 * @param {boolean} opts.lower   true (default) ⇒ L is lower-triangular
 * @param {boolean} opts.adjoint true ⇒ solve Lᵀ · X = B instead of L · X = B
 * @returns {tf.Tensor} X with shape [..., N, K]
 */
export function triangularSolve(matrix, rhs, { lower = true, adjoint = false } = {}) {
  return tf.tidy(() => {
    let mat = matrix instanceof tf.Tensor ? matrix : tf.tensor(matrix)
    let b = rhs instanceof tf.Tensor ? rhs : tf.tensor(rhs)

    const matShape = mat.shape
    const matRank = matShape.length
    if (matRank < 2) {
      throw new Error(
        `triangularSolve: matrix must be at least 2D, got shape [${matShape}]`
      )
    }
    const n = matShape[matRank - 1]
    const m = matShape[matRank - 2]
    if (n !== m) {
      throw new Error(
        `triangularSolve: matrix must be square, got [..., ${m}, ${n}]`
      )
    }

    // Handle rhs: allow 1D (vector) or 2D+ (matrix)
    const rhsShape = b.shape
    const rhsRank = rhsShape.length
    let squeezed = false
    if (rhsRank === matRank - 1) {
      // rhs is a vector [..., N] — add trailing dim to get [..., N, 1]
      b = b.expandDims(-1)
      squeezed = true
    }

    const bShape = b.shape
    const bRank = bShape.length
    if (bRank < 2) {
      throw new Error(
        `triangularSolve: rhs must be at least 1D, got shape [${rhsShape}]`
      )
    }
    const bRows = bShape[bRank - 2]
    const k = bShape[bRank - 1]
    if (bRows !== n) {
      throw new Error(
        `triangularSolve: incompatible shapes, matrix is [..., ${n}, ${n}] but rhs is [..., ${bRows}, ${k}]`
      )
    }

    // Compute batch dimensions
    const matBatch = matShape.slice(0, matRank - 2)
    const bBatch = bShape.slice(0, bRank - 2)
    const batchSize = matBatch.reduce((a, c) => a * c, 1) || 1

    // Reshape to [batchSize, n, n] and [batchSize, n, k]
    const flatMat = mat.reshape([batchSize, n, n])
    const flatB = b.reshape([batchSize, n, k])
    const matData = flatMat.arraySync()
    const bData = flatB.arraySync()

    const results = new Float32Array(batchSize * n * k)

    for (let batch = 0; batch < batchSize; batch++) {
      const L = matData[batch]
      const B = bData[batch]
      const off = batch * n * k

      if (lower && !adjoint) {
        // Forward substitution: L · X = B
        _forwardSolve(L, B, n, k, results, off)
      } else if (lower && adjoint) {
        // Back substitution with Lᵀ: Lᵀ · X = B
        _backSolveTranspose(L, B, n, k, results, off)
      } else if (!lower && !adjoint) {
        // Back substitution: U · X = B
        _backSolve(L, B, n, k, results, off)
      } else {
        // Forward substitution with Uᵀ: Uᵀ · X = B
        _forwardSolveTranspose(L, B, n, k, results, off)
      }
    }

    let result = tf.tensor(results, [...matBatch, n, k])
    if (squeezed) {
      result = result.squeeze([-1])
    }
    return result
  })
}

/** Forward substitution: solve L · X = B where L is lower-triangular */
function _forwardSolve(L, B, n, k, out, off) {
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < n; i++) {
      if (L[i][i] === 0) {
        throw new Error(`triangularSolve: singular matrix (zero diagonal at index ${i})`)
      }
      let sum = B[i][j]
      for (let p = 0; p < i; p++) {
        sum -= L[i][p] * out[off + p * k + j]
      }
      out[off + i * k + j] = sum / L[i][i]
    }
  }
}

/** Back substitution with Lᵀ: solve Lᵀ · X = B where L is lower-triangular */
function _backSolveTranspose(L, B, n, k, out, off) {
  for (let j = 0; j < k; j++) {
    for (let i = n - 1; i >= 0; i--) {
      if (L[i][i] === 0) {
        throw new Error(`triangularSolve: singular matrix (zero diagonal at index ${i})`)
      }
      let sum = B[i][j]
      for (let p = i + 1; p < n; p++) {
        sum -= L[p][i] * out[off + p * k + j]
      }
      out[off + i * k + j] = sum / L[i][i]
    }
  }
}

/** Back substitution: solve U · X = B where U is upper-triangular */
function _backSolve(U, B, n, k, out, off) {
  for (let j = 0; j < k; j++) {
    for (let i = n - 1; i >= 0; i--) {
      if (U[i][i] === 0) {
        throw new Error(`triangularSolve: singular matrix (zero diagonal at index ${i})`)
      }
      let sum = B[i][j]
      for (let p = i + 1; p < n; p++) {
        sum -= U[i][p] * out[off + p * k + j]
      }
      out[off + i * k + j] = sum / U[i][i]
    }
  }
}

/** Forward substitution with Uᵀ: solve Uᵀ · X = B where U is upper-triangular */
function _forwardSolveTranspose(U, B, n, k, out, off) {
  for (let j = 0; j < k; j++) {
    for (let i = 0; i < n; i++) {
      if (U[i][i] === 0) {
        throw new Error(`triangularSolve: singular matrix (zero diagonal at index ${i})`)
      }
      let sum = B[i][j]
      for (let p = 0; p < i; p++) {
        sum -= U[p][i] * out[off + p * k + j]
      }
      out[off + i * k + j] = sum / U[i][i]
    }
  }
}
