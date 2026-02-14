import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { logGamma } from '../math/special'

/**
 * Wishart distribution over d×d positive-definite matrices.
 *
 * Parameterized by degrees of freedom df and scale matrix Cholesky factor scaleTril.
 * The scale matrix is V = L·Lᵀ.
 *
 * Uses Bartlett decomposition for sampling.
 *
 * @param {number} df - degrees of freedom (must be > d-1)
 * @param {tf.Tensor|Array} scaleTril - [..., d, d] lower-triangular Cholesky of scale
 */
export class Wishart extends Distribution {
  constructor({ df, scaleTril, validateArgs, name } = {}) {
    super({ dtype: 'float32', validateArgs, name: name || 'Wishart' })

    this._df = typeof df === 'number' ? tf.scalar(df) : (df instanceof tf.Tensor ? df : tf.tensor(df))
    this._scaleTril = scaleTril instanceof tf.Tensor
      ? scaleTril
      : tf.tensor(scaleTril, undefined, 'float32')

    const rank = this._scaleTril.shape.length
    this._d = this._scaleTril.shape[rank - 1]

    if (validateArgs !== false) {
      const dfVal = this._df.dataSync()[0]
      if (dfVal <= this._d - 1) {
        throw new Error(`Wishart: df (${dfVal}) must be > d-1 (${this._d - 1})`)
      }
    }
  }

  get df() { return this._df }
  get scaleTril() { return this._scaleTril }

  _eventShape() {
    return [this._d, this._d]
  }

  _logProb(value) {
    const d = this._d
    const dfVal = this._df.dataSync()[0]

    const valData = value.dataSync()
    const LData = this._scaleTril.dataSync()
    const batchShape = value.shape.slice(0, -2)
    const batchSize = batchShape.reduce((a, b) => a * b, 1) || 1
    const result = new Float32Array(batchSize)

    const logZ = this._logNormConst(dfVal, d, LData)

    const L = []
    for (let i = 0; i < d; i++) {
      L[i] = []
      for (let j = 0; j < d; j++) {
        L[i][j] = LData[i * d + j]
      }
    }

    for (let b = 0; b < batchSize; b++) {
      const xOff = b * d * d

      const X = []
      for (let i = 0; i < d; i++) {
        X[i] = []
        for (let j = 0; j < d; j++) {
          X[i][j] = valData[xOff + i * d + j]
        }
      }

      const cholX = _choleskyJS(X, d)
      if (!cholX) {
        result[b] = -Infinity
        continue
      }

      let logDetX = 0
      for (let i = 0; i < d; i++) logDetX += 2 * Math.log(cholX[i][i])

      // tr(V⁻¹X) = ||L⁻¹ chol(X)||²_F
      const Y = _forwardSolveJS(L, cholX, d)
      let traceVinvX = 0
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          traceVinvX += Y[i][j] * Y[i][j]
        }
      }

      result[b] = (dfVal - d - 1) / 2 * logDetX - 0.5 * traceVinvX - logZ
    }

    return batchShape.length === 0 ? tf.scalar(result[0]) : tf.tensor(result, batchShape)
  }

  _logNormConst(dfVal, d, LData) {
    let logDetV = 0
    for (let i = 0; i < d; i++) logDetV += 2 * Math.log(Math.abs(LData[i * d + i]))

    // Multivariate log gamma: logΓ_d(a) = d(d-1)/4*log(π) + sum_{j=1}^d logΓ(a + (1-j)/2)
    const a = dfVal / 2
    let logMvGamma = d * (d - 1) / 4 * Math.log(Math.PI)
    for (let j = 1; j <= d; j++) {
      logMvGamma += _logGammaScalar(a + (1 - j) / 2)
    }

    return dfVal * d / 2 * Math.log(2) + dfVal / 2 * logDetV + logMvGamma
  }

  _sampleN(n) {
    const d = this._d
    const dfVal = this._df.dataSync()[0]
    const LData = this._scaleTril.dataSync()

    const L = []
    for (let i = 0; i < d; i++) {
      L[i] = []
      for (let j = 0; j < d; j++) {
        L[i][j] = LData[i * d + j]
      }
    }

    const result = new Float32Array(n * d * d)

    for (let s = 0; s < n; s++) {
      // Bartlett decomposition
      const A = []
      for (let i = 0; i < d; i++) {
        A[i] = new Array(d).fill(0)
        const shape = (dfVal - i) / 2
        A[i][i] = Math.sqrt(2 * _gammaRandom(shape))
        for (let j = 0; j < i; j++) {
          A[i][j] = _normalRandom()
        }
      }

      const LA = _matMulJS(L, A, d)

      const off = s * d * d
      for (let i = 0; i < d; i++) {
        for (let j = 0; j < d; j++) {
          let sum = 0
          for (let k = 0; k < d; k++) {
            sum += LA[i][k] * LA[j][k]
          }
          result[off + i * d + j] = sum
        }
      }
    }

    return tf.tensor(result, [n, d, d])
  }

  _mean() {
    const LLt = tf.matMul(this._scaleTril, this._scaleTril, false, true)
    return tf.mul(this._df, LLt)
  }

  _entropy() {
    const d = this._d
    const dfVal = this._df.dataSync()[0]
    const LData = this._scaleTril.dataSync()

    let logDetV = 0
    for (let i = 0; i < d; i++) logDetV += 2 * Math.log(Math.abs(LData[i * d + i]))

    const logZ = this._logNormConst(dfVal, d, LData)

    let sumDigamma = 0
    for (let i = 1; i <= d; i++) {
      sumDigamma += _digammaApprox((dfVal + 1 - i) / 2)
    }
    const ElogDetX = d * Math.log(2) + logDetV + sumDigamma
    const entropy = logZ + (d + 1 - dfVal) / 2 * ElogDetX + dfVal * d / 2

    return tf.scalar(entropy)
  }

  dispose() {
    if (this._df instanceof tf.Tensor && !this._df.isDisposed) this._df.dispose()
    if (this._scaleTril instanceof tf.Tensor && !this._scaleTril.isDisposed) this._scaleTril.dispose()
  }
}

// --- Helper functions ---

function _logGammaScalar(x) {
  // Scalar logGamma using the tensor version, extracting the value
  const t = logGamma(x)
  const val = t.dataSync()[0]
  t.dispose()
  return val
}

function _choleskyJS(mat, n) {
  const L = []
  for (let i = 0; i < n; i++) L[i] = new Array(n).fill(0)
  for (let j = 0; j < n; j++) {
    let sum = 0
    for (let k = 0; k < j; k++) sum += L[j][k] * L[j][k]
    const diag = mat[j][j] - sum
    if (diag <= 0) return null
    L[j][j] = Math.sqrt(diag)
    for (let i = j + 1; i < n; i++) {
      let s = 0
      for (let k = 0; k < j; k++) s += L[i][k] * L[j][k]
      L[i][j] = (mat[i][j] - s) / L[j][j]
    }
  }
  return L
}

function _forwardSolveJS(L, B, n) {
  const X = []
  for (let i = 0; i < n; i++) X[i] = new Array(n).fill(0)
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      let s = B[i][j]
      for (let k = 0; k < i; k++) s -= L[i][k] * X[k][j]
      X[i][j] = s / L[i][i]
    }
  }
  return X
}

function _matMulJS(A, B, n) {
  const C = []
  for (let i = 0; i < n; i++) {
    C[i] = new Array(n).fill(0)
    for (let j = 0; j < n; j++) {
      for (let k = 0; k < n; k++) {
        C[i][j] += A[i][k] * B[k][j]
      }
    }
  }
  return C
}

function _normalRandom() {
  const u1 = Math.random()
  const u2 = Math.random()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

function _gammaRandom(shape) {
  if (shape < 1) {
    return _gammaRandom(shape + 1) * Math.pow(Math.random(), 1 / shape)
  }
  const d = shape - 1 / 3
  const c = 1 / Math.sqrt(9 * d)
  while (true) {
    let x, v
    do {
      x = _normalRandom()
      v = 1 + c * x
    } while (v <= 0)
    v = v * v * v
    const u = Math.random()
    if (u < 1 - 0.0331 * x * x * x * x) return d * v
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v
  }
}

function _digammaApprox(x) {
  if (x < 6) {
    return _digammaApprox(x + 1) - 1 / x
  }
  return Math.log(x) - 1 / (2 * x) - 1 / (12 * x * x) + 1 / (120 * x * x * x * x)
}
