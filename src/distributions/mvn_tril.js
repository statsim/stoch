import * as tf from '@tensorflow/tfjs'
import { Distribution } from './distribution'
import { triangularSolve } from '../math/triangularSolve'
import { LOG_2PI } from '../math/numeric'

/**
 * Multivariate Normal distribution with a lower-triangular scale matrix.
 *
 * The covariance is Σ = L · Lᵀ where L = scaleTril (lower triangular).
 *
 * Example:
 *   const mvn = new MultivariateNormalTriL({
 *     loc: [0, 0],
 *     scaleTril: [[1, 0], [0.5, 0.866]]
 *   })
 *   mvn.sample()    // shape [2]
 *   mvn.logProb([0, 0])  // scalar
 */
export class MultivariateNormalTriL extends Distribution {
  constructor({ loc, scaleTril, validateArgs, name } = {}) {
    super({ dtype: 'float32', validateArgs, name: name || 'MultivariateNormalTriL' })

    this._loc = loc instanceof tf.Tensor ? loc : tf.tensor(loc, undefined, 'float32')
    this._scaleTril = scaleTril instanceof tf.Tensor
      ? scaleTril
      : tf.tensor(scaleTril, undefined, 'float32')

    const rank = this._scaleTril.shape.length
    this._d = this._scaleTril.shape[rank - 1]
  }

  get loc() { return this._loc }
  get scaleTril() { return this._scaleTril }

  _eventShape() {
    return [this._d]
  }

  _logProb(value) {
    // log p(x) = -0.5*d*log(2π) - sum(log|diag(L)|) - 0.5*||L⁻¹(x-μ)||²
    const d = this._d
    const diff = tf.sub(value, this._loc) // [..., d]

    // Solve L*z = diff → z = L⁻¹*diff
    const diffExpanded = diff.expandDims(-1) // [..., d, 1]
    const z = triangularSolve(this._scaleTril, diffExpanded) // [..., d, 1]
    const zSqueezed = z.squeeze([-1]) // [..., d]

    // Mahalanobis distance: ||z||²
    const mahal = tf.sum(tf.mul(zSqueezed, zSqueezed), -1)

    // Log determinant: sum of log|diag(L)| — extract diag via dataSync
    const logDetL = this._logDetL()

    return tf.sub(
      tf.sub(tf.scalar(-0.5 * d * LOG_2PI), logDetL),
      tf.mul(0.5, mahal)
    )
  }

  _logDetL() {
    const d = this._d
    const data = this._scaleTril.dataSync()
    let logDet = 0
    for (let i = 0; i < d; i++) {
      logDet += Math.log(Math.abs(data[i * d + i]))
    }
    return tf.scalar(logDet)
  }

  _sampleN(n) {
    const d = this._d
    const batchShape = this._loc.shape.slice(0, -1)
    // z ~ N(0, I), shape [n, ...batch, d]
    const z = tf.randomNormal([n, ...batchShape, d])
    // x = μ + L*z
    const zExpanded = z.expandDims(-1) // [n, ...batch, d, 1]
    const Lz = tf.matMul(this._scaleTril, zExpanded) // [n, ...batch, d, 1]
    const LzSqueezed = Lz.squeeze([-1]) // [n, ...batch, d]
    return tf.add(this._loc, LzSqueezed)
  }

  _mean() {
    return this._loc
  }

  _variance() {
    // diag(L*Lᵀ) = sum of L[i,j]² over j for each i
    const LLt = tf.matMul(this._scaleTril, this._scaleTril, false, true)
    const rank = LLt.shape.length
    // Extract diagonal: element-wise multiply with identity, sum last axis
    const d = this._d
    const eye = tf.eye(d)
    return tf.sum(tf.mul(LLt, eye), rank - 1)
  }

  _entropy() {
    const d = this._d
    // H = 0.5*d*(1 + log(2π)) + sum(log|diag(L)|)
    const logDetL = this._logDetL()
    return tf.add(tf.scalar(0.5 * d * (1 + LOG_2PI)), logDetL)
  }

  dispose() {
    if (this._loc instanceof tf.Tensor && !this._loc.isDisposed) this._loc.dispose()
    if (this._scaleTril instanceof tf.Tensor && !this._scaleTril.isDisposed) this._scaleTril.dispose()
  }
}
